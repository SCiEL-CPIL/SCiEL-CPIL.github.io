import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit

    # Fallback: n=9 modular coloring (valid)
    def mod_color(n, mod=3):
        return [0 if (a + b + c) % mod == 0 else 1 for a, b, c in combinations(range(n), 3)]

    best_n = 9
    best_colors = mod_color(9, 3)

    def is_valid(n, colors):
        """Verify a coloring is valid (no red K4, no blue K5)."""
        if len(colors) != len(list(combinations(range(n), 3))):
            return False
        cm = {}
        for i, t in enumerate(combinations(range(n), 3)):
            cm[t] = colors[i]
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            if cm[(a,b,c)] == 0 and cm[(a,b,d)] == 0 and cm[(a,c,d)] == 0 and cm[(b,c,d)] == 0:
                return False
        for quint in combinations(range(n), 5):
            all_blue = True
            for t in combinations(quint, 3):
                if cm[t] != 1:
                    all_blue = False
                    break
            if all_blue:
                return False
        return True

    def build_structures(target_n):
        """Precompute all structures needed for fast local search."""
        triple_list = list(combinations(range(target_n), 3))
        num_triples = len(triple_list)
        triple_to_idx = {t: i for i, t in enumerate(triple_list)}

        all_quads = list(combinations(range(target_n), 4))
        quad_triples_idx = []
        for a, b, c, d in all_quads:
            quad_triples_idx.append([
                triple_to_idx[(a,b,c)], triple_to_idx[(a,b,d)],
                triple_to_idx[(a,c,d)], triple_to_idx[(b,c,d)]
            ])
        num_quads = len(quad_triples_idx)

        all_quints = list(combinations(range(target_n), 5))
        quint_triples_idx = []
        for q5 in all_quints:
            quint_triples_idx.append([triple_to_idx[t] for t in combinations(q5, 3)])
        num_quints = len(quint_triples_idx)

        # For each triple, list of quad indices containing it
        triple_in_quads = [[] for _ in range(num_triples)]
        for qi, qt in enumerate(quad_triples_idx):
            for tidx in qt:
                triple_in_quads[tidx].append(qi)

        # For each triple, list of quint indices containing it
        triple_in_quints = [[] for _ in range(num_triples)]
        for qi, qt in enumerate(quint_triples_idx):
            for tidx in qt:
                triple_in_quints[tidx].append(qi)

        # Bitsets for quads and quints (using Python ints as arbitrary precision bitsets)
        quad_bitsets = []
        for qt in quad_triples_idx:
            bs = 0
            for i in qt:
                bs |= (1 << i)
            quad_bitsets.append(bs)

        quint_bitsets = []
        for qt in quint_triples_idx:
            bs = 0
            for i in qt:
                bs |= (1 << i)
            quint_bitsets.append(bs)

        return (triple_list, num_triples, triple_to_idx, quad_triples_idx, num_quads,
                quint_triples_idx, num_quints, triple_in_quads, triple_in_quints,
                quad_bitsets, quint_bitsets)

    def count_violations(color_arr, num_triples, quad_bitsets, quint_bitsets, num_quads, num_quints):
        """Count red K4 and blue K5 violations."""
        red_mask = 0
        for i in range(num_triples):
            if color_arr[i] == 0:
                red_mask |= (1 << i)
        rk4 = 0
        for qbs in quad_bitsets:
            if (red_mask & qbs) == qbs:
                rk4 += 1
        bk5 = 0
        for qbs in quint_bitsets:
            if (red_mask & qbs) == 0:
                bk5 += 1
        return rk4, bk5, red_mask

    def local_search(target_n, init_colors, deadline, structures):
        """Run local search to eliminate violations. Returns (colors, success)."""
        (triple_list, num_triples, triple_to_idx, quad_triples_idx, num_quads,
         quint_triples_idx, num_quints, triple_in_quads, triple_in_quints,
         quad_bitsets, quint_bitsets) = structures

        color_arr = np.array(init_colors, dtype=np.int8)
        
        rk4, bk5, red_mask = count_violations(color_arr, num_triples, quad_bitsets, quint_bitsets, num_quads, num_quints)

        if rk4 == 0 and bk5 == 0:
            return color_arr.tolist(), True

        max_iters = 200000
        temp_start = 2.0
        temp_end = 0.001

        for it in range(max_iters):
            if time.time() > deadline - 2:
                break
            if rk4 == 0 and bk5 == 0:
                return color_arr.tolist(), True

            temp = temp_start * (temp_end / temp_start) ** (it / max_iters)

            # Choose a triple to flip
            tidx = -1
            r = random.random()
            
            # Heuristic: pick from violating structures if they exist
            if rk4 > 0 and r < 0.5:
                # Find a violating red K4
                for _ in range(20):
                    qi = random.randint(0, num_quads - 1)
                    if (red_mask & quad_bitsets[qi]) == quad_bitsets[qi]:
                        qt = quad_triples_idx[qi]
                        tidx = qt[random.randint(0, 3)]
                        break
            elif bk5 > 0 and r < 0.85:
                # Find a violating blue K5
                for _ in range(20):
                    qi = random.randint(0, num_quints - 1)
                    if (red_mask & quint_bitsets[qi]) == 0:
                        qt = quint_triples_idx[qi]
                        tidx = qt[random.randint(0, 9)]
                        break

            if tidx == -1:
                tidx = random.randint(0, num_triples - 1)

            old_color = color_arr[tidx]
            new_color = 1 - old_color

            # Compute delta in violations
            delta_rk4 = 0
            delta_bk5 = 0

            # Update red counts for quads containing tidx
            for qi in triple_in_quads[tidx]:
                qt = quad_triples_idx[qi]
                others_all_red = True
                for p in range(4):
                    if qt[p] == tidx:
                        continue
                    if color_arr[qt[p]] != 0:
                        others_all_red = False
                        break
                
                if others_all_red:
                    if old_color == 0:
                        delta_rk4 -= 1
                    else:
                        delta_rk4 += 1

            # Update blue counts for quints containing tidx
            for qi in triple_in_quints[tidx]:
                qt = quint_triples_idx[qi]
                others_all_blue = True
                for p in range(10):
                    if qt[p] == tidx:
                        continue
                    if color_arr[qt[p]] != 1:
                        others_all_blue = False
                        break
                
                if others_all_blue:
                    if old_color == 1:
                        delta_bk5 -= 1
                    else:
                        delta_bk5 += 1

            delta = delta_rk4 + delta_bk5

            if delta <= 0 or random.random() < np.exp(-delta / max(temp, 1e-10)):
                color_arr[tidx] = new_color
                if new_color == 0:
                    red_mask |= (1 << tidx)
                else:
                    red_mask &= ~(1 << tidx)
                rk4 += delta_rk4
                bk5 += delta_bk5

        if rk4 == 0 and bk5 == 0:
            return color_arr.tolist(), True
        return color_arr.tolist(), False

    def genetic_crossover(parent1, parent2, structures):
        """
        Specialized crossover that preserves valid 4-subsets from parents.
        For each 4-subset (quad), if a parent has no red K4 on that quad,
        prefer inheriting that parent's coloring for the triples in that quad.
        This preserves local structural constraints preventing red K4s.
        """
        (triple_list, num_triples, triple_to_idx, quad_triples_idx, num_quads,
         quint_triples_idx, num_quints, triple_in_quads, triple_in_quints,
         quad_bitsets, quint_bitsets) = structures

        p1 = np.array(parent1, dtype=np.int8)
        p2 = np.array(parent2, dtype=np.int8)
        child = np.zeros(num_triples, dtype=np.int8)

        # Compute red masks for parents
        p1_red = 0
        p2_red = 0
        for i in range(num_triples):
            if p1[i] == 0:
                p1_red |= (1 << i)
            if p2[i] == 0:
                p2_red |= (1 << i)

        # For each quad, determine which parent has no red K4
        # Prefer the parent with fewer violations on this quad
        for qi in range(num_quads):
            qt = quad_triples_idx[qi]
            p1_violates = ((p1_red & quad_bitsets[qi]) == quad_bitsets[qi])
            p2_violates = ((p2_red & quad_bitsets[qi]) == quad_bitsets[qi])

            if not p1_violates and p2_violates:
                src = p1
            elif p1_violates and not p2_violates:
                src = p2
            elif not p1_violates and not p2_violates:
                # Both valid on this quad - pick randomly
                src = p1 if random.random() < 0.5 else p2
            else:
                # Both violate - pick the one with more reds (fewer reds = less likely to form K4)
                p1_reds = sum(1 for t in qt if p1[t] == 0)
                p2_reds = sum(1 for t in qt if p2[t] == 0)
                src = p1 if p1_reds <= p2_reds else p2

            for tidx in qt:
                child[tidx] = src[tidx]

        return child.tolist()

    def ga_mutation(child, mutation_rate, bias_to_break_blue_k5, structures):
        """
        Mutation biased toward breaking blue K5 violations.
        Triples in blue K5 violations are more likely to be mutated to red.
        """
        (triple_list, num_triples, triple_to_idx, quad_triples_idx, num_quads,
         quint_triples_idx, num_quints, triple_in_quads, triple_in_quints,
         quad_bitsets, quint_bitsets) = structures

        color_arr = np.array(child, dtype=np.int8)
        _, bk5, red_mask = count_violations(color_arr, num_triples, quad_bitsets, quint_bitsets, num_quads, num_quints)

        # Find triples in blue K5 violations
        blue_k5_triples = set()
        if bk5 > 0:
            for qi in range(num_quints):
                if (red_mask & quint_bitsets[qi]) == 0:
                    for tidx in quint_triples_idx[qi]:
                        blue_k5_triples.add(tidx)

        for i in range(num_triples):
            if i in blue_k5_triples and color_arr[i] == 1:
                # In a blue K5 and is blue - bias toward making it red
                if random.random() < mutation_rate * bias_to_break_blue_k5:
                    color_arr[i] = 0
            else:
                if random.random() < mutation_rate:
                    color_arr[i] = 1 - color_arr[i]

        return color_arr.tolist()

    def ga_local_search_hybrid(target_n, init_colors, deadline, structures):
        """Run GA-style local search: local search with periodic crossover/mutation."""
        (triple_list, num_triples, triple_to_idx, quad_triples_idx, num_quads,
         quint_triples_idx, num_quints, triple_in_quads, triple_in_quints,
         quad_bitsets, quint_bitsets) = structures

        # Start with local search
        colors, success = local_search(target_n, init_colors, deadline, structures)
        if success:
            return colors, True

        # If local search didn't succeed, try GA approach
        if time.time() > deadline - 5:
            return colors, False

        # Keep a small population
        pop_size = 4
        population = [colors]
        
        # Add some perturbed variants
        for _ in range(pop_size - 1):
            perturbed = colors[:]
            for i in range(num_triples):
                if random.random() < 0.05:
                    perturbed[i] = 1 - perturbed[i]
            population.append(perturbed)

        best_solution = colors
        best_violations = count_violations(np.array(colors, dtype=np.int8), num_triples, quad_bitsets, quint_bitsets, num_quads, num_quints)[0] + count_violations(np.array(colors, dtype=np.int8), num_triples, quad_bitsets, quint_bitsets, num_quads, num_quints)[1]

        max_generations = 50
        mutation_rate = 0.03
        bias_to_break_blue = 2.0

        for gen in range(max_generations):
            if time.time() > deadline - 3:
                break

            new_population = []

            # Elitism: keep best
            new_population.append(best_solution[:])

            # Crossover and mutation
            for _ in range(pop_size - 1):
                # Select two parents (tournament selection)
                p1_idx = random.randint(0, len(population) - 1)
                p2_idx = random.randint(0, len(population) - 1)
                
                # Crossover preserving valid 4-subsets
                child = genetic_crossover(population[p1_idx], population[p2_idx], structures)
                
                # Mutation
                child = ga_mutation(child, mutation_rate, bias_to_break_blue, structures)
                
                # Local search on child
                child, _ = local_search(target_n, child, deadline, structures)
                
                new_population.append(child)

            population = new_population

            # Update best
            for ind in population:
                rk4, bk5, _ = count_violations(np.array(ind, dtype=np.int8), num_triples, quad_bitsets, quint_bitsets, num_quads, num_quints)
                total_v = rk4 + bk5
                if total_v < best_violations:
                    best_violations = total_v
                    best_solution = ind[:]
                if total_v == 0:
                    return ind, True

        return best_solution, best_violations == 0

    # Try increasing n
    for target_n in range(10, 36):
        if time.time() > deadline - 15:
            break

        structures = build_structures(target_n)
        triple_list, num_triples = structures[0], structures[1]

        found = False
        max_restarts = 100 if target_n <= 20 else 50
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break

            if restart == 0:
                init = [0 if (a+b+c) % 3 == 0 else 1 for (a,b,c) in triple_list]
            elif restart == 1:
                init = [0 if (a+b+c) % 4 == 0 else 1 for (a,b,c) in triple_list]
            elif restart == 2:
                init = [0 if (a+b+c) % 5 == 0 else 1 for (a,b,c) in triple_list]
            elif restart == 3:
                init = [0 if (a+b+c) % 2 == 0 else 1 for (a,b,c) in triple_list]
            elif restart == 4:
                init = list(np.random.randint(0, 2, size=num_triples))
            elif restart == 5:
                init = [0 if random.random() < 0.4 else 1 for _ in range(num_triples)]
            elif restart == 6:
                init = [0 if random.random() < 0.3 else 1 for _ in range(num_triples)]
            elif restart == 7:
                base = [0 if (a+b+c) % 3 == 0 else 1 for (a,b,c) in triple_list]
                init = base[:]
                for i in range(num_triples):
                    if random.random() < 0.1:
                        init[i] = 1 - init[i]
            elif restart == 8:
                init = [0 if a % 2 == 0 else 1 for (a,b,c) in triple_list]
            elif restart == 9:
                init = [0 if c % 3 == 0 else 1 for (a,b,c) in triple_list]
            else:
                base = [0 if (a+b+c) % 4 == 0 else 1 for (a,b,c) in triple_list]
                init = base[:]
                for i in range(num_triples):
                    if random.random() < 0.15:
                        init[i] = 1 - init[i]

            # Use GA-hybrid approach for more diverse search
            if restart % 3 == 2:
                colors, success = ga_local_search_hybrid(target_n, init, deadline, structures)
            else:
                colors, success = local_search(target_n, init, deadline, structures)
            
            if success:
                if is_valid(target_n, colors):
                    best_n = target_n
                    best_colors = colors
                    found = True
                    break
                else:
                    found = False

        if found:
            continue
        if target_n > 12 and not found:
             if time.time() < deadline - 20:
                 for restart in range(max_restarts, max_restarts + 50):
                     if time.time() > deadline - 5:
                         break
                     init = list(np.random.randint(0, 2, size=num_triples))
                     # Alternate between pure local search and GA hybrid
                     if restart % 2 == 0:
                         colors, success = ga_local_search_hybrid(target_n, init, deadline, structures)
                     else:
                         colors, success = local_search(target_n, init, deadline, structures)
                     if success and is_valid(target_n, colors):
                         best_n = target_n
                         best_colors = colors
                         found = True
                         break
             if not found:
                 break

    # Final safety verification
    if best_n > 9 and not is_valid(best_n, best_colors):
        best_n = 9
        best_colors = mod_color(9, 3)

    return {'n': best_n, 'colors': best_colors}
