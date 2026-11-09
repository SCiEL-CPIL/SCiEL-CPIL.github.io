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
        """Precompute all structural mappings for a given n."""
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

        # Inverse mappings: for each triple, which quads/quints contain it
        triple_in_quads = [[] for _ in range(num_triples)]
        for qi, qt in enumerate(quad_triples_idx):
            for pos, tidx in enumerate(qt):
                triple_in_quads[tidx].append((qi, pos))

        triple_in_quints = [[] for _ in range(num_triples)]
        for qi, qt in enumerate(quint_triples_idx):
            for pos, tidx in enumerate(qt):
                triple_in_quints[tidx].append((qi, pos))

        quad_triples_np = np.array(quad_triples_idx, dtype=np.int32)
        quint_triples_np = np.array(quint_triples_idx, dtype=np.int32)

        return {
            'triple_list': triple_list,
            'num_triples': num_triples,
            'quad_triples_np': quad_triples_np,
            'quint_triples_np': quint_triples_np,
            'num_quads': num_quads,
            'num_quints': num_quints,
            'triple_in_quads': triple_in_quads,
            'triple_in_quints': triple_in_quints,
        }

    def compute_violations(color_arr, structs):
        """Compute violation counts using vectorized numpy."""
        quad_colors = structs['quad_triples_np']
        quad_red_counts = np.sum(color_arr[quad_colors] == 0, axis=1)
        rk4 = int(np.sum(quad_red_counts == 4))

        quint_colors = structs['quint_triples_np']
        quint_blue_counts = np.sum(color_arr[quint_colors] == 1, axis=1)
        bk5 = int(np.sum(quint_blue_counts == 10))

        return rk4, bk5, quad_red_counts, quint_blue_counts

    def local_search(color_arr, structs, max_iters, deadline, temp_start=2.0, temp_end=0.01):
        """Run simulated annealing local search, return (colors, rk4, bk5)."""
        num_triples = structs['num_triples']
        quad_triples_np = structs['quad_triples_np']
        quint_triples_np = structs['quint_triples_np']
        triple_in_quads = structs['triple_in_quads']
        triple_in_quints = structs['triple_in_quints']

        rk4, bk5, quad_red_counts, quint_blue_counts = compute_violations(color_arr, structs)

        if rk4 == 0 and bk5 == 0:
            return color_arr, rk4, bk5

        it = 0
        while it < max_iters:
            if time.time() > deadline - 2:
                break
            if rk4 == 0 and bk5 == 0:
                break

            temp = temp_start * (temp_end / temp_start) ** (it / max_iters)

            # Choose a triple to flip - bias toward violating structures
            tidx = -1
            r = random.random()
            if rk4 > 0 and r < 0.5:
                violating_quads = np.where(quad_red_counts == 4)[0]
                if len(violating_quads) > 0:
                    qi = violating_quads[random.randint(0, len(violating_quads)-1)]
                    qt = quad_triples_np[qi]
                    tidx = int(qt[random.randint(0, 3)])
            elif bk5 > 0 and r < 0.85:
                violating_quints = np.where(quint_blue_counts == 10)[0]
                if len(violating_quints) > 0:
                    qi = violating_quints[random.randint(0, len(violating_quints)-1)]
                    qt = quint_triples_np[qi]
                    tidx = int(qt[random.randint(0, 9)])

            if tidx == -1:
                tidx = random.randint(0, num_triples - 1)

            old_color = int(color_arr[tidx])

            # Compute delta_rk4 incrementally
            delta_rk4 = 0
            for (qi, pos) in triple_in_quads[tidx]:
                qt = quad_triples_np[qi]
                others = [qt[p] for p in range(4) if p != pos]
                others_red = all(int(color_arr[t]) == 0 for t in others)
                if old_color == 0 and others_red:
                    delta_rk4 -= 1
                elif old_color == 1 and others_red:
                    delta_rk4 += 1

            # Compute delta_bk5 incrementally
            delta_bk5 = 0
            for (qi, pos) in triple_in_quints[tidx]:
                qt = quint_triples_np[qi]
                others = [qt[p] for p in range(10) if p != pos]
                others_blue = all(int(color_arr[t]) == 1 for t in others)
                if old_color == 1 and others_blue:
                    delta_bk5 -= 1
                elif old_color == 0 and others_blue:
                    delta_bk5 += 1

            delta_total = delta_rk4 + delta_bk5

            # Acceptance criterion
            if delta_total <= 0:
                color_arr[tidx] = 1 - old_color
                rk4 += delta_rk4
                bk5 += delta_bk5
            else:
                if random.random() < np.exp(-delta_total / max(temp, 1e-10)):
                    color_arr[tidx] = 1 - old_color
                    rk4 += delta_rk4
                    bk5 += delta_bk5

            # Update violation counts for biased selection
            if rk4 == 0 and bk5 == 0:
                break
            quad_red_counts = np.sum(color_arr[structs['quad_triples_np']] == 0, axis=1)
            quint_blue_counts = np.sum(color_arr[structs['quint_triples_np']] == 1, axis=1)

            it += 1

        return color_arr, rk4, bk5

    def generate_initial_coloring(target_n, restart_idx, num_triples, triple_list):
        """Generate diverse initial colorings."""
        if restart_idx == 0:
            return np.array([0 if (a+b+c) % 3 == 0 else 1 for (a,b,c) in triple_list], dtype=np.int8)
        elif restart_idx == 1:
            return np.array([0 if (a+b+c) % 4 == 0 else 1 for (a,b,c) in triple_list], dtype=np.int8)
        elif restart_idx == 2:
            return np.array([0 if (a+b+c) % 5 == 0 else 1 for (a,b,c) in triple_list], dtype=np.int8)
        elif restart_idx == 3:
            return np.random.randint(0, 2, size=num_triples, dtype=np.int8)
        elif restart_idx == 4:
            arr = np.random.randint(0, 2, size=num_triples, dtype=np.int8)
            arr[np.random.random(num_triples) < 0.4] = 0
            return arr
        elif restart_idx == 5:
            arr = np.random.randint(0, 2, size=num_triples, dtype=np.int8)
            arr[np.random.random(num_triples) < 0.4] = 1
            return arr
        else:
            base_mod = 3 + (restart_idx % 3)
            base = np.array([0 if (a+b+c) % base_mod == 0 else 1 for (a,b,c) in triple_list], dtype=np.int8)
            flip_mask = np.random.random(num_triples) < 0.15
            base[flip_mask] = 1 - base[flip_mask]
            return base

    def structural_crossover(parent1, parent2, structs):
        """
        Specialized crossover that preserves valid 4-subsets from parents.
        
        For each quad (4 vertices), if parent1 has no red K4 on those vertices,
        we prefer parent1's coloring for the 4 triples in that quad.
        Similarly for quints with blue K5 avoidance.
        
        This preserves local structural constraints that prevent violations.
        """
        num_triples = structs['num_triples']
        quad_triples_np = structs['quad_triples_np']
        quint_triples_np = structs['quint_triples_np']
        num_quads = structs['num_quads']
        num_quints = structs['num_quints']

        child = np.empty(num_triples, dtype=np.int8)
        votes = np.zeros(num_triples, dtype=np.int8)
        weights = np.zeros(num_triples, dtype=np.float64)

        # For each quad, decide which parent's coloring to use
        for qi in range(num_quads):
            qt = quad_triples_np[qi]
            # Check if parent1 has a red K4 on this quad
            p1_red = all(int(parent1[t]) == 0 for t in qt)
            # Check if parent2 has a red K4 on this quad
            p2_red = all(int(parent2[t]) == 0 for t in qt)

            if p1_red and not p2_red:
                # Parent2 is better for this quad
                for tidx in qt:
                    votes[tidx] += 1
                    weights[tidx] += 2.0
            elif p2_red and not p1_red:
                # Parent1 is better for this quad
                for tidx in qt:
                    votes[tidx] += 0
                    weights[tidx] += 2.0
            else:
                # Neither or both have red K4 - random choice
                if random.random() < 0.5:
                    for tidx in qt:
                        votes[tidx] += 1
                        weights[tidx] += 1.0
                else:
                    for tidx in qt:
                        votes[tidx] += 0
                        weights[tidx] += 1.0

        # For each quint, decide based on blue K5
        for qi in range(num_quints):
            qt = quint_triples_np[qi]
            # Check if parent1 has a blue K5 on this quint
            p1_blue = all(int(parent1[t]) == 1 for t in qt)
            # Check if parent2 has a blue K5 on this quint
            p2_blue = all(int(parent2[t]) == 1 for t in qt)

            if p1_blue and not p2_blue:
                # Parent2 is better
                for tidx in qt:
                    votes[tidx] += 1
                    weights[tidx] += 1.5
            elif p2_blue and not p1_blue:
                # Parent1 is better
                for tidx in qt:
                    votes[tidx] += 0
                    weights[tidx] += 1.5
            else:
                if random.random() < 0.5:
                    for tidx in qt:
                        votes[tidx] += 1
                        weights[tidx] += 0.5
                else:
                    for tidx in qt:
                        votes[tidx] += 0
                        weights[tidx] += 0.5

        # Resolve conflicts: use weighted vote
        for i in range(num_triples):
            if weights[i] > 0:
                if votes[i] / weights[i] > 0.5:
                    child[i] = parent2[i]
                else:
                    child[i] = parent1[i]
            else:
                # No constraint info, random
                child[i] = random.choice([parent1[i], parent2[i]])

        return child

    def biased_mutation(child, structs, mutation_rate=0.05):
        """
        Mutation biased toward breaking blue K5 violations.
        
        For each triple, compute how many blue K5 violations it participates in.
        Higher probability of flipping triples that are in more blue K5s.
        """
        num_triples = structs['num_triples']
        quint_triples_np = structs['quint_triples_np']
        triple_in_quints = structs['triple_in_quints']
        num_quints = structs['num_quints']

        # Count blue K5 violations per triple
        blue_k5_count = np.zeros(num_triples, dtype=np.int32)
        for qi in range(num_quints):
            qt = quint_triples_np[qi]
            if all(int(child[t]) == 1 for t in qt):
                for tidx in qt:
                    blue_k5_count[tidx] += 1

        # Also count red K4 violations
        quad_triples_np = structs['quad_triples_np']
        num_quads = structs['num_quads']
        red_k4_count = np.zeros(num_triples, dtype=np.int32)
        for qi in range(num_quads):
            qt = quad_triples_np[qi]
            if all(int(child[t]) == 0 for t in qt):
                for tidx in qt:
                    red_k4_count[tidx] += 1

        # Mutation probability based on violation counts
        for i in range(num_triples):
            # Higher probability if in more violations
            prob = mutation_rate * (1.0 + blue_k5_count[i] * 0.5 + red_k4_count[i] * 0.5)
            if random.random() < min(prob, 0.5):
                child[i] = 1 - child[i]

        return child

    def genetic_algorithm(target_n, structs, deadline, pop_size=20, generations=100):
        """
        Genetic algorithm with structural crossover and biased mutation.
        
        Population of colorings, evolved to minimize violations.
        """
        num_triples = structs['num_triples']
        triple_list = structs['triple_list']

        # Initialize population with diverse colorings
        population = []
        for i in range(pop_size):
            coloring = generate_initial_coloring(target_n, i, num_triples, triple_list)
            population.append(coloring)

        # Evaluate initial population
        scores = []
        for coloring in population:
            rk4, bk5, _, _ = compute_violations(coloring, structs)
            scores.append(rk4 + bk5)

        best_idx = int(np.argmin(scores))
        best_coloring = population[best_idx].copy()
        best_score = scores[best_idx]

        for gen in range(generations):
            if time.time() > deadline - 5:
                break

            if best_score == 0:
                break

            # Selection: tournament selection
            new_population = []

            # Elitism: keep best
            new_population.append(best_coloring.copy())

            while len(new_population) < pop_size:
                if time.time() > deadline - 3:
                    break

                # Tournament selection
                t1 = random.randint(0, len(population) - 1)
                t2 = random.randint(0, len(population) - 1)
                t3 = random.randint(0, len(population) - 1)

                candidates = [t1, t2, t3]
                winner = min(candidates, key=lambda i: scores[i])
                parent1 = population[winner]

                # Second parent
                t4 = random.randint(0, len(population) - 1)
                t5 = random.randint(0, len(population) - 1)
                t6 = random.randint(0, len(population) - 1)
                candidates2 = [t4, t5, t6]
                winner2 = min(candidates2, key=lambda i: scores[i])
                parent2 = population[winner2]

                # Structural crossover
                child = structural_crossover(parent1, parent2, structs)

                # Biased mutation
                child = biased_mutation(child, structs, mutation_rate=0.05)

                new_population.append(child)

            # Evaluate new population
            population = new_population
            new_scores = []
            for coloring in population:
                rk4, bk5, _, _ = compute_violations(coloring, structs)
                new_scores.append(rk4 + bk5)
            scores = new_scores

            # Update best
            best_idx = int(np.argmin(scores))
            if scores[best_idx] < best_score:
                best_score = scores[best_idx]
                best_coloring = population[best_idx].copy()

            # Periodically inject diversity
            if gen % 10 == 9:
                diversity_coloring = generate_initial_coloring(
                    target_n, gen, num_triples, triple_list)
                population[-1] = diversity_coloring
                rk4, bk5, _, _ = compute_violations(diversity_coloring, structs)
                scores[-1] = rk4 + bk5

        return best_coloring, best_score

    def hybrid_search(target_n, structs, deadline):
        """
        Hybrid search: combine GA with local search refinement.
        
        Strategy:
        1. Run GA to find promising colorings
        2. Refine best GA results with local search
        3. If local search succeeds, return valid coloring
        """
        num_triples = structs['num_triples']

        # Phase 1: Genetic Algorithm
        ga_result, ga_score = genetic_algorithm(target_n, structs, deadline, pop_size=15, generations=80)

        if ga_score == 0:
            return ga_result

        # Phase 2: Local search refinement on GA result
        refined, rk4, bk5 = local_search(ga_result, structs, max_iters=30000, deadline=deadline)

        if rk4 == 0 and bk5 == 0:
            return refined

        # Phase 3: Try more GA restarts if time allows
        if time.time() < deadline - 15:
            for restart in range(3):
                if time.time() > deadline - 5:
                    break
                ga_result2, ga_score2 = genetic_algorithm(target_n, structs, deadline, pop_size=10, generations=50)
                if ga_score2 < ga_score:
                    ga_result = ga_result2
                    ga_score = ga_score2
                if ga_score2 == 0:
                    return ga_result2

        # Phase 4: Final local search
        final, rk4, bk5 = local_search(ga_result, structs, max_iters=50000, deadline=deadline)

        if rk4 == 0 and bk5 == 0:
            return final

        return final

    # Main loop: try increasing n with hybrid search
    for target_n in range(10, 35):
        if time.time() > deadline - 10:
            break

        structs = build_structures(target_n)
        num_triples = structs['num_triples']

        found = False
        result_colors = None

        # Strategy 1: Hybrid GA + local search
        result_colors = hybrid_search(target_n, structs, deadline)
        if result_colors is not None:
            rk4, bk5, _, _ = compute_violations(result_colors, structs)
            if rk4 == 0 and bk5 == 0:
                found = True

        # Strategy 2: If hybrid failed, try direct local search with multiple restarts
        if not found and time.time() < deadline - 5:
            max_restarts = 10
            for restart in range(max_restarts):
                if time.time() > deadline - 3:
                    break

                color_arr = generate_initial_coloring(target_n, restart, num_triples, structs['triple_list'])
                color_arr, rk4, bk5 = local_search(color_arr, structs, max_iters=30000, deadline=deadline)

                if rk4 == 0 and bk5 == 0:
                    result_colors = color_arr
                    found = True
                    break

        if found:
            colors_list = result_colors.tolist()
            if is_valid(target_n, colors_list):
                best_n = target_n
                best_colors = colors_list
            else:
                break
        else:
            if target_n >= 25:
                break

    return {'n': best_n, 'colors': best_colors}
