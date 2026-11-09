import numpy as np
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 112  # safety margin before 120s kill

    # Fallback: n=9 modular coloring (valid)
    def mod_color(n, mod=3):
        return [0 if (a + b + c) % mod == 0 else 1 for a, b, c in combinations(range(n), 3)]

    best_n = 9
    best_colors = mod_color(9, 3)

    def build_structures(n):
        triple_list = list(combinations(range(n), 3))
        num_triples = len(triple_list)
        triple_to_idx = {t: i for i, t in enumerate(triple_list)}

        all_quads = list(combinations(range(n), 4))
        quad_triples_idx = []
        for a, b, c, d in all_quads:
            quad_triples_idx.append([
                triple_to_idx[(a,b,c)], triple_to_idx[(a,b,d)],
                triple_to_idx[(a,c,d)], triple_to_idx[(b,c,d)]
            ])
        num_quads = len(quad_triples_idx)

        all_quints = list(combinations(range(n), 5))
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

        return (triple_list, num_triples, triple_to_idx, quad_triples_idx, num_quads,
                quint_triples_idx, num_quints, triple_in_quads, triple_in_quints)

    def count_violations_fast(color_arr, triple_in_quads, triple_in_quints, num_quads, num_quints, quad_triples_idx, quint_triples_idx):
        """Count red K4 violations and blue K5 violations using incremental counting."""
        # For each quad, count how many of its 4 triples are red (color 0)
        quad_red_counts = np.zeros(num_quads, dtype=np.int8)
        for qi in range(num_quads):
            qt = quad_triples_idx[qi]
            for tidx in qt:
                if color_arr[tidx] == 0:
                    quad_red_counts[qi] += 1
        rk4 = int(np.sum(quad_red_counts == 4))

        # For each quint, count how many of its 10 triples are blue (color 1)
        quint_blue_counts = np.zeros(num_quints, dtype=np.int8)
        for qi in range(num_quints):
            qt = quint_triples_idx[qi]
            for tidx in qt:
                if color_arr[tidx] == 1:
                    quint_blue_counts[qi] += 1
        bk5 = int(np.sum(quint_blue_counts == 10))

        return rk4, bk5

    def local_search(n, init_colors, deadline, structures=None):
        """Run local search on a single coloring to reduce violations."""
        if structures is None:
            structures = build_structures(n)
        (triple_list, num_triples, triple_to_idx, quad_triples_idx, num_quads,
         quint_triples_idx, num_quints, triple_in_quads, triple_in_quints) = structures

        color_arr = np.array(init_colors, dtype=np.int8).copy()

        rk4, bk5 = count_violations_fast(color_arr, triple_in_quads, triple_in_quints, num_quads, num_quints, quad_triples_idx, quint_triples_idx)
        if rk4 == 0 and bk5 == 0:
            return color_arr.tolist(), True

        max_iters = 50000
        temp = 1.0
        temp_min = 0.01
        cooling = 0.9999

        # Track counts incrementally
        quad_red_counts = np.zeros(num_quads, dtype=np.int8)
        for qi in range(num_quads):
            qt = quad_triples_idx[qi]
            for tidx in qt:
                if color_arr[tidx] == 0:
                    quad_red_counts[qi] += 1

        quint_blue_counts = np.zeros(num_quints, dtype=np.int8)
        for qi in range(num_quints):
            qt = quint_triples_idx[qi]
            for tidx in qt:
                if color_arr[tidx] == 1:
                    quint_blue_counts[qi] += 1

        for it in range(max_iters):
            if time.time() > deadline - 2:
                break
            if rk4 == 0 and bk5 == 0:
                return color_arr.tolist(), True

            # Choose a triple to flip, biased toward violating structures
            tidx = -1
            r = random.random()
            if rk4 > 0 and r < 0.4:
                # Pick a random quad that is a red K4 violation
                for _ in range(10):
                    qi = random.randint(0, num_quads - 1)
                    if quad_red_counts[qi] == 4:
                        qt = quad_triples_idx[qi]
                        tidx = qt[random.randint(0, 3)]
                        break
            elif bk5 > 0 and r < 0.8:
                # Pick a random quint that is a blue K5 violation
                for _ in range(10):
                    qi = random.randint(0, num_quints - 1)
                    if quint_blue_counts[qi] == 10:
                        qt = quint_triples_idx[qi]
                        tidx = qt[random.randint(0, 9)]
                        break

            if tidx == -1:
                tidx = random.randint(0, num_triples - 1)

            old_color = color_arr[tidx]
            new_color = 1 - old_color

            # Compute delta in violations
            delta_rk4 = 0
            for qi in triple_in_quads[tidx]:
                if old_color == 0:  # was red, becoming blue
                    if quad_red_counts[qi] == 4:
                        delta_rk4 -= 1  # breaks a red K4
                else:  # was blue, becoming red
                    if quad_red_counts[qi] == 3:
                        delta_rk4 += 1  # creates a red K4

            delta_bk5 = 0
            for qi in triple_in_quints[tidx]:
                if old_color == 1:  # was blue, becoming red
                    if quint_blue_counts[qi] == 10:
                        delta_bk5 -= 1  # breaks a blue K5
                else:  # was red, becoming blue
                    if quint_blue_counts[qi] == 9:
                        delta_bk5 += 1  # creates a blue K5

            delta = delta_rk4 + delta_bk5

            if delta <= 0 or random.random() < np.exp(-delta / max(temp, 1e-10)):
                # Accept the flip
                color_arr[tidx] = new_color
                for qi in triple_in_quads[tidx]:
                    if old_color == 0:
                        quad_red_counts[qi] -= 1
                    else:
                        quad_red_counts[qi] += 1
                for qi in triple_in_quints[tidx]:
                    if old_color == 1:
                        quint_blue_counts[qi] -= 1
                    else:
                        quint_blue_counts[qi] += 1
                rk4 += delta_rk4
                bk5 += delta_bk5
            else:
                # Revert (no change since we didn't apply)
                pass

            temp *= cooling
            if temp < temp_min:
                temp = temp_min

        return color_arr.tolist(), False

    def ga_crossover(parent1, parent2, structures):
        """Specialized crossover that preserves locally-valid 4-subsets from parents."""
        (triple_list, num_triples, triple_to_idx, quad_triples_idx, num_quads,
         quint_triples_idx, num_quints, triple_in_quads, triple_in_quints) = structures

        p1 = np.array(parent1, dtype=np.int8)
        p2 = np.array(parent2, dtype=np.int8)
        child = np.zeros(num_triples, dtype=np.int8)

        # For each triple, decide which parent to inherit from
        # Bias: prefer the parent that has fewer violations in the local neighborhood
        # (i.e., the parent where this triple's containing quads/quints have fewer violations)

        # Precompute per-parent violation contribution for each triple
        # A triple "contributes" to violations if it's part of a violating structure
        # For simplicity, use a heuristic: for each triple, check if it's in a violating
        # structure in each parent, and prefer the parent with fewer such violations

        # Quick approach: for each triple, randomly pick a parent but with bias
        # toward the parent where the triple is "safer"
        for tidx in range(num_triples):
            # Count how many containing quads are red-K4 violations in each parent
            # and how many containing quints are blue-K5 violations in each parent
            p1_risk = 0
            p2_risk = 0

            # Check quads containing this triple
            for qi in triple_in_quads[tidx]:
                qt = quad_triples_idx[qi]
                # Check if all 4 triples in this quad are red (color 0) in parent1
                all_red_p1 = all(p1[t] == 0 for t in qt)
                all_red_p2 = all(p2[t] == 0 for t in qt)
                if all_red_p1:
                    p1_risk += 1
                if all_red_p2:
                    p2_risk += 1

            # Check quints containing this triple (sample for speed)
            for qi in triple_in_quints[tidx]:
                qt = quint_triples_idx[qi]
                # Check if all 10 triples in this quint are blue (color 1) in parent
                all_blue_p1 = all(p1[t] == 1 for t in qt)
                all_blue_p2 = all(p2[t] == 1 for t in qt)
                if all_blue_p1:
                    p1_risk += 1
                if all_blue_p2:
                    p2_risk += 1

            # Prefer the parent with lower risk; if tied, random
            if p1_risk < p2_risk:
                child[tidx] = p1[tidx]
            elif p2_risk < p1_risk:
                child[tidx] = p2[tidx]
            else:
                child[tidx] = p1[tidx] if random.random() < 0.5 else p2[tidx]

        return child.tolist()

    def ga_mutation(colors, structures, mutation_rate=0.05):
        """Mutation biased toward breaking blue K5 violations."""
        (triple_list, num_triples, triple_to_idx, quad_triples_idx, num_quads,
         quint_triples_idx, num_quints, triple_in_quads, triple_in_quints) = structures

        color_arr = np.array(colors, dtype=np.int8).copy()

        # Identify triples that are part of blue K5 violations
        # and bias mutation toward flipping them to red
        blue_k5_triples = set()
        for qi in range(num_quints):
            qt = quint_triples_idx[qi]
            if all(color_arr[t] == 1 for t in qt):
                for t in qt:
                    blue_k5_triples.add(t)

        for tidx in range(num_triples):
            if random.random() < mutation_rate:
                if tidx in blue_k5_triples:
                    # Higher probability to flip blue K5 members to red
                    if color_arr[tidx] == 1 and random.random() < 0.7:
                        color_arr[tidx] = 0
                else:
                    # Normal random flip
                    color_arr[tidx] = 1 - color_arr[tidx]

        return color_arr.tolist()

    def genetic_algorithm(n, deadline, structures=None, pop_size=20, max_generations=50):
        """Run a genetic algorithm to find a valid coloring."""
        if structures is None:
            structures = build_structures(n)
        (triple_list, num_triples, triple_to_idx, quad_triples_idx, num_quads,
         quint_triples_idx, num_quints, triple_in_quads, triple_in_quints) = structures

        # Initialize population with diverse strategies
        population = []

        # Strategy 1: Random with low red probability
        p_red = 0.05
        colors = np.random.binomial(1, p_red, num_triples).tolist()
        population.append(colors)

        # Strategy 2: Modular coloring mod 3
        population.append([0 if (a + b + c) % 3 == 0 else 1 for a, b, c in triple_list])

        # Strategy 3: Modular coloring mod 4
        population.append([0 if (a + b + c) % 4 == 0 else 1 for a, b, c in triple_list])

        # Strategy 4: Vertex-based pattern
        population.append([0 if (a % 2 == 0 and b % 2 == 1) or (a % 2 == 1 and b % 2 == 0) else 1 for a, b, c in triple_list])

        # Strategy 5: Random with higher red probability
        p_red = 0.1
        colors = np.random.binomial(1, p_red, num_triples).tolist()
        population.append(colors)

        # Fill rest with random
        while len(population) < pop_size:
            p_red = random.uniform(0.03, 0.15)
            colors = np.random.binomial(1, p_red, num_triples).tolist()
            population.append(colors)

        # Evaluate initial population
        def fitness(colors):
            rk4, bk5 = count_violations_fast(np.array(colors, dtype=np.int8), triple_in_quads, triple_in_quints, num_quads, num_quints, quad_triples_idx, quint_triples_idx)
            return rk4 + bk5

        for i in range(len(population)):
            if time.time() > deadline - 3:
                break
            # Local search on each individual
            improved, valid = local_search(n, population[i], deadline - 3, structures)
            population[i] = improved
            if valid:
                return improved

        # GA loop
        best_individual = None
        best_fitness = float('inf')

        for gen in range(max_generations):
            if time.time() > deadline - 3:
                break

            # Evaluate fitness
            fitnesses = [fitness(ind) for ind in population]

            # Track best
            min_idx = int(np.argmin(fitnesses))
            if fitnesses[min_idx] < best_fitness:
                best_fitness = fitnesses[min_idx]
                best_individual = population[min_idx]
                if best_fitness == 0:
                    return best_individual

            # Selection: tournament selection
            new_population = []

            # Elitism: keep the best individual
            new_population.append(population[min_idx])

            while len(new_population) < pop_size:
                if time.time() > deadline - 3:
                    break

                # Tournament selection
                tournament_size = 3
                contenders = random.sample(range(pop_size), min(tournament_size, pop_size))
                winner = min(contenders, key=lambda i: fitnesses[i])
                loser_candidates = [i for i in range(pop_size) if i != winner]
                loser = random.choice(loser_candidates)

                # Crossover
                if random.random() < 0.7:
                    child = ga_crossover(population[winner], population[loser], structures)
                else:
                    child = population[winner][:]

                # Mutation
                child = ga_mutation(child, structures, mutation_rate=0.03)

                # Local search on child
                improved, valid = local_search(n, child, deadline - 3, structures)
                new_population.append(improved)
                if valid:
                    return improved

            population = new_population[:pop_size]

        return best_individual

    # Try to find valid colorings for decreasing n
    for n in range(34, 8, -1):
        if time.time() > deadline - 5:
            break

        try:
            structures = build_structures(n)
        except Exception:
            break

        # Try GA
        result = genetic_algorithm(n, deadline - 5, structures, pop_size=15, max_generations=30)
        if result is not None:
            # Verify
            rk4, bk5 = count_violations_fast(np.array(result, dtype=np.int8),
                                             structures[7], structures[8],
                                             structures[5], structures[6],
                                             structures[3], structures[5])
            if rk4 == 0 and bk5 == 0:
                best_n = n
                best_colors = result
                break

        # Also try direct local search with different initializations
        for restart in range(10):
            if time.time() > deadline - 5:
                break

            if restart == 0:
                init = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in structures[0]]
            elif restart == 1:
                init = [0 if (a + b + c) % 4 == 0 else 1 for a, b, c in structures[0]]
            else:
                p_red = 0.03 + 0.01 * restart
                init = np.random.binomial(1, p_red, structures[1]).tolist()

            improved, valid = local_search(n, init, deadline - 5, structures)
            if valid:
                best_n = n
                best_colors = improved
                break

        if best_n >= n:
            break

    if best_n > 9:
        return {'n': best_n, 'colors': best_colors}
    else:
        return {'n': 9, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(9), 3)]}
