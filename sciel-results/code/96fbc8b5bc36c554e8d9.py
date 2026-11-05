import numpy as np
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safe margin before 150s kill

    def get_triple_indices(n):
        """Return list of all triples (a,b,c) with a<b<c."""
        return list(combinations(range(n), 3))

    def build_structure(n):
        """Build K4 and K5 structures for fast violation checking."""
        triples = get_triple_indices(n)
        num_triples = len(triples)
        triple_to_idx = {t: i for i, t in enumerate(triples)}

        # K4: 4 vertices -> 4 triples
        k4s = []
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            k4s.append((
                triple_to_idx[(a, b, c)],
                triple_to_idx[(a, b, d)],
                triple_to_idx[(a, c, d)],
                triple_to_idx[(b, c, d)]
            ))

        # K5: 5 vertices -> 10 triples
        k5s = []
        for quint in combinations(range(n), 5):
            idxs = []
            for triple in combinations(quint, 3):
                idxs.append(triple_to_idx[triple])
            k5s.append(tuple(idxs))

        # Reverse indices: triple_idx -> list of clique indices
        triple_to_k4s = [[] for _ in range(num_triples)]
        triple_to_k5s = [[] for _ in range(num_triples)]

        for i, idxs in enumerate(k4s):
            for t in idxs:
                triple_to_k4s[t].append(i)

        for i, idxs in enumerate(k5s):
            for t in idxs:
                triple_to_k5s[t].append(i)

        return triples, num_triples, k4s, k5s, triple_to_k4s, triple_to_k5s

    def count_violations(colors, k4s, k5s):
        """Count red K4s and blue K5s."""
        red_count = 0
        for idxs in k4s:
            if all(colors[i] == 0 for i in idxs):
                red_count += 1
        blue_count = 0
        for idxs in k5s:
            if all(colors[i] == 1 for i in idxs):
                blue_count += 1
        return red_count + blue_count

    def delta_cost(colors, t, old_color, new_color, k4s, k5s, triple_to_k4s, triple_to_k5s):
        """Compute cost delta when flipping triple t from old_color to new_color."""
        delta = 0

        # Check K4s containing t
        for k4_idx in triple_to_k4s[t]:
            idxs = k4s[k4_idx]
            red_count = sum(1 for i in idxs if colors[i] == 0)

            if old_color == 0:
                if red_count == 4:
                    delta -= 1
            else:
                if red_count == 3:
                    delta += 1

        # Check K5s containing t
        for k5_idx in triple_to_k5s[t]:
            idxs = k5s[k5_idx]
            blue_count = sum(1 for i in idxs if colors[i] == 1)

            if old_color == 1:
                if blue_count == 10:
                    delta -= 1
            else:
                if blue_count == 9:
                    delta += 1

        return delta

    def gf3_structure(n, variant=0):
        """
        Generate structured colorings using finite field GF(3) arithmetic.
        Assign each vertex a value in GF(3), then derive triple color from
        linear combinations of vertex values.
        """
        triples = get_triple_indices(n)
        colors = np.zeros(len(triples), dtype=np.int8)

        # Assign vertex values in GF(3)
        if variant == 0:
            # v[i] = i % 3
            v = np.array([i % 3 for i in range(n)], dtype=np.int8)
        elif variant == 1:
            # v[i] = (2*i) % 3
            v = np.array([(2 * i) % 3 for i in range(n)], dtype=np.int8)
        elif variant == 2:
            # v[i] = (i*i) % 3
            v = np.array([(i * i) % 3 for i in range(n)], dtype=np.int8)
        elif variant == 3:
            # v[i] = (i + (i//3)) % 3
            v = np.array([(i + (i // 3)) % 3 for i in range(n)], dtype=np.int8)
        elif variant == 4:
            # v[i] = (2*i + (i//3)) % 3
            v = np.array([(2 * i + (i // 3)) % 3 for i in range(n)], dtype=np.int8)
        elif variant == 5:
            # v[i] = (i*i + i) % 3
            v = np.array([(i * i + i) % 3 for i in range(n)], dtype=np.int8)
        else:
            # v[i] = (i + (i//2)) % 3
            v = np.array([(i + (i // 2)) % 3 for i in range(n)], dtype=np.int8)

        # Different linear combinations for triple color
        if variant == 0:
            # (v[a] + v[b] + v[c]) % 3 == 0 -> red
            for idx, (a, b, c) in enumerate(triples):
                colors[idx] = 0 if (v[a] + v[b] + v[c]) % 3 == 0 else 1
        elif variant == 1:
            # (v[a] + 2*v[b] + v[c]) % 3 == 0 -> red
            for idx, (a, b, c) in enumerate(triples):
                colors[idx] = 0 if (v[a] + 2 * v[b] + v[c]) % 3 == 0 else 1
        elif variant == 2:
            # (v[a] + v[b] - v[c]) % 3 == 0 -> red
            for idx, (a, b, c) in enumerate(triples):
                colors[idx] = 0 if (v[a] + v[b] - v[c]) % 3 == 0 else 1
        elif variant == 3:
            # (v[a]*v[b] + v[b]*v[c] + v[c]*v[a]) % 3 == 0 -> red
            for idx, (a, b, c) in enumerate(triples):
                colors[idx] = 0 if (v[a] * v[b] + v[b] * v[c] + v[c] * v[a]) % 3 == 0 else 1
        elif variant == 4:
            # (v[a]*v[b]*v[c]) % 3 == 0 -> red
            for idx, (a, b, c) in enumerate(triples):
                colors[idx] = 0 if (v[a] * v[b] * v[c]) % 3 == 0 else 1
        elif variant == 5:
            # (v[a] + v[b] + v[c]) % 3 == 1 -> red
            for idx, (a, b, c) in enumerate(triples):
                colors[idx] = 0 if (v[a] + v[b] + v[c]) % 3 == 1 else 1
        else:
            # (v[a] + v[b] + v[c]) % 3 == 2 -> red
            for idx, (a, b, c) in enumerate(triples):
                colors[idx] = 0 if (v[a] + v[b] + v[c]) % 3 == 2 else 1

        return colors

    def pair_based_structure(n, variant=0):
        """
        Generate structured colorings using pair-based values.
        Assign each pair (i,j) a value, then derive triple color from
        sum of pair values.
        """
        triples = get_triple_indices(n)
        colors = np.zeros(len(triples), dtype=np.int8)

        # Define pair value function
        if variant == 0:
            # pair_val(i,j) = (i+j) % 3
            def pair_val(i, j):
                return (i + j) % 3
        elif variant == 1:
            # pair_val(i,j) = (i*j) % 3
            def pair_val(i, j):
                return (i * j) % 3
        elif variant == 2:
            # pair_val(i,j) = (i-j) % 3
            def pair_val(i, j):
                return (i - j) % 3
        elif variant == 3:
            # pair_val(i,j) = (i+j) % 2
            def pair_val(i, j):
                return (i + j) % 2
        else:
            # pair_val(i,j) = (i*j + i + j) % 3
            def pair_val(i, j):
                return (i * j + i + j) % 3

        # Derive triple color from pair values
        for idx, (a, b, c) in enumerate(triples):
            s = pair_val(a, b) + pair_val(b, c) + pair_val(a, c)
            if variant < 3:
                # GF(3) based
                colors[idx] = 0 if s % 3 == 0 else 1
            else:
                # GF(2) based
                colors[idx] = 0 if s % 2 == 0 else 1

        return colors

    def structured_init(n, variant=0):
        """Generate structured initial colorings."""
        triples = get_triple_indices(n)
        colors = []

        if variant < 7:
            # GF(3) vertex-based
            return gf3_structure(n, variant)
        elif variant < 12:
            # Pair-based
            return pair_based_structure(n, variant - 7)
        elif variant == 12:
            # (a+b+c) mod 3
            for a, b, c in triples:
                colors.append(0 if (a + b + c) % 3 == 0 else 1)
        elif variant == 13:
            # (a+b+2*c) mod 3
            for a, b, c in triples:
                colors.append(0 if (a + b + 2 * c) % 3 == 0 else 1)
        elif variant == 14:
            # (a*b + b*c + c*a) mod 2
            for a, b, c in triples:
                colors.append((a * b + b * c + c * a) % 2)
        elif variant == 15:
            # (a+b+c) mod 2
            for a, b, c in triples:
                colors.append((a + b + c) % 2)
        elif variant == 16:
            # (a*b*c) mod 2
            for a, b, c in triples:
                colors.append((a * b * c) % 2)
        elif variant == 17:
            # (a+b) mod 3 == 0 -> red
            for a, b, c in triples:
                colors.append(0 if (a + b) % 3 == 0 else 1)
        elif variant == 18:
            # (b+c) mod 3 == 0 -> red
            for a, b, c in triples:
                colors.append(0 if (b + c) % 3 == 0 else 1)
        else:
            # (a+c) mod 3 == 0 -> red
            for a, b, c in triples:
                colors.append(0 if (a + c) % 3 == 0 else 1)

        return np.array(colors, dtype=np.int8)

    def local_search(n, time_limit, seed=0, init_variant=0):
        """Local search with simulated annealing."""
        triples, num_triples, k4s, k5s, triple_to_k4s, triple_to_k5s = build_structure(n)
        np.random.seed(seed)

        # Initialize with structured coloring
        colors = structured_init(n, init_variant)
        current_cost = count_violations(colors, k4s, k5s)
        best_cost = current_cost
        best_colors = colors.copy()

        if current_cost == 0:
            return colors, True

        # Track which cliques are violated
        violated_k4s = set()
        violated_k5s = set()
        for i, idxs in enumerate(k4s):
            if all(colors[j] == 0 for j in idxs):
                violated_k4s.add(i)
        for i, idxs in enumerate(k5s):
            if all(colors[j] == 1 for j in idxs):
                violated_k5s.add(i)

        temp = 1.0
        cool_rate = 0.9998
        min_temp = 0.005

        step = 0
        no_improve = 0
        max_no_improve = 5000

        while time.time() < time_limit:
            step += 1

            # Strategy: 70% focus on violating cliques, 30% random
            if (violated_k4s or violated_k5s) and np.random.rand() < 0.7:
                # Pick a random violated clique and flip one of its triples
                if violated_k4s and (not violated_k5s or np.random.rand() < 0.5):
                    k4_idx = np.random.choice(list(violated_k4s))
                    idxs = k4s[k4_idx]
                    t = idxs[np.random.randint(4)]
                else:
                    k5_idx = np.random.choice(list(violated_k5s))
                    idxs = k5s[k5_idx]
                    t = idxs[np.random.randint(10)]
            else:
                t = np.random.randint(num_triples)

            old_color = colors[t]
            new_color = 1 - old_color

            dc = delta_cost(colors, t, old_color, new_color, k4s, k5s, triple_to_k4s, triple_to_k5s)
            new_cost = current_cost + dc

            # Acceptance
            if dc <= 0:
                accept = True
            else:
                if temp > min_temp:
                    accept = np.random.rand() < np.exp(-dc / temp)
                else:
                    accept = False

            if accept:
                colors[t] = new_color
                current_cost = new_cost

                # Update violation sets incrementally
                for k4_i in triple_to_k4s[t]:
                    idxs = k4s[k4_i]
                    if all(colors[j] == 0 for j in idxs):
                        violated_k4s.add(k4_i)
                    else:
                        violated_k4s.discard(k4_i)

                for k5_i in triple_to_k5s[t]:
                    idxs = k5s[k5_i]
                    if all(colors[j] == 1 for j in idxs):
                        violated_k5s.add(k5_i)
                    else:
                        violated_k5s.discard(k5_i)

                if current_cost < best_cost:
                    best_cost = current_cost
                    best_colors = colors.copy()
                    no_improve = 0
                    if best_cost == 0:
                        return best_colors, True
                else:
                    no_improve += 1
            else:
                no_improve += 1

            temp *= cool_rate

            if no_improve > max_no_improve and temp > min_temp:
                temp = max(0.5, temp * 1.5)
                no_improve = 0

            if no_improve > max_no_improve * 4:
                colors = best_colors.copy()
                current_cost = best_cost
                temp = max(0.8, temp)
                no_improve = 0
                # Rebuild violation sets
                violated_k4s = set()
                violated_k5s = set()
                for i, idxs in enumerate(k4s):
                    if all(colors[j] == 0 for j in idxs):
                        violated_k4s.add(i)
                for i, idxs in enumerate(k5s):
                    if all(colors[j] == 1 for j in idxs):
                        violated_k5s.add(i)

        return best_colors, (best_cost == 0)

    def genetic_algorithm(n, time_limit, seed=0):
        """
        Genetic algorithm to optimize pair-value assignments.
        Represent each vertex with a value in GF(3), then derive triple colors
        from linear combinations.
        """
        np.random.seed(seed)
        triples = get_triple_indices(n)
        num_triples = len(triples)
        _, _, k4s, k5s, triple_to_k4s, triple_to_k5s = build_structure(n)

        pop_size = min(40, max(10, n))
        elite_size = 5

        # Each individual: vertex values in GF(3) + linear combination coefficients
        # v[i] in {0,1,2}, coeffs (c1,c2,c3) in {0,1,2}
        # color(a,b,c) = 0 if (c1*v[a] + c2*v[b] + c3*v[c]) % 3 == 0 else 1

        def evaluate(individual):
            v, c1, c2, c3 = individual
            colors = np.zeros(num_triples, dtype=np.int8)
            for idx, (a, b, c) in enumerate(triples):
                s = (c1 * v[a] + c2 * v[b] + c3 * v[c]) % 3
                colors[idx] = 0 if s == 0 else 1
            cost = count_violations(colors, k4s, k5s)
            return cost, colors

        # Initialize population
        population = []
        for _ in range(pop_size):
            v = np.random.randint(0, 3, n, dtype=np.int8)
            c1 = np.random.randint(0, 3)
            c2 = np.random.randint(0, 3)
            c3 = np.random.randint(0, 3)
            population.append((v, c1, c2, c3))

        # Evaluate initial population
        scores = []
        for ind in population:
            cost, _ = evaluate(ind)
            scores.append(cost)

        best_idx = np.argmin(scores)
        best_cost = scores[best_idx]
        best_colors = evaluate(population[best_idx])[1]

        if best_cost == 0:
            return best_colors, True

        # Genetic algorithm loop
        generation = 0
        while time.time() < time_limit:
            generation += 1

            # Sort by fitness
            order = np.argsort(scores)
            population = [population[i] for i in order]
            scores = [scores[i] for i in order]

            # Elitism
            new_population = [population[i] for i in range(min(elite_size, len(population)))]

            # Fill rest with offspring
            while len(new_population) < pop_size:
                # Select parents (tournament selection)
                def tournament_select():
                    idxs = np.random.randint(0, len(population), 3)
                    best = min(idxs, key=lambda i: scores[i])
                    return population[best]

                parent1 = tournament_select()
                parent2 = tournament_select()

                # Crossover: combine vertex values
                v1, c1_1, c2_1, c3_1 = parent1
                v2, c1_2, c2_2, c3_2 = parent2

                # Uniform crossover on vertex values
                mask = np.random.randint(0, 2, n)
                v_child = np.where(mask == 0, v1, v2).astype(np.int8)

                # Coefficients: pick from either parent
                if np.random.rand() < 0.5:
                    c1, c2, c3 = c1_1, c2_1, c3_1
                else:
                    c1, c2, c3 = c1_2, c2_2, c3_2

                # Mutation
                if np.random.rand() < 0.3:
                    mut_idx = np.random.randint(0, n)
                    v_child[mut_idx] = np.random.randint(0, 3)

                if np.random.rand() < 0.1:
                    c1 = np.random.randint(0, 3)
                if np.random.rand() < 0.1:
                    c2 = np.random.randint(0, 3)
                if np.random.rand() < 0.1:
                    c3 = np.random.randint(0, 3)

                new_population.append((v_child, c1, c2, c3))

            # Evaluate new population
            new_scores = []
            for ind in new_population:
                cost, _ = evaluate(ind)
                new_scores.append(cost)

            population = new_population
            scores = new_scores

            best_idx = np.argmin(scores)
            if scores[best_idx] < best_cost:
                best_cost = scores[best_idx]
                best_colors = evaluate(population[best_idx])[1]
                if best_cost == 0:
                    return best_colors, True

        return best_colors, (best_cost == 0)

    def verify(n, colors):
        """Verify coloring has no red K4 and no blue K5."""
        triples = get_triple_indices(n)
        triple_to_idx = {t: i for i, t in enumerate(triples)}

        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            if (colors[triple_to_idx[(a, b, c)]] == 0 and
                colors[triple_to_idx[(a, b, d)]] == 0 and
                colors[triple_to_idx[(a, c, d)]] == 0 and
                colors[triple_to_idx[(b, c, d)]] == 0):
                return False

        for quint in combinations(range(n), 5):
            all_blue = True
            for triple in combinations(quint, 3):
                if colors[triple_to_idx[triple]] != 1:
                    all_blue = False
                    break
            if all_blue:
                return False

        return True

    final_n = 0
    final_colors = []

    target_ns = [34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9]

    for n in target_ns:
        if time.time() > deadline - 5:
            break

        if n <= final_n:
            continue

        remaining = deadline - time.time()
        if remaining < 3:
            break

        if n >= 30:
            time_budget = min(25.0, remaining * 0.6)
        elif n >= 25:
            time_budget = min(18.0, remaining * 0.5)
        elif n >= 20:
            time_budget = min(12.0, remaining * 0.4)
        else:
            time_budget = min(8.0, remaining * 0.3)

        if time_budget < 1.0:
            break

        time_limit = time.time() + time_budget

        found = False

        # Try multiple strategies: structured inits with local search, then GA
        num_attempts = 15 if n >= 30 else 10

        for attempt in range(num_attempts):
            if time.time() > time_limit:
                break

            # Alternate between local search with different inits and GA
            if attempt < num_attempts // 2:
                # Local search with various structured inits
                seed = attempt * 7 + 1
                variant = attempt % 19
                try:
                    colors, is_valid = local_search(n, time_limit, seed=seed, init_variant=variant)
                    if is_valid:
                        if verify(n, colors.tolist()):
                            final_n = n
                            final_colors = colors.tolist()
                            found = True
                            break
                except Exception:
                    continue
            else:
                # Genetic algorithm
                seed = attempt * 13 + 3
                try:
                    colors, is_valid = genetic_algorithm(n, time_limit, seed=seed)
                    if is_valid:
                        if verify(n, colors.tolist()):
                            final_n = n
                            final_colors = colors.tolist()
                            found = True
                            break
                except Exception:
                    continue

        if found:
            break

    if final_n == 0 or not final_colors:
        n = 9
        triples = get_triple_indices(n)
        colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in triples]

        if verify(n, colors):
            final_n = n
            final_colors = colors
        else:
            np.random.seed(42)
            for _ in range(1000):
                colors = [np.random.randint(0, 2) for _ in range(len(triples))]
                if verify(n, colors):
                    final_n = n
                    final_colors = colors
                    break
            else:
                final_n = 1
                final_colors = [0]

    return {'n': final_n, 'colors': final_colors}
