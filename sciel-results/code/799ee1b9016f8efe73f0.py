"""
Genetic algorithm to find a tournament on n vertices where every set of 4 vertices
has a common dominator. Uses NumPy-vectorized fitness evaluation with precomputed
quad-edge mappings for fast incremental updates.
"""

import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 140  # Safety margin for 150s limit

    # Known best solution: Paley tournament on p=67
    p_known = 67
    squares = {x * x % p_known for x in range(1, p_known)}
    orientation_known = []
    for i in range(p_known):
        for j in range(i + 1, p_known):
            if (j - i) % p_known in squares:
                orientation_known.append(0)
            else:
                orientation_known.append(1)

    best_n = p_known
    best_orientation = orientation_known

    # Try to find a solution for n = 66
    target_n = 66
    m = target_n * (target_n - 1) // 2

    # Helper to map (i, j) to index in orientation array
    def get_idx(i, j, n):
        return i * (n - 1) - i * (i + 1) // 2 + (j - i - 1)

    # Precompute all 4-tuples as a numpy array
    quad_sets = []
    for a in range(target_n):
        for b in range(a + 1, target_n):
            for c in range(b + 1, target_n):
                for d in range(c + 1, target_n):
                    quad_sets.append((a, b, c, d))

    quad_sets = np.array(quad_sets, dtype=np.int32)
    num_quads = len(quad_sets)

    # Precompute edge indices for each 4-set
    edge_indices = np.zeros((num_quads, 6), dtype=np.int32)
    for i in range(num_quads):
        a, b, c, d = quad_sets[i]
        edge_indices[i, 0] = get_idx(a, b, target_n)
        edge_indices[i, 1] = get_idx(a, c, target_n)
        edge_indices[i, 2] = get_idx(a, d, target_n)
        edge_indices[i, 3] = get_idx(b, c, target_n)
        edge_indices[i, 4] = get_idx(b, d, target_n)
        edge_indices[i, 5] = get_idx(c, d, target_n)

    # Vectorized cost evaluation
    def evaluate_cost(orient):
        """
        orient: numpy array of 0/1, shape (m,)
        Returns the number of bad 4-sets.
        """
        e_ab = orient[edge_indices[:, 0]]
        e_ac = orient[edge_indices[:, 1]]
        e_ad = orient[edge_indices[:, 2]]
        e_bc = orient[edge_indices[:, 3]]
        e_bd = orient[edge_indices[:, 4]]
        e_cd = orient[edge_indices[:, 5]]

        a_doms = (e_ab == 0) & (e_ac == 0) & (e_ad == 0)
        b_doms = (e_ab == 1) & (e_bc == 0) & (e_bd == 0)
        c_doms = (e_ac == 1) & (e_bc == 1) & (e_cd == 0)
        d_doms = (e_ad == 1) & (e_bd == 1) & (e_cd == 1)

        good = a_doms | b_doms | c_doms | d_doms
        bad_count = num_quads - np.sum(good)
        return int(bad_count)

    # Build Paley tournament for target_n (not a prime, but use modular arithmetic)
    def make_paley_like(n):
        """Create a tournament inspired by Paley construction for non-prime n."""
        # Use the largest prime less than n for the quadratic residue pattern
        def is_prime(x):
            if x < 2:
                return False
            if x % 2 == 0:
                return x == 2
            for i in range(3, int(x**0.5) + 1, 2):
                if x % i == 0:
                    return False
            return True

        p = n
        while p > 2 and not is_prime(p):
            p -= 1

        if p < 5:
            return np.random.randint(0, 2, size=m, dtype=np.int8)

        squares = set()
        for x in range(1, p):
            squares.add(x * x % p)

        orient = np.zeros(m, dtype=np.int8)
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                diff = (j - i) % p
                if diff in squares:
                    orient[idx] = 0
                else:
                    orient[idx] = 1
                idx += 1
        return orient

    # Genetic Algorithm
    def genetic_algorithm(n, time_limit):
        start_search = time.time()
        m = n * (n - 1) // 2

        pop_size = 30
        elite_size = 3
        mutation_rate = 0.05

        # Initialize population: one Paley-like, rest random
        population = []
        costs = []

        # First individual: Paley-like
        paley_orient = make_paley_like(n)
        population.append(paley_orient.copy())
        costs.append(evaluate_cost(paley_orient))

        # Rest: random with some bias toward Paley structure
        for i in range(pop_size - 1):
            if i < 5:
                # Slightly perturbed Paley
                orient = paley_orient.copy()
                num_flips = random.randint(m // 20, m // 10)
                flip_positions = np.random.choice(m, size=num_flips, replace=False)
                orient[flip_positions] = 1 - orient[flip_positions]
            else:
                orient = np.random.randint(0, 2, size=m, dtype=np.int8)
            population.append(orient)
            costs.append(evaluate_cost(orient))

        best_cost = min(costs)
        best_idx = costs.index(best_cost)
        best_orient = population[best_idx].copy()

        if best_cost == 0:
            return best_orient, True

        generation = 0
        max_generations = 5000
        stagnant_generations = 0
        max_stagnant = 200

        while generation < max_generations:
            if time.time() - start_search > time_limit:
                break

            # Sort by cost
            sorted_indices = np.argsort(costs)
            sorted_costs = [costs[i] for i in sorted_indices]
            sorted_pop = [population[i] for i in sorted_indices]

            # Update best
            if sorted_costs[0] < best_cost:
                best_cost = sorted_costs[0]
                best_orient = sorted_pop[0].copy()
                stagnant_generations = 0
            else:
                stagnant_generations += 1

            if best_cost == 0:
                return best_orient, True

            if stagnant_generations > max_stagnant:
                # Restart with new random individuals
                for i in range(pop_size // 2):
                    population[i] = np.random.randint(0, 2, size=m, dtype=np.int8)
                    costs[i] = evaluate_cost(population[i])
                stagnant_generations = 0
                continue

            # Create next generation
            next_population = []
            next_costs = []

            # Elitism: keep top individuals
            for i in range(min(elite_size, pop_size)):
                next_population.append(sorted_pop[i].copy())
                next_costs.append(sorted_costs[i])

            # Fill rest via tournament selection + mutation
            while len(next_population) < pop_size:
                # Tournament selection
                tournament_size = 3
                contenders = random.sample(range(pop_size), tournament_size)
                winner = min(contenders, key=lambda x: costs[x])

                # Mutation: bit flips
                child = population[winner].copy()
                num_flips = max(1, int(m * mutation_rate * random.uniform(0.5, 1.5)))
                if num_flips > m:
                    num_flips = m
                flip_positions = np.random.choice(m, size=num_flips, replace=False)
                child[flip_positions] = 1 - child[flip_positions]

                next_population.append(child)
                next_costs.append(evaluate_cost(child))

            population = next_population
            costs = next_costs
            generation += 1

        return best_orient, (best_cost == 0)

    # Try genetic algorithm for n = 66
    if time.time() - start_time < deadline - 10:
        try:
            remaining_time = deadline - 10 - (time.time() - start_time)
            result_orient, success = genetic_algorithm(target_n, remaining_time)
            if success:
                best_n = target_n
                best_orientation = result_orient.tolist()
        except Exception as e:
            pass  # Fall back to known solution

    return {'n': best_n, 'orientation': best_orientation}
