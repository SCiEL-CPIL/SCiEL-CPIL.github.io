"""Odd-even transposition family with budgeted comparator reduction for 18 wires."""
import time
import numpy as np

N = 18
SEED_COUNT = N * (N - 1) // 2          # 153
DEADLINE = 85.0                        # safety margin under the 90 s kill
_seed_time = time.time()


# EVOLVE-BLOCK-START
def construct():
    """Return a correct compare-exchange network (list of (i, j), i<j) for N wires.

    Uses a genetic algorithm with bit-parallel NumPy fitness evaluation,
    followed by greedy local refinement.
    """
    deadline = _seed_time + DEADLINE

    # Precompute all 2^N boolean inputs as a (2^N, N) uint8 array
    all_inputs = np.zeros((1 << N, N), dtype=np.uint8)
    for i in range(N):
        all_inputs[:, i] = (np.arange(1 << N) >> i) & 1

    # Precompute comparator indices for vectorized application
    # For each comparator (i, j), we need masks to extract, compare, and write back

    def _apply_comparators(inputs, net):
        """Apply all comparators in net to inputs array. Returns (inputs, sorted_mask)."""
        v = inputs.copy()
        for i, j in net:
            a = v[:, i].astype(np.int8)
            b = v[:, j].astype(np.int8)
            # min goes to i, max goes to j
            v[:, i] = np.minimum(a, b)
            v[:, j] = np.maximum(a, b)
        return v

    def _count_unsorted(inputs, net):
        """Count how many of the 2^N inputs are NOT sorted after applying net."""
        v = _apply_comparators(inputs, net)
        # Check if sorted: v[:, k] <= v[:, k+1] for all k
        diffs = np.diff(v, axis=1)  # shape (2^N, N-1)
        unsorted_mask = np.any(diffs < 0, axis=1)
        return int(np.sum(unsorted_mask))

    def _is_fully_sorted(net):
        return _count_unsorted(all_inputs, net) == 0

    # Start from odd-even transposition network
    net = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    best = list(net)
    best_len = len(net)

    # Verify seed is correct
    assert _is_fully_sorted(best), "Seed network should be correct"

    # --- Phase 1: Genetic Algorithm ---
    pop_size = 60
    mutation_rate = 0.15
    elite_count = 4

    # Initialize population with variations
    population = [list(net)]
    # Add some random valid networks (subsets of odd-even or random comparators)
    all_comparators = [(i, j) for i in range(N) for j in range(i + 1, N)]
    rng = np.random.default_rng(42)

    for _ in range(pop_size - 1):
        # Start from odd-even and randomly remove/swap some
        cand = list(net)
        # Random removals
        for _ in range(rng.integers(0, 10)):
            if len(cand) > 10:
                idx = rng.integers(0, len(cand))
                cand.pop(idx)
        # Random swaps
        for _ in range(rng.integers(0, 3)):
            if len(cand) >= 2:
                i1 = rng.integers(0, len(cand))
                i2 = rng.integers(0, len(cand))
                cand[i1], cand[i2] = cand[i2], cand[i1]
        # Random insertions
        for _ in range(rng.integers(0, 3)):
            c = tuple(rng.choice(all_comparators))
            pos = rng.integers(0, len(cand) + 1)
            cand.insert(pos, c)
        population.append(cand)

    # Evaluate initial population
    fitness = [_count_unsorted(all_inputs, ind) for ind in population]

    # Sort by fitness
    order = np.argsort(fitness)
    population = [population[i] for i in order]
    fitness = [fitness[i] for i in order]

    # Keep best
    if fitness[0] == 0 and len(population[0]) < best_len:
        best = list(population[0])
        best_len = len(best)

    gen = 0
    while time.time() < deadline - 15.0:  # Reserve 15s for local search
        gen += 1
        # Selection: tournament
        def tournament_select(pop, fit, k=3):
            indices = rng.integers(0, len(pop), size=k)
            best_idx = indices[np.argmin([fit[i] for i in indices])]
            return list(pop[best_idx])

        new_pop = []
        # Elitism
        for i in range(min(elite_count, len(population))):
            new_pop.append(list(population[i]))

        # Fill rest with crossover + mutation
        while len(new_pop) < pop_size:
            if time.time() >= deadline - 15.0:
                break
            p1 = tournament_select(population, fitness)
            p2 = tournament_select(population, fitness)
            # Crossover: uniform
            child = []
            for i in range(max(len(p1), len(p2))):
                if i < len(p1) and (i < len(p2) and rng.random() < 0.5):
                    child.append(p1[i])
                elif i < len(p1):
                    child.append(p1[i])
                elif i < len(p2):
                    child.append(p2[i])
            # Mutation
            # Random remove
            if len(child) > 10 and rng.random() < mutation_rate:
                idx = rng.integers(0, len(child))
                child.pop(idx)
            # Random add
            if rng.random() < mutation_rate * 0.5:
                c = tuple(rng.choice(all_comparators))
                pos = rng.integers(0, len(child) + 1)
                child.insert(pos, c)
            # Random swap
            if len(child) >= 2 and rng.random() < mutation_rate * 0.5:
                i1 = rng.integers(0, len(child))
                i2 = rng.integers(0, len(child))
                child[i1], child[i2] = child[i2], child[i1]
            new_pop.append(child)

        # Evaluate new population
        new_fitness = [_count_unsorted(all_inputs, ind) for ind in new_pop]
        order = np.argsort(new_fitness)
        population = [new_pop[i] for i in order]
        fitness = [new_fitness[i] for i in order]

        # Update best
        if fitness[0] == 0 and len(population[0]) < best_len:
            best = list(population[0])
            best_len = len(best)
            # If we found a fully sorted network, try to refine it
            break

    # --- Phase 2: Greedy local refinement on best network ---
    # Use the best fully-sorted network if we have one, else the best fitness network
    if fitness[0] == 0:
        start_net = list(population[0])
    else:
        start_net = list(best)

    # Try to remove redundant comparators from the best network
    current = list(start_net)
    current_len = len(current)

    while time.time() < deadline - 2.0:
        improved = False
        # Try removing from the end first (often redundant)
        for idx in range(len(current) - 1, -1, -1):
            if time.time() >= deadline - 2.0:
                break
            cand = current[:idx] + current[idx + 1:]
            if _is_fully_sorted(cand):
                current = cand
                current_len = len(current)
                improved = True
                break
        if not improved:
            # Try swapping adjacent comparators to enable further removals
            swapped = False
            for idx in range(len(current) - 1):
                if time.time() >= deadline - 2.0:
                    break
                cand = list(current)
                cand[idx], cand[idx + 1] = cand[idx + 1], cand[idx]
                if _is_fully_sorted(cand):
                    current = cand
                    swapped = True
                    break
            if not swapped:
                break

    if len(current) < best_len:
        best = current
        best_len = len(best)

    return best
# EVOLVE-BLOCK-END
