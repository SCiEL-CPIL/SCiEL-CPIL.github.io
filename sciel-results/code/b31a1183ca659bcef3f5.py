import numpy as np
import time
import random

N = 256
MAX_D = N // 3  # 85


def _precompute_ap_matrix():
    """
    Build a (N, N, N) boolean-like structure? Too large.
    Instead, for each position i, store the list of 4-APs containing i.
    But for vectorized coordinate descent, we need to compute delta for ALL i at once.
    
    Alternative: For each 4-AP (a, b, c, d), it contributes to the count if all 4 are same color.
    We can precompute for each position i, the set of 4-APs it belongs to.
    
    Actually, let's use a different vectorized approach for local search:
    For a given coloring c (as int8 array), the number of monochromatic 4-APs is:
    sum over all 4-APs of [c[a]==c[b]==c[c]==c[d]]
    
    For coordinate descent, flipping position i changes the count by:
    delta_i = (number of 4-APs containing i that become mono) - (number that were mono and become non-mono)
    
    To vectorize this: for each i, we need to know for each 4-AP containing i, whether the other 3 are equal to each other and to c[i] (before flip) or to 1-c[i] (after flip).
    
    Let's precompute for each position i:
      - list of (j, k, l) for each 4-AP containing i
    
    Then for vectorized coordinate descent, we can't easily vectorize over all i simultaneously with simple numpy ops because the structure is irregular.
    
    Better approach: Use the precomputed neighbor lists but compute deltas in a semi-vectorized way.
    Or: Just do a fast sequential coordinate descent with precomputed neighbors, which is still much faster than the current _delta_flip because we avoid function call overhead and use local variables.
    
    Actually, let's try a different vectorized local search:
    Represent the coloring as an int8 array.
    For each 4-AP, we can compute a "mono score" array.
    
    Let's try this: Precompute for each 4-AP its 4 indices. Store them in a list of arrays.
    Then, the total count is sum of (all four equal) over all APs.
    
    For coordinate descent, flipping i:
    For each AP containing i, check if the other 3 are equal. If they are equal to each other:
      - If they equal old c[i]: flipping makes it non-mono (delta -1)
      - If they equal new c[i]: flipping makes it mono (delta +1)
      - If they are not all equal to each other: flipping doesn't change mono status (delta 0)
    
    So for each i, delta_i = (# of APs containing i where other 3 are all equal to 1-c[i]) - (# of APs containing i where other 3 are all equal to c[i])
    
    We can precompute for each i the list of (j,k,l).
    Then for a given c, we can compute for each i:
      For each (j,k,l) in neighbors[i]:
        if c[j]==c[k]==c[l]:
          if c[j] == 1-c[i]: delta_i += 1
          else: delta_i -= 1
    
    This is still O(total_APs * 4) per full coordinate descent pass, but with low constant.
    Let's just implement a very fast sequential coordinate descent with precomputed neighbors and inline logic.
    """
    neighbors = [[] for _ in range(N)]
    for d in range(1, MAX_D + 1):
        max_a = N - 3 * d
        for a in range(1, max_a + 1):
            p0 = a - 1
            p1 = p0 + d
            p2 = p0 + 2 * d
            p3 = p0 + 3 * d
            neighbors[p0].append((p1, p2, p3))
            neighbors[p1].append((p0, p2, p3))
            neighbors[p2].append((p0, p1, p3))
            neighbors[p3].append((p0, p1, p2))
    return neighbors


_NEIGHBORS = _precompute_ap_matrix()


def _count_mono4(c):
    """Fast count using precomputed structure."""
    count = 0
    for d in range(1, MAX_D + 1):
        max_a = N - 3 * d
        if max_a <= 0:
            continue
        c0 = c[:max_a]
        c1 = c[d:d+max_a]
        c2 = c[2*d:2*d+max_a]
        c3 = c[3*d:3*d+max_a]
        count += int(np.sum((c0 == c1) & (c0 == c2) & (c0 == c3)))
    return count


def _fast_coordinate_descent(c, max_passes=10, time_limit=5.0):
    """
    Fast coordinate descent: repeatedly scan all positions, flip if it improves.
    Use precomputed neighbors for O(1) delta per position (amortized).
    """
    start = time.time()
    c = c.copy()
    best_count = _count_mono4(c)
    
    for _ in range(max_passes):
        if time.time() - start > time_limit:
            break
        improved = False
        for i in range(N):
            if time.time() - start > time_limit:
                break
            old_val = c[i]
            new_val = 1 - old_val
            delta = 0
            for j, k, l in _NEIGHBORS[i]:
                if c[j] == c[k] == c[l]:
                    if c[j] == new_val:
                        delta += 1
                    else:
                        delta -= 1
            if delta < 0:
                c[i] = new_val
                best_count += delta
                improved = True
        if not improved:
            break
    return c, best_count


def _ga_generation(pop, pop_size, elite_size, crossover_rate, mutation_rate, rng, time_limit):
    """
    One generation of GA.
    pop: list of (coloring, count)
    Returns new population.
    """
    # Sort by count
    pop_sorted = sorted(pop, key=lambda x: x[1])
    
    # Elitism
    new_pop = [pop_sorted[i] for i in range(min(elite_size, len(pop_sorted)))]
    
    # Fill rest with offspring
    while len(new_pop) < pop_size:
        if time.time() > time_limit:
            break
        # Tournament selection
        p1 = rng.choice(pop_sorted)
        p2 = rng.choice(pop_sorted)
        parent1 = p1[0] if p1[1] < p2[1] else p2[0]
        
        p3 = rng.choice(pop_sorted)
        p4 = rng.choice(pop_sorted)
        parent2 = p3[0] if p3[1] < p4[1] else p4[0]
        
        # Crossover
        if rng.random() < crossover_rate:
            point = rng.randint(1, N-1)
            child = np.concatenate([parent1[:point], parent2[point:]])
        else:
            child = parent1.copy()
        
        # Mutation
        if rng.random() < mutation_rate:
            num_mut = rng.randint(1, 5)
            for _ in range(num_mut):
                pos = rng.randint(0, N-1)
                child[pos] = 1 - child[pos]
        
        # Local search on child
        child, child_count = _fast_coordinate_descent(child, max_passes=3, time_limit=0.3)
        new_pop.append((child, child_count))
    
    return new_pop


def _thue_morse(n):
    c = np.zeros(n, dtype=np.int8)
    for i in range(1, n):
        c[i] = c[i >> 1] ^ (i & 1)
    return c


def construct():
    deadline = time.time() + 140  # Safety margin
    rng = random.Random(42)
    
    # Initialize population with diverse starting points
    pop_size = 20
    elite_size = 4
    crossover_rate = 0.8
    mutation_rate = 0.5
    
    pop = []
    # Thue-Morse and shifts
    base = _thue_morse(N)
    pop.append((base.copy(), _count_mono4(base)))
    for shift in range(1, 10):
        c = np.roll(base, shift)
        pop.append((c.copy(), _count_mono4(c)))
    
    # Random initializations
    while len(pop) < pop_size:
        c = np.random.randint(0, 2, N, dtype=np.int8)
        c, cnt = _fast_coordinate_descent(c, max_passes=3, time_limit=0.2)
        pop.append((c, cnt))
    
    # Run GA
    gen = 0
    best_c, best_count = min(pop, key=lambda x: x[1])
    
    while time.time() < deadline - 10:
        pop = _ga_generation(pop, pop_size, elite_size, crossover_rate, mutation_rate, rng, deadline - 10)
        gen += 1
        
        # Update best
        for c, cnt in pop:
            if cnt < best_count:
                best_count = cnt
                best_c = c.copy()
        
        # If no improvement in last 5 generations, perturb best
        if gen % 5 == 0:
            perturbed = best_c.copy()
            for _ in range(rng.randint(5, 15)):
                perturbed[rng.randint(0, N-1)] = 1 - perturbed[rng.randint(0, N-1)]
            perturbed, perturbed_count = _fast_coordinate_descent(perturbed, max_passes=5, time_limit=1.0)
            pop.append((perturbed, perturbed_count))
            if perturbed_count < best_count:
                best_count = perturbed_count
                best_c = perturbed.copy()
    
    # Final polish
    if time.time() < deadline - 2:
        best_c, best_count = _fast_coordinate_descent(best_c, max_passes=20, time_limit=deadline - time.time() - 1)
    
    result = [int(x) for x in best_c]
    assert len(result) == 256
    assert all(x in (0, 1) for x in result)
    return result
