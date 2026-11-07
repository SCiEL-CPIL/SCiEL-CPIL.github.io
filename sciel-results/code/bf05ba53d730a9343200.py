"""
Combined approach: structural multiplicative orbits + CSP-style greedy with
frequency-aware edge ordering, followed by SA and a final high-precision local
polish. The value sets are designed so that products collide heavily (smooth
numbers), while the edge-selection phase uses a greedy frequency heuristic to
pick edges that reuse existing sum/product values, then SA refines and a final
local search polishes.
"""
# EVOLVE-BLOCK-START
import time
import random
import math
from collections import Counter, defaultdict
from itertools import combinations

P = {'n': 14, 'm': 35, 'maximum': 10000}


def _raw_score(values, edges):
    sums = set()
    prods = set()
    for u, v in edges:
        a, b = values[u], values[v]
        sums.add(a + b)
        prods.add(a * b)
    return max(len(sums), len(prods))


def _all_edges(n):
    return list(combinations(range(n), 2))


def _make_value_sets():
    """
    Generate a diverse set of candidate value sets designed to maximize
    product and sum collisions.
    """
    candidates = []

    # Smooth numbers (2,3,5-smooth) - excellent for product collisions
    smooth = []
    for a in range(0, 14):
        for b in range(0, 14):
            for c in range(0, 14):
                v = (2 ** a) * (3 ** b) * (5 ** c)
                if 0 < v <= 10000:
                    smooth.append(v)
    smooth = sorted(set(smooth))
    if len(smooth) >= 14:
        # Take first 14 (smallest) for maximum density
        candidates.append(smooth[:14])
        # Also try a spread selection
        idxs = [int(i * (len(smooth) - 1) / 13) for i in range(14)]
        spread = sorted(set(smooth[i] for i in idxs))
        if len(spread) == 14:
            candidates.append(spread)

    # Two-prime structure: 2^a * 3^b
    two_prime = []
    for a in range(0, 14):
        for b in range(0, 14):
            v = (2 ** a) * (3 ** b)
            if 0 < v <= 10000:
                two_prime.append(v)
    two_prime = sorted(set(two_prime))
    if len(two_prime) >= 14:
        candidates.append(two_prime[:14])

    # Small integers 1-14
    candidates.append(list(range(1, 15)))

    # Arithmetic progression
    candidates.append([10 * i for i in range(1, 15)])

    # Geometric progression (powers of 2)
    candidates.append([2 ** i for i in range(14)])

    # Divisors of 2520 (highly composite)
    base = 2520
    divisors = []
    for d in range(1, int(base ** 0.5) + 1):
        if base % d == 0:
            divisors.append(d)
            if d != base // d:
                divisors.append(base // d)
    divisors.sort()
    if len(divisors) >= 14:
        candidates.append(divisors[:14])

    # Mixed smooth: small values with shared factors
    mixed = [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36]
    if len(set(mixed)) == 14:
        candidates.append(mixed)

    # Another mixed set
    mixed2 = [2, 4, 6, 8, 12, 16, 18, 24, 32, 36, 48, 54, 64, 72]
    if len(set(mixed2)) == 14:
        candidates.append(mixed2)

    # Randomized smooth numbers with fixed seed
    random.seed(123)
    smooth_primes = [2, 3, 5]
    rand_smooth = set()
    while len(rand_smooth) < 14:
        v = 1
        for _ in range(random.randint(1, 5)):
            v *= random.choice(smooth_primes)
        if v <= 10000:
            rand_smooth.add(v)
    if len(rand_smooth) == 14:
        candidates.append(sorted(rand_smooth))

    # Deduplicate candidates
    seen = set()
    unique_candidates = []
    for c in candidates:
        key = tuple(sorted(c))
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)

    return unique_candidates


def _greedy_edge_selection(values, all_edges, m):
    """
    Greedy edge selection using frequency-aware ordering.
    Edges whose sums/products are more frequent in the full graph are preferred,
    as they are more likely to collide with already-selected edges.
    """
    n = len(values)

    # Precompute edge attributes
    edge_data = []
    for u, v in all_edges:
        a, b = values[u], values[v]
        s = a + b
        p = a * b
        edge_data.append((s, p, (u, v)))

    # Compute frequency of each sum and product across all edges
    sum_freq = Counter()
    prod_freq = Counter()
    for s, p, _ in edge_data:
        sum_freq[s] += 1
        prod_freq[p] += 1

    # Sort edges by descending (sum_freq, prod_freq) to favor collision-prone edges
    ordered = sorted(range(len(edge_data)),
                     key=lambda i: (sum_freq[edge_data[i][0]],
                                    prod_freq[edge_data[i][1]]),
                     reverse=True)

    selected = []
    sums_set = set()
    prods_set = set()

    for idx in ordered:
        if len(selected) == m:
            break
        s, p, e = edge_data[idx]
        selected.append(e)
        sums_set.add(s)
        prods_set.add(p)

    return selected


def _sa_optimize(values, all_edges, deadline, num_restarts=10, steps_per_restart=100000):
    """
    Simulated annealing to find m edges minimizing max(distinct_sums, distinct_products).
    """
    n = len(values)
    m = P['m']

    # Precompute edge attributes
    edge_sums = []
    edge_prods = []
    for i, j in all_edges:
        a, b = values[i], values[j]
        edge_sums.append(a + b)
        edge_prods.append(a * b)

    num_total_edges = len(all_edges)
    best_score = float('inf')
    best_edges = None

    for r in range(num_restarts):
        if time.time() > deadline - 2:
            break

        # Start from greedy solution if available, else random
        if r == 0:
            greedy = _greedy_edge_selection(values, all_edges, m)
            init_indices = []
            edge_to_idx = {all_edges[i]: i for i in range(num_total_edges)}
            for e in greedy:
                init_indices.append(edge_to_idx[e])
        else:
            init_indices = random.sample(range(num_total_edges), m)

        current_edges = set(init_indices)

        # Counters
        sum_counts = Counter()
        prod_counts = Counter()
        active_sums = set()
        active_prods = set()

        for idx in current_edges:
            s = edge_sums[idx]
            p = edge_prods[idx]
            sum_counts[s] += 1
            prod_counts[p] += 1
            active_sums.add(s)
            active_prods.add(p)

        current_score = max(len(active_sums), len(active_prods))

        # Simulated Annealing
        T = 1.0
        cooling = 0.9997
        min_T = 0.005

        for step in range(steps_per_restart):
            if time.time() > deadline - 1:
                break

            # Pick a random edge to remove
            e_out = random.choice(list(current_edges))

            # Pick a random edge to add (not in current)
            e_in = random.randrange(num_total_edges)
            if e_in in current_edges:
                continue

            # Calculate delta score
            s_out = edge_sums[e_out]
            p_out = edge_prods[e_out]
            s_in = edge_sums[e_in]
            p_in = edge_prods[e_in]

            # Tentative removal
            sum_counts[s_out] -= 1
            if sum_counts[s_out] == 0:
                active_sums.discard(s_out)
            prod_counts[p_out] -= 1
            if prod_counts[p_out] == 0:
                active_prods.discard(p_out)

            # Tentative addition
            sum_counts[s_in] = sum_counts.get(s_in, 0) + 1
            if sum_counts[s_in] == 1:
                active_sums.add(s_in)
            prod_counts[p_in] = prod_counts.get(p_in, 0) + 1
            if prod_counts[p_in] == 1:
                active_prods.add(p_in)

            new_score = max(len(active_sums), len(active_prods))

            if new_score <= current_score:
                # Accept
                current_edges.remove(e_out)
                current_edges.add(e_in)
                current_score = new_score
            else:
                # SA: accept with probability exp(-delta/T)
                delta = new_score - current_score
                if random.random() < math.exp(-delta / T):
                    current_edges.remove(e_out)
                    current_edges.add(e_in)
                    current_score = new_score
                else:
                    # Revert
                    sum_counts[s_in] -= 1
                    if sum_counts[s_in] == 0:
                        active_sums.discard(s_in)
                    prod_counts[p_in] -= 1
                    if prod_counts[p_in] == 0:
                        active_prods.discard(p_in)

                    sum_counts[s_out] = sum_counts.get(s_out, 0) + 1
                    if sum_counts[s_out] == 1:
                        active_sums.add(s_out)
                    prod_counts[p_out] = prod_counts.get(p_out, 0) + 1
                    if prod_counts[p_out] == 1:
                        active_prods.add(p_out)

            T *= cooling
            if T < min_T:
                T = min_T

        # Update global best
        if current_score < best_score:
            best_score = current_score
            best_edges = [list(all_edges[idx]) for idx in current_edges]

    return best_score, best_edges


def _local_search_polish(values, edges, all_edges, deadline, max_iters=60):
    """
    High-precision local search: try all single-edge swaps to improve score.
    """
    n = len(values)
    m = len(edges)

    # Precompute edge attributes
    edge_attrs = {}
    for u, v in all_edges:
        a, b = values[u], values[v]
        edge_attrs[(u, v)] = (a + b, a * b)

    current_edges = [list(e) for e in edges]
    edge_set = set(tuple(e) for e in current_edges)

    # Current score state
    sum_counts = Counter()
    prod_counts = Counter()
    for u, v in current_edges:
        s, p = edge_attrs[(u, v)]
        sum_counts[s] += 1
        prod_counts[p] += 1

    def _current_score():
        n_sums = sum(1 for c in sum_counts.values() if c > 0)
        n_prods = sum(1 for c in prod_counts.values() if c > 0)
        return max(n_sums, n_prods)

    current_score = _current_score()

    for _ in range(max_iters):
        if time.time() > deadline:
            break

        improved = False
        for i in range(m):
            if time.time() > deadline:
                break
            u1, v1 = current_edges[i]
            e1 = (u1, v1)
            s1, p1 = edge_attrs[e1]

            for e2 in all_edges:
                if e2 == e1 or e2 in edge_set:
                    continue

                s2, p2 = edge_attrs[e2]

                # Simulate swap: remove e1, add e2
                old_s_count = sum_counts[s1]
                old_p_count = prod_counts[p1]
                new_s_count = sum_counts.get(s2, 0)
                new_p_count = prod_counts.get(p2, 0)

                # After removal
                sum_counts[s1] = old_s_count - 1
                prod_counts[p1] = old_p_count - 1
                if sum_counts[s1] == 0:
                    del sum_counts[s1]
                if prod_counts[p1] == 0:
                    del prod_counts[p1]

                # After addition
                sum_counts[s2] = new_s_count + 1
                prod_counts[p2] = new_p_count + 1

                new_score = _current_score()

                if new_score < current_score:
                    # Accept the swap
                    current_edges[i] = list(e2)
                    edge_set.remove(e1)
                    edge_set.add(e2)
                    current_score = new_score
                    improved = True
                    break
                else:
                    # Revert
                    sum_counts[s1] = old_s_count
                    prod_counts[p1] = old_p_count
                    sum_counts[s2] = new_s_count
                    if new_s_count == 0:
                        del sum_counts[s2]
                    prod_counts[p2] = new_p_count
                    if new_p_count == 0:
                        del prod_counts[p2]

            if improved:
                break

        if not improved:
            break

    return current_score, current_edges


def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill

    all_edges = _all_edges(P['n'])

    # Generate candidate value sets
    value_sets = _make_value_sets()

    best_score = float('inf')
    best_edges = None
    best_values = None

    # Try each value set with SA + local search
    for values in value_sets:
        if time.time() > deadline - 5:
            break

        if len(set(values)) != 14:
            continue
        if not all(isinstance(v, int) and 0 < v <= 10000 for v in values):
            continue

        # SA optimization
        score, edges = _sa_optimize(values, all_edges, deadline,
                                    num_restarts=8, steps_per_restart=100000)

        if edges is not None and score < best_score:
            best_score = score
            best_edges = [list(e) for e in edges]
            best_values = values[:]

        # Local search polish
        if best_edges is not None and time.time() < deadline - 2:
            score2, edges2 = _local_search_polish(values, best_edges, all_edges,
                                                  deadline, max_iters=50)
            if score2 < best_score:
                best_score = score2
                best_edges = [list(e) for e in edges2]
                best_values = values[:]

    # Fallback
    if best_edges is None:
        values = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
        edges = [list(e) for e in all_edges[:P['m']]]
        best_values = values
        best_edges = edges
        best_score = _raw_score(best_values, best_edges)

    # Ensure types are correct
    final_values = [int(v) for v in best_values]
    final_edges = [[int(u), int(v)] for u, v in best_edges]

    return {'values': final_values, 'edges': final_edges}
# EVOLVE-BLOCK-END
