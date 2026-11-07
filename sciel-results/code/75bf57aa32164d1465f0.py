"""Exhaustive size-8 search + SA/local search for larger sizes to maximize distinct induced degrees."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import math

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}


def _compute_distinct_degrees(combo, adj, n):
    """Compute number of distinct degrees in induced subgraph on combo."""
    mask = 0
    for v in combo:
        mask |= (1 << v)
    degs = set()
    for v in combo:
        x = adj[v] & mask
        c = 0
        while x:
            x &= x - 1
            c += 1
        degs.add(c)
    return len(degs)


def _local_search(combo, adj, n, minv, deadline, rng):
    """Hill-climbing local search: try single swaps to improve distinct degree count."""
    combo_set = set(combo)
    current_score = _compute_distinct_degrees(combo, adj, n)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for v_in in combo:
            for v_out in range(n):
                if v_out in combo_set:
                    continue
                new_combo = list(combo)
                new_combo.remove(v_in)
                new_combo.append(v_out)
                new_score = _compute_distinct_degrees(new_combo, adj, n)
                if new_score > current_score:
                    combo = new_combo
                    combo_set = set(combo)
                    current_score = new_score
                    improved = True
                    break
            if improved:
                break
    return combo, current_score


def construct():
    deadline = time.time() + 85.0  # safety margin under 90s wall-clock
    n = P['n']
    edges = P['edges']
    minv = P['minimum_vertices']

    # Build adjacency bitmasks
    adj = [0] * n
    for a, b in edges:
        adj[a] |= (1 << b)
        adj[b] |= (1 << a)

    rng = random.Random(42)  # deterministic for reproducibility

    best_vertices = None
    best_score = -1
    vertices = list(range(n))

    # ---- Phase 1: EXHAUSTIVE search over all C(17, 8) = 24310 subsets of size 8 ----
    # This is the binding constraint ("at least 8 vertices") and the space is small
    # enough to enumerate completely, guaranteeing the optimal size-8 answer.
    for combo in itertools.combinations(vertices, minv):
        if time.time() > deadline:
            break
        score = _compute_distinct_degrees(combo, adj, n)
        if score > best_score:
            best_score = score
            best_vertices = list(combo)
        # Theoretical max for size 8 is 8; if we hit it, we can't do better at size 8
        if score == minv:
            break

    # ---- Phase 2: try larger sizes (9..17) with SA + local search ----
    # Only worth it if a larger size could exceed the exhaustive best.
    # For size s, the max distinct degrees is s, so we need s > best_score to improve.
    for size in range(minv + 1, n + 1):
        if time.time() > deadline:
            break
        if size <= best_score:
            # Even the theoretical max for this size can't beat best_score
            continue

        # Multiple SA restarts for this size
        max_restarts = 40
        for restart in range(max_restarts):
            if time.time() > deadline:
                break

            # Random initial subset of given size
            combo = rng.sample(vertices, size)
            current_score = _compute_distinct_degrees(combo, adj, n)

            if current_score > best_score:
                best_score = current_score
                best_vertices = list(combo)

            # Simulated annealing
            T = 1.0
            T_min = 0.001
            cooling = 0.95
            max_iters = 150

            for it in range(max_iters):
                if time.time() > deadline:
                    break
                if T < T_min:
                    break

                # Generate neighbor: swap one vertex in with one vertex out
                v_in = rng.choice(combo)
                candidates_out = [v for v in vertices if v not in combo]
                if not candidates_out:
                    break
                v_out = rng.choice(candidates_out)

                new_combo = list(combo)
                new_combo.remove(v_in)
                new_combo.append(v_out)
                new_score = _compute_distinct_degrees(new_combo, adj, n)

                delta = new_score - current_score
                if delta > 0:
                    combo = new_combo
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_vertices = list(combo)
                else:
                    if delta < 0:
                        if rng.random() < math.exp(delta / T):
                            combo = new_combo
                            current_score = new_score

                T *= cooling

            # Local search from current position
            combo, local_score = _local_search(combo, adj, n, minv, deadline, rng)
            if local_score > best_score:
                best_score = local_score
                best_vertices = list(combo)

            # If we hit the theoretical max for this size, stop this size
            if best_score == size:
                break

    # Fallback if nothing found
    if best_vertices is None:
        best_vertices = list(range(minv))

    # Ensure exact Python int types
    out = [int(v) for v in best_vertices]
    return out
# EVOLVE-BLOCK-END
