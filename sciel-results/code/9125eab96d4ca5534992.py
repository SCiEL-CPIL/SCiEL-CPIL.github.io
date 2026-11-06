"""Rewritten program: maximum clique in pairwise-gcd>1 graph on [1,255] containing 255."""

import math
import time
import random

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 80.0  # safety margin before 90s kill
    N = 255
    target = 255  # must be in the set

    # Prime factors of 255
    pf = [3, 5, 17]

    # Candidate pool: numbers in [1,255] that share a factor with 255
    # (since every element must have gcd>1 with 255)
    pool = [x for x in range(1, N + 1) if any(x % p == 0 for p in pf)]

    # Precompute gcd matrix for the pool
    # Use index mapping
    idx = {v: i for i, v in enumerate(pool)}
    m = len(pool)

    # Build adjacency: edge between i,j if gcd(pool[i], pool[j]) > 1
    adj = [[False] * m for _ in range(m)]
    for i in range(m):
        for j in range(i + 1, m):
            if math.gcd(pool[i], pool[j]) > 1:
                adj[i][j] = adj[j][i] = True

    # Target index
    t_idx = idx[target]

    # The trivial solution: all multiples of 3
    multiples_of_3 = [x for x in pool if x % 3 == 0]
    best_set = set(multiples_of_3)
    best_size = len(best_set)

    # Verify the trivial solution is valid
    for i in range(len(multiples_of_3)):
        for j in range(i + 1, len(multiples_of_3)):
            if math.gcd(multiples_of_3[i], multiples_of_3[j]) == 1:
                raise RuntimeError("Trivial solution invalid")

    # Try to improve via local search / greedy expansion
    # Strategy: start from multiples of 3, try adding elements not divisible by 3
    # that are connected to all current members.
    # Then try removing some multiples of 3 to add more non-3 elements.

    # Greedy: try to add each non-3 element if it's connected to all in current set
    current = list(best_set)
    current_set = set(current)

    for x in pool:
        if x in current_set:
            continue
        # Check if x is connected to all in current_set
        x_idx = idx[x]
        ok = True
        for y in current:
            y_idx = idx[y]
            if not adj[x_idx][y_idx]:
                ok = False
                break
        if ok:
            current_set.add(x)
            current.append(x)

    if len(current_set) > best_size:
        best_set = current_set
        best_size = len(best_set)

    # Now try more aggressive local search:
    # Randomized greedy: start with a random seed set containing 255, grow greedily
    random.seed(42)
    for trial in range(500):
        if time.time() > deadline - 5.0:
            break

        # Start with 255 and a few random neighbors
        start = {target}
        # Add 2-5 random neighbors of 255
        neighbors_of_target = [pool[j] for j in range(m) if j != t_idx and adj[t_idx][j]]
        random.shuffle(neighbors_of_target)
        for nb in neighbors_of_target[:random.randint(2, 5)]:
            start.add(nb)

        # Greedy grow: repeatedly add the vertex with highest degree within current set
        current = set(start)
        while True:
            if time.time() > deadline - 5.0:
                break
            candidates = [x for x in pool if x not in current]
            if not candidates:
                break
            best_cand = None
            best_score = -1
            for x in candidates:
                x_idx = idx[x]
                score = sum(1 for y in current if adj[x_idx][idx[y]])
                if score == len(current):  # connected to all
                    if best_cand is None or score > best_score:
                        best_cand = x
                        best_score = score
                elif best_cand is None:
                    # fallback: pick one with max connections
                    if score > best_score:
                        best_cand = x
                        best_score = score
            if best_cand is None:
                break
            current.add(best_cand)

        if len(current) > best_size:
            best_set = current
            best_size = len(current)

    # Final check: ensure best_set is valid
    best_list = list(best_set)
    for i in range(len(best_list)):
        for j in range(i + 1, len(best_list)):
            if math.gcd(best_list[i], best_list[j]) == 1:
                # Fallback to multiples of 3
                return [3 * k for k in range(1, 86)]

    if target not in best_set:
        best_set.add(target)

    return list(best_set)
# EVOLVE-BLOCK-END
