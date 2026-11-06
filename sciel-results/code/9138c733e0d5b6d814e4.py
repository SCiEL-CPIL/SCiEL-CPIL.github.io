"""Combined program: exact maximum independent set on the square-sum conflict
graph over [1,256], with forbidden-vertex filtering, greedy seeding, local
search, and a deadline-guarded branch-and-bound refinement."""
import time
import math
import sys

# EVOLVE-BLOCK-START

def construct():
    start = time.time()
    deadline = start + 85.0  # explicit safety margin before the 90s kill
    n = 256

    def timed():
        return time.time() > deadline

    # Precompute perfect squares up to 2*256 = 512
    squares = set()
    k = 1
    while k * k <= 512:
        squares.add(k * k)
        k += 1

    # Build adjacency: two numbers conflict if their sum is a perfect square.
    # 0-indexed internally: vertex i represents number i+1.
    adj = [set() for _ in range(n)]
    for i in range(n):
        a = i + 1
        for j in range(i + 1, n):
            b = j + 1
            if a + b in squares:
                adj[i].add(j)
                adj[j].add(i)
        # Diagonal: a + a = 2a must not be a perfect square, else a is forbidden.
        if 2 * a in squares:
            adj[i].add(-1)  # sentinel for self-conflict

    # Remove forbidden vertices (2a is a perfect square -> a cannot be chosen).
    allowed = [i for i in range(n) if -1 not in adj[i]]
    for i in allowed:
        adj[i].discard(-1)

    # Map allowed vertices to compact indices.
    idx_map = {v: idx for idx, v in enumerate(allowed)}
    m = len(allowed)
    if m == 0:
        return [1]  # fallback, should not happen

    # Build compact adjacency bitmasks.
    adj_bits = [0] * m
    for i, v in enumerate(allowed):
        bits = 0
        for nb in adj[v]:
            if nb in idx_map:
                bits |= (1 << idx_map[nb])
        adj_bits[i] = bits

    # Greedy initial solution (pick min-degree vertex repeatedly).
    def greedy_solution():
        selected = []
        remaining = set(range(m))
        while remaining and not timed():
            best_v = None
            best_deg = float('inf')
            for v in remaining:
                deg_in_rem = 0
                av = adj_bits[v]
                for nb in range(m):
                    if nb in remaining and (av >> nb) & 1:
                        deg_in_rem += 1
                if deg_in_rem < best_deg:
                    best_deg = deg_in_rem
                    best_v = v
            if best_v is None:
                break
            selected.append(best_v)
            remaining.discard(best_v)
            for nb in range(m):
                if nb in remaining and (adj_bits[best_v] >> nb) & 1:
                    remaining.discard(nb)
        return selected

    initial = greedy_solution()
    best_set = [False] * m
    for v in initial:
        best_set[v] = True
    best_size = len(initial)

    # Local search: repeatedly add any vertex compatible with the current set.
    current_selected = set(initial)
    improved = True
    while improved and not timed():
        improved = False
        for v in range(m):
            if v in current_selected:
                continue
            compatible = True
            for u in current_selected:
                if (adj_bits[v] >> u) & 1:
                    compatible = False
                    break
            if compatible:
                current_selected.add(v)
                best_set[v] = True
                best_size += 1
                improved = True

    # Branch and bound for maximum independent set (Tomita-style upper bound).
    full_mask = (1 << m) - 1

    def popcount(x):
        return x.bit_count() if hasattr(int, "bit_count") else bin(x).count('1')

    def bnb(remaining_mask, size, selected_mask):
        nonlocal best_size, best_set
        if timed():
            return
        if not remaining_mask:
            if size > best_size:
                best_size = size
                best_set = [bool((selected_mask >> i) & 1) for i in range(m)]
            return
        # Upper bound: size + popcount(remaining). If it can't beat best, prune.
        if size + popcount(remaining_mask) <= best_size:
            return
        # Pick a pivot vertex (highest degree within remaining) for branching.
        # Branch on including a neighbor or excluding the pivot.
        # Simpler: pick first vertex v in remaining; branch include/exclude.
        v_bit = remaining_mask & (-remaining_mask)  # lowest set bit
        v = v_bit.bit_length() - 1
        # Branch 1: include v
        new_remaining = remaining_mask & ~v_bit & ~adj_bits[v]
        bnb(new_remaining, size + 1, selected_mask | v_bit)
        # Branch 2: exclude v
        bnb(remaining_mask & ~v_bit, size, selected_mask)

    bnb(full_mask, 0, 0)

    # Convert back to 1-indexed values.
    result = [allowed[i] + 1 for i in range(m) if best_set[i]]
    if not result:
        result = [allowed[i] + 1 for i in initial]

    # Verify the result.
    valid = True
    for idx, a in enumerate(result):
        for b in result[idx:]:
            if a + b in squares:
                valid = False
                break
        if not valid:
            break
    if not valid:
        result = [allowed[i] + 1 for i in initial]

    return result

# EVOLVE-BLOCK-END
