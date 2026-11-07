"""
Maximum sum of pairwise-coprime integers from 1..256.

Strategy:
1. Build a conflict graph where vertices are numbers 1..256 and edges
   connect pairs with gcd > 1.
2. Find a maximum weight independent set (MWIS) using a branch-and-bound
   algorithm with bitset operations and greedy coloring for upper bounds.
3. Track the actual selected set to return a valid solution.
"""
import time
import math
import sys


def construct():
    t_start = time.time()
    DEADLINE = 85.0  # seconds

    N = 256
    weights = [i + 1 for i in range(N)]

    # Build conflict graph using bitsets
    # neighbors[i] is a bitmask of all j where gcd(i+1, j+1) > 1
    neighbors = [0] * N
    for i in range(N):
        vi = i + 1
        mask = 0
        for j in range(N):
            if j != i and math.gcd(vi, j + 1) > 1:
                mask |= (1 << j)
        neighbors[i] = mask

    all_mask = (1 << N) - 1
    vertex_bit = [1 << i for i in range(N)]

    # Helper: sum of weights in a mask
    def sum_weights(mask):
        s = 0
        m = mask
        while m:
            low = m & (-m)
            j = low.bit_length() - 1
            s += weights[j]
            m ^= low
        return s

    # Greedy solution for initial lower bound
    def greedy_solution():
        selected = set()
        candidates = set(range(N))
        while candidates:
            best = max(candidates, key=lambda x: weights[x])
            selected.add(best)
            candidates.discard(best)
            m = neighbors[best]
            removed = set()
            while m:
                low = m & (-m)
                j = low.bit_length() - 1
                if j in candidates:
                    removed.add(j)
                m ^= low
            candidates -= removed
        return sum(weights[i] for i in selected), selected

    best_lb, _ = greedy_solution()

    # Branch and Bound with bitset optimizations
    # We use a recursive approach with the following:
    # - At each step, pick a vertex to branch on
    # - Use greedy coloring on the remaining available set for upper bound
    # - Track the actual set of selected vertices

    # For efficiency, we'll use an iterative DFS with a stack
    # Each stack entry: (available_mask, current_sum, selected_tuple)

    best_sum = [best_lb]
    best_set = [set()]

    # To improve pruning, we'll use a simple coloring-based upper bound
    # For the remaining available vertices, we compute a greedy coloring
    # and use the sum of the maximum weight in each color class as an upper bound

    def greedy_color_upper_bound(available):
        """
        Compute a greedy coloring of the subgraph induced by available vertices.
        Return the sum of max weights per color class as an upper bound.
        """
        if available == 0:
            return 0

        # Get list of vertices in available
        verts = []
        m = available
        while m:
            low = m & (-m)
            j = low.bit_length() - 1
            verts.append(j)
            m ^= low

        # Sort by weight descending for better coloring
        verts.sort(key=lambda x: -weights[x])

        # Greedy coloring
        color_of = {}
        color_max_weight = {}
        for v in verts:
            # Find the smallest color not used by any neighbor of v
            used_colors = set()
            m = neighbors[v] & available
            while m:
                low = m & (-m)
                u = low.bit_length() - 1
                if u in color_of:
                    used_colors.add(color_of[u])
                m ^= low

            c = 0
            while c in used_colors:
                c += 1
            color_of[v] = c
            if c not in color_max_weight or weights[v] > color_max_weight[c]:
                color_max_weight[c] = weights[v]

        return sum(color_max_weight.values())

    # B&B with stack
    # We process by picking a vertex and branching include/exclude
    # To keep the stack manageable, we'll use a priority queue approach
    # or simply DFS with good pruning

    # Let's use DFS with the following strategy:
    # - Pick the highest-weight vertex in available
    # - Branch include first (to find good solutions early)
    # - Use coloring upper bound for pruning

    sys.setrecursionlimit(10000)

    # Use iterative DFS to avoid recursion depth issues
    # Stack entries: (available_mask, current_sum, selected_frozen)

    # To avoid storing huge frozensets, we'll track the selected set
    # differently: we'll store a list of selected vertices and rebuild

    # Actually, let's just store the selected set as a frozenset for simplicity
    # The maximum size is at most ~100 vertices, so this should be fine

    stack = [(all_mask, 0, frozenset())]
    visited = 0

    while stack:
        if time.time() > DEADLINE:
            break

        available, current_sum, selected = stack.pop()
        visited += 1

        if available == 0:
            if current_sum > best_sum[0]:
                best_sum[0] = current_sum
                best_set[0] = set(selected)
            continue

        # Upper bound using greedy coloring
        ub = current_sum + greedy_color_upper_bound(available)
        if ub <= best_sum[0]:
            continue

        # Find the highest-weight vertex in available
        best_v = -1
        best_w = -1
        m = available
        while m:
            low = m & (-m)
            j = low.bit_length() - 1
            if weights[j] > best_w:
                best_w = weights[j]
                best_v = j
            m ^= low

        if best_v == -1:
            continue

        # Branch 1: Include best_v
        new_available = available & ~vertex_bit[best_v]
        new_available &= ~neighbors[best_v]
        new_sum = current_sum + weights[best_v]
        stack.append((new_available, new_sum, selected | frozenset([best_v])))

        # Branch 2: Exclude best_v
        stack.append((available & ~vertex_bit[best_v], current_sum, selected))

    # Convert best_set to list of actual numbers
    if best_set[0]:
        result = sorted([v + 1 for v in best_set[0]])
    else:
        # Fallback: primes + 1
        result = [1] + [a for a in range(2, N + 1)
                        if all(a % d for d in range(2, math.isqrt(a) + 1))]

    # Ensure all are Python ints
    result = [int(x) for x in result]

    return result


if __name__ == "__main__":
    res = construct()
    print(f"Sum: {sum(res)}")
    print(f"Count: {len(res)}")
    # Verify pairwise coprimality
    valid = True
    for i in range(len(res)):
        for j in range(i + 1, len(res)):
            if math.gcd(res[i], res[j]) != 1:
                valid = False
                print(f"Conflict: {res[i]} and {res[j]}")
                break
        if not valid:
            break
    print(f"Valid: {valid}")
