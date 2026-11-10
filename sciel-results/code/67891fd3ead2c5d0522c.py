"""Combined program: bitmask conflict graph + branch-and-bound MWIS + heuristic local search."""
# EVOLVE-BLOCK-START
import time
from math import lcm
from fractions import Fraction
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit

    N = 256
    candidates = list(range(2, N + 1))
    n_cand = len(candidates)

    # Precompute incompatibility as bitmasks: conflict[i] has bit j set if lcm(candidates[i], candidates[j]) <= 256
    idx_map = {val: i for i, val in enumerate(candidates)}
    conflict = [0] * n_cand

    for i in range(n_cand):
        a = candidates[i]
        for j in range(i + 1, n_cand):
            b = candidates[j]
            if lcm(a, b) <= 256:
                conflict[i] |= (1 << j)
                conflict[j] |= (1 << i)

    def is_compat_idx(i, j):
        if i == j:
            return True
        return (conflict[i] >> j) & 1 == 0

    def is_compat_val(a, b):
        if a == b:
            return True
        return is_compat_idx(idx_map[a], idx_map[b])

    def reciprocal_sum(subset):
        total = Fraction(0, 1)
        for x in subset:
            total += Fraction(1, x)
        return total

    def is_valid(subset):
        s = set(subset)
        for i, a in enumerate(subset):
            for b in subset[i + 1:]:
                if not is_compat_val(a, b):
                    return False
        return True

    # --- Heuristic phase: greedy + local search ---
    def can_add_idx(idx, selected_idxs):
        for si in selected_idxs:
            if not is_compat_idx(idx, si):
                return False
        return True

    def greedy(order):
        selected = []
        selected_set = set()
        for x in order:
            if can_add_idx(idx_map[x], [idx_map(y) for y in selected]):
                selected.append(x)
                selected_set.add(x)
        return selected

    best_solution = None
    best_sum = Fraction(0, 1)

    # Strategy 1: Ascending
    sol1 = greedy(candidates)
    s1 = reciprocal_sum(sol1)
    if s1 > best_sum:
        best_sum = s1
        best_solution = sol1[:]

    # Strategy 2: Fewest conflicts first
    inc_count = {}
    for x in candidates:
        ix = idx_map[x]
        count = bin(conflict[ix]).count('1')
        inc_count[x] = count
    order3 = sorted(candidates, key=lambda x: (inc_count[x], x))
    sol3 = greedy(order3)
    s3 = reciprocal_sum(sol3)
    if s3 > best_sum:
        best_sum = s3
        best_solution = sol3[:]

    # Strategy 3: Randomized greedy
    for r in range(30):
        if time.time() > deadline - 30:
            break
        shuffled = candidates[:]
        random.shuffle(shuffled)
        sol_r = greedy(shuffled)
        s_r = reciprocal_sum(sol_r)
        if s_r > best_sum:
            best_sum = s_r
            best_solution = sol_r[:]

    # Local search: 1-swap improvements
    current = best_solution[:]
    current_set = set(current)
    current_sum = best_sum

    improved = True
    while improved and time.time() < deadline - 25:
        improved = False
        current.sort()
        for i in range(len(current)):
            if time.time() > deadline - 25:
                break
            old_elem = current[i]
            others = [current[j] for j in range(len(current)) if j != i]
            for x in sorted(candidates, key=lambda v: 1.0 / v):
                if x in current_set:
                    continue
                if can_add_idx(idx_map[x], [idx_map(y) for y in others]):
                    new_sum = current_sum - Fraction(1, old_elem) + Fraction(1, x)
                    if new_sum > current_sum:
                        new_set = current[:]
                        new_set[i] = x
                        new_set.sort()
                        current = new_set
                        current_set = set(new_set)
                        current_sum = new_sum
                        improved = True
                        break
            if improved:
                break

        # Try adding any compatible element
        for x in sorted(candidates, key=lambda v: 1.0 / v):
            if x in current_set:
                continue
            if can_add_idx(idx_map[x], [idx_map(y) for y in current]):
                current.append(x)
                current.sort()
                current_set = set(current)
                current_sum += Fraction(1, x)
                improved = True
                break

    if current_sum > best_sum:
        best_sum = current_sum
        best_solution = current[:]

    # --- Branch-and-bound exact solver ---
    # Order vertices by weight (1/x) descending for better pruning
    # weight[i] = 1/candidates[i], we sort indices by this descending
    order = sorted(range(n_cand), key=lambda i: -1.0 / candidates[i])
    # Re-index: new position p corresponds to old index order[p]
    # Build re-indexed conflict masks
    pos_of = [0] * n_cand  # pos_of[old_idx] = new position
    for p, old_idx in enumerate(order):
        pos_of[old_idx] = p

    # Re-indexed weights (float for B&B, Fraction for exact tracking)
    weights_f = [1.0 / candidates[order[p]] for p in range(n_cand)]
    weights_frac = [Fraction(1, candidates[order[p]]) for p in range(n_cand)]

    # Re-indexed conflict masks
    rconflict = [0] * n_cand
    for p in range(n_cand):
        old = order[p]
        mask = conflict[old]
        new_mask = 0
        # Map each set bit in old mask to new position
        m = mask
        while m:
            low_bit = m & (-m)
            old_j = low_bit.bit_length() - 1
            new_j = pos_of[old_j]
            if new_j != p:
                new_mask |= (1 << new_j)
            m ^= low_bit
        rconflict[p] = new_mask

    # Fractional knapsack upper bound: given a set of available vertices (bitmask),
    # compute upper bound on weight sum using fractional knapsack (sort by weight, take greedily)
    # For efficiency, precompute sorted order and use prefix sums
    # Since vertices are already sorted by weight descending, the fractional knapsack
    # upper bound for a subset is: sum of weights of all vertices in the subset
    # (because we can take all of them fractionally with no conflicts in the relaxed problem)
    # Actually, the proper bound: sort available by weight, take fully until we hit a conflict,
    # then take fraction. But a simpler valid upper bound is just sum of all weights in the candidate set.
    # A tighter bound: use the fact that in the conflict graph, we can't take conflicting vertices.
    # For the fractional relaxation of MWIS: it's the LP relaxation. A simple upper bound is
    # the sum of weights of vertices that are "available" (not yet excluded).
    # Let's use: upper_bound = sum of weights of all available vertices.
    # This is valid but loose. A tighter bound: for each vertex, if it conflicts with k other available
    # vertices, we can at most take 1/(k+1) of it... that's complex. Let's use the simple sum bound
    # combined with the ordering (process highest weight first, prune when current + remaining <= best).

    # Precompute suffix sums of weights for quick upper bound
    suffix_sum = [0.0] * (n_cand + 1)
    for p in range(n_cand - 1, -1, -1):
        suffix_sum[p] = suffix_sum[p + 1] + weights_f[p]

    best_bb_weight = float(best_sum)
    best_bb_set = set(best_solution)  # values
    best_bb_frac = best_sum
    nodes_visited = 0
    max_nodes = 500000  # Limit to avoid timeout

    def bb_solve():
        nonlocal best_bb_weight, best_bb_set, best_bb_frac, nodes_visited

        # State: position in order, available mask, current weight, current set
        # We process vertices in order (highest weight first)
        # At each step, we can include or exclude the current vertex

        # available_mask: bitmask of vertices still available (not excluded by conflicts)
        # We use a recursive approach with memoization on (pos, available_mask) for pruning

        # For efficiency, use iterative deepening or just DFS with the ordering
        # Since n_cand = 255, we can't enumerate all subsets. We rely on pruning.

        # Key insight: if we process in weight-descending order, and at position p,
        # the remaining vertices are p..n_cand-1, the upper bound on additional weight
        # is suffix_sum[p] (sum of all remaining weights).
        # If current_weight + suffix_sum[p] <= best_bb_weight, prune.

        # Additionally, if a vertex conflicts with many available vertices, including it
        # removes those from consideration, so the actual bound is tighter.

        # Let's implement a DFS with the following pruning:
        # 1. current_weight + suffix_sum[p] <= best_bb_weight + epsilon
        # 2. Node limit

        import sys
        sys.setrecursionlimit(10000)

        def dfs(p, available_mask, current_weight, current_set_vals):
            nonlocal best_bb_weight, best_bb_set, best_bb_frac, nodes_visited

            if nodes_visited > max_nodes:
                return

            if p >= n_cand:
                if current_weight > best_bb_weight:
                    best_bb_weight = current_weight
                    best_bb_set = set(current_set_vals)
                    best_bb_frac = reciprocal_sum(best_bb_set)
                return

            nodes_visited += 1

            # Upper bound: current + sum of all remaining available weights
            # Compute sum of weights of available vertices from p onward
            remaining_mask = available_mask & ((1 << (n_cand - p)) - 1)  # vertices p..n_cand-1
            if remaining_mask == 0:
                if current_weight > best_bb_weight:
                    best_bb_weight = current_weight
                    best_bb_set = set(current_set_vals)
                    best_bb_frac = reciprocal_sum(best_bb_set)
                return

            # Quick upper bound: sum of weights of all remaining vertices (loose but fast)
            ub = current_weight + suffix_sum[p]
            if ub <= best_bb_weight + 1e-12:
                return

            # Option 1: Exclude vertex p
            # Check if vertex p is even available
            if (available_mask >> p) & 1:
                # Option 2: Include vertex p
                new_weight = current_weight + weights_f[p]
                # New available mask: remove p and all its conflicts
                new_available = available_mask & ~(1 << p)
                new_available &= ~rconflict[p]
                new_set = current_set_vals + [candidates[order[p]]]

                # Prune: if new_weight + suffix_sum[p+1] <= best, skip
                if new_weight + suffix_sum[p + 1] > best_bb_weight + 1e-12:
                    dfs(p + 1, new_available, new_weight, new_set)

            # Option 1: Exclude vertex p (always valid)
            dfs(p + 1, available_mask, current_weight, current_set_vals)

        # Initial available mask: all vertices
        initial_mask = (1 << n_cand) - 1
        dfs(0, initial_mask, 0.0, [])

    if time.time() < deadline - 10:
        bb_solve()

        if best_bb_frac > best_sum:
            best_sum = best_bb_frac
            best_solution = list(best_bb_set)

    # Final validation
    if not best_solution:
        # Fallback
        selected = []
        for x in range(2, N + 1):
            if can_add_idx(idx_map[x], [idx_map(y) for y in selected]):
                selected.append(x)
        return selected

    if not is_valid(best_solution):
        selected = []
        for x in range(2, N + 1):
            if can_add_idx(idx_map[x], [idx_map(y) for y in selected]):
                selected.append(x)
        return selected

    return best_solution
# EVOLVE-BLOCK-END
