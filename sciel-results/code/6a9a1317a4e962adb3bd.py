import math
import time
from fractions import Fraction
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s kill

    N = 256
    domain = list(range(2, N + 1))
    n = len(domain)

    # Precompute weights as Fractions for exact arithmetic
    weights = [Fraction(1, x) for x in domain]

    # Precompute lcm conflict: conflict[i][j] = True if lcm(domain[i], domain[j]) <= 256
    # Build conflict as a list of bitmasks or adjacency
    # n=255, use Python ints as bitmasks for speed
    conflict_mask = [0] * n  # conflict_mask[i] has bit j set if i,j conflict
    for i in range(n):
        vi = domain[i]
        for j in range(i + 1, n):
            vj = domain[j]
            if math.lcm(vi, vj) <= 256:
                conflict_mask[i] |= (1 << j)
                conflict_mask[j] |= (1 << i)

    # Check time
    if time.time() > deadline - 60:
        # Fallback greedy
        return _greedy_fallback(domain, conflict_mask, weights)

    # Branch and bound with exact arithmetic
    # State: (index, selected_mask, current_sum)
    # We branch on including or excluding each element in order of weight (descending)
    # Upper bound: current_sum + sum of weights of all remaining non-conflicting elements

    # Sort indices by weight descending (i.e., value ascending) for tighter bounds
    order = sorted(range(n), key=lambda i: domain[i])  # 2, 3, 4, ... smallest first = highest weight

    # Precompute for each position in order: which subsequent positions conflict with it
    # pos_conflict[k] = bitmask of positions > k in order that conflict with order[k]
    pos_conflict = [0] * n
    for k in range(n):
        i = order[k]
        mask = 0
        for k2 in range(k + 1, n):
            j = order[k2]
            if (conflict_mask[i] >> j) & 1:
                mask |= (1 << (k2 - k - 1))  # relative offset
        # Actually, let's use absolute position masks for clarity
        # Rebuild: pos_conflict[k] = bitmask over positions k+1..n-1
        pos_conflict[k] = mask

    # Better: use absolute index masks
    # conflict_abs[k] = set of absolute indices > order[k] that conflict
    # For bounding, we need: given a set of "forbidden" indices (already selected),
    # the upper bound is sum of weights of all unprocessed indices not forbidden.

    # Let's use a cleaner B&B:
    # Process indices in 'order' (smallest value first = highest weight).
    # At each step k, decide include or exclude order[k].
    # If include: add its weight, mark all its conflicts as forbidden.
    # If exclude: just skip.
    # Upper bound at step k: current_sum + sum(weights[order[t]] for t >= k if order[t] not forbidden)

    # Precompute suffix sums of weights in order for quick upper bound
    suffix_sum = [Fraction(0)] * (n + 1)
    for k in range(n - 1, -1, -1):
        suffix_sum[k] = suffix_sum[k + 1] + weights[order[k]]

    best_sum = Fraction(0)
    best_set = set()

    # For the bound to be tight, we need to know which elements are forbidden.
    # forbidden_mask: bitmask of absolute indices that are forbidden.
    # When we include order[k], we forbid all j where conflict_mask[order[k]] has bit j.

    # To make the bound tighter: upper_bound = current_sum + sum of weights of
    # order[t] for t >= k where order[t] is NOT in forbidden_mask.
    # This is an optimistic bound (ignores conflicts among remaining elements).

    # For n=255, this B&B might be too slow without good pruning.
    # Let's add: if suffix_sum[k] + current_sum <= best_sum, prune.

    # Also: use a greedy initial solution to set a good lower bound.
    greedy_set = _greedy_solve(domain, conflict_mask, weights, order)
    best_sum = sum(weights[i] for i in greedy_set)
    best_set = set(greedy_set)

    if time.time() > deadline - 50:
        return [domain[i] for i in best_set]

    # B&B with recursion limit and time check
    import sys
    sys.setrecursionlimit(10000)

    forbidden = [0] * n  # forbidden[k] = bitmask of absolute indices forbidden at level k
    # Actually, we'll pass forbidden as a frozenset or bitmask.

    # Use iterative B&B with a stack to avoid recursion depth issues
    # Stack items: (k, current_sum, forbidden_mask, selected_list)
    # forbidden_mask: Python int bitmask of absolute indices that are forbidden

    # Initial: k=0, sum=0, forbidden=0, selected=[]
    # But we want to explore include-branch first (higher weight elements first)

    # For efficiency, precompute for each k: the bitmask of conflicts of order[k]
    # in terms of absolute indices
    conflict_abs_mask = [0] * n
    for k in range(n):
        i = order[k]
        cm = conflict_mask[i]
        # cm already has bits for all conflicting absolute indices
        conflict_abs_mask[k] = cm

    # Upper bound function: given k, current_sum, forbidden_mask
    # UB = current_sum + sum(weights[order[t]] for t in range(k, n) if not (forbidden_mask >> order[t]) & 1)
    # This is O(n) per call, which is fine.

    def upper_bound(k, current_sum, forbidden_mask):
        if k >= n:
            return current_sum
        ub = current_sum
        for t in range(k, n):
            idx = order[t]
            if not (forbidden_mask >> idx) & 1:
                ub += weights[idx]
        return ub

    # B&B stack: (k, current_sum, forbidden_mask, selected_tuple)
    # We'll use a list as stack, exploring include-branch first
    stack = [(0, Fraction(0), 0, ())]
    nodes_explored = 0

    while stack:
        if time.time() > deadline - 15:
            break

        k, current_sum, forbidden_mask, selected = stack.pop()

        if k >= n:
            if current_sum > best_sum:
                best_sum = current_sum
                best_set = set(selected)
            continue

        # Prune by upper bound
        ub = upper_bound(k, current_sum, forbidden_mask)
        if ub <= best_sum:
            continue

        idx = order[k]

        # Branch 1: Exclude idx (push first so include is explored first via stack LIFO)
        stack.append((k + 1, current_sum, forbidden_mask, selected))

        # Branch 2: Include idx (if not forbidden)
        if not (forbidden_mask >> idx) & 1:
            new_forbidden = forbidden_mask | conflict_abs_mask[k] | (1 << idx)
            new_sum = current_sum + weights[idx]
            # Quick check: if new_sum + suffix_sum[k+1] <= best_sum, skip
            if new_sum + suffix_sum[k + 1] > best_sum:
                stack.append((k + 1, new_sum, new_forbidden, selected + (idx,)))

        nodes_explored += 1

    if time.time() > deadline - 10:
        pass

    # Local improvement: try 1-for-1 and 2-for-1 swaps
    best_list = list(best_set)
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        # 1-for-1 swap
        for i in range(len(best_list)):
            a = best_list[i]
            for j in range(n):
                if j in best_set:
                    continue
                b = j
                # Check if b conflicts with any in best_list except a
                ok = True
                for other in best_list:
                    if other == a:
                        continue
                    if (conflict_mask[b] >> other) & 1:
                        ok = False
                        break
                if not ok:
                    continue
                new_sum = best_sum - weights[a] + weights[b]
                if new_sum > best_sum:
                    best_sum = new_sum
                    best_list[i] = b
                    best_set = set(best_list)
                    improved = True
                    break
            if improved:
                break

        if not improved and time.time() < deadline - 5:
            # 2-for-1: remove two, add one
            for i in range(len(best_list)):
                for j in range(i + 1, len(best_list)):
                    a1 = best_list[i]
                    a2 = best_list[j]
                    for k2 in range(n):
                        if k2 in best_set:
                            continue
                        b = k2
                        # Check b doesn't conflict with any remaining element
                        ok = True
                        for other in best_list:
                            if other == a1 or other == a2:
                                continue
                            if (conflict_mask[b] >> other) & 1:
                                ok = False
                                break
                        if not ok:
                            continue
                        new_sum = best_sum - weights[a1] - weights[a2] + weights[b]
                        if new_sum > best_sum:
                            best_sum = new_sum
                            best_list.remove(a1)
                            best_list.remove(a2)
                            best_list.append(b)
                            best_set = set(best_list)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break

    if not best_set:
        best_set = {order[0]}
        best_sum = weights[order[0]]

    return [domain[i] for i in best_set]


def _greedy_fallback(domain, conflict_mask, weights):
    n = len(domain)
    order = sorted(range(n), key=lambda i: domain[i])
    selected = []
    selected_set = set()
    for idx in order:
        ok = True
        for other in selected_set:
            if (conflict_mask[idx] >> other) & 1:
                ok = False
                break
        if ok:
            selected.append(idx)
            selected_set.add(idx)
    if not selected:
        selected = [0]
        selected_set = {0}
    return [domain[i] for i in selected]


def _greedy_solve(domain, conflict_mask, weights, order):
    n = len(domain)
    selected = []
    selected_set = set()
    for idx in order:
        ok = True
        for other in selected_set:
            if (conflict_mask[idx] >> other) & 1:
                ok = False
                break
        if ok:
            selected.append(idx)
            selected_set.add(idx)
    if not selected:
        selected = [order[0]]
        selected_set = {order[0]}
    return selected_set
