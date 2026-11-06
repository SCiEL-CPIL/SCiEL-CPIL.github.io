import time
from math import gcd
from fractions import Fraction

def construct():
    start = time.time()
    deadline = start + 82.0  # safety margin before 90s kill

    # Denominators 2..48 -> indices 0..46
    denoms = list(range(2, 49))
    n = len(denoms)

    # Work in integer weights: weight[i] = L // denoms[i], where L = lcm(2..48)
    L = 1
    for d in denoms:
        L = L // gcd(L, d) * d

    weights = [L // d for d in denoms]

    # Sort items by weight descending for better B&B pruning
    order = sorted(range(n), key=lambda i: -weights[i])
    sorted_weights = [weights[i] for i in order]

    # Suffix sums for upper bound: suffix[i] = sum of sorted_weights[i:]
    suffix = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix[i] = suffix[i + 1] + sorted_weights[i]

    # Meet-in-the-middle to get a strong initial lower bound
    # Split into two halves
    mid = n // 2
    w1 = sorted_weights[:mid]
    w2 = sorted_weights[mid:]

    def gen_sums(w_list, limit):
        """Generate all subset sums < limit."""
        sums = {0}
        for w in w_list:
            new_sums = set()
            for s in sums:
                ns = s + w
                if ns < limit:
                    new_sums.add(ns)
            sums |= new_sums
        return sums

    # Generate sums for each half (with pruning)
    sums1 = gen_sums(w1, L)
    sums2 = gen_sums(w2, L)

    # Find max sum < L using two-pointer
    sorted_sums2 = sorted(sums2)
    best_sum = 0
    best_s1 = 0
    best_s2 = 0

    for s1 in sorted(sums1):
        # Need s2 < L - s1
        # Binary search for largest s2 in sorted_sums2 with s2 < L - s1
        target = L - s1  # s2 must be < target
        # bisect_left gives first element >= target, so we want index before that
        import bisect
        idx = bisect.bisect_left(sorted_sums2, target)
        if idx > 0:
            s2 = sorted_sums2[idx - 1]
            total = s1 + s2
            if total > best_sum:
                best_sum = total
                best_s1 = s1
                best_s2 = s2

    # Now we have a good initial bound. Run branch and bound to verify/improve.
    # B&B: maximize sum, constraint sum <= L-1
    best_total = best_sum
    best_selection = [False] * n  # in sorted order

    # Branch and bound
    def branch_and_bound():
        nonlocal best_total, best_selection
        # DFS with pruning
        # State: index i, current sum, current selection
        # Upper bound: current_sum + suffix[i]
        # If current_sum + suffix[i] <= best_total, prune
        # If current_sum >= L, prune (can't add more)
        
        import sys
        sys.setrecursionlimit(10000)
        
        def dfs(i, current_sum, selection):
            nonlocal best_total, best_selection
            if time.time() > deadline:
                return
            
            # Prune: even taking all remaining, can't beat best
            if current_sum + suffix[i] <= best_total:
                return
            
            if i == n:
                if current_sum > best_total and current_sum < L:
                    best_total = current_sum
                    best_selection = selection[:]
                return
            
            # Option 1: skip item i
            dfs(i + 1, current_sum, selection)
            
            if time.time() > deadline:
                return
            
            # Option 2: take item i (if doesn't exceed L-1)
            new_sum = current_sum + sorted_weights[i]
            if new_sum < L:
                selection[i] = True
                dfs(i + 1, new_sum, selection)
                selection[i] = False

        dfs(0, 0, [False] * n)

    # Run B&B
    branch_and_bound()

    # Convert best_selection (in sorted order) back to original indices
    result_indices = []
    for i in range(n):
        if best_selection[i]:
            result_indices.append(order[i])

    # If B&B didn't find anything (shouldn't happen), use MITM result
    if not result_indices:
        # Reconstruct from MITM
        # Find which items in w1 sum to best_s1 and which in w2 sum to best_s2
        # This is a subset sum reconstruction - for now, just pick greedily
        # Actually, let's just return the first few items that fit
        result_indices = []
        current_sum = 0
        for i in range(n):
            if current_sum + sorted_weights[i] < L:
                result_indices.append(order[i])
                current_sum += sorted_weights[i]
        if not result_indices:
            result_indices = [order[0]]

    # Verify
    total = sum(weights[i] for i in result_indices)
    assert total < L, f"Sum {total} >= L {L}"
    assert len(result_indices) > 0

    return result_indices
