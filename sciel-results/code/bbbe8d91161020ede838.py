import time
from math import factorial, log

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    target = factorial(12)  # 479001600

    def time_check():
        return time.time() - start_time > DEADLINE

    # Generate all divisors of 12! = 2^10 * 3^5 * 5^2 * 7 * 11
    def get_divisors():
        divs = [1]
        for p, e in [(2, 10), (3, 5), (5, 2), (7, 1), (11, 1)]:
            new_divs = []
            for d in divs:
                for k in range(e + 1):
                    new_divs.append(d * (p ** k))
            divs = new_divs
        return sorted(divs)

    all_divs = get_divisors()
    # Valid divisors: > 12 and <= target
    valid_divs = [d for d in all_divs if d > 12]
    n = len(valid_divs)

    # Precompute logs for pruning
    log_valid = [log(d) for d in valid_divs]
    log_target = log(target)

    # Suffix sum of logs: suffix_log[i] = sum of log_valid[i:]
    # Used for upper bound: max product achievable from valid_divs[i:] is at most exp(suffix_log[i])
    # But we can only pick a subset, so the true max is product of largest k elements.
    # For pruning: if current_prod * (product of largest min(remaining, n-i) from valid_divs[i:]) < target, prune.
    # Precompute suffix products of the largest elements is tricky; use a simpler bound.

    # Better pruning: 
    # 1. If current_prod * (max_candidate)^(remaining_slots) < target, can't reach target.
    # 2. If current_prod * (min_candidate_from_i)^(remaining_slots) > target, already too big.
    # But these are loose. Let's use the actual largest available elements.

    # Precompute: for each index i, the product of the largest k elements from valid_divs[i:]
    # for k = 1..12. This is expensive but n is only 792.
    # Actually, let's just use a simpler approach: 
    # The maximum product we can get from valid_divs[i:] picking at most r elements
    # is the product of the largest min(r, n-i) elements.
    # Since valid_divs is sorted ascending, the largest elements are at the end.

    # For efficiency, precompute suffix_max_prod[i][k] = product of largest k elements from valid_divs[i:]
    # But that's O(n^2 * 12) which is ~792^2 * 12 ≈ 7.5M, might be slow.
    # Instead, use a simpler bound: current_prod * (valid_divs[-1])^remaining < target => prune
    # And: current_prod * (valid_divs[i])^remaining > target => prune (since all remaining >= valid_divs[i])

    # Actually, let's use a tighter bound. Precompute for each i the product of the 
    # largest k elements from valid_divs[i:] for k up to 12.
    # Since n=792 and we need k up to 12, this is manageable.

    # suffix_max_prod[i] = list of length 13, where suffix_max_prod[i][k] = product of largest k from valid_divs[i:]
    # If k > n-i, set to infinity or handle separately.

    # Let's build this:
    # For i from n-1 down to 0:
    #   suffix_max_prod[i][0] = 1
    #   For k from 1 to min(12, n-i):
    #     The largest k from valid_divs[i:] are valid_divs[n-k:n] (since sorted)
    #     But wait, that's not right. The largest k from valid_divs[i:] are the last k elements 
    #     of valid_divs[i:], i.e., valid_divs[n-k:n] if n-k >= i, else... 
    #     Actually, valid_divs[i:] has n-i elements. The largest k are valid_divs[n-k:n] if k <= n-i.
    #     So product = prod(valid_divs[n-k:n])
    #     This is independent of i as long as k <= n-i!
    #     So we can precompute global_max_prod[k] = product of largest k elements of valid_divs.

    global_max_prod = [1] * 13
    for k in range(1, 13):
        if k <= n:
            prod = 1
            for j in range(k):
                prod *= valid_divs[n - 1 - j]
            global_max_prod[k] = prod
        else:
            global_max_prod[k] = float('inf')

    # Now the DFS:
    # We pick factors in increasing order to ensure distinctness and natural ordering.
    # State: (start_idx, current_prod, chosen_count)
    # We want to find ANY valid completion, and by always trying smallest first, 
    # the first solution found will have the smallest possible largest factor? 
    # Not necessarily. We need to be more careful.

    # Better approach: DFS that finds the minimum maximum factor.
    # We'll do a DFS that tries to build a solution, and we track the best (smallest max) found.
    # To prune: if the current largest chosen factor is already >= best_max, we can prune 
    # only if we're sure we can't do better. Actually, we want to minimize the max, so 
    # if we've already found a solution with max M, we only need to find solutions with max < M.

    # Let's do: iterative deepening on the maximum factor M.
    # For a given M, check if there's a decomposition with all factors in [13, M].
    # Binary search on M.

    # But the previous approach was slow. Let's optimize the DFS for a fixed M.

    def can_solve(M):
        """Check if 12! can be decomposed into distinct factors all in [13, M]."""
        cands = [d for d in valid_divs if d <= M]
        nc = len(cands)
        if nc == 0:
            return None

        # Precompute log of cands
        log_cands = [log(c) for c in cands]

        # Precompute global max prod for cands
        cands_max_prod = [1] * 13
        for k in range(1, 13):
            if k <= nc:
                prod = 1
                for j in range(k):
                    prod *= cands[nc - 1 - j]
                cands_max_prod[k] = prod
            else:
                cands_max_prod[k] = float('inf')

        def _dfs(start_idx, current_prod, chosen):
            if time_check():
                return None

            if current_prod == target:
                if 1 <= len(chosen) <= 12:
                    return list(chosen)
                return None

            if current_prod > target:
                return None

            if len(chosen) >= 12:
                return None

            remaining_slots = 12 - len(chosen)
            avail = nc - start_idx
            if avail == 0:
                return None

            # Prune 1: max possible product from cands[start_idx:] picking at most remaining_slots
            pick_k = min(remaining_slots, avail)
            if current_prod * cands_max_prod[pick_k] < target:
                return None

            # Prune 2: min possible product from cands[start_idx:] picking exactly 1 element
            # If current_prod * cands[start_idx] > target, prune (but this is handled by current_prod > target)
            # Better: if even the smallest available factor makes current_prod exceed target, prune
            if current_prod * cands[start_idx] > target:
                return None

            # Try each candidate
            for i in range(start_idx, nc):
                d = cands[i]

                if current_prod * d > target:
                    break

                if target % d != 0:
                    continue

                new_chosen = chosen + [d]
                res = _dfs(i + 1, current_prod * d, new_chosen)
                if res is not None:
                    return res

                if time_check():
                    return None

            return None

        return _dfs(0, 1, [])

    # Binary search on M
    lo = 13
    hi = target
    best_sol = [target]
    best_M = target

    while lo <= hi:
        if time_check():
            break

        mid = (lo + hi) // 2
        sol = can_solve(mid)
        if sol is not None:
            best_sol = sol
            best_M = mid
            hi = mid - 1
        else:
            lo = mid + 1

    return best_sol
