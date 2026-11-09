"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq
from itertools import combinations

N = 5040
TARGET = 4979
DEADLINE_S = 80  # safety margin before 90s kill

def _divisors_of(n):
    divs = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
    return sorted(divs)

_DIVS = _divisors_of(N)
_DIVS_SET = set(_DIVS)
_DIVS_DESC = sorted(_DIVS, reverse=True)

def _greedy_solution():
    """Greedy: pick largest divisor <= remainder."""
    rem = TARGET
    chosen = []
    for d in _DIVS_DESC:
        if d <= rem:
            chosen.append(d)
            rem -= d
            if rem == 0:
                break
    if rem == 0:
        return chosen
    return None

def _best_known():
    """Return the best (fewest terms) solution found so far, or None."""
    sol = _greedy_solution()
    if sol is not None:
        return sol
    # Fallback: try to find any valid solution via small-sum DP
    return _dp_solution()

def _dp_solution():
    """DP to find minimum number of distinct divisors summing to TARGET."""
    # dp[s] = (min_count, frozenset_of_divisors) — but storing sets is expensive.
    # Instead, store min_count and reconstruct via parent pointers.
    # Since TARGET=4979 and divisors are ~60, this is feasible.
    dp = [None] * (TARGET + 1)
    dp[0] = (0, frozenset())
    
    for d in _DIVS:
        # Process in reverse to ensure each divisor used at most once
        for s in range(TARGET, d - 1, -1):
            if dp[s - d] is not None:
                cnt, fs = dp[s - d]
                new_cnt = cnt + 1
                if dp[s] is None or new_cnt < dp[s][0]:
                    dp[s] = (new_cnt, fs | {d})
    
    if dp[TARGET] is not None:
        return list(dp[TARGET][1])
    return None

def _branch_and_bound(best_size):
    """B&B to find solution with fewer than best_size terms."""
    # Sort divisors descending for B&B
    divs = _DIVS_DESC
    n = len(divs)
    best = [None]
    start = time.time()
    
    # Precompute suffix sums for pruning
    suffix_sum = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + divs[i]
    
    def dfs(idx, remaining, chosen, depth, max_depth):
        if time.time() - start > DEADLINE_S:
            return
        if remaining == 0:
            if len(chosen) < best_size:
                best[0] = list(chosen)
            return
        if idx >= n or depth == max_depth:
            return
        if suffix_sum[idx] < remaining:
            return  # Even taking all remaining divisors won't reach target
        
        for i in range(idx, n):
            d = divs[i]
            if d > remaining:
                continue
            # Prune: if we take this divisor, can we still reach remaining - d
            # with at most (max_depth - depth - 1) more divisors?
            new_rem = remaining - d
            if new_rem == 0:
                if depth + 1 < best_size:
                    best[0] = list(chosen) + [d]
                continue
            # Simple bound: need at least ceil(new_rem / max_remaining_divisor) more terms
            # But we'll rely on suffix_sum and depth limit
            chosen.append(d)
            dfs(i + 1, new_rem, chosen, depth + 1, max_depth)
            chosen.pop()
            if best[0] is not None:
                return  # Found a better solution, stop
    
    # Try smaller and smaller max_depth
    for md in range(1, best_size):
        start = time.time()
        dfs(0, TARGET, [], 0, md)
        if best[0] is not None:
            return best[0]
        if time.time() - start > DEADLINE_S - 5:
            break
    return best[0]

def construct():
    start = time.time()
    
    # Step 1: Get initial solution
    sol = _greedy_solution()
    if sol is None:
        sol = _dp_solution()
    if sol is None:
        # Should not happen, but return empty as fallback
        return []
    
    best_size = len(sol)
    
    # Step 2: Try to improve with B&B if time permits
    if best_size > 1 and time.time() - start < DEADLINE_S - 10:
        improved = _branch_and_bound(best_size)
        if improved is not None:
            sol = improved
            best_size = len(sol)
    
    # Final validation
    assert len(sol) <= 256, "Too many divisors"
    assert len(set(sol)) == len(sol), "Duplicate divisors"
    for d in sol:
        assert d in _DIVS_SET, f"{d} is not a divisor of {N}"
    assert sum(sol) == TARGET, f"Sum is {sum(sol)}, expected {TARGET}"
    
    return sol

# EVOLVE-BLOCK-END
