"""Rewritten solver for the finite divisor-sum instance (n=5040, target=4979)."""
import time
from itertools import combinations

_N = 5040
_TARGET = 4979

def _divisors(n):
    divs = []
    for d in range(1, int(n ** 0.5) + 1):
        if n % d == 0:
            divs.append(d)
            if d != n // d:
                divs.append(n // d)
    divs.sort(reverse=True)
    return divs

_DIVS = _divisors(_N)
_DIVS_SET = set(_DIVS)

# Precompute all divisor sums up to a small limit to speed up membership tests.
# We only need sums up to _TARGET.
_MAX_SUM = _TARGET
# Use a bitset-like approach with a set of achievable sums? That's too large.
# Instead, we will use a meet-in-the-middle or dynamic programming approach for small k.

def _can_sum_to(target, divs, k):
    """
    Check if there exists a subset of 'divs' of size exactly k that sums to 'target'.
    Uses meet-in-the-middle for efficiency when len(divs) is moderate.
    """
    n = len(divs)
    if k > n:
        return False
    if k == 0:
        return target == 0
    if k == 1:
        return target in _DIVS_SET and target in divs
    if k == n:
        return sum(divs) == target
    
    # Meet in the middle
    mid = n // 2
    left = divs[:mid]
    right = divs[mid:]
    
    # Generate all subset sums for left and right with their sizes
    # Since n is at most 60, 2^30 is too large. We need a smarter way or rely on k being small.
    # The divisors of 5040 are 60. 2^60 is huge.
    # However, k is likely small. Let's use a DP or bounded search.
    
    # Alternative: Since we are minimizing k, we can iterate k from 1 upwards.
    # For a fixed k, we can use a recursive backtracking with pruning, or generate combinations.
    # If k is small (e.g., <= 5), itertools.combinations is fast enough?
    # C(60, 5) = 5,461,512. This is borderline in Python but might be okay if optimized or if k is very small.
    # C(60, 6) = 50,063,860. Too slow.
    
    # Let's use a DP table for achievable sums with specific counts?
    # dp[count][sum] = True/False
    # Max sum is 4979. Max count is probably small (say 10).
    # States: 10 * 5000 = 50,000. Transitions: 60 * 50,000 = 3,000,000 per count step?
    # No, we iterate through divisors and update DP.
    # dp[i][j][s] = True if we can pick j divisors from the first i divisors to sum to s.
    # i up to 60, j up to K, s up to 4979.
    # 60 * 10 * 5000 = 3,000,000 states. This is very fast in Python if using lists of booleans or bitsets.
    
    # Let's implement DP with bitsets for speed.
    # dp[j] is a set of sums achievable with j divisors.
    # Or better, a list of sets.
    
    # Actually, since we just need to find *one* solution for the minimal k, we can do BFS/DFS with pruning.
    # But DP is more robust.
    
    # Let's use a list of sets for sums.
    # dp[j] = set of sums achievable using exactly j divisors from the processed list.
    
    # Optimization: We only care about sums <= target.
    
    dp = [set() for _ in range(k + 1)]
    dp[0].add(0)
    
    for d in divs:
        if d > target:
            continue
        # Update in reverse order of j to avoid reusing the same divisor
        for j in range(k, 0, -1):
            # For each sum in dp[j-1], add d
            new_sums = set()
            for s in dp[j-1]:
                ns = s + d
                if ns <= target:
                    new_sums.add(ns)
            dp[j] |= new_sums
            if target in dp[j]:
                return True
    return target in dp[k]

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    # Sort divisors in descending order to find large sums with fewer terms?
    # Actually, for minimization of count, we should try to use the largest divisors first.
    # The greedy approach in the seed program likely used many terms.
    # Let's find the minimal k.
    
    # Upper bound on k: The greedy solution gives an upper bound.
    # Let's compute the greedy solution first to get an upper bound and a fallback.
    remainder = _TARGET
    greedy_chosen = []
    for d in _DIVS:
        if d <= remainder:
            greedy_chosen.append(d)
            remainder -= d
    # Note: The greedy approach might not reach 0 if we just pick largest <= remainder?
    # 5040 divisors: 5040, 2520, 1680, 1260, 1008, 840, 720, 630, 560, 504, 420, 360, 336, 315, 280, 252, 240, 210, 180, 168, 140, 126, 120, 105, 90, 84, 80, 70, 63, 56, 45, 42, 40, 35, 30, 28, 24, 21, 20, 18, 15, 14, 12, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1.
    # Sum of all divisors is 21612. Target 4979 is much smaller.
    # Greedy:
    # 4979. Largest div <= 4979 is 2520. Rem 2459.
    # Next 1680. Rem 779.
    # Next 720. Rem 59.
    # Next 56. Rem 3.
    # Next 3. Rem 0.
    # So greedy uses 5 divisors: [2520, 1680, 720, 56, 3].
    # Can we do it with fewer?
    # k=4?
    # k=3?
    # k=2?
    # k=1? 4979 is not a divisor.
    
    # Let's check k=1 to k=4 quickly.
    
    # k=1
    if _TARGET in _DIVS_SET:
        return [_TARGET]
        
    # k=2
    # Check all pairs
    for i in range(len(_DIVS)):
        d1 = _DIVS[i]
        if d1 > _TARGET:
            continue
        rem = _TARGET - d1
        if rem in _DIVS_SET and rem != d1:
            return [d1, rem]
            
    # k=3
    # Check all triples? C(60,3) = 34220. Very fast.
    for i in range(len(_DIVS)):
        d1 = _DIVS[i]
        if d1 > _TARGET:
            continue
        rem1 = _TARGET - d1
        for j in range(i+1, len(_DIVS)):
            d2 = _DIVS[j]
            if d2 > rem1:
                continue
            rem2 = rem1 - d2
            if rem2 in _DIVS_SET and rem2 > d2: # ensure distinct and ordered? 
                # We need to ensure rem2 is a divisor and distinct from d1, d2.
                # Since we iterate d1, d2 from sorted list, and check rem2 in set,
                # we just need to ensure rem2 != d1 and rem2 != d2.
                # Also rem2 should be a divisor.
                # To avoid duplicates and ensure we find a valid set, we can just return.
                # But we must ensure they are distinct.
                # d1, d2 are distinct. rem2 is distinct from d1, d2?
                # If rem2 == d1, then d1 + d2 + d1 = Target -> 2*d1 + d2 = Target.
                # If rem2 == d2, then d1 + 2*d2 = Target.
                # The problem requires DISTINCT divisors.
                if rem2 != d1 and rem2 != d2:
                    return [d1, d2, rem2]
                    
    # k=4
    # C(60,4) = 487635. Still fast enough?
    # 487k iterations is fine in Python.
    for i in range(len(_DIVS)):
        d1 = _DIVS[i]
        if d1 > _TARGET:
            continue
        rem1 = _TARGET - d1
        for j in range(i+1, len(_DIVS)):
            d2 = _DIVS[j]
            if d2 > rem1:
                continue
            rem2 = rem1 - d2
            for k in range(j+1, len(_DIVS)):
                d3 = _DIVS[k]
                if d3 > rem2:
                    continue
                rem3 = rem2 - d3
                if rem3 in _DIVS_SET:
                    # Check distinctness
                    if rem3 != d1 and rem3 != d2 and rem3 != d3:
                        return [d1, d2, d3, rem3]
                        
    # k=5
    # Greedy already found a solution with 5.
    # We can just return the greedy solution if it's valid.
    if sum(greedy_chosen) == _TARGET and len(greedy_chosen) == 5:
        return greedy_chosen
        
    # If we reach here, we need to search for k=5 properly or higher.
    # But greedy likely works. Let's verify greedy.
    # Greedy: 2520, 1680, 720, 56, 3. Sum = 4979. Distinct? Yes. Divisors? Yes.
    # So k=5 is achievable.
    # Did we miss a k=4 solution?
    # The loop above should have found it if it existed.
    # So the answer is likely 5.
    
    # Just to be safe, if greedy failed (unlikely), we do a DP search for k=5.
    # But for now, return greedy.
    return greedy_chosen
