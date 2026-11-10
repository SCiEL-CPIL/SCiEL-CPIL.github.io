"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
from functools import lru_cache

P = {'n': 12, 'maximum_start': 1000000}

def construct():
    """
    Find m in [0, 1000000] that maximizes f(12, m), where f(12, m) is the
    least positive L such that (m, m+L] contains 12 distinct integers b_1..b_12
    with i | b_i.
    
    Uses Hall's theorem with incremental window updates for efficient search.
    """
    n = 12
    max_m = 1000000
    period = 27720  # lcm(1,2,...,12)
    full_mask = (1 << 12) - 1
    
    # Precompute mask for each residue r in [0, 27720)
    mask_by_residue = [0] * period
    for r in range(period):
        mask = 0
        for i in range(1, 13):
            if r % i == 0:
                mask |= (1 << (i - 1))
        mask_by_residue[r] = mask
    
    # Precompute popcount
    popcount = [0] * 4096
    for i in range(4096):
        popcount[i] = bin(i).count('1')
    
    # For each L, precompute the SOS DP structure to check Hall's condition
    # But we need to do this for each m... Let's use a different approach.
    
    # For a given L and m, we have freq[mask] for masks in (m, m+L].
    # We need to check: for all S, sum_{mask ⊆ comp(S)} freq[mask] <= L - popcount[S].
    
    # Let's precompute for each L, and for each m, the F array via SOS DP.
    # But that's 27720 * 144 * 49152 ops = too much.
    
    # Alternative: For each m, binary search on L. For each L in binary search,
    # build freq and do SOS DP. 
    # SOS DP is 12 * 4096 = 49152 ops. Binary search is ~8 iterations.
    # So ~393k ops per m. 27720 m values -> ~10.9e9 ops. Still too slow.
    
    # Let's try a different approach: for each L from 144 down to 1,
    # find all m where can_match_hall(m, L) is True. The answer for m is
    # the smallest L where it's True.
    
    # For a fixed L, can we efficiently find all m where Hall's condition holds?
    # The condition depends on the multiset of masks in (m, m+L].
    # As m varies, the window slides. We can use a sliding window approach.
    
    # For fixed L, initialize freq for m=0: window (0, L] = {1, 2, ..., L}.
    # Then for m=1: remove mask of 1, add mask of L+1.
    # After each update, we need to check if Hall's condition holds.
    # Checking Hall's condition requires SOS DP (49152 ops) + 4096 checks.
    # For 27720 m values and 144 L values, that's 27720 * 144 * 53248 = ~2.1e11. Too slow.
    
    # We need a faster Hall check. 
    # Key observation: Hall's condition fails iff there exists S with 
    # |{x in window : mask(x) & S == 0}| > L - |S|.
    # The LHS is the number of elements in the window whose mask is a subset of comp(S).
    # Let G[T] = count of elements in window with mask ⊆ T. Then condition is G[comp(S)] <= L - |S|.
    # G[T] = sum_{mask ⊆ T} freq[mask]. This is what SOS DP computes.
    
    # Can we avoid full SOS DP? Note that we only need G[comp(S)] for all S,
    # which is G[T] for all T (since comp is a bijection). So we need all G[T].
    # SOS DP is the standard way to compute all G[T] from freq in O(12 * 2^12).
    
    # Let's just optimize the constants and hope it fits in 60s.
    # Actually, let's reduce the search space. The maximum f(12,m) is at most 144.
    # We can do binary search on L for each m, but we can also try to find the max L
    # by checking from high L downward.
    
    # Better idea: For each m, we can check L=144, 143, ..., 1 and stop at the first
    # that works. But that's up to 144 checks per m, each with SOS DP. Worse than binary search.
    
    # Let's just do the exhaustive search with binary search and SOS DP, and optimize the inner loop.
    
    def can_match_hall(m, L):
        """Check Hall's condition for interval (m, m+L]."""
        if L <= 0:
            return False
        
        freq = [0] * 4096
        for x in range(m + 1, m + L + 1):
            r = x % period
            freq[mask_by_residue[r]] += 1
        
        # SOS DP
        F = freq[:]
        for i in range(12):
            bit = 1 << i
            for mask in range(4096):
                if mask & bit:
                    F[mask] += F[mask ^ bit]
        
        for S in range(4096):
            comp = full_mask ^ S
            if F[comp] > L - popcount[S]:
                return False
        return True
    
    def f_value(m):
        """Compute f(12, m)."""
        lo, hi = 1, 144
        while lo < hi:
            mid = (lo + hi) // 2
            if can_match_hall(m, mid):
                hi = mid
            else:
                lo = mid + 1
        return lo
    
    best_m = 0
    best_f = 0
    
    for m in range(period):
        f = f_value(m)
        if f > best_f:
            best_f = f
            best_m = m
    
    return int(best_m)

# EVOLVE-BLOCK-END
