import time
import sys
from functools import lru_cache

def construct():
    n = 12
    max_L = 144
    period = 27720  # LCM(1,2,...,12)
    deadline = time.time() + 85  # safety margin before 90s kill

    # Precompute masks: masks[m][i] = bitmask of positions j in [0, max_L)
    # where (m + 1 + j) % i == 0
    masks = [[0] * (n + 1) for _ in range(period)]
    
    for m in range(period):
        for i in range(1, n + 1):
            target = (- (m + 1)) % i
            mask = 0
            if target < max_L:
                j = target
                while j < max_L:
                    mask |= (1 << j)
                    j += i
            masks[m][i] = mask

    def _can_match(m, L):
        """Check if there exists a matching for window length L using Hall's condition."""
        limit_mask = (1 << L) - 1
        
        # Build the restricted masks and check individual non-emptiness
        r_masks = []
        for i in range(1, n + 1):
            rm = masks[m][i] & limit_mask
            if rm == 0:
                return False
            r_masks.append(rm)
        
        # Hall's condition: for every subset S of {0..11}, |union of r_masks[S]| >= |S|
        # We check all 2^12 = 4096 subsets
        # Precompute union for each subset using DP
        # subset_union[s] = union of r_masks[i] for i in subset s
        # Use iterative method
        N = 1 << n
        # We'll compute union sizes incrementally
        # For efficiency, use the fact that we can build up from singletons
        
        # Check all subsets of size 1 (already done above)
        # For subsets of size >= 2, compute union and check |union| >= |subset|
        
        # Optimization: process subsets in order of increasing size
        # Use bit DP: union_mask[s] computed from union_mask[s without lsb] | r_masks[lsb_index]
        
        # But 4096 * 144-bit OR is fast enough
        # Let's just iterate all subsets
        
        for s in range(1, N):
            # Compute |S|
            popcount = bin(s).count('1')
            if popcount <= 1:
                continue  # already checked
            # Compute union of r_masks for indices in s
            union = 0
            ss = s
            while ss:
                lsb = ss & -ss
                idx = lsb.bit_length() - 1
                union |= r_masks[idx]
                ss ^= lsb
            # Check |union| >= popcount
            if bin(union).count('1') < popcount:
                return False
        
        return True

    def _f_val(m):
        """Find minimum L such that matching exists."""
        lo, hi = 1, max_L
        ans = max_L
        while lo <= hi:
            mid = (lo + hi) // 2
            if _can_match(m, mid):
                ans = mid
                hi = mid - 1
            else:
                lo = mid + 1
        return ans

    best_m = 0
    best_f = 0
    
    # First pass: scan all m in the period
    for m in range(period):
        if time.time() > deadline:
            break
        f = _f_val(m)
        if f > best_f:
            best_f = f
            best_m = m

    return int(best_m)

if __name__ == "__main__":
    construct()
