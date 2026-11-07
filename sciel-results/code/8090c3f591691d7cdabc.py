"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    limit = 85  # safety margin before 90s kill

    N = 12
    # lcm(1..12) = 27720
    period = 27720

    best_m = 0
    best_f = 0

    # Precompute divisors info for speed
    # For each i in 1..12, we need to find multiples in (m, m+L]
    
    def get_candidates(m, L, i):
        """Return list of x in (m, m+L] divisible by i"""
        start = m + 1
        rem = start % i
        if rem == 0:
            first = start
        else:
            first = start + (i - rem)
        
        if first > m + L:
            return []
        
        res = []
        for x in range(first, m + L + 1, i):
            res.append(x)
        return res

    def can_match(m, L):
        # Build candidate lists for i=1..12
        cands = []
        for i in range(1, N + 1):
            cands.append(get_candidates(m, L, i))
        
        # Check if any i has no candidates
        for i in range(N):
            if len(cands[i]) == 0:
                return False
        
        # Sort indices by length of candidates (most constrained first)
        indices = list(range(N))
        indices.sort(key=lambda idx: len(cands[idx]))
        
        used = set()
        
        def backtrack(k):
            if k == N:
                return True
            i = indices[k]
            for x in cands[i]:
                if x not in used:
                    used.add(x)
                    if backtrack(k + 1):
                        return True
                    used.remove(x)
            return False
        
        return backtrack(0)

    def compute_f(m):
        # Binary search on L
        # Max possible L is small, try up to 200
        lo = 1
        hi = 200
        while lo < hi:
            mid = (lo + hi) // 2
            if can_match(m, mid):
                hi = mid
            else:
                lo = mid + 1
        return lo

    # Search m in one period
    # We can parallelize or just loop. 27720 iterations is fast.
    # Check time periodically
    check_interval = 200
    
    for m in range(period):
        if m % check_interval == 0:
            if time.time() - start_time > limit - 5:
                break
        
        f_val = compute_f(m)
        if f_val > best_f:
            best_f = f_val
            best_m = m

    return best_m
# EVOLVE-BLOCK-END
