import time
import math
import numpy as np

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    
    start_time = time.time()
    deadline = start_time + 85.0
    
    def lcm(a, b):
        return a * b // math.gcd(a, b)
    
    lcm_val = 1
    for d in divisors:
        lcm_val = lcm(lcm_val, d)
    
    # Known good solution with range 19
    known_good = [24, 40, 28, 30, 21, 22, 26, 33, 35, 39]
    best_range = max(known_good) - min(known_good)
    best_vals = known_good[:]
    
    # For each divisor, precompute a boolean mask over one LCM period
    # mask[d_idx][r] = True if (r mod d) == 0, i.e., r is a multiple of d
    masks = []
    for d in divisors:
        mask = np.zeros(lcm_val, dtype=bool)
        mask[::d] = True
        masks.append(mask)
    
    # For a window [A, A+W], the available residues for divisor d are
    # those r in [A, A+W] (mod lcm_val) where r % d == 0.
    # We need to find W such that for some A, we can assign distinct values.
    
    # Strategy: For each W, for each A (0 to lcm_val-1), check if each divisor
    # has at least one multiple in [A, A+W]. Then use backtracking to find
    # distinct assignments.
    
    # To speed up: precompute for each W and A, the count of available multiples
    # per divisor. But that's lcm_val * max_W * n which is too large.
    
    # Alternative: For each W, slide A from 0 to lcm_val-1. For each A, compute
    # the set of available values for each divisor. Use a bitset to check
    # if any value is shared across divisors (which would make assignment harder).
    
    # Let's use a different approach: for each W, iterate A, and for each divisor,
    # find the multiples in [A, A+W]. If any divisor has 0 multiples, skip.
    # If all have >= 1, try backtracking.
    
    # Precompute multiples lists for quick access
    max_W = 25
    multiples_by_divisor = []
    for d in divisors:
        m = 0
        mults = []
        while m < lcm_val:
            mults.append(m)
            m += d
        multiples_by_divisor.append(np.array(mults, dtype=np.int32))
    
    # Search for better solutions
    for W in range(9, max_W + 1):
        if time.time() > deadline:
            break
        if W >= best_range:
            break
        
        found = False
        for A in range(0, lcm_val):
            if time.time() > deadline:
                break
            
            candidates = []
            ok = True
            for d_idx, d in enumerate(divisors):
                mults = multiples_by_divisor[d_idx]
                # Find multiples in [A, A+W] considering wrap-around in LCM period
                # Values are in [0, lcm_val-1], window is [A, A+W]
                # A multiple m is in window if (m - A) mod lcm_val <= W
                # Equivalently: m in [A, A+W] or (m >= A+W-lcm_val and m < A) if wrap
                
                # Simpler: compute (m - A) % lcm_val <= W
                # But let's use searchsorted on the circular array
                
                # For circular window [A, A+W] mod lcm_val:
                # Case 1: A + W < lcm_val, window is [A, A+W]
                # Case 2: A + W >= lcm_val, window is [A, lcm_val-1] U [0, (A+W)%lcm_val]
                
                if A + W < lcm_val:
                    left = np.searchsorted(mults, A, side='left')
                    right = np.searchsorted(mults, A + W, side='right')
                    if left >= right:
                        ok = False
                        break
                    candidates.append(mults[left:right])
                else:
                    # Wrap around
                    left1 = np.searchsorted(mults, A, side='left')
                    right1 = len(mults)
                    left2 = 0
                    right2 = np.searchsorted(mults, (A + W) % lcm_val, side='right')
                    if left1 >= right1 and left2 >= right2:
                        ok = False
                        break
                    if left1 >= right1:
                        candidates.append(mults[left2:right2])
                    elif left2 >= right2:
                        candidates.append(mults[left1:right1])
                    else:
                        candidates.append(np.concatenate([mults[left1:right1], mults[left2:right2]]))
            
            if not ok:
                continue
            
            # Prune: if any divisor has more than a few candidates, skip for speed
            # Actually, let's try if the minimum candidate count is small
            min_count = min(len(c) for c in candidates)
            if min_count > 4:
                continue
            
            # Backtracking
            order = sorted(range(n), key=lambda i: len(candidates[i]))
            
            used = set()
            assignment = [None] * n
            
            def bt(idx):
                if idx == n:
                    return True
                i = order[idx]
                for v in candidates[i]:
                    vi = int(v)
                    if vi not in used:
                        used.add(vi)
                        assignment[i] = vi
                        if bt(idx + 1):
                            return True
                        used.discard(vi)
                        assignment[i] = None
                return False
            
            if bt(0):
                vals = assignment[:]
                r = max(vals) - min(vals)
                if r < best_range:
                    best_range = r
                    best_vals = vals[:]
                    found = True
        
        if best_range <= W:
            break
    
    return [int(v) for v in best_vals]
