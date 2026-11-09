import time
from math import gcd
from functools import reduce

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    MAX_V = 1000000
    deadline = time.time() + 85.0

    # Compute LCM of all divisors for periodicity
    def _lcm(a, b):
        return a // gcd(a, b) * b

    period = reduce(_lcm, divisors)  # 30030

    best_val = float('inf')
    best_assign = None

    # Precompute for each divisor d and window width W, the possible offsets
    # For a given L, candidates for divisor d are multiples in [L, L+W]
    # We use bitmask: bit i set means candidate at offset i (value L+i) is available for divisor d
    
    def get_candidate_offsets(L, W, d):
        """Return list of offsets (0..W) such that L+offset is divisible by d and L+offset <= MAX_V"""
        start_k = (L + d - 1) // d
        end_k = (L + W) // d
        if end_k < start_k:
            return []
        offsets = []
        for k in range(start_k, end_k + 1):
            val = k * d
            offset = val - L
            if 0 <= offset <= W and val <= MAX_V:
                offsets.append(offset)
        return offsets

    def backtrack(idx, used_mask, assign, L, W, cand_offsets):
        nonlocal best_val, best_assign
        if idx == n:
            val = max(assign) - min(assign)
            if val < best_val:
                best_val = val
                best_assign = assign[:]
            return True
        
        # MRV: pick the unassigned divisor with fewest available candidates
        best_i = -1
        best_count = float('inf')
        for i in range(idx, n):
            avail = 0
            for off in cand_offsets[i]:
                if not (used_mask & (1 << off)):
                    avail += 1
            if avail == 0:
                return False
            if avail < best_count:
                best_count = avail
                best_i = i
        
        if best_i < idx:
            # No valid assignment for remaining
            return False
        
        # Swap to use MRV order
        cand_offsets[idx], cand_offsets[best_i] = cand_offsets[best_i], cand_offsets[idx]
        divisors[idx], divisors[best_i] = divisors[best_i], divisors[idx]
        
        d = divisors[idx]
        for off in cand_offsets[idx]:
            if not (used_mask & (1 << off)):
                new_mask = used_mask | (1 << off)
                assign.append(L + off)
                if backtrack(idx + 1, new_mask, assign, L, W, cand_offsets):
                    # Restore before returning
                    cand_offsets[idx], cand_offsets[best_i] = cand_offsets[best_i], cand_offsets[idx]
                    divisors[idx], divisors[best_i] = divisors[best_i], divisors[idx]
                    return True
                assign.pop()
        
        # Restore
        cand_offsets[idx], cand_offsets[best_i] = cand_offsets[best_i], cand_offsets[idx]
        divisors[idx], divisors[best_i] = divisors[best_i], divisors[idx]
        return False

    # Search W from small values upward
    for W in range(16, 30):
        if time.time() > deadline:
            break
        found = False
        
        # Only need to check L in [1, period] due to periodicity
        # But also ensure L+W <= MAX_V
        max_L = min(period, MAX_V - W)
        
        for L in range(1, max_L + 1):
            if time.time() > deadline:
                break
            
            # Quick feasibility: each divisor must have at least one multiple in [L, L+W]
            feasible = True
            cand_offsets = []
            for d in divisors:
                offsets = get_candidate_offsets(L, W, d)
                if not offsets:
                    feasible = False
                    break
                cand_offsets.append(offsets)
            
            if not feasible:
                continue
            
            # Try backtracking with MRV
            used_mask = 0
            assign = []
            if backtrack(0, used_mask, assign, L, W, cand_offsets):
                found = True
                break
        
        if found:
            break

    if best_assign is None:
        return divisors[:]

    return [int(x) for x in best_assign]
