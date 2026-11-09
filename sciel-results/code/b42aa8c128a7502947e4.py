"""Optimized search for distinct multiples of divisor set."""
import time
from math import gcd
from functools import reduce

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    MAX_V = 1000000
    deadline = time.time() + 85.0
    
    lcm_all = reduce(_lcm, divisors)
    
    best_val = float('inf')
    best_assign = None
    
    # Precompute all subsets of indices
    subsets = []
    for mask in range(1 << n):
        bits = []
        for i in range(n):
            if mask & (1 << i):
                bits.append(i)
        subsets.append(bits)
        
    # Sort subsets by size descending? No, Hall's condition: |Union| >= |Set|.
    # If we check smaller sets first, we might fail early.
    # But usually larger unions are more likely to satisfy size >= count.
    # Actually, if a small set fails, the whole thing fails.
    # So check small sets first?
    # Let's just iterate all.
    
    for W in range(9, 150):
        if time.time() > deadline:
            break
            
        # Precompute masks for this W
        # masks[i][r] is a bitmask of offsets m in [0, W] such that m % divisors[i] == r
        masks = []
        for i, d in enumerate(divisors):
            m_r = [0] * d
            for r in range(d):
                mask = 0
                # m in [0, W] such that m % d == r
                # m = r, r+d, r+2d, ... <= W
                start = r
                if start > W:
                    continue
                for m in range(start, W + 1, d):
                    mask |= (1 << m)
                m_r[r] = mask
            masks.append(m_r)
            
        # Iterate L
        for L in range(lcm_all):
            if time.time() > deadline:
                break
                
            # Get masks for this L
            current_masks = []
            for i, d in enumerate(divisors):
                r = (-L) % d
                current_masks.append(masks[i][r])
                
            # Check Hall's Condition
            valid = True
            for bits in subsets:
                if not bits:
                    continue
                union_mask = 0
                for idx in bits:
                    union_mask |= current_masks[idx]
                if union_mask.bit_count() < len(bits):
                    valid = False
                    break
                    
            if valid:
                # Find assignment via backtracking
                # Order by popcount
                order = sorted(range(n), key=lambda i: current_masks[i].bit_count())
                result = [None] * n
                used_offsets = set()
                
                def backtrack(idx):
                    if idx == len(order):
                        return True
                    i = order[idx]
                    mask = current_masks[i]
                    # Iterate through set bits
                    m = 0
                    temp_mask = mask
                    while temp_mask:
                        if not (temp_mask & 1):
                            temp_mask >>= 1
                            m += 1
                            continue
                        # bit m is set
                        if m not in used_offsets:
                            used_offsets.add(m)
                            result[i] = L + m
                            if backtrack(idx + 1):
                                return True
                            used_offsets.remove(m)
                            result[i] = None
                        temp_mask &= ~(1 << m)
                        temp_mask >>= 1
                        m += 1
                    return False
                
                if backtrack(0):
                    val = max(result) - min(result)
                    if val < best_val:
                        best_val = val
                        best_assign = result[:]
                    # Since we are iterating W in increasing order, 
                    # the first W that works gives the minimum possible span <= W.
                    # Is it possible that a larger W gives a smaller span?
                    # No, span is max-min. If span is S, it fits in window S.
                    # So min span = min W such that solution exists in window W.
                    return [int(x) for x in best_assign]
        
    if best_assign is None:
        # Fallback
        best_assign = divisors[:]
        
    return [int(x) for x in best_assign]
