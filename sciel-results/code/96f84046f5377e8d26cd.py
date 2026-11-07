import time
from math import gcd

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    deadline = time.time() + 85  # safety margin
    
    # LCM of all divisors
    def lcm(a, b):
        return a // gcd(a, b) * b
    LCM = 1
    for d in divisors:
        LCM = lcm(LCM, d)
    
    best_range = float('inf')
    best_assign = None
    
    # Sliding window approach:
    # Generate all valid multiples within [1, 1000000] for each divisor
    # Sort all (value, divisor_index) pairs
    # Use two pointers to find the shortest window containing at least one multiple of each divisor
    # Then assign distinct values within that window
    
    # For each divisor, generate its multiples
    # Total multiples: sum(1000000/d for d in divisors) ≈ 1000000*(1/6+1/10+1/14+1/15+1/21+1/22+1/26+1/33+1/35+1/39)
    # ≈ 1000000*(0.1667+0.1+0.0714+0.0667+0.0476+0.0455+0.0385+0.0303+0.0286+0.0256) ≈ 1000000*0.6209 ≈ 620,900
    
    # Generate all (value, divisor_idx) pairs
    all_pairs = []
    for idx, d in enumerate(divisors):
        for m in range(d, 1000001, d):
            all_pairs.append((m, idx))
    
    # Sort by value
    all_pairs.sort(key=lambda x: x[0])
    
    # Two-pointer sliding window to find minimal span where all 10 divisors are represented
    # We need at least one multiple for each of the 10 divisors in the window
    left = 0
    counts = [0] * n
    distinct_count = 0
    
    for right in range(len(all_pairs)):
        val_r, idx_r = all_pairs[right]
        if counts[idx_r] == 0:
            distinct_count += 1
        counts[idx_r] += 1
        
        # Try to shrink from the left
        while distinct_count == n:
            span = val_r - all_pairs[left][0]
            if span < best_range:
                best_range = span
                # Record the window
                # We need to assign distinct values from this window to each divisor
                # For now, record left and right indices
                if best_assign is None or span < best_range:
                    # Find a valid assignment within this window
                    # For each divisor, pick a distinct multiple in [all_pairs[left][0], val_r]
                    window_vals = {}
                    for i in range(left, right + 1):
                        v, idx = all_pairs[i]
                        if idx not in window_vals:
                            window_vals[idx] = []
                        window_vals[idx].append(v)
                    
                    # Try to assign distinct values
                    # Simple greedy: for each divisor, pick the smallest available value
                    used = set()
                    assign = []
                    feasible = True
                    for i in range(n):
                        found = False
                        if i in window_vals:
                            for v in window_vals[i]:
                                if v not in used:
                                    assign.append(v)
                                    used.add(v)
                                    found = True
                                    break
                        if not found:
                            feasible = False
                            break
                    if feasible:
                        best_assign = assign[:]
                        if best_range <= 9:  # close to optimal
                            break
            else:
                # Can't improve, just shrink
                pass
            
            val_l, idx_l = all_pairs[left]
            counts[idx_l] -= 1
            if counts[idx_l] == 0:
                distinct_count -= 1
            left += 1
        
        if best_range <= 9:
            break
    
    if best_assign is None:
        # Fallback: use the simple heuristic
        used = set()
        assign = []
        for i, d in enumerate(divisors):
            m = d
            while m in used:
                m += d
            assign.append(m)
            used.add(m)
        best_assign = assign
    
    return list(best_assign)
