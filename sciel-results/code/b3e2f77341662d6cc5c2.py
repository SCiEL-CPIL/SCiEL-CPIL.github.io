import time
import math

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    # Sort indices by divisor value descending to constrain search early
    sorted_indices = sorted(range(n), key=lambda i: divisors[i], reverse=True)
    sorted_divisors = [divisors[i] for i in sorted_indices]
    
    # LCM of the divisors to limit the periodicity of the pattern
    # 6, 10, 14, 15, 21, 22, 26, 33, 35, 39
    # Primes: 2, 3, 5, 7, 11, 13
    # LCM = 2 * 3 * 5 * 7 * 11 * 13 = 30030
    lcm_val = 30030
    
    start_time = time.time()
    time_limit = 85.0 # Safety margin
    
    # Heuristic fallback if search fails or times out
    def get_fallback():
        # Try to pack around a central value
        # Greedy: pick multiples close to 1000
        target = 1000
        assigned = {}
        used_vals = set()
        # Sort by divisor descending for greedy too
        for d in sorted_divisors:
            # Find multiple of d closest to target
            k = target // d
            candidates = [k*d, (k+1)*d, (k-1)*d, (k+2)*d, (k-2)*d]
            found = None
            for c in candidates:
                if c > 0 and c not in used_vals:
                    found = c
                    break
            if found is None:
                # Extend search
                for offset in range(3, 100):
                    c1 = (k+offset)*d
                    c2 = (k-offset)*d
                    if c1 > 0 and c1 not in used_vals:
                        found = c1
                        break
                    if c2 > 0 and c2 not in used_vals:
                        found = c2
                        break
            if found is None:
                # Last resort: just pick k*d and shift if needed? 
                # Or pick a large number
                found = d * 10000
                while found in used_vals:
                    found += d
            assigned[d] = found
            used_vals.add(found)
        
        # Map back to original order
        result = [0] * n
        for i in sorted_indices:
            result[i] = assigned[divisors[i]]
        return result

    best_result = None
    best_val = float('inf')

    # Iterate window size W
    # Minimum span for 10 distinct integers is 9
    for W in range(9, 100):
        if time.time() - start_time > time_limit:
            break
        
        # Iterate start L in one period
        for L in range(lcm_val):
            if time.time() - start_time > time_limit:
                break
            
            # Backtracking search for distinct multiples in [L, L+W]
            used = set()
            success = False
            values = [0] * n
            
            # Pre-calculate candidate lists for each divisor for this L, W
            candidates = []
            feasible = True
            for i in range(n):
                d = sorted_divisors[i]
                start_k = (L + d - 1) // d
                end_k = (L + W) // d
                cand_list = []
                for k in range(start_k, end_k + 1):
                    v = k * d
                    cand_list.append(v)
                candidates.append(cand_list)
                if len(cand_list) == 0:
                    feasible = False
                    break
            
            if not feasible:
                continue

            def backtrack(assigned_count):
                nonlocal success
                if assigned_count == n:
                    success = True
                    return True
                
                # MRV Heuristic: Pick unassigned divisor with fewest available candidates
                best_i = -1
                best_avail = []
                min_count = float('inf')
                
                for i in range(n):
                    if values[sorted_indices[i]] != 0:
                        continue
                    
                    avail = [v for v in candidates[i] if v not in used]
                    count = len(avail)
                    if count == 0:
                        return False # Prune
                    if count < min_count:
                        min_count = count
                        best_i = i
                        best_avail = avail
                
                if best_i == -1:
                    return False # Should not happen if assigned_count < n

                d = sorted_divisors[best_i]
                orig_idx = sorted_indices[best_i]
                
                for v in best_avail:
                    used.add(v)
                    values[orig_idx] = v
                    if backtrack(assigned_count + 1):
                        return True
                    used.remove(v)
                    values[orig_idx] = 0
                return False
            
            if backtrack(0):
                # Found a valid configuration for this W and L
                # Since we iterate W from 9 up, this is the minimal W
                # The raw value is W
                return values[:]
            
    # If loop finishes without finding (should not happen for small W), or timeout
    if best_result is not None:
        return best_result
    
    return get_fallback()
