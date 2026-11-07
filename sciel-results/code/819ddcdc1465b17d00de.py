import time
import sys

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    max_val = 1000000
    
    # Large divisors (d > 16) have at most 1 multiple in a window of length 16.
    # Small divisors (d <= 16) can have 1 or 2.
    # We search for minimal S.
    # Min S is at least 9 (10 distinct ints).
    
    # Periodicity for the large divisors
    # LCM of 21, 22, 26, 33, 35, 39
    # 21=3*7, 22=2*11, 26=2*13, 33=3*11, 35=5*7, 39=3*13
    # LCM = 2*3*5*7*11*13 = 30030
    
    large_divs = [21, 22, 26, 33, 35, 39]
    small_divs = [6, 10, 14, 15]
    
    # Map divisor to index in original list
    # divisors: 6, 10, 14, 15, 21, 22, 26, 33, 35, 39
    # indices:   0,  1,  2,  3,  4,  5,  6,  7,  8,  9
    
    # Precompute indices
    idx_map = {d: i for i, d in enumerate(divisors)}
    
    start_time = time.time()
    
    # Search S from 9 to 16 (current best is 16, so we hope to find < 16)
    for S in range(9, 17):
        # If time is running out, break
        if time.time() - start_time > 50:
            break
            
        for L in range(30030):
            # Check large divisors
            large_vals = {}
            valid = True
            for d in large_divs:
                # Find multiple of d in [L, L+S]
                # First multiple >= L
                m = ((L + d - 1) // d) * d
                if m <= L + S:
                    large_vals[d] = m
                else:
                    valid = False
                    break
            
            if not valid:
                continue
                
            # Check distinctness of large values
            vals = list(large_vals.values())
            if len(set(vals)) != len(vals):
                continue
                
            occupied = set(vals)
            
            # Check small divisors
            # We need to assign values to 6, 10, 14, 15
            # Candidates for each
            candidates = {}
            for d in small_divs:
                cands = []
                m = ((L + d - 1) // d) * d
                while m <= L + S:
                    if m not in occupied:
                        cands.append(m)
                    m += d
                candidates[d] = cands
                
            # Backtrack/assign distinct values
            # Since sizes are small, we can just try permutations
            # Order: 6, 10, 14, 15
            assigned = {}
            used_small = set()
            
            def backtrack(i):
                if i == 4:
                    return True
                d = small_divs[i]
                for v in candidates[d]:
                    if v not in used_small:
                        used_small.add(v)
                        assigned[d] = v
                        if backtrack(i + 1):
                            return True
                        used_small.remove(v)
                return False
                
            if backtrack(0):
                # Construct result
                res = [0] * 10
                for d, v in assigned.items():
                    res[idx_map[d]] = v
                for d, v in large_vals.items():
                    res[idx_map[d]] = v
                
                # Final validation
                if len(set(res)) == 10 and all(x > 0 for x in res) and all(x <= max_val for x in res):
                    # Ensure types are int
                    return [int(x) for x in res]
                    
    # Fallback: return the seed solution if nothing better found
    return divisors[:]
