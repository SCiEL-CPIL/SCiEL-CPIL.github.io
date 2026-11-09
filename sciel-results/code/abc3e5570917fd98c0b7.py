import time
import math

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    MAX_V = 1000000
    
    # LCM of divisors
    period = 30030
    
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Search for W from 1 upwards
    for W in range(1, 1000):
        if time.time() > deadline:
            return divisors[:]
            
        # Precompute offsets for this W
        # offsets_table[i][r] -> list of offsets o in [0, W] such that (L+r+o) % d_i == 0
        # where L % d_i = r
        offsets_table = []
        for d in divisors:
            table = []
            for r in range(d):
                target = (-r) % d
                lst = []
                for o in range(W + 1):
                    if o % d == target:
                        lst.append(o)
                table.append(lst)
            offsets_table.append(table)
        
        # Iterate L from 1 to period
        for L in range(1, period + 1):
            if time.time() > deadline:
                return divisors[:]
                
            candidates = []
            valid = True
            for i, d in enumerate(divisors):
                r = L % d
                offs = offsets_table[i][r]
                if not offs:
                    valid = False
                    break
                candidates.append([(L + o) for o in offs])
            
            if not valid:
                continue
                
            # Fast check for uniqueness of single-candidate divisors
            used_vals = set()
            single_vals = []
            multi_indices = []
            
            for i in range(n):
                if len(candidates[i]) == 1:
                    val = candidates[i][0]
                    if val in used_vals:
                        valid = False
                        break
                    used_vals.add(val)
                    single_vals.append((i, val))
                else:
                    multi_indices.append(i)
                    
            if not valid:
                continue
                
            # Assign multi-candidate divisors
            assignment = [None] * n
            for i, val in single_vals:
                assignment[i] = val
                
            # Backtrack for multi
            def backtrack_multi(current_multi_idx):
                if current_multi_idx == len(multi_indices):
                    return True
                orig_idx = multi_indices[current_multi_idx]
                c_list = candidates[orig_idx]
                for val in c_list:
                    if val not in used_vals:
                        used_vals.add(val)
                        assignment[orig_idx] = val
                        if backtrack_multi(current_multi_idx + 1):
                            return True
                        used_vals.remove(val)
                        assignment[orig_idx] = None
                return False
            
            if backtrack_multi(0):
                # Verify bounds
                if all(1 <= v <= MAX_V for v in assignment):
                    return assignment
                    
    return divisors[:]
