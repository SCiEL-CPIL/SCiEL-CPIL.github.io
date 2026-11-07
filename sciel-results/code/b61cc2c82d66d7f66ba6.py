import time
from math import gcd

DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
N = len(DIVISORS)
MAX_VAL = 1000000

def _lcm(a, b):
    return a * b // gcd(a, b)

# Compute LCM of all divisors for periodicity
LCM_ALL = 1
for d in DIVISORS:
    LCM_ALL = _lcm(LCM_ALL, d)
# LCM_ALL = 30030

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    # Pre-compute divisors for faster access
    divs = DIVISORS
    
    # We search for the minimum width W.
    for W in range(9, 500):
        if time.time() > deadline:
            break
            
        # Scan starting positions L.
        # Optimization: We can step L by 1, but we can also check if L is valid quickly.
        # The condition for L to have a multiple of d in [L, L+W] is:
        # Let rem = L % d.
        # If rem == 0, first mult is L.
        # If rem != 0, first mult is L + (d - rem).
        # We need first mult <= L + W.
        # This is equivalent to:
        # If rem == 0, 0 <= W (always true).
        # If rem != 0, d - rem <= W => rem >= d - W.
        # So for each d, valid remainders are:
        # 0, or d-W ... d-1 (if W < d).
        # If W >= d, all remainders are valid.
        
        # We can pre-calculate the set of valid remainders for each d for this W.
        valid_rems = []
        for d in divs:
            if W >= d:
                valid_rems.append(None) # All remainders valid
            else:
                # Valid: 0, d-W, d-W+1, ..., d-1
                # Count: 1 + W
                valid_rems.append(set([0] + list(range(d-W, d))))
        
        # Now iterate L
        for L in range(1, LCM_ALL + 1):
            if time.time() > deadline:
                break
                
            if L + W > MAX_VAL:
                break
            
            # Check feasibility and build candidates
            feasible = True
            candidates = []
            
            for i in range(N):
                d = divs[i]
                rem = L % d
                vr = valid_rems[i]
                
                if vr is not None and rem not in vr:
                    feasible = False
                    break
                
                # Generate candidates
                # First multiple >= L
                if rem == 0:
                    first_val = L
                else:
                    first_val = L + (d - rem)
                
                cand_list = []
                v = first_val
                while v <= L + W:
                    cand_list.append(v)
                    v += d
                candidates.append(cand_list)
            
            if not feasible:
                continue
            
            # Solve SDR
            # Since domains are small, we can use a simple backtracking
            # We'll use a recursive function with MRV
            
            domains = candidates
            assignment = [None] * N
            used = set()
            
            # Unit propagation
            changed = True
            while changed:
                changed = False
                for i in range(N):
                    if assignment[i] is None:
                        if not domains[i]:
                            feasible = False
                            break
                        if len(domains[i]) == 1:
                            val = domains[i][0]
                            if val in used:
                                feasible = False
                                break
                            assignment[i] = val
                            used.add(val)
                            changed = True
                if not feasible:
                    break
                if changed:
                    for i in range(N):
                        if assignment[i] is None:
                            new_d = [v for v in domains[i] if v not in used]
                            if len(new_d) != len(domains[i]):
                                domains[i] = new_d
                                if not domains[i]:
                                    feasible = False
                                    break
                                changed = True
            
            if not feasible:
                continue
            
            if all(a is not None for a in assignment):
                return [int(x) for x in assignment]
            
            # Backtracking
            unassigned = [i for i in range(N) if assignment[i] is None]
            unassigned.sort(key=lambda i: len(domains[i]))
            
            def backtrack(idx):
                if idx == len(unassigned):
                    return True
                i = unassigned[idx]
                for val in domains[i]:
                    if val not in used:
                        assignment[i] = val
                        used.add(val)
                        
                        saved = []
                        valid = True
                        for j in unassigned[idx+1:]:
                            if assignment[j] is None:
                                saved.append((j, domains[j][:]))
                                new_d = [v for v in domains[j] if v != val]
                                if not new_d:
                                    valid = False
                                    break
                                domains[j] = new_d
                        
                        if valid and backtrack(idx + 1):
                            return True
                        
                        used.remove(val)
                        assignment[i] = None
                        if valid:
                            for j, d in saved:
                                domains[j] = d
                return False
            
            if backtrack(0):
                return [int(x) for x in assignment]
    
    # Fallback
    return [int(d * (i + 1) * 100) for i, d in enumerate(DIVISORS)]
