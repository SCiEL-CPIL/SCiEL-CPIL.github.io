import time
import sys
import math

# EVOLVE-BLOCK-START

DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
MAX_VAL = 1000000

def construct():
    start_time = time.time()
    deadline = 55.0  # seconds, keep margin
    
    # Calculate LCM of divisors to determine the period of the pattern
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a
        
    lcm_val = 1
    for d in DIVISORS:
        lcm_val = lcm_val * d // gcd(lcm_val, d)
        
    # The pattern of valid starts L repeats every lcm_val.
    # We only need to search L in [1, lcm_val + W_max].
    # If a solution exists for W, it exists in the first period.
    
    # We search for the minimum window width W
    for W in range(1, 100):
        if time.time() - start_time > deadline:
            break
            
        # Pre-compute valid residues for this W
        valid_residues = []
        for d in DIVISORS:
            if W >= d - 1:
                valid_residues.append(None) # All residues valid
            else:
                # r in {0} U [d-W, d-1]
                valid_residues.append((d - W, d - 1))
        
        # Iterate L over one period + W
        # Max L we need to check is lcm_val + W
        # Because if L is valid, L + lcm_val is also valid.
        # And we can shift the window by lcm_val.
        # Since we want to minimize max-min, and values are bounded by 1000000,
        # we can pick the smallest L that works.
        
        for L in range(1, lcm_val + W + 1):
            if time.time() - start_time > deadline:
                # Fallback
                return DIVISORS[:]
                
            # Check if L is a candidate start
            valid = True
            for i, d in enumerate(DIVISORS):
                r = L % d
                vr = valid_residues[i]
                if vr is None:
                    continue
                if r != 0 and not (vr[0] <= r <= vr[1]):
                    valid = False
                    break
            
            if not valid:
                continue
                
            # If valid, assign distinct values
            avail = []
            for d in DIVISORS:
                first_m = L + ((-L) % d)
                ms = []
                m = first_m
                while m <= L + W:
                    ms.append(m)
                    m += d
                avail.append(ms)
                
            # Check if we have enough candidates (at least 1 for each)
            # And if a matching exists
            indices = list(range(len(DIVISORS)))
            indices.sort(key=lambda i: len(avail[i]))
            
            used = set()
            assignment = [None] * len(DIVISORS)
            
            def bt(idx):
                if idx == len(indices):
                    return True
                i = indices[idx]
                for m in avail[i]:
                    if m not in used:
                        used.add(m)
                        assignment[i] = m
                        if bt(idx + 1):
                            return True
                        used.remove(m)
                        assignment[i] = None
                return False
            
            if bt(0):
                # Ensure values are within MAX_VAL
                # Since L <= lcm_val + W and W is small, L+W is small.
                # lcm_val is 30030. So values are < 30100. Well within 1000000.
                return [int(x) for x in assignment]
        
    # If loop finishes without returning, return fallback
    return DIVISORS[:]
# EVOLVE-BLOCK-END
