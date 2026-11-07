import math
import time

# EVOLVE-BLOCK-START
DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
MAX_VAL = 1000000
N = len(DIVISORS)

def _lcm(a, b):
    return a * b // math.gcd(a, b)

# Calculate LCM of all divisors to limit the search period
LCM_VAL = 1
for d in DIVISORS:
    LCM_VAL = _lcm(LCM_VAL, d)

def _find_sdr(candidate_sets):
    """
    Find a System of Distinct Representatives from a list of sets.
    Uses backtracking.
    Returns a list of chosen values or None.
    """
    n = len(candidate_sets)
    # Sort by size of set to prune early (heuristic)
    indices = sorted(range(n), key=lambda i: len(candidate_sets[i]))
    ordered_sets = [candidate_sets[i] for i in indices]
    
    chosen = [None] * n
    used = set()
    
    def backtrack(idx):
        if idx == n:
            return True
        for val in ordered_sets[idx]:
            if val not in used:
                used.add(val)
                chosen[idx] = val
                if backtrack(idx + 1):
                    return True
                used.remove(val)
                chosen[idx] = None
        return False
    
    if backtrack(0):
        # Map back to original order
        result = [None] * n
        for i, orig_idx in enumerate(indices):
            result[orig_idx] = chosen[i]
        return result
    return None

def _check_width(W):
    """
    Check if there exists a window of width W containing distinct multiples.
    Returns the list of representatives if found, else None.
    """
    # We only need to check L in [0, LCM_VAL)
    for L in range(LCM_VAL):
        candidates = []
        valid = True
        for d in DIVISORS:
            # Find multiples of d in [L, L+W]
            # First multiple >= L
            start_k = (L + d - 1) // d
            start_val = start_k * d
            end_val = L + W
            
            vals = []
            if start_val <= end_val:
                # Add multiples
                # Usually only 1 or 2 values for small W
                vals.append(start_val)
                next_val = start_val + d
                if next_val <= end_val:
                    vals.append(next_val)
                    # If W is large, there could be more, but we expect small W
                    # To be safe, loop
                    while next_val + d <= end_val:
                        next_val += d
                        vals.append(next_val)
            
            if not vals:
                valid = False
                break
            candidates.append(vals)
        
        if not valid:
            continue
            
        sdr = _find_sdr(candidates)
        if sdr:
            return sdr
            
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    # Binary search for the minimum width
    lo = 0
    hi = 1000  # Upper bound guess
    best_reps = None
    best_W = None
    
    # First, ensure hi is valid
    reps_hi = _check_width(hi)
    if reps_hi is None:
        # If 1000 is not enough, increase it, though unlikely
        hi = 10000
        reps_hi = _check_width(hi)
        if reps_hi is None:
            # Fallback to a simple greedy if something goes wrong
            # Just return the divisors themselves (valid but wide)
            return DIVISORS[:]
    
    while lo < hi:
        if time.time() > deadline - 5.0:
            break
        mid = (lo + hi) // 2
        reps_mid = _check_width(mid)
        if reps_mid is not None:
            hi = mid
            best_reps = reps_mid
            best_W = mid
        else:
            lo = mid + 1
            
    if best_reps is None:
        best_reps = reps_hi
        best_W = hi
        
    return best_reps
# EVOLVE-BLOCK-END
