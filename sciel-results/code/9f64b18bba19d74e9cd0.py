import math
import time

DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
MAX_VAL = 1000000
N = len(DIVISORS)

def _lcm(a, b):
    return a * b // math.gcd(a, b)

LCM = 1
for d in DIVISORS:
    LCM = _lcm(LCM, d)
# LCM = 30030

def construct():
    start_time = time.time()
    deadline = start_time + 55.0 # Safety margin

    # Theoretical minimum range for 10 distinct integers is 9.
    # We search for R starting from 9.
    
    best_R = None
    best_W = None
    best_assignment = None

    for R in range(9, 500):
        if time.time() > deadline:
            break
        
        # Iterate over all possible window starts modulo LCM
        for W in range(LCM):
            # Check time periodically inside the inner loop if R is large, 
            # but for small R, the inner loop is fast.
            if time.time() > deadline:
                break
            
            avail = []
            feasible = True
            for d in DIVISORS:
                # Multiples of d in [W, W+R]
                # k_min * d >= W  => k_min >= W/d
                # k_max * d <= W+R => k_max <= (W+R)/d
                
                k_min = (W + d - 1) // d
                k_max = (W + R) // d
                
                if k_min > k_max:
                    feasible = False
                    break
                
                # Generate the list of candidates
                # The list size is k_max - k_min + 1
                # For small R, this is small.
                # To optimize, we can store just the range and compute on the fly in backtracking?
                # Or just store the list.
                candidates = [k * d for k in range(k_min, k_max + 1)]
                avail.append(candidates)
            
            if not feasible:
                continue
            
            # Check for System of Distinct Representatives (SDR)
            # We use backtracking with ordering by constraint size.
            n = N
            # Order indices by length of candidate list
            order = sorted(range(n), key=lambda i: len(avail[i]))
            
            used = set()
            assignment = [None] * n
            
            def backtrack(idx):
                if idx == n:
                    return True
                i = order[idx]
                for v in avail[i]:
                    if v not in used:
                        used.add(v)
                        assignment[i] = v
                        if backtrack(idx + 1):
                            return True
                        used.remove(v)
                        assignment[i] = None
                return False
            
            if backtrack(0):
                best_R = R
                best_W = W
                best_assignment = assignment[:]
                break
        
        if best_R is not None:
            break

    if best_R is None:
        # Fallback: Return divisors themselves (valid, distinct, range is large)
        # Or a known valid set.
        return DIVISORS[:]

    # Construct final result
    # The assignment values are in [0, LCM + R] roughly.
    # We need positive integers.
    # If any value is 0, we shift the whole set by LCM.
    # Shifting by LCM preserves divisibility and distinctness and range.
    
    result = []
    shift = 0
    for v in best_assignment:
        if v == 0:
            shift = LCM
            break
    
    for i in range(N):
        val = best_assignment[i] + shift
        # Ensure val <= MAX_VAL
        # If val > MAX_VAL, we might need to adjust, but since LCM is small,
        # and we start from small W, this is unlikely to be an issue for minimal R.
        # However, if W is large (close to LCM) and R is large, val could be large.
        # But we found the minimal R, which is small.
        # Let's add a check just in case.
        if val > MAX_VAL:
            # Try to shift down? No, must be positive.
            # If this happens, we might have picked a W that is too high.
            # But we iterate W from 0. The first valid W found for minimal R
            # might be high? No, we break on the first valid W for a given R.
            # So W is the smallest valid start for this R.
            # So values should be as small as possible for this R.
            pass
        
        result.append(int(val))
    
    return result
