import time
import numpy as np
from collections import defaultdict

MAX_VAL = 10_000_000
TIME_LIMIT = 85  # seconds, safety margin before 90s kill

def _sigma_sieve(n):
    """Compute sigma(k) for all k in [1, n] using a sieve-like approach."""
    sigma = np.zeros(n + 1, dtype=np.int64)
    for d in range(1, n + 1):
        sigma[d::d] += d
    return sigma

def construct():
    start_time = time.time()
    
    # Compute sigma for all n up to MAX_VAL
    sigma_arr = _sigma_sieve(MAX_VAL)
    
    # Build fibers
    fiber_map = defaultdict(list)
    for n in range(1, MAX_VAL + 1):
        s = int(sigma_arr[n])
        fiber_map[s].append(n)
        
    # Get max value for each fiber
    fiber_maxes = {s: max(vals) for s, vals in fiber_map.items()}
    
    # Sort fibers by max value descending
    sorted_fibers = sorted(fiber_map.items(), key=lambda x: fiber_maxes[x[0]], reverse=True)
    
    best_num = None
    best_den = None
    best_pair = (11, 6) # Fallback: sigma(11)=12, sigma(6)=12, error=|22-18|/12 = 4/12 = 1/3
    
    for s_val, vals in sorted_fibers:
        if time.time() - start_time > TIME_LIMIT:
            break
            
        max_val_in_fiber = fiber_maxes[s_val]
        
        # Check if this fiber can possibly improve the best error
        # Minimum possible error in this fiber is 1 / (2 * max_val_in_fiber)
        if best_num is not None:
            # If 1 / (2 * max_val_in_fiber) >= best_num / best_den, we can stop
            # 1 * best_den >= best_num * 2 * max_val_in_fiber
            if best_den >= best_num * 2 * max_val_in_fiber:
                break
        
        # Search for best pair in this fiber
        # vals is sorted
        n_vals = len(vals)
        if n_vals < 2:
            continue
            
        # For each b, find a closest to 1.5 * b
        # We can use binary search for efficiency
        # Or two pointers. Let's use binary search for simplicity and robustness.
        
        # To speed up, we can skip if the fiber is too small to have a good pair?
        # No, even small fibers can have k=1.
        
        # We'll iterate through b and binary search for a.
        # To avoid O(N log N) on very large fibers, we might need to be careful.
        # But we expect to process few fibers.
        
        for b in vals:
            target = 1.5 * b
            # Binary search for target in vals
            # np.searchsorted
            idx = np.searchsorted(vals, target)
            
            # Check idx and idx-1
            candidates = []
            if idx < n_vals:
                candidates.append(vals[idx])
            if idx > 0:
                candidates.append(vals[idx-1])
                
            for a in candidates:
                if a == b:
                    continue
                if 2 * a == 3 * b:
                    continue # Error 0, invalid
                
                error_num = abs(2 * a - 3 * b)
                error_den = 2 * b
                
                if best_num is None:
                    best_num = error_num
                    best_den = error_den
                    best_pair = (a, b)
                else:
                    # Compare error_num/error_den < best_num/best_den
                    if error_num * best_den < best_num * error_den:
                        best_num = error_num
                        best_den = error_den
                        best_pair = (a, b)
                        
        # Optional: Early termination if we found a very good pair
        # If best_num == 1, we found the smallest possible numerator.
        # We can only improve by increasing the denominator (i.e., larger b).
        # Since we are processing fibers in descending order of max value,
        # any subsequent fiber will have a smaller max value, so its denominator will be smaller.
        # Thus, if we found best_num == 1, we can break immediately?
        # Wait. The denominator is 2*b. b is from the current fiber.
        # If we found a pair with num=1 in a fiber with max M, the error is 1/(2b).
        # The next fiber has max M' < M. Any pair there will have b' <= M' < M.
        # So the error will be 1/(2b') > 1/(2M) >= 1/(2b) (since b <= M).
        # Actually, b could be smaller than M. But the *best possible* error in the next fiber is 1/(2M').
        # Since M' < M, 1/(2M') > 1/(2M).
        # If our current error is 1/(2b_current), and b_current is close to M, it's likely better than 1/(2M').
        # However, it's possible that in the current fiber, the best pair had a small b,
        # while in a previous fiber (larger M), we missed a pair with large b?
        # No, we process in descending M. So we've already checked all fibers with larger M.
        # So if we have best_num == 1, we have the best possible numerator.
        # We just need the largest possible denominator among all pairs with num=1.
        # Since we process fibers in descending M, the first fiber we encounter with a num=1 pair
        # will have the largest M. But does it have the largest b for that pair?
        # We found the best pair in that fiber.
        # Could a later fiber (smaller M) have a pair with num=1 and larger b?
        # No, because b in later fiber <= M_later < M_current.
        # And b in current fiber <= M_current.
        # So the maximum possible b for any num=1 pair is in the fiber with the largest M that contains such a pair.
        # Since we process in descending M, the first such fiber we find will contain the global best pair with num=1.
        # Therefore, if we find best_num == 1, we can break.
        
        if best_num == 1:
            break

    a, b = best_pair
    return [int(a), int(b)]
