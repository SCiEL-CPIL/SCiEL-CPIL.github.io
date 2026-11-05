import time
import math
import random
import numpy as np

def _full_calc_disc(signs):
    """Calculate the exact maximum absolute signed sum on any AP."""
    n = len(signs)
    signs_arr = np.asarray(signs, dtype=np.int8)
    max_abs = 0
    
    for d in range(1, n + 1):
        for r in range(min(d, n)):
            sub = signs_arr[r::d]
            if len(sub) == 0:
                continue
            cum = np.cumsum(sub)
            curr_max = np.max(np.abs(cum))
            if curr_max > max_abs:
                max_abs = curr_max
                # Early exit optimization: if we hit a theoretical max?
                # Max possible sum for step d is len(sub).
                # If max_abs is already very high, we might still need to check others?
                # No, we need the global max.
    return int(max_abs)

def _fast_calc_disc(signs, bound):
    """
    Quickly check if the max AP sum is < bound.
    Returns the max sum found among APs with length >= bound.
    If any such AP has sum >= bound, returns that sum (or >= bound).
    If all such APs have sum < bound, returns the max of those sums.
    Note: APs with length < bound are ignored.
    """
    n = len(signs)
    if bound <= 0:
        return 0
    
    signs_arr = np.asarray(signs, dtype=np.int8)
    max_d = n // bound
    if max_d > n:
        max_d = n
        
    max_val = 0
    
    for d in range(1, max_d + 1):
        for r in range(min(d, n)):
            sub = signs_arr[r::d]
            # Only check if length is sufficient?
            # The condition for checking was L_d >= bound.
            # L_d is the max length for this start/step.
            # len(sub) is the actual length.
            # If len(sub) < bound, the max sum is <= len(sub) < bound.
            # So we can skip if len(sub) < bound?
            # Yes, because we are looking for values >= bound to reject, or max < bound to accept.
            # If len(sub) < bound, it cannot contribute to a value >= bound.
            # And it contributes at most len(sub) to the max.
            # We are tracking max_val.
            if len(sub) < bound:
                # It can't exceed bound, but it might be the new max if current max_val is small?
                # No, we are comparing against 'bound'.
                # If we find a value < bound, we update max_val.
                # But if len(sub) < bound, max sum <= len(sub).
                # We can just update max_val with len(sub) if we want tight bound?
                # Actually, we just need to know if ANY AP has sum >= bound.
                # If we find one, we return it.
                # If we don't find any, we return the max of the checked ones.
                # But the unchecked ones (len < bound) have max <= len < bound.
                # So the global max is max(max_checked, max_unchecked).
                # max_unchecked <= bound - 1.
                # So if max_checked < bound, global max <= bound - 1.
                # We don't need the exact value of unchecked ones if we just want to know if it's < bound.
                pass
            else:
                cum = np.cumsum(sub)
                curr_max = np.max(np.abs(cum))
                if curr_max > max_val:
                    max_val = curr_max
                    if max_val >= bound:
                        return int(max_val)
                        
    return int(max_val)

def _thue_morse(n):
    """Generate Thue-Morse sequence of length n."""
    # t_i = popcount(i) % 2
    # Sign: 1 if even, -1 if odd
    return [1 if bin(i).count('1') % 2 == 0 else -1 for i in range(n)]

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 5s safety margin
    
    n = 256
    
    # Initial candidates
    candidates = []
    
    # 1. Thue-Morse
    candidates.append(_thue_morse(n))
    
    # 2. Random
    rng = random.Random(42)
    for _ in range(5):
        candidates.append([rng.choice((-1, 1)) for _ in range(n)])
    
    # 3. Alternating
    candidates.append([1 if i % 2 == 0 else -1 for i in range(n)])
    
    # Evaluate initial candidates
    best_signs = candidates[0][:]
    best_disc = _full_calc_disc(best_signs)
    
    for c in candidates[1:]:
        if time.time() > deadline - 10:
            break
        d = _full_calc_disc(c)
        if d < best_disc:
            best_disc = d
            best_signs = c[:]
            
    signs = best_signs[:]
    current_disc = best_disc
    
    # Local Search
    # Strategy: Flip random indices. Use fast_calc to filter.
    # If fast_calc suggests improvement (val < current_disc), do full calc and accept.
    # If fast_calc suggests no improvement (val >= current_disc), revert.
    # Occasionally do full calc to ensure we aren't stuck in a "false" local minimum
    # where fast_calc misses a decrease?
    # Actually, if fast_calc returns U >= current_disc, the true max D_new <= max(U, current_disc-1).
    # If U == current_disc, D_new could be < current_disc.
    # So we should do full calc if U == current_disc.
    
    steps = 0
    no_improve_count = 0
    max_no_improve = 500 # Check for stagnation
    
    while time.time() < deadline - 2:
        idx = rng.randint(0, n - 1)
        old_val = signs[idx]
        signs[idx] = -old_val
        
        # Fast check
        # We want to know if new_disc < current_disc
        # fast_calc returns max sum for APs with L >= current_disc
        # If this max < current_disc, then new_disc < current_disc
        # If this max >= current_disc, new_disc could be < or >= current_disc
        # But if max >= current_disc, it's likely new_disc >= current_disc
        
        fast_val = _fast_calc_disc(signs, current_disc)
        
        if fast_val < current_disc:
            # Definite improvement
            new_disc = _full_calc_disc(signs)
            if new_disc < current_disc:
                current_disc = new_disc
                best_signs = signs[:]
                best_disc = current_disc
                no_improve_count = 0
            else:
                # Should not happen if fast_val < current_disc
                signs[idx] = old_val
        elif fast_val == current_disc:
            # Possible improvement
            new_disc = _full_calc_disc(signs)
            if new_disc < current_disc:
                current_disc = new_disc
                best_signs = signs[:]
                best_disc = current_disc
                no_improve_count = 0
            else:
                signs[idx] = old_val
                no_improve_count += 1
        else:
            # fast_val > current_disc
            # Likely worse, but could be equal?
            # If fast_val > current_disc, then some AP with L >= current_disc has sum > current_disc.
            # So new_disc > current_disc.
            # Revert.
            signs[idx] = old_val
            no_improve_count += 1
            
        steps += 1
        
        # If stuck, try a random restart or perturbation
        if no_improve_count > max_no_improve:
            # Perturb current best
            signs = best_signs[:]
            for _ in range(10):
                p_idx = rng.randint(0, n - 1)
                signs[p_idx] = -signs[p_idx]
            current_disc = _full_calc_disc(signs)
            no_improve_count = 0
            max_no_improve = 1000 # Be more patient next time?
            
    return best_signs
