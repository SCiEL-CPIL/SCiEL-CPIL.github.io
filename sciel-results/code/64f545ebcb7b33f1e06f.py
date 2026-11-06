import time
import random
import numpy as np
from math import exp

MAX_N = 20
MAX_VAL = 10000
TIME_LIMIT = 85  # Safety margin (90s limit)

def _objective(A):
    """Compute max(|A+A|, |A*A|) for a set A of distinct positive integers using NumPy."""
    A_arr = np.array(A, dtype=np.int64)
    n = len(A_arr)
    
    # Generate all pairs (i, j) with i <= j
    # There are n*(n+1)/2 such pairs
    indices = np.triu_indices(n)
    
    # Vectorized sum and product calculation
    sums = A_arr[indices[0]] + A_arr[indices[1]]
    products = A_arr[indices[0]] * A_arr[indices[1]]
    
    # Count unique values
    # np.unique is efficient for sorting and finding unique elements
    unique_sums = len(np.unique(sums))
    unique_products = len(np.unique(products))
    
    return max(unique_sums, unique_products)

def _is_valid(A):
    """Check if A has 20 distinct positive integers <= 10000."""
    if len(A) != 20:
        return False
    if len(set(A)) != 20:
        return False
    for x in A:
        if not isinstance(x, int) or x < 1 or x > MAX_VAL:
            return False
    return True

def _generate_rational_gp(start, num, den, n):
    """
    Generate n distinct integers from a rational geometric progression:
    a_k = start * (num/den)^k
    This helps create multiplicative collisions.
    """
    A = []
    current = float(start)
    ratio = float(num) / float(den)
    
    for _ in range(n):
        val = int(round(current))
        if val < 1:
            val = 1
        if val > MAX_VAL:
            break
        if val not in A:
            A.append(val)
        current *= ratio
        
    # If we didn't get enough, fill with small increments
    if len(A) < n:
        last = A[-1] if A else start
        while len(A) < n:
            last += 1
            if last > MAX_VAL: break
            if last not in A:
                A.append(last)
            
    return A[:n]

def _local_search(A, time_limit, rng):
    """Local search with simulated annealing to minimize the objective."""
    A = sorted(A)
    current_obj = _objective(A)
    best_A = list(A)
    best_obj = current_obj
    
    start_time = time.time()
    temperature = 50.0
    cooling_rate = 0.995
    
    while temperature > 0.01:
        if time.time() - start_time > time_limit:
            break
        
        # Pick a random element to modify
        idx = rng.randint(0, 19)
        old_val = A[idx]
        
        # Try a few random changes
        accepted = False
        for _ in range(15):
            if time.time() - start_time > time_limit:
                break
            
            # Random delta
            # Scale of change depends on the value.
            # Small values can change by small amounts. Large values by larger.
            # But we are bounded by 10000.
            # Let's use a mix.
            if rng.random() < 0.5:
                # Small jump
                delta = rng.randint(-5, 5)
            else:
                # Larger jump
                delta = rng.randint(-50, 50)
            
            new_val = old_val + delta
            
            if new_val < 1 or new_val > MAX_VAL:
                continue
            if new_val in A:
                continue
            
            # Test the change
            A[idx] = new_val
            new_obj = _objective(A)
            A[idx] = old_val
            
            delta_obj = new_obj - current_obj
            # Accept if better or with probability exp(-delta/T)
            if delta_obj <= 0 or rng.random() < exp(-delta_obj / (temperature + 1e-10)):
                A[idx] = new_val
                current_obj = new_obj
                if new_obj < best_obj:
                    best_obj = new_obj
                    best_A = list(A)
                accepted = True
                break
        
        temperature *= cooling_rate
    
    return best_A, best_obj

def construct():
    start_time = time.time()
    time_budget = TIME_LIMIT - 5  # Leave margin for safety
    
    rng = random.Random(42)
    best_A = None
    best_obj = float('inf')
    
    starts = []
    
    # Start 1: Arithmetic Progression
    starts.append(list(range(1, 21)))
    
    # Start 2: Powers of 2 {1, 2, 4, ..., 2^13} + tail
    # 2^13 = 8192. Next would be 16384 > 10000.
    # So we have 14 elements. Need 6 more.
    # Add 8193..8198?
    s2 = [1 << i for i in range(14)]
    s2.extend([8193, 8194, 8195, 8196, 8197, 8198])
    starts.append(s2)
    
    # Start 3: Rational GP (3/2)
    s3 = _generate_rational_gp(1, 3, 2, 20)
    starts.append(s3)
    
    # Start 4: Rational GP (5/4)
    s4 = _generate_rational_gp(1, 5, 4, 20)
    starts.append(s4)
    
    # Start 5: Rational GP (4/3)
    s5 = _generate_rational_gp(1, 4, 3, 20)
    starts.append(s5)
    
    # Start 6: Mixed (small AP + large GP)
    starts.append(list(range(1, 11)) + [100 * (i+1) for i in range(10)])
    
    # Start 7: Random
    starts.append(rng.sample(range(1, MAX_VAL + 1), 20))
    
    # Start 8: Rational GP (7/5)
    s8 = _generate_rational_gp(1, 7, 5, 20)
    starts.append(s8)
    
    # Start 9: Rational GP (2/1) - same as powers of 2 but generated differently?
    # Let's try a different start for GP
    s9 = _generate_rational_gp(2, 2, 1, 20)
    starts.append(s9)
    
    # Filter valid starts
    valid_starts = []
    for s in starts:
        if _is_valid(s):
            valid_starts.append(s)
        else:
            # Try to fix
            s = sorted(set(s))
            if len(s) < 20:
                # Add random distinct numbers
                available = [x for x in range(1, MAX_VAL+1) if x not in s]
                s += rng.sample(available, 20 - len(s))
            elif len(s) > 20:
                s = s[:20]
            if _is_valid(s):
                valid_starts.append(s)
    
    if not valid_starts:
        # Fallback
        valid_starts = [list(range(1, 21))]
    
    # Run local search on each start
    # Allocate time evenly, but prioritize structural starts?
    # Let's just split time.
    num_starts = len(valid_starts)
    time_per_start = time_budget / num_starts
    
    for i, A in enumerate(valid_starts):
        remaining_time = time_budget - (time.time() - start_time)
        if remaining_time < 0.5:
            break
        
        # Allocate slightly less than remaining to ensure we finish
        alloc = min(time_per_start, remaining_time)
        
        A_opt, obj_opt = _local_search(A, alloc, rng)
        
        if obj_opt < best_obj:
            best_obj = obj_opt
            best_A = A_opt
    
    # Final refinement on the best found
    remaining = time_budget - (time.time() - start_time)
    if remaining > 0.5 and best_A:
        A_ref, obj_ref = _local_search(best_A, remaining, rng)
        if obj_ref < best_obj:
            best_A = A_ref
            best_obj = obj_ref
    
    if not best_A:
        best_A = list(range(1, 21))
        
    return sorted(best_A)
