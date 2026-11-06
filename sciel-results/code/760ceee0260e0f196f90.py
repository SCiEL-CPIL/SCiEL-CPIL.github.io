import time
import random
import numpy as np

MAX_N = 20
MAX_VAL = 10000
TIME_LIMIT = 85  # Safety margin for 90s limit

# Pre-compute indices for A+A and A*A (i <= j)
# 20*21/2 = 210 pairs
INDICES = np.triu_indices(MAX_N)

def _objective(A):
    """
    Compute max(|A+A|, |A*A|) for a set A of 20 distinct positive integers.
    A is a list or array of 20 ints.
    """
    A_arr = np.array(A, dtype=np.int64)
    sums = A_arr[INDICES[0]] + A_arr[INDICES[1]]
    products = A_arr[INDICES[0]] * A_arr[INDICES[1]]
    
    # Count unique values
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
    Generate n distinct integers approximating a geometric progression with ratio num/den.
    We want to ensure they are distinct and <= MAX_VAL.
    """
    A = []
    val = float(start)
    ratio = float(num) / float(den)
    
    for _ in range(n):
        val_int = int(round(val))
        if val_int < 1:
            val_int = 1
        if val_int > MAX_VAL:
            break
        if val_int not in A:
            A.append(val_int)
        val *= ratio
        
    # If we didn't get enough, fill with small increments
    if len(A) < n:
        last = A[-1] if A else start
        while len(A) < n:
            last += 1
            if last > MAX_VAL:
                break
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
    temperature = 100.0
    cooling_rate = 0.995
    
    # Precompute a set for O(1) membership checks
    A_set = set(A)
    
    while temperature > 0.1:
        if time.time() - start_time > time_limit:
            break
        
        # Pick a random element to modify
        idx = rng.randint(0, 19)
        old_val = A[idx]
        
        # Try a few random changes
        accepted = False
        for _ in range(30):
            if time.time() - start_time > time_limit:
                break
            
            # Strategy: Mix of small perturbations and larger jumps
            if rng.random() < 0.5:
                # Small perturbation
                scale = max(1, old_val // 10)
                new_val = old_val + rng.randint(-scale, scale)
            else:
                # Larger jump or swap-like move
                # Try replacing with a value from a different range
                if rng.random() < 0.5:
                    new_val = rng.randint(1, MAX_VAL)
                else:
                    # Try to move towards the median or extremes
                    new_val = old_val + rng.randint(-max(1, old_val//2), max(1, old_val//2))
            
            if new_val < 1 or new_val > MAX_VAL:
                continue
            if new_val in A_set:
                continue
            
            # Test the change
            A[idx] = new_val
            A_set.remove(old_val)
            A_set.add(new_val)
            new_obj = _objective(A)
            
            delta = new_obj - current_obj
            if delta <= 0 or rng.random() < min(1.0, max(0.0, -delta / (temperature + 1e-10))):
                # Accept
                current_obj = new_obj
                if new_obj < best_obj:
                    best_obj = new_obj
                    best_A = list(A)
                accepted = True
                break
            else:
                # Reject
                A[idx] = old_val
                A_set.remove(new_val)
                A_set.add(old_val)
        
        temperature *= cooling_rate
    
    return best_A, best_obj

def construct():
    start_time = time.time()
    time_budget = TIME_LIMIT - 5  # Leave margin for safety
    
    rng = random.Random(42)
    best_A = None
    best_obj = float('inf')
    
    starts = []
    
    # Start 1: Arithmetic Progression {1..20}
    starts.append(list(range(1, 21)))
    
    # Start 2: Powers of 2 {1, 2, 4, ..., 2^13} + tail
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
    starts.append(sorted(rng.sample(range(1, MAX_VAL + 1), 20)))
    
    # Start 8: Rational GP (7/5)
    s8 = _generate_rational_gp(1, 7, 5, 20)
    starts.append(s8)
    
    # Start 9: Rational GP (2/1) - same as powers of 2 but generated differently?
    s9 = _generate_rational_gp(2, 2, 1, 20)
    starts.append(s9)
    
    # Start 10: Rational GP (5/3)
    s10 = _generate_rational_gp(1, 5, 3, 20)
    starts.append(s10)
    
    # Start 11: Rational GP (6/5)
    s11 = _generate_rational_gp(1, 6, 5, 20)
    starts.append(s11)
    
    # Start 12: Rational GP (8/7)
    s12 = _generate_rational_gp(1, 8, 7, 20)
    starts.append(s12)
    
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
