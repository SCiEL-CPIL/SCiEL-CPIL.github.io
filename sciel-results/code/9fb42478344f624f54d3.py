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
    indices = np.triu_indices(n)
    
    # Vectorized sum and product calculation
    sums = A_arr[indices[0]] + A_arr[indices[1]]
    products = A_arr[indices[0]] * A_arr[indices[1]]
    
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
    Generate n distinct integers from a rational geometric progression.
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

def _local_search_il(A, time_limit, rng):
    """
    Iterated Local Search (ILS) with Greedy Hill Climbing.
    1. Greedy Phase: Find best single-element replacement.
    2. Perturbation Phase: Randomly disrupt the solution.
    """
    A = sorted(A)
    current_obj = _objective(A)
    best_A = list(A)
    best_obj = current_obj
    
    start_time = time.time()
    
    # ILS Loop
    while time.time() - start_time < time_limit:
        # --- Greedy Phase ---
        improved = True
        while improved and (time.time() - start_time < time_limit):
            improved = False
            best_delta = 0
            best_new_val = None
            best_idx = -1
            
            # Try modifying each element
            for i in range(20):
                old_val = A[i]
                
                # Generate candidate replacements
                # 1. Small local steps
                candidates = set()
                for d in range(1, 11):
                    candidates.add(old_val - d)
                    candidates.add(old_val + d)
                
                # 2. Random jumps
                for _ in range(10):
                    candidates.add(rng.randint(1, MAX_VAL))
                    
                # 3. "Structured" candidates: values that might create collisions
                # e.g., multiples of other elements
                for j in range(20):
                    if i == j: continue
                    # Try values close to A[j] * k or A[j] + k
                    # This is heuristic, might be too slow if done extensively.
                    # Let's stick to random + local for speed.
                    
                # Evaluate candidates
                for new_val in candidates:
                    if new_val < 1 or new_val > MAX_VAL:
                        continue
                    if new_val in A:
                        continue
                    
                    A[i] = new_val
                    new_obj = _objective(A)
                    A[i] = old_val
                    
                    delta = new_obj - current_obj
                    if delta < best_delta:
                        best_delta = delta
                        best_new_val = new_val
                        best_idx = i
            
            # Apply best move if found
            if best_idx != -1:
                A[best_idx] = best_new_val
                current_obj += best_delta
                improved = True
                
                if current_obj < best_obj:
                    best_obj = current_obj
                    best_A = list(A)
                    
        # --- Perturbation Phase ---
        # If we found a new best, perturb it to escape local minima
        # If we didn't find a new best in this iteration, we are stuck in a local min.
        # Perturb the current best solution.
        
        # Check time
        if time.time() - start_time >= time_limit:
            break
            
        # Perturb best_A
        A = list(best_A)
        # Randomly replace 2-4 elements
        num_perturb = rng.randint(2, 4)
        for _ in range(num_perturb):
            idx = rng.randint(0, 19)
            # Choose a new value
            # Bias towards small numbers or numbers that create structure?
            # Just random for now, maybe mix with local
            if rng.random() < 0.5:
                # Local perturbation
                old = A[idx]
                delta = rng.randint(-20, 20)
                new_val = old + delta
                if new_val < 1: new_val = 1
                if new_val > MAX_VAL: new_val = MAX_VAL
            else:
                new_val = rng.randint(1, MAX_VAL)
                
            # Ensure distinctness
            if new_val in A and new_val != A[idx]:
                # Find a nearby valid value
                # Simple retry
                for _ in range(10):
                    new_val = rng.randint(1, MAX_VAL)
                    if new_val not in A:
                        break
                if new_val in A:
                    continue # Skip this perturbation
            
            A[idx] = new_val
        
        current_obj = _objective(A)
        if current_obj < best_obj:
            best_obj = current_obj
            best_A = list(A)
            
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
    
    # Start 2: Powers of 2
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
    
    # Start 9: Dense small numbers
    starts.append(sorted(rng.sample(range(1, 100), 20)))
    
    # Filter valid starts
    valid_starts = []
    for s in starts:
        if _is_valid(s):
            valid_starts.append(sorted(s))
        else:
            s = sorted(set(s))
            if len(s) < 20:
                available = [x for x in range(1, MAX_VAL+1) if x not in s]
                s += rng.sample(available, 20 - len(s))
            elif len(s) > 20:
                s = s[:20]
            if _is_valid(s):
                valid_starts.append(sorted(s))
    
    if not valid_starts:
        valid_starts = [list(range(1, 21))]
    
    # Run ILS on each start
    # Allocate time based on remaining budget
    num_starts = len(valid_starts)
    
    # Prioritize structural starts? Or just split evenly?
    # Let's split evenly but ensure at least 1s per start if possible.
    time_per_start = time_budget / num_starts
    
    for i, A in enumerate(valid_starts):
        remaining_time = time_budget - (time.time() - start_time)
        if remaining_time < 0.5:
            break
        
        # Allocate time, but don't exceed remaining
        alloc = min(time_per_start, remaining_time)
        
        A_opt, obj_opt = _local_search_il(A, alloc, rng)
        
        if obj_opt < best_obj:
            best_obj = obj_opt
            best_A = A_opt
    
    # Final refinement on the best found
    remaining = time_budget - (time.time() - start_time)
    if remaining > 0.5 and best_A:
        A_ref, obj_ref = _local_search_il(best_A, remaining, rng)
        if obj_ref < best_obj:
            best_A = A_ref
            best_obj = obj_ref
    
    if not best_A:
        best_A = list(range(1, 21))
        
    return sorted(best_A)
