import time
import random
import numpy as np

MAX_N = 20
MAX_VAL = 10000
TIME_LIMIT = 85  # Safety margin

def _objective(A):
    """Compute max(|A+A|, |A*A|) for a set A of distinct positive integers using NumPy."""
    A_arr = np.array(A, dtype=np.int64)
    # Generate all pairs (i, j) with i <= j
    indices = np.triu_indices(MAX_N)
    sums = A_arr[indices[0]] + A_arr[indices[1]]
    products = A_arr[indices[0]] * A_arr[indices[1]]
    
    # Count unique values
    # Using np.unique is faster than set() for numpy arrays in this context? 
    # Actually, set() on a list of ints is very fast. 
    # But converting numpy array to set is: set(sums.tolist())
    # Let's try np.unique first, it's C-optimized.
    unique_sums = np.unique(sums)
    unique_products = np.unique(products)
    
    return max(len(unique_sums), len(unique_products))

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

def _local_search(A, time_limit, rng):
    """Local search with simulated annealing to minimize the objective."""
    A = sorted(A)
    current_obj = _objective(A)
    best_A = list(A)
    best_obj = current_obj
    
    start_time = time.time()
    temperature = 100.0
    cooling_rate = 0.999
    
    while temperature > 0.01:
        if time.time() - start_time > time_limit:
            break
        
        # Pick a random element to modify
        idx = rng.randint(0, 19)
        old_val = A[idx]
        
        # Try a few random changes
        # We want to explore the neighborhood.
        # Scale of change depends on the value.
        # Small values can change by small amounts. Large values can change by larger amounts.
        # But we are bounded by 10000.
        
        accepted = False
        # Try multiple candidates for this index before moving on
        for _ in range(10):
            if time.time() - start_time > time_limit:
                break
            
            # Random delta
            # Use a mix of small and large jumps
            if rng.random() < 0.5:
                # Small jump
                delta = rng.randint(-10, 10)
            else:
                # Larger jump
                delta = rng.randint(-100, 100)
            
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
            if delta_obj <= 0 or rng.random() < np.exp(-delta_obj / (temperature + 1e-10)):
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
    
    # Start 1: Arithmetic Progression {1..20}
    # Sumset size: 39. Product set size: likely large.
    starts.append(list(range(1, 21)))
    
    # Start 2: Powers of 2 {1, 2, 4, ..., 2^19}
    # Product set: All powers of 2 from 2^0 to 2^38. Size 39.
    # Sumset: Minkowski sum of GP. Size is larger? 
    # {1,2,4,8} + {1,2,4,8} = {2,3,4,5,6,8,10,12,16} -> 9 elements.
    # For n=20, sumset size is roughly n^2/2? No, for powers of 2, sums are unique?
    # 2^i + 2^j = 2^k + 2^l implies {i,j}={k,l} if i!=j and k!=l.
    # If i=j, 2*2^i = 2^(i+1).
    # So sums are mostly unique.
    # Let's just evaluate it.
    starts.append([1 << i for i in range(20)])
    
    # Start 3: Small AP {1..10} and Large GP {100, 200, ...}
    # Mix of structures.
    starts.append(list(range(1, 11)) + [100 * (i+1) for i in range(10)])
    
    # Start 4: Random
    starts.append(random.sample(range(1, MAX_VAL + 1), 20))
    
    # Start 5: Geometric ratio 3
    starts.append([3**i for i in range(20) if 3**i <= 10000] + [10000, 9999, 9998, 9997, 9996, 9995, 9994, 9993, 9992, 9991])
    # Fix duplicates and count
    s5 = sorted(set(starts[4]))
    if len(s5) > 20: s5 = s5[:20]
    while len(s5) < 20:
        s5.append(s5[-1] + 1)
    starts.append(s5)
    
    # Run local search on each start
    for i, A in enumerate(starts):
        if not _is_valid(A):
            A = sorted(set(A))
            if len(A) < 20:
                A += random.sample([x for x in range(1, MAX_VAL+1) if x not in A], 20 - len(A))
            elif len(A) > 20:
                A = A[:20]
        
        remaining_time = time_budget - (time.time() - start_time)
        if remaining_time < 1:
            continue
        
        # Allocate time
        # Give more time to promising starts? 
        # Just split evenly for simplicity, or give more to first ones.
        alloc = max(0.5, remaining_time / (len(starts) - i))
        
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
