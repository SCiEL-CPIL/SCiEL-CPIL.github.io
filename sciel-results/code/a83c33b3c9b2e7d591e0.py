import numpy as np
import time
import random
import math

def _objective_np(A):
    """Fast objective using NumPy vectorization: max(|A+A|, |A*A|)."""
    A = np.array(A, dtype=np.int64)
    sums = A[:, None] + A[None, :]
    prods = A[:, None] * A[None, :]
    n_sums = len(np.unique(sums))
    n_prods = len(np.unique(prods))
    return max(n_sums, n_prods)

def _is_valid(A):
    if len(set(A)) != 20:
        return False
    if any(x < 1 or x > 10000 for x in A):
        return False
    return True

def _geometric_start(ratio=1.5, n=20, max_val=10000):
    """Generate a geometric-like starting set with integer rounding."""
    A = []
    val = 2.0
    for i in range(n):
        v = int(round(val))
        if v < 1:
            v = 1
        if v > max_val:
            v = max_val
        if v not in A:
            A.append(v)
        val *= ratio
    # Ensure distinct and fill if needed
    A = sorted(set(A))
    while len(A) < n:
        # Add a value that is not in A
        candidates = []
        for x in range(1, min(max_val + 1, int(A[-1] * 1.2) + 1) if A else 100):
            if x not in A:
                candidates.append(x)
        if not candidates:
            candidates = [x for x in range(1, max_val + 1) if x not in A]
        if not candidates:
            break
        A.append(random.choice(candidates))
    A = sorted(set(A))[:n]
    while len(A) < n:
        A.append(max(A) + 1)
    return A[:n]

def _local_search_np(A, time_limit=55):
    """
    Local search with NumPy objective.
    Uses small perturbations and occasional random jumps.
    """
    start_time = time.time()
    best_A = sorted(A[:])
    best_obj = _objective_np(best_A)
    current_A = best_A[:]
    current_obj = best_obj
    
    n = len(current_A)
    max_val = 10000
    
    T = 15.0
    cooling = 0.998
    min_T = 0.5
    
    # Precompute some useful info
    while (time.time() - start_time) < time_limit and T > min_T:
        # Pick a random index
        i = random.randint(0, n - 1)
        old_val = current_A[i]
        
        # Generate new value
        r = random.random()
        if r < 0.6:
            # Small perturbation: +/- 1 to 10
            delta = random.randint(1, 10) * (1 if random.random() < 0.5 else -1)
            new_val = old_val + delta
        elif r < 0.8:
            # Medium perturbation: +/- 1 to 50
            delta = random.randint(1, 50) * (1 if random.random() < 0.5 else -1)
            new_val = old_val + delta
        else:
            # Large jump: random value
            new_val = random.randint(1, max_val)
        
        # Validate
        if new_val < 1 or new_val > max_val:
            T *= cooling
            continue
        if new_val in current_A:
            T *= cooling
            continue
            
        trial = current_A[:]
        trial[i] = new_val
        if len(set(trial)) != n:
            T *= cooling
            continue
            
        trial_obj = _objective_np(trial)
        delta_obj = trial_obj - current_obj
        
        # Accept if better or with probability exp(-delta/T)
        if delta_obj < 0 or random.random() < math.exp(-delta_obj / T):
            current_A = trial
            current_obj = trial_obj
            if current_obj < best_obj:
                best_obj = current_obj
                best_A = current_A[:]
        
        T *= cooling
    
    return best_A, best_obj

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    best_A = None
    best_obj = float('inf')
    
    # Generate diverse starting points
    starts = []
    
    # Geometric starts with various ratios
    for ratio in [1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0, 2.5]:
        A = _geometric_start(ratio=ratio, n=20, max_val=10000)
        if _is_valid(A):
            starts.append(A)
    
    # Small consecutive integers
    starts.append(list(range(1, 21)))
    starts.append(list(range(2, 22)))
    starts.append(list(range(1, 16)) + list(range(20, 26)))
    
    # Random starts
    for _ in range(5):
        A = random.sample(range(1, 10001), 20)
        starts.append(sorted(A))
    
    # Power-of-2 based
    starts.append([2**i for i in range(20)])
    
    # Mixed: some small, some geometric
    A = list(range(1, 11))
    for i in range(10):
        A.append(100 * (i + 1))
    A = sorted(set(A))[:20]
    if _is_valid(A):
        starts.append(A)
    
    # Evaluate all starts
    for A in starts:
        if (time.time() - start_time) > deadline:
            break
        obj = _objective_np(A)
        if obj < best_obj:
            best_obj = obj
            best_A = A[:]
    
    # Local search from the best start
    if best_A is not None:
        remaining = deadline - (time.time() - start_time)
        if remaining > 5:
            A_ls, obj_ls = _local_search_np(best_A, time_limit=remaining - 2)
            if obj_ls < best_obj:
                best_obj = obj_ls
                best_A = A_ls[:]
    
    # If time remains, try local search from a few other good starts
    remaining = deadline - (time.time() - start_time)
    if remaining > 10:
        # Sort starts by objective value and try the next few
        scored_starts = []
        for A in starts:
            obj = _objective_np(A)
            scored_starts.append((obj, A))
        scored_starts.sort()
        
        for obj_s, A_s in scored_starts[:3]:
            if (time.time() - start_time) > deadline - 3:
                break
            if obj_s >= best_obj:
                continue
            time_left = deadline - (time.time() - start_time) - 2
            if time_left < 5:
                break
            A_ls, obj_ls = _local_search_np(A_s, time_limit=time_left)
            if obj_ls < best_obj:
                best_obj = obj_ls
                best_A = A_ls[:]
    
    if best_A is None:
        best_A = list(range(1, 21))
    
    return sorted(best_A)
