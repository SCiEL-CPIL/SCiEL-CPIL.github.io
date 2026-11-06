import time
import random
import itertools

MAX_N = 20
MAX_VAL = 10000
TIME_LIMIT = 85  # safety margin before 90s kill

def _objective(A):
    """Compute max(|A+A|, |A*A|) for a set A of distinct positive integers."""
    A_sorted = sorted(A)
    n = len(A_sorted)
    
    sums = set()
    for i in range(n):
        for j in range(i, n):
            sums.add(A_sorted[i] + A_sorted[j])
    
    products = set()
    for i in range(n):
        for j in range(i, n):
            products.add(A_sorted[i] * A_sorted[j])
    
    return max(len(sums), len(products))

def _is_valid(A):
    if len(A) != 20:
        return False
    if len(set(A)) != 20:
        return False
    for x in A:
        if not isinstance(x, int) or x < 1 or x > 10000:
            return False
    return True

def _smooth_numbers(limit, count):
    """Generate the 'count' smallest smooth numbers <= limit."""
    # Generate all numbers of form 2^i * 3^j * 5^k <= limit
    smooths = set()
    i_max = int(limit.bit_length())
    j_max = int(limit.bit_length())
    k_max = int(limit.bit_length())
    
    for i in range(i_max + 1):
        pow2 = 1 << i
        if pow2 > limit: break
        for j in range(j_max + 1):
            pow3 = pow2 * (3 ** j)
            if pow3 > limit: break
            for k in range(k_max + 1):
                val = pow3 * (5 ** k)
                if val > limit: break
                smooths.add(val)
    
    sorted_smooths = sorted(smooths)
    return sorted_smooths[:count]

def _construct_candidates():
    """Generate a list of candidate starting sets."""
    candidates = []
    
    # 1. Smooth numbers (powers of 2, 3, 5)
    try:
        smooths = _smooth_numbers(10000, 20)
        if len(smooths) == 20:
            candidates.append(smooths)
    except:
        pass
        
    # 2. Geometric Progressions
    for ratio in [1.2, 1.3, 1.5, 1.8, 2.0]:
        for start in [1, 2, 3]:
            A = []
            current = float(start)
            for _ in range(20):
                val = int(round(current))
                if val in A:
                    val += 1
                    while val in A:
                        val += 1
                if val > 10000:
                    break
                A.append(val)
                current *= ratio
            if len(A) == 20 and _is_valid(A):
                candidates.append(A)
                
    # 3. Arithmetic Progressions
    for start in [1, 2, 5]:
        for step in [1, 2, 3]:
            A = [start + i * step for i in range(20)]
            if max(A) <= 10000 and _is_valid(A):
                candidates.append(A)
                
    # 4. Random
    candidates.append(random.sample(range(1, 10001), 20))
    candidates.append(random.sample(range(1, 1001), 20))
    
    return candidates

def _local_search(A, time_limit, rng):
    """Improved Local Search with Simulated Annealing and Polish."""
    A = sorted(A)
    current_obj = _objective(A)
    best_A = list(A)
    best_obj = current_obj
    
    start_time = time.time()
    temperature = 100.0
    cooling_rate = 0.999
    min_temp = 0.1
    
    # SA Phase
    while temperature > min_temp:
        if time.time() - start_time > time_limit:
            break
            
        # Choose move type
        r = rng.random()
        if r < 0.7:
            move_type = 'single'
        else:
            move_type = 'double'

        if move_type == 'single':
            idx = rng.randint(0, 19)
            old_val = A[idx]
            
            # Generate candidate
            if rng.random() < 0.5:
                # Local
                step = rng.randint(1, 20)
                new_val = old_val + rng.choice([-1, 1]) * step
            else:
                # Global
                new_val = rng.randint(1, 10000)
                
            if 1 <= new_val <= 10000 and new_val not in A:
                A[idx] = new_val
                new_obj = _objective(A)
                
                delta = new_obj - current_obj
                if delta <= 0 or rng.random() < min(1.0, max(0.0, -delta / (temperature + 1e-10))):
                    current_obj = new_obj
                    if new_obj < best_obj:
                        best_obj = new_obj
                        best_A = list(A)
                else:
                    A[idx] = old_val
        else:
            # Double move
            idx1 = rng.randint(0, 19)
            idx2 = rng.randint(0, 19)
            if idx1 == idx2:
                idx2 = (idx2 + 1) % 20
                
            old_val1 = A[idx1]
            old_val2 = A[idx2]
            
            # Generate candidates
            step1 = rng.randint(1, 20)
            step2 = rng.randint(1, 20)
            new_val1 = old_val1 + rng.choice([-1, 1]) * step1
            new_val2 = old_val2 + rng.choice([-1, 1]) * step2
            
            if 1 <= new_val1 <= 10000 and 1 <= new_val2 <= 10000:
                if new_val1 not in A and new_val2 not in A and new_val1 != new_val2:
                    A[idx1] = new_val1
                    A[idx2] = new_val2
                    new_obj = _objective(A)
                    
                    delta = new_obj - current_obj
                    if delta <= 0 or rng.random() < min(1.0, max(0.0, -delta / (temperature + 1e-10))):
                        current_obj = new_obj
                        if new_obj < best_obj:
                            best_obj = new_obj
                            best_A = list(A)
                    else:
                        A[idx1] = old_val1
                        A[idx2] = old_val2

        temperature *= cooling_rate
        
    # Polish Phase: Strict Hill Climbing
    polish_time_limit = 5.0
    polish_start = time.time()
    improved = True
    while improved and (time.time() - polish_start) < polish_time_limit:
        improved = False
        for i in range(20):
            if (time.time() - polish_start) > polish_time_limit:
                break
            old_val = best_A[i]
            # Try small perturbations
            for step in range(1, 51):
                for direction in [-1, 1]:
                    new_val = old_val + direction * step
                    if new_val < 1 or new_val > 10000:
                        continue
                    if new_val in best_A:
                        continue
                    
                    # Test
                    temp_A = list(best_A)
                    temp_A[i] = new_val
                    new_obj = _objective(temp_A)
                    
                    if new_obj < best_obj:
                        best_obj = new_obj
                        best_A = temp_A
                        improved = True
                        old_val = new_val # Update base for subsequent steps in this iteration?
                        # Break to restart step loop with new base?
                        # Yes, because the neighborhood changed.
                        break
                else:
                    continue
                break
    return best_A, best_obj

def construct():
    start_time = time.time()
    time_budget = TIME_LIMIT - 5
    
    candidates = _construct_candidates()
    
    best_A = None
    best_obj = float('inf')
    
    rng = random.Random(42)
    
    for A in candidates:
        if time.time() - start_time > time_budget:
            break
        if not _is_valid(A):
            continue
        refined_A, obj = _local_search(A, time_budget - (time.time() - start_time), rng)
        if obj < best_obj:
            best_obj = obj
            best_A = refined_A
            
    if best_A is None:
        best_A = list(range(1, 21))
        best_obj = _objective(best_A)
        
    # Final refinement
    remaining = TIME_LIMIT - 5 - (time.time() - start_time)
    if remaining > 1:
        try:
            refined_A, refined_obj = _local_search(best_A, remaining, rng)
            if refined_obj < best_obj:
                best_A = refined_A
        except Exception:
            pass

    if not _is_valid(best_A):
        best_A = list(range(1, 21))
        
    return sorted(best_A)
