import time
import random
import math
import numpy as np

def objective(A):
    """Vectorized objective: returns max(|A+A|, |A*A|) using NumPy for speed."""
    arr = np.array(A, dtype=np.int64)
    n = len(arr)
    # Compute all i <= j pairs
    i_idx, j_idx = np.triu_indices(n, k=0)
    ai = arr[i_idx]
    aj = arr[j_idx]
    sums = ai + aj
    prods = ai * aj
    # Count unique values
    n_sums = len(np.unique(sums))
    n_prods = len(np.unique(prods))
    return max(n_sums, n_prods)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin under 90s
    
    # We search for the optimal "shape" in a small integer range [1, SMALL_MAX]
    # and then scale it up.
    # Scale invariance: |kA+kA| = |A+A| and |kA*kA| = |A*A|
    
    SMALL_MAX = 120 
    n = 20
    
    best_obj_global = float('inf')
    best_set_global = None

    def local_search(current_set, max_iters=200, deadline=None):
        """Steepest descent local search to polish a candidate."""
        current_set = list(current_set)
        current_obj = objective(current_set)
        current_set_set = set(current_set)
        
        for _ in range(max_iters):
            if deadline is not None and time.time() > deadline:
                break
            
            best_neighbor = None
            best_neighbor_obj = current_obj
            
            # Try replacing each element with nearby values
            for i in range(n):
                old_val = current_set[i]
                
                # Try nearby values (within ±8) - wider neighborhood for better escape
                candidates = set()
                for delta in range(-8, 9):
                    if delta == 0:
                        continue
                    v = old_val + delta
                    if 1 <= v <= SMALL_MAX:
                        candidates.add(v)
                # Add 2 random candidates
                for _ in range(2):
                    candidates.add(random.randint(1, SMALL_MAX))
                
                for new_val in candidates:
                    if new_val == old_val or new_val in current_set_set:
                        continue
                    neighbor = current_set[:]
                    neighbor[i] = new_val
                    n_obj = objective(neighbor)
                    if n_obj < best_neighbor_obj:
                        best_neighbor_obj = n_obj
                        best_neighbor = neighbor[:]
                        best_neighbor_set = current_set_set.copy()
                        best_neighbor_set.discard(old_val)
                        best_neighbor_set.add(new_val)
            
            if best_neighbor is not None and best_neighbor_obj < current_obj:
                current_set = best_neighbor
                current_set_set = best_neighbor_set
                current_obj = best_neighbor_obj
            else:
                break
                
        return current_set, current_obj

    def simulated_annealing(start_set, max_iters=5000):
        current_set = list(start_set)
        current_obj = objective(current_set)
        current_set_set = set(current_set)
        
        # Initial temperature based on typical objective variation
        T = 12.0
        T_min = 0.1
        cooling_rate = 0.997
        
        best_local_set = current_set[:]
        best_local_obj = current_obj
        
        for i in range(max_iters):
            if time.time() > deadline:
                break
            
            neighbor = current_set[:]
            move_type = random.random()
            
            if move_type < 0.4:
                # Replace one element with nearby value
                idx = random.randint(0, n-1)
                delta = random.randint(-6, 6)
                if delta == 0:
                    delta = random.choice([-1, 1])
                new_val = current_set[idx] + delta
                if 1 <= new_val <= SMALL_MAX and new_val not in current_set_set:
                    neighbor[idx] = new_val
                else:
                    new_val = random.randint(1, SMALL_MAX)
                    if new_val not in current_set_set:
                        neighbor[idx] = new_val
                    else:
                        continue
            elif move_type < 0.7:
                # Replace one element with random value
                idx = random.randint(0, n-1)
                new_val = random.randint(1, SMALL_MAX)
                if new_val not in current_set_set:
                    neighbor[idx] = new_val
                else:
                    continue
            elif move_type < 0.85:
                # Swap two elements with new random values (2-move)
                i1, i2 = random.sample(range(n), 2)
                v1 = random.randint(1, SMALL_MAX)
                v2 = random.randint(1, SMALL_MAX)
                if (v1 not in current_set_set or v1 == current_set[i1]) and \
                   (v2 not in current_set_set or v2 == current_set[i2]) and \
                   v1 != v2:
                    neighbor[i1] = v1
                    neighbor[i2] = v2
                else:
                    # Fallback to single replace
                    idx = random.randint(0, n-1)
                    new_val = random.randint(1, SMALL_MAX)
                    if new_val not in current_set_set:
                        neighbor[idx] = new_val
                    else:
                        continue
            else:
                # Large perturbation: replace 2-3 random elements
                num_replace = random.randint(2, 3)
                indices = random.sample(range(n), num_replace)
                valid = True
                for idx in indices:
                    new_val = random.randint(1, SMALL_MAX)
                    if new_val not in current_set_set and new_val != current_set[idx]:
                        neighbor[idx] = new_val
                    else:
                        valid = False
                        break
                if not valid:
                    continue

            if len(set(neighbor)) != n:
                continue
                
            neighbor_obj = objective(neighbor)
            delta_obj = neighbor_obj - current_obj
            
            if delta_obj < 0 or (T > T_min and random.random() < math.exp(-delta_obj / T)):
                current_set = neighbor
                current_set_set = set(neighbor)
                current_obj = neighbor_obj
                
                if current_obj < best_local_obj:
                    best_local_obj = current_obj
                    best_local_set = current_set[:]
            
            T = max(T * cooling_rate, T_min)
            
        # Polish with local search
        final_set, final_obj = local_search(best_local_set, max_iters=100, deadline=deadline)
        if final_obj < best_local_obj:
            best_local_set = final_set
            best_local_obj = final_obj
            
        return best_local_set, best_local_obj

    # Generate diverse seeds
    random.seed(42)
    seeds = []
    
    # 1. Arithmetic Progression
    seeds.append(list(range(1, n+1)))
    seeds.append(list(range(2, n+2)))
    
    # 2. 7-smooth numbers (numbers of form 2^a * 3^b * 5^c * 7^d)
    # These have small product sets due to restricted prime factors
    smooth_numbers = set()
    for a in range(0, 8):
        p2 = 2**a
        if p2 > SMALL_MAX:
            break
        for b in range(0, 6):
            p3 = p2 * 3**b
            if p3 > SMALL_MAX:
                break
            for c in range(0, 5):
                p5 = p3 * 5**c
                if p5 > SMALL_MAX:
                    break
                for d in range(0, 4):
                    val = p5 * 7**d
                    if 1 <= val <= SMALL_MAX:
                        smooth_numbers.add(val)
    
    # Take the first n 7-smooth numbers
    smooth_sorted = sorted(smooth_numbers)
    if len(smooth_sorted) >= n:
        seeds.append(smooth_sorted[:n])
        # Also try a subset with some variation
        seeds.append(smooth_sorted[1:n+1])
    
    # 3. 5-smooth numbers (Hamming numbers)
    smooth5 = set()
    for a in range(0, 8):
        p2 = 2**a
        if p2 > SMALL_MAX:
            break
        for b in range(0, 6):
            p3 = p2 * 3**b
            if p3 > SMALL_MAX:
                break
            for c in range(0, 5):
                val = p3 * 5**c
                if 1 <= val <= SMALL_MAX:
                    smooth5.add(val)
    smooth5_sorted = sorted(smooth5)
    if len(smooth5_sorted) >= n:
        seeds.append(smooth5_sorted[:n])
    
    # 4. Random sets
    for _ in range(15):
        seeds.append(random.sample(range(1, SMALL_MAX+1), n))
        
    # 5. "Clustered" sets (dense near 1, sparse elsewhere)
    base = 1
    s = [base]
    while len(s) < n:
        cand = s[-1] + random.randint(1, 3)
        if cand <= SMALL_MAX and cand not in s:
            s.append(cand)
        else:
            cand = random.randint(1, SMALL_MAX)
            if cand not in s:
                s.append(cand)
    seeds.append(s)
    
    # 6. "Mixed" - AP for first 10, random for next 10
    s = list(range(1, 11))
    while len(s) < n:
        cand = random.randint(10, SMALL_MAX)
        if cand not in s:
            s.append(cand)
    seeds.append(s)
    
    # 7. "Sparse" sets with larger gaps
    s = [1, 2]
    while len(s) < n:
        cand = s[-1] + random.randint(3, 10)
        if cand <= SMALL_MAX and cand not in s:
            s.append(cand)
        else:
            cand = random.randint(1, SMALL_MAX)
            if cand not in s:
                s.append(cand)
    if len(set(s)) == n:
        seeds.append(s)
    
    # 8. Powers of 2 mixed with small numbers
    s = [1, 2, 4, 8, 16, 32, 64]
    while len(s) < n:
        cand = random.randint(1, SMALL_MAX)
        if cand not in s:
            s.append(cand)
    seeds.append(s)
    
    # 9. Fibonacci-like
    s = [1, 1]
    while len(s) < n:
        cand = s[-1] + s[-2]
        if cand <= SMALL_MAX and cand not in s:
            s.append(cand)
        else:
            cand = random.randint(1, SMALL_MAX)
            if cand not in s:
                s.append(cand)
    if len(set(s)) == n:
        seeds.append(s)
    
    best_found_obj = float('inf')
    best_found_set = None

    # Run SA on each seed
    for seed_idx, seed in enumerate(seeds):
        if time.time() > deadline - 8.0:  # Leave 8s for final polish
            break
            
        # Ensure seed is valid
        if len(set(seed)) != n or any(x < 1 or x > SMALL_MAX for x in seed):
            continue
            
        # Allocate more iterations to promising seeds (smooth numbers first)
        if seed_idx < 4:
            max_iters_sa = 4000
        else:
            max_iters_sa = 2500
            
        sa_set, sa_obj = simulated_annealing(seed, max_iters=max_iters_sa)
        
        if sa_obj < best_found_obj:
            best_found_obj = sa_obj
            best_found_set = sa_set[:]
            
    # Final intensive polish on the best solution found
    if best_found_set is not None and time.time() < deadline - 3.0:
        polished_set, polished_obj = local_search(best_found_set, max_iters=500, deadline=deadline - 1.0)
        if polished_obj < best_found_obj:
            best_found_obj = polished_obj
            best_found_set = polished_set
        
    if best_found_set is None:
        best_found_set = list(range(1, n+1))
        best_found_obj = objective(best_found_set)
        
    # Scale up to fit 10000
    max_val = max(best_found_set)
    scale = 10000 // max_val
    if scale < 1:
        scale = 1
        
    final_set = [x * scale for x in best_found_set]
    
    # Verify final set
    if len(set(final_set)) != n or any(x > 10000 or x < 1 for x in final_set):
        # Fallback to simple AP scaled
        ap = list(range(1, n+1))
        max_ap = max(ap)
        sc = 10000 // max_ap
        final_set = [x * sc for x in ap]
        
    return final_set
