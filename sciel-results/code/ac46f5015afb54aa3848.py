import time
import random
import math
import numpy as np

def set_sizes_fast(A):
    """
    Computes the sizes of the sumset A+A and product set A*A using NumPy.
    A is a list of distinct positive integers.
    """
    A = np.array(A, dtype=np.int64)
    n = len(A)
    # Compute all pairwise sums and products (including diagonal)
    A_col = A[:, np.newaxis]
    A_row = A[np.newaxis, :]
    sums = A_col + A_row
    prods = A_col * A_row
    # Only need upper triangle including diagonal for unique pairs, but since we're counting distinct values,
    # we can just use all entries (duplicates don't matter for set size)
    sum_set = np.unique(sums)
    prod_set = np.unique(prods)
    return len(sum_set), len(prod_set)

def objective(A):
    """
    Returns the maximum of the sumset and product set sizes.
    """
    s, p = set_sizes_fast(A)
    return max(s, p)

def local_search(current_set, max_iters=100, small_max=100, deadline=None):
    """
    Steepest descent local search to polish a candidate.
    Tries to replace each element with a value in [1, small_max] that improves the objective.
    """
    n = len(current_set)
    current_set = list(current_set)
    current_obj = objective(current_set)
    
    improved = True
    iter_count = 0
    while improved and iter_count < max_iters:
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        iter_count += 1
        best_neighbor = None
        best_neighbor_obj = current_obj
        
        for i in range(n):
            old_val = current_set[i]
            # Try all values in a focused range around current value
            search_range = max(1, min(small_max, old_val + 10))
            for new_val in range(1, search_range + 1):
                if new_val == old_val or new_val in current_set:
                    continue
                neighbor = current_set[:]
                neighbor[i] = new_val
                n_obj = objective(neighbor)
                if n_obj < best_neighbor_obj:
                    best_neighbor_obj = n_obj
                    best_neighbor = neighbor[:]
            
            # Also try nearby values beyond search range
            for delta in [-5, -3, -2, -1, 1, 2, 3, 5]:
                new_val = old_val + delta
                if 1 <= new_val <= small_max and new_val != old_val and new_val not in current_set:
                    neighbor = current_set[:]
                    neighbor[i] = new_val
                    n_obj = objective(neighbor)
                    if n_obj < best_neighbor_obj:
                        best_neighbor_obj = n_obj
                        best_neighbor = neighbor[:]

        if best_neighbor is not None and best_neighbor_obj < current_obj:
            current_set = best_neighbor
            current_obj = best_neighbor_obj
            improved = True
            
    return current_set, current_obj

def simulated_annealing(start_set, max_iters=2000, small_max=100, deadline=None):
    """
    Simulated Annealing to find a good candidate in the small domain.
    """
    n = len(start_set)
    current_set = list(start_set)
    current_obj = objective(current_set)
    
    T = 20.0
    T_min = 0.1
    cooling_rate = 0.995
    
    best_local_set = current_set[:]
    best_local_obj = current_obj
    
    for i in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        neighbor = current_set[:]
        move_type = random.random()
        
        if move_type < 0.6:
            # Replace one element
            idx = random.randint(0, n-1)
            new_val = random.randint(1, small_max)
            if new_val not in current_set or new_val == current_set[idx]:
                neighbor[idx] = new_val
        elif move_type < 0.8:
            # Swap two elements with new random values
            i1, i2 = random.sample(range(n), 2)
            v1 = random.randint(1, small_max)
            v2 = random.randint(1, small_max)
            if (v1 not in current_set or v1 == current_set[i1]) and \
               (v2 not in current_set or v2 == current_set[i2]) and \
               v1 != v2:
                neighbor[i1] = v1
                neighbor[i2] = v2
            else:
                # Fallback to single replace
                idx = random.randint(0, n-1)
                new_val = random.randint(1, small_max)
                if new_val not in current_set or new_val == current_set[idx]:
                    neighbor[idx] = new_val
        else:
            # Small perturbation
            idx = random.randint(0, n-1)
            delta = random.choice([-5, -2, -1, 1, 2, 5])
            new_val = current_set[idx] + delta
            if 1 <= new_val <= small_max and new_val not in current_set:
                neighbor[idx] = new_val
            else:
                new_val = random.randint(1, small_max)
                if new_val not in current_set:
                    neighbor[idx] = new_val
                else:
                    continue

        if len(set(neighbor)) != n:
            continue
            
        neighbor_obj = objective(neighbor)
        delta_obj = neighbor_obj - current_obj
        
        if delta_obj < 0 or random.random() < math.exp(-delta_obj / T):
            current_set = neighbor
            current_obj = neighbor_obj
            
            if current_obj < best_local_obj:
                best_local_obj = current_obj
                best_local_set = current_set[:]
        
        T = max(T * cooling_rate, T_min)
        
    # Polish with local search
    final_set, final_obj = local_search(best_local_set, max_iters=50, small_max=small_max, deadline=deadline)
    if final_obj < best_local_obj:
        best_local_set = final_set
        best_local_obj = final_obj
        
    return best_local_set, best_local_obj

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin under 90s
    
    n = 20
    SMALL_MAX = 100
    
    best_obj_global = float('inf')
    best_set_global = None

    # Generate diverse seeds
    random.seed(42)
    seeds = []
    
    # 1. Arithmetic Progression
    seeds.append(list(range(1, n+1)))
    seeds.append(list(range(2, n+2)))
    
    # 2. Random sets
    for _ in range(10):
        seeds.append(random.sample(range(1, SMALL_MAX+1), n))
        
    # 3. "Clustered" sets (dense near 1)
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
    
    # 4. "Mixed" - AP for first 10, random for next 10
    s = list(range(1, 11))
    while len(s) < n:
        cand = random.randint(10, SMALL_MAX)
        if cand not in s:
            s.append(cand)
    seeds.append(s)
    
    # 5. "Sparse" - Geometric-like (small ratio)
    s = [1]
    while len(s) < n:
        # Try to add next geometric term, but if it's too big, add random
        next_geometric = s[-1] * 2
        if next_geometric <= SMALL_MAX and next_geometric not in s:
            s.append(next_geometric)
        else:
            cand = random.randint(1, SMALL_MAX)
            if cand not in s:
                s.append(cand)
    seeds.append(s)

    # 6. 2x3 lattice seeds: {2^i * 3^j} with small exponents
    # This structurally minimizes |A*A| since products map to exponent pairs
    lattice_seeds = []
    for i_max in range(1, 5):
        for j_max in range(1, 5):
            # Generate all 2^i * 3^j for 0<=i<=i_max, 0<=j<=j_max
            candidates = []
            for i in range(i_max + 1):
                for j in range(j_max + 1):
                    val = (2 ** i) * (3 ** j)
                    if val <= SMALL_MAX:
                        candidates.append(val)
            candidates = sorted(set(candidates))
            if len(candidates) >= n:
                # Take first n
                lattice_seeds.append(candidates[:n])
            elif len(candidates) > 0:
                # Pad with random values
                s = list(candidates)
                while len(s) < n:
                    cand = random.randint(1, SMALL_MAX)
                    if cand not in s:
                        s.append(cand)
                lattice_seeds.append(s)
    
    seeds.extend(lattice_seeds)

    best_found_obj = float('inf')
    best_found_set = None

    # Run SA on each seed
    for seed in seeds:
        if time.time() > deadline:
            break
            
        # Ensure seed is valid
        if len(set(seed)) != n or any(x < 1 or x > SMALL_MAX for x in seed):
            continue
            
        sa_set, sa_obj = simulated_annealing(seed, max_iters=1500, small_max=SMALL_MAX, deadline=deadline)
        
        if sa_obj < best_found_obj:
            best_found_obj = sa_obj
            best_found_set = sa_set[:]

    # If no valid set found (should not happen), fallback
    if best_found_set is None:
        best_found_set = list(range(1, n+1))
        best_found_obj = objective(best_found_set)
    
    # Final polish with more iterations if time allows
    if time.time() < deadline - 5:
        polished_set, polished_obj = local_search(best_found_set, max_iters=100, small_max=SMALL_MAX, deadline=deadline)
        if polished_obj < best_found_obj:
            best_found_set = polished_set
            best_found_obj = polished_obj
    
    # Scale up to fit 10000
    max_val = max(best_found_set)
    scale = 10000 // max_val
    if scale < 1:
        scale = 1
        
    final_set = [x * scale for x in best_found_set]
    
    # Verify final set
    if len(set(final_set)) != n or any(x > 10000 for x in final_set):
        # Fallback to simple AP
        return list(range(1, n+1))
        
    return final_set
