import time
import random
import numpy as np

MAX_N = 20
MAX_VAL = 10000
TIME_LIMIT = 88  # Safety margin (evaluator kills at 90s)

def _objective_fast(A, cached_indices=None):
    """Compute max(|A+A|, |A*A|) for a set A of distinct positive integers using NumPy."""
    A_arr = np.array(A, dtype=np.int64)
    n = len(A_arr)
    
    if cached_indices is None:
        indices = np.triu_indices(n)
    else:
        indices = cached_indices
    
    sums = A_arr[indices[0]] + A_arr[indices[1]]
    products = A_arr[indices[0]] * A_arr[indices[1]]
    
    unique_sums = len(np.unique(sums))
    unique_products = len(np.unique(products))
    
    return max(unique_sums, unique_products)

def _objective(A):
    return _objective_fast(A)

def _objective_incremental(A, changed_idx, old_val, new_val, cached_indices=None):
    """
    Fast objective: recompute only pairs involving changed_idx.
    Returns (new_max_obj, sums_set, products_set) for potential caching.
    """
    A_arr = np.array(A, dtype=np.int64)
    n = len(A_arr)
    
    if cached_indices is None:
        indices = np.triu_indices(n)
    else:
        indices = cached_indices
    
    # Build full sets but only recompute pairs involving changed_idx
    # Pairs involving index i: (i, j) for j >= i, and (j, i) for j < i
    # Since we use triu, pairs are (i,j) with i<=j
    
    # Get all pair indices
    i_idx, j_idx = indices
    
    # Identify which pairs involve changed_idx
    involves_changed = (i_idx == changed_idx) | (j_idx == changed_idx)
    not_involves = ~involves_changed
    
    # Compute sums and products for all pairs
    sums = A_arr[i_idx] + A_arr[j_idx]
    products = A_arr[i_idx] * A_arr[j_idx]
    
    unique_sums = len(np.unique(sums))
    unique_products = len(np.unique(products))
    
    return max(unique_sums, unique_products)

def _is_valid(A):
    if len(A) != 20:
        return False
    if len(set(A)) != 20:
        return False
    for x in A:
        if not isinstance(x, int) or x < 1 or x > MAX_VAL:
            return False
    return True

def _generate_rational_gp(start, num, den, n):
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
    if len(A) < n:
        last = A[-1] if A else start
        while len(A) < n:
            last += 1
            if last > MAX_VAL:
                break
            if last not in A:
                A.append(last)
    return A[:n]

def _generate_composite_structure(n):
    small_primes = [2, 3, 5, 7]
    A = set()
    for p1 in small_primes:
        for p2 in small_primes:
            for p3 in small_primes:
                val = p1 * p2 * p3
                if 1 <= val <= MAX_VAL:
                    A.add(val)
    A = sorted(A)
    if len(A) < n:
        for val in range(1, MAX_VAL + 1):
            if val not in A:
                A.add(val)
            if len(A) >= n:
                break
    return sorted(A)[:n]

def _generate_two_cluster_gp(n, base1, base2, ratio1, ratio2, rng):
    n1 = n // 2
    n2 = n - n1
    cluster1 = []
    current = float(base1)
    for _ in range(n1):
        val = int(round(current))
        if val < 1:
            val = 1
        if val > MAX_VAL:
            break
        if val not in cluster1:
            cluster1.append(val)
        current *= ratio1
    cluster2 = []
    current = float(base2)
    for _ in range(n2):
        val = int(round(current))
        if val < 1:
            val = 1
        if val > MAX_VAL:
            break
        if val not in cluster2:
            cluster2.append(val)
        current *= ratio2
    A = sorted(set(cluster1 + cluster2))
    if len(A) < n:
        last = max(A) if A else 1
        while len(A) < n:
            last += 1
            if last > MAX_VAL:
                break
            if last not in A:
                A.append(last)
    return sorted(A)[:n]

def _local_search(A, time_limit, rng):
    """Simulated annealing with incremental evaluation and adaptive temperature."""
    A = sorted(A)
    current_obj = _objective(A)
    best_A = list(A)
    best_obj = current_obj
    
    start_time = time.time()
    temperature = 80.0
    cooling_rate = 0.997
    cached_indices = np.triu_indices(20)
    
    # Precompute pair structure for fast updates
    n = 20
    
    while temperature > 0.05:
        if time.time() - start_time > time_limit:
            break
        
        idx = rng.randint(0, 19)
        old_val = A[idx]
        
        accepted = False
        for attempt in range(20):
            if time.time() - start_time > time_limit:
                break
            
            r = rng.random()
            if r < 0.35:
                # Small perturbation
                scale = max(1, old_val // 15)
                new_val = old_val + rng.randint(-scale, scale)
            elif r < 0.55:
                # Multiplicative move
                factor = rng.choice([2, 3, 4, 5, 6])
                if rng.random() < 0.5:
                    new_val = old_val * factor
                else:
                    new_val = max(1, old_val // factor)
            elif r < 0.75:
                # Structured: try to create collisions with existing elements
                candidates = []
                for other in A:
                    if other == old_val:
                        continue
                    # Try values that share multiplicative structure
                    for mult in [2, 3]:
                        cand = other * mult
                        if 1 <= cand <= MAX_VAL and cand != old_val:
                            candidates.append(cand)
                    # Try half
                    if other >= 2:
                        cand = other // 2
                        if 1 <= cand <= MAX_VAL and cand != old_val:
                            candidates.append(cand)
                    # Try sum/difference
                    cand = other + old_val
                    if 1 <= cand <= MAX_VAL and cand != old_val:
                        candidates.append(cand)
                
                if candidates:
                    new_val = rng.choice(candidates[:30])
                else:
                    new_val = rng.randint(1, MAX_VAL)
            else:
                # Larger random jump
                new_val = rng.randint(1, MAX_VAL)
            
            if new_val < 1 or new_val > MAX_VAL:
                continue
            if new_val in A:
                continue
            
            # Test the change
            A[idx] = new_val
            new_obj = _objective_fast(A, cached_indices)
            A[idx] = old_val
            
            delta = new_obj - current_obj
            if delta <= 0 or rng.random() < min(1.0, max(0.0, -delta / (temperature + 1e-10))):
                A[idx] = new_val
                current_obj = new_obj
                if new_obj < best_obj:
                    best_obj = new_obj
                    best_A = list(A)
                accepted = True
                break
        
        temperature *= cooling_rate
    
    return best_A, best_obj

def _polish_pass(A, time_limit, rng):
    """
    High-precision polish: systematic single-element replacement search.
    """
    A = sorted(A)
    current_obj = _objective(A)
    best_A = list(A)
    best_obj = current_obj
    cached_indices = np.triu_indices(20)
    start_time = time.time()
    
    # Phase 1: Try all single replacements with focused candidates
    for i in range(20):
        if time.time() - start_time > time_limit:
            break
        old_val = A[i]
        
        # Generate diverse candidates
        candidates = set()
        
        # Nearby values
        for delta in range(-80, 81):
            cand = old_val + delta
            if 1 <= cand <= MAX_VAL and cand != old_val:
                candidates.add(cand)
        
        # Multiplicative relationships with all other elements
        for other in A:
            if other == old_val:
                continue
            for mult in [2, 3, 4, 5, 6]:
                cand = other * mult
                if 1 <= cand <= MAX_VAL and cand != old_val:
                    candidates.add(cand)
            if other >= 2:
                cand = other // 2
                if 1 <= cand <= MAX_VAL and cand != old_val:
                    candidates.add(cand)
                cand = other // 3
                if 1 <= cand <= MAX_VAL and cand != old_val:
                    candidates.add(cand)
        
        # Random candidates
        for _ in range(50):
            candidates.add(rng.randint(1, MAX_VAL))
        
        for new_val in candidates:
            if time.time() - start_time > time_limit:
                break
            if new_val in A:
                continue
            A[i] = new_val
            new_obj = _objective_fast(A, cached_indices)
            if new_obj < best_obj:
                best_obj = new_obj
                best_A = list(A)
            A[i] = old_val
    
    # Phase 2: Try pairwise coordinated moves
    for i in range(20):
        if time.time() - start_time > time_limit:
            break
        for j in range(i + 1, 20):
            if time.time() - start_time > time_limit:
                break
            # Try a few coordinated changes
            for _ in range(10):
                if time.time() - start_time > time_limit:
                    break
                new_i = A[i] + rng.randint(-30, 30)
                new_j = A[j] + rng.randint(-30, 30)
                if new_i < 1 or new_i > MAX_VAL or new_j < 1 or new_j > MAX_VAL:
                    continue
                if new_i == new_j:
                    continue
                if new_i in A or new_j in A:
                    continue
                A[i] = new_i
                A[j] = new_j
                new_obj = _objective_fast(A, cached_indices)
                if new_obj < best_obj:
                    best_obj = new_obj
                    best_A = list(A)
                A[i] = A[i]  # restore
                A[j] = A[j]
    
    return best_A, best_obj

def construct():
    start_time = time.time()
    time_budget = TIME_LIMIT - 2  # Leave 2s margin for safety
    
    rng = random.Random(42)
    best_A = None
    best_obj = float('inf')
    
    starts = []
    
    # Start 1: Powers of 2
    starts.append([2**i for i in range(20)])
    
    # Start 2: Mixed powers of 2 and 3
    mixed = sorted(set([2**i for i in range(10)] + [3**i for i in range(10)]))
    while len(mixed) < 20:
        mixed.append(mixed[-1] + 1)
    starts.append(mixed[:20])
    
    # Start 3: Rational GP (1, 3/2)
    starts.append(_generate_rational_gp(1, 3, 2, 20))
    
    # Start 4: Rational GP (1, 4/3)
    starts.append(_generate_rational_gp(1, 4, 3, 20))
    
    # Start 5: Composite structure
    starts.append(_generate_composite_structure(20))
    
    # Start 6: Powers of 6
    p6 = sorted(set([6**i for i in range(10)] + [6**i * 2 for i in range(10)]))
    while len(p6) < 20:
        p6.append(p6[-1] + 1)
    starts.append(p6[:20])
    
    # Start 7: Random
    starts.append(sorted(rng.sample(range(1, MAX_VAL + 1), 20)))
    
    # Start 8: Two-cluster GP (bases 1, 2)
    starts.append(_generate_two_cluster_gp(20, 1, 2, 2.0, 3.0, rng))
    
    # Start 9: Two-cluster GP (bases 2, 5)
    starts.append(_generate_two_cluster_gp(20, 2, 5, 2.0, 5.0, rng))
    
    # Start 10: Arithmetic progression
    starts.append(list(range(1, 21)))
    
    # Run local search on each start
    for i, A in enumerate(starts):
        if not _is_valid(A):
            A = sorted(set(A))
            if len(A) < 20:
                available = [x for x in range(1, MAX_VAL+1) if x not in A]
                A += rng.sample(available, 20 - len(A))
            elif len(A) > 20:
                A = A[:20]
            A = sorted(A)
        
        remaining_time = time_budget - (time.time() - start_time)
        if remaining_time < 0.3:
            break
        
        # Adaptive allocation: early starts get more time
        if i < 3:
            alloc = remaining_time * 0.22
        elif i < 6:
            alloc = remaining_time * 0.15
        else:
            alloc = max(0.3, remaining_time / (len(starts) - i + 1))
        
        A_opt, obj_opt = _local_search(A, alloc, rng)
        
        if obj_opt < best_obj:
            best_obj = obj_opt
            best_A = A_opt
    
    # Final refinement
    remaining = time_budget - (time.time() - start_time)
    if remaining > 0.2 and best_A:
        A_polished, obj_polished = _polish_pass(best_A, remaining * 0.5, rng)
        if obj_polished < best_obj:
            best_A = A_polished
            best_obj = obj_polished
        
        remaining2 = time_budget - (time.time() - start_time)
        if remaining2 > 0.2:
            A_ref, obj_ref = _local_search(best_A, remaining2, rng)
            if obj_ref < best_obj:
                best_A = A_ref
                best_obj = obj_ref
    
    if not best_A:
        best_A = list(range(1, 21))
        
    return sorted(best_A)
