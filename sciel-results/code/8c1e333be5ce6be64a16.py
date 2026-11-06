import time
import random
import itertools
import numpy as np

MAX_N = 20
MAX_VAL = 10000
TIME_LIMIT = 85  # safety margin before 90s kill

def _objective(A):
    """Compute max(|A+A|, |A*A|) for a set A of distinct positive integers.
    Uses NumPy vectorization for speed."""
    arr = np.array(sorted(A), dtype=np.int64)
    n = len(arr)
    # Upper-triangular indices including diagonal
    i_idx, j_idx = np.triu_indices(n)
    sums = arr[i_idx] + arr[j_idx]
    prods = arr[i_idx] * arr[j_idx]
    return max(len(np.unique(sums)), len(np.unique(prods)))

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

def _geometric_start(start, ratio_num, ratio_den):
    """Generate a geometric progression with rational ratio p/q."""
    A = []
    val = start
    for _ in range(20):
        if val > MAX_VAL:
            break
        if val in A:
            while val in A:
                val += 1
            if val > MAX_VAL:
                break
        A.append(val)
        val = (val * ratio_num + ratio_den // 2) // ratio_den
    if len(A) == 20 and _is_valid(A):
        return A
    return None

def _logarithmic_set(base, count=20):
    """Generate numbers approximately at logarithmic spacing."""
    A = []
    for i in range(count):
        val = int(base * (1.3 ** i))
        if val > MAX_VAL:
            break
        if val not in A and val >= 1:
            A.append(val)
    while len(A) < count:
        A.append(len(A) + 1)
    if len(A) == 20 and _is_valid(A):
        return A
    return None

def _dual_geometric_set(start1, ratio1_num, ratio1_den, n1, start2, ratio2_num, ratio2_den, n2):
    """Generate a set that is a union of two geometric progressions.
    This maximizes multiplicative collisions: products within each GP collide
    due to shared prime factors, and cross-products may also collide."""
    A = []
    # First GP
    val = start1
    for _ in range(n1):
        if val > MAX_VAL:
            break
        if val not in A and val >= 1:
            A.append(val)
        val = (val * ratio1_num + ratio1_den // 2) // ratio1_den
    # Second GP
    val = start2
    for _ in range(n2):
        if val > MAX_VAL:
            break
        if val not in A and val >= 1:
            A.append(val)
        val = (val * ratio2_num + ratio2_den // 2) // ratio2_den
    # Fill remaining if needed
    while len(A) < 20:
        for v in range(1, MAX_VAL + 1):
            if v not in A:
                A.append(v)
                break
        if len(A) < 20:
            break
    if len(A) == 20 and _is_valid(A):
        return A
    return None

def _powers_of_two_and_three(n2, n3):
    """Generate set of powers of 2 and powers of 3, choosing n2+n3=20 total."""
    A = []
    # Powers of 2
    v = 1
    for _ in range(n2):
        if v > MAX_VAL:
            break
        if v not in A:
            A.append(v)
        v *= 2
    # Powers of 3
    v = 1
    for _ in range(n3):
        if v > MAX_VAL:
            break
        if v not in A:
            A.append(v)
        v *= 3
    # Fill if needed
    while len(A) < 20:
        for v in range(1, MAX_VAL + 1):
            if v not in A:
                A.append(v)
                break
        if len(A) < 20:
            break
    if len(A) == 20 and _is_valid(A):
        return A
    return None

def _construct_candidates():
    """Generate a list of candidate starting sets."""
    candidates = []
    
    # 1. Smooth numbers
    try:
        smooths = _smooth_numbers(10000, 20)
        if len(smooths) == 20:
            candidates.append(smooths)
    except:
        pass
        
    # 2. Dual geometric sets (key strategy: two GPs for multiplicative collisions)
    # Powers of 2 and powers of 3 with various splits
    for n2 in range(5, 16):
        n3 = 20 - n2
        A = _powers_of_two_and_three(n2, n3)
        if A is not None:
            candidates.append(A)
    
    # More dual GPs with different bases
    dual_configs = [
        # (start1, r1n, r1d, n1, start2, r2n, r2d, n2)
        (1, 2, 1, 10, 3, 3, 1, 10),   # 2^0..2^9, 3^0..3^9
        (1, 2, 1, 12, 5, 5, 1, 8),    # more 2s, fewer 5s
        (1, 3, 1, 8, 2, 2, 1, 12),    # more 2s, fewer 3s
        (2, 2, 1, 10, 3, 3, 1, 10),   # shifted
        (1, 4, 1, 8, 3, 3, 1, 12),    # powers of 4 and 3
        (1, 2, 1, 10, 9, 9, 1, 10),   # powers of 2 and 9
        (1, 3, 1, 10, 4, 4, 1, 10),   # powers of 3 and 4
        (1, 2, 1, 15, 5, 5, 1, 5),    # mostly 2s
        (1, 5, 1, 8, 2, 2, 1, 12),    # mostly 2s, some 5s
        (1, 2, 1, 10, 7, 7, 1, 10),   # powers of 2 and 7
    ]
    for cfg in dual_configs:
        A = _dual_geometric_set(*cfg)
        if A is not None:
            candidates.append(A)
    
    # 3. Geometric Progressions
    rational_ratios = [
        (6, 5), (7, 6), (8, 7), (5, 4), (4, 3),
        (9, 8), (11, 10), (3, 2), (5, 3), (7, 5),
        (10, 9), (12, 11), (13, 12), (14, 13), (15, 14),
    ]
    for ratio_num, ratio_den in rational_ratios:
        for start in [1, 2, 3, 4, 5]:
            A = _geometric_start(start, ratio_num, ratio_den)
            if A is not None:
                candidates.append(A)
                
    # 4. Arithmetic Progressions
    for start in [1, 2, 5, 10]:
        for step in [1, 2, 3, 5]:
            A = [start + i * step for i in range(20)]
            if max(A) <= 10000 and _is_valid(A):
                candidates.append(A)
    
    # 5. Logarithmic sets
    for base in [1, 2, 3, 5]:
        A = _logarithmic_set(base)
        if A is not None:
            candidates.append(A)
    
    # 6. Random with different ranges
    rng = random.Random(123)
    for _ in range(3):
        candidates.append(rng.sample(range(1, 10001), 20))
    for _ in range(3):
        candidates.append(rng.sample(range(1, 1001), 20))
    for _ in range(2):
        candidates.append(rng.sample(range(1, 200), 20))
    
    return candidates

def _local_search(A, time_limit, rng):
    """Local search with simulated annealing, paired swaps, and cluster-aware moves."""
    A = sorted(A)
    current_obj = _objective(A)
    best_A = list(A)
    best_obj = current_obj
    
    start_time = time.time()
    temperature = 30.0
    cooling_rate = 0.9985  # slightly slower cooling for more iterations
    
    # Identify potential "cluster" structure: which elements are close in log space
    # This helps with cluster-aware moves
    log_vals = np.log2(np.array(A, dtype=np.float64) + 1)
    # Simple clustering: elements within 0.5 log units are in same cluster
    cluster_ids = np.zeros(20, dtype=int)
    cluster_ids[0] = 0
    next_cluster = 1
    for i in range(1, 20):
        if log_vals[i] - log_vals[i-1] > 0.5:
            next_cluster += 1
        cluster_ids[i] = next_cluster
    
    while temperature > 0.01:
        if time.time() - start_time > time_limit:
            break
        
        # Choose move type
        move_type = rng.random()
        
        if move_type < 0.35:
            # Single element change
            idx = rng.randint(0, 19)
            old_val = A[idx]
            
            r = rng.random()
            if r < 0.3:
                # Small perturbation
                new_val = old_val + rng.randint(-5, 5)
            elif r < 0.5:
                # Medium perturbation
                new_val = old_val + rng.randint(-50, 50)
            elif r < 0.7:
                # Try to create multiplicative collision: pick value that is a multiple
                # of an existing element or a divisor
                existing = rng.choice(A)
                if rng.random() < 0.5:
                    # Try new_val = k * existing for small k
                    k = rng.randint(2, 5)
                    new_val = existing * k
                else:
                    # Try new_val = existing // k
                    k = rng.randint(2, 5)
                    new_val = existing // k
                    if new_val < 1:
                        new_val = existing * k
            elif r < 0.85:
                # Random
                new_val = rng.randint(1, 10000)
            else:
                # Try to hit a power of 2 or 3
                if rng.random() < 0.5:
                    p = rng.randint(0, 13)
                    new_val = 2 ** p
                else:
                    p = rng.randint(0, 8)
                    new_val = 3 ** p
            
            if new_val < 1 or new_val > 10000 or new_val in A:
                continue
            
            A[idx] = new_val
            new_obj = _objective(A)
            A[idx] = old_val
            
            delta = new_obj - current_obj
            if delta <= 0 or rng.random() < min(1.0, max(0.0, -delta / (temperature + 1e-10))):
                A[idx] = new_val
                current_obj = new_obj
                if new_obj < best_obj:
                    best_obj = new_obj
                    best_A = list(A)
        
        elif move_type < 0.65:
            # Paired swap: change two elements simultaneously
            idx1, idx2 = rng.sample(range(20), 2)
            old1, old2 = A[idx1], A[idx2]
            
            new1 = old1 + rng.randint(-20, 20)
            new2 = old2 + rng.randint(-20, 20)
            
            if new1 < 1 or new1 > 10000 or new2 < 1 or new2 > 10000:
                continue
            if new1 == new2 or new1 in A or new2 in A:
                continue
            
            A[idx1] = new1
            A[idx2] = new2
            new_obj = _objective(A)
            A[idx1] = old1
            A[idx2] = old2
            
            delta = new_obj - current_obj
            if delta <= 0 or rng.random() < min(1.0, max(0.0, -delta / (temperature + 1e-10))):
                A[idx1] = new1
                A[idx2] = new2
                current_obj = new_obj
                if new_obj < best_obj:
                    best_obj = new_obj
                    best_A = list(A)
        
        else:
            # Cluster-aware move: try to make two elements form a geometric pair
            # Pick two elements and try to make one a small multiple of the other
            idx1, idx2 = rng.sample(range(20), 2)
            old1, old2 = A[idx1], A[idx2]
            
            # Try new2 = k * new1 for small k, or new1 = k * new2
            k = rng.choice([2, 3, 4, 5])
            if rng.random() < 0.5:
                # Make new2 = k * old1
                new1 = old1
                new2 = old1 * k
            else:
                # Make new1 = k * old2
                new2 = old2
                new1 = old2 * k
            
            if new1 < 1 or new1 > 10000 or new2 < 1 or new2 > 10000:
                continue
            if new1 == new2 or new1 in A or new2 in A:
                continue
            
            A[idx1] = new1
            A[idx2] = new2
            new_obj = _objective(A)
            A[idx1] = old1
            A[idx2] = old2
            
            delta = new_obj - current_obj
            if delta <= 0 or rng.random() < min(1.0, max(0.0, -delta / (temperature + 1e-10))):
                A[idx1] = new1
                A[idx2] = new2
                current_obj = new_obj
                if new_obj < best_obj:
                    best_obj = new_obj
                    best_A = list(A)
        
        temperature *= cooling_rate
    
    # Final polish: try all single-element replacements with nearby values
    if time.time() - start_time < time_limit - 1:
        A_polish = list(best_A)
        obj_polish = best_obj
        for i in range(20):
            if time.time() - start_time > time_limit - 1:
                break
            old = A_polish[i]
            # Try nearby values and values that create multiplicative structure
            candidates_nv = list(range(max(1, old - 15), min(10001, old + 16)))
            # Add some multiplicative candidates
            for k in [2, 3, 4, 5]:
                if old * k <= 10000:
                    candidates_nv.append(old * k)
                if old % k == 0 and old // k >= 1:
                    candidates_nv.append(old // k)
            
            for nv in candidates_nv:
                if nv == old or nv < 1 or nv > 10000 or nv in A_polish:
                    continue
                A_polish[i] = nv
                o = _objective(A_polish)
                if o < obj_polish:
                    obj_polish = o
                    best_A = list(A_polish)
                    best_obj = obj_polish
                A_polish[i] = old
    
    return best_A, best_obj

def construct():
    start_time = time.time()
    time_budget = TIME_LIMIT - 5
    
    candidates = _construct_candidates()
    
    best_A = None
    best_obj = float('inf')
    
    rng = random.Random(42)
    
    # Phase 1: Quick evaluation of all candidates
    candidate_objs = []
    for A in candidates:
        if not _is_valid(A):
            continue
        obj = _objective(A)
        candidate_objs.append((obj, A))
    
    candidate_objs.sort(key=lambda x: x[0])
    
    n_cands = len(candidate_objs)
    if n_cands == 0:
        best_A = list(range(1, 21))
        best_obj = _objective(best_A)
    else:
        # Try top candidates with local search
        # Give more time to top candidates since dual GPs should be strong starts
        top_n = min(10, n_cands)
        for i, (obj_init, A) in enumerate(candidate_objs[:top_n]):
            elapsed = time.time() - start_time
            remaining = time_budget - elapsed
            if remaining < 2:
                break
            # Allocate more time to better initial candidates
            if i < 3:
                alloc = min(remaining, 15.0)
            elif i < 6:
                alloc = min(remaining, 10.0)
            else:
                alloc = min(remaining, 6.0)
            refined_A, obj = _local_search(A, alloc, rng)
            if obj < best_obj:
                best_obj = obj
                best_A = refined_A
        
        # Try more candidates if time permits
        for obj_init, A in candidate_objs[top_n:]:
            elapsed = time.time() - start_time
            remaining = time_budget - elapsed
            if remaining < 3:
                break
            if obj_init >= best_obj:
                continue
            refined_A, obj = _local_search(A, min(remaining, 5.0), rng)
            if obj < best_obj:
                best_obj = obj
                best_A = refined_A
    
    if best_A is None:
        best_A = list(range(1, 21))
        best_obj = _objective(best_A)
        
    # Final refinement
    remaining = TIME_LIMIT - 5 - (time.time() - start_time)
    if remaining > 2:
        try:
            refined_A, refined_obj = _local_search(best_A, remaining, rng)
            if refined_obj < best_obj:
                best_A = refined_A
        except Exception:
            pass

    if not _is_valid(best_A):
        best_A = list(range(1, 21))
        
    return sorted(best_A)
