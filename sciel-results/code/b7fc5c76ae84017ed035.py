"""Optimized search for integer sum-product extremal sets."""
import random
import time
import numpy as np
import bisect
import numpy as np

def _objective(A):
    """Compute max(|A+A|, |A*A|) for set A of 20 distinct positive integers."""
    A = np.array(A, dtype=np.int64)
    n = len(A)
    # Sumset: all a_i + a_j for i <= j (distinct sums)
    sums = np.add(A[:, None], A[None, :])
    sums = sums[np.triu_indices(n)]
    n_sums = len(np.unique(sums))
    # Product set: all a_i * a_j for i <= j
    prods = np.multiply(A[:, None], A[None, :])
    prods = prods[np.triu_indices(n)]
    n_prods = len(np.unique(prods))
    return max(n_sums, n_prods)

def _is_valid(A):
    """Check A has 20 distinct positive integers <= 10000."""
    if len(A) != 20:
        return False
    if len(set(A)) != 20:
        return False
    for x in A:
        if x < 1 or x > 10000:
            return False
    return True

def _generate_structured_candidates():
    """Generate diverse initial candidates using different structures."""
    candidates = []
    
    # 1. Small consecutive integers
    candidates.append(list(range(1, 21)))
    candidates.append(list(range(2, 22)))
    candidates.append(list(range(5, 25)))
    
    # 2. Powers of 2 (minimizes product set, maximizes sum set)
    powers2 = [2**i for i in range(20)]
    if max(powers2) <= 10000:
        candidates.append(powers2)
    else:
        # Truncated powers
        p = []
        v = 1
        while len(p) < 20 and v <= 10000:
            p.append(v)
            v *= 2
        while len(p) < 20:
            p.append(p[-1] + 1)
        candidates.append(p)
    
    # 3. Geometric progression with small ratio
    for ratio in [1.1, 1.2, 1.3, 1.5]:
        gp = []
        v = 1.0
        while len(gp) < 20 and v <= 10000:
            gp.append(int(round(v)))
            v *= ratio
        # Remove duplicates and ensure 20 distinct
        gp = list(dict.fromkeys(gp))
        while len(gp) < 20:
            gp.append(max(gp) + 1)
        if _is_valid(gp):
            candidates.append(gp[:20])
    
    # 4. Mixed: small numbers plus some large
    base = list(range(1, 11))
    large = [500, 600, 700, 800, 900, 1000, 1500, 2000, 3000, 5000]
    mixed = base + large
    if _is_valid(mixed):
        candidates.append(mixed)
    
    # 5. Random samples
    for _ in range(10):
        r = random.sample(range(1, 10001), 20)
        candidates.append(sorted(r))
    
    # 6. Arithmetic progressions
    for start in [1, 2, 5, 10]:
        for step in [1, 2, 3, 5, 10]:
            ap = [start + i * step for i in range(20)]
            if max(ap) <= 10000 and _is_valid(ap):
                candidates.append(ap)
    
    # 7. Fibonacci-like
    fib = [1, 1]
    while len(fib) < 20:
        fib.append(fib[-1] + fib[-2])
    # Remove duplicate 1s
    fib = list(dict.fromkeys(fib))
    while len(fib) < 20:
        fib.append(fib[-1] + fib[-2])
    if _is_valid(fib):
        candidates.append(fib[:20])
    
    # 8. Dual arithmetic progressions (from directive)
    # Union of two APs with same difference, different starts/lengths
    # This creates collisions in both sumset and product set
    for d in [1, 2, 3, 4, 5, 6, 8, 10]:
        for start1 in [1, 2, 3, 4, 5]:
            for start2 in [start1 + d, start1 + 2*d, start1 + 3*d]:
                for len1 in [10, 11, 12]:
                    len2 = 20 - len1
                    if len2 < 1:
                        continue
                    ap1 = [start1 + i * d for i in range(len1)]
                    ap2 = [start2 + i * d for i in range(len2)]
                    combined = ap1 + ap2
                    if len(set(combined)) == 20 and max(combined) <= 10000 and min(combined) >= 1:
                        combined = sorted(combined)
                        if _is_valid(combined):
                            candidates.append(combined)
    
    # 9. Smooth number based sets (powers of 2 and 3)
    smooth = []
    for a in range(0, 15):
        for b in range(0, 15):
            val = (2 ** a) * (3 ** b)
            if 1 <= val <= 10000:
                smooth.append(val)
    smooth = sorted(set(smooth))
    # Pick 20 from smooth numbers
    if len(smooth) >= 20:
        # Try different subsets
        for i in range(0, max(1, len(smooth) - 19), 5):
            subset = smooth[i:i+20]
            if _is_valid(subset):
                candidates.append(subset)
    
    # 10. Hybrid: mix of small AP and smooth numbers
    for d in [1, 2, 3]:
        for start in [1, 2, 3]:
            small_ap = [start + i * d for i in range(10)]
            remaining_smooth = [x for x in smooth if x not in small_ap]
            if len(remaining_smooth) >= 10:
                for offset in [0, 5, 10, 15]:
                    large_smooth = remaining_smooth[offset:offset+10]
                    combined = small_ap + large_smooth
                    if len(set(combined)) == 20 and max(combined) <= 10000:
                        combined = sorted(combined)
                        if _is_valid(combined):
                            candidates.append(combined)
    
    return candidates

def _greedy_coordinate_descent(A, deadline, best_val, best_A):
    """
    Structural change: Instead of random perturbations, use systematic
    coordinate descent. For each element, try a curated set of candidate
    values and keep the best improvement. This is more deterministic and
    less likely to miss good moves.
    """
    A = list(A)
    cur_val = _objective(A)
    if cur_val < best_val:
        best_val = cur_val
        best_A = list(A)
    
    n = 20
    max_passes = 100
    no_improve_passes = 0
    max_no_improve = 5
    
    # Precompute smooth numbers for candidate generation
    smooth_set = set()
    for a in range(0, 14):
        for b in range(0, 14):
            val = (2 ** a) * (3 ** b)
            if 1 <= val <= 10000:
                smooth_set.add(val)
    smooth_list = sorted(smooth_set)
    
    for _pass in range(max_passes):
        if time.time() > deadline:
            break
        
        improved_this_pass = False
        
        for i in range(n):
            if time.time() > deadline:
                break
            
            old = A[i]
            best_candidate = None
            best_candidate_val = cur_val
            
            # Generate candidate values for position i
            # 1. Neighbors: old ± d for small d
            for d in [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233]:
                for sign in [-1, 1]:
                    cand = old + sign * d
                    if 1 <= cand <= 10000 and cand != old and cand not in A:
                        A[i] = cand
                        v = _objective(A)
                        if v < best_candidate_val:
                            best_candidate_val = v
                            best_candidate = cand
            
            # 2. Geometric neighbors: old * (1±0.1), old * (1±0.2), etc.
            for factor in [0.8, 0.9, 1.1, 1.2, 0.5, 2.0, 0.75, 1.33]:
                cand = int(round(old * factor))
                if 1 <= cand <= 10000 and cand != old and cand not in A:
                    A[i] = cand
                    v = _objective(A)
                    if v < best_candidate_val:
                        best_candidate_val = v
                        best_candidate = cand
            
            # 3. Targeted values: try to move toward median or mean of A
            other_vals = [x for j, x in enumerate(A) if j != i]
            other_vals_sorted = sorted(other_vals)
            median_val = other_vals_sorted[n // 2 - 1]
            # Try values near the median
            for offset in [-5, -3, -1, 0, 1, 3, 5]:
                cand = median_val + offset
                if 1 <= cand <= 10000 and cand != old and cand not in A:
                    A[i] = cand
                    v = _objective(A)
                    if v < best_candidate_val:
                        best_candidate_val = v
                        best_candidate = cand
            
            # 4. Smooth number candidates near old value
            old_idx = bisect.bisect_left(smooth_list, old)
            for offset in [-5, -3, -2, -1, 1, 2, 3, 5]:
                idx = old_idx + offset
                if 0 <= idx < len(smooth_list):
                    cand = smooth_list[idx]
                    if 1 <= cand <= 10000 and cand != old and cand not in A:
                        A[i] = cand
                        v = _objective(A)
                        if v < best_candidate_val:
                            best_candidate_val = v
                            best_candidate = cand
            
            # 5. Random candidates
            for _ in range(5):
                cand = random.randint(1, 10000)
                if cand != old and cand not in A:
                    A[i] = cand
                    v = _objective(A)
                    if v < best_candidate_val:
                        best_candidate_val = v
                        best_candidate = cand
            
            # Apply best candidate if found
            if best_candidate is not None:
                A[i] = best_candidate
                cur_val = best_candidate_val
                improved_this_pass = True
                if cur_val < best_val:
                    best_val = cur_val
                    best_A = list(A)
            else:
                A[i] = old
        
        if not improved_this_pass:
            no_improve_passes += 1
            if no_improve_passes >= max_no_improve:
                break
        else:
            no_improve_passes = 0
    
    return best_val, best_A

def _local_search(A, deadline, best_val, best_A):
    """Perform local search with random replacements and swaps."""
    A = list(A)
    cur_val = _objective(A)
    if cur_val < best_val:
        best_val = cur_val
        best_A = list(A)
    
    n = 20
    max_iter = 50000
    no_improve = 0
    max_no_improve = 5000
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Choose a move type
        move_type = random.random()
        
        if move_type < 0.4:
            # Replace one element
            idx = random.randint(0, n - 1)
            old = A[idx]
            # Try values near old or random
            if random.random() < 0.5:
                new = old + random.randint(-50, 50)
            else:
                new = random.randint(1, 10000)
            new = max(1, min(10000, new))
            if new == old or new in A:
                continue
            A[idx] = new
            new_val = _objective(A)
            if new_val < cur_val:
                cur_val = new_val
                no_improve = 0
                if cur_val < best_val:
                    best_val = cur_val
                    best_A = list(A)
            else:
                # Accept with probability based on improvement (simulated annealing)
                delta = new_val - cur_val
                if delta <= 0 or random.random() < 0.1:
                    no_improve += 1
                else:
                    A[idx] = old
                    no_improve += 1
        elif move_type < 0.7:
            # Swap two elements
            i, j = random.sample(range(n), 2)
            A[i], A[j] = A[j], A[i]
            new_val = _objective(A)
            if new_val < cur_val:
                cur_val = new_val
                no_improve = 0
                if cur_val < best_val:
                    best_val = cur_val
                    best_A = list(A)
            else:
                A[i], A[j] = A[j], A[i]
                no_improve += 1
        else:
            # Perturb multiple elements
            for _ in range(random.randint(2, 5)):
                idx = random.randint(0, n - 1)
                new = A[idx] + random.randint(-100, 100)
                new = max(1, min(10000, new))
                if new not in A or new == A[idx]:
                    A[idx] = new
            
            new_val = _objective(A)
            if new_val < cur_val:
                cur_val = new_val
                no_improve = 0
                if cur_val < best_val:
                    best_val = cur_val
                    best_A = list(A)
            else:
                no_improve += 1
        
        if no_improve > max_no_improve:
            break
    
    return best_val, best_A

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    random.seed(42)
    
    # Generate initial candidates
    candidates = _generate_structured_candidates()
    
    # Evaluate all candidates
    best_val = float('inf')
    best_A = None
    for c in candidates:
        if not _is_valid(c):
            continue
        v = _objective(c)
        if v < best_val:
            best_val = v
            best_A = list(c)
    
    # Use greedy coordinate descent as primary optimization
    if time.time() < deadline - 5.0:
        best_val, best_A = _greedy_coordinate_descent(best_A, deadline - 5.0, best_val, best_A)
    
    # Then use local search for further refinement
    if time.time() < deadline - 5.0:
        best_val, best_A = _local_search(best_A, deadline - 5.0, best_val, best_A)
    
    # Multiple restarts from different starting points
    restarts = 0
    while time.time() < deadline - 2.0 and restarts < 5:
        # Generate a new random starting point
        new_start = sorted(random.sample(range(1, 10001), 20))
        v = _objective(new_start)
        if v < best_val:
            best_val = v
            best_A = list(new_start)
        
        # Use greedy coordinate descent from this point
        if time.time() < deadline - 2.0:
            best_val, best_A = _greedy_coordinate_descent(new_start, deadline - 2.0, best_val, best_A)
        
        restarts += 1
    
    # Final polish: try small perturbations on best_A
    if time.time() < deadline - 1.0:
        A = list(best_A)
        cur_val = best_val
        # Precompute smooth numbers
        smooth_set = set()
        for a in range(0, 14):
            for b in range(0, 14):
                val = (2 ** a) * (3 ** b)
                if 1 <= val <= 10000:
                    smooth_set.add(val)
        smooth_list = sorted(smooth_set)
        
        for _ in range(10000):
            if time.time() > deadline - 1.0:
                break
            idx = random.randint(0, 19)
            old = A[idx]
            # Try both small perturbations and smooth number replacements
            if random.random() < 0.6:
                new = old + random.randint(-10, 10)
                new = max(1, min(10000, new))
            else:
                # Try smooth number near old
                old_idx = bisect.bisect_left(smooth_list, old)
                if old_idx > 0:
                    offsets = random.sample(range(max(0, old_idx-5), min(len(smooth_list), old_idx+6)), 
                                           k=min(3, max(1, min(old_idx, len(smooth_list)-old_idx))))
                    new = smooth_list[old_idx + random.choice(offsets)]
                else:
                    new = smooth_list[random.randint(0, min(5, len(smooth_list)-1))]
            
            if new == old or new in A:
                continue
            A[idx] = new
            v = _objective(A)
            if v < cur_val:
                cur_val = v
                best_val = v
                best_A = list(A)
            else:
                A[idx] = old
    
    # Ensure validity
    if not _is_valid(best_A):
        best_A = list(range(1, 21))
    
    return best_A

# EVOLVE-BLOCK-START
# (construct is defined above)
# EVOLVE-BLOCK-END
