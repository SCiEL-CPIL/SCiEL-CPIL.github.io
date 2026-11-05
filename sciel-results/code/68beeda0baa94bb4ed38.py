import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill

    n = 128
    min_sum = 128
    max_sum = 256
    target_range = max_sum - min_sum + 1  # 129

    # Precompute: for each element x in [0,128], which target sums does it contribute to?
    # If x is in A, then for each y in A, x+y contributes to sum s=x+y.
    # Better: precompute the pair-sum incidence matrix.
    # pair_incidence[x, s] = 1 if there exists y in [0,128] such that x+y=s and y in [0,128]
    # But we need unordered pairs with repetition.
    
    # Let's use a different approach: 
    # For a binary vector v of length 129 (v[x]=1 if x in A),
    # the number of unordered pairs (a,b) with a+b=s, a<=b, is:
    #   count[s] = sum_{a=0}^{min(s,128)} v[a]*v[s-a] / 2 + (v[s/2] if s even else 0)/2
    # Actually simpler: ordered_count[s] = sum_a v[a]*v[s-a]
    # unordered_count[s] = (ordered_count[s] + diag[s]) / 2
    # where diag[s] = v[s//2] if s%2==0 else 0
    
    # Precompute for each x, the list of (y, s) pairs where y >= x and x+y in [128,256]
    # This lets us do incremental updates.
    
    # For speed, let's precompute the full convolution kernel approach but vectorized.
    # Actually, let's just use a fast method:
    
    # Precompute: for each target sum s in [128,256], the list of pairs (a,b) with a<=b, a+b=s, a,b in [0,128]
    # Then count[s] = number of such pairs where both a and b are in A.
    
    # Even faster: use the fact that ordered_conv = np.convolve(v, v) is fast in numpy.
    # Let's benchmark: np.convolve on length-129 arrays is very fast.
    
    def evaluate(v):
        """v is a binary numpy array of length 129. Returns (peak, valid, counts_array)."""
        if not np.any(v):
            return float('inf'), False, None
        
        conv = np.convolve(v, v, mode='full')
        # conv[s] for s in [0, 256]
        conv_target = conv[min_sum:max_sum+1].astype(np.float64)
        
        # Diagonal correction
        diag = np.zeros(target_range)
        for idx in range(target_range):
            s = min_sum + idx
            if s % 2 == 0:
                mid = s // 2
                if 0 <= mid <= n:
                    diag[idx] = v[mid]
        
        counts = (conv_target + diag) / 2.0
        counts = np.round(counts).astype(np.int32)
        
        if np.any(counts == 0):
            return float('inf'), False, counts
        
        peak = int(np.max(counts))
        return peak, True, counts
    
    # Build initial solution: two intervals
    best_initial_peak = float('inf')
    best_initial_A = None
    best_initial_v = None
    best_initial_counts = None
    
    # Try structured initializations more efficiently
    for a in range(0, 65, 1):
        for b in range(64, 129, 1):
            if a >= b:
                continue
            v = np.zeros(129, dtype=np.float64)
            v[0:a+1] = 1.0
            v[b:129] = 1.0
            peak, valid, counts = evaluate(v)
            if valid and peak < best_initial_peak:
                best_initial_peak = peak
                best_initial_A = set(range(0, a+1)) | set(range(b, 129))
                best_initial_v = v.copy()
                best_initial_counts = counts.copy()
    
    # Also try single interval
    for a in range(0, 65, 1):
        for b in range(a, 129, 1):
            v = np.zeros(129, dtype=np.float64)
            v[a:b+1] = 1.0
            peak, valid, counts = evaluate(v)
            if valid and peak < best_initial_peak:
                best_initial_peak = peak
                best_initial_A = set(range(a, b+1))
                best_initial_v = v.copy()
                best_initial_counts = counts.copy()
    
    if best_initial_A is None:
        # Fallback
        best_initial_A = set(range(64, 129))
        best_initial_v = np.zeros(129, dtype=np.float64)
        best_initial_v[64:129] = 1.0
        best_initial_peak, _, best_initial_counts = evaluate(best_initial_v)
    
    current_A = best_initial_A.copy()
    current_v = best_initial_v.copy()
    current_peak = best_initial_peak
    current_counts = best_initial_counts.copy()
    best_A = current_A.copy()
    best_v = current_v.copy()
    best_peak = current_peak
    best_counts = current_counts.copy()
    
    random.seed(42)
    np.random.seed(42)
    
    # Balanced local search: target the max and min counts
    T = 5.0
    T_min = 0.1
    cooling_rate = 0.9998
    max_iters = 1000000
    iter_count = 0
    no_improve = 0
    max_no_improve = 10000
    
    # Precompute for fast move evaluation:
    # For each x in [0,128], precompute the vector of contributions to each target sum.
    # If we add x to A, the change in counts is: for each y already in A, 
    #   if x <= y: count[x+y] increases by 1 (unordered pair (x,y))
    #   if x > y: count[x+y] increases by 1 (same, unordered)
    # Plus if x+x is in range: count[2x] increases by 1
    # Actually, when adding x, for each y in A:
    #   The new unordered pairs are (min(x,y), max(x,y)) for each y in A, plus (x,x) if not already
    #   But (x,x) is only added once.
    #   For y != x: pair (min(x,y), max(x,y)) is new, count increases by 1
    #   For y == x: pair (x,x) is new, count increases by 1
    # So total increase: for each y in A, count[x+y] += 1, and if x not in A, also count[2x] += 1? 
    # No wait: when we add x, the new pairs involving x are:
    #   (x, y) for each y in A (including y=x if x was not in A)
    #   But (x,y) and (y,x) are the same unordered pair.
    #   So for each y in A, we get one new unordered pair (x,y) if x!=y, or (x,x) if x==y.
    #   The sum is x+y.
    # So: delta_count[s] += 1 for each y in A where x+y=s.
    # That is: delta_count[x+y] += 1 for each y in A.
    
    # Precompute contribution vectors for each x
    # contrib[x] is a vector of length target_range
    # contrib[x][idx] = 1 if there exists y such that x+y = min_sum+idx and y in [0,128]
    # But we need to know how many y in A satisfy this, which depends on A.
    # So we can't fully precompute; we need to compute on the fly.
    
    # Alternative: use the convolution approach but make it faster.
    # np.convolve on 129-element arrays should be ~microseconds.
    
    # Let's just use the evaluate function and make the search smarter.
    
    while time.time() < deadline and iter_count < max_iters and no_improve < max_no_improve:
        iter_count += 1
        
        # Choose move type based on current state
        if current_counts is not None:
            max_count = np.max(current_counts)
            min_count = np.min(current_counts)
            # Sums that are most overloaded
            overloaded_sums = np.where(current_counts >= max_count)[0]
            # Sums that are most underloaded (but still > 0 for validity)
            underloaded_sums = np.where(current_counts == min_count)[0]
        else:
            overloaded_sums = np.arange(target_range)
            underloaded_sums = np.arange(target_range)
        
        move_type = random.random()
        
        if move_type < 0.35:
            # Swap: remove one element, add one
            if len(current_A) <= 5:
                continue
            # Try to remove an element that contributes to overloaded sums
            # and add one that contributes to underloaded sums
            remove_candidates = list(current_A)
            add_candidates = [x for x in range(129) if x not in current_A]
            if not add_candidates:
                continue
            
            # Score removal candidates by how much they contribute to overloaded sums
            best_remove = None
            best_remove_score = -float('inf')
            for x in remove_candidates:
                # How many overloaded sums does x contribute to?
                score = 0
                for s in overloaded_sums:
                    target_s = min_sum + s
                    y = target_s - x
                    if 0 <= y <= n and y in current_A:
                        score += 1
                if score > best_remove_score:
                    best_remove_score = score
                    best_remove = x
            
            # Score addition candidates by how much they contribute to underloaded sums
            best_add = None
            best_add_score = -float('inf')
            for x in add_candidates:
                score = 0
                for s in underloaded_sums:
                    target_s = min_sum + s
                    y = target_s - x
                    if 0 <= y <= n and y in current_A:
                        score += 1
                if score > best_add_score:
                    best_add_score = score
                    best_add = x
            
            if best_remove is None or best_add is None:
                best_remove = random.choice(remove_candidates)
                best_add = random.choice(add_candidates)
            
            new_A = current_A.copy()
            new_A.discard(best_remove)
            new_A.add(best_add)
            
        elif move_type < 0.65:
            # Add an element that helps underloaded sums
            add_candidates = [x for x in range(129) if x not in current_A]
            if not add_candidates:
                continue
            
            best_add = None
            best_add_score = -float('inf')
            for x in add_candidates:
                score = 0
                for s in underloaded_sums:
                    target_s = min_sum + s
                    y = target_s - x
                    if 0 <= y <= n and y in current_A:
                        score += 1
                if score > best_add_score:
                    best_add_score = score
                    best_add = x
            
            if best_add is None:
                best_add = random.choice(add_candidates)
            
            new_A = current_A.copy()
            new_A.add(best_add)
            
        else:
            # Remove an element that hurts overloaded sums
            if len(current_A) <= 5:
                continue
            remove_candidates = list(current_A)
            
            best_remove = None
            best_remove_score = float('inf')
            for x in remove_candidates:
                # How many overloaded sums does x contribute to? (we want to minimize this)
                score = 0
                for s in overloaded_sums:
                    target_s = min_sum + s
                    y = target_s - x
                    if 0 <= y <= n and y in current_A:
                        score += 1
                if score < best_remove_score:
                    best_remove_score = score
                    best_remove = x
            
            if best_remove is None:
                best_remove = random.choice(remove_candidates)
            
            new_A = current_A.copy()
            new_A.discard(best_remove)
        
        # Evaluate new solution
        new_v = np.zeros(129, dtype=np.float64)
        for x in new_A:
            new_v[x] = 1.0
        new_peak, new_valid, new_counts = evaluate(new_v)
        
        if not new_valid:
            no_improve += 1
            T *= cooling_rate
            if T < T_min:
                T = T_min
            continue
        
        # Acceptance criterion
        if new_peak < current_peak:
            current_A = new_A
            current_v = new_v
            current_peak = new_peak
            current_counts = new_counts
            no_improve = 0
            if current_peak < best_peak:
                best_peak = current_peak
                best_A = current_A.copy()
                best_v = current_v.copy()
                best_counts = current_counts.copy()
        else:
            # Accept with probability exp(-(new_peak - current_peak) / T)
            delta = new_peak - current_peak
            if T > T_min and random.random() < np.exp(-delta / T):
                current_A = new_A
                current_v = new_v
                current_peak = new_peak
                current_counts = new_counts
            no_improve += 1
        
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
    
    # Return the best solution found
    if best_A is None or len(best_A) == 0:
        return list(range(64, 129))
    
    return sorted(best_A)
