import random
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # Safety margin before 90s kill
    
    n = 128
    target_lo = 128
    target_hi = 256
    target_range = np.arange(target_lo, target_hi + 1)
    
    def get_counts(A):
        """Calculate representation counts for all target sums."""
        counts = np.zeros(target_hi + 1, dtype=np.int32)
        A_arr = np.array(A, dtype=np.int32)
        # For each pair (i,j) with i<=j, add to counts[A[i]+A[j]]
        # Efficient: iterate over all pairs
        for i in range(len(A_arr)):
            for j in range(i, len(A_arr)):
                s = A_arr[i] + A_arr[j]
                if target_lo <= s <= target_hi:
                    counts[s] += 1
        return counts
    
    def get_score(counts):
        """Return (uncovered_count, peak_count)."""
        target_counts = counts[target_lo:target_hi+1]
        uncovered = np.sum(target_counts == 0)
        peak = np.max(target_counts) if len(target_counts) > 0 else 0
        return (int(uncovered), int(peak))
    
    def get_variance(counts):
        """Return variance of target counts (for balance polish)."""
        target_counts = counts[target_lo:target_hi+1].astype(np.float64)
        if len(target_counts) == 0:
            return 0.0
        return float(np.var(target_counts))
    
    def evaluate_set(A):
        """Quick evaluation of set A."""
        counts = get_counts(A)
        return get_score(counts)
    
    def compute_delta_counts(A_set, x, y, counts):
        """
        Compute the delta in counts if we remove x and add y from A_set.
        Returns a dict of {sum: delta} for sums in target range.
        """
        deltas = {}
        # Remove x: for each z in A_set, z != x, sum = x+z loses 1
        # Also 2*x loses 1
        for z in A_set:
            if z == x:
                continue
            s = x + z
            if target_lo <= s <= target_hi:
                deltas[s] = deltas.get(s, 0) - 1
        s2x = 2 * x
        if target_lo <= s2x <= target_hi:
            deltas[s2x] = deltas.get(s2x, 0) - 1
        
        # Add y: for each z in A_set, z != y, sum = y+z gains 1
        # Also 2*y gains 1
        for z in A_set:
            if z == y:
                continue
            s = y + z
            if target_lo <= s <= target_hi:
                deltas[s] = deltas.get(s, 0) + 1
        s2y = 2 * y
        if target_lo <= s2y <= target_hi:
            deltas[s2y] = deltas.get(s2y, 0) + 1
        
        return deltas
    
    best_A = None
    best_score = (float('inf'), float('inf'))  # (uncovered, peak)
    best_var = float('inf')
    
    # Phase 1: Generate structured initial candidates
    candidates = []
    
    # 1. Even numbers in range
    candidates.append(list(range(64, 129, 2)))  # 32 elements
    
    # 2. Multiples of small integers
    candidates.append(list(range(0, 129, 2)))  # All even: 65 elements
    candidates.append(list(range(0, 129, 3)))  # Multiples of 3
    
    # 3. Random sets of various sizes
    for k in [30, 35, 40, 45, 50]:
        for _ in range(5):
            candidates.append(random.sample(range(n + 1), k))
    
    # 4. Greedy balanced construction: start from quasi-random spread
    def greedy_balanced_build(target_size=45):
        """Build a set greedily, preferring elements that reduce peak/variance."""
        A = set()
        # Start with a quasi-random spread
        base_elements = random.sample(range(0, n + 1), min(10, target_size))
        A.update(base_elements)
        
        counts = get_counts(list(A))
        
        while len(A) < target_size:
            if time.time() > deadline - 5:
                break
            best_y = None
            best_metric = None
            
            # Try candidate elements
            candidates_y = [y for y in range(n + 1) if y not in A]
            random.shuffle(candidates_y)
            # Limit candidates to speed up
            candidates_y = candidates_y[:30]
            
            for y in candidates_y:
                # Compute what happens if we add y
                temp_counts = counts.copy()
                for z in A:
                    s = y + z
                    if target_lo <= s <= target_hi:
                        temp_counts[s] += 1
                s2y = 2 * y
                if target_lo <= s2y <= target_hi:
                    temp_counts[s2y] += 1
                
                score = get_score(temp_counts)
                var = get_variance(temp_counts)
                
                # Prefer: lower uncovered, then lower peak, then lower variance
                metric = (score[0], score[1], var)
                if best_metric is None or metric < best_metric:
                    best_metric = metric
                    best_y = y
            
            if best_y is not None:
                A.add(best_y)
                counts = get_counts(list(A))
        
        return list(A)
    
    # Add greedy balanced builds
    for _ in range(3):
        candidates.append(greedy_balanced_build(45))
    
    # Evaluate all candidates and pick best few for local search
    scored_candidates = []
    for A in candidates:
        if time.time() > deadline - 5:
            break
        score = evaluate_set(A)
        counts = get_counts(A)
        var = get_variance(counts)
        scored_candidates.append((score, var, A))
    
    scored_candidates.sort(key=lambda x: (x[0][0], x[0][1], x[1]))
    
    # Phase 2: Enhanced local search with balance polish and SA escape
    def local_search_enhanced(initial_A, max_steps=5000):
        """Local search with swap moves, balance polish, and SA escape."""
        A = list(initial_A)
        A_set = set(A)
        counts = get_counts(A)
        curr_score = get_score(counts)
        curr_var = get_variance(counts)
        
        best_A_local = A[:]
        best_score_local = curr_score
        best_var_local = curr_var
        
        no_improve = 0
        max_no_improve = 300
        sa_temperature = 0.5
        sa_cooling = 0.995
        
        for step in range(max_steps):
            if time.time() > deadline - 3:
                break
            if no_improve > max_no_improve:
                # SA escape: accept a worse move with probability based on temperature
                if random.random() < sa_temperature:
                    # Do a random swap (even if worse)
                    x = random.choice(A)
                    y = random.randint(0, n)
                    attempts = 0
                    while y in A_set and attempts < 15:
                        y = random.randint(0, n)
                        attempts += 1
                    if y not in A_set:
                        old_A = A[:]
                        old_counts = counts.copy()
                        old_var = curr_var
                        
                        A.remove(x)
                        A_set.remove(x)
                        for z in A:
                            s = x + z
                            if target_lo <= s <= target_hi:
                                counts[s] -= 1
                        s2x = 2 * x
                        if target_lo <= s2x <= target_hi:
                            counts[s2x] -= 1
                        
                        A.append(y)
                        A_set.add(y)
                        for z in A:
                            if z == y:
                                continue
                            s = y + z
                            if target_lo <= s <= target_hi:
                                counts[s] += 1
                        s2y = 2 * y
                        if target_lo <= s2y <= target_hi:
                            counts[s2y] += 1
                        
                        new_score = get_score(counts)
                        new_var = get_variance(counts)
                        
                        # Accept (SA escape)
                        curr_score = new_score
                        curr_var = new_var
                        no_improve = 0
                        sa_temperature *= sa_cooling
                else:
                    break
                continue
            
            # Choose move type: balance polish vs peak reduction
            # Balance polish: if peak is not improving, try to reduce variance
            target_counts = counts[target_lo:target_hi+1]
            peak_val = np.max(target_counts)
            mean_val = np.mean(target_counts)
            
            # Find high-count and low-count sums
            high_sums = np.where(target_counts > mean_val + 0.5)[0] + target_lo
            low_sums = np.where(target_counts < mean_val - 0.5)[0] + target_lo
            
            # Try a swap that moves weight from high to low
            if len(high_sums) > 0 and len(low_sums) > 0 and random.random() < 0.5:
                # Balance polish move
                x = random.choice(A)
                # Try to find y such that removing x and adding y reduces variance
                best_y = None
                best_delta_var = 0
                
                candidates_y = [y for y in range(n + 1) if y not in A_set]
                random.shuffle(candidates_y)
                candidates_y = candidates_y[:20]
                
                for y in candidates_y:
                    # Quick estimate: compute delta variance
                    old_counts_copy = counts.copy()
                    
                    # Remove x
                    for z in A_set:
                        if z == x:
                            continue
                        s = x + z
                        if target_lo <= s <= target_hi:
                            old_counts_copy[s] -= 1
                    s2x = 2 * x
                    if target_lo <= s2x <= target_hi:
                        old_counts_copy[s2x] -= 1
                    
                    # Add y
                    for z in A_set:
                        if z == y:
                            continue
                        s = y + z
                        if target_lo <= s <= target_hi:
                            old_counts_copy[s] += 1
                    s2y = 2 * y
                    if target_lo <= s2y <= target_hi:
                        old_counts_copy[s2y] += 1
                    
                    new_var = get_variance(old_counts_copy)
                    delta_var = curr_var - new_var
                    
                    if delta_var > best_delta_var:
                        best_delta_var = delta_var
                        best_y = y
                
                if best_y is not None and best_delta_var > 0:
                    y = best_y
                else:
                    y = random.choice(candidates_y) if candidates_y else None
                    if y is None or y in A_set:
                        continue
            else:
                # Peak reduction move
                x = random.choice(A)
                y = random.randint(0, n)
                attempts = 0
                while y in A_set and attempts < 15:
                    y = random.randint(0, n)
                    attempts += 1
                if y in A_set:
                    continue
            
            # Try the swap
            old_A = A[:]
            old_counts = counts.copy()
            old_var = curr_var
            
            # Remove x
            A.remove(x)
            A_set.remove(x)
            for z in A:
                s = x + z
                if target_lo <= s <= target_hi:
                    counts[s] -= 1
            s2x = 2 * x
            if target_lo <= s2x <= target_hi:
                counts[s2x] -= 1
            
            # Add y
            A.append(y)
            A_set.add(y)
            for z in A:
                if z == y:
                    continue
                s = y + z
                if target_lo <= s <= target_hi:
                    counts[s] += 1
            s2y = 2 * y
            if target_lo <= s2y <= target_hi:
                counts[s2y] += 1
            
            new_score = get_score(counts)
            new_var = get_variance(counts)
            
            # Accept if better on primary metric, or if same primary but better variance
            if (new_score < curr_score) or \
               (new_score == curr_score and new_var < curr_var):
                curr_score = new_score
                curr_var = new_var
                no_improve = 0
                if (curr_score < best_score_local) or \
                   (curr_score == best_score_local and curr_var < best_var_local):
                    best_score_local = curr_score
                    best_var_local = curr_var
                    best_A_local = A[:]
            else:
                # Rollback
                A = old_A
                A_set = set(A)
                counts = old_counts
                curr_var = old_var
                no_improve += 1
                sa_temperature *= sa_cooling
                
        return best_A_local, best_score_local, best_var_local
    
    # Run enhanced local search on top candidates
    for score, var, A in scored_candidates[:8]:
        if time.time() > deadline - 5:
            break
        
        A_res, score_res, var_res = local_search_enhanced(A, max_steps=4000)
        
        if score_res < best_score or (score_res == best_score and var_res < best_var):
            best_score = score_res
            best_var = var_res
            best_A = A_res
        
        # Early exit if we found a good solution
        if best_score[0] == 0 and best_score[1] <= 3:
            break
    
    # If still not good enough, try more aggressive search with different set sizes
    if best_score[0] > 0 or best_score[1] > 4:
        if time.time() < deadline - 10:
            # Try various set sizes with enhanced local search
            for k in [40, 45, 50, 55, 60]:
                if time.time() > deadline - 5:
                    break
                for _ in range(2):
                    if time.time() > deadline - 3:
                        break
                    init_A = random.sample(range(n + 1), k)
                    A_res, score_res, var_res = local_search_enhanced(init_A, max_steps=2500)
                    if score_res < best_score or (score_res == best_score and var_res < best_var):
                        best_score = score_res
                        best_var = var_res
                        best_A = A_res
                    if best_score[0] == 0 and best_score[1] <= 3:
                        break
                if best_score[0] == 0 and best_score[1] <= 3:
                    break
    
    if best_A is None:
        # Fallback
        return list(range(64, 129))
    
    return best_A
