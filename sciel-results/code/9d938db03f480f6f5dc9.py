import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    N = 128
    min_sum = 128
    max_sum = 256
    target_sums = list(range(min_sum, max_sum + 1))
    num_targets = len(target_sums)  # 129
    
    def get_counts_fast(A):
        """Returns array of counts for sums in [0, 256] using numpy convolution-like approach"""
        A_arr = np.array(sorted(A), dtype=np.int64)
        k = len(A_arr)
        if k == 0:
            return np.zeros(max_sum + 1, dtype=int)
        # Compute all pair sums (unordered, with self-pairs)
        if k <= 50:
            A_i = A_arr[:, None]
            A_j = A_arr[None, :]
            sums = A_i + A_j
            mask = np.triu(np.ones((k, k), dtype=bool), k=0)
            valid_sums = sums[mask]
            valid_sums = valid_sums[valid_sums <= max_sum]
            if len(valid_sums) > 0:
                counts = np.bincount(valid_sums, minlength=max_sum + 1)
            else:
                counts = np.zeros(max_sum + 1, dtype=int)
            return counts
        else:
            counts = np.zeros(max_sum + 1, dtype=int)
            for i in range(k):
                ai = A_arr[i]
                s = ai + ai
                if s <= max_sum:
                    counts[s] += 1
                for j in range(i + 1, k):
                    aj = A_arr[j]
                    s = ai + aj
                    if s <= max_sum:
                        counts[s] += 1
            return counts
    
    def get_cost(A):
        counts = get_counts_fast(A)
        target_counts = counts[min_sum:max_sum + 1]
        if np.any(target_counts == 0):
            uncovered = np.sum(target_counts == 0)
            return 1000 + uncovered
        return int(np.max(target_counts))
    
    def get_cost_detailed(A):
        """Returns (max_mult, uncovered_count, counts_array)"""
        counts = get_counts_fast(A)
        target_counts = counts[min_sum:max_sum + 1]
        uncovered = np.sum(target_counts == 0)
        if uncovered > 0:
            return (1000 + uncovered, uncovered, counts)
        return (int(np.max(target_counts)), 0, counts)
    
    def smooth_cost(counts, t=2.0):
        """Smooth surrogate: log-sum-exp of target counts, penalizes peak"""
        target_counts = counts[min_sum:max_sum + 1].astype(float)
        if np.any(target_counts == 0):
            return 1000.0 + np.sum(target_counts == 0)
        m = np.max(target_counts)
        # Log-sum-exp for numerical stability
        lse = m + np.log(np.sum(np.exp(target_counts - m)))
        return lse
    
    # Direct construction: try several structured initial sets
    best_A = None
    best_cost = float('inf')
    
    candidates = []
    
    for k in [16, 17, 18, 19, 20]:
        base_points = np.linspace(0, 128, k, dtype=int)
        base_set = set(int(x) for x in base_points)
        if len(base_set) < k:
            base_set = set()
            used = set()
            for i in range(k):
                x = round(i * 128 / (k - 1))
                while x in used:
                    x += 1
                    if x > 128:
                        x = 0
                base_set.add(x)
                used.add(x)
            base_set = set(x for x in base_set if 0 <= x <= 128)
        candidates.append(set(base_set))
        
        for offset in range(-3, 4):
            perturbed = set()
            for i in range(k):
                x = int(round(i * 128 / (k - 1))) + offset
                if 0 <= x <= 128:
                    perturbed.add(x)
            if len(perturbed) >= k - 2:
                candidates.append(perturbed)
    
    for a in range(5, 15):
        for b in range(113, 128):
            if b > a + 1:
                clus = set(range(0, a + 1)) | set(range(b, 129))
                candidates.append(clus)
    
    for m in range(6, 12):
        small = set(range(0, m))
        large = set(range(128 - m + 1, 129))
        mid = set(range(60 - m // 2, 60 + m // 2 + 1))
        mixed = small | large | mid
        if mixed <= set(range(129)):
            candidates.append(mixed)
    
    # Evaluate all candidates
    for cand in candidates:
        cand = set(x for x in cand if 0 <= x <= 128)
        if len(cand) == 0:
            continue
        cost, uncovered, _ = get_cost_detailed(cand)
        if cost < best_cost:
            best_cost = cost
            best_A = set(cand)
    
    # Efficient local search with incremental count updates
    if best_A is not None:
        A = set(best_A)
        current_cost, _, current_counts = get_cost_detailed(A)
        
        # Build index for fast lookups
        A_list = sorted(A)
        A_set = set(A)
        outside = list(set(range(129)) - A_set)
        
        def update_counts_add(A_arr, x, counts):
            """Add element x to A_arr, update counts incrementally"""
            k = len(A_arr)
            # Add x+x
            s = x + x
            if s <= max_sum:
                counts[s] += 1
            # Add x + a for each a in A_arr
            for a in A_arr:
                s = x + a
                if s <= max_sum:
                    counts[s] += 1
            return counts
        
        def update_counts_remove(A_arr, x, counts):
            """Remove element x from A_arr, update counts incrementally"""
            # Remove x+x
            s = x + x
            if s <= max_sum:
                counts[s] -= 1
            # Remove x + a for each a in A_arr (excluding x itself)
            for a in A_arr:
                if a != x:
                    s = x + a
                    if s <= max_sum:
                        counts[s] -= 1
            return counts
        
        def swap_update(A_arr, x_out, x_in, counts):
            """Remove x_out, add x_in, return updated counts"""
            # Remove x_out
            s = x_out + x_out
            if s <= max_sum:
                counts[s] -= 1
            for a in A_arr:
                if a != x_out:
                    s = x_out + a
                    if s <= max_sum:
                        counts[s] -= 1
            # Add x_in
            s = x_in + x_in
            if s <= max_sum:
                counts[s] += 1
            for a in A_arr:
                if a != x_in:
                    s = x_in + a
                    if s <= max_sum:
                        counts[s] += 1
            return counts
        
        rng = random.Random(12345)
        A_arr = np.array(A_list, dtype=np.int64)
        
        # Congestion-aware local search
        for iteration in range(50000):
            if time.time() > deadline - 5.0:
                break
            
            move_type = rng.random()
            
            if move_type < 0.7 and len(outside) > 0:
                # Swap with congestion awareness
                x_in = rng.choice(list(A_arr))
                x_out = rng.choice(outside)
                
                # Compute new counts
                new_counts = current_counts.copy()
                new_counts = swap_update(A_arr, x_in, x_out, new_counts)
                
                # Check coverage and compute smooth cost
                target_new = new_counts[min_sum:max_sum + 1]
                if np.any(target_new == 0):
                    continue  # Revert, coverage lost
                
                new_smooth = smooth_cost(new_counts)
                old_smooth = smooth_cost(current_counts)
                
                # Accept if better or with small probability (simulated annealing)
                if new_smooth < old_smooth or (iteration < 5000 and rng.random() < 0.1):
                    # Accept the swap
                    idx = np.where(A_arr == x_in)[0][0]
                    A_arr[idx] = x_out
                    A_arr = np.sort(A_arr)
                    A_set.remove(x_in)
                    A_set.add(x_out)
                    outside.remove(x_out)
                    outside.append(x_in)
                    current_counts = new_counts
                    current_cost, _, _ = get_cost_detailed(set(A_arr))
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_A = set(A_arr)
                else:
                    continue  # Revert (counts not modified in place)
            
            elif move_type < 0.85 and len(A_arr) > 14:
                # Removal
                idx = rng.randint(0, len(A_arr) - 1)
                x = A_arr[idx]
                
                new_counts = current_counts.copy()
                new_counts = update_counts_remove(A_arr, x, new_counts)
                
                target_new = new_counts[min_sum:max_sum + 1]
                if np.any(target_new == 0):
                    continue
                
                new_smooth = smooth_cost(new_counts)
                old_smooth = smooth_cost(current_counts)
                
                if new_smooth < old_smooth:
                    # Accept removal
                    A_arr = np.delete(A_arr, idx)
                    A_set.remove(x)
                    outside.append(x)
                    current_counts = new_counts
                    current_cost, _, _ = get_cost_detailed(set(A_arr))
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_A = set(A_arr)
            
            elif move_type < 1.0 and len(A_arr) < 30:
                # Addition
                x = rng.choice(outside)
                
                new_counts = current_counts.copy()
                new_counts = update_counts_add(A_arr, x, new_counts)
                
                target_new = new_counts[min_sum:max_sum + 1]
                new_smooth = smooth_cost(new_counts)
                old_smooth = smooth_cost(current_counts)
                
                if new_smooth < old_smooth:
                    # Accept addition
                    A_arr = np.append(A_arr, x)
                    A_arr = np.sort(A_arr)
                    A_set.add(x)
                    outside.remove(x)
                    current_counts = new_counts
                    current_cost, _, _ = get_cost_detailed(set(A_arr))
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_A = set(A_arr)
            
            # Early exit
            if best_cost <= 2:
                break
            if best_cost == 3 and iteration > 10000:
                # Try harder but with diminishing returns
                pass
    
    # Final verification and cleanup
    if best_A is None:
        best_A = set(range(0, 129, 8))
    
    return list(best_A)
