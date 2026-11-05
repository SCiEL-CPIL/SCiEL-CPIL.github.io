import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin (evaluator kills at 90s)
    
    N = 128
    min_sum = 128
    max_sum = 256
    target_sums = range(min_sum, max_sum + 1)
    target_len = max_sum - min_sum + 1  # 129
    
    def compute_counts_fast(A_list):
        """Fast computation of target counts using vectorized approach."""
        if len(A_list) == 0:
            return np.zeros(target_len, dtype=np.int32)
        
        A_arr = np.array(A_list, dtype=np.int32)
        sums = np.add.outer(A_arr, A_arr)
        flat = sums.ravel()
        
        # Only keep sums in target range
        mask = (flat >= min_sum) & (flat <= max_sum)
        if not np.any(mask):
            return np.zeros(target_len, dtype=np.int32)
        
        valid = flat[mask] - min_sum
        f = np.bincount(valid, minlength=target_len).astype(np.int32)
        
        # Compute g(s) for unordered pair correction
        A_set = set(A_list)
        g = np.zeros(target_len, dtype=np.int32)
        for a in A_list:
            s = 2 * a
            if min_sum <= s <= max_sum:
                g[s - min_sum] += 1
        
        counts = (f + g) // 2
        return counts
    
    def get_cost(tc):
        """Compute cost from target counts."""
        if np.any(tc == 0):
            return 1000 + int(np.sum(tc == 0))
        return int(np.max(tc))
    
    def smooth_cost(tc):
        """Smooth surrogate using log-sum-exp."""
        tc_safe = np.maximum(tc, 1e-6)
        max_tc = np.max(tc_safe)
        lse = max_tc + np.log(np.sum(np.exp(tc_safe - max_tc)))
        uncovered = np.sum(tc == 0)
        return lse + 100.0 * uncovered
    
    # Better initial set: structured approach
    random.seed(42)
    
    # Start with a spread set that likely covers many sums
    # Use a combination of small and large values
    initial_candidates = list(range(0, N + 1, 4))  # Every 4th number
    # Add some key values
    key_values = [0, 16, 32, 48, 64, 80, 96, 112, 128, 8, 24, 40, 56, 72, 88, 104, 120]
    for v in key_values:
        if v not in initial_candidates:
            initial_candidates.append(v)
    
    # Shuffle and take a good subset
    random.shuffle(initial_candidates)
    A_list = initial_candidates[:24]
    A_set = set(A_list)
    
    # Ensure we have a reasonable starting set
    tc = compute_counts_fast(A_list)
    best_cost = get_cost(tc)
    best_A = list(A_list)
    
    # Phase 1: Greedy coverage
    max_phase1_time = 10.0
    phase1_deadline = start_time + max_phase1_time
    
    while time.time() < phase1_deadline:
        tc = compute_counts_fast(A_list)
        uncovered_mask = (tc == 0)
        
        if not np.any(uncovered_mask):
            break
        
        # Find best element to add
        best_x = None
        best_gain = -1
        
        for x in range(0, N + 1):
            if x in A_set:
                continue
            
            # Compute gain: how many uncovered sums get covered
            gain = 0
            for y in A_list:
                s = x + y
                if min_sum <= s <= max_sum and tc[s - min_sum] == 0:
                    gain += 1
            s = 2 * x
            if min_sum <= s <= max_sum and tc[s - min_sum] == 0:
                gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_x = x
        
        if best_x is None or best_gain == 0:
            break
        
        A_list.append(best_x)
        A_set.add(best_x)
        
        # Prune if too large
        if len(A_list) > 28:
            for _ in range(5):
                if time.time() >= phase1_deadline:
                    break
                for idx in range(len(A_list)):
                    x = A_list[idx]
                    A_list.remove(x)
                    A_set.remove(x)
                    tc = compute_counts_fast(A_list)
                    if np.any(tc == 0):
                        A_list.append(x)
                        A_set.add(x)
                    else:
                        break
    
    # Phase 2: Simulated annealing with incremental updates
    tc = compute_counts_fast(A_list)
    current_cost = get_cost(tc)
    current_smooth = smooth_cost(tc)
    
    if current_cost < best_cost:
        best_cost = current_cost
        best_A = list(A_list)
    
    T_start = 3.0
    T_end = 0.05
    iteration = 0
    stuck_count = 0
    max_stuck = 300
    
    all_elements = list(range(0, N + 1))
    
    while time.time() < deadline:
        iteration += 1
        progress = min(iteration / 20000.0, 1.0)
        T = T_start * (T_end / T_start) ** progress
        
        r = random.random()
        
        if r < 0.65 and len(A_list) > 0:
            # Swap
            idx = random.randint(0, len(A_list) - 1)
            x_in = A_list[idx]
            x_out = random.choice([x for x in all_elements if x not in A_set])
            
            # Quick delta estimate before full recompute
            A_list[idx] = x_out
            A_set.remove(x_in)
            A_set.add(x_out)
            
            new_tc = compute_counts_fast(A_list)
            new_cost = get_cost(new_tc)
            new_smooth = smooth_cost(new_tc)
            
            delta = new_smooth - current_smooth
            if delta < 0 or random.random() < np.exp(-delta / max(T, 1e-6)):
                tc = new_tc
                current_cost = new_cost
                current_smooth = new_smooth
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_A = list(A_list)
                    stuck_count = 0
                else:
                    stuck_count += 1
            else:
                A_list[idx] = x_in
                A_set.remove(x_out)
                A_set.add(x_in)
                stuck_count += 1
                
        elif r < 0.80 and len(A_list) > 16:
            # Remove
            idx = random.randint(0, len(A_list) - 1)
            x = A_list[idx]
            A_list.pop(idx)
            A_set.remove(x)
            
            new_tc = compute_counts_fast(A_list)
            new_cost = get_cost(new_tc)
            new_smooth = smooth_cost(new_tc)
            
            delta = new_smooth - current_smooth
            if delta < 0 or random.random() < np.exp(-delta / max(T, 1e-6)):
                tc = new_tc
                current_cost = new_cost
                current_smooth = new_smooth
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_A = list(A_list)
                    stuck_count = 0
                else:
                    stuck_count += 1
            else:
                A_list.append(x)
                A_set.add(x)
                stuck_count += 1
                
        elif r < 0.92 and len(A_list) < 32:
            # Add
            x = random.choice([x for x in all_elements if x not in A_set])
            A_list.append(x)
            A_set.add(x)
            
            new_tc = compute_counts_fast(A_list)
            new_cost = get_cost(new_tc)
            new_smooth = smooth_cost(new_tc)
            
            delta = new_smooth - current_smooth
            if delta < 0 or random.random() < np.exp(-delta / max(T, 1e-6)):
                tc = new_tc
                current_cost = new_cost
                current_smooth = new_smooth
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_A = list(A_list)
                    stuck_count = 0
                else:
                    stuck_count += 1
            else:
                A_list.remove(x)
                A_set.remove(x)
                stuck_count += 1
                
        else:
            # Double swap
            if len(A_list) >= 2:
                idx1, idx2 = random.sample(range(len(A_list)), 2)
                x1 = A_list[idx1]
                x2 = A_list[idx2]
                
                available = [x for x in all_elements if x not in A_set]
                if len(available) >= 2:
                    x_out1, x_out2 = random.sample(available, 2)
                    
                    A_list[idx1] = x_out1
                    A_list[idx2] = x_out2
                    A_set.remove(x1)
                    A_set.remove(x2)
                    A_set.add(x_out1)
                    A_set.add(x_out2)
                    
                    new_tc = compute_counts_fast(A_list)
                    new_cost = get_cost(new_tc)
                    new_smooth = smooth_cost(new_tc)
                    
                    delta = new_smooth - current_smooth
                    if delta < 0 or random.random() < np.exp(-delta / max(T, 1e-6)):
                        tc = new_tc
                        current_cost = new_cost
                        current_smooth = new_smooth
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_A = list(A_list)
                            stuck_count = 0
                        else:
                            stuck_count += 1
                    else:
                        A_list[idx1] = x1
                        A_list[idx2] = x2
                        A_set.remove(x_out1)
                        A_set.remove(x_out2)
                        A_set.add(x1)
                        A_set.add(x2)
                        stuck_count += 1
                else:
                    stuck_count += 1
            else:
                stuck_count += 1
        
        # Restart if stuck
        if stuck_count > max_stuck:
            A_list = list(best_A)
            A_set = set(A_list)
            tc = compute_counts_fast(A_list)
            current_cost = get_cost(tc)
            current_smooth = smooth_cost(tc)
            stuck_count = 0
        
        if best_cost <= 2:
            break
    
    # Final polish: aggressive local search
    if time.time() < deadline - 3:
        polish_A = list(best_A)
        polish_set = set(polish_A)
        polish_tc = compute_counts_fast(polish_A)
        polish_cost = get_cost(polish_tc)
        
        improved = True
        max_polish_iters = 50
        polish_iter = 0
        
        while improved and time.time() < deadline - 1 and polish_iter < max_polish_iters:
            improved = False
            polish_iter += 1
            
            # Try removing each element
            for idx in range(len(polish_A)):
                x = polish_A[idx]
                polish_A.remove(x)
                polish_set.remove(x)
                new_tc = compute_counts_fast(polish_A)
                new_cost = get_cost(new_tc)
                if new_cost < polish_cost:
                    polish_cost = new_cost
                    polish_tc = new_tc
                    improved = True
                    break
                else:
                    polish_A.append(x)
                    polish_set.add(x)
            
            if not improved:
                # Try swaps
                for idx in range(len(polish_A)):
                    x_in = polish_A[idx]
                    for x_out in all_elements:
                        if x_out in polish_set:
                            continue
                        polish_A[idx] = x_out
                        polish_set.remove(x_in)
                        polish_set.add(x_out)
                        new_tc = compute_counts_fast(polish_A)
                        new_cost = get_cost(new_tc)
                        if new_cost < polish_cost:
                            polish_cost = new_cost
                            polish_tc = new_tc
                            improved = True
                            break
                        else:
                            polish_A[idx] = x_in
                            polish_set.remove(x_out)
                            polish_set.add(x_in)
                    if improved:
                        break
            
            if polish_cost < best_cost:
                best_cost = polish_cost
                best_A = list(polish_A)
    
    return list(best_A)
