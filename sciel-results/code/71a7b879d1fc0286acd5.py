import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin (evaluator kills at 90s)
    
    N = 128
    min_sum = 128
    max_sum = 256
    target_sums = range(min_sum, max_sum + 1)
    target_len = max_sum - min_sum + 1  # 129
    
    # Precompute pair indices for fast count updates
    # For a set A of size k, the number of unordered pairs (with repetition) is k*(k+1)/2
    # We need to efficiently compute the sum counts
    
    def compute_counts(A_list):
        """Compute counts of all sums from pairs in A_list using convolution (fast)."""
        if not A_list:
            return np.zeros(max_sum + 1, dtype=np.int32)
        
        # Binary indicator vector: v[a] = 1 if a in A_list
        v = np.zeros(N + 1, dtype=np.int64)
        for a in A_list:
            v[a] = 1
        
        # Convolution: (v*v)[s] = number of ordered pairs (a,b) in A with a+b=s
        conv = np.convolve(v, v, mode='full')
        
        # f(s) = conv[s] = ordered pair count
        # g(s) = 1 if s is even and s/2 in A, else 0
        g = np.zeros(len(conv), dtype=np.int64)
        for a in A_list:
            s = 2 * a
            if s < len(conv):
                g[s] = 1  # exactly one element a gives 2a=s
        
        # Unordered pair count = (f(s) + g(s)) / 2
        counts = (conv + g) // 2
        
        return counts[:max_sum + 1].astype(np.int32)
    
    def get_target_counts(A_list):
        """Get counts only for target sums [128, 256]."""
        counts = compute_counts(A_list)
        return counts[min_sum:max_sum + 1]
    
    def get_cost(A_list):
        """Compute the cost: max representation multiplicity in target range, with penalty for uncovered."""
        tc = get_target_counts(A_list)
        if np.any(tc == 0):
            uncovered = np.sum(tc == 0)
            return 1000 + uncovered
        return int(np.max(tc))
    
    def smooth_cost(A_list, beta=1.0):
        """Smooth surrogate for max multiplicity using log-sum-exp with coverage penalty."""
        tc = get_target_counts(A_list).astype(np.float64)
        # Coverage penalty: large penalty for uncovered sums
        uncovered = np.sum(tc == 0)
        if uncovered > 0:
            return 1000.0 + 10.0 * uncovered
        # Log-sum-exp for smooth max
        tc_safe = np.maximum(tc, 1e-6)
        max_tc = np.max(tc_safe)
        lse = max_tc + np.log(np.sum(np.exp(tc_safe - max_tc)))
        return lse
    
    # Initial set: structured start with good coverage properties
    random.seed(42)
    
    # Start with a spread set: include 0, 128, and evenly spaced elements
    # This gives better initial coverage than random
    base_elements = [0, 128]
    # Add elements spread across the range
    for i in range(1, 16):
        base_elements.append(i * 8)  # 8, 16, 24, ..., 120
    # Remove duplicates and ensure in range
    base_elements = list(set(x for x in base_elements if 0 <= x <= N))
    random.shuffle(base_elements)
    A_list = base_elements[:20]
    A_set = set(A_list)
    
    best_cost = get_cost(A_list)
    best_A = list(A_list)
    
    # Phase 1: Ensure coverage by adding elements
    while time.time() < deadline:
        tc = get_target_counts(A_list)
        uncovered = [s for s in target_sums if tc[s - min_sum] == 0]
        
        if not uncovered:
            break
        
        # Find best element to add
        best_x = None
        best_gain = -1
        
        # Try adding each candidate
        for x in range(0, N + 1):
            if x in A_set:
                continue
            # Gain: how many uncovered sums get covered
            gain = 0
            for y in A_list:
                s = x + y
                if min_sum <= s <= max_sum and tc[s - min_sum] == 0:
                    gain += 1
            # Also x+x
            s = 2 * x
            if min_sum <= s <= max_sum and tc[s - min_sum] == 0:
                gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_x = x
        
        if best_x is None or best_gain == 0:
            # Stuck, try random
            candidates = [x for x in range(0, N + 1) if x not in A_set]
            if candidates:
                best_x = random.choice(candidates)
            else:
                break
        
        A_list.append(best_x)
        A_set.add(best_x)
        
        # If set is getting too large, try to prune
        if len(A_list) > 25:
            # Try removing elements that don't break coverage
            for _ in range(3):
                if time.time() >= deadline:
                    break
                for idx in range(len(A_list)):
                    x = A_list[idx]
                    A_list.remove(x)
                    A_set.remove(x)
                    tc = get_target_counts(A_list)
                    if np.any(tc == 0):
                        A_list.append(x)
                        A_set.add(x)
                    else:
                        break
    
    # Phase 2: Local search with simulated annealing
    current_cost = get_cost(A_list)
    current_smooth = smooth_cost(A_list)
    
    if current_cost < best_cost:
        best_cost = current_cost
        best_A = list(A_list)
    
    # Simulated annealing parameters - tuned for faster convergence
    T_start = 10.0
    T_end = 0.05
    cooling_rate = 0.9995  # Slower cooling for better exploration
    
    # Precompute all possible elements for faster swaps
    all_elements = list(range(0, N + 1))
    
    iteration = 0
    stuck_count = 0
    max_stuck = 300  # Reset sooner to escape local minima
    best_smooth = smooth_cost(A_list)
    
    while time.time() < deadline:
        iteration += 1
        # Exponential cooling with floor
        T = T_start * (T_end / T_start) ** (iteration / 5000.0)
        if T < T_end:
            T = T_end
        
        # Choose move type - weighted for effectiveness
        r = random.random()
        
        if r < 0.60 and len(A_list) > 0:
            # Swap: remove one, add one (most common)
            idx = random.randint(0, len(A_list) - 1)
            x_in = A_list[idx]
            # Use precomputed list for speed
            x_out = random.choice(all_elements)
            while x_out in A_set:
                x_out = random.choice(all_elements)
            
            A_list[idx] = x_out
            A_set.remove(x_in)
            A_set.add(x_out)
            
            new_smooth = smooth_cost(A_list)
            
            # Accept if better or with probability exp(-delta/T)
            delta = new_smooth - current_smooth
            if delta < 0 or random.random() < np.exp(-delta / max(T, 1e-6)):
                current_smooth = new_smooth
                new_cost = get_cost(A_list)
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_A = list(A_list)
                    best_smooth = current_smooth
                    stuck_count = 0
                else:
                    stuck_count += 1
            else:
                # Revert
                A_list[idx] = x_in
                A_set.remove(x_out)
                A_set.add(x_in)
                stuck_count += 1
                
        elif r < 0.75 and len(A_list) > 14:
            # Remove an element (to find minimal sets)
            idx = random.randint(0, len(A_list) - 1)
            x = A_list[idx]
            A_list.pop(idx)
            A_set.remove(x)
            
            new_smooth = smooth_cost(A_list)
            
            delta = new_smooth - current_smooth
            if delta < 0 or random.random() < np.exp(-delta / max(T, 1e-6)):
                current_smooth = new_smooth
                new_cost = get_cost(A_list)
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_A = list(A_list)
                    best_smooth = current_smooth
                    stuck_count = 0
                else:
                    stuck_count += 1
            else:
                A_list.append(x)
                A_set.add(x)
                stuck_count += 1
                
        elif r < 0.90 and len(A_list) < 28:
            # Add an element
            x = random.choice(all_elements)
            while x in A_set:
                x = random.choice(all_elements)
            A_list.append(x)
            A_set.add(x)
            
            new_smooth = smooth_cost(A_list)
            
            delta = new_smooth - current_smooth
            if delta < 0 or random.random() < np.exp(-delta / max(T, 1e-6)):
                current_smooth = new_smooth
                new_cost = get_cost(A_list)
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_A = list(A_list)
                    best_smooth = current_smooth
                    stuck_count = 0
                else:
                    stuck_count += 1
            else:
                A_list.remove(x)
                A_set.remove(x)
                stuck_count += 1
                
        else:
            # Double swap: remove two, add two (for escaping local minima)
            if len(A_list) >= 2:
                idx1, idx2 = random.sample(range(len(A_list)), 2)
                x1 = A_list[idx1]
                x2 = A_list[idx2]
                x_out1 = random.choice(all_elements)
                while x_out1 in A_set:
                    x_out1 = random.choice(all_elements)
                x_out2 = random.choice(all_elements)
                while x_out2 in A_set or x_out2 == x_out1:
                    x_out2 = random.choice(all_elements)
                
                A_list[idx1] = x_out1
                A_list[idx2] = x_out2
                A_set.remove(x1)
                A_set.remove(x2)
                A_set.add(x_out1)
                A_set.add(x_out2)
                
                new_smooth = smooth_cost(A_list)
                
                delta = new_smooth - current_smooth
                if delta < 0 or random.random() < np.exp(-delta / max(T, 1e-6)):
                    current_smooth = new_smooth
                    new_cost = get_cost(A_list)
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_A = list(A_list)
                        best_smooth = current_smooth
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
        
        # If stuck too long, restart from best with perturbation
        if stuck_count > max_stuck:
            A_list = list(best_A)
            A_set = set(A_list)
            current_cost = get_cost(A_list)
            current_smooth = smooth_cost(A_list)
            stuck_count = 0
            # Perturb: swap 2-3 random elements to escape
            for _ in range(2):
                if len(A_list) > 0:
                    idx = random.randint(0, len(A_list) - 1)
                    x_out = random.choice(all_elements)
                    while x_out in A_set:
                        x_out = random.choice(all_elements)
                    x_in = A_list[idx]
                    A_list[idx] = x_out
                    A_set.remove(x_in)
                    A_set.add(x_out)
            current_smooth = smooth_cost(A_list)
            current_cost = get_cost(A_list)
            
        # Early termination if we found a very good solution
        if best_cost <= 2:
            break
    
    # Final polish: aggressive greedy local search to minimize peak multiplicity
    if time.time() < deadline - 3:
        polish_A = list(best_A)
        polish_set = set(polish_A)
        polish_cost = get_cost(polish_A)
        
        improved = True
        max_polish_iters = 50
        polish_iter = 0
        while improved and polish_iter < max_polish_iters and time.time() < deadline - 1:
            improved = False
            polish_iter += 1
            
            # Try removing each element (greedy: keep only if it helps)
            for idx in range(len(polish_A)):
                x = polish_A[idx]
                polish_A.remove(x)
                polish_set.remove(x)
                new_cost = get_cost(polish_A)
                if new_cost < polish_cost:
                    polish_cost = new_cost
                    improved = True
                    break
                else:
                    polish_A.append(x)
                    polish_set.add(x)
            
            if not improved:
                # Try swapping each element with every other
                for idx in range(len(polish_A)):
                    x_in = polish_A[idx]
                    for x_out in all_elements:
                        if x_out in polish_set:
                            continue
                        polish_A[idx] = x_out
                        polish_set.remove(x_in)
                        polish_set.add(x_out)
                        new_cost = get_cost(polish_A)
                        if new_cost < polish_cost:
                            polish_cost = new_cost
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
