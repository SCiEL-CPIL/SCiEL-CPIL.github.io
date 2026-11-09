import time
import random
import numpy as np

def construct():
    n = 256
    t0 = time.time()
    deadline = t0 + 55.0  # Safety margin

    # Precompute index lists for each step d and start s?
    # Slicing x[s::d] is very fast in numpy. We will rely on that.
    
    def evaluate(x):
        """
        Compute max absolute sum over all arithmetic progressions in x.
        x is a numpy array of length n with values -1, 1.
        """
        max_abs = 1
        # We iterate over all steps d from 1 to n-1
        # For each step d, we iterate over all starts s from 0 to d-1
        # The progression is x[s], x[s+d], x[s+2d], ...
        
        # Optimization: 
        # For a fixed d, the number of progressions is d.
        # The total number of elements across all progressions for step d is n.
        # So total work is O(n^2).
        
        for d in range(1, n):
            for s in range(d):
                # Extract the progression
                # x[s::d] creates a view, which is fast.
                vals = x[s::d]
                if len(vals) == 0:
                    continue
                
                # Compute prefix sums
                # We need to consider subarrays.
                # Sum of subarray [i, j] is P[j] - P[i-1].
                # Let P_full = [0, P[0], P[1], ...]
                # Max absolute sum is max(P_full) - min(P_full)
                
                P = np.cumsum(vals)
                # We can check min/max including 0
                # p_min = min(0, P.min())
                # p_max = max(0, P.max())
                # local_max = p_max - p_min
                
                # To avoid creating a new array with 0 prepended:
                p_min = P.min()
                p_max = P.max()
                
                if p_min > 0:
                    local_max = p_max
                elif p_max < 0:
                    local_max = -p_min
                else:
                    local_max = p_max - p_min
                
                if local_max > max_abs:
                    max_abs = local_max
                    # Optional: Early termination if max_abs is already very high?
                    # No, we need the exact value for the search.
                    
        return int(max_abs)

    def find_worst_ap(x):
        """
        Find the AP with the maximum absolute sum.
        Returns (start, step, max_abs_sum)
        """
        max_abs = 1
        best_s = 0
        best_d = 1
        
        for d in range(1, n):
            for s in range(d):
                vals = x[s::d]
                if len(vals) == 0:
                    continue
                P = np.cumsum(vals)
                p_min = P.min()
                p_max = P.max()
                
                if p_min > 0:
                    local_max = p_max
                elif p_max < 0:
                    local_max = -p_min
                else:
                    local_max = p_max - p_min
                
                if local_max > max_abs:
                    max_abs = local_max
                    best_s = s
                    best_d = d
                    
        return best_s, best_d, max_abs

    def optimize(x_init, max_time=10.0):
        """
        Perform local search with perturbation (Iterated Local Search).
        """
        x = x_init.copy()
        current_val = evaluate(x)
        best_x = x.copy()
        best_val = current_val
        
        t_start = time.time()
        stuck_count = 0
        max_stuck = 100 # Perturb after 100 non-improving steps
        
        while (time.time() - t_start) < max_time:
            # 1. Identify worst AP
            s_w, d_w, val_w = find_worst_ap(x)
            
            # 2. Generate candidate moves
            # Prioritize indices in the worst AP
            # The worst AP is x[s_w::d_w]
            # Let's get the indices of this AP
            ap_indices = np.arange(s_w, n, d_w)
            
            # Shuffle the AP indices to pick a random one from the worst AP
            np.random.shuffle(ap_indices)
            
            # Also generate some random indices to escape local traps
            # We'll try a mix: 50% from worst AP, 50% random
            
            improved = False
            
            # Try flipping indices from the worst AP
            for idx in ap_indices:
                if time.time() > t_start + max_time:
                    break
                old = x[idx]
                x[idx] = -old
                new_val = evaluate(x)
                if new_val < current_val:
                    current_val = new_val
                    improved = True
                    if current_val < best_val:
                        best_val = current_val
                        best_x = x.copy()
                    stuck_count = 0
                    break # Accept first improvement, restart loop to re-evaluate worst AP
                else:
                    x[idx] = old
            
            if improved:
                continue
                
            # If no improvement from worst AP, try random flips
            # Try a few random flips
            for _ in range(10):
                if time.time() > t_start + max_time:
                    break
                idx = np.random.randint(n)
                old = x[idx]
                x[idx] = -old
                new_val = evaluate(x)
                if new_val < current_val:
                    current_val = new_val
                    improved = True
                    if current_val < best_val:
                        best_val = current_val
                        best_x = x.copy()
                    stuck_count = 0
                    break
                else:
                    x[idx] = old
            
            if improved:
                continue
                
            # Stuck!
            stuck_count += 1
            if stuck_count >= max_stuck:
                # Perturb the best solution found so far
                # Flip 5-10% of the bits
                num_flips = random.randint(5, 10)
                flip_indices = np.random.choice(n, num_flips, replace=False)
                x = best_x.copy()
                x[flip_indices] *= -1
                current_val = evaluate(x)
                stuck_count = 0
                # Reset max_stuck slightly? Or keep it constant.
                # Let's keep it constant for simplicity.

        return best_x, best_val

    # Strategy: Multiple restarts
    best_overall = None
    best_overall_val = 999
    
    rng = np.random.RandomState(42)
    
    # 1. Thue-Morse
    def thue_morse(n):
        seq = [1]
        while len(seq) < n:
            seq = seq + [-s for s in seq]
        return np.array(seq[:n], dtype=np.int8)
    
    x_tm = thue_morse(n)
    x_opt, val_opt = optimize(x_tm, max_time=10.0)
    if val_opt < best_overall_val:
        best_overall_val = val_opt
        best_overall = x_opt
    
    # 2. Random
    x_rand = np.array(rng.choice([-1, 1], n), dtype=np.int8)
    x_opt, val_opt = optimize(x_rand, max_time=10.0)
    if val_opt < best_overall_val:
        best_overall_val = val_opt
        best_overall = x_opt
        
    # 3. Alternating
    x_alt = np.array([(-1)**i for i in range(n)], dtype=np.int8)
    x_opt, val_opt = optimize(x_alt, max_time=10.0)
    if val_opt < best_overall_val:
        best_overall_val = val_opt
        best_overall = x_opt
        
    # 4. Random again
    x_rand2 = np.array(rng.choice([-1, 1], n), dtype=np.int8)
    x_opt, val_opt = optimize(x_rand2, max_time=10.0)
    if val_opt < best_overall_val:
        best_overall_val = val_opt
        best_overall = x_opt
        
    # 5. Block Alternating (period 4)
    x_block = np.array([(-1)**(i//4) for i in range(n)], dtype=np.int8)
    x_opt, val_opt = optimize(x_block, max_time=10.0)
    if val_opt < best_overall_val:
        best_overall_val = val_opt
        best_overall = x_opt

    # If time remains, do a final polish on the best solution
    if time.time() < deadline - 2:
        remaining = deadline - 2 - time.time()
        x_opt, val_opt = optimize(best_overall, max_time=remaining)
        if val_opt < best_overall_val:
            best_overall_val = val_opt
            best_overall = x_opt

    return [int(v) for v in best_overall]
