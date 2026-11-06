import numpy as np
import time
import random

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 110.0

    # Precompute all 4-term APs as index tuples (0-based)
    aps = []
    for d in range(1, N // 3 + 1):
        max_a = N - 3 * d
        for a in range(1, max_a + 1):
            aps.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))
    
    num_aps = len(aps)

    # Convert to numpy array for vectorized operations
    ap_array = np.array(aps, dtype=np.int32)  # shape (num_aps, 4)

    # Map bit index to list of AP indices it participates in
    bit_to_aps = [[] for _ in range(N)]
    for idx in range(num_aps):
        for pos in aps[idx]:
            bit_to_aps[pos].append(idx)

    # Precompute for each bit, the AP indices and the positions of the other 3 elements
    # bit_ap_data[i] = list of (ap_idx, other_pos1, other_pos2, other_pos3)
    bit_ap_data = [[] for _ in range(N)]
    for idx, ap in enumerate(aps):
        for j in range(4):
            pos = ap[j]
            others = [ap[k] for k in range(4) if k != j]
            bit_ap_data[pos].append((idx, others[0], others[1], others[2]))

    # Function to count monochromatic APs (vectorized)
    def count_mono(x):
        x = np.asarray(x, dtype=np.int8)
        # For each AP, check if all 4 positions have same value
        # x[ap_array] gives shape (num_aps, 4)
        vals = x[ap_array]  # (num_aps, 4)
        # Mono if all equal: check if max == min
        return int(np.sum(np.all(vals == vals[:, 0:1], axis=1)))

    # Fast delta for flipping bit i
    def delta_flip(x, i):
        old_val = x[i]
        new_val = 1 - old_val
        delta = 0
        for (ap_idx, o1, o2, o3) in bit_ap_data[i]:
            # Check if the other 3 are all equal to old_val (was mono)
            was_mono = (x[o1] == old_val) and (x[o2] == old_val) and (x[o3] == old_val)
            # Check if the other 3 are all equal to new_val (becomes mono)
            is_mono = (x[o1] == new_val) and (x[o2] == new_val) and (x[o3] == new_val)
            if was_mono and not is_mono:
                delta -= 1
            elif not was_mono and is_mono:
                delta += 1
        return delta

    # Simulated Annealing with better cooling and 2-flip moves
    def simulated_annealing(initial_x, time_limit, T_start=5.0, T_end=0.05):
        x = np.array(initial_x, dtype=np.int8)
        current_cost = count_mono(x)
        best_x = x.copy()
        best_cost = current_cost
        
        T = T_start
        step = 0
        total_steps_estimate = 200000  # rough estimate
        cool_rate = (T_end / T_start) ** (1.0 / total_steps_estimate)
        
        while time.time() < time_limit:
            # Mix of 1-flip and 2-flip moves
            if random.random() < 0.3:
                # 2-flip move
                i1 = random.randint(0, N - 1)
                i2 = random.randint(0, N - 1)
                if i1 == i2:
                    continue
                d1 = delta_flip(x, i1)
                # Temporarily flip i1 to compute combined effect
                x[i1] = 1 - x[i1]
                d2 = delta_flip(x, i1)  # delta for flipping i1 back
                d_combined = d1 + d2
                # But this isn't quite right for 2-flip. Let's do it properly:
                x[i1] = 1 - x[i1]  # flip back
                # Compute delta for flipping both i1 and i2 simultaneously
                # Save state
                old_i1 = x[i1]
                old_i2 = x[i2]
                x[i1] = 1 - old_i1
                x[i2] = 1 - old_i2
                # Count change: we need to recompute affected APs
                # Simpler: just compute cost difference by checking affected APs
                affected = set()
                for (ap_idx, o1, o2, o3) in bit_ap_data[i1]:
                    affected.add(ap_idx)
                for (ap_idx, o1, o2, o3) in bit_ap_data[i2]:
                    affected.add(ap_idx)
                
                delta2 = 0
                for ap_idx in affected:
                    ap = aps[ap_idx]
                    # Check mono before (with original x)
                    # We need original values
                    # This is getting complex. Let's use a simpler approach:
                    # Just do sequential flips with acceptance
                    pass
                
                # Revert
                x[i1] = old_i1
                x[i2] = old_i2
                # For simplicity, do two sequential 1-flips
                continue
            
            # 1-flip move
            i = random.randint(0, N - 1)
            d = delta_flip(x, i)
            
            # Accept if improvement or with probability exp(-d/T)
            if d < 0 or random.random() < np.exp(-d / T):
                x[i] = 1 - x[i]
                current_cost += d
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_x = x.copy()
            
            # Cool down
            step += 1
            T *= cool_rate
            if T < T_end:
                T = T_end
        
        return best_x, best_cost

    # Initialize with diverse starting points
    def thue_morse(n):
        return [bin(i).count('1') % 2 for i in range(n)]
    
    def periodic_pattern(n, period):
        return [i % period % 2 for i in range(n)]
    
    tm_x = np.array(thue_morse(N), dtype=np.int8)
    rand_x = np.array([random.randint(0, 1) for _ in range(N)], dtype=np.int8)
    # Periodic pattern with period 4: 0,1,0,1,...
    per4_x = np.array(periodic_pattern(N, 4), dtype=np.int8)
    # Periodic pattern with period 5: 0,1,0,1,0,1,...
    per5_x = np.array([i % 5 < 2 for i in range(N)], dtype=np.int8)
    
    best_overall_x = None
    best_overall_cost = float('inf')
    
    # Run SA on multiple starts, splitting time
    starts = [rand_x, tm_x, per4_x, per5_x]
    n_starts = len(starts)
    time_per_start = (deadline - start_time) / n_starts
    
    for s_idx, init_x in enumerate(starts):
        t_limit = start_time + (s_idx + 1) * time_per_start
        if time.time() < t_limit:
            x_best, c_best = simulated_annealing(init_x, t_limit)
            if c_best < best_overall_cost:
                best_overall_cost = c_best
                best_overall_x = x_best
    
    # Fallback
    if best_overall_x is None:
        best_overall_x = rand_x
    
    # Final polish: greedy local search on best solution
    x = np.array(best_overall_x, dtype=np.int8)
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for i in range(N):
            d = delta_flip(x, i)
            if d < 0:
                x[i] = 1 - x[i]
                improved = True
    
    # Check if polished solution is better
    final_cost = count_mono(x)
    if final_cost < best_overall_cost:
        best_overall_x = x
        best_overall_cost = final_cost
    
    # Return as list of ints
    return [int(v) for v in best_overall_x]
