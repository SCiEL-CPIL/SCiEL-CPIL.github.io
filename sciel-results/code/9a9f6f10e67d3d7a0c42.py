import time
import random
import math
import numpy as np

def construct():
    N = 256
    
    # Precompute all 4-term APs
    aps = []
    for d in range(1, N // 3 + 1):
        max_a = N - 3 * d
        for a in range(1, max_a + 1):
            aps.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))
    
    num_aps = len(aps)
    
    # NumPy array for vectorized full evaluation
    ap_indices = np.array(aps, dtype=np.int32)  # shape (num_aps, 4)
    
    def count_mono_np(x):
        """Vectorized count using NumPy. x is np.array of 0s and 1s."""
        vals = x[ap_indices]  # shape (num_aps, 4)
        prod1 = np.prod(vals, axis=1)
        prod0 = np.prod(1 - vals, axis=1)
        return int(np.sum(prod1 + prod0))
    
    # Precompute link_info for fast delta_flip
    link_info = [[] for _ in range(N)]
    for i, ap in enumerate(aps):
        for pos, idx in enumerate(ap):
            if pos == 0:
                others = (ap[1], ap[2], ap[3])
            elif pos == 1:
                others = (ap[0], ap[2], ap[3])
            elif pos == 2:
                others = (ap[0], ap[1], ap[3])
            else:
                others = (ap[0], ap[1], ap[2])
            link_info[idx].append(others)
    
    def count_mono(colors):
        count = 0
        for ap in aps:
            c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
            if c0 == c1 == c2 == c3:
                count += 1
        return count
    
    def delta_flip(colors, k):
        old_c = colors[k]
        new_c = 1 - old_c
        delta = 0
        for others in link_info[k]:
            c1, c2, c3 = colors[others[0]], colors[others[1]], colors[others[2]]
            if c1 == c2 == c3 == new_c:
                delta += 1
            if c1 == c2 == c3 == old_c:
                delta -= 1
        return delta
    
    # Thue-Morse
    def thue_morse(n):
        return bin(n).count('1') % 2
    
    # Generate diverse initializations
    init_strategies = []
    
    # Thue-Morse
    tm = [thue_morse(i) for i in range(N)]
    init_strategies.append(tm)
    
    # Random seeds
    random.seed(42)
    for seed in [42, 123, 456, 789]:
        random.seed(seed)
        init_strategies.append([random.randint(0, 1) for _ in range(N)])
    
    # Block patterns
    for block_size in [4, 8, 16, 32]:
        block_init = [(i // block_size) % 2 for i in range(N)]
        init_strategies.append(block_init)
    
    # Modular patterns
    for mod in [3, 5, 7]:
        mod_init = [(i % mod) < mod // 2 for i in range(N)]
        init_strategies.append(mod_init)
    
    # Evaluate all and pick best
    best_init = None
    best_init_cost = float('inf')
    for init in init_strategies:
        cost = count_mono(init)
        if cost < best_init_cost:
            best_init_cost = cost
            best_init = init[:]
    
    start_time = time.time()
    deadline = start_time + 115.0
    
    global_best_colors = best_init[:]
    global_best_cost = best_init_cost
    
    # SA with multiple restarts
    num_restarts = 4
    time_per_restart = (deadline - start_time) / num_restarts
    
    for restart in range(num_restarts):
        restart_deadline = start_time + time_per_restart * (restart + 1)
        
        if restart == 0:
            current_colors = best_init[:]
            current_cost = best_init_cost
        else:
            current_colors = global_best_colors[:]
            # Perturb with varying strength
            flip_prob = 0.08 + 0.04 * restart
            for i in range(N):
                if random.random() < flip_prob:
                    current_colors[i] = 1 - current_colors[i]
            current_cost = count_mono(current_colors)
        
        T = 1.5
        T_min = 0.005
        steps_since_improvement = 0
        max_steps_no_improve = 80000
        
        while time.time() < restart_deadline:
            k = random.randint(0, N - 1)
            d_cost = delta_flip(current_colors, k)
            
            if d_cost < 0 or (T > T_min and random.random() < math.exp(-d_cost / T)):
                current_colors[k] = 1 - current_colors[k]
                current_cost += d_cost
                if current_cost < global_best_cost:
                    global_best_cost = current_cost
                    global_best_colors = current_colors[:]
                    steps_since_improvement = 0
                else:
                    steps_since_improvement += 1
            else:
                steps_since_improvement += 1
            
            T *= 0.99997
            if T < T_min:
                T = T_min
            
            if T <= T_min and steps_since_improvement > max_steps_no_improve:
                break
    
    # Final greedy polish: try flipping each bit, keep if improves
    if time.time() < deadline - 5.0:
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            for pos in range(N):
                if time.time() > deadline - 5.0:
                    break
                old_val = global_best_colors[pos]
                global_best_colors[pos] = 1 - old_val
                new_cost = global_best_cost + delta_flip(global_best_colors, pos)
                # delta_flip computed with flipped value, so we need to undo
                global_best_colors[pos] = old_val
                # Recompute: flip, check, unflip
                global_best_colors[pos] = 1 - old_val
                d = delta_flip(global_best_colors, pos)
                # d is change if we flip back, so actual change from flipping was -d
                actual_delta = -d
                if -d < 0:  # flipping improves
                    global_best_cost += actual_delta
                    improved = True
                else:
                    global_best_colors[pos] = old_val
    
    return global_best_colors
