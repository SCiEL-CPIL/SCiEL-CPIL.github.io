import time
import random
import math

def construct():
    N = 256
    # Precompute all 4-term APs
    # (a, a+d, a+2d, a+3d) with 1 <= a, a+3d <= N
    # Indices 0..N-1 correspond to 1..N
    aps = []
    for d in range(1, N // 3 + 1):
        max_a = N - 3 * d
        for a in range(1, max_a + 1):
            # a, a+d, a+2d, a+3d
            # indices: a-1, a+d-1, a+2d-1, a+3d-1
            aps.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))
    
    num_aps = len(aps)
    
    # Precompute links for each index: list of (ap_index, position_in_ap)
    links = [[] for _ in range(N)]
    for i, ap in enumerate(aps):
        for pos, idx in enumerate(ap):
            links[idx].append((i, pos))
    
    # Precompute for each (ap_idx, pos): the other 3 indices as a tuple
    # This avoids repeated indexing in delta_flip
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
        """
        Calculate change in monochromatic AP count if we flip colors[k].
        Returns delta. Does NOT modify colors.
        """
        old_c = colors[k]
        new_c = 1 - old_c
        delta = 0
        for others in link_info[k]:
            c1, c2, c3 = colors[others[0]], colors[others[1]], colors[others[2]]
            
            # Check if it becomes mono
            if c1 == c2 == c3 == new_c:
                delta += 1
            # Check if it was mono
            if c1 == c2 == c3 == old_c:
                delta -= 1
        return delta

    # Thue-Morse initialization
    def thue_morse(n):
        return bin(n).count('1') % 2
    
    # Try multiple initializations and pick the best one
    init_strategies = []
    
    # Strategy 1: Thue-Morse
    tm = [thue_morse(i) for i in range(N)]
    init_strategies.append(tm)
    
    # Strategy 2: Random
    random.seed(42)
    rand_init = [random.randint(0, 1) for _ in range(N)]
    init_strategies.append(rand_init)
    
    # Strategy 3: Alternating blocks of size 8
    block_init = []
    for i in range(N):
        block_init.append((i // 8) % 2)
    init_strategies.append(block_init)
    
    # Strategy 4: Alternating blocks of size 16
    block16_init = []
    for i in range(N):
        block16_init.append((i // 16) % 2)
    init_strategies.append(block16_init)
    
    # Evaluate all initializations and pick the best
    best_init = None
    best_init_cost = float('inf')
    for init in init_strategies:
        cost = count_mono(init)
        if cost < best_init_cost:
            best_init_cost = cost
            best_init = init[:]
    
    # Simulated Annealing with restarts
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin for safety
    
    global_best_colors = best_init[:]
    global_best_cost = best_init_cost
    
    # Multiple restarts
    num_restarts = 3
    time_per_restart = (deadline - start_time) / num_restarts
    
    for restart in range(num_restarts):
        restart_deadline = start_time + time_per_restart * (restart + 1)
        
        # For first restart, use best_init; for others, perturb global best
        if restart == 0:
            current_colors = best_init[:]
            current_cost = best_init_cost
        else:
            current_colors = global_best_colors[:]
            # Perturb: flip ~10% of bits
            for i in range(N):
                if random.random() < 0.1:
                    current_colors[i] = 1 - current_colors[i]
            current_cost = count_mono(current_colors)
        
        T = 1.0
        T_min = 0.01
        
        steps_since_improvement = 0
        max_steps_no_improve = 50000
        
        while time.time() < restart_deadline:
            # Pick random index
            k = random.randint(0, N - 1)
            d_cost = delta_flip(current_colors, k)
            
            # Acceptance criterion
            if d_cost < 0 or (T > T_min and random.random() < math.exp(-d_cost / T)):
                current_colors[k] = 1 - current_colors[k]
                current_cost += d_cost
                if current_cost < global_best_cost:
                    global_best_cost = current_cost
                    global_best_colors = current_colors[:]
                    steps_since_improvement = 0
                elif d_cost >= 0:
                    steps_since_improvement += 1
                else:
                    # Improved current but not global
                    steps_since_improvement += 1
            else:
                steps_since_improvement += 1
            
            # Cool down - use a slower cooling schedule
            T *= 0.99995
            if T < T_min:
                T = T_min
            
            # If stuck for too long at low T, break early and restart
            if T <= T_min and steps_since_improvement > max_steps_no_improve:
                break
    
    return global_best_colors

# EVOLVE-BLOCK-END
