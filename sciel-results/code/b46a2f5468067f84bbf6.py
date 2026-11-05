import math
import time
import random

def construct():
    N = 256
    deadline = time.time() + 110
    
    # Precompute all 4-APs
    aps = []
    aps_at = [[] for _ in range(N)]
    
    for d in range(1, N // 3 + 1):
        max_a = N - 3 * d
        for a in range(max_a):
            idxs = (a, a+d, a+2*d, a+3*d)
            aps.append(idxs)
            for i in range(4):
                cur = idxs[i]
                others = [idxs[j] for j in range(4) if j != i]
                aps_at[cur].append(others)
    
    def count_mono(cols):
        cnt = 0
        for ap in aps:
            if cols[ap[0]] == cols[ap[1]] == cols[ap[2]] == cols[ap[3]]:
                cnt += 1
        return cnt
    
    def delta_for_flip(cols, i):
        """Calculate cost delta if we flip position i."""
        c_i = cols[i]
        delta = 0
        for others in aps_at[i]:
            j, k, l = others
            c_j = cols[j]
            c_k = cols[k]
            c_l = cols[l]
            if c_j == c_k == c_l:
                if c_i == c_j:
                    delta -= 1
                else:
                    delta += 1
        return delta
    
    def greedy_polish(colors, current_cost):
        """Try all single flips and apply any that reduce cost."""
        improved = True
        while improved:
            improved = False
            for i in range(N):
                d = delta_for_flip(colors, i)
                if d < 0:
                    colors[i] = 1 - colors[i]
                    current_cost += d
                    improved = True
        return colors, current_cost
    
    def simulated_annealing(initial_colors, T_start, cooling, min_T, time_limit):
        colors = initial_colors[:]
        current_cost = count_mono(colors)
        best_cost = current_cost
        best_colors = colors[:]
        
        T = T_start
        end_time = time.time() + time_limit
        
        while time.time() < end_time:
            i = random.randint(0, N - 1)
            delta = delta_for_flip(colors, i)
            
            if delta < 0:
                colors[i] = 1 - colors[i]
                current_cost += delta
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_colors = colors[:]
            elif delta == 0:
                colors[i] = 1 - colors[i]
            else:
                if T > 0:
                    prob = math.exp(-delta / T)
                else:
                    prob = 0.0
                
                if random.random() < prob:
                    colors[i] = 1 - colors[i]
                    current_cost += delta
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_colors = colors[:]
            
            T *= cooling
            if T < min_T:
                T = min_T
        
        return best_colors, best_cost
    
    # Global best
    global_best_colors = None
    global_best_cost = float('inf')
    
    # Multiple restarts with different strategies
    num_restarts = 8
    time_per_restart = 110.0 / num_restarts
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Different initial colorings for diversity
        if restart == 0:
            # All zeros
            initial = [0] * N
        elif restart == 1:
            # All ones
            initial = [1] * N
        elif restart == 2:
            # Alternating
            initial = [i % 2 for i in range(N)]
        elif restart == 3:
            # Block pattern
            initial = [(i // 8) % 2 for i in range(N)]
        elif restart == 4:
            # Random
            initial = [random.randint(0, 1) for _ in range(N)]
        else:
            # Perturb previous best
            if global_best_colors is not None:
                initial = global_best_colors[:]
                for _ in range(N // 4):
                    idx = random.randint(0, N - 1)
                    initial[idx] = 1 - initial[idx]
            else:
                initial = [random.randint(0, 1) for _ in range(N)]
        
        # Simulated annealing with varying parameters
        T_start = 1.0 + 0.5 * (restart % 3)
        cooling = 0.9999 - 0.00001 * (restart % 5)
        min_T = 0.005
        
        sa_colors, sa_cost = simulated_annealing(
            initial, T_start, cooling, min_T, time_per_restart * 0.8
        )
        
        # Greedy polish
        polished_colors, polished_cost = greedy_polish(sa_colors, sa_cost)
        
        if polished_cost < global_best_cost:
            global_best_cost = polished_cost
            global_best_colors = polished_colors[:]
    
    # Final polish on the best result
    if global_best_colors is not None:
        final_colors, final_cost = greedy_polish(global_best_colors, global_best_cost)
        if final_cost < global_best_cost:
            global_best_cost = final_cost
            global_best_colors = final_colors
    
    return global_best_colors
