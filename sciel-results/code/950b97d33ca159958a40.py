import math
import time
import random
import numpy as np

def construct():
    N = 256
    deadline = time.time() + 115
    
    # Precompute all 4-term arithmetic progressions
    # AP: (a, a+d, a+2d, a+3d) with 0-based indices
    aps = []
    aps_at = [[] for _ in range(N)]
    
    for d in range(1, N // 3 + 1):
        max_a = N - 3 * d
        for a in range(max_a):
            idxs = (a, a + d, a + 2 * d, a + 3 * d)
            aps.append(idxs)
            for i in range(4):
                cur = idxs[i]
                others = [idxs[j] for j in range(4) if j != i]
                aps_at[cur].append(others)
    
    total_aps = len(aps)
    
    # Count monochromatic APs
    def count_mono(cols):
        cnt = 0
        for ap in aps:
            if cols[ap[0]] == cols[ap[1]] == cols[ap[2]] == cols[ap[3]]:
                cnt += 1
        return cnt
    
    # Greedy construction: assign each element to minimize current mono count
    def greedy_construct(seed_val):
        random.seed(seed_val)
        cols = [None] * N
        # Assign in random order
        order = list(range(N))
        random.shuffle(order)
        
        for idx in order:
            # Try both colors, pick the one that adds fewer mono APs
            best_c = 0
            best_delta = float('inf')
            for c in [0, 1]:
                delta = 0
                for others in aps_at[idx]:
                    j, k, l = others
                    if cols[j] is not None and cols[k] is not None and cols[l] is not None:
                        if c == cols[j] == cols[k] == cols[l]:
                            delta += 1
                if delta < best_delta:
                    best_delta = delta
                    best_c = c
            cols[idx] = best_c
        return cols
    
    # Simulated annealing
    def simulate_annealing(initial_cols, T_start, cooling, min_T, time_limit):
        cols = initial_cols[:]
        current_cost = count_mono(cols)
        best_cost = current_cost
        best_cols = cols[:]
        T = T_start
        start_time = time.time()
        
        while time.time() - start_time < time_limit:
            # Pick random index
            i = random.randint(0, N - 1)
            
            # Calculate delta
            delta = 0
            c_i = cols[i]
            for others in aps_at[i]:
                j, k, l = others
                c_j = cols[j]
                c_k = cols[k]
                c_l = cols[l]
                
                # Before flip
                if c_i == c_j == c_k == c_l:
                    delta -= 1
                # After flip
                if (1 - c_i) == c_j == c_k == c_l:
                    delta += 1
            
            if delta < 0:
                cols[i] = 1 - c_i
                current_cost += delta
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_cols = cols[:]
            else:
                if delta == 0:
                    cols[i] = 1 - c_i
                else:
                    r = random.random()
                    if r < math.exp(-delta / T):
                        cols[i] = 1 - c_i
                        current_cost += delta
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_cols = cols[:]
            
            T *= cooling
            if T < min_T:
                T = min_T
        
        return best_cols, best_cost
    
    # Local search: try all single flips, accept if improves
    def local_search(cols, max_rounds=10):
        cols = cols[:]
        current_cost = count_mono(cols)
        best_cost = current_cost
        best_cols = cols[:]
        
        for _ in range(max_rounds):
            improved = False
            for i in range(N):
                c_i = cols[i]
                delta = 0
                for others in aps_at[i]:
                    j, k, l = others
                    c_j = cols[j]
                    c_k = cols[k]
                    c_l = cols[l]
                    if c_i == c_j == c_k == c_l:
                        delta -= 1
                    if (1 - c_i) == c_j == c_k == c_l:
                        delta += 1
                if delta < 0:
                    cols[i] = 1 - c_i
                    current_cost += delta
                    improved = True
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_cols = cols[:]
            if not improved:
                break
        
        return best_cols, best_cost
    
    # Run multiple strategies
    best_overall = None
    best_cost_overall = float('inf')
    
    # Strategy 1: Greedy + SA + local search
    for seed in range(5):
        if time.time() > deadline:
            break
        cols = greedy_construct(seed * 1000 + 42)
        cols, cost = simulate_annealing(cols, T_start=2.0, cooling=0.9999, min_T=0.05, time_limit=15)
        cols, cost = local_search(cols, max_rounds=20)
        if cost < best_cost_overall:
            best_cost_overall = cost
            best_overall = cols[:]
    
    # Strategy 2: Random init + SA + local search
    for seed in range(5):
        if time.time() > deadline:
            break
        random.seed(seed * 2000 + 7)
        cols = [random.randint(0, 1) for _ in range(N)]
        cols, cost = simulate_annealing(cols, T_start=1.0, cooling=0.99995, min_T=0.01, time_limit=15)
        cols, cost = local_search(cols, max_rounds=20)
        if cost < best_cost_overall:
            best_cost_overall = cost
            best_overall = cols[:]
    
    # Strategy 3: Pattern-based initialization
    # Try periodic patterns
    for period in [2, 3, 4, 5, 6, 7, 8]:
        if time.time() > deadline:
            break
        for offset in range(period):
            cols = [((i + offset) % period) % 2 for i in range(N)]
            cols, cost = simulate_annealing(cols, T_start=1.5, cooling=0.9999, min_T=0.05, time_limit=10)
            cols, cost = local_search(cols, max_rounds=15)
            if cost < best_cost_overall:
                best_cost_overall = cost
                best_overall = cols[:]
    
    # Final local search on best result
    if best_overall is not None:
        best_overall, best_cost_overall = local_search(best_overall, max_rounds=50)
    
    if best_overall is None:
        # Fallback: random
        best_overall = [random.randint(0, 1) for _ in range(N)]
    
    return best_overall
