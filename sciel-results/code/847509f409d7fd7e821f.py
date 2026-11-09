import random
import time
import math

def construct():
    n = 256
    
    # Precompute all 4-APs
    all_aps = []
    pos_to_aps = [[] for _ in range(n)]
    
    for d in range(1, (n - 3) // 3 + 1):
        for a in range(0, n - 3 * d):
            ap = (a, a + d, a + 2 * d, a + 3 * d)
            idx = len(all_aps)
            all_aps.append(ap)
            for j in range(4):
                pos = ap[j]
                pos_to_aps[pos].append((idx, j))
                
    def count_mono(colors):
        count = 0
        for ap in all_aps:
            if colors[ap[0]] == colors[ap[1]] == colors[ap[2]] == colors[ap[3]]:
                count += 1
        return count

    def calc_delta(colors, i):
        old_c = colors[i]
        new_c = 1 - old_c
        delta = 0
        for ap_idx, pos_in_ap in pos_to_aps[i]:
            ap = all_aps[ap_idx]
            c0 = colors[ap[0]]
            c1 = colors[ap[1]]
            c2 = colors[ap[2]]
            c3 = colors[ap[3]]
            
            before_mono = (c0 == c1 == c2 == c3)
            
            if pos_in_ap == 0:
                after_mono = (new_c == c1 == c2 == c3)
            elif pos_in_ap == 1:
                after_mono = (c0 == new_c == c2 == c3)
            elif pos_in_ap == 2:
                after_mono = (c0 == c1 == new_c == c3)
            else:
                after_mono = (c0 == c1 == c2 == new_c)
                
            if before_mono and not after_mono:
                delta -= 1
            elif not before_mono and after_mono:
                delta += 1
        return delta

    # Iterated Local Search
    def local_search(colors, current_cost):
        # Hill climbing: keep flipping if it improves
        # To avoid infinite loops or slow convergence, we use a random restart of the scan order
        while True:
            improved = False
            # Shuffle indices to avoid bias
            indices = list(range(n))
            random.shuffle(indices)
            for i in indices:
                d_cost = calc_delta(colors, i)
                if d_cost < 0:
                    colors[i] = 1 - colors[i]
                    current_cost += d_cost
                    improved = True
                    # Restart scan? Or continue? 
                    # Continuing with the shuffled list is fine, but the delta for other indices 
                    # might have changed. 
                    # For simplicity and speed, we do one pass. If any improvement is made, 
                    # we repeat the whole pass.
            if not improved:
                break
        return colors, current_cost

    def perturb(colors, k):
        for _ in range(k):
            i = random.randint(0, n - 1)
            colors[i] = 1 - colors[i]

    # Initialization
    colors = [random.randint(0, 1) for _ in range(n)]
    current_cost = count_mono(colors)
    best_cost = current_cost
    best_colors = colors.copy()
    
    start_time = time.time()
    time_limit = 115.0
    
    # Perturbation strength: number of bits to flip
    k_perturb = 15
    
    while (time.time() - start_time) < time_limit:
        # 1. Local Search
        colors, current_cost = local_search(colors, current_cost)
        
        # 2. Update Best
        if current_cost < best_cost:
            best_cost = current_cost
            best_colors = colors.copy()
            
        # 3. Perturb
        # We perturb the current state (which is a local optimum) to get a new state
        # We might want to perturb the best state if we are stuck, but standard ILS perturbs current.
        # To ensure we explore different basins, we perturb current.
        perturb(colors, k_perturb)
        
        # Recalculate cost after perturbation? 
        # We can just let the next local search handle it, but we need current_cost to be accurate 
        # if we were to compare, but local_search takes cost as input. 
        # Since we flipped bits, the cost changed. 
        # We can estimate or just recalculate. Recalculation is O(APs) ~ 10k ops. Fast enough.
        current_cost = count_mono(colors)

    return best_colors
