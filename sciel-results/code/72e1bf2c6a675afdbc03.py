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
        for ap_idx, pos in links[k]:
            ap = aps[ap_idx]
            # Get other 3 indices
            # Optimization: avoid list creation
            if pos == 0:
                o1, o2, o3 = ap[1], ap[2], ap[3]
            elif pos == 1:
                o1, o2, o3 = ap[0], ap[2], ap[3]
            elif pos == 2:
                o1, o2, o3 = ap[0], ap[1], ap[3]
            else:
                o1, o2, o3 = ap[0], ap[1], ap[2]
            
            c1, c2, c3 = colors[o1], colors[o2], colors[o3]
            
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
    
    init_colors = [thue_morse(i) for i in range(N)]
    
    # Simulated Annealing
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    best_colors = init_colors[:]
    best_cost = count_mono(best_colors)
    
    current_colors = init_colors[:]
    current_cost = best_cost
    
    T = 1.0
    T_min = 0.001
    # Estimate number of steps we can do
    # Let's just run until time limit
    
    while time.time() < deadline:
        # Pick random index
        k = random.randint(0, N - 1)
        d_cost = delta_flip(current_colors, k)
        
        # Acceptance criterion
        if d_cost < 0 or random.random() < math.exp(-d_cost / T):
            current_colors[k] = 1 - current_colors[k]
            current_cost += d_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_colors = current_colors[:]
        
        # Cool down
        # Simple geometric cooling
        T *= 0.9999
        if T < T_min:
            T = T_min
            # Optionally restart with higher T if stuck? 
            # For now, just keep going with low T (local search)
            
    return best_colors

# EVOLVE-BLOCK-END
