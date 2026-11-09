import time
import numpy as np

def construct():
    N = 256
    start_time = time.time()
    time_limit = 120.0
    
    # Precompute all 4-APs
    # A 4-AP is (a, d) such that a, a+d, a+2d, a+3d < N
    aps = []
    # pos_to_aps[i] will store indices into 'aps' list for APs containing i
    pos_to_aps = [[] for _ in range(N)]
    
    for d in range(1, (N - 3) // 3 + 1):
        for a in range(N - 3 * d):
            ap = (a, a + d, a + 2 * d, a + 3 * d)
            idx = len(aps)
            aps.append(ap)
            for pos in ap:
                pos_to_aps[pos].append(idx)
                
    # Helper to count monochromatic APs
    def count_mono(colors):
        count = 0
        for ap in aps:
            c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
            if c0 == c1 == c2 == c3:
                count += 1
        return count

    # Thue-Morse initialization
    def thue_morse(n):
        return bin(n).count('1') % 2

    best_colors = np.array([thue_morse(i) for i in range(N)], dtype=np.int8)
    best_cost = count_mono(best_colors)
    
    # Local search: Hill Climbing with incremental update
    # We maintain 'colors' and 'cost'.
    # For each position i, we can compute delta if we flip i.
    
    # To make it faster, we can precompute the "state" of each AP.
    # But recalculating delta on the fly is O(degree(i)).
    # Degree of a position in 4-AP graph is roughly O(N) in worst case?
    # Number of APs is ~ N^2/6. Each AP has 4 nodes. Total incidences ~ 4 * N^2/6 = 2/3 N^2.
    # Average degree ~ 2/3 N. For N=256, avg degree ~ 170.
    # Calculating delta for all 256 positions takes 256 * 170 ~ 43k operations.
    # This is very fast. We can do many iterations.
    
    def get_delta(colors, i):
        """Calculate change in cost if we flip colors[i]."""
        old_c = colors[i]
        new_c = 1 - old_c
        delta = 0
        for ap_idx in pos_to_aps[i]:
            ap = aps[ap_idx]
            # Get the other 3 positions
            others = [p for p in ap if p != i]
            # Count how many others match old_c and new_c
            n_old = 0
            n_new = 0
            for p in others:
                if colors[p] == old_c:
                    n_old += 1
                else:
                    n_new += 1
            
            # Before flip:
            # Mono if all 4 are old_c. Since colors[i] is old_c, this happens if n_old == 3.
            before = 1 if n_old == 3 else 0
            
            # After flip:
            # colors[i] becomes new_c.
            # Mono if all 4 are new_c. This happens if n_new == 3.
            after = 1 if n_new == 3 else 0
            
            delta += after - before
        return delta

    def hill_climb(colors):
        colors = colors.copy()
        cost = count_mono(colors)
        improved = True
        while improved:
            improved = False
            best_delta = 0
            best_i = -1
            for i in range(N):
                d = get_delta(colors, i)
                if d < best_delta:
                    best_delta = d
                    best_i = i
            if best_delta < 0:
                colors[best_i] = 1 - colors[best_i]
                cost += best_delta
                improved = True
        return colors, cost

    # Run hill climb from Thue-Morse
    colors, cost = hill_climb(best_colors)
    if cost < best_cost:
        best_cost = cost
        best_colors = colors

    # Try some random restarts / annealing kicks if time permits
    # We'll do a simple loop: pick random k bits, flip them, hill climb, keep best.
    # k=1 is just hill climbing from a neighbor. k=2,3... helps escape.
    # Let's try k up to 5 or so.
    
    import random
    random.seed(42)
    
    current_colors = best_colors.copy()
    current_cost = best_cost
    
    # We can also try to optimize the hill climbing by checking if we can just
    # do a few passes of "flip if beneficial".
    
    while time.time() - start_time < time_limit - 10:
        # Pick a random perturbation size
        # Small perturbations are more likely to find local improvements or nearby basins
        # Larger perturbations explore more.
        k = random.randint(1, 5)
        new_colors = current_colors.copy()
        for _ in range(k):
            i = random.randint(0, N - 1)
            new_colors[i] = 1 - new_colors[i]
        
        # Hill climb from this new state
        res_colors, res_cost = hill_climb(new_colors)
        
        if res_cost < best_cost:
            best_cost = res_cost
            best_colors = res_colors
            current_colors = best_colors.copy()
            current_cost = best_cost
        else:
            # If we didn't improve, we might want to stick to the best found so far
            # or accept worse states (annealing). 
            # Simple hill climbing from random kicks might get stuck.
            # Let's just keep the best found.
            pass
            
    return best_colors.tolist()
