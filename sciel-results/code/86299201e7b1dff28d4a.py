import time
import random

def count_mono_ap(colors):
    """Count monochromatic 4-term arithmetic progressions."""
    n = len(colors)
    count = 0
    # d max is (n-1)//3
    for d in range(1, (n - 1) // 3 + 1):
        # a max is n - 3*d
        for a in range(n - 3 * d):
            if colors[a] == colors[a + d] == colors[a + 2 * d] == colors[a + 3 * d]:
                count += 1
    return count

def construct():
    n = 256
    max_time = 110  # seconds, leaving buffer for overhead
    start_time = time.time()
    
    # Precompute for each index i, the list of 4-APs (represented by the other 3 indices)
    # that contain i.
    # An AP is defined by (a, d). Indices are a, a+d, a+2d, a+3d.
    # We store for each index i, a list of tuples (j, k, l) where {j,k,l} are the other indices in the AP.
    aps_involving = [[] for _ in range(n)]
    
    for d in range(1, (n - 1) // 3 + 1):
        for a in range(n - 3 * d):
            idx0 = a
            idx1 = a + d
            idx2 = a + 2 * d
            idx3 = a + 3 * d
            
            # For each index in the AP, store the other three
            aps_involving[idx0].append((idx1, idx2, idx3))
            aps_involving[idx1].append((idx0, idx2, idx3))
            aps_involving[idx2].append((idx0, idx1, idx3))
            aps_involving[idx3].append((idx0, idx1, idx2))
            
    def get_delta(colors, i):
        """Calculate the change in score if we flip colors[i]."""
        c_i = colors[i]
        delta = 0
        for (j, k, l) in aps_involving[i]:
            c_j, c_k, c_l = colors[j], colors[k], colors[l]
            if c_j == c_k == c_l:
                if c_j == c_i:
                    # Currently mono, will become non-mono
                    delta -= 1
                else:
                    # Currently non-mono, will become mono
                    delta += 1
        return delta

    best_colors = None
    best_score = float('inf')
    
    # Run multiple simulated annealing instances
    num_runs = 10
    for run in range(num_runs):
        # Initialize with random colors
        colors = [random.randint(0, 1) for _ in range(n)]
        current_score = count_mono_ap(colors)
        
        if current_score < best_score:
            best_score = current_score
            best_colors = colors[:]
            
        # Simulated Annealing parameters
        # Start temperature high enough to accept bad moves
        T = 10.0
        T_min = 0.1
        cooling_rate = 0.99
        
        # Number of steps per run depends on time
        # We will loop until time runs out, but we need to manage the loop carefully
        while time.time() - start_time < max_time:
            # Pick a random index to flip
            i = random.randint(0, n - 1)
            
            delta = get_delta(colors, i)
            
            if delta < 0:
                # Accept
                colors[i] ^= 1
                current_score += delta
            else:
                # Accept with probability exp(-delta / T)
                if random.random() < pow(2.718281828, -delta / T):
                    colors[i] ^= 1
                    current_score += delta
            
            # Cool down
            T *= cooling_rate
            if T < T_min:
                # Restart with higher temperature to escape local minima?
                # Or just break and start a new run?
                # Let's break and start a new run if we have time for another full run
                # But we need to check time.
                # If we just broke out of the inner while, we go to the next run.
                break
                
        if current_score < best_score:
            best_score = current_score
            best_colors = colors[:]
            
        # Check if we have time for more runs
        if time.time() - start_time > max_time - 5:
            break
            
    return best_colors
