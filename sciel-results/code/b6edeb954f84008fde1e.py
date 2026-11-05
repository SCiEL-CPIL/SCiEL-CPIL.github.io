import time
import random
import math

def construct():
    deadline = time.time() + 115
    
    def time_left():
        return deadline - time.time()

    def edge_idx(i, j, n):
        return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)

    def compute_energy(adj, n):
        total = 0
        for c in range(5):
            for i in range(n):
                for j in range(i + 1, n):
                    common = adj[c][i] & adj[c][j]
                    k = common.bit_count()
                    if k >= 2:
                        total += k * (k - 1) // 2
        return total

    best_n = 1
    best_colors = [0]
    best_energy_global = float('inf')

    # Try high n first to maximize score
    # Allocate most time to 26
    time_alloc = {
        26: 0.8,
        25: 0.1,
        24: 0.05,
        23: 0.03,
        22: 0.02
    }
    
    # We will iterate through targets, but break early if we find a solution for the current target.
    # However, if we don't find 26, we should keep the best found so far.
    
    for target_n in [26, 25, 24, 23, 22]:
        if time_left() < 2:
            break
        
        n = target_n
        num_edges = n * (n - 1) // 2
        
        # Precompute edge list
        edge_list = []
        for i in range(n):
            for j in range(i + 1, n):
                edge_list.append((i, j, edge_idx(i, j, n)))
        
        found = False
        
        # Number of restarts depends on time left
        # Estimate time per restart.
        # Greedy init is O(n^2 * n) = O(n^3).
        # SA steps: Let's say 50k steps per restart.
        # Each step is O(n).
        # Total ops ~ 50k * 26 ~ 1.3M ops.
        # Python is slow, maybe 0.1s per restart?
        # Let's try 10-20 restarts.
        
        max_restarts = 20
        if time_left() < 30:
            max_restarts = 5
            
        for restart in range(max_restarts):
            if time_left() < 5:
                break
            
            colors = [0] * num_edges
            adj = [[0] * n for _ in range(5)]
            
            # Greedy Initialization
            # Sort edges by some heuristic? Or just random order?
            # Random order is better to avoid bias.
            edge_indices = list(range(num_edges))
            random.shuffle(edge_indices)
            
            for idx in edge_indices:
                u, v, e_idx = edge_list[idx]
                best_c = 0
                min_delta = float('inf')
                
                for c in range(5):
                    # Calculate energy increase if we add (u,v) with color c
                    # This is the same as the "add to c_new" part of the delta calculation
                    # assuming the edge is not currently in any color.
                    delta = 0
                    for w in range(n):
                        if w == u or w == v:
                            continue
                        mask_w = adj[c][w]
                        if mask_w & (1 << v):
                            k = (adj[c][u] & mask_w).bit_count()
                            delta += k
                        if mask_w & (1 << u):
                            k = (adj[c][v] & mask_w).bit_count()
                            delta += k
                    
                    if delta < min_delta:
                        min_delta = delta
                        best_c = c
                
                c = best_c
                colors[idx] = c
                adj[c][u] |= (1 << v)
                adj[c][v] |= (1 << u)
            
            energy = compute_energy(adj, n)
            
            if energy == 0:
                best_n = n
                best_colors = colors[:]
                found = True
                break
            
            best_energy_local = energy
            best_colors_local = colors[:]
            
            # Simulated Annealing
            temp = 5.0 # Higher temp
            cooling = 0.99995
            min_temp = 0.01
            max_steps = 50000 # More steps
            stuck_count = 0
            max_stuck = 5000
            
            for step in range(max_steps):
                if time_left() < 3:
                    break
                if temp < min_temp:
                    break
                if stuck_count > max_stuck:
                    break
                
                ei = random.randint(0, num_edges - 1)
                u, v, e_idx = edge_list[ei]
                c_old = colors[e_idx]
                c_new = random.randint(0, 4)
                if c_new == c_old:
                    continue
                
                delta = 0
                
                # Remove from c_old
                for w in range(n):
                    if w == u or w == v:
                        continue
                    mask_w = adj[c_old][w]
                    if mask_w & (1 << v):
                        k = (adj[c_old][u] & mask_w).bit_count()
                        delta -= (k - 1)
                    if mask_w & (1 << u):
                        k = (adj[c_old][v] & mask_w).bit_count()
                        delta -= (k - 1)
                
                # Add to c_new
                for w in range(n):
                    if w == u or w == v:
                        continue
                    mask_w = adj[c_new][w]
                    if mask_w & (1 << v):
                        k = (adj[c_new][u] & mask_w).bit_count()
                        delta += k
                    if mask_w & (1 << u):
                        k = (adj[c_new][v] & mask_w).bit_count()
                        delta += k
                
                new_energy = energy + delta
                
                if delta <= 0 or random.random() < math.exp(-delta / temp):
                    energy = new_energy
                    adj[c_old][u] &= ~(1 << v)
                    adj[c_old][v] &= ~(1 << u)
                    adj[c_new][u] |= (1 << v)
                    adj[c_new][v] |= (1 << u)
                    colors[e_idx] = c_new
                    
                    if energy < best_energy_local:
                        best_energy_local = energy
                        best_colors_local = colors[:]
                        stuck_count = 0
                        if energy == 0:
                            best_n = n
                            best_colors = colors[:]
                            found = True
                            break
                else:
                    stuck_count += 1
                
                temp *= cooling
            
            if found:
                break
        
        if found:
            # Found a solution for this n, return it immediately as it's the max possible we're trying
            # (since we go down from 26)
            break
        else:
            # If we didn't find a valid coloring for this n, 
            # but we might have found a "good" one with low energy?
            # The task requires a VALID coloring (energy 0).
            # So if energy > 0, it's invalid.
            # We must return a valid coloring.
            # If we didn't find one for n, we try n-1.
            pass

    return {'n': best_n, 'colors': best_colors}

if __name__ == '__main__':
    result = construct()
    print(f"Found n = {result['n']}")
