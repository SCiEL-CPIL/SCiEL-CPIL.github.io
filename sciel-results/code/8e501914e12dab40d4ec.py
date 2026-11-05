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
    best_colors = [] # Fixed: empty list for n=1
    best_energy_global = float('inf')

    # Time allocation
    # Prioritize 26, but keep enough time for fallbacks
    time_alloc = {
        26: 0.6,
        25: 0.25,
        24: 0.1,
        23: 0.03,
        22: 0.02
    }
    
    # Targets
    targets = [26, 25, 24, 23, 22]
    
    # We need to track time spent per target to respect allocation?
    # Or just use a global deadline and check time_left before starting each target?
    # The current code uses `time_left() < 2` to break.
    # Let's stick to that but ensure we don't waste too much time on 26 if it's failing.
    
    for target_n in targets:
        if time_left() < 5:
            break
            
        n = target_n
        num_edges = n * (n - 1) // 2
        
        # Precompute edge list
        edge_list = []
        for i in range(n):
            for j in range(i + 1, n):
                edge_list.append((i, j, edge_idx(i, j, n)))
                
        found = False
        
        # Dynamic restarts based on time
        # Estimate: optimized SA is ~5x faster.
        # Previous: 20 restarts. Now maybe 50?
        # Let's say 1 restart ~ 0.05s (optimistic).
        # 115s * 0.6 = 69s for 26.
        # 69 / 0.05 = 1380 restarts?
        # That's a lot. Let's cap it.
        
        max_restarts = 100
        if time_left() < 30:
            max_restarts = 20
        if time_left() < 10:
            max_restarts = 5

        for restart in range(max_restarts):
            if time_left() < 5:
                break
            
            colors = [0] * num_edges
            adj = [[0] * n for _ in range(5)]
            
            # Greedy Initialization
            edge_indices = list(range(num_edges))
            random.shuffle(edge_indices)
            
            for idx in edge_indices:
                u, v, e_idx = edge_list[idx]
                best_c = 0
                min_delta = float('inf')
                
                for c in range(5):
                    delta = 0
                    # Iterate neighbors of u and v in color c
                    # Actually, we need to check C4s formed by adding (u,v) to c.
                    # C4s: u-v-w-x-u.
                    # w is neighbor of v in c.
                    # x is neighbor of w and u in c.
                    
                    mask_v = adj[c][v]
                    mask_v &= ~(1 << u) # u is not neighbor yet
                    
                    m = mask_v
                    while m:
                        lsb = m & -m
                        w = lsb.bit_length() - 1
                        m ^= lsb
                        
                        # w is neighbor of v
                        # Count common neighbors of u and w
                        common = adj[c][u] & adj[c][w]
                        delta += common.bit_count()
                        
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
            temp = 5.0
            cooling = 0.99995
            min_temp = 0.01
            max_steps = 100000 # Increased steps due to speedup
            stuck_count = 0
            max_stuck = 10000
            
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
                # Break C4s using (u,v) in c_old
                # C4s: u-v-w-x-u
                # w is neighbor of v in c_old (excluding u)
                mask_v_old = adj[c_old][v]
                mask_v_old &= ~(1 << u)
                
                m = mask_v_old
                while m:
                    lsb = m & -m
                    w = lsb.bit_length() - 1
                    m ^= lsb
                    
                    # w is neighbor of v
                    # Count common neighbors of u and w in c_old
                    # One common neighbor is v? 
                    # adj[c_old][u] has v. adj[c_old][w] has v.
                    # So common includes v.
                    # Valid x excludes v.
                    common = adj[c_old][u] & adj[c_old][w]
                    k = common.bit_count()
                    delta -= (k - 1)
                
                # Add to c_new
                # Form C4s using (u,v) in c_new
                # C4s: u-v-w-x-u
                # w is neighbor of v in c_new (excluding u, since u-v not in c_new yet)
                mask_v_new = adj[c_new][v]
                mask_v_new &= ~(1 << u)
                
                m = mask_v_new
                while m:
                    lsb = m & -m
                    w = lsb.bit_length() - 1
                    m ^= lsb
                    
                    # w is neighbor of v
                    # Count common neighbors of u and w in c_new
                    # v is NOT neighbor of u in c_new (yet).
                    # So common does NOT include v.
                    common = adj[c_new][u] & adj[c_new][w]
                    delta += common.bit_count()
                
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
            break
        # If not found, loop continues to next smaller n

    return {'n': best_n, 'colors': best_colors}

if __name__ == '__main__':
    result = construct()
    print(f"Found n = {result['n']}")
