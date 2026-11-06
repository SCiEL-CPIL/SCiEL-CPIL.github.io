import time
import random
import math

def construct():
    """
    Find a 5-coloring of K_n with no monochromatic C4, maximizing n.
    Uses Simulated Annealing with incremental energy updates.
    """
    deadline = time.time() + 115
    
    def time_left():
        return deadline - time.time()
    
    def edge_idx(i, j, n):
        return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)
    
    def build_adj(colors, n):
        adj = [[0] * n for _ in range(5)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                idx += 1
        return adj
    
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
        
    def delta_energy(adj, n, u, v, c_old, c_new):
        # Calculate change in energy when changing edge (u,v) from c_old to c_new
        delta = 0
        
        # Remove (u,v) from c_old
        # Count C4s in c_old containing (u,v)
        # For each w neighbor of v in c_old (w != u), count common neighbors of u and w in c_old (excluding v)
        # This is equivalent to: sum_{w in N_c(v)\{u}} (|N_c(u) \cap N_c(w)| - 1)
        # Note: v is always in N_c(u) \cap N_c(w) if w is neighbor of v?
        # If w is neighbor of v, and u is neighbor of v, then v is common neighbor of u and w.
        # So |N_c(u) \cap N_c(w)| includes v.
        # We want common neighbors OTHER than v.
        # So (bit_count - 1).
        
        mask_v = adj[c_old][v]
        mask_u = adj[c_old][u]
        
        for w in range(n):
            if w == u or w == v:
                continue
            if mask_v & (1 << w):
                common = mask_u & adj[c_old][w]
                k = common.bit_count()
                # v is in common, so subtract 1
                delta -= (k - 1)
                
        # Add (u,v) to c_new
        # Count C4s in c_new that will be formed
        # For each w neighbor of v in c_new (w != u), count common neighbors of u and w in c_new
        # Note: v is NOT yet added to adj[c_new].
        # So v is NOT in N_c_new(u) or N_c_new(w).
        # So we just count |N_c_new(u) \cap N_c_new(w)|.
        
        mask_v_new = adj[c_new][v]
        mask_u_new = adj[c_new][u]
        
        for w in range(n):
            if w == u or w == v:
                continue
            if mask_v_new & (1 << w):
                common = mask_u_new & adj[c_new][w]
                k = common.bit_count()
                delta += k
                
        return delta

    best_n = 1
    best_colors = [0]
    
    # Try n=26 first, as it's a known bound
    target_ns = [26, 27, 25, 28]
    
    for n in target_ns:
        if time_left() < 20:
            break
            
        if n <= best_n:
            continue
            
        num_edges = n * (n - 1) // 2
        found = False
        
        # Multiple restarts
        max_restarts = 10 if n <= 26 else 5
        
        for restart in range(max_restarts):
            if time_left() < 10:
                break
                
            # Initialize
            colors = [random.randint(0, 4) for _ in range(num_edges)]
            adj = build_adj(colors, n)
            energy = compute_energy(adj, n)
            
            if energy == 0:
                best_n = n
                best_colors = colors[:]
                found = True
                break
            
            best_energy = energy
            best_colors_local = colors[:]
            
            temp = 1.0
            cooling = 0.9999
            min_temp = 0.01
            max_steps = 50000
            stuck_count = 0
            max_stuck = 5000
            
            for step in range(max_steps):
                if time_left() < 5:
                    break
                if temp < min_temp:
                    break
                if stuck_count > max_stuck:
                    break
                    
                ei = random.randint(0, num_edges - 1)
                # Decode ei to u, v
                # Faster: precompute edge list
                # For speed, we can just pick u, v randomly
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u == v:
                    continue
                if u > v:
                    u, v = v, u
                
                e_idx = edge_idx(u, v, n)
                c_old = colors[e_idx]
                c_new = random.randint(0, 4)
                if c_new == c_old:
                    continue
                
                delta = delta_energy(adj, n, u, v, c_old, c_new)
                new_energy = energy + delta
                
                if delta <= 0 or random.random() < math.exp(-delta / temp):
                    energy = new_energy
                    adj[c_old][u] &= ~(1 << v)
                    adj[c_old][v] &= ~(1 << u)
                    adj[c_new][u] |= (1 << v)
                    adj[c_new][v] |= (1 << u)
                    colors[e_idx] = c_new
                    
                    if energy < best_energy:
                        best_energy = energy
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
            
            # If not found, keep the best local solution for next restart?
            # Or just restart fresh.
            
        if found:
            continue
            
    return {'n': best_n, 'colors': best_colors}

if __name__ == '__main__':
    result = construct()
    print(f"Found n = {result['n']}")
