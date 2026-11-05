import time
import random
import math
import sys

def construct():
    """
    Find a 5-coloring of K_n with no monochromatic C4, maximizing n.
    Uses incremental energy update for fast local search.
    """
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
    
    for target_n in range(8, 29):
        if time_left() < 10:
            break
        
        n = target_n
        num_edges = n * (n - 1) // 2
        
        # Precompute edge list
        edge_list = []
        for i in range(n):
            for j in range(i + 1, n):
                edge_list.append((i, j, edge_idx(i, j, n)))
        
        found = False
        max_restarts = 200
        
        for restart in range(max_restarts):
            if time_left() < 5:
                break
            
            colors = [random.randint(0, 4) for _ in range(num_edges)]
            
            adj = [[0] * n for _ in range(5)]
            idx = 0
            for i in range(n):
                for j in range(i + 1, n):
                    c = colors[idx]
                    adj[c][i] |= (1 << j)
                    adj[c][j] |= (1 << i)
                    idx += 1
            
            energy = compute_energy(adj, n)
            
            if energy == 0:
                best_n = n
                best_colors = colors[:]
                found = True
                break
            
            best_energy = energy
            best_colors_local = colors[:]
            
            temp = 2.0
            cooling = 0.99995
            min_temp = 0.05
            max_steps = 500000
            stuck_count = 0
            max_stuck = 20000
            
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
                
                # Calculate delta
                delta = 0
                
                # Remove from c_old
                # For all w != u, v
                # Check if v in N_c_old(w) -> update pair (u, w)
                # Check if u in N_c_old(w) -> update pair (v, w)
                
                # Optimization: Iterate over w
                for w in range(n):
                    if w == u or w == v:
                        continue
                    
                    mask_w = adj[c_old][w]
                    
                    # If v is neighbor of w in c_old
                    if mask_w & (1 << v):
                        k = (adj[c_old][u] & mask_w).bit_count()
                        delta -= (k - 1)
                    
                    # If u is neighbor of w in c_old
                    if mask_w & (1 << u):
                        k = (adj[c_old][v] & mask_w).bit_count()
                        delta -= (k - 1)
                
                # Add to c_new
                for w in range(n):
                    if w == u or w == v:
                        continue
                    
                    mask_w = adj[c_new][w]
                    
                    # If v is neighbor of w in c_new
                    if mask_w & (1 << v):
                        k = (adj[c_new][u] & mask_w).bit_count()
                        delta += k
                    
                    # If u is neighbor of w in c_new
                    if mask_w & (1 << u):
                        k = (adj[c_new][v] & mask_w).bit_count()
                        delta += k
                
                new_energy = energy + delta
                
                if delta <= 0 or random.random() < math.exp(-delta / temp):
                    # Accept
                    energy = new_energy
                    
                    # Update adj
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

        if found:
            continue

    return {'n': best_n, 'colors': best_colors}

if __name__ == '__main__':
    result = construct()
    print(f"Found n = {result['n']}")
