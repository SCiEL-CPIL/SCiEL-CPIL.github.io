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
    
    # Try high n first
    for target_n in range(26, 10, -1):
        if time_left() < 10:
            break
        
        n = target_n
        num_edges = n * (n - 1) // 2
        
        edge_list = []
        for i in range(n):
            for j in range(i + 1, n):
                edge_list.append((i, j, edge_idx(i, j, n)))
        
        found = False
        
        # More restarts for higher n if time permits
        max_restarts = 100
        if time_left() > 60:
            max_restarts = 200
        else:
            max_restarts = 50

        for restart in range(max_restarts):
            if time_left() < 5:
                break
            
            # Greedy Initialization with random tie-breaking
            colors = [0] * num_edges
            adj = [[0] * n for _ in range(5)]
            
            edge_indices = list(range(num_edges))
            random.shuffle(edge_indices)
            
            for idx in edge_indices:
                u, v, e_idx = edge_list[idx]
                best_c = 0
                min_common = float('inf')
                
                for c in range(5):
                    common = adj[c][u] & adj[c][v]
                    cnt = common.bit_count()
                    if cnt < min_common:
                        min_common = cnt
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
            
            best_energy = energy
            best_colors_local = colors[:]
            
            # Simulated Annealing
            temp = 5.0 # Higher initial temp
            cooling = 0.9995
            min_temp = 0.01
            max_steps = 100000
            stuck_count = 0
            max_stuck = 10000
            
            for step in range(max_steps):
                if time_left() < 3:
                    break
                if temp < min_temp:
                    break
                if stuck_count > max_stuck:
                    if best_energy < energy:
                        for i in range(num_edges):
                            colors[i] = best_colors_local[i]
                        for c in range(5):
                            for i in range(n):
                                adj[c][i] = 0
                        for i in range(num_edges):
                            u, v, _ = edge_list[i]
                            c = colors[i]
                            adj[c][u] |= (1 << v)
                            adj[c][v] |= (1 << u)
                        energy = best_energy
                        temp = 5.0
                        stuck_count = 0
                    else:
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
            break

    return {'n': best_n, 'colors': best_colors}

if __name__ == '__main__':
    result = construct()
    print(f"Found n = {result['n']}")
