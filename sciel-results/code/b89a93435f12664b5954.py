import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    def lazebnik_woldar_coloring(n):
        """
        Generate an initial coloring based on algebraic structure.
        For n <= 26, use a structured approach; for larger n, random.
        """
        if n <= 26:
            # Use a simple modular structure as starting point
            colors = []
            for i in range(n):
                for j in range(i + 1, n):
                    d = (j - i) % 5
                    colors.append(d)
            return colors
        else:
            num_edges = n * (n - 1) // 2
            return [random.randint(0, 4) for _ in range(num_edges)]
    
    def count_violations(colors, n):
        """
        Calculate total violations.
        """
        if len(colors) != n * (n - 1) // 2:
            return float('inf')
        
        neighbors = [ [set() for _ in range(5)] for _ in range(n) ]
        
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                neighbors[i][c].add(j)
                neighbors[j][c].add(i)
                idx += 1
        
        total_cost = 0.0
        for c in range(5):
            for u in range(n):
                for v in range(u + 1, n):
                    common = len(neighbors[u][c] & neighbors[v][c])
                    if common > 1:
                        total_cost += common * (common - 1) / 2.0
        return total_cost

    def solve_n(n, initial_colors=None, max_restarts=30):
        num_edges = n * (n - 1) // 2
        best_cost_found = float('inf')
        best_colors = None
        
        # Precompute edge endpoints and indices
        edge_index = {}
        endpoints = []
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_index[(i, j)] = idx
                endpoints.append((i, j))
                idx += 1

        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            if initial_colors and restart == 0:
                colors = initial_colors.copy()
            else:
                # Use structured initial coloring for better starting point
                if n <= 26:
                    colors = lazebnik_woldar_coloring(n)
                else:
                    colors = [random.randint(0, 4) for _ in range(num_edges)]
            
            # Initialize neighbors and common_counts
            neighbors = [ [set() for _ in range(5)] for _ in range(n) ]
            for e_idx in range(num_edges):
                u, v = endpoints[e_idx]
                c = colors[e_idx]
                neighbors[u][c].add(v)
                neighbors[v][c].add(u)
            
            common_counts = [ [[0]*n for _ in range(n)] for _ in range(5) ]
            
            for c in range(5):
                for w in range(n):
                    nbrs = list(neighbors[w][c])
                    for i in range(len(nbrs)):
                        u = nbrs[i]
                        for j in range(i + 1, len(nbrs)):
                            v = nbrs[j]
                            common_counts[c][u][v] += 1
                            common_counts[c][v][u] += 1
            
            current_cost = 0.0
            for c in range(5):
                for u in range(n):
                    for v in range(u + 1, n):
                        cnt = common_counts[c][u][v]
                        if cnt > 1:
                            current_cost += cnt * (cnt - 1) / 2.0
            
            if current_cost == 0:
                return colors, n
            
            # Simulated annealing
            temperature = max(current_cost * 0.1, 1.0)
            cooling_rate = 0.999
            min_temperature = 0.01
            max_iter = 50000
            
            for i in range(max_iter):
                if time.time() > deadline - 5:
                    if best_cost_found > current_cost:
                        best_cost_found = current_cost
                        best_colors = colors.copy()
                    break
                    
                if current_cost == 0:
                    return colors, n
                
                if temperature < min_temperature:
                    break
                    
                # Pick random edge
                e_idx = random.randint(0, num_edges - 1)
                u, v = endpoints[e_idx]
                old_c = colors[e_idx]
                new_c = random.randint(0, 4)
                if old_c == new_c:
                    continue
                
                # Calculate delta cost
                delta = 0.0
                
                # 1. Remove edge (u,v) from old_c
                for w in range(n):
                    if w == u or w == v:
                        continue
                    
                    key_vw = (v, w) if v < w else (w, v)
                    idx_vw = edge_index[key_vw]
                    if colors[idx_vw] == old_c:
                        cnt = common_counts[old_c][u][w]
                        delta -= (cnt - 1)
                        common_counts[old_c][u][w] -= 1
                        common_counts[old_c][w][u] -= 1
                    
                    key_uw = (u, w) if u < w else (w, u)
                    idx_uw = edge_index[key_uw]
                    if colors[idx_uw] == old_c:
                        cnt = common_counts[old_c][v][w]
                        delta -= (cnt - 1)
                        common_counts[old_c][v][w] -= 1
                        common_counts[old_c][w][v] -= 1
                        
                # 2. Add edge (u,v) to new_c
                for w in range(n):
                    if w == u or w == v:
                        continue
                    
                    key_vw = (v, w) if v < w else (w, v)
                    idx_vw = edge_index[key_vw]
                    if colors[idx_vw] == new_c:
                        cnt = common_counts[new_c][u][w]
                        delta += cnt
                        common_counts[new_c][u][w] += 1
                        common_counts[new_c][w][u] += 1
                        
                    key_uw = (u, w) if u < w else (w, u)
                    idx_uw = edge_index[key_uw]
                    if colors[idx_uw] == new_c:
                        cnt = common_counts[new_c][v][w]
                        delta += cnt
                        common_counts[new_c][v][w] += 1
                        common_counts[new_c][w][v] += 1
                        
                new_cost = current_cost + delta
                
                # Accept with probability based on temperature
                if delta < 0 or random.random() < np.exp(-delta / temperature):
                    current_cost = new_cost
                    colors[e_idx] = new_c
                    neighbors[u][old_c].remove(v)
                    neighbors[v][old_c].remove(u)
                    neighbors[u][new_c].add(v)
                    neighbors[v][new_c].add(u)
                    
                    if current_cost < best_cost_found:
                        best_cost_found = current_cost
                        best_colors = colors.copy()
                else:
                    # Revert common_counts changes
                    for w in range(n):
                        if w == u or w == v:
                            continue
                        key_vw = (v, w) if v < w else (w, v)
                        idx_vw = edge_index[key_vw]
                        if colors[idx_vw] == old_c:
                            common_counts[old_c][u][w] += 1
                            common_counts[old_c][w][u] += 1
                            
                        key_uw = (u, w) if u < w else (w, u)
                        idx_uw = edge_index[key_uw]
                        if colors[idx_uw] == old_c:
                            common_counts[old_c][v][w] += 1
                            common_counts[old_c][w][v] += 1
                            
                    for w in range(n):
                        if w == u or w == v:
                            continue
                        key_vw = (v, w) if v < w else (w, v)
                        idx_vw = edge_index[key_vw]
                        if colors[idx_vw] == new_c:
                            common_counts[new_c][u][w] -= 1
                            common_counts[new_c][w][u] -= 1
                            
                        key_uw = (u, w) if u < w else (w, u)
                        idx_uw = edge_index[key_uw]
                        if colors[idx_uw] == new_c:
                            common_counts[new_c][v][w] -= 1
                            common_counts[new_c][w][v] -= 1
                
                temperature *= cooling_rate
        
        return best_colors, n if best_colors else None

    # Try to find solutions for increasing n
    current_colors = None
    best_n = 7
    
    # Start with small n and build up
    for n in range(7, 25):
        if time.time() > deadline - 10:
            break
            
        res_colors, res_n = solve_n(n, None)
        
        if res_colors:
            best_n = n
            current_colors = res_colors
        else:
            # If we fail, keep trying with more restarts for a bit
            if n <= 15:
                res_colors, res_n = solve_n(n, None, max_restarts=50)
                if res_colors:
                    best_n = n
                    current_colors = res_colors
                else:
                    break
            else:
                break

    # Fallback if nothing found
    if best_n < 7:
        n_fallback = 7
        colors_fallback = [random.randint(0, 4) for _ in range(n_fallback * (n_fallback - 1) // 2)]
        res_colors, _ = solve_n(n_fallback, colors_fallback)
        if res_colors:
            best_n = n_fallback
            current_colors = res_colors
        else:
            current_colors = colors_fallback

    return {'n': best_n, 'colors': current_colors}
