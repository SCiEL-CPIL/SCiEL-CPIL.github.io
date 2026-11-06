import time
import random
import math
import sys
import numpy as np

def construct():
    """
    Find a 5-coloring of K_n with no monochromatic C4, maximizing n.
    Uses a greedy algorithm with C4-potential minimization, followed by local search.
    """
    deadline = time.time() + 115
    
    def time_left():
        return deadline - time.time()
    
    def edge_idx(i, j, n):
        return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)

    def calculate_cost(adj_matrices, u, v, n):
        """
        Calculate the number of new C4s created if edge (u,v) is added to color c.
        This is the number of paths of length 3 from u to v in the graph of color c.
        """
        costs = [0] * 5
        for c in range(5):
            A = adj_matrices[c]
            # (A^3)[u, v] = sum_{x, y} A[u, x] A[x, y] A[y, v]
            # Optimization: Precompute A^2? No, just loop.
            # n is small.
            path_count = 0
            Au = A[u]
            for x in range(n):
                if Au[x]:
                    Ax = A[x]
                    for y in range(n):
                        if Ax[y] and A[y, v]:
                            path_count += 1
            costs[c] = path_count
        return costs

    def is_valid(adj_bitmasks, n):
        """
        Check if any color class contains a C4.
        C4 exists if there are two vertices u, w with at least 2 common neighbors in color c.
        """
        for c in range(5):
            adj_c = adj_bitmasks[c]
            for u in range(n):
                for w in range(u + 1, n):
                    common = adj_c[u] & adj_c[w]
                    if common.bit_count() >= 2:
                        return False
        return True

    def calc_energy(adj_bitmasks, n):
        total = 0
        for c in range(5):
            adj_c = adj_bitmasks[c]
            for u in range(n):
                for v in range(u + 1, n):
                    k = (adj_c[u] & adj_c[v]).bit_count()
                    if k >= 2:
                        total += k * (k - 1) // 2
        return total

    def greedy_coloring(n, seed):
        random.seed(seed)
        adj_matrices = [np.zeros((n, n), dtype=int) for _ in range(5)]
        adj_bitmasks = [[0] * n for _ in range(5)]
        
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(edges)
        
        num_edges = n * (n - 1) // 2
        colors = [0] * num_edges
        edge_idx_map = {}
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_idx_map[(i, j)] = idx
                idx += 1

        for u, v in edges:
            costs = calculate_cost(adj_matrices, u, v, n)
            min_cost = min(costs)
            candidates = [c for c in range(5) if costs[c] == min_cost]
            c = random.choice(candidates)
            
            # Add edge
            adj_matrices[c][u, v] = 1
            adj_matrices[c][v, u] = 1
            adj_bitmasks[c][u] |= (1 << v)
            adj_bitmasks[c][v] |= (1 << u)
            colors[edge_idx_map[(u, v)]] = c
            
        return colors, adj_bitmasks

    def local_search(colors, adj_bitmasks, n, time_limit):
        start_time = time.time()
        energy = calc_energy(adj_bitmasks, n)
        
        if energy == 0:
            return colors, adj_bitmasks, 0

        best_energy = energy
        best_colors = colors[:]
        best_adj = [row[:] for row in adj_bitmasks]
        
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        while time.time() - start_time < time_limit:
            if energy == 0:
                break
                
            u, v = random.choice(edges)
            e_idx = edge_idx(u, v, n)
            c_old = colors[e_idx]
            
            best_delta = 10**9
            best_c_new = -1
            
            for c_new in range(5):
                if c_new == c_old:
                    continue
                
                delta = 0
                adj_u_old = adj_bitmasks[c_old][u]
                adj_v_old = adj_bitmasks[c_old][v]
                
                for w in range(n):
                    if w == u or w == v:
                        continue
                    mask_w = adj_bitmasks[c_old][w]
                    
                    if mask_w & (1 << v):
                        k = (adj_u_old & mask_w).bit_count()
                        delta -= (k - 1)
                        
                    if mask_w & (1 << u):
                        k = (adj_v_old & mask_w).bit_count()
                        delta -= (k - 1)
                        
                adj_u_new = adj_bitmasks[c_new][u]
                adj_v_new = adj_bitmasks[c_new][v]
                
                for w in range(n):
                    if w == u or w == v:
                        continue
                    mask_w = adj_bitmasks[c_new][w]
                    
                    if mask_w & (1 << v):
                        k = (adj_u_new & mask_w).bit_count()
                        delta += (k + 1)
                        
                    if mask_w & (1 << u):
                        k = (adj_v_new & mask_w).bit_count()
                        delta += (k + 1)
                        
                if delta < best_delta:
                    best_delta = delta
                    best_c_new = c_new
                    
            if best_c_new != -1 and best_delta < 0:
                adj_bitmasks[c_old][u] &= ~(1 << v)
                adj_bitmasks[c_old][v] &= ~(1 << u)
                adj_bitmasks[best_c_new][u] |= (1 << v)
                adj_bitmasks[best_c_new][v] |= (1 << u)
                colors[e_idx] = best_c_new
                energy += best_delta
                
                if energy < best_energy:
                    best_energy = energy
                    best_colors = colors[:]
                    best_adj = [row[:] for row in adj_bitmasks]
                    
                    if energy == 0:
                        return colors, adj_bitmasks, 0
            else:
                # Random restart or jump?
                # Just continue for now.
                pass
                
        return best_colors, best_adj, best_energy

    best_n = 1
    best_colors = [0]
    
    # Try n from 29 down to 26
    for n in range(29, 25, -1):
        if time_left() < 10:
            break
            
        found = False
        # More restarts for larger n? No, larger n is harder.
        # But we want to find a solution quickly.
        num_restarts = 10
        
        for r in range(num_restarts):
            if time_left() < 5:
                break
                
            colors, adj = greedy_coloring(n, seed=42 + r)
            
            if is_valid(adj, n):
                best_n = n
                best_colors = colors
                found = True
                break
                
            # Local search
            time_limit = min(5, time_left() / (num_restarts - r))
            colors, adj, energy = local_search(colors, adj, n, time_limit)
            
            if energy == 0:
                best_n = n
                best_colors = colors
                found = True
                break
                
        if found:
            break
            
    return {'n': best_n, 'colors': best_colors}

if __name__ == '__main__':
    result = construct()
    print(f"Found n = {result['n']}")
