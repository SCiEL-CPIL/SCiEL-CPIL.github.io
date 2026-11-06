import random
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 110.0
    
    n = 16
    num_colors = 15
    
    # Precompute edges
    edges = []
    edge_to_idx = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))
            edge_to_idx[(i, j)] = idx
            idx += 1
    num_edges = len(edges) # 120
    
    # Precompute K4s
    k4s = []
    k4_edges = [] # list of lists of edge indices
    k4_idx = 0
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                for d in range(c + 1, n):
                    k4s.append((a, b, c, d))
                    e_indices = [
                        edge_to_idx[(a, b)],
                        edge_to_idx[(a, c)],
                        edge_to_idx[(a, d)],
                        edge_to_idx[(b, c)],
                        edge_to_idx[(b, d)],
                        edge_to_idx[(c, d)]
                    ]
                    k4_edges.append(e_indices)
                    k4_idx += 1
    num_k4s = len(k4s) # 1820
    
    # Precompute edge to k4s mapping
    edge_to_k4s = [[] for _ in range(num_edges)]
    for k_idx in range(num_k4s):
        for e_idx in k4_edges[k_idx]:
            edge_to_k4s[e_idx].append(k_idx)
            
    def solve():
        # Initialize random coloring
        colors = np.random.randint(0, num_colors, num_edges)
        
        # Initialize counts
        counts = np.zeros((num_k4s, num_colors), dtype=np.int8)
        k4_num_colors = np.zeros(num_k4s, dtype=np.int8)
        
        for k_idx in range(num_k4s):
            e_indices = k4_edges[k_idx]
            for e_idx in e_indices:
                c = colors[e_idx]
                if counts[k_idx][c] == 0:
                    k4_num_colors[k_idx] += 1
                counts[k_idx][c] += 1
                
        current_violations = np.sum(k4_num_colors < 5)
        
        if current_violations == 0:
            return list(colors)
            
        T = 1.0
        cooling = 0.9995
        no_improve = 0
        
        while time.time() < deadline - 2.0:
            if current_violations == 0:
                break
                
            # Pick a random violating K4
            violating_k4s = np.where(k4_num_colors < 5)[0]
            if len(violating_k4s) == 0:
                break
            k_idx = violating_k4s[np.random.randint(len(violating_k4s))]
            
            # Pick a random edge in this K4
            e_idx = k4_edges[k_idx][np.random.randint(6)]
            
            c_old = colors[e_idx]
            
            # Find absent colors in this K4
            # present_mask
            present_mask = counts[k_idx] > 0
            absent_colors = np.where(~present_mask)[0]
            
            if len(absent_colors) == 0:
                continue
                
            best_delta = 100
            best_c_new = -1
            
            for c_new in absent_colors:
                delta_v = 0
                for k2_idx in edge_to_k4s[e_idx]:
                    if counts[k2_idx][c_old] == 1:
                        if k4_num_colors[k2_idx] == 5:
                            delta_v += 1
                    if counts[k2_idx][c_new] == 0:
                        if k4_num_colors[k2_idx] == 4:
                            delta_v -= 1
                            
                if delta_v < best_delta:
                    best_delta = delta_v
                    best_c_new = c_new
                    
            # Accept best move
            if best_c_new != -1:
                # Check acceptance
                if best_delta < 0 or (best_delta == 0 and np.random.random() < 0.5) or (best_delta > 0 and np.random.random() < np.exp(-best_delta / T)):
                    # Apply move
                    colors[e_idx] = best_c_new
                    for k2_idx in edge_to_k4s[e_idx]:
                        if counts[k2_idx][c_old] == 1:
                            k4_num_colors[k2_idx] -= 1
                        counts[k2_idx][c_old] -= 1
                        
                        if counts[k2_idx][best_c_new] == 0:
                            k4_num_colors[k2_idx] += 1
                        counts[k2_idx][best_c_new] += 1
                            
                    current_violations += best_delta
                    if best_delta < 0:
                        no_improve = 0
                    else:
                        no_improve += 1
                    T *= cooling
                else:
                    no_improve += 1
            else:
                no_improve += 1
                
            if no_improve > 1000:
                break
                
        if current_violations == 0:
            return list(colors)
        return None

    result = solve()
    if result:
        return result
        
    # Fallback: Return a 17-coloring (cyclic)
    # (i+j) % 17
    return [(i + j) % 17 for i in range(n) for j in range(i + 1, n)]
