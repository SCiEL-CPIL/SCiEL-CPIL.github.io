import numpy as np
import itertools
import time
import random

def construct():
    start_time = time.time()
    timeout = 110.0
    
    n = 16
    # Precompute edge indices
    edge_to_idx = {}
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            edge_to_idx[(i,j)] = idx
            idx += 1
            
    # Precompute K4s and their edge indices
    k4s = list(itertools.combinations(range(n), 4))
    num_k4s = len(k4s)
    
    k4_edges = []
    for k4 in k4s:
        edges = []
        for i in range(4):
            for j in range(i+1, 4):
                u, v = k4[i], k4[j]
                if u > v: u, v = v, u
                edges.append(edge_to_idx[(u,v)])
        k4_edges.append(edges)
        
    # Precompute which K4s contain which edge
    edge_to_k4s = [[] for _ in range(120)]
    for k_idx, edges in enumerate(k4_edges):
        for e in edges:
            edge_to_k4s[e].append(k_idx)
            
    # Fallback seed solution: (i + j) % 17
    seed_result = [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
    
    best_result = None
    best_colours = 17 # Seed uses 17
    
    # Try to find solutions with K = 13, 14, 15
    for K in [13, 14, 15]:
        if time.time() - start_time > timeout - 5:
            break
            
        found = False
        # Number of restarts depends on time left
        # Estimate time per restart ~ 0.5s? Let's be conservative.
        max_restarts = 100
        for restart in range(max_restarts):
            if time.time() - start_time > timeout - 5:
                break
                
            # Random initialization
            colors = np.random.randint(0, K, 120)
            
            # Initialize state
            # freq[k_idx, c] = count of color c in K4 k_idx
            freq = np.zeros((num_k4s, K), dtype=np.int32)
            distinct = np.zeros(num_k4s, dtype=np.int32)
            violations = 0
            
            for k_idx, edges in enumerate(k4_edges):
                for e in edges:
                    freq[k_idx, colors[e]] += 1
                distinct[k_idx] = np.count_nonzero(freq[k_idx])
                if distinct[k_idx] < 5:
                    violations += 1
            
            if violations == 0:
                best_colours = K
                best_result = colors.copy()
                found = True
                break
            
            # Simulated Annealing
            T = 1.0
            T_min = 0.01
            alpha = 0.99
            max_iter = 20000
            
            for i in range(max_iter):
                if time.time() - start_time > timeout - 5:
                    break
                if T < T_min:
                    break
                    
                e = random.randint(0, 119)
                old_c = colors[e]
                new_c = random.randint(0, K-1)
                if new_c == old_c:
                    continue
                
                delta_v = 0
                # Calculate delta violations
                for k_idx in edge_to_k4s[e]:
                    old_count = freq[k_idx, old_c]
                    new_count = freq[k_idx, new_c]
                    
                    distinct_change = 0
                    if old_count == 1:
                        distinct_change -= 1 
                    if new_count == 0:
                        distinct_change += 1
                    
                    new_distinct = distinct[k_idx] + distinct_change
                    old_viol = 1 if distinct[k_idx] < 5 else 0
                    new_viol = 1 if new_distinct < 5 else 0
                    delta_v += new_viol - old_viol
                
                # Acceptance
                if delta_v <= 0:
                    # Apply change
                    colors[e] = new_c
                    for k_idx in edge_to_k4s[e]:
                        freq[k_idx, old_c] -= 1
                        freq[k_idx, new_c] += 1
                        if freq[k_idx, old_c] == 0:
                            distinct[k_idx] -= 1
                        if freq[k_idx, new_c] == 1:
                            distinct[k_idx] += 1
                    violations += delta_v
                    if violations == 0:
                        best_colours = K
                        best_result = colors.copy()
                        found = True
                        break
                else:
                    if random.random() < np.exp(-delta_v / T):
                        # Apply change
                        colors[e] = new_c
                        for k_idx in edge_to_k4s[e]:
                            freq[k_idx, old_c] -= 1
                            freq[k_idx, new_c] += 1
                            if freq[k_idx, old_c] == 0:
                                distinct[k_idx] -= 1
                            if freq[k_idx, new_c] == 1:
                                distinct[k_idx] += 1
                        violations += delta_v
                
                T *= alpha
                
            if found:
                break
                
        if found:
            break
            
    if best_result is not None:
        return best_result.tolist()
    else:
        return seed_result
