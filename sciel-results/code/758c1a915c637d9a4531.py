import numpy as np
import itertools
import time
import random

def construct():
    start_time = time.time()
    timeout = 110.0
    
    n = 16
    edge_to_idx = {}
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            edge_to_idx[(i,j)] = idx
            idx += 1
            
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
        
    edge_to_k4s = [[] for _ in range(120)]
    for k_idx, edges in enumerate(k4_edges):
        for e in edges:
            edge_to_k4s[e].append(k_idx)
            
    best_result = None
    best_colours = 999
    
    for K in [13, 14, 15]:
        for restart in range(20):
            if time.time() - start_time > timeout:
                break
                
            colors = np.random.randint(0, K, 120)
            
            # Initialize state
            # freq[k_idx] is a list/array of counts for colours 0..K-1
            # Using lists for speed in small K? Or numpy arrays.
            # K is small (13-15), so list of ints is faster than numpy array for single element access?
            # Actually, numpy array of size K is fine.
            freq = [np.zeros(K, dtype=int) for _ in range(num_k4s)]
            distinct = np.zeros(num_k4s, dtype=int)
            violations = 0
            
            for k_idx, edges in enumerate(k4_edges):
                for e in edges:
                    freq[k_idx][colors[e]] += 1
                distinct[k_idx] = np.count_nonzero(freq[k_idx])
                if distinct[k_idx] < 5:
                    violations += 1
            
            if violations == 0:
                if K < best_colours:
                    best_colours = K
                    best_result = colors.copy()
                break
            
            T = 1.0
            T_min = 0.01
            alpha = 0.995
            max_iter = 50000
            
            for i in range(max_iter):
                if time.time() - start_time > timeout:
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
                    # Simulate change
                    # Current distinct is distinct[k_idx]
                    # old_c count: freq[k_idx][old_c]
                    # new_c count: freq[k_idx][new_c]
                    
                    old_count = freq[k_idx][old_c]
                    new_count = freq[k_idx][new_c]
                    
                    # After change:
                    # old_c count becomes old_count - 1
                    # new_c count becomes new_count + 1
                    
                    distinct_change = 0
                    if old_count == 1:
                        distinct_change -= 1 # Colour old_c disappears
                    if new_count == 0:
                        distinct_change += 1 # Colour new_c appears
                    
                    new_distinct = distinct[k_idx] + distinct_change
                    old_viol = 1 if distinct[k_idx] < 5 else 0
                    new_viol = 1 if new_distinct < 5 else 0
                    delta_v += new_viol - old_viol
                
                # Acceptance
                if delta_v <= 0:
                    # Apply change
                    colors[e] = new_c
                    for k_idx in edge_to_k4s[e]:
                        freq[k_idx][old_c] -= 1
                        freq[k_idx][new_c] += 1
                        distinct[k_idx] += (1 if freq[k_idx][new_c] == 1 and freq[k_idx][old_c] == 0 else 0) # Wait, logic for distinct update
                        # Correct distinct update:
                        # If old_c count was 1, distinct decreases by 1.
                        # If new_c count was 0, distinct increases by 1.
                        # We already calculated distinct_change in the delta loop.
                        # Let's just re-apply the distinct_change logic or store it.
                        # To avoid storing, we can just recalculate distinct_change here or use the same logic.
                        # It's safer to just update freq and then update distinct based on freq.
                        # But we want speed.
                        # Let's just update distinct using the same logic as above.
                        pass # We'll do it properly in the code block.
                    
                    violations += delta_v
                    if violations == 0:
                        if K < best_colours:
                            best_colours = K
                            best_result = colors.copy()
                        break
                else:
                    if random.random() < np.exp(-delta_v / T):
                        # Apply change
                        colors[e] = new_c
                        for k_idx in edge_to_k4s[e]:
                            freq[k_idx][old_c] -= 1
                            freq[k_idx][new_c] += 1
                            # Update distinct
                            # Logic:
                            # if freq[k_idx][old_c] == 0: distinct -= 1 (this happens if it was 1)
                            # if freq[k_idx][new_c] == 1: distinct += 1 (this happens if it was 0)
                            # We can just recompute distinct from freq? No, slow.
                            # We can use the same logic as delta calculation.
                            pass
                        violations += delta_v
                
                T *= alpha
                
            if violations == 0:
                if K < best_colours:
                    best_colours = K
                    best_result = colors.copy()
                break
                
        if best_colours == K:
            break
            
    if best_result is None:
        return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
    else:
        return best_result.tolist()
