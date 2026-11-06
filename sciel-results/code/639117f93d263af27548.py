import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    best_edges = None
    best_n = 0
    best_count = float('inf')
    
    # Focus on n from 9 to 13 where search is tractable
    for n in range(9, 14):
        if time.time() > deadline:
            break
            
        all_edges = list(combinations(range(n), 5))
        num_edges = len(all_edges)
        total_colorings = 1 << n
        all_ones_mask = (1 << total_colorings) - 1
        
        # Precompute M0 and M1 efficiently using pattern generation
        M0 = [0] * n
        M1 = [0] * n
        
        for v in range(n):
            half = 1 << v
            period = 1 << (v + 1)
            # In each period of length 2^(v+1), positions half..period-1 have bit v = 1
            pattern = ((1 << half) - 1) << half
            num_periods = total_colorings // period
            m1 = 0
            for p in range(num_periods):
                m1 |= (pattern << (p * period))
            M1[v] = m1
            M0[v] = all_ones_mask ^ m1
        
        # Compute edge masks
        edge_masks = []
        for e in all_edges:
            m0 = M0[e[0]] & M0[e[1]] & M0[e[2]] & M0[e[3]] & M0[e[4]]
            m1 = M1[e[0]] & M1[e[1]] & M1[e[2]] & M1[e[3]] & M1[e[4]]
            edge_masks.append(m0 | m1)
        
        best_local_count = float('inf')
        best_local_edges = None
        
        # More restarts for smaller n
        restarts = 300 if n <= 10 else 150 if n <= 11 else 80
        
        for r in range(restarts):
            if time.time() > deadline:
                break
                
            covered = 0
            selected_indices = []
            remaining = list(range(num_edges))
            random.shuffle(remaining)
            
            # Greedy Set Cover
            while covered != all_ones_mask:
                best_gain = -1
                best_idx = -1
                
                for idx in remaining:
                    gain = (edge_masks[idx] & ~covered).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_idx = idx
                        if best_gain == 0:
                            break
                            
                if best_gain <= 0:
                    break
                    
                selected_indices.append(best_idx)
                covered |= edge_masks[best_idx]
                remaining.remove(best_idx)
                
            if covered == all_ones_mask:
                # Refinement: remove redundant edges
                improved = True
                while improved and time.time() < deadline:
                    improved = False
                    random.shuffle(selected_indices)
                    for i in range(len(selected_indices) - 1, -1, -1):
                        e_idx = selected_indices[i]
                        new_covered = 0
                        for j in range(len(selected_indices)):
                            if j != i:
                                new_covered |= edge_masks[selected_indices[j]]
                        if new_covered == all_ones_mask:
                            selected_indices.pop(i)
                            improved = True
                            break
                
                # Local search: try swapping edges to reduce count
                if time.time() < deadline:
                    for _ in range(30):
                        if time.time() > deadline:
                            break
                        random.shuffle(selected_indices)
                        improved_swap = False
                        for i in range(len(selected_indices)):
                            if time.time() > deadline:
                                break
                            e_idx = selected_indices[i]
                            without_e = 0
                            for j in range(len(selected_indices)):
                                if j != i:
                                    without_e |= edge_masks[selected_indices[j]]
                            uncovered = all_ones_mask & ~without_e
                            if uncovered == 0:
                                selected_indices.pop(i)
                                improved_swap = True
                                break
                            # Try to find a replacement
                            for j in range(num_edges):
                                if j in selected_indices:
                                    continue
                                if (edge_masks[j] & uncovered) != 0:
                                    new_covered = without_e | edge_masks[j]
                                    if new_covered == all_ones_mask:
                                        selected_indices[i] = j
                                        improved_swap = True
                                        break
                            if improved_swap:
                                break
                        if not improved_swap:
                            break
                    
                if len(selected_indices) < best_local_count:
                    best_local_count = len(selected_indices)
                    best_local_edges = [all_edges[i] for i in selected_indices]
                    
        if best_local_edges is not None:
            if best_local_count < best_count:
                best_count = best_local_count
                best_n = n
                best_edges = [list(e) for e in best_local_edges]
                print(f"Found better: n={n}, edges={best_count}")
                
    if best_edges is None:
        best_n = 9
        best_edges = [list(e) for e in combinations(range(9), 5)]
        best_count = len(best_edges)

    return {'n': best_n, 'edges': best_edges}
