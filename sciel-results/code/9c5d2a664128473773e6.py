import time
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_n = 0
    best_edges = []
    min_edge_count = float('inf')
    
    for n in [10, 11]:
        if time.time() > deadline:
            break
            
        vertices = list(range(n))
        all_edges = list(combinations(vertices, 5))
        num_edges = len(all_edges)
        num_colorings = 1 << n
        
        all_colorings = np.arange(num_colorings, dtype=np.int32)
        
        # Build masks array (num_edges, num_colorings) of uint8
        masks = np.zeros((num_edges, num_colorings), dtype=np.uint8)
        for j, e in enumerate(all_edges):
            mask_val = 0
            for v in e:
                mask_val |= (1 << v)
            vals = (all_colorings & mask_val).astype(np.int32)
            masks[j] = ((vals == 0) | (vals == mask_val)).astype(np.uint8)
            
        # Greedy
        covered = np.zeros(num_colorings, dtype=bool)
        selected = []
        remaining = np.arange(num_edges)
        
        while not np.all(covered):
            if time.time() > deadline:
                break
            
            if len(remaining) == 0:
                break
                
            uncovered = ~covered
            # new_covs for all remaining edges
            # masks[remaining] is (K, 2^n)
            # uncovered is (2^n,)
            new_covs = np.sum(masks[remaining] * uncovered[np.newaxis, :], axis=1)
            best_local_idx = np.argmax(new_covs)
            best_global_idx = remaining[best_local_idx]
            max_new = new_covs[best_local_idx]
            
            if max_new == 0:
                break
                
            selected.append(best_global_idx)
            covered |= masks[best_global_idx]
            # Remove from remaining
            remaining = np.delete(remaining, best_local_idx)
            
        # Local Search with Counts
        # cover_counts: how many times each coloring is covered
        cover_counts = np.sum(masks[selected], axis=0).astype(np.int32)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try to remove each selected edge
            # Iterate in reverse to make popping easier? Or just loop and break on success
            for i in range(len(selected)):
                idx = selected[i]
                cover_counts -= masks[idx]
                if np.min(cover_counts) > 0:
                    selected.pop(i)
                    improved = True
                    break
                else:
                    cover_counts += masks[idx]
                    
        if len(selected) < min_edge_count:
            min_edge_count = len(selected)
            best_n = n
            best_edges = [all_edges[i] for i in selected]
            
    if best_edges:
        return {'n': best_n, 'edges': [list(e) for e in best_edges]}
    else:
        return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
