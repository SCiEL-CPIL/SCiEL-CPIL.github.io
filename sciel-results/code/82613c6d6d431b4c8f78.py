import time
import random
from itertools import combinations

START_TIME = time.time()
TIME_LIMIT = 140.0

def time_left():
    return TIME_LIMIT - (time.time() - START_TIME)

def construct():
    n = 11
    num_colorings = 1 << n  # 2048
    
    all_edges = list(combinations(range(n), 5))
    
    # Fast mask computation: for each edge, enumerate the 128 monochromatic colorings
    edge_masks = []
    for e in all_edges:
        v1, v2, v3, v4, v5 = e
        edge_verts = frozenset((v1, v2, v3, v4, v5))
        free_vertices = [v for v in range(n) if v not in edge_verts]
        base_ones = 0
        for v in (v1, v2, v3, v4, v5):
            base_ones |= (1 << v)
        
        mask = 0
        for free_mask in range(1 << len(free_vertices)):
            # All-1 case
            c1 = base_ones
            for i, fv in enumerate(free_vertices):
                if (free_mask >> i) & 1:
                    c1 |= (1 << fv)
            mask |= (1 << c1)
            # All-0 case
            c0 = 0
            for i, fv in enumerate(free_vertices):
                if (free_mask >> i) & 1:
                    c0 |= (1 << fv)
            mask |= (1 << c0)
        edge_masks.append(mask)
    
    full_mask = (1 << num_colorings) - 1
    num_edges = len(all_edges)
    
    best_indices = None
    best_len = float('inf')
    
    for trial in range(1000):
        if time_left() < 20:
            break
        
        uncovered = full_mask
        selected_indices = []
        
        while uncovered:
            best_gain = -1
            best_idx = -1
            # Collect all edges with max gain for random tie-breaking
            max_gain_edges = []
            
            for idx in range(num_edges):
                em = edge_masks[idx]
                gain = (em & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx
                    max_gain_edges = [idx]
                elif gain == best_gain and gain > 0:
                    max_gain_edges.append(idx)
            
            if best_idx == -1 or best_gain <= 0:
                break
            
            # Random tie-breaking
            if len(max_gain_edges) > 1 and trial > 0:
                best_idx = random.choice(max_gain_edges)
            
            selected_indices.append(best_idx)
            uncovered &= ~edge_masks[best_idx]
            
            if len(selected_indices) >= best_len:
                break
        
        if not uncovered and len(selected_indices) < best_len:
            best_len = len(selected_indices)
            best_indices = selected_indices[:]
            
            if best_len <= 51:
                # Keep searching briefly for potential improvement
                if trial < 50:
                    continue
    
    if best_indices is None:
        return {'n': n, 'edges': [list(e) for e in all_edges[:60]]}
    
    # Local search: remove redundant edges
    current_indices = best_indices[:]
    improved = True
    while improved and time_left() > 20:
        improved = False
        
        total_coverage = 0
        for idx in current_indices:
            total_coverage |= edge_masks[idx]
        
        for i in range(len(current_indices)):
            if time_left() < 20:
                break
            
            cov_without = 0
            for j in range(len(current_indices)):
                if i != j:
                    cov_without |= edge_masks[current_indices[j]]
            
            if cov_without == total_coverage:
                current_indices.pop(i)
                improved = True
                break
    
    final_edges = [list(all_edges[idx]) for idx in current_indices]
    return {'n': n, 'edges': final_edges}

if __name__ == "__main__":
    res = construct()
    print(f"Found hypergraph with n={res['n']}, m={len(res['edges'])}")
