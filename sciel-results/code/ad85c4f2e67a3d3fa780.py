import time
import random
from itertools import combinations

START_TIME = time.time()
TIME_LIMIT = 110.0

def time_left():
    return TIME_LIMIT - (time.time() - START_TIME)

def construct():
    n = 11
    num_colorings = 1 << n
    all_edges = list(combinations(range(n), 5))
    num_edges = len(all_edges)
    
    # Precompute edge masks: which colorings make this edge monochromatic
    edge_masks = []
    for e in all_edges:
        v1, v2, v3, v4, v5 = e
        mask = 0
        for c in range(num_colorings):
            b1 = (c >> v1) & 1
            b2 = (c >> v2) & 1
            b3 = (c >> v3) & 1
            b4 = (c >> v4) & 1
            b5 = (c >> v5) & 1
            if b1 == b2 == b3 == b4 == b5:
                mask |= (1 << c)
        edge_masks.append(mask)
        
    full_mask = (1 << num_colorings) - 1
    best_indices = None
    best_len = float('inf')
    
    def local_search(selected_indices):
        """Try to remove redundant edges and do edge swaps."""
        if not selected_indices:
            return selected_indices
        
        sel_set = set(selected_indices)
        current = list(selected_indices)
        
        # Phase 1: Remove redundant edges
        changed = True
        while changed and time_left() > 3:
            changed = False
            # Compute total coverage
            total_cov = 0
            for idx in current:
                total_cov |= edge_masks[idx]
            
            for i in range(len(current)):
                if time_left() < 3:
                    break
                # Check if removing current[i] still covers everything
                cov_without = 0
                for j in range(len(current)):
                    if j != i:
                        cov_without |= edge_masks[current[j]]
                if cov_without == total_cov:
                    current.pop(i)
                    changed = True
                    break
        
        # Phase 2: Try edge swaps - replace an edge with another that allows further reduction
        improved = True
        while improved and time_left() > 3:
            improved = False
            current_set = set(current)
            for i in range(len(current)):
                if time_left() < 3:
                    break
                for j in range(num_edges):
                    if j in current_set:
                        continue
                    # Try replacing current[i] with j
                    new_set = list(current)
                    new_set[i] = j
                    # Check if this still covers all
                    cov = 0
                    for idx in new_set:
                        cov |= edge_masks[idx]
                    if cov == full_mask:
                        # Check if we can now remove some other edge
                        for k in range(len(new_set)):
                            cov_without_k = 0
                            for l in range(len(new_set)):
                                if l != k:
                                    cov_without_k |= edge_masks[new_set[l]]
                            if cov_without_k == full_mask:
                                new_set.pop(k)
                                current = new_set
                                improved = True
                                break
                        if improved:
                            break
                if improved:
                    break
        
        return current
    
    # Multiple greedy trials with different randomizations
    for trial in range(200):
        if time_left() < 8:
            break
        
        uncovered = full_mask
        selected_indices = []
        
        # Randomize tie-breaking
        tie_prob = 0.3 + 0.7 * (trial / 200.0)
        
        while uncovered and time_left() > 5:
            best_gain = -1
            best_idx = -1
            candidates = []
            
            for idx in range(num_edges):
                em = edge_masks[idx]
                gain = (em & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx
                    candidates = [idx]
                elif gain == best_gain and gain > 0:
                    candidates.append(idx)
            
            if best_idx == -1:
                break
            
            # Random tie-breaking
            if len(candidates) > 1 and random.random() < tie_prob:
                best_idx = random.choice(candidates)
            
            selected_indices.append(best_idx)
            uncovered &= ~edge_masks[best_idx]
            
            if len(selected_indices) >= best_len:
                break
        
        if not uncovered:
            # Apply local search
            reduced = local_search(selected_indices)
            
            if len(reduced) < best_len:
                best_len = len(reduced)
                best_indices = reduced[:]
                if best_len <= 51:
                    # Try more local search
                    for _ in range(5):
                        if time_left() < 5:
                            break
                        new_reduced = local_search(best_indices[:])
                        if len(new_reduced) < best_len:
                            best_len = len(new_reduced)
                            best_indices = new_reduced
                    if best_len <= 51:
                        break
    
    if best_indices is None:
        # Fallback: deterministic greedy
        uncovered = full_mask
        selected_indices = []
        while uncovered:
            best_gain = -1
            best_idx = -1
            for idx in range(num_edges):
                em = edge_masks[idx]
                gain = (em & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx
            if best_idx == -1:
                break
            selected_indices.append(best_idx)
            uncovered &= ~edge_masks[best_idx]
        best_indices = selected_indices
    
    return {'n': n, 'edges': [list(all_edges[idx]) for idx in best_indices]}

if __name__ == "__main__":
    res = construct()
    print(f"Found hypergraph with n={res['n']}, m={len(res['edges'])}")
