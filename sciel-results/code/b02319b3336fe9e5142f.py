# EVOLVE-BLOCK-START
import numpy as np
import itertools
from functools import lru_cache

def no_5_on_a_sphere8() -> np.ndarray:
    n = 8
    total = n ** 3
    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    sq_norms = np.sum(grid**2, axis=1)
    
    def can_add(selected_list, idx):
        """Check if idx can be added to selected_list."""
        if len(selected_list) < 4:
            return True
        p = grid[idx]
        p_norm = sq_norms[idx]
        sel_indices = np.array(selected_list)
        Q = grid[sel_indices]
        diff = Q - p
        norm_diff = sq_norms[sel_indices] - p_norm
        W = np.column_stack((diff, norm_diff))
        
        K = len(selected_list)
        combos = list(itertools.combinations(range(K), 4))
        if not combos:
            return True
            
        M = np.empty((len(combos), 4, 4), dtype=np.float64)
        for i, c in enumerate(combos):
            M[i] = W[list(c)]
        dets = np.linalg.det(M)
        return not np.any(np.abs(dets) < 0.5)

    np.random.seed(42)
    best = None
    best_len = 0
    
    for restart in range(50):
        order = list(range(total))
        np.random.shuffle(order)
        
        selected = []
        for idx in order:
            if can_add(selected, idx):
                selected.append(idx)
        
        # Local Search
        # Try to improve by swapping
        for _ in range(20): # Iterations
            improved = False
            # Try to remove one and add two
            for i in range(len(selected)):
                # Remove i
                temp_selected = selected[:i] + selected[i+1:]
                # Find candidates to add
                # We need to find 2 points from unselected that fit with temp_selected
                # This is expensive. 
                # Simplified: Try to add any point j. If it fits, try to add any point k.
                # If both fit, we gain 1.
                
                # To make it fast, only try a few candidates
                unselected = [x for x in range(total) if x not in temp_selected]
                np.random.shuffle(unselected)
                
                found_pair = False
                for j in unselected[:10]: # Limit candidates
                    if not can_add(temp_selected, j):
                        continue
                    # Try to add k
                    temp_selected2 = temp_selected + [j]
                    for k in unselected[:10]:
                        if k == j: continue
                        if can_add(temp_selected2, k):
                            # Success
                            selected = temp_selected2
                            improved = True
                            found_pair = True
                            break
                    if found_pair:
                        break
                if improved:
                    break
            
            if not improved:
                break
                
        if len(selected) > best_len:
            best_len = len(selected)
            best = selected
    
    result = np.array([grid[i] for i in best], dtype=np.int64)
    return result

# EVOLVE-BLOCK-END
