import time
import random
import math
import numpy as np

N = 12
M = 40
FULL_MASK = (1 << N) - 1
NUM_MASKS = 1 << N

def _build_comparability_matrix():
    """Precompute comparability: comp[i,j] = 1 if set i is comparable with set j."""
    comp = np.zeros((NUM_MASKS, NUM_MASKS), dtype=np.uint8)
    for i in range(NUM_MASKS):
        for j in range(i + 1, NUM_MASKS):
            if (i & j) == i or (i & j) == j:
                comp[i, j] = 1
                comp[j, i] = 1
    return comp

# Precompute once at module load
_COMP_MATRIX = _build_comparability_matrix()

def _count_pairs(masks):
    """Count comparable pairs in a list of masks using precomputed matrix."""
    count = 0
    m = len(masks)
    for i in range(m):
        for j in range(i + 1, m):
            count += _COMP_MATRIX[masks[i], masks[j]]
    return count

def _incremental_score(masks, idx, new_mask):
    """Compute score change when replacing masks[idx] with new_mask."""
    old_mask = masks[idx]
    delta = 0
    for j in range(M):
        if j == idx:
            continue
        c = masks[j]
        old_comp = _COMP_MATRIX[old_mask, c]
        new_comp = _COMP_MATRIX[new_mask, c]
        delta += new_comp - old_comp
    return delta

def _greedy_construct():
    """Vectorized greedy construction of 40 masks."""
    chosen = []
    chosen_set = set()
    
    for _ in range(M):
        best_mask = -1
        best_score = -1
        for mask in range(NUM_MASKS):
            if mask in chosen_set:
                continue
            # Count how many chosen masks are comparable with this mask
            score = sum(_COMP_MATRIX[mask, c] for c in chosen)
            if score > best_score:
                best_score = score
                best_mask = mask
        chosen.append(best_mask)
        chosen_set.add(best_mask)
    
    return chosen

def _full_scan_local_search(masks, deadline=None):
    """Full-scan local search: for each position, try all 4096 masks to find best replacement."""
    current_score = _count_pairs(masks)
    mask_set = set(masks)
    improved = True
    
    while improved:
        if deadline and time.time() > deadline:
            break
        improved = False
        for idx in range(M):
            if deadline and time.time() > deadline:
                break
            old_mask = masks[idx]
            mask_set.discard(old_mask)
            
            best_mask = old_mask
            best_delta = 0
            
            for new_mask in range(NUM_MASKS):
                if new_mask in mask_set:
                    continue
                delta = _incremental_score(masks, idx, new_mask)
                if delta > best_delta:
                    best_delta = delta
                    best_mask = new_mask
            
            mask_set.add(old_mask)
            
            if best_delta > 0:
                mask_set.discard(old_mask)
                masks[idx] = best_mask
                mask_set.add(best_mask)
                current_score += best_delta
                improved = True
    
    return masks, current_score

def _simulated_annealing(masks, deadline=None):
    """Simulated annealing to escape local optima."""
    current_score = _count_pairs(masks)
    best_score = current_score
    best_masks = list(masks)
    
    T = 3.0
    T_min = 0.05
    cooling = 0.9995
    
    while time.time() < deadline:
        if T < T_min:
            T = 1.5
            cooling = 0.9998
        
        idx = random.randint(0, M - 1)
        old_mask = masks[idx]
        
        new_mask = random.randint(0, FULL_MASK)
        if new_mask == old_mask:
            continue
        if new_mask in masks:
            continue
        
        delta = _incremental_score(masks, idx, new_mask)
        
        if delta >= 0 or random.random() < math.exp(delta / T):
            masks[idx] = new_mask
            current_score += delta
            
            if current_score > best_score:
                best_score = current_score
                best_masks = list(masks)
        
        T *= cooling
    
    return best_masks, best_score

def _swap_local_search(masks, deadline=None):
    """Try all swaps between pairs of positions to find improvements."""
    current_score = _count_pairs(masks)
    improved = True
    
    while improved:
        if deadline and time.time() > deadline:
            break
        improved = False
        for i in range(M):
            if deadline and time.time() > deadline:
                break
            for j in range(i + 1, M):
                old_i = masks[i]
                old_j = masks[j]
                
                # Compute delta for swapping i and j
                delta = 0
                for k in range(M):
                    if k == i or k == j:
                        continue
                    c = masks[k]
                    old_comp_i = _COMP_MATRIX[old_i, c]
                    old_comp_j = _COMP_MATRIX[old_j, c]
                    new_comp_i = _COMP_MATRIX[old_j, c]
                    new_comp_j = _COMP_MATRIX[old_i, c]
                    delta += (new_comp_i + new_comp_j) - (old_comp_i + old_comp_j)
                
                old_ij = _COMP_MATRIX[old_i, old_j]
                new_ij = _COMP_MATRIX[old_j, old_i]
                delta += new_ij - old_ij
                
                if delta > 0:
                    masks[i], masks[j] = old_j, old_i
                    current_score += delta
                    improved = True
    
    return masks, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin
    
    random.seed(42)
    
    best_overall = None
    best_overall_score = -1
    
    # Multiple restarts
    for restart in range(8):
        if time.time() >= deadline:
            break
        
        random.seed(42 + restart * 137)
        
        # Greedy initialization
        masks = _greedy_construct()
        
        # Add perturbation for diversity
        for _ in range(restart * 3):
            idx = random.randint(0, M - 1)
            new_mask = random.randint(0, FULL_MASK)
            if new_mask not in masks:
                masks[idx] = new_mask
        
        # Full scan local search
        remaining = deadline - time.time()
        if remaining > 2:
            masks, score = _full_scan_local_search(masks, deadline - 1)
        else:
            score = _count_pairs(masks)
        
        # Simulated annealing
        remaining = deadline - time.time()
        if remaining > 3:
            masks, score = _simulated_annealing(masks, deadline - 1)
        
        # Swap local search polish
        remaining = deadline - time.time()
        if remaining > 1:
            masks, score = _swap_local_search(masks, deadline - 0.5)
        
        # Final full scan
        remaining = deadline - time.time()
        if remaining > 0.5:
            masks, score = _full_scan_local_search(masks, deadline - 0.1)
        
        if score > best_overall_score:
            best_overall_score = score
            best_overall = list(masks)
        
        if best_overall_score >= 220:
            break
    
    if best_overall is None:
        best_overall = _greedy_construct()
    
    # Convert masks to lists of vertices
    result = []
    for mask in best_overall:
        result.append([b for b in range(N) if (mask >> b) & 1])
    
    return result
