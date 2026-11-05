import time
import random
import numpy as np
from itertools import combinations

# Caches for masks to avoid recomputation
_MASKS_6_CACHE = {}
_ALL_MASKS_CACHE = {}

def _get_masks_6(n):
    """Precompute masks for subsets of size 6."""
    if n not in _MASKS_6_CACHE:
        masks = []
        for combo in combinations(range(n), 6):
            mask = 0
            for v in combo:
                mask |= (1 << v)
            masks.append(mask)
        _MASKS_6_CACHE[n] = masks
    return _MASKS_6_CACHE[n]

def _get_all_masks(n):
    """Precompute masks for all subsets of size >= 6."""
    if n not in _ALL_MASKS_CACHE:
        _ALL_MASKS_CACHE[n] = {}
        for size in range(6, n + 1):
            masks = []
            for combo in combinations(range(n), size):
                mask = 0
                for v in combo:
                    mask |= (1 << v)
                masks.append(mask)
            _ALL_MASKS_CACHE[n][size] = masks
    return _ALL_MASKS_CACHE[n]

def _precompute_mask_arrays(n):
    """Precompute NumPy arrays for all masks of size >= 6.
    Returns a list of (size, mask_indices_array) where mask_indices_array
    has shape (num_masks, size) with vertex indices.
    """
    all_masks = _get_all_masks(n)
    result = []
    for size in range(6, n + 1):
        if size not in all_masks:
            continue
        masks = all_masks[size]
        if not masks:
            continue
        # Convert each mask to array of vertex indices
        indices = np.zeros((len(masks), size), dtype=np.int32)
        for i, mask in enumerate(masks):
            m = mask
            col = 0
            while m:
                lb = m & (-m)
                idx = lb.bit_length() - 1
                indices[i, col] = idx
                col += 1
                m ^= lb
        result.append((size, indices))
    return result

def _count_violations_vectorized(adj_matrix, mask_arrays):
    """
    Count the number of regular induced subgraphs using NumPy.
    adj_matrix: (n, n) boolean numpy array
    mask_arrays: list of (size, indices_array)
    Returns total count of violations.
    """
    total = 0
    for size, indices in mask_arrays:
        if indices.size == 0:
            continue
        num_masks = indices.shape[0]
        # Get adjacency submatrix for all masks at once
        # adj_sub: (num_masks, size, size)
        adj_sub = adj_matrix[indices[:, :, None], indices[:, None, :]]
        # Degrees: sum along axis 2 for each mask
        degs = adj_sub.sum(axis=2)  # (num_masks, size)
        # Check if all degrees in each mask are equal
        # A mask is regular if min(degs) == max(degs) for that mask
        min_degs = degs.min(axis=1)
        max_degs = degs.max(axis=1)
        regular = (min_degs == max_degs)
        total += regular.sum()
    return total

def _find_violation_vectorized(adj_matrix, mask_arrays):
    """
    Find the first regular induced subgraph.
    Returns (size, indices_array_row) or None.
    """
    for size, indices in mask_arrays:
        if indices.size == 0:
            continue
        num_masks = indices.shape[0]
        adj_sub = adj_matrix[indices[:, :, None], indices[:, None, :]]
        degs = adj_sub.sum(axis=2)
        min_degs = degs.min(axis=1)
        max_degs = degs.max(axis=1)
        regular = (min_degs == max_degs)
        if regular.any():
            idx = np.argmax(regular)
            return (size, indices[idx])
    return None

def _edges_from_adj_matrix(adj_matrix):
    """Convert (n,n) boolean numpy adjacency matrix to edge list."""
    n = adj_matrix.shape[0]
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj_matrix[i, j]:
                edges.append([i, j])
    return edges

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    best_n = 5
    best_edges = []
    
    # Try to find valid graphs for n = 22, 21, 20, 19
    for target_n in [22, 21, 20, 19]:
        if time.time() > deadline - 10:
            break
            
        n = target_n
        
        # Precompute mask arrays for vectorized operations
        mask_arrays = _precompute_mask_arrays(n)
        
        found_valid = False
        restarts_attempted = 0
        max_restarts = 50  # More restarts since each check is faster
        
        while restarts_attempted < max_restarts and not found_valid:
            if time.time() > deadline - 10:
                break
            restarts_attempted += 1
            
            # Initialize random graph as (n, n) boolean numpy array
            adj_matrix = np.random.random((n, n)) < 0.5
            # Make symmetric and zero diagonal
            adj_matrix = ((adj_matrix + adj_matrix.T) > 0)
            np.fill_diagonal(adj_matrix, False)
            
            # Simulated annealing
            current_violations = _count_violations_vectorized(adj_matrix, mask_arrays)
            best_violations = current_violations
            best_adj = adj_matrix.copy()
            
            if current_violations == 0:
                edges = _edges_from_adj_matrix(adj_matrix)
                best_n = n
                best_edges = edges
                found_valid = True
                break
            
            # Annealing parameters
            T = 1.0
            cooling = 0.995
            min_T = 0.01
            steps = 0
            max_steps = 3000  # More steps possible due to vectorized checks
            
            while steps < max_steps and time.time() < deadline - 10:
                steps += 1
                
                if T < min_T:
                    break
                
                # Pick a random edge to flip
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                if i == j:
                    continue
                
                # Flip edge (i, j)
                old_val = adj_matrix[i, j]
                adj_matrix[i, j] = not old_val
                adj_matrix[j, i] = not old_val
                
                new_violations = _count_violations_vectorized(adj_matrix, mask_arrays)
                delta = new_violations - current_violations
                
                if delta <= 0 or random.random() < np.exp(-delta / T):
                    # Accept
                    current_violations = new_violations
                    if new_violations < best_violations:
                        best_violations = new_violations
                        best_adj = adj_matrix.copy()
                else:
                    # Reject: undo the flip
                    adj_matrix[i, j] = old_val
                    adj_matrix[j, i] = old_val
                
                T *= cooling
                
                if current_violations == 0:
                    edges = _edges_from_adj_matrix(adj_matrix)
                    best_n = n
                    best_edges = edges
                    found_valid = True
                    break
            
            # If we didn't find valid, check the best we found
            if not found_valid and best_violations == 0:
                edges = _edges_from_adj_matrix(best_adj)
                best_n = n
                best_edges = edges
                found_valid = True
                break
            
            # If best found is valid, use it
            if not found_valid:
                # Check if best_adj is valid (it might have been valid at some point)
                bv = _count_violations_vectorized(best_adj, mask_arrays)
                if bv == 0:
                    edges = _edges_from_adj_matrix(best_adj)
                    best_n = n
                    best_edges = edges
                    found_valid = True
                    break
        
        if found_valid:
            break
    
    # Fallback
    if best_n <= 5:
        return {'n': 5, 'edges': []}
        
    return {'n': best_n, 'edges': best_edges}
