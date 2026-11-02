import numpy as np
import time
import itertools
import sys

D = 11

def _generate_weight_vectors(weight):
    """Generate all integer vectors of a given weight (number of non-zero entries)."""
    if weight > D:
        return np.empty((0, D), dtype=np.int64)
    
    vectors = []
    indices = range(D)
    for comb in itertools.combinations(indices, weight):
        for signs in itertools.product([1, -1], repeat=weight):
            v = np.zeros(D, dtype=np.int64)
            for idx, sign in zip(comb, signs):
                v[idx] = sign
            vectors.append(v)
    
    if not vectors:
        return np.empty((0, D), dtype=np.int64)
    return np.array(vectors, dtype=np.int64)

def _greedy_select(candidates, max_dot, deadline):
    """
    Greedily select a subset of candidates such that pairwise dot products <= max_dot.
    Uses vectorized operations for speed.
    """
    n_cand = len(candidates)
    if n_cand == 0:
        return np.empty((0, D), dtype=np.int64)
    
    # Normalize candidates? No, the condition is on raw dot products for fixed weights.
    # If weights are mixed, we need to check dist^2 >= max(norm^2).
    # For homogeneous weight w, norm_sq = w.
    # dist^2 = 2w - 2 dot.
    # Condition: 2w - 2 dot >= w => w >= 2 dot => dot <= w/2.
    # For w=2, dot <= 1.
    # For w=3, dot <= 1.5 => dot <= 1.
    # For w=4, dot <= 2.
    
    selected = []
    # Randomize order for restarts
    # The caller should pass a shuffled candidates array
    
    # To speed up, we can maintain a set of selected vectors as a matrix.
    # However, checking against a growing matrix is O(k * D) per candidate.
    # Total O(N * k * D).
    
    # Optimization: If the selected set becomes large, we can use a bitset or other structures,
    # but for N~5000 and k~300, NumPy is fine.
    
    selected_arr = []
    
    for i in range(n_cand):
        if time.time() > deadline:
            break
            
        v = candidates[i]
        
        if not selected_arr:
            selected_arr.append(v)
            continue
            
        # Check against all selected vectors
        # selected_matrix shape (k, D)
        # dot_products = selected_matrix @ v
        # if max(dot_products) <= max_dot:
        
        # We can batch this if we maintain the matrix.
        # But appending one by one is safer for memory if k is small.
        
        # Convert selected to array for dot product
        # This is inefficient if done inside the loop.
        # Better: maintain selected_matrix.
        
        # Let's refactor to maintain selected_matrix.
        
    # Refactored approach:
    selected_list = []
    selected_matrix = None
    
    for i in range(n_cand):
        if time.time() > deadline:
            break
            
        v = candidates[i]
        
        if selected_matrix is None:
            selected_list.append(v)
            selected_matrix = v.reshape(1, D)
            continue
            
        # Compute dot products with all selected
        dots = selected_matrix @ v
        
        if np.max(dots) <= max_dot:
            selected_list.append(v)
            selected_matrix = np.vstack([selected_matrix, v.reshape(1, D)])
            
    if not selected_list:
        return np.empty((0, D), dtype=np.int64)
        
    return np.array(selected_list, dtype=np.int64)

def _greedy_select_mixed(candidates, norms_sq, deadline):
    """
    Greedy selection for mixed weights.
    Condition: ||u - v||^2 >= max(||u||^2, ||v||^2)
    ||u||^2 + ||v||^2 - 2 u.v >= max(||u||^2, ||v||^2)
    Let A = ||u||^2, B = ||v||^2.
    A + B - 2 u.v >= max(A, B)
    If A >= B: A + B - 2 u.v >= A => B >= 2 u.v => u.v <= B/2
    If B > A: A + B - 2 u.v >= B => A >= 2 u.v => u.v <= A/2
    
    So u.v <= min(A, B) / 2.
    """
    n_cand = len(candidates)
    if n_cand == 0:
        return np.empty((0, D), dtype=np.int64)
        
    selected_list = []
    selected_matrix = None
    selected_norms = []
    
    for i in range(n_cand):
        if time.time() > deadline:
            break
            
        v = candidates[i]
        norm_sq_v = norms_sq[i]
        
        if selected_matrix is None:
            selected_list.append(v)
            selected_matrix = v.reshape(1, D)
            selected_norms.append(norm_sq_v)
            continue
            
        # Compute dot products with all selected
        dots = selected_matrix @ v
        
        # Determine max allowed dot for each selected vector u_j
        # allowed_j = min(norm_sq_u_j, norm_sq_v) / 2.0
        # We need dots[j] <= allowed_j for all j.
        
        min_norms = np.minimum(selected_norms, norm_sq_v)
        max_allowed = min_norms / 2.0
        
        # Check if all dots are <= max_allowed
        if np.all(dots <= max_allowed):
            selected_list.append(v)
            selected_matrix = np.vstack([selected_matrix, v.reshape(1, D)])
            selected_norms.append(norm_sq_v)
            
    if not selected_list:
        return np.empty((0, D), dtype=np.int64)
        
    return np.array(selected_list, dtype=np.int64)

def construct():
    start_time = time.time()
    deadline = start_time + 115.0 # Safety margin
    
    # Generate candidates
    w2 = _generate_weight_vectors(2)
    w3 = _generate_weight_vectors(3)
    w4 = _generate_weight_vectors(4)
    
    best_set = w2 # Fallback: 220 vectors
    best_size = len(w2)
    
    rng = np.random.default_rng(42)
    
    # Try Greedy on W4
    # Max dot for W4 is 2.
    for restart in range(5):
        if time.time() > deadline:
            break
        shuffled_w4 = w4[rng.permutation(len(w4))]
        # We can stop early if we exceed best_size? 
        # The greedy function doesn't support early exit on size, but we can check after.
        # Actually, we can modify _greedy_select to stop if we can't beat best_size?
        # No, the final size is only known at the end.
        # But if we have 500 candidates left and current size + remaining < best_size, we can stop.
        # For now, just run it.
        
        # Optimization: If best_size is 220, and we are looking for W4, 
        # we might want to stop if we have processed half and size is low? 
        # Let's just run a few restarts.
        
        result = _greedy_select(shuffled_w4, 2, deadline)
        if len(result) > best_size:
            best_size = len(result)
            best_set = result
            
    # Try Greedy on W3
    # Max dot for W3 is 1.
    for restart in range(5):
        if time.time() > deadline:
            break
        shuffled_w3 = w3[rng.permutation(len(w3))]
        result = _greedy_select(shuffled_w3, 1, deadline)
        if len(result) > best_size:
            best_size = len(result)
            best_set = result
            
    # Try Mixed W3 + W4
    # This is more complex. We need to interleave them.
    # Let's just concatenate and shuffle.
    mixed = np.vstack([w3, w4])
    norms_sq = np.sum(mixed * mixed, axis=1)
    
    for restart in range(3):
        if time.time() > deadline:
            break
        shuffled_mixed = mixed[rng.permutation(len(mixed))]
        shuffled_norms = norms_sq[rng.permutation(len(mixed))]
        
        # Note: _greedy_select_mixed expects candidates and norms_sq to be aligned.
        # We shuffled both, so they are aligned.
        
        result = _greedy_select_mixed(shuffled_mixed, shuffled_norms, deadline)
        if len(result) > best_size:
            best_size = len(result)
            best_set = result
            
    # Try Mixed W2 + W4? 
    # W2 is 220. If we add W4, we lose some W2.
    # The mixed greedy might find a better combination.
    mixed24 = np.vstack([w2, w4])
    norms_sq24 = np.sum(mixed24 * mixed24, axis=1)
    
    for restart in range(3):
        if time.time() > deadline:
            break
        shuffled_24 = mixed24[rng.permutation(len(mixed24))]
        shuffled_norms24 = norms_sq24[rng.permutation(len(mixed24))]
        
        result = _greedy_select_mixed(shuffled_24, shuffled_norms24, deadline)
        if len(result) > best_size:
            best_size = len(result)
            best_set = result
            
    return best_set

