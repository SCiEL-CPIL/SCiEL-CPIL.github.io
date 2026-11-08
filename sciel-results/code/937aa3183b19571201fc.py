import numpy as np
import itertools
import random
import time

D = 11

def construct():
    """
    Construct nonzero integer vectors in R^11 whose minimum pairwise distance
    is at least the maximum vector norm.
    
    Strategy:
    1. Generate a pool of candidate integer vectors (weight-2 and weight-4)
    2. Use random orthonormal projections to create new candidate vectors
    3. Greedy selection with multiple random restarts
    4. Local improvement by trying to add more vectors
    """
    
    deadline = time.time() + 140  # 10s safety margin before 150s limit
    
    # Generate base candidate pool: weight-2 and weight-4 vectors
    candidates = []
    
    # Weight 2 vectors: norm^2 = 2
    for comb in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = [0] * D
            v[comb[0]] = signs[0]
            v[comb[1]] = signs[1]
            candidates.append(v)
    
    # Weight 4 vectors: norm^2 = 4
    for comb in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, val in zip(comb, signs):
                v[idx] = val
            candidates.append(v)
    
    # Also add some weight-3 vectors (norm^2 = 3)
    for comb in itertools.combinations(range(D), 3):
        for signs in itertools.product([1, -1], repeat=3):
            v = [0] * D
            for idx, val in zip(comb, signs):
                v[idx] = val
            candidates.append(v)
    
    N = len(candidates)
    cand_arr = np.array(candidates, dtype=np.float64)
    
    # Project through random orthonormal bases to create new candidates
    all_candidates = [cand_arr]
    
    # Generate several random orthonormal projections
    for proj_iter in range(20):
        if time.time() > deadline - 10:
            break
        
        # Generate random orthonormal matrix using QR decomposition
        M = np.random.randn(D, D)
        Q, _ = np.linalg.qr(M)
        
        # Project and round
        projected = cand_arr @ Q.T
        rounded = np.round(projected).astype(np.int64)
        
        # Filter: remove zero vectors and keep only those with small norm
        norms_sq = np.sum(rounded ** 2, axis=1)
        nonzero_mask = norms_sq > 0
        small_norm_mask = norms_sq <= 8  # Keep vectors with norm^2 <= 8
        
        valid_mask = nonzero_mask & small_norm_mask
        if np.any(valid_mask):
            all_candidates.append(rounded[valid_mask])
    
    # Combine all candidates and remove duplicates
    all_cands = np.vstack(all_candidates)
    # Remove exact duplicates
    all_cands_unique = np.unique(all_cands, axis=0)
    N_total = len(all_cands_unique)
    
    norms_sq_all = np.sum(all_cands_unique ** 2, axis=1)
    
    # Greedy selection with multiple random restarts
    best_set = []
    best_size = 0
    
    # Pre-sort candidates by norm^2 (prefer smaller norms)
    order_by_norm = np.argsort(norms_sq_all)
    
    for attempt in range(200):
        if time.time() > deadline - 5:
            break
        
        # Mix: some attempts use norm order, others use random
        if attempt % 3 == 0:
            order = list(order_by_norm)
            random.shuffle(order)
        else:
            order = list(range(N_total))
            random.shuffle(order)
        
        selected = []
        selected_indices = []
        
        for idx in order:
            if time.time() > deadline - 3:
                break
            
            v = all_cands_unique[idx]
            norm_sq = norms_sq_all[idx]
            
            if not selected:
                selected.append(idx)
                continue
            
            sel_idx = np.array(selected)
            sel_vectors = all_cands_unique[sel_idx]
            sel_norms_sq = norms_sq_all[sel_idx]
            
            # Check compatibility: ||v - u||^2 >= max(norm_sq, max_sel_norm_sq)
            dots = sel_vectors @ v
            min_rhs = max(norm_sq, np.max(sel_norms_sq))
            lhs = norm_sq + sel_norms_sq - 2 * dots
            
            if np.all(lhs >= min_rhs - 1e-9):
                selected.append(idx)
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = selected
    
    # Local improvement: try to add more vectors to the best set
    if best_set and time.time() < deadline - 10:
        current_set = list(best_set)
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            current_arr = all_cands_unique[current_set]
            current_norms_sq = norms_sq_all[current_set]
            
            for idx in range(N_total):
                if time.time() > deadline - 10:
                    break
                if idx in current_set:
                    continue
                
                v = all_cands_unique[idx]
                norm_sq = norms_sq_all[idx]
                
                dots = current_arr @ v
                min_rhs = max(norm_sq, np.max(current_norms_sq))
                lhs = norm_sq + current_norms_sq - 2 * dots
                
                if np.all(lhs >= min_rhs - 1e-9):
                    current_set.append(idx)
                    current_arr = np.vstack([current_arr, v])
                    current_norms_sq = np.append(current_norms_sq, norm_sq)
                    improved = True
                    if len(current_set) > best_size:
                        best_size = len(current_set)
                        best_set = current_set
    
    if not best_set:
        # Fallback: return a small valid set
        result = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)
        return result
    
    result = all_cands_unique[best_set]
    return result
