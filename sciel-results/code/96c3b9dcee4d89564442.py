"""Kissing number in dimension 11: spectral relaxation + local search for large independent sets."""
import numpy as np
import itertools
import time
import random
import heapq
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh


def construct():
    start_time = time.time()
    deadline = start_time + 118.0

    D = 11

    # Generate candidate vectors
    candidates_list = []

    # Weight-4 vectors (norm^2=4): C(11,4)*16 = 5280
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates_list.append(v)

    # Weight-3 vectors (norm^2=3): C(11,3)*8 = 1320
    for positions in itertools.combinations(range(D), 3):
        for signs in itertools.product([1, -1], repeat=3):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates_list.append(v)

    # Weight-2 vectors (norm^2=2): C(11,2)*4 = 220
    for positions in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates_list.append(v)

    # Weight-1 vectors (norm^2=1): 22
    for pos in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[pos] = s
            candidates_list.append(v)

    candidates = np.array(candidates_list, dtype=np.int8)
    n = len(candidates)

    # Precompute norms squared
    norms_sq = np.sum(candidates.astype(np.int32) ** 2, axis=1)
    max_norm_sq = int(norms_sq.max())  # 4

    # Precompute dot product matrix
    dot = candidates.astype(np.int32) @ candidates.T.astype(np.int32)

    # Constraint: ||v_i - v_j||^2 >= max_norm_sq
    threshold = (norms_sq[:, np.newaxis] + norms_sq[np.newaxis, :] - max_norm_sq) / 2.0
    compat = (dot <= threshold).astype(np.uint8)
    np.fill_diagonal(compat, 1)

    # conflict[i,j] = 1 if i and j CANNOT both be selected (i != j)
    conflict = (1 - compat).astype(np.uint8)
    np.fill_diagonal(conflict, 0)

    # Precompute neighbors for faster degree updates
    # neighbors[i] is a list of indices j such that conflict[i,j] == 1
    neighbors = [np.where(conflict[i] == 1)[0] for i in range(n)]
    full_deg = np.array([len(neigh) for neigh in neighbors])

    # Spectral approach: compute Laplacian of conflict graph
    conflict_sparse = csr_matrix(conflict.astype(np.float64))
    degree_vec = full_deg.astype(np.float64)
    L = csr_matrix((degree_vec[:, np.newaxis] - conflict_sparse))

    try:
        k_eig = min(10, n - 1)
        eigenvalues, eigenvectors = eigsh(L, k=k_eig, which='SM', return_eigenvectors=True,
                                          v0=np.ones(n))
    except Exception:
        eigenvalues, eigenvectors = [], np.zeros((n, 1))

    best_selected = []

    # Strategy 1: Spectral-based independent set extraction
    if len(eigenvectors) > 0:
        for ev_idx in range(eigenvectors.shape[1]):
            if time.time() > deadline - 30.0:
                break
            ev = eigenvectors[:, ev_idx]
            threshold_val = np.percentile(ev, 25)
            spectral_set = np.where(ev <= threshold_val)[0]
            
            # Use fast greedy on this subset
            selected = greedy_min_degree_fast(spectral_set, deadline - 25.0)
            if len(selected) > len(best_selected):
                best_selected = selected

            # Try above-median
            threshold_val = np.percentile(ev, 75)
            spectral_set = np.where(ev >= threshold_val)[0]
            selected = greedy_min_degree_fast(spectral_set, deadline - 25.0)
            if len(selected) > len(best_selected):
                best_selected = selected

    # Strategy 2: Fast Min-degree greedy on full set
    def greedy_min_degree_fast(initial_indices, time_limit):
        # initial_indices is an array of indices to consider
        if len(initial_indices) == 0:
            return []
        
        remaining = set(initial_indices.tolist())
        if not remaining:
            return []
            
        # Initialize heap with (degree, index)
        # Degree is relative to the remaining set
        degs = {i: 0 for i in remaining}
        for i in remaining:
            for j in neighbors[i]:
                if j in remaining:
                    degs[i] += 1
        
        heap = [(degs[i], i) for i in remaining]
        heapq.heapify(heap)
        
        selected = []
        # To handle stale entries in heap, we keep a set of removed nodes
        removed = set()
        
        while remaining and time.time() < time_limit:
            # Get min degree node
            while heap:
                d, v = heap[0]
                if v in removed:
                    heapq.heappop(heap)
                    continue
                # Check if degree is still correct? 
                # We update degrees of neighbors when we remove a node.
                # So if v is still in remaining, its degree in `degs` is correct 
                # IF we updated it properly. 
                # Let's just pop and verify.
                break
            
            if not heap:
                break
                
            d, v = heapq.heappop(heap)
            
            if v in removed:
                continue
                
            # Select v
            selected.append(v)
            removed.add(v)
            remaining.discard(v)
            
            # Update degrees of neighbors of v
            for u in neighbors[v]:
                if u in remaining:
                    degs[u] -= 1
                    # Push new degree
                    heapq.heappush(heap, (degs[u], u))
                    
        return selected

    if time.time() < deadline - 20.0:
        sel = greedy_min_degree_fast(np.arange(n), deadline - 20.0)
        if len(sel) > len(best_selected):
            best_selected = sel

    # Strategy 3: Randomized greedy restarts
    np.random.seed(42)
    random.seed(42)
    for trial in range(20): # Reduced count, faster greedy
        if time.time() > deadline - 10.0:
            break
        # Random subset to break symmetry / find different local optima
        subset_size = int(n * np.random.uniform(0.5, 1.0))
        subset_indices = np.random.choice(n, size=subset_size, replace=False)
        sel = greedy_min_degree_fast(subset_indices, deadline - 10.0)
        if len(sel) > len(best_selected):
            best_selected = sel

    # Strategy 4: Local search refinement (1-for-2 swaps)
    def local_search(selected_list, time_limit):
        selected = list(selected_list)
        if not selected:
            return selected
        
        sel_set = set(selected)
        # compat_with_all[i] = 1 if i is compatible with all in selected
        compat_with_all = np.ones(n, dtype=np.uint8)
        for s in selected:
            compat_with_all = compat_with_all & compat[s]
        for s in selected:
            compat_with_all[s] = 0
            
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            # 1. Try to add any compatible node
            addable = np.where(compat_with_all > 0)[0]
            if len(addable) > 0:
                # Pick one with min degree (greedy)
                best_add = addable[np.argmin(full_deg[addable])]
                selected.append(best_add)
                sel_set.add(best_add)
                compat_with_all = compat_with_all & compat[best_add]
                compat_with_all[best_add] = 0
                improved = True
                continue
            
            # 2. Try 1-for-2 swap
            # For each s in selected, check if we can remove s and add two others
            # To speed up, we only check a random subset of s if list is large
            trial_size = min(50, len(selected))
            trial_indices = np.random.choice(len(selected), size=trial_size, replace=False) if len(selected) > trial_size else range(len(selected))
            
            for si in trial_indices:
                if time.time() > time_limit:
                    break
                s = selected[si]
                
                # Compute mask of nodes compatible with (selected \ {s})
                # This is expensive if done from scratch. 
                # Optimization: Precompute "compat_except" for each s? 
                # Or just do it if needed. Given n~6800 and selected~100-200, 
                # iterating over selected to AND masks is O(K*N) which is ~1.3M ops. 
                # Doing this 50 times is ~65M ops. Might be slow in Python loop.
                
                # Alternative: 
                # mask_except_s = compat_with_all | (compat[s] & ~compat_with_all) ? No.
                # compat_with_all = AND_{u in selected} compat[u]
                # We want AND_{u in selected, u!=s} compat[u]
                # This is not easily derived from compat_with_all without knowing which u caused the 0.
                
                # So we must recompute.
                mask_except_s = np.ones(n, dtype=np.uint8)
                for s2 in selected:
                    if s2 != s:
                        mask_except_s = mask_except_s & compat[s2]
                mask_except_s[s] = 0
                # Exclude already selected
                mask_except_s[list(sel_set)] = 0
                
                candidates_for_swap = np.where(mask_except_s > 0)[0]
                if len(candidates_for_swap) < 2:
                    continue
                
                # Find pair (c1, c2) in candidates_for_swap that are compatible
                # Sort by degree to prioritize "safe" nodes
                sorted_cands = candidates_for_swap[np.argsort(full_deg[candidates_for_swap])]
                
                found_pair = False
                # Check first few candidates
                limit_check = min(20, len(sorted_cands))
                for i in range(limit_check):
                    c1 = sorted_cands[i]
                    # Check compatibility of c1 with other candidates
                    # compat[c1] is a vector. We can check against sorted_cands
                    # We need c2 such that compat[c1, c2] == 1
                    # And c2 is in sorted_cands[i+1:]
                    
                    # Vectorized check
                    # mask_c2 = compat[c1][sorted_cands[i+1:]]
                    # If any is 1, we found one.
                    sub_cands = sorted_cands[i+1:limit_check] # Only check within window for speed
                    if len(sub_cands) == 0:
                        continue
                    compat_mask = compat[c1, sub_cands]
                    if np.any(compat_mask == 1):
                        idx_rel = np.argmax(compat_mask)
                        c2 = sub_cands[idx_rel]
                        
                        # Perform swap
                        selected.remove(s)
                        sel_set.discard(s)
                        selected.append(c1)
                        sel_set.add(c1)
                        selected.append(c2)
                        sel_set.add(c2)
                        
                        # Update compat_with_all
                        compat_with_all = np.ones(n, dtype=np.uint8)
                        for s3 in selected:
                            compat_with_all = compat_with_all & compat[s3]
                        for s3 in selected:
                            compat_with_all[s3] = 0
                            
                        improved = True
                        found_pair = True
                        break
                if found_pair:
                    break
        
        return selected

    if best_selected and time.time() < deadline - 5.0:
        improved = local_search(best_selected, deadline - 5.0)
        if len(improved) > len(best_selected):
            best_selected = improved

    # Final greedy fill
    if best_selected and time.time() < deadline - 2.0:
        sel_set = set(best_selected)
        compat_with_all = np.ones(n, dtype=np.uint8)
        for s in best_selected:
            compat_with_all = compat_with_all & compat[s]
        for s in best_selected:
            compat_with_all[s] = 0
        addable = np.where(compat_with_all > 0)[0]
        for idx in addable[np.argsort(full_deg[addable])]:
            if time.time() > deadline - 2.0:
                break
            best_selected.append(idx)
            sel_set.add(idx)
            compat_with_all = compat_with_all & compat[idx]
            compat_with_all[idx] = 0

    if not best_selected:
        v = np.zeros(D, dtype=np.int64)
        v[0] = 1
        v[1] = 1
        return v.reshape(1, -1)

    result = candidates[np.array(best_selected, dtype=np.int64)]
    return result.astype(np.int64)
