import numpy as np
import itertools
import time
import sys

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit

    # Generate all weight-4 vectors in {-1, 0, 1}^11 (Norm^2 = 4)
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
            
    # Generate all weight-1 vectors in {-2, 0, 2}^11 (Norm^2 = 4)
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)
            
    vectors_v4 = np.array(v4_candidates, dtype=np.int64)
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)
    
    N4 = len(vectors_v4)
    
    if N4 == 0:
        return vectors_v1
    
    # Compute conflict graph: conflict if dot > 2
    dots = vectors_v4 @ vectors_v4.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Spectral method: Max-Cut relaxation
    # Build Laplacian L = D - A where A is adjacency (conflict) matrix
    degree = conflicts.sum(axis=1)
    L = np.diag(degree) - conflicts.astype(np.float64)
    
    # Find Fiedler vector (second smallest eigenvector)
    # Use scipy for eigendecomposition
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import eigsh
    
    L_sparse = csr_matrix(L)
    
    try:
        # Find smallest eigenvalues
        eigvals, eigvecs = eigsh(L_sparse, k=3, sigma=0, which='LM')
        # Sort by eigenvalue
        idx = np.argsort(eigvals)
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        
        # Fiedler vector is the second eigenvector
        fiedler = eigvecs[:, 1]
    except Exception:
        # Fallback: use random vector
        rng = np.random.default_rng(42)
        fiedler = rng.standard_normal(N4)
    
    # Assign: +1 if fiedler[i] >= 0, -1 otherwise
    assignment = np.where(fiedler >= 0, 1, -1)
    
    # Count satisfied edges (non-conflict edges where both endpoints have same sign)
    # Actually for Max-Cut, we want to cut conflict edges
    # For independent set, we want to select vertices with no conflicts between them
    # Let's think differently: we want a subset S such that no two vertices in S conflict
    # This is maximum independent set
    
    # Better approach: use the spectral method to get an initial independent set
    # Then refine with local search
    
    # From the spectral assignment, take one side (the larger or better side)
    pos_set = np.where(assignment == 1)[0]
    neg_set = np.where(assignment == -1)[0]
    
    # For each side, find the maximum independent set within that side
    # Actually, the spectral cut gives us a partition, but we need an independent set
    
    # Let me use a different approach: greedy with spectral initialization
    
    # Use the Fiedler vector to order vertices: process in order of |fiedler[i]|
    # Vertices with large |fiedler[i]| are more "central" to the cut
    
    # Alternative: use the spectral method to get a good initial solution,
    # then use local search to improve
    
    # Let's try: start with all vertices assigned to "selected" or "not selected"
    # based on the Fiedler vector, then refine
    
    # Actually, let me use a cleaner approach:
    # 1. Get initial solution from spectral method (take one side of the cut)
    # 2. Refine with local search
    
    # For maximum independent set, a good heuristic is:
    # - Start with a vertex ordering
    # - Greedily add vertices if they don't conflict with already added
    
    # Use the Fiedler vector to create an ordering: vertices with more negative 
    # Fiedler values first (or some other criterion)
    
    # Let me try multiple strategies and keep the best:
    
    best_selected = None
    best_count = 0
    
    # Strategy 1: Greedy minimum degree (current approach)
    alive = np.ones(N4, dtype=bool)
    selected = np.zeros(N4, dtype=bool)
    degrees = conflicts.sum(axis=1).astype(np.int64)
    rng = np.random.default_rng(42)
    
    while np.any(alive):
        alive_indices = np.where(alive)[0]
        if len(alive_indices) == 0:
            break
        alive_degrees = degrees[alive_indices]
        min_deg = np.min(alive_degrees)
        cands = alive_indices[alive_degrees == min_deg]
        chosen = cands[rng.integers(len(cands))]
        
        selected[chosen] = True
        to_remove = np.where(conflicts[chosen] | (np.arange(N4) == chosen))[0]
        alive[to_remove] = False
        remaining = np.where(alive)[0]
        if len(remaining) > 0:
            degrees[remaining] -= conflicts[remaining][:, to_remove].sum(axis=1)
    
    if selected.sum() > best_count:
        best_count = selected.sum()
        best_selected = selected.copy()
    
    # Strategy 2: Spectral-based ordering
    # Order by Fiedler vector value, process in that order
    order = np.argsort(fiedler)  # most negative first
    
    selected = np.zeros(N4, dtype=bool)
    alive = np.ones(N4, dtype=bool)
    
    for i in order:
        if not alive[i]:
            continue
        selected[i] = True
        to_remove = np.where(conflicts[i] | (np.arange(N4) == i))[0]
        alive[to_remove] = False
    
    if selected.sum() > best_count:
        best_count = selected.sum()
        best_selected = selected.copy()
    
    # Strategy 3: Strong local search with 2-opt swaps and iterative refinement
    if best_selected is not None:
        selected = best_selected.copy()
        rng_ls = np.random.default_rng(77)
        
        # Phase 1: Add all free vertices (no conflicts with selected)
        for _ in range(200):
            if time.time() > deadline - 15:
                break
            changed = False
            non_sel = np.where(~selected)[0]
            for v in non_sel:
                if np.any(conflicts[v] & selected):
                    continue
                selected[v] = True
                changed = True
            if not changed:
                break
        
        if selected.sum() > best_count:
            best_count = selected.sum()
            best_selected = selected.copy()
        
        # Phase 2: 2-opt swap search - try removing one selected, adding one non-selected
        # that was blocked by exactly that one selected vertex
        improved = True
        max_swaps = 3000
        swap_iter = 0
        while improved and swap_iter < max_swaps:
            if time.time() > deadline - 15:
                break
            improved = False
            swap_iter += 1
            
            sel_idx = np.where(selected)[0]
            non_sel_idx = np.where(~selected)[0]
            
            # For each non-selected v with exactly 1 conflict to selected u,
            # try swap: remove u, add v. Then check if we can add more free vertices.
            # We accept if the net gain after cleanup is positive.
            best_gain = 0
            best_swap_pair = None
            
            # Sample a subset for speed if too many
            if len(non_sel_idx) > 2000:
                sample_size = 2000
                sample_idx = rng_ls.choice(len(non_sel_idx), size=sample_size, replace=False)
                non_sel_sample = non_sel_idx[sample_idx]
            else:
                non_sel_sample = non_sel_idx
            
            for v in non_sel_sample:
                if time.time() > deadline - 15:
                    break
                sel_neighbors = np.where(conflicts[v] & selected)[0]
                if len(sel_neighbors) != 1:
                    continue
                u = sel_neighbors[0]
                
                # Tentative swap
                selected[u] = False
                selected[v] = True
                
                # Try to add free vertices
                gain = 0
                added = []
                non_sel_now = np.where(~selected)[0]
                for w in non_sel_now:
                    if np.any(conflicts[w] & selected):
                        continue
                    selected[w] = True
                    added.append(w)
                    gain += 1
                
                net_gain = gain  # -1 (removed u) + 1 (added v) + gain (added w's) = gain
                if net_gain > best_gain:
                    best_gain = net_gain
                    best_swap_pair = (u, v, added.copy())
                
                # Revert
                selected[u] = True
                selected[v] = False
                for w in added:
                    selected[w] = False
            
            if best_swap_pair is not None and best_gain > 0:
                u, v, added = best_swap_pair
                selected[u] = False
                selected[v] = True
                for w in added:
                    selected[w] = True
                improved = True
            
            # Also try: remove a selected vertex that has many conflicts, see if we can add more
            if not improved:
                # Try removing a high-degree selected vertex
                sel_degrees = (conflicts[sel_idx] & selected).sum(axis=1)
                if len(sel_degrees) > 0:
                    # Remove the one with highest degree to selected set
                    max_deg_idx = np.argmax(sel_degrees)
                    u = sel_idx[max_deg_idx]
                    if sel_degrees[max_deg_idx] > 0:
                        selected[u] = False
                        added = []
                        non_sel_now = np.where(~selected)[0]
                        for w in non_sel_now:
                            if np.any(conflicts[w] & selected):
                                continue
                            selected[w] = True
                            added.append(w)
                        if len(added) >= 1:  # net gain >= 0
                            improved = True
                        else:
                            selected[u] = True  # revert
        
        if selected.sum() > best_count:
            best_count = selected.sum()
            best_selected = selected.copy()
    
    # Strategy 4: Multiple random restarts with different orderings
    for restart in range(15):
        if time.time() > deadline - 10:
            break
        
        rng2 = np.random.default_rng(100 + restart)
        alive = np.ones(N4, dtype=bool)
        selected = np.zeros(N4, dtype=bool)
        
        # Random ordering
        order = rng2.permutation(N4)
        
        for i in order:
            if not alive[i]:
                continue
            selected[i] = True
            to_remove = np.where(conflicts[i] | (np.arange(N4) == i))[0]
            alive[to_remove] = False
        
        count = selected.sum()
        if count > best_count:
            best_count = count
            best_selected = selected.copy()
    
    # Strategy 5: Final polish - add all free vertices
    if best_selected is not None:
        selected = best_selected.copy()
        for _ in range(500):
            if time.time() > deadline - 5:
                break
            changed = False
            non_sel = np.where(~selected)[0]
            for v in non_sel:
                if np.any(conflicts[v] & selected):
                    continue
                selected[v] = True
                changed = True
            if not changed:
                break
        if selected.sum() > best_count:
            best_count = selected.sum()
            best_selected = selected.copy()
    
    result_v4 = vectors_v4[best_selected]
    
    # Add all V1 vectors (they're all compatible with V4 and each other)
    final_vectors = np.vstack([result_v4, vectors_v1])
    
    # Verify the constraint: min pairwise distance >= max norm
    # For safety, verify and remove any violating vectors
    if len(final_vectors) > 1:
        norms = np.sqrt(np.sum(final_vectors.astype(np.float64)**2, axis=1))
        max_norm = np.max(norms)
        
        # Compute all pairwise distances
        # This can be expensive, so do it in chunks if needed
        n_final = len(final_vectors)
        valid = np.ones(n_final, dtype=bool)
        
        # Check in batches to save memory
        batch_size = 500
        for i in range(0, n_final, batch_size):
            if time.time() > deadline - 2:
                break
            batch = final_vectors[i:i+batch_size]
            if len(batch) == 0:
                break
            # Distance from batch to all other vectors
            # d(a,b)^2 = |a|^2 + |b|^2 - 2*a.b
            dots_batch = batch @ final_vectors.T
            norms_sq = np.sum(batch.astype(np.float64)**2, axis=1)
            all_norms_sq = np.sum(final_vectors.astype(np.float64)**2, axis=1)
            dist_sq = norms_sq[:, None] + all_norms_sq[None, :] - 2 * dots_batch
            
            # For each vector in batch, check against all others (excluding itself)
            rows = np.arange(len(batch))
            cols = np.arange(n_final)
            # dist_sq[rows, cols] where col != i (index in batch)
            for r in range(len(batch)):
                idx_in_final = i + r
                if not valid[idx_in_final]:
                    continue
                # Check against all valid vectors
                d_sq = dist_sq[r, :]
                # Exclude self
                d_sq[idx_in_final] = np.inf
                # Check against valid vectors only
                for c in range(n_final):
                    if c == idx_in_final:
                        continue
                    if not valid[c]:
                        continue
                    if d_sq[c] < max_norm**2 - 1e-6:
                        valid[idx_in_final] = False
                        valid[c] = False
                        break
        
        final_vectors = final_vectors[valid]
    
    return final_vectors
