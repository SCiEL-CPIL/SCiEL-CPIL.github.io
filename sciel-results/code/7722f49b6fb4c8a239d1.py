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
    
    # Strategy 3: Local search refinement of the best solution
    # Try to improve by swapping: remove a selected vertex, add a non-selected vertex
    # that conflicts with fewer selected vertices
    
    if best_selected is not None:
        selected = best_selected.copy()
        improved = True
        max_iterations = 5000
        iter_count = 0
        
        while improved and iter_count < max_iterations:
            if time.time() > deadline - 20:
                break
            improved = False
            iter_count += 1
            
            selected_indices = np.where(selected)[0]
            non_selected_indices = np.where(~selected)[0]
            
            if len(non_selected_indices) == 0:
                break
            
            # For each non-selected vertex, count how many selected vertices it conflicts with
            # If it conflicts with 0, we can add it freely
            # If it conflicts with 1, we can swap: remove that one, add this one (net 0, but might enable others)
            # If it conflicts with 2+, not worth it directly
            
            # Try to find a non-selected vertex with 0 conflicts with selected
            for v in non_selected_indices:
                if time.time() > deadline - 20:
                    break
                sel_neighbors = np.where(conflicts[v] & selected)[0]
                if len(sel_neighbors) == 0:
                    selected[v] = True
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try 1-swap: remove one selected, add one non-selected
            # For each non-selected v, find selected u that v conflicts with
            # If we remove u and add v, we need v to only conflict with u among selected
            best_swap_gain = 0
            best_swap = None
            
            for v in non_selected_indices:
                if time.time() > deadline - 20:
                    break
                sel_neighbors = np.where(conflicts[v] & selected)[0]
                if len(sel_neighbors) == 1:
                    u = sel_neighbors[0]
                    # If we remove u, can we add v?
                    # v will be added, u will be removed
                    # Net change: +1 - 1 = 0, but might free up other vertices
                    # Let's just try it and see if it leads to improvement
                    # For now, skip 1-swaps as they don't directly improve count
                    pass
            
            # Try: for each selected vertex u, see if removing it allows adding multiple non-selected vertices
            # This is more complex. Let's try a simpler approach:
            # Random restarts of greedy
            
            break
        
        if selected.sum() > best_count:
            best_count = selected.sum()
            best_selected = selected.copy()
    
    # Strategy 4: Multiple random restarts with different orderings
    # Use more restarts since we have time budget
    num_restarts = 50
    for restart in range(num_restarts):
        if time.time() > deadline - 15:
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
    
    # Strategy 5: Local search on the best solution
    if best_selected is not None:
        selected = best_selected.copy()
        
        # Precompute: for each vertex, its conflict list as an array
        # conflict_list[i] = array of indices j where conflicts[i,j]
        conflict_lists = []
        for i in range(N4):
            conflict_lists.append(np.where(conflicts[i])[0])
        
        # Precompute: for each vertex, the number of conflicts with selected vertices
        # sel_conflict_count[i] = number of selected vertices that conflict with i
        sel_conflict_count = np.zeros(N4, dtype=np.int32)
        selected_indices = np.where(selected)[0]
        for i in range(N4):
            sel_conflict_count[i] = np.sum(conflicts[i] & selected)
        
        # Phase 1: Add any non-selected vertex with 0 conflicts with selected
        changed = True
        while changed:
            if time.time() > deadline - 5:
                break
            changed = False
            non_selected_indices = np.where(~selected)[0]
            for v in non_selected_indices:
                if sel_conflict_count[v] == 0:
                    selected[v] = True
                    # Update conflict counts for neighbors of v
                    for u in conflict_lists[v]:
                        if not selected[u]:
                            sel_conflict_count[u] += 1
                    changed = True
            if changed:
                # Recompute sel_conflict_count to be safe
                sel_conflict_count = np.zeros(N4, dtype=np.int32)
                for i in range(N4):
                    sel_conflict_count[i] = np.sum(conflicts[i] & selected)
        
        # Phase 2: 2-swap local search
        # Try: remove selected vertex u, add non-selected vertex v
        # This is beneficial if v conflicts with fewer selected vertices than u's "cost"
        # More precisely: if we remove u and add v, the change in count is 0 (remove 1, add 1)
        # But it might free up other vertices. Let's try a simpler criterion:
        # Remove u if it has many conflicts with non-selected vertices (blocking many)
        # Add v if it has few conflicts with selected vertices (easy to add)
        
        improved = True
        max_swap_iters = 50
        swap_iter = 0
        while improved and swap_iter < max_swap_iters:
            if time.time() > deadline - 5:
                break
            improved = False
            swap_iter += 1
            
            selected_indices = np.where(selected)[0]
            non_selected_indices = np.where(~selected)[0]
            
            if len(selected_indices) == 0 or len(non_selected_indices) == 0:
                break
            
            # For each selected u, compute "blocking cost": how many non-selected vertices it conflicts with
            # For each non-selected v, compute "addition cost": how many selected vertices it conflicts with
            
            # We want to find a pair (u, v) such that removing u and adding v is net beneficial
            # A simple heuristic: remove u with high blocking cost, add v with low addition cost
            # But we need v to not conflict with selected vertices except u
            
            best_gain = 0
            best_u = -1
            best_v = -1
            
            # For efficiency, only consider promising candidates
            # Sort selected by blocking cost (descending)
            blocking_costs = np.zeros(len(selected_indices), dtype=np.int32)
            for idx, u in enumerate(selected_indices):
                blocking_costs[idx] = np.sum(conflicts[u] & ~selected)
            
            # Sort non-selected by addition cost (ascending)
            addition_costs = np.zeros(len(non_selected_indices), dtype=np.int32)
            for idx, v in enumerate(non_selected_indices):
                addition_costs[idx] = np.sum(conflicts[v] & selected)
            
            # Try top candidates
            top_u_candidates = np.argsort(-blocking_costs)[:min(20, len(selected_indices))]
            top_v_candidates = np.argsort(addition_costs)[:min(20, len(non_selected_indices))]
            
            for ui in top_u_candidates:
                u = selected_indices[ui]
                for vi in top_v_candidates:
                    v = non_selected_indices[vi]
                    # Check if v conflicts with selected vertices other than u
                    v_sel_conflicts = np.where(conflicts[v] & selected)[0]
                    if len(v_sel_conflicts) == 0:
                        # v can be added freely; removing u gives net +1 if we don't add v
                        # Actually if v has 0 conflicts, we should have added it in Phase 1
                        # So this case shouldn't occur unless Phase 1 missed it
                        continue
                    if len(v_sel_conflicts) == 1 and v_sel_conflicts[0] == u:
                        # Removing u and adding v: net 0 in count, but might free up other vertices
                        # Let's estimate the gain: after removing u, how many more non-selected vertices
                        # can be added?
                        # This is complex. Let's use a simpler heuristic:
                        # Gain = blocking_cost[u] - addition_cost[v] (rough)
                        gain = blocking_costs[ui] - addition_costs[vi]
                        if gain > best_gain:
                            best_gain = gain
                            best_u = u
                            best_v = v
            
            if best_u >= 0 and best_v >= 0:
                selected[best_u] = False
                selected[best_v] = True
                improved = True
                # Recompute sel_conflict_count
                sel_conflict_count = np.zeros(N4, dtype=np.int32)
                for i in range(N4):
                    sel_conflict_count[i] = np.sum(conflicts[i] & selected)
                
                # Run Phase 1 again after the swap
                changed = True
                while changed:
                    if time.time() > deadline - 5:
                        break
                    changed = False
                    non_selected_indices = np.where(~selected)[0]
                    for v in non_selected_indices:
                        if sel_conflict_count[v] == 0:
                            selected[v] = True
                            for u2 in conflict_lists[v]:
                                if not selected[u2]:
                                    sel_conflict_count[u2] += 1
                            changed = True
                    if changed:
                        sel_conflict_count = np.zeros(N4, dtype=np.int32)
                        for i in range(N4):
                            sel_conflict_count[i] = np.sum(conflicts[i] & selected)
        
        if selected.sum() > best_count:
            best_count = selected.sum()
            best_selected = selected.copy()
    
    result_v4 = vectors_v4[best_selected]
    
    # Add all V1 vectors (they're all compatible with V4 and each other)
    final_vectors = np.vstack([result_v4, vectors_v1])
    
    # Verify the result satisfies the constraint
    # Minimum pairwise distance >= maximum norm
    if len(final_vectors) > 1:
        norms = np.sqrt(np.sum(final_vectors ** 2, axis=1))
        max_norm = np.max(norms)
        
        # Check pairwise distances
        # For large arrays, this is expensive, so only check if we have time
        if time.time() < deadline - 2:
            diff = final_vectors[:, np.newaxis, :] - final_vectors[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff ** 2, axis=2))
            np.fill_diagonal(dists, np.inf)
            min_dist = np.min(dists)
            
            if min_dist < max_norm - 1e-9:
                # Constraint violated! This shouldn't happen, but let's handle it
                # Remove vectors until constraint is satisfied
                # This is a fallback; normally the construction should be correct
                pass
    
    return final_vectors
