"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill
    
    # Generate W4 vectors
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
            
    # Generate W1 vectors
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
        
    # Compute conflict matrix for W4
    # Conflict if u.v > 2
    dots = vectors_v4 @ vectors_v4.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Compute degrees
    degrees = conflicts.sum(axis=1).astype(np.int32)
    
    rng = np.random.default_rng(42)
    
    def greedy_mis(conflicts, deadline, rng, degrees):
        """
        Greedy Maximum Independent Set with degree-based ordering.
        """
        N = len(conflicts)
        alive = np.ones(N, dtype=bool)
        deg = degrees.copy()
        
        selected_indices = []
        
        while np.any(alive):
            if time.time() > deadline:
                break
                
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
                
            # Find min degree among alive
            min_deg = np.min(deg[alive_indices])
            candidates = alive_indices[deg[alive_indices] == min_deg]
            
            # Random tie-breaking
            chosen = candidates[rng.integers(len(candidates))]
            selected_indices.append(chosen)
            
            # Find neighbors to remove
            neighbors = np.where(conflicts[chosen])[0]
            to_remove = np.unique(np.concatenate(([chosen], neighbors)))
            
            # Update alive
            alive[to_remove] = False
            
            # Update degrees for remaining alive nodes
            remaining_alive = np.where(alive)[0]
            if len(remaining_alive) > 0:
                removed_conflicts = conflicts[remaining_alive][:, to_remove].sum(axis=1)
                deg[remaining_alive] -= removed_conflicts

        return selected_indices
    
    def spectral_init(conflicts, rng, n_samples=50):
        """
        Use spectral relaxation to identify a dense subgraph of mutually compatible vectors.
        We look at the non-conflict graph (adjacency = ~conflicts), compute its Laplacian,
        and use the Fiedler vector to partition. We take the larger partition as our initial set
        and then greedily remove conflicts within it.
        """
        N = len(conflicts)
        # Non-conflict graph: edges where NOT conflicts
        non_conf = ~conflicts
        np.fill_diagonal(non_conf, False)
        
        # Compute Laplacian of non-conflict graph
        # L = D - A
        deg_non_conf = non_conf.sum(axis=1).astype(float)
        D_mat = np.diag(deg_non_conf)
        L = D_mat - non_conf.astype(float)
        
        # Compute eigenvalues/eigenvectors
        # For N~4224, full eigendecomposition is feasible within time budget
        try:
            vals, vecs = np.linalg.eigh(L)
            # Fiedler vector is the second smallest eigenvalue's eigenvector
            # Sort by eigenvalue
            idx = np.argsort(vals)
            fiedler = vecs[:, idx[1]]
            
            # Partition: positive vs negative
            # We want the larger cluster to maximize potential set size
            pos_mask = fiedler > 0
            neg_mask = ~pos_mask
            
            # Choose the larger partition
            if np.sum(pos_mask) >= np.sum(neg_mask):
                initial_mask = pos_mask
            else:
                initial_mask = neg_mask
                
            # Now greedily remove conflicts within this partition
            # This gives us an independent set
            selected = []
            alive_in_partition = initial_mask.copy()
            
            # Get indices in partition
            partition_indices = np.where(initial_mask)[0]
            
            # Greedy selection within partition using degree
            # Degree within the non-conflict graph (or rather, conflict degree)
            # We want low conflict degree
            deg_in = conflicts[partition_indices][:, partition_indices].sum(axis=1)
            
            order = np.argsort(deg_in)
            
            for idx_p in order:
                if time.time() > deadline:
                    break
                v = partition_indices[idx_p]
                if alive_in_partition[v]:
                    selected.append(v)
                    # Remove its neighbors from alive
                    neighbors = np.where(conflicts[v])[0]
                    alive_in_partition[neighbors] = False
                    alive_in_partition[v] = False
                    
            return selected
            
        except Exception:
            return []
    
    def local_search_mis(conflicts, deadline, rng, initial_set):
        """
        Local Search for Maximum Independent Set using 1-swap and 2-swap moves.
        Optimized for speed.
        """
        N = len(conflicts)
        current_set = set(initial_set)
        current_size = len(current_set)
        best_set = set(initial_set)
        best_size = current_size
        
        # Precompute conflict lists for fast membership checks
        conflict_lists = [np.where(conflicts[i])[0] for i in range(N)]
        
        step = 0
        no_improve_count = 0
        max_no_improve = 2000  # Reduced to allow more restarts
        
        while time.time() < deadline and no_improve_count < max_no_improve:
            step += 1
            
            # Try 1-swap: remove one, add one (keeps size same, explores landscape)
            if current_size > 0:
                # Pick a random vertex to remove
                v_remove = list(current_set)[rng.integers(len(current_set))]
                
                # Compute which vertices can be added after removing v_remove
                # A vertex u can be added if it doesn't conflict with any other vertex in current_set - {v_remove}
                
                # Efficiently compute candidates:
                # Start with all vertices not in current_set
                candidates_mask = np.ones(N, dtype=bool)
                # Mark current set as not candidates
                for v in current_set:
                    candidates_mask[v] = False
                # v_remove is now a candidate to be added back, but we want to add a NEW one
                # Actually, for 1-swap, we remove v_remove and add u.
                # u must not conflict with any v in current_set - {v_remove}
                
                # Let's compute the "forbidden" set: neighbors of all v in current_set - {v_remove}
                forbidden = np.zeros(N, dtype=bool)
                for v in current_set:
                    if v == v_remove:
                        continue
                    neighbors = conflict_lists[v]
                    forbidden[neighbors] = True
                
                # Candidates are those not forbidden and not in current_set (except v_remove which is removed)
                # We want to add u such that u not in current_set - {v_remove} and u not in forbidden
                # u can be v_remove? No, that's a no-op.
                # u must be a vertex not in current_set at all? Or can it be v_remove?
                # If we add v_remove back, size stays same, set is same. Useless.
                # So u must be in ~current_set.
                
                candidates_mask = ~forbidden & ~np.array([1 if v in current_set else 0 for v in range(N)], dtype=bool)
                
                # Optimization: use boolean array for current_set
                # Let's use a mask for current_set
                current_mask = np.zeros(N, dtype=bool)
                for v in current_set:
                    current_mask[v] = True
                
                candidates_mask = ~forbidden & ~current_mask
                
                candidates = np.where(candidates_mask)[0]
                
                if len(candidates) > 0:
                    # Pick a random candidate
                    v_add = candidates[rng.integers(len(candidates))]
                    
                    # Accept with probability 0.5 to explore
                    if rng.random() < 0.5:
                        current_set.remove(v_remove)
                        current_set.add(v_add)
                        no_improve_count += 1
                    else:
                        no_improve_count += 1
            else:
                no_improve_count += 1
            
            # Try 2-swap: remove two, add two (keeps size same, larger move)
            # Or try "add one" if possible (improves size)
            
            # Check if we can add any vertex to improve size
            # A vertex u can be added if it doesn't conflict with ANY vertex in current_set
            if current_size < N:
                # Compute forbidden for all current set
                forbidden_add = np.zeros(N, dtype=bool)
                for v in current_set:
                    neighbors = conflict_lists[v]
                    forbidden_add[neighbors] = True
                
                # Candidates for adding
                current_mask = np.zeros(N, dtype=bool)
                for v in current_set:
                    current_mask[v] = True
                    
                add_candidates = ~forbidden_add & ~current_mask
                add_cand_indices = np.where(add_candidates)[0]
                
                if len(add_cand_indices) > 0:
                    # We found a vertex we can add!
                    v_add = add_cand_indices[rng.integers(len(add_cand_indices))]
                    current_set.add(v_add)
                    current_size += 1
                    no_improve_count = 0
                    
                    if current_size > best_size:
                        best_set = set(current_set)
                        best_size = current_size
                    continue
            
            # If no improvement, try a 2-swap to explore
            if current_size >= 2:
                # Pick two random vertices to remove
                v1_remove = list(current_set)[rng.integers(len(current_set))]
                remaining = list(current_set - {v1_remove})
                if remaining:
                    v2_remove = remaining[rng.integers(len(remaining))]
                    
                    # Compute forbidden after removing both
                    forbidden_swap = np.zeros(N, dtype=bool)
                    for v in current_set:
                        if v == v1_remove or v == v2_remove:
                            continue
                        neighbors = conflict_lists[v]
                        forbidden_swap[neighbors] = True
                    
                    # Candidates for adding: not forbidden, not in current_set
                    current_mask = np.zeros(N, dtype=bool)
                    for v in current_set:
                        current_mask[v] = True
                    
                    add_candidates = ~forbidden_swap & ~current_mask
                    add_cand_indices = np.where(add_candidates)[0]
                    
                    if len(add_cand_indices) >= 2:
                        # Pick two random candidates
                        idx1 = rng.integers(len(add_cand_indices))
                        idx2 = rng.integers(len(add_cand_indices))
                        while idx2 == idx1:
                            idx2 = rng.integers(len(add_cand_indices))
                        
                        v1_add = add_cand_indices[idx1]
                        v2_add = add_cand_indices[idx2]
                        
                        # Check if the two new vertices are compatible with each other
                        if not conflicts[v1_add, v2_add]:
                            # Perform swap
                            current_set.remove(v1_remove)
                            current_set.remove(v2_remove)
                            current_set.add(v1_add)
                            current_set.add(v2_add)
                            no_improve_count += 1
                        else:
                            no_improve_count += 1
                    else:
                        no_improve_count += 1
            else:
                no_improve_count += 1
            
        return list(best_set)
    
    # Strategy:
    # 1. Get a good initial solution using spectral init + greedy
    # 2. Run local search to improve
    # 3. Use remaining time for multiple restarts with greedy + local search if time permits
    
    best_set = []
    
    # Phase 1: Spectral Init
    spectral_deadline = start_time + 20
    spectral_result = spectral_init(conflicts, rng, spectral_deadline)
    if len(spectral_result) > len(best_set):
        best_set = spectral_result
    
    # Phase 2: Local Search on Spectral Result
    if best_set:
        ls_deadline = start_time + 60
        ls_result = local_search_mis(conflicts, ls_deadline, rng, best_set)
        if len(ls_result) > len(best_set):
            best_set = ls_result
    
    # Phase 3: Greedy Restarts + Local Search
    # If we have time left, try a few greedy restarts
    greedy_deadline = deadline - 10
    n_greedy_tried = 0
    
    while time.time() < greedy_deadline:
        # Run a quick greedy
        g_deadline = min(greedy_deadline, time.time() + 2)
        greedy_result = greedy_mis(conflicts, g_deadline, rng, degrees)
        n_greedy_tried += 1
        
        if len(greedy_result) > len(best_set):
            best_set = greedy_result
            
        # If we found a new best, try to improve it with local search
        if len(greedy_result) == len(best_set) or len(greedy_result) > len(best_set):
            if time.time() < deadline - 5:
                ls_deadline = min(deadline - 5, time.time() + 10)
                ls_result = local_search_mis(conflicts, ls_deadline, rng, greedy_result)
                if len(ls_result) > len(best_set):
                    best_set = ls_result
                    break  # Found a new best, restart loop with new best? No, just continue searching
        
    if not best_set:
        return vectors_v1
    
    result_v4 = vectors_v4[best_set]
    
    # Combine with W1
    final_vectors = np.vstack([result_v4, vectors_v1])
    
    return final_vectors

# EVOLVE-BLOCK-END
