import numpy as np
import time
import itertools

def construct():
    start_time = time.time()
    deadline = 145.0

    D = 11
    
    # Generate supports (4-subsets)
    supports = list(itertools.combinations(range(D), 4))
    n_supports = len(supports) # 330
    
    # Map support to index
    support_to_idx = {s: i for i, s in enumerate(supports)}
    
    # Build adjacency list for supports with intersection size 3
    # For each support, find others with 3 common elements
    adj = [[] for _ in range(n_supports)]
    
    # To speed up, group by 3-subsets
    # If two 4-subsets share 3 elements, they share a 3-subset.
    # Each 4-subset contains 4 3-subsets.
    # If two 4-subsets share a 3-subset, their intersection is at least 3.
    # Since they are distinct 4-subsets, intersection is exactly 3.
    
    # Map 3-subset to list of 4-subsets containing it
    from collections import defaultdict
    tri_to_tet = defaultdict(list)
    
    for i, s in enumerate(supports):
        # Generate 3-subsets of s
        for tri in itertools.combinations(s, 3):
            tri_to_tet[tri].append(i)
    
    for tri, tets in tri_to_tet.items():
        # All pairs in tets are connected
        for u in range(len(tets)):
            for v in range(u + 1, len(tets)):
                i = tets[u]
                j = tets[v]
                adj[i].append(j)
                adj[j].append(i)
    
    # Greedy assignment of patterns
    # Patterns are 3-bit integers 0..7
    # For each support, we assign a set of patterns
    # We process supports in order of increasing degree (or random)
    
    # Let's try multiple random orderings
    rng = np.random.default_rng(42)
    
    best_vectors = []
    best_count = 0
    
    max_restarts = 20
    
    for restart in range(max_restarts):
        elapsed = time.time() - start_time
        if elapsed > deadline - 5:
            break
        
        order = rng.permutation(n_supports)
        
        # assigned_patterns[i] = set of patterns assigned to support i
        assigned_patterns = [set() for _ in range(n_supports)]
        
        # Process in order
        for idx in order:
            # Find patterns used by already processed neighbors
            used_by_neighbors = set()
            for neighbor in adj[idx]:
                # Only consider if neighbor was processed before idx in this restart?
                # In a simple greedy, we usually process in the given order.
                # So we only look at neighbors that are earlier in 'order'.
                # We can maintain a 'processed' boolean array or just check index in order?
                # Easier: maintain a 'blocked_patterns' set for current node based on processed neighbors.
                # But we need to know which neighbors are processed.
                pass
            
            # This simple loop is O(N^2) per restart. 330^2 is small.
            # Let's refine:
            # We iterate 'order'. For each node, we check its neighbors.
            # We need to know if the neighbor has been assigned yet.
            
            processed = [False] * n_supports
            # Re-initialize for each restart? No, 'order' defines the sequence.
            
            # Let's re-implement the inner loop properly
            # We need to track which patterns are "blocked" for the current node.
            # A pattern is blocked if ANY processed neighbor has it.
            
            # To make it fast, we can just iterate.
            
            # Reset assigned for this restart
            assigned_patterns = [set() for _ in range(n_supports)]
            processed = [False] * n_supports
            
            current_vectors = []
            
            for idx in order:
                blocked = set()
                for neighbor in adj[idx]:
                    if processed[neighbor]:
                        blocked |= assigned_patterns[neighbor]
                
                available = set(range(8)) - blocked
                # Assign all available patterns
                assigned_patterns[idx] = available
                processed[idx] = True
                
                # Generate vectors for this support
                support = supports[idx]
                # Map pattern bits to signs
                # Pattern p (0..7) corresponds to signs on the 3 common indices?
                # Wait, the "3 common indices" are relative to the neighbor.
                # The pattern assignment is local to the support's internal structure?
                # No, the conflict depends on the intersection.
                # The "pattern" is the sign configuration on the intersection.
                # But the intersection is different for each neighbor!
                
                # This approach is flawed. The "pattern" must be consistent across all neighbors.
                # The sign of a vector is fixed.
                # So we must assign signs to the 4 positions of the support.
                # There are 16 sign combinations.
                # For a neighbor with intersection 3, the "pattern" is the projection of the signs onto the 3 common indices.
                
                # So, for support idx, we select a subset of the 16 sign vectors.
                # Let S_idx be the set of sign vectors selected.
                # For each neighbor j, let P_j(S_idx) be the set of projections of S_idx onto the common indices of idx and j.
                # Constraint: P_j(S_idx) \cap P_k(S_j) = empty for all neighbors.
                
                # This is complex to check greedily with just "patterns".
                # However, if we just pick random sign vectors and check compatibility, it might work.
                
                # Let's go back to the vector-level greedy but optimized with the support structure.
                # Since supports with intersection < 3 are always compatible,
                # we can pick ALL 16 vectors for a support if it has no processed neighbors with intersection 3.
                # If it has processed neighbors with intersection 3, we must filter.
                
                # Let's try a simpler vector greedy:
                # 1. Shuffle all 5280 vectors.
                # 2. Greedy select.
                # This is what the current program does.
                # The current program scores 1.2682.
                
                # Can we do better?
                # The support-based insight:
                # If we process supports in an order, and for each support we pick the max subset compatible with previous supports.
                
                # Let's implement this specific greedy.
                
                # For support idx:
                # Candidates: 16 vectors.
                # For each candidate v:
                #   Check compatibility with ALL selected vectors from previous supports.
                #   If compatible, add to local_selected.
                # Add local_selected to global selected.
                
                # This is O(16 * N_selected * 11) per support.
                # Total O(330 * 16 * 5280 * 11) ~ 3e9. Too slow.
                
                # We can optimize the check.
                # Only neighbors with intersection 3 matter.
                # For support idx, let Neighbors3 be the list of processed neighbors with intersection 3.
                # For each candidate v in idx:
                #   For each neighbor j in Neighbors3:
                #      For each u in Selected[j]:
                #          if dot(v, u) > 2: conflict.
                
                # Number of Neighbors3 is small (<=28).
                # Size of Selected[j] is <=16.
                # So check is 16 * 28 * 16 * 11 ~ 78k ops per support.
                # Total 330 * 78k ~ 25M ops. Feasible.
                
                # Let's implement this.
                
                pass # Skip this restart's inner logic for now, will rewrite below

    # Rewrite the main loop with the optimized support-greedy
    
    best_result_vectors = None
    best_k = 0
    
    for restart in range(max_restarts):
        elapsed = time.time() - start_time
        if elapsed > deadline - 5:
            break
        
        order = rng.permutation(n_supports)
        
        # selected_by_support[i] = list of vectors (indices in global vector list or actual arrays)
        # We'll store actual arrays for simplicity, or indices.
        # Let's store indices in the global all_vectors array.
        
        # Precompute global vector list
        all_vectors = []
        vec_idx_map = {} # (support_idx, sign_tuple) -> global_idx
        for s_idx, s in enumerate(supports):
            for signs in itertools.product([1, -1], repeat=4):
                v = [0] * D
                for pos, sign in zip(s, signs):
                    v[pos] = sign
                all_vectors.append(v)
                vec_idx_map[(s_idx, signs)] = len(all_vectors) - 1
        
        all_vectors_np = np.array(all_vectors, dtype=np.int8)
        
        # Precompute dot products? No, just compute on the fly for the small sets.
        
        selected_indices = []
        selected_by_support = [[] for _ in range(n_supports)]
        
        for s_idx in order:
            # Get neighbors with intersection 3 that are already processed
            # We need to know which neighbors are processed.
            # We can just check if selected_by_support[neighbor] is not empty?
            # No, a neighbor might be processed but have 0 selected vectors.
            # We need a 'processed' flag.
            
            # Let's maintain a processed set
            # Actually, 'order' implies sequence.
            # We can just check if the neighbor's turn has passed.
            # Since we are iterating 'order', we can keep a set of processed support indices.
            
            # Optimization: Precompute list of neighbors for each support.
            # neighbors3[s_idx] = list of support indices j such that |intersection| == 3
            
            # We already have 'adj' which contains exactly these neighbors.
            
            local_candidates = []
            for signs in itertools.product([1, -1], repeat=4):
                v = all_vectors_np[vec_idx_map[(s_idx, signs)]]
                
                compatible = True
                # Check against selected vectors in neighbors
                for n_idx in adj[s_idx]:
                    if not selected_by_support[n_idx]:
                        continue
                    # Check against all vectors in selected_by_support[n_idx]
                    for u_idx in selected_by_support[n_idx]:
                        u = all_vectors_np[u_idx]
                        # Compute dot product
                        # Since we are in Python loop, this might be slow.
                        # But 330 * 16 * 28 * 16 is small.
                        dot = np.dot(v, u)
                        if dot > 2:
                            compatible = False
                            break
                    if not compatible:
                        break
                
                if compatible:
                    local_candidates.append(vec_idx_map[(s_idx, signs)])
            
            # Add all compatible candidates to selected
            for idx in local_candidates:
                selected_indices.append(idx)
                selected_by_support[s_idx].append(idx)
        
        k = len(selected_indices)
        if k > best_k:
            best_k = k
            best_result_vectors = [all_vectors_np[i] for i in selected_indices]
        
        if best_k > 400:
            break

    if best_result_vectors is None:
        return np.array([], dtype=np.int64).reshape(0, D)
    
    return np.array(best_result_vectors, dtype=np.int64)
