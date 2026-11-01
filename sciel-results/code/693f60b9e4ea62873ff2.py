"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    # Generate W4 vectors (weight 4, entries in {-1,0,1}, norm^2=4)
    # There are C(11,4) * 2^4 = 330 * 16 = 5280 such vectors
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
            
    # Generate W1 vectors (weight 1, entries in {-2,0,2}, norm^2=4)
    # We can add 11 of these (one per dimension, e.g., all +2)
    # They do not conflict with W4 vectors (dot <= 2)
    # And we can pick at most 11 non-conflicting W1s (e.g., 2e0...2e10)
    v1_final = []
    for pos in range(D):
        v = [0] * D
        v[pos] = 2
        v1_final.append(v)
        
    vectors_v4 = np.array(v4_candidates, dtype=np.int64)
    N4 = len(vectors_v4)
    
    if N4 == 0:
        return np.array(v1_final)

    # Compute conflict matrix for W4 vectors
    # Conflict if dot > 2
    # As analyzed, conflicts only happen if supports intersect in 3 and signs match on those 3.
    # We can compute this efficiently or just use matrix multiplication.
    # 5280 x 5280 matrix is ~28M entries. Memory ~200MB (int64) or 100MB (int32).
    # NumPy matmul is fast.
    
    dots_v4 = vectors_v4 @ vectors_v4.T
    conflicts_v4 = dots_v4 > 2
    np.fill_diagonal(conflicts_v4, False)
    
    # Convert to adjacency list for fast lookup in local search
    # neighbors[i] is a list of indices j such that v4[i] conflicts with v4[j]
    neighbors = [np.where(conflicts_v4[i])[0] for i in range(N4)]
    
    # We will find the maximum independent set in this graph.
    # Then add the 11 W1 vectors.
    
    rng = np.random.default_rng(42)
    
    best_in_set = np.zeros(N4, dtype=bool)
    best_size = 0
    
    def greedy_init(seed):
        """Randomized greedy independent set."""
        local_rng = np.random.default_rng(seed)
        order = local_rng.permutation(N4)
        in_set = np.zeros(N4, dtype=bool)
        blocked = np.zeros(N4, dtype=bool)
        size = 0
        for v in order:
            if not blocked[v]:
                in_set[v] = True
                size += 1
                blocked[v] = True
                if len(neighbors[v]) > 0:
                    blocked[neighbors[v]] = True
        return in_set, size

    def local_search(in_set, max_iters=1000):
        """
        Local search to improve the independent set.
        Moves:
        1. Add a vertex with 0 conflicts.
        2. Swap: Remove a vertex, add a vertex that was blocked only by it.
        """
        in_set = in_set.copy()
        size = int(in_set.sum())
        
        # Conflict count for each vertex: number of neighbors in the set
        conflict_count = np.zeros(N4, dtype=np.int32)
        selected_idx = np.where(in_set)[0]
        for v in selected_idx:
            conflict_count[neighbors[v]] += 1
            
        # Also track which vertices are "blocked" (conflict_count > 0)
        # and which are free (conflict_count == 0)
        
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline - 5:
            improved = False
            iters += 1
            
            # 1. Try to add any free vertex
            free_vertices = np.where(conflict_count == 0)[0]
            # Exclude those already in set?
            # conflict_count is 0 for free vertices.
            # If a vertex is in set, its neighbors have count > 0, but itself has count 0?
            # No, conflict_count[v] counts neighbors in set.
            # If v is in set, it is not a neighbor of itself (no self loops).
            # So conflict_count[v] for v in set is the number of OTHER selected neighbors?
            # No, an independent set has NO selected neighbors for any selected vertex.
            # So for any v in in_set, conflict_count[v] == 0.
            # For any v NOT in in_set, conflict_count[v] > 0 iff it has a neighbor in in_set.
            
            # So, free_vertices includes all in_set vertices and all addable vertices.
            # We need to filter out in_set.
            addable = free_vertices[~in_set[free_vertices]]
            
            if len(addable) > 0:
                # Add a random addable vertex
                v = addable[rng.integers(len(addable))]
                in_set[v] = True
                size += 1
                # Update conflict counts for neighbors
                for u in neighbors[v]:
                    conflict_count[u] += 1
                improved = True
                continue
            
            # 2. Try 1-for-1 swaps
            # If no free vertices to add, try to swap.
            # Remove a selected vertex s, which unblocks some neighbors.
            # If any unblocked neighbor v can be added, we do the swap.
            # To be efficient, we can check for each selected vertex s,
            # if there is a neighbor v such that conflict_count[v] == 1 and that 1 is s.
            
            # This is expensive to check naively.
            # Alternative: Randomly pick a selected vertex s to remove.
            # Check if any of its neighbors become addable.
            if len(selected_idx) > 0:
                s = selected_idx[rng.integers(len(selected_idx))]
                
                # Temporarily remove s
                in_set[s] = False
                size -= 1
                for u in neighbors[s]:
                    conflict_count[u] -= 1
                
                # Find any u that is now addable (conflict_count[u] == 0 and not in_set)
                # We only need to check neighbors of s, as only they changed.
                # Actually, other vertices might have been addable before?
                # No, we established no addable vertices existed.
                # So only neighbors of s can become addable.
                
                candidate_adds = []
                for u in neighbors[s]:
                    if not in_set[u] and conflict_count[u] == 0:
                        candidate_adds.append(u)
                        
                if len(candidate_adds) > 0:
                    # Pick one to add
                    v = candidate_adds[rng.integers(len(candidate_adds))]
                    in_set[v] = True
                    size += 1
                    for u in neighbors[v]:
                        conflict_count[u] += 1
                    # Update selected_idx? We can just recompute if needed, 
                    # but for the next iteration's random pick, we can use a stale list 
                    # or just check in_set directly.
                    # For efficiency, let's not update selected_idx immediately, 
                    # but the next iteration's "random pick from selected_idx" might pick s (which is now False).
                    # We should handle that.
                    improved = True
                else:
                    # Swap failed, put s back
                    in_set[s] = True
                    size += 1
                    for u in neighbors[s]:
                        conflict_count[u] += 1
                    # Update selected_idx to be accurate for next iter?
                    # Or just rely on the fact that we check in_set later?
                    # The code above uses selected_idx for random pick.
                    # If we don't update it, we might pick a vertex not in set.
                    # Let's update selected_idx if we fail? Or just at the end of loop?
                    # Better: update selected_idx at the start of each iteration if it got stale?
                    # Or just use np.where(in_set) each time? N=5280, it's fast enough.
            
            # Update selected_idx for next iteration if it changed
            if improved:
                selected_idx = np.where(in_set)[0]
                
        return in_set, size

    # Run multiple restarts
    n_restarts = 0
    while time.time() < deadline - 10 and n_restarts < 1000:
        n_restarts += 1
        seed = n_restarts * 13 + 7
        in_set, size = greedy_init(seed)
        
        # Polish with local search
        in_set_polished, size_polished = local_search(in_set, max_iters=500)
        
        if size_polished > best_size:
            best_size = size_polished
            best_in_set = in_set_polished.copy()
            
        # If we get a very large set, we can reduce restarts
        # Max possible is roughly N/2? No, degree 56, so ~N/57.
        # 5280/57 ~ 92.
        # If we get > 90, it's likely near optimal.
        if best_size > 95:
            break

    # Final result: Best W4 set + 11 W1 vectors
    final_v4_indices = np.where(best_in_set)[0]
    result_v4 = vectors_v4[final_v4_indices]
    
    # Stack W4 and W1
    final_vectors = np.vstack([result_v4, np.array(v1_final)])
    
    return final_vectors

# EVOLVE-BLOCK-END
