import numpy as np
import itertools
import time

D = 11

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill

    # Generate W4 vectors (weight 4, entries in {-1,0,1}, norm^2=4)
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)

    # Generate W1 vectors (weight 1, entries in {-2,0,2}, norm^2=4)
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)

    # Combine all candidates
    all_vectors = np.array(v4_candidates + v1_candidates, dtype=np.int64)
    N = len(all_vectors)
    
    # Precompute norms squared (all are 4)
    # Norms = np.sum(all_vectors**2, axis=1)
    
    # Compute dot product matrix
    # Dot[i, j] = all_vectors[i] . all_vectors[j]
    # We need |Dot[i, j]| <= 2 for compatibility (distance >= norm)
    # Actually, condition is ||v_i - v_j|| >= ||v_i||
    # ||v_i - v_j||^2 = ||v_i||^2 + ||v_j||^2 - 2 v_i . v_j >= ||v_i||^2
    # Since ||v_i||^2 = ||v_j||^2 = 4:
    # 4 + 4 - 2 v_i . v_j >= 4  =>  4 >= 2 v_i . v_j  =>  v_i . v_j <= 2
    # Also need to check lower bound? No, distance is symmetric.
    # Wait, the condition is "minimum pairwise distance is at least the maximum vector norm".
    # Max norm is 2 (since norm^2=4).
    # So dist(u, v) >= 2.
    # dist^2 = ||u-v||^2 = ||u||^2 + ||v||^2 - 2 u.v = 4 + 4 - 2 u.v = 8 - 2 u.v.
    # We need 8 - 2 u.v >= 4  =>  4 >= 2 u.v  =>  u.v <= 2.
    # Is there a lower bound on u.v?
    # If u.v is very negative, dist is large, which is fine.
    # So the constraint is simply u.v <= 2.
    
    # Let's verify with W1 vectors.
    # v1 = [2, 0, ...], v2 = [-2, 0, ...]
    # v1.v2 = -4. dist^2 = 8 - 2(-4) = 16. dist = 4 >= 2. OK.
    # v1 = [2, 0, ...], v3 = [2, 0, ...] (same vector, not allowed in set)
    # v1 = [2, 0, ...], v4 = [0, 2, ...]
    # v1.v4 = 0. dist^2 = 8. dist = sqrt(8) > 2. OK.
    
    # So the constraint for any pair i, j in the set is: all_vectors[i] . all_vectors[j] <= 2.
    
    Dot = all_vectors @ all_vectors.T
    # Conflict if Dot[i, j] > 2
    # Note: Dot[i, i] = 4, which is > 2, but we don't consider self-conflicts in the independent set sense (we just pick unique indices).
    # We build a boolean conflict matrix
    conflict = (Dot > 2)
    # Set diagonal to False to avoid self-conflict in graph logic if needed, though we track "in_set" separately.
    np.fill_diagonal(conflict, False)
    
    # Initial greedy construction
    in_set = np.zeros(N, dtype=bool)
    # Start with W1 vectors? They are often compatible with many W4s.
    # Actually, let's just do a greedy pass over all vectors.
    # To be fast, we can order by "potential" or just iterate.
    # A good heuristic: pick vectors that have few conflicts with already selected ones.
    
    # We maintain conflict_count[i] = number of j in in_set such that conflict[i, j] is True.
    conflict_count = np.zeros(N, dtype=np.int32)
    
    # Greedy Phase
    # Randomize order slightly to allow restarts
    rng = np.random.default_rng(42)
    
    # We will do multiple restarts with different greedy strategies
    
    best_in_set = np.zeros(N, dtype=bool)
    best_size = 0
    
    n_restarts = 0
    max_restarts = 50 # Adjust based on time
    
    while time.time() < deadline - 10 and n_restarts < max_restarts:
        n_restarts += 1
        
        in_set = np.zeros(N, dtype=bool)
        conflict_count = np.zeros(N, dtype=np.int32)
        alive = np.ones(N, dtype=bool)
        
        # Greedy Min-Conflict Selection
        # We pick a vector that minimizes conflict_count among alive vectors.
        # If ties, break randomly.
        while np.any(alive) and time.time() < deadline - 5:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            
            counts = conflict_count[alive_indices]
            min_count = np.min(counts)
            cands = alive_indices[counts == min_count]
            
            # Pick one
            chosen = cands[rng.integers(len(cands))]
            
            in_set[chosen] = True
            # Update conflict counts for neighbors of chosen
            nbrs = np.where(conflict[chosen])[0]
            conflict_count[nbrs] += 1
            
            # Remove chosen and its neighbors from alive?
            # No, we can't remove neighbors from alive if we want to potentially pick them later?
            # Wait, if we pick a node, its neighbors CANNOT be picked.
            # So we MUST remove them from the pool of candidates for the current set.
            # This is the standard greedy independent set algorithm.
            
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            alive[to_remove] = False
            
        current_size = in_set.sum()
        
        if current_size > best_size:
            best_size = current_size
            best_in_set = in_set.copy()
            
        # Simulated Annealing / Local Search
        # We need a good local search to improve the greedy result.
        # Moves:
        # 1. Add a vertex v if conflict_count[v] == 0.
        # 2. Remove a vertex u.
        # 3. Swap: Remove u, Add v if v becomes conflict-free after removing u.
        
        def local_search(in_set_init, max_time=10.0, seed_rng=None):
            r = seed_rng if seed_rng is not None else rng
            in_set = in_set_init.copy()
            current_size = in_set.sum()
            
            # Recalculate conflict_count
            conflict_count = np.zeros(N, dtype=np.int32)
            in_set_indices = np.where(in_set)[0]
            for i in in_set_indices:
                conflict_count[conflict[i]] += 1
            
            T = 1.0
            min_T = 0.01
            cooling = 0.9995
            local_best_size = current_size
            local_best_in_set = in_set.copy()
            
            sa_deadline = time.time() + max_time
            stuck_counter = 0
            step = 0
            
            while time.time() < sa_deadline:
                step += 1
                
                # Determine move
                # Find vertices not in set
                not_in_set = np.where(~in_set)[0]
                if len(not_in_set) == 0:
                    break
                    
                # Option 1: Add
                # Find vertices with 0 conflicts
                counts = conflict_count[not_in_set]
                zero_cands = not_in_set[counts == 0]
                
                if len(zero_cands) > 0 and (len(in_set_indices) == 0 or r.random() < 0.7):
                    # Add a zero-conflict vertex
                    v = zero_cands[r.integers(len(zero_cands))]
                    in_set[v] = True
                    current_size += 1
                    conflict_count[conflict[v]] += 1
                    
                    if current_size > local_best_size:
                        local_best_size = current_size
                        local_best_in_set = in_set.copy()
                        stuck_counter = 0
                    else:
                        stuck_counter += 1
                    continue
                
                # Option 2: Swap (Remove one, Add one or more)
                # This is more complex.
                # Pick a random vertex u in set to remove.
                if current_size > 0:
                    u = in_set_indices[rng.integers(len(in_set_indices))]
                    
                    # Temporarily remove u
                    in_set[u] = False
                    current_size -= 1
                    conflict_count[conflict[u]] -= 1
                    
                    # Find vertices that were conflicting with u but are now potentially addable
                    # Candidates are neighbors of u that are not in set.
                    u_nbrs = np.where(conflict[u])[0]
                    # Filter to those not in set
                    cand_mask = ~in_set[u_nbrs]
                    cands = u_nbrs[cand_mask]
                    
                    if len(cands) > 0:
                        # Check conflict counts for these candidates
                        cand_counts = conflict_count[cands]
                        addable = cands[cand_counts == 0]
                        
                        if len(addable) > 0:
                            # Add one (or more if possible, but let's keep it simple for now)
                            # Pick the one that frees up the most space? Or just random?
                            # Let's pick random for diversity, or the one with most neighbors (greedy)
                            v = addable[rng.integers(len(addable))]
                            in_set[v] = True
                            current_size += 1
                            conflict_count[conflict[v]] += 1
                            
                            # Net change in size is 0.
                            # But we might have improved the "quality" of the set.
                            # If current_size == local_best_size, we might want to accept based on some heuristic?
                            # For now, accept if we didn't get stuck.
                            if current_size > local_best_size:
                                local_best_size = current_size
                                local_best_in_set = in_set.copy()
                                stuck_counter = 0
                            else:
                                stuck_counter += 1
                                # Accept swap with probability exp(-delta/T) if delta > 0?
                                # Here delta = 0. So always accept? Or accept with prob 0.5?
                                if r.random() < 0.5:
                                    pass # Keep the swap
                                else:
                                    # Revert
                                    in_set[v] = False
                                    in_set[u] = True
                                    conflict_count[conflict[v]] -= 1
                                    conflict_count[conflict[u]] += 1
                        else:
                            # No addable, revert
                            in_set[u] = True
                            current_size += 1
                            conflict_count[conflict[u]] += 1
                            stuck_counter += 1
                    else:
                        # No candidates, revert
                        in_set[u] = True
                        current_size += 1
                        conflict_count[conflict[u]] += 1
                        stuck_counter += 1
                else:
                    stuck_counter += 1
                
                if stuck_counter > 500:
                    T = max(T, 1.0)
                    stuck_counter = 0
                
                T *= cooling
                if T < min_T:
                    T = min_T
            
            return local_best_in_set, local_best_size

        # Run Local Search
        remaining_time = deadline - 5 - time.time()
        if remaining_time > 2:
            ls_time = min(remaining_time * 0.5, 15.0)
            if ls_time > 1:
                ls_in_set, ls_size = local_search(in_set, max_time=ls_time)
                if ls_size > best_size:
                    best_size = ls_size
                    best_in_set = ls_in_set.copy()
    
    # Final Polish
    # Try to add any remaining 0-conflict vectors
    if best_size > 0:
        in_set = best_in_set.copy()
        conflict_count = np.zeros(N, dtype=np.int32)
        in_set_indices = np.where(in_set)[0]
        for i in in_set_indices:
            conflict_count[conflict[i]] += 1
        
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            not_in_set = np.where(~in_set)[0]
            if len(not_in_set) == 0:
                break
            counts = conflict_count[not_in_set]
            zero_cands = not_in_set[counts == 0]
            if len(zero_cands) > 0:
                for v in zero_cands:
                    in_set[v] = True
                    conflict_count[conflict[v]] += 1
                    improved = True
        
        if in_set.sum() > best_size:
            best_size = in_set.sum()
            best_in_set = in_set.copy()

    final_vectors = all_vectors[best_in_set]
    return final_vectors

# EVOLVE-BLOCK-END
