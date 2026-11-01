"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct():
    """
    Construct nonzero integer vectors in R^11 whose minimum pairwise distance
    is at least the maximum vector norm.
    
    Strategy:
    1. Generate all minimal norm vectors (norm^2 = 2 and 4).
       Type 1: two +/-1s (norm^2=2)
       Type 2: one +/-2 (norm^2=4)
       Type 3: four +/-1s (norm^2=4)
       
    2. The constraint is: min_pairwise_dist >= max_norm.
       If we include any norm^2=4 vector, max_norm=2, so we need dist^2 >= 4 for ALL pairs.
       If we only include norm^2=2 vectors, max_norm=sqrt(2), so we need dist^2 >= 2.
       
       However, mixing them forces the stricter constraint (dist^2 >= 4) on the entire set.
       The set of all vectors with dist^2 >= 4 is an independent set in the graph where edges connect pairs with dist^2 < 4.
       
       Note: Type 1 vectors with dist^2=2 (dot=1) conflict with each other under the strict constraint.
       Type 1 vectors with dist^2=4 (dot=0) are compatible.
       
       We will find the Maximum Independent Set (MIS) in the conflict graph defined by dist^2 < 4.
       This guarantees validity for any subset.
       
    3. Use a fast greedy algorithm with randomized tie-breaking and local search (2-opt/3-opt) 
       to improve the solution within the time budget.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # --- Generate Candidates ---
    candidates = []
    
    # Type 1: two +/-1's, rest 0 (norm^2 = 2)
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i] = si
                v[j] = sj
                candidates.append(v)
                
    # Type 2: one +/-2, rest 0 (norm^2 = 4)
    for i in range(D):
        for s in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = s * 2
            candidates.append(v)
            
    # Type 3: four +/-1's, rest 0 (norm^2 = 4)
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for pos, sign in zip(positions, signs):
                v[pos] = sign
            candidates.append(v)
            
    candidates = np.array(candidates, dtype=np.int64)
    n = len(candidates)
    
    # Compute norms squared
    norms_sq = np.sum(candidates**2, axis=1)
    
    # Compute dot products
    dots = candidates @ candidates.T
    
    # Compute squared distances
    # dist^2 = ||u||^2 + ||v||^2 - 2*u.v
    dist_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * dots
    
    # Conflict if dist_sq < 4 (strictly less than max_norm^2 = 4)
    # This ensures min_dist >= 2 (max_norm) for any set containing norm-2 vectors.
    # If the final set only contains norm-sqrt(2) vectors, this is a stricter constraint than needed,
    # but it's a safe starting point. We could potentially relax this later if we detect no norm-2 vectors.
    # However, for simplicity and robustness, we stick to the stricter constraint.
    # Actually, if we select a set S, the condition is min_dist(S) >= max_norm(S).
    # If we enforce dist_sq >= 4 for all pairs in S, then min_dist >= 2.
    # If max_norm(S) <= 2, the condition holds.
    # All our candidates have norm <= 2. So this is sufficient.
    
    conflict = (dist_sq < 4)
    np.fill_diagonal(conflict, False)
    
    # Build adjacency lists and neighbor sets for fast operations
    adj_lists = [np.where(conflict[i])[0].tolist() for i in range(n)]
    neighbor_sets = [set(adj) for adj in adj_lists]
    degrees = np.array([len(adj) for adj in adj_lists])
    
    def greedy_independent_set_fast(degrees, n, rng, neighbor_sets):
        """
        Fast greedy MIS.
        Repeatedly picks the node with minimum degree among remaining nodes.
        """
        remaining = set(range(n))
        deg = degrees.copy()
        selected = []
        
        while remaining:
            # Find min degree among remaining
            # Optimization: if n is large, this loop is expensive.
            # But n ~ 5500, so it's manageable.
            min_deg = min(deg[i] for i in remaining)
            candidates_for_pick = [i for i in remaining if deg[i] == min_deg]
            chosen = rng.choice(candidates_for_pick)
            
            selected.append(chosen)
            
            # Remove chosen and neighbors
            to_remove = neighbor_sets[chosen] | {chosen}
            to_remove_in_remaining = to_remove & remaining
            
            for node in to_remove_in_remaining:
                remaining.discard(node)
                # Update degrees of neighbors that are still remaining
                for neighbor in neighbor_sets[node]:
                    if neighbor in remaining:
                        deg[neighbor] -= 1
                        
        return selected

    def local_search_2opt(selected, n, neighbor_sets, rng, max_iterations=100):
        """
        Improve independent set via 2-opt swaps.
        Try to add a node if it doesn't conflict.
        Try to remove one node and add two others.
        """
        selected_set = set(selected)
        current_size = len(selected_set)
        improved = True
        iterations = 0
        
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            
            # Phase 1: Add any compatible node
            for i in range(n):
                if i not in selected_set:
                    if not (neighbor_sets[i] & selected_set):
                        selected_set.add(i)
                        current_size += 1
                        improved = True
                        
            # Phase 2: 2-opt swap (remove 1, add 2)
            if not improved:
                # Sort selected by degree or just iterate
                for remove_candidate in list(selected_set):
                    # Find nodes that are currently blocked ONLY by remove_candidate
                    # and don't conflict with other selected nodes
                    freed_nodes = []
                    for i in range(n):
                        if i not in selected_set:
                            # Check if i conflicts with any selected node
                            conflicts = neighbor_sets[i] & selected_set
                            if conflicts == {remove_candidate}:
                                freed_nodes.append(i)
                    
                    if len(freed_nodes) >= 2:
                        # We can remove 1 and add 2
                        # Pick the first 2
                        add_nodes = freed_nodes[:2]
                        selected_set.discard(remove_candidate)
                        for node in add_nodes:
                            selected_set.add(node)
                        current_size += 1
                        improved = True
                        break
                        
        return list(selected_set)

    best_selected = []
    best_size = 0
    
    # Run multiple restarts
    attempts = 0
    consecutive_no_improvement = 0
    
    while time.time() < deadline:
        attempts += 1
        rng = np.random.RandomState(attempts * 7919 + 42)
        
        # Greedy pass
        selected = greedy_independent_set_fast(degrees, n, rng, neighbor_sets)
        
        # Local search improvement
        selected = local_search_2opt(selected, n, neighbor_sets, rng, max_iterations=50)
        
        if len(selected) > best_size:
            best_size = len(selected)
            best_selected = selected.copy()
            consecutive_no_improvement = 0
        else:
            consecutive_no_improvement += 1
            
        # Early stop if stuck
        if consecutive_no_improvement > 20 and best_size > 200:
            break
            
        # Check time
        if attempts % 5 == 0 and time.time() > deadline - 10:
            break

    if not best_selected:
        # Fallback: return all Type 1 vectors? No, they might conflict.
        # Just return empty or a small valid set.
        # Actually, greedy should always find something.
        return np.array([], dtype=np.int64).reshape(0, D)

    result = candidates[best_selected]
    return result

# EVOLVE-BLOCK-END
