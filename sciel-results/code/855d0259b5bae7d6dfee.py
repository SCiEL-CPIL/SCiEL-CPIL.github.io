"""Kissing number in dimension 11: Greedy + Local Search on Conflict Graph."""
import numpy as np
import itertools
import time
import random


def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    D = 11

    # Generate candidate vectors
    candidates_list = []

    # Weight-4 vectors (norm^2=4)
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates_list.append(v)

    # Weight-3 vectors (norm^2=3)
    for positions in itertools.combinations(range(D), 3):
        for signs in itertools.product([1, -1], repeat=3):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates_list.append(v)

    # Weight-2 vectors (norm^2=2)
    for positions in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates_list.append(v)

    # Weight-1 vectors (norm^2=1)
    for pos in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[pos] = s
            candidates_list.append(v)

    candidates = np.array(candidates_list, dtype=np.int32)
    n = len(candidates)

    # Precompute norms squared
    norms_sq = np.sum(candidates ** 2, axis=1)
    
    # Constraint: ||v_i - v_j||^2 >= max_norm_sq
    # We want to maximize the set size. The "radius" is determined by the max norm in the set?
    # The problem statement says: "minimum pairwise distance is at least the maximum vector norm".
    # If we normalize each vector to norm 2, this means distance >= 2.
    # Original: dist(v_i, v_j) >= max(||v_k||).
    # If we pick a set S, let R = max_{k in S} ||v_k||.
    # Then for all i,j in S, ||v_i - v_j|| >= R.
    # This implies R^2 <= ||v_i - v_j||^2 = ||v_i||^2 + ||v_j||^2 - 2 v_i.v_j.
    # Since ||v_i|| <= R and ||v_j|| <= R, this is not a simple fixed threshold.
    # However, if we restrict ourselves to vectors of the SAME norm, say norm^2 = W,
    # then R = sqrt(W). Condition: W <= W + W - 2 v_i.v_j => 2 v_i.v_j <= W => v_i.v_j <= W/2.
    
    # The previous code assumed a global max_norm_sq = 4 (from weight 4 vectors).
    # If we include weight 4 vectors, R^2 = 4.
    # Condition for any pair (i,j): ||v_i - v_j||^2 >= 4.
    # This is a valid constraint for a set containing weight 4 vectors.
    # If we only use weight 4 vectors, condition is v_i.v_j <= 2.
    # If we mix, e.g., weight 4 and weight 3:
    # R^2 = 4.
    # dist^2(v4, v3) = 4 + 3 - 2 v4.v3 >= 4 => 3 >= 2 v4.v3 => v4.v3 <= 1.5 => v4.v3 <= 1 (since integer).
    # dist^2(v3, v3) = 3 + 3 - 2 v3.v3 >= 4 => 6 - 2 v3.v3 >= 4 => 2 v3.v3 <= 2 => v3.v3 <= 1.
    
    # So, if we include ANY weight 4 vector, the constraint is:
    # 1. v_i.v_j <= 2 if both are weight 4.
    # 2. v_i.v_j <= 1 if one is weight 4, other weight 3.
    # 3. v_i.v_j <= 1 if both are weight 3? No, if we have a weight 4 vector in the set, R=2.
    #    dist(v3, v3) >= 2 => 6 - 2 v3.v3 >= 4 => v3.v3 <= 1.
    # 4. If we have weight 4 and weight 2:
    #    dist(v4, v2) >= 2 => 4 + 2 - 2 v4.v2 >= 4 => 2 >= 2 v4.v2 => v4.v2 <= 1.
    # 5. If we have weight 4 and weight 1:
    #    dist(v4, v1) >= 2 => 4 + 1 - 2 v4.v1 >= 4 => 1 >= 2 v4.v1 => v4.v1 <= 0.
    
    # This mixed constraint is complex.
    # A simpler strategy: Just use Weight 4 vectors.
    # Why? The kissing number is dominated by the densest packings which often use a single shell or symmetric shells.
    # For D=11, the 1656 kissing number is achieved by a specific construction.
    # Let's try to maximize the set of Weight 4 vectors first.
    # If we only use Weight 4 vectors, the constraint is simply:
    # v_i . v_j <= 2 for all i != j.
    
    # Let's build the conflict matrix for Weight 4 vectors only to start, as it's a large subset (5280).
    # If we can find a large independent set here, it's a good baseline.
    # We can later try to add lower weight vectors if they fit.
    
    # Filter to Weight 4
    w4_indices = np.where(norms_sq == 4)[0]
    candidates_w4 = candidates[w4_indices]
    n_w4 = len(candidates_w4)
    
    # Compute dot products for W4
    dot_w4 = candidates_w4 @ candidates_w4.T
    
    # Conflict if dot > 2
    conflict_w4 = (dot_w4 > 2).astype(np.uint8)
    np.fill_diagonal(conflict_w4, 0)
    
    # Greedy + Local Search on W4
    best_set = []
    
    def greedy_select(conflict, n_nodes, rng, bias=0.0):
        """
        Greedy selection with random tie-breaking and optional bias.
        Returns list of selected indices.
        """
        selected = []
        remaining = np.ones(n_nodes, dtype=bool)
        
        while remaining.any():
            # Calculate degree in remaining subgraph
            deg = conflict[remaining, :].sum(axis=1)
            # Actually, we need degree within the remaining set.
            # conflict is (n, n).
            # deg[i] = sum(conflict[i, j] for j in remaining)
            
            # Vectorized degree calculation
            deg = np.sum(conflict[:, remaining], axis=1)
            deg[~remaining] = n_nodes + 1
            
            # Find min degree
            min_deg = deg.min()
            if min_deg >= n_nodes:
                break
                
            # Get candidates with min degree
            candidates_mask = (deg == min_deg) & remaining
            candidates_indices = np.where(candidates_mask)[0]
            
            if len(candidates_indices) == 0:
                break
                
            # Random choice among min-degree nodes
            # If bias > 0, prefer nodes with lower global degree?
            # For now, random.
            chosen = candidates_indices[rng.randint(0, len(candidates_indices))]
            
            selected.append(chosen)
            # Remove chosen and its neighbors from remaining
            remaining[chosen] = False
            neighbors = np.where(conflict[chosen] > 0)[0]
            remaining[neighbors] = False
            
        return selected

    def local_search(selected_indices, conflict, n_nodes, rng, max_iters=1000):
        """
        Try to improve the set by swapping nodes.
        """
        selected_set = set(selected_indices)
        size = len(selected_set)
        best_size = size
        best_set = list(selected_set)
        
        # Precompute compatibility with all selected for each node?
        # This is expensive to update.
        # Instead, for each node in selected, try to swap with a node not in selected.
        
        # We can try:
        # 1. Remove i.
        # 2. Find a node j not in selected that is compatible with all remaining selected.
        # 3. If found, swap.
        
        # To speed up, we can maintain a count of conflicts for each non-selected node with the selected set.
        # conflict_count[j] = number of selected nodes k such that conflict[j,k] == 1.
        # If conflict_count[j] == 0, j can be added.
        
        conflict_counts = np.zeros(n_nodes, dtype=np.int32)
        for s in selected_set:
            conflict_counts += conflict[s]
            
        # Note: conflict[s][s] is 0, so this is correct.
        # A node j is addable if conflict_counts[j] == 0 and j not in selected.
        
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline - 5.0:
            improved = False
            iters += 1
            
            # Try to add a node
            addable = np.where((conflict_counts == 0) & (~np.isin(np.arange(n_nodes), list(selected_set))))[0]
            if len(addable) > 0:
                # Add a random addable node
                j = addable[rng.randint(0, len(addable))]
                selected_set.add(j)
                conflict_counts += conflict[j]
                size += 1
                improved = True
                if size > best_size:
                    best_size = size
                    best_set = list(selected_set)
                continue # Restart search with new state
            
            # Try to swap
            # For each selected i, if we remove it, which nodes become addable?
            # Removing i decreases conflict_counts by conflict[i].
            # A node j (not in selected) becomes addable if:
            #   conflict_counts[j] - conflict[i][j] == 0
            #   and conflict[i][j] must be 1? No, if conflict[i][j] is 0, count doesn't change.
            #   So we look for j such that conflict_counts[j] == 1 and conflict[i][j] == 1.
            #   Then after removing i, count becomes 0.
            
            # This is a 1-for-1 swap.
            # We can vectorize the search for such (i, j) pairs.
            
            # For each i in selected:
            #   potential_j = nodes where conflict_counts[j] == 1 AND conflict[i][j] == 1 AND j not in selected.
            
            # This is still O(N^2) per iteration if not careful.
            # Given N=5280, N^2 is 28M. With 1000 iters, 28B ops. Too slow in Python loops.
            
            # Alternative: Just try random swaps.
            i = list(selected_set)[rng.randint(0, len(selected_set)-1)]
            
            # Temporarily remove i
            old_counts = conflict_counts.copy()
            conflict_counts -= conflict[i]
            selected_set.discard(i)
            
            # Find addable nodes
            addable = np.where((conflict_counts == 0) & (~np.isin(np.arange(n_nodes), list(selected_set))))[0]
            
            if len(addable) > 0:
                j = addable[rng.randint(0, len(addable))]
                # Accept swap
                selected_set.add(j)
                conflict_counts += conflict[j]
                # Size remains same, but structure changed.
                # We accept if it allows further growth? 
                # For now, accept with some probability or always if it reduces "potential" conflicts?
                # Let's accept always to explore.
                improved = True
            else:
                # Revert
                conflict_counts = old_counts
                selected_set.add(i)
                
        return best_set

    rng = np.random.RandomState(42)
    
    # Run multiple restarts
    current_best = []
    
    for restart in range(50):
        if time.time() > deadline - 10.0:
            break
            
        # Greedy initial solution
        init_set = greedy_select(conflict_w4, n_w4, rng)
        
        # Local search
        improved_set = local_search(init_set, conflict_w4, n_w4, rng, max_iters=200)
        
        if len(improved_set) > len(current_best):
            current_best = improved_set
            print(f"Restart {restart}: Found set of size {len(current_best)}")

    # Map back to original indices
    if not current_best:
        # Fallback: just return one vector
        return candidates[w4_indices[0]].reshape(1, -1).astype(np.int64)
        
    final_indices = w4_indices[current_best]
    result = candidates[final_indices]
    return result.astype(np.int64)
