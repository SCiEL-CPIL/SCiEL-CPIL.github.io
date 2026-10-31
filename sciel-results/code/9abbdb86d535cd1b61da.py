import numpy as np
import itertools
import time

D = 11

def generate_weight_4_vectors():
    """Generate all vectors of weight 4 in {-1, 0, 1}^D"""
    candidates = []
    for combo in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, s in zip(combo, signs):
                v[idx] = s
            candidates.append(v)
    return np.array(candidates, dtype=np.int8)

def hypergraph_greedy_mis(conflicts, deadline, seed=0):
    """
    Hypergraph-aware greedy MIS.
    Uses a scoring function that considers both degree and "clique coverage"
    (how many high-degree neighbors the vertex has), which helps identify
    vertices in dense conflict clusters.
    """
    N = len(conflicts)
    alive = np.ones(N, dtype=bool)
    degrees = conflicts.sum(axis=1).astype(np.float64)
    
    # Precompute "clique coverage": for each vertex, count neighbors with high degree
    # A vertex with many high-degree neighbors is in a dense cluster
    high_deg_threshold = np.percentile(degrees, 75)
    high_deg_neighbors = (degrees >= high_deg_threshold)
    clique_coverage = (conflicts & high_deg_neighbors[np.newaxis, :]).sum(axis=1).astype(np.float64)
    
    selected_indices = []
    rng = np.random.RandomState(seed)
    
    while np.any(alive):
        if time.time() > deadline:
            break
        
        alive_indices = np.where(alive)[0]
        if len(alive_indices) == 0:
            break
        
        # Score: prefer vertices with low degree and low clique coverage
        # This tends to pick vertices in sparse regions first
        alive_degrees = degrees[alive_indices]
        alive_cq = clique_coverage[alive_indices]
        
        # Combined score: lower is better
        score = alive_degrees + 0.5 * alive_cq
        
        # Add small random noise for tie-breaking
        noise = rng.random(len(alive_indices)) * 0.01
        score_with_noise = score + noise
        
        # Pick the vertex with minimum score
        best_local_idx = np.argmin(score_with_noise)
        chosen = alive_indices[best_local_idx]
        selected_indices.append(chosen)
        
        # Find neighbors to remove
        neighbors = np.where(conflicts[chosen])[0]
        to_remove = np.unique(np.concatenate(([chosen], neighbors)))
        
        # Update alive
        alive[to_remove] = False
        
        # Update degrees and clique coverage for remaining alive nodes
        remaining_alive = np.where(alive)[0]
        if len(remaining_alive) > 0:
            removed_conflicts = conflicts[remaining_alive][:, to_remove].sum(axis=1)
            degrees[remaining_alive] -= removed_conflicts
            
            # Update clique coverage
            removed_high = to_remove[high_deg_neighbors[to_remove]]
            if len(removed_high) > 0:
                removed_high_conflicts = conflicts[remaining_alive][:, removed_high].sum(axis=1)
                clique_coverage[remaining_alive] -= removed_high_conflicts

    return selected_indices

def local_improve(selected_indices, conflicts, vectors, deadline):
    """
    Try to add more vectors or swap to improve the solution.
    For each unselected vector, check if it can be added.
    Also try swapping: remove one selected, add two unselected.
    """
    N = len(conflicts)
    selected_set = set(selected_indices)
    unselected = [i for i in range(N) if i not in selected_set]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try adding vectors
        for i in unselected[:]:
            if time.time() > deadline:
                break
            if i in selected_set:
                continue
            # Check if i conflicts with any selected
            if not np.any(conflicts[i, list(selected_set)]):
                selected_set.add(i)
                unselected.remove(i)
                improved = True
        
        # Try swaps: remove one selected, add one or more unselected
        if not improved:
            for s in list(selected_set):
                if time.time() > deadline:
                    break
                # Temporarily remove s
                selected_set.remove(s)
                added = 0
                for i in unselected[:]:
                    if time.time() > deadline:
                        break
                    if i in selected_set:
                        continue
                    if not np.any(conflicts[i, list(selected_set)]):
                        selected_set.add(i)
                        unselected.remove(i)
                        added += 1
                if added > 1:
                    improved = True
                else:
                    # Put s back
                    selected_set.add(s)
        
        if not improved:
            # Try pairwise swaps
            for s1 in list(selected_set):
                if time.time() > deadline:
                    break
                for s2 in list(selected_set):
                    if s1 == s2 or time.time() > deadline:
                        continue
                    # Try removing s1 and s2, adding two unselected
                    selected_set.discard(s1)
                    selected_set.discard(s2)
                    added = 0
                    for i in unselected[:]:
                        if time.time() > deadline:
                            break
                        if i in selected_set:
                            continue
                        if not np.any(conflicts[i, list(selected_set)]):
                            selected_set.add(i)
                            unselected.remove(i)
                            added += 1
                            if added >= 2:
                                break
                    if added >= 2:
                        improved = True
                    else:
                        # Put back
                        selected_set.add(s1)
                        selected_set.add(s2)
                    if improved:
                        break
                if improved:
                    break
    
    return list(selected_set)

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Generate candidates
    vectors = generate_weight_4_vectors()
    N = len(vectors)
    
    if N == 0:
        return np.empty((0, D), dtype=np.int64)
        
    # Compute dot products
    dots = vectors @ vectors.T
    
    # Conflict if dot > 2
    conflicts = dots > 2
    
    # Run multiple random restarts with hypergraph-aware greedy
    best_selected = []
    num_restarts = 200
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10:  # Leave time for improvement
            break
        
        seed = restart
        selected = hypergraph_greedy_mis(conflicts, deadline - 10, seed=seed)
        
        if len(selected) > len(best_selected):
            best_selected = selected
            # Try local improvement on the best so far
            if time.time() < deadline - 5:
                best_selected = local_improve(best_selected, conflicts, vectors, deadline - 5)
    
    if not best_selected:
        # Fallback: return empty or a single vector
        return np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)
        
    result_vectors = vectors[best_selected]
    
    return result_vectors.astype(np.int64)
