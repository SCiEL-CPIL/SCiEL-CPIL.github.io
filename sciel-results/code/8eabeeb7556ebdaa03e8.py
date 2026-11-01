"""Rewritten program for Kissing Number in dimension 11."""
import numpy as np
import itertools
import time

D = 11

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
            
    vectors_v4 = np.array(v4_candidates, dtype=np.int64)
    N4 = len(vectors_v4)
    
    # Compute conflict matrix: conflict if dot > 2
    # Using matrix multiplication for speed
    dots = vectors_v4 @ vectors_v4.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Convert to neighbor lists for faster local search updates
    neighbors = [np.where(conflicts[i])[0] for i in range(N4)]
    neighbor_counts = np.array([len(n) for n in neighbors], dtype=np.int32)
    
    # Precompute neighbor arrays for faster access
    neighbor_arrays = []
    for i in range(N4):
        neighbor_arrays.append(neighbors[i])
    
    rng = np.random.default_rng(42)
    
    best_in_set = np.zeros(N4, dtype=bool)
    best_size = 0

    def greedy_min_degree():
        in_set = np.zeros(N4, dtype=bool)
        alive = np.ones(N4, dtype=bool)
        degrees = neighbor_counts.copy()
        
        while np.any(alive) and time.time() < deadline - 5:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            
            min_deg = np.min(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] == min_deg]
            chosen = cands[rng.integers(len(cands))]
            
            in_set[chosen] = True
            nbrs = neighbor_arrays[chosen]
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            alive[to_remove] = False
            
            rem = np.where(alive)[0]
            if len(rem) > 0:
                degrees[rem] -= conflicts[rem][:, to_remove].sum(axis=1)
                
        return in_set

    def greedy_fill(in_set, counts):
        """
        Add all possible nodes to in_set, updating counts.
        Returns number of nodes added.
        """
        added = 0
        # Iterative greedy fill
        improved = True
        while improved:
            improved = False
            candidates = np.where((counts == 0) & (~in_set))[0]
            for v in candidates:
                if counts[v] == 0 and not in_set[v]:
                    in_set[v] = True
                    added += 1
                    improved = True
                    for u in neighbor_arrays[v]:
                        counts[u] += 1
        return added

    def local_search(in_set):
        """
        Simulated Annealing based local search with proper conflict tracking.
        """
        in_set = in_set.copy()
        size = in_set.sum()
        
        # Calculate initial global_conflicts
        global_conflicts = 0
        in_set_indices = np.where(in_set)[0]
        for i in in_set_indices:
            global_conflicts += np.count_nonzero(in_set[neighbors[i]])
        global_conflicts //= 2  # Each conflict counted twice
        
        best_in_set = in_set.copy()
        best_size = size
        
        T = 2.0
        cooling = 0.9998
        lambda_pen = 5.0
        rng = np.random.default_rng(42)
        
        while time.time() < deadline - 10 and T > 0.05:
            v = rng.integers(N4)
            nbrs = neighbors[v]
            conflict_count = np.count_nonzero(in_set[nbrs])
            
            if in_set[v]:
                # Remove v
                delta_E = 1.0 - lambda_pen * conflict_count
                if delta_E < 0 or rng.random() < np.exp(-delta_E / T):
                    in_set[v] = False
                    size -= 1
                    global_conflicts -= conflict_count
            else:
                # Add v
                delta_E = -1.0 + lambda_pen * conflict_count
                if delta_E < 0 or rng.random() < np.exp(-delta_E / T):
                    in_set[v] = True
                    size += 1
                    global_conflicts += conflict_count
            
            if global_conflicts == 0 and size > best_size:
                best_size = size
                best_in_set = in_set.copy()
                
            T *= cooling
            
        # Final Polish: Iterative greedy fill
        improved = True
        while improved:
            improved = False
            counts = np.zeros(N4, dtype=np.int32)
            for i in np.where(best_in_set)[0]:
                counts[neighbors[i]] += 1
            
            candidates = np.where((counts == 0) & (~best_in_set))[0]
            for v in candidates:
                if counts[v] == 0 and not best_in_set[v]:
                    best_in_set[v] = True
                    improved = True
                    for u in neighbors[v]:
                        counts[u] += 1
        
        best_size = best_in_set.sum()

        return best_in_set

    # Main Loop: Multiple Restarts
    n_restarts = 0
    max_restarts = 30
    
    while time.time() < deadline - 15 and n_restarts < max_restarts:
        n_restarts += 1
        
        # 1. Greedy Init
        in_set = greedy_min_degree()
        
        # 2. Local Search
        remaining_time = deadline - 15 - time.time()
        if remaining_time < 3:
            break
            
        # Allocate time to local search
        # Local search is the main driver
        in_set = local_search(in_set)
        
        current_size = in_set.sum()
        if current_size > best_size:
            best_size = current_size
            best_in_set = in_set.copy()
        
        # 3. Try multiple perturbations of the best set found so far
        if n_restarts > 1 and time.time() < deadline - 10:
            for _ in range(3):
                if time.time() > deadline - 10:
                    break
                perturbed = best_in_set.copy()
                # Remove 3-5 random elements
                indices = np.where(perturbed)[0]
                if len(indices) > 5:
                    n_remove = rng.integers(3, 6)
                    to_remove = rng.choice(indices, size=n_remove, replace=False)
                    perturbed[to_remove] = False
                perturbed = local_search(perturbed)
                current_size = perturbed.sum()
                if current_size > best_size:
                    best_size = current_size
                    best_in_set = perturbed.copy()

    # Final result
    final_v4_indices = np.where(best_in_set)[0]
    result_v4 = vectors_v4[final_v4_indices]
    
    # Add W1 vectors (norm 2, weight 1) and W2 vectors (norm 2, weight 2)
    # W1 vectors: (2,0,0...), (-2,0,0...), etc.
    # W2 vectors: (1,1,0...), (1,-1,0...), etc.
    
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)
    
    v2_candidates = []
    for positions in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v2_candidates.append(v)
    vectors_v2 = np.array(v2_candidates, dtype=np.int64)
    
    # Combine all vectors
    all_vectors = np.vstack([result_v4, vectors_v1, vectors_v2])
    
    # Verify and filter: ensure min pairwise distance >= max norm
    n_total = len(all_vectors)
    if n_total > 0:
        norms = np.linalg.norm(all_vectors, axis=1)
        max_norm = np.max(norms)
        
        # Compute pairwise distances
        dots = all_vectors @ all_vectors.T
        dists_sq = norms[:, None]**2 + norms[None, :]**2 - 2 * dots
        np.fill_diagonal(dists_sq, np.inf)
        
        # Find valid subset: keep vectors where all pairwise distances >= max_norm
        # This is complex, so we'll do a greedy filter
        keep = np.ones(n_total, dtype=bool)
        
        # Sort by norm descending (larger norms are more restrictive)
        order = np.argsort(-norms)
        
        for i in order:
            if not keep[i]:
                continue
            # Check against all kept vectors
            for j in order:
                if j == i or not keep[j]:
                    continue
                if dists_sq[i, j] < max_norm**2 - 1e-9:
                    keep[i] = False
                    break
        
        final_vectors = all_vectors[keep]
    else:
        final_vectors = all_vectors
    
    return final_vectors

# EVOLVE-BLOCK-END
