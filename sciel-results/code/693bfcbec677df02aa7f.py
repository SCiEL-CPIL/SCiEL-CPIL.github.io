import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 120s limit
    
    # Generate all weight-4 vectors in {-1, 0, 1}^11 (norm^2 = 4)
    candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates.append(v)
    
    vectors = np.array(candidates, dtype=np.int64)
    N = len(vectors)
    
    if N == 0:
        return np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)
    
    # Compute norms
    norms2 = (vectors ** 2).sum(axis=1)
    max_norm2 = norms2.max()
    
    # Compute dot products
    dots = vectors @ vectors.T
    
    # dist^2(i,j) = norms2[i] + norms2[j] - 2*dots[i,j]
    dist2 = norms2[:, None] + norms2[None, :] - 2 * dots
    np.fill_diagonal(dist2, 0)
    
    min_dist2 = dist2.min()
    
    if min_dist2 >= max_norm2:
        # All vectors are valid together
        return vectors
    
    # Conflict matrix: dist2[i,j] < max_norm2
    conflicts = dist2 < max_norm2
    np.fill_diagonal(conflicts, False)
    
    # Build sparse neighbor list
    neighbors_list = []
    for i in range(N):
        neighbors_list.append(np.where(conflicts[i])[0])
        
    # Free memory
    del conflicts
    del dist2
    del dots
    del norms2
    
    def greedy_mis_sparse(neighbors_list, N, seed, deadline):
        alive = np.ones(N, dtype=bool)
        selected = []
        degrees = np.array([len(neighbors_list[i]) for i in range(N)], dtype=np.int32)
        rng = np.random.default_rng(seed)
        
        while np.any(alive) and time.time() < deadline:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            min_deg = np.min(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] == min_deg]
            chosen = cands[rng.integers(len(cands))]
            selected.append(chosen)
            
            # Update degrees and alive status for neighbors
            for n in neighbors_list[chosen]:
                if alive[n]:
                    degrees[n] -= 1
                    alive[n] = False
            alive[chosen] = False
            
        return selected
    
    def local_improve_sparse(selected, neighbors_list, N, deadline, rng):
        """Try to add more vectors by checking if they conflict with any selected."""
        selected_set = set(selected)
        improved = list(selected)
        
        # Get indices of unselected vectors
        unselected = np.setdiff1d(np.arange(N), np.array(selected))
        
        # Randomize order
        rng.shuffle(unselected)
        
        for v in unselected:
            if time.time() > deadline:
                break
            # Check if v conflicts with any selected vector
            # v conflicts if any neighbor of v is in selected_set
            has_conflict = False
            for n in neighbors_list[v]:
                if n in selected_set:
                    has_conflict = True
                    break
            if not has_conflict:
                improved.append(v)
                selected_set.add(v)
        
        return improved
    
    best_selected = None
    best_len = 0
    
    # Try multiple seeds
    num_seeds = 100
    for seed in range(num_seeds):
        if time.time() > deadline - 10:
            break
        selected = greedy_mis_sparse(neighbors_list, N, seed, deadline - 10)
        # Local improvement
        rng_local = np.random.default_rng(seed + 1000)
        selected = local_improve_sparse(selected, neighbors_list, N, deadline - 10, rng_local)
        
        if len(selected) > best_len:
            best_len = len(selected)
            best_selected = selected
            
        # If we found a very large set, maybe stop early?
        # Theoretical max is 554.
        if best_len >= 554:
            break

    if best_selected is None:
        best_selected = list(range(N))  # fallback
    
    result_vectors = vectors[best_selected]
    return result_vectors
