import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin

    # Generate candidates: weights 1, 2, 3, 4
    candidates = []
    # Weight 1
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = sign
            candidates.append(v)
    # Weight 2
    for positions in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates.append(v)
    # Weight 3
    for positions in itertools.combinations(range(D), 3):
        for signs in itertools.product([1, -1], repeat=3):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates.append(v)
    # Weight 4
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates.append(v)

    V = np.array(candidates, dtype=np.int8)
    N = len(V)
    
    # Compute norms squared
    norms_sq = np.sum(V**2, axis=1)
    max_norm_sq = np.max(norms_sq) # Should be 4

    # Compute dot products
    # To save memory, we can compute block by block or just use full matrix if N is small enough
    # N is approx 22 + 220 + 1320 + 5280 = 6842
    # 6842 x 6842 matrix of int8 is ~46 MB. Acceptable.
    dots = V @ V.T
    
    # Conflict condition: ||u - v||^2 < max_norm_sq
    # ||u||^2 + ||v||^2 - 2*dot < max_norm_sq
    # 2*dot > ||u||^2 + ||v||^2 - max_norm_sq
    # dot > (||u||^2 + ||v||^2 - max_norm_sq) / 2
    
    # Precompute thresholds for each pair?
    # threshold[i, j] = (norms_sq[i] + norms_sq[j] - max_norm_sq) / 2
    # Conflict if dots[i, j] > threshold[i, j]
    
    # Since dots are integers, we can use:
    # 2 * dots > norms_sq[:, None] + norms_sq[None, :] - max_norm_sq
    # 2 * dots - norms_sq[:, None] - norms_sq[None, :] > -max_norm_sq
    # This is vectorized.
    
    lhs = 2 * dots
    rhs = norms_sq[:, None] + norms_sq[None, :] - max_norm_sq
    conflicts = lhs > rhs
    
    # Ensure diagonal is False (no self conflict)
    np.fill_diagonal(conflicts, False)
    
    # Store conflicts as a list of neighbors for faster traversal in greedy
    # Or keep as boolean matrix for vectorized operations in local search
    # For N=6800, boolean matrix is fine.
    
    # Greedy MIS with Local Search
    rng = np.random.default_rng(42)
    best_selected = np.zeros(N, dtype=bool)
    best_count = 0
    
    def greedy_mis(conflicts, rng_seed):
        r = np.random.default_rng(rng_seed)
        alive = np.ones(N, dtype=bool)
        selected = np.zeros(N, dtype=bool)
        # Initialize degrees
        degrees = conflicts.sum(axis=1).astype(np.int32)
        
        while np.any(alive):
            if time.time() > deadline - 5:
                break
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            # Find min degree among alive
            alive_degrees = degrees[alive_indices]
            min_deg = np.min(alive_degrees)
            cands = alive_indices[alive_degrees == min_deg]
            
            # Tie breaking: random
            chosen = cands[r.integers(len(cands))]
            
            selected[chosen] = True
            # Remove chosen and its neighbors from alive
            to_remove = np.where(conflicts[chosen] | (np.arange(N) == chosen))[0]
            alive[to_remove] = False
            
            # Update degrees for remaining alive nodes
            remaining = np.where(alive)[0]
            if len(remaining) > 0:
                # degrees[remaining] -= conflicts[remaining][:, to_remove].sum(axis=1)
                # This is O(N * K).
                # Faster:
                degrees[remaining] -= np.sum(conflicts[remaining][:, to_remove], axis=1)
                
        return selected

    def local_search(selected, conflicts, rng_seed):
        r = np.random.default_rng(rng_seed)
        selected = selected.copy()
        count = selected.sum()
        
        for iter in range(50):
            if time.time() > deadline - 5:
                break
            improved = False
            
            # 1-swap
            non_selected = np.where(~selected)[0]
            if len(non_selected) > 0:
                # Try random non-selected nodes
                # Or try all? N is large. Try random subset.
                sample_size = min(100, len(non_selected))
                samples = r.choice(non_selected, size=sample_size, replace=False)
                for v in samples:
                    if selected[v]: continue
                    sel_neighbors = np.where(selected & conflicts[v])[0]
                    net_gain = 1 - len(sel_neighbors)
                    if net_gain > 0:
                        selected[v] = True
                        for sn in sel_neighbors:
                            selected[sn] = False
                        count += net_gain
                        improved = True
                        break
            
            if not improved:
                # 2-swap
                if len(non_selected) >= 2:
                    sample_size = min(20, len(non_selected)//2)
                    for _ in range(sample_size):
                        i, j = r.choice(non_selected, size=2, replace=False)
                        if selected[i] or selected[j]: continue
                        if conflicts[i, j]: continue
                        
                        sel_neigh_i = np.where(selected & conflicts[i])[0]
                        sel_neigh_j = np.where(selected & conflicts[j])[0]
                        combined = np.unique(np.concatenate([sel_neigh_i, sel_neigh_j]))
                        net_gain = 2 - len(combined)
                        if net_gain > 0:
                            selected[i] = True
                            selected[j] = True
                            for sn in combined:
                                selected[sn] = False
                            count += net_gain
                            improved = True
                            break
            
            if not improved:
                break
                
        return selected, count

    # Run multiple restarts
    num_restarts = 20
    for i in range(num_restarts):
        if time.time() > deadline - 10:
            break
        seed = 42 + i * 13
        sel = greedy_mis(conflicts, seed)
        sel, cnt = local_search(sel, conflicts, seed)
        
        if cnt > best_count:
            best_count = cnt
            best_selected = sel

    # Extract result
    result_vectors = V[best_selected]
    
    # Final Validation
    if len(result_vectors) > 1:
        # Check validity
        norms_sq_res = np.sum(result_vectors**2, axis=1)
        max_norm_sq_res = np.max(norms_sq_res)
        diff = result_vectors[:, None, :] - result_vectors[None, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        conflicts_res = dists_sq < max_norm_sq_res
        np.fill_diagonal(conflicts_res, False)
        
        if np.any(conflicts_res):
            # Repair: remove conflicting vectors
            valid = np.ones(len(result_vectors), dtype=bool)
            changed = True
            while changed and time.time() < deadline - 2:
                changed = False
                # Find first conflict
                i, j = np.where(conflicts_res & np.triu(np.ones_like(conflicts_res, dtype=bool), k=1))
                if len(i) == 0:
                    break
                i, j = i[0], j[0]
                # Remove j (or the one with more conflicts? just j for simplicity)
                valid[j] = False
                conflicts_res[:, j] = False
                conflicts_res[j, :] = False
                changed = True
            
            result_vectors = result_vectors[valid]

    if len(result_vectors) == 0:
        return np.array([[1] + [0]*10], dtype=np.int64)

    return result_vectors.astype(np.int64)
