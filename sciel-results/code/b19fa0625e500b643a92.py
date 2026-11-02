"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115

    # Generate all candidate vectors with norm^2 <= 4
    # W4: weight 4, entries +/-1, norm^2 = 4
    # W1: weight 1, entries +/-2, norm^2 = 4
    # W3: weight 3, entries +/-1, norm^2 = 3
    # W2: weight 2, entries +/-1, norm^2 = 2
    # W1b: weight 1, entries +/-1, norm^2 = 1 (but these are suboptimal)
    
    candidates = []
    
    # W4 vectors
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates.append(v)
    
    # W1 vectors (entries +/-2)
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            candidates.append(v)
    
    # W3 vectors
    for positions in itertools.combinations(range(D), 3):
        for signs in itertools.product([1, -1], repeat=3):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates.append(v)
    
    # W2 vectors
    for positions in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates.append(v)
    
    candidates = np.array(candidates, dtype=np.int64)
    N = len(candidates)
    
    # Compute norms squared
    norms_sq = np.sum(candidates ** 2, axis=1)
    max_norm = np.max(norms_sq)  # Should be 4
    
    # The constraint: for all pairs i,j: ||v_i - v_j||^2 >= max_norm
    # ||v_i - v_j||^2 = ||v_i||^2 + ||v_j||^2 - 2*v_i.v_j >= max_norm
    # So: v_i.v_j <= (||v_i||^2 + ||v_j||^2 - max_norm) / 2
    
    # For two W4 vectors: dot <= (4+4-4)/2 = 2
    # For W4 and W3: dot <= (4+3-4)/2 = 1.5 -> dot <= 1
    # For W4 and W2: dot <= (4+2-4)/2 = 1
    # For W4 and W1: dot <= (4+4-4)/2 = 2
    # For two W3: dot <= (3+3-4)/2 = 1
    # For W3 and W2: dot <= (3+2-4)/2 = 0.5 -> dot <= 0
    # For W3 and W1: dot <= (3+4-4)/2 = 1.5 -> dot <= 1
    # For two W2: dot <= (2+2-4)/2 = 0
    # For W2 and W1: dot <= (2+4-4)/2 = 1
    # For two W1: dot <= (4+4-4)/2 = 2
    
    # Build conflict matrix: conflict[i,j] = True if v_i.v_j > threshold(i,j)
    # threshold(i,j) = (norms_sq[i] + norms_sq[j] - max_norm) // 2 (floor for integer)
    
    # Actually, we need: 2*dot <= norms_sq[i] + norms_sq[j] - max_norm
    # dot <= (norms_sq[i] + norms_sq[j] - max_norm) / 2
    # Since dot is integer: dot <= floor((norms_sq[i] + norms_sq[j] - max_norm) / 2)
    
    # Compute dot products in batches to save memory
    # N can be large: C(11,4)*16 + 22 + C(11,3)*8 + C(11,2)*4 = 330*16 + 22 + 165*8 + 55*4 = 5280 + 22 + 1320 + 220 = 6842
    
    # Build conflict matrix
    # Process in chunks to avoid O(N^2) memory issues for large N
    chunk_size = 500
    conflict = np.zeros((N, N), dtype=bool)
    
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        # Compute dot products between candidates[start:end] and all candidates
        dots = candidates[start:end] @ candidates.T  # (chunk, N)
        thresholds = (norms_sq[start:end, None] + norms_sq[None, :] - max_norm) // 2
        # Conflict if dot > threshold
        conflict[start:end, :] = dots > thresholds
    
    # Symmetrize (should already be symmetric)
    conflict = conflict | conflict.T
    
    # Now find maximum independent set
    # Use greedy + local search
    
    rng = np.random.default_rng(42)
    neighbors = [np.where(conflict[i])[0] for i in range(N)]
    neighbor_counts = np.array([len(n) for n in neighbors], dtype=np.int32)
    
    best_set = np.zeros(N, dtype=bool)
    best_size = 0
    
    def greedy_min_degree():
        in_set = np.zeros(N, dtype=bool)
        alive = np.ones(N, dtype=bool)
        degrees = neighbor_counts.copy()
        
        while np.any(alive) and time.time() < deadline - 5:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0:
                break
            
            min_deg = np.min(degrees[alive_indices])
            cands = alive_indices[degrees[alive_indices] == min_deg]
            chosen = cands[rng.integers(len(cands))]
            
            in_set[chosen] = True
            nbrs = neighbors[chosen]
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            alive[to_remove] = False
            
            rem = np.where(alive)[0]
            if len(rem) > 0:
                degrees[rem] -= conflict[rem][:, to_remove].sum(axis=1)
        
        return in_set
    
    def local_search(in_set_init, max_time=10.0):
        in_set = in_set_init.copy()
        current_size = in_set.sum()
        
        T = 2.0
        cooling = 0.999
        min_T = 0.03
        
        count_conflicts = np.zeros(N, dtype=np.int32)
        for i in np.where(in_set)[0]:
            count_conflicts[neighbors[i]] += 1
        
        stuck_counter = 0
        local_best_size = current_size
        local_best_in_set = in_set.copy()
        sa_deadline = time.time() + max_time
        
        while time.time() < sa_deadline:
            move_type = rng.random()
            
            if move_type < 0.7:
                not_in_set = np.where(~in_set)[0]
                if len(not_in_set) == 0:
                    continue
                
                counts = count_conflicts[not_in_set].astype(float)
                weights = np.exp(-counts / max(T, 0.1))
                weights /= weights.sum()
                s = not_in_set[rng.choice(len(not_in_set), p=weights)]
                
                k = count_conflicts[s]
                delta = 1 - k
                
                if delta > 0:
                    accept = True
                elif delta == 0:
                    accept = rng.random() < 0.5
                else:
                    accept = rng.random() < np.exp(delta / T)
                    
                if accept:
                    to_remove = np.where(conflict[s] & in_set)[0]
                    if len(to_remove) > 0:
                        in_set[to_remove] = False
                        for u in to_remove:
                            count_conflicts[neighbors[u]] -= 1
                    
                    in_set[s] = True
                    current_size += delta
                    count_conflicts[neighbors[s]] += 1
            elif move_type < 0.9 and current_size > 0:
                in_set_indices = np.where(in_set)[0]
                if len(in_set_indices) == 0:
                    continue
                    
                s = in_set_indices[rng.integers(len(in_set_indices))]
                accept = rng.random() < np.exp(-1.0 / T)
                
                if accept:
                    in_set[s] = False
                    current_size -= 1
                    count_conflicts[neighbors[s]] -= 1
            else:
                in_set_indices = np.where(in_set)[0]
                not_in_set = np.where(~in_set)[0]
                if len(in_set_indices) == 0 or len(not_in_set) == 0:
                    continue
                
                s_remove = in_set_indices[rng.integers(len(in_set_indices))]
                s_add = not_in_set[rng.integers(len(not_in_set))]
                
                in_set_minus = in_set.copy()
                in_set_minus[s_remove] = False
                conflicts_after = np.sum(conflict[s_add] & in_set_minus)
                
                if conflicts_after == 0:
                    in_set[s_remove] = False
                    in_set[s_add] = True
                    count_conflicts[neighbors[s_remove]] -= 1
                    count_conflicts[neighbors[s_add]] += 1
                    # Update conflict counts for neighbors of removed and added
                    for u in neighbors[s_remove]:
                        if in_set[u]:
                            count_conflicts[u] -= 1
                    for u in neighbors[s_add]:
                        if in_set[u] and u != s_remove:
                            count_conflicts[u] += 1
            
            if current_size > local_best_size:
                local_best_size = current_size
                local_best_in_set = in_set.copy()
                stuck_counter = 0
            else:
                stuck_counter += 1
            
            if stuck_counter > 2000:
                T = max(T, 1.5)
                stuck_counter = 0
            
            T *= cooling
            if T < min_T:
                T = min_T
                
        return local_best_in_set, local_best_size
    
    n_restarts = 0
    max_restarts = 200
    
    while time.time() < deadline - 8 and n_restarts < max_restarts:
        n_restarts += 1
        
        in_set = greedy_min_degree()
        current_size = in_set.sum()
        
        if current_size > best_size:
            best_size = current_size
            best_set = in_set.copy()
        
        remaining_time = deadline - 8 - time.time()
        if remaining_time < 3:
            break
        
        sa_time = min(remaining_time * 0.5, 15.0)
        if sa_time < 2:
            sa_time = 2.0
            
        sa_result, sa_size = local_search(in_set, max_time=sa_time)
        
        if sa_size > best_size:
            best_size = sa_size
            best_set = sa_result.copy()
        
        # Polish
        if time.time() < deadline - 3:
            polish_set = best_set.copy()
            polished = True
            while polished and time.time() < deadline - 3:
                polished = False
                not_in = np.where(~polish_set)[0]
                if len(not_in) == 0:
                    break
                for s in not_in:
                    if not np.any(conflict[s] & polish_set):
                        polish_set[s] = True
                        polished = True
                        if polish_set.sum() > best_size:
                            best_size = polish_set.sum()
                            best_set = polish_set.copy()
    
    # Return the selected vectors
    result = candidates[best_set]
    
    if len(result) == 0:
        # Fallback: just return W1 vectors
        result = candidates[5280:5302]  # W1 vectors are at indices 5280-5301
    
    return result

# EVOLVE-BLOCK-END
