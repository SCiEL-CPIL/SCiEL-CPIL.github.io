import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    # 1. Generate Type B vectors (Norm^2 = 2)
    # These are always a valid configuration.
    vB_candidates = []
    for i in range(D):
        for j in range(i+1, D):
            for si in [1, -1]:
                for sj in [1, -1]:
                    v = [0] * D
                    v[i] = si
                    v[j] = sj
                    vB_candidates.append(v)
    
    vectors_B = np.array(vB_candidates, dtype=np.int64)
    k_B = len(vectors_B) # 220

    # 2. Generate Type A vectors (Norm^2 = 4)
    vA_candidates = []
    for i in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[i] = 2 * s
            vA_candidates.append(v)
    
    vectors_A = np.array(vA_candidates, dtype=np.int64)
    k_A = len(vectors_A) # 22

    # 3. Generate Type C vectors (Norm^2 = 4)
    vC_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            vC_candidates.append(v)
            
    vectors_C = np.array(vC_candidates, dtype=np.int64)
    N_C = len(vectors_C) # 5280

    # 4. Build Conflict Graph for Type C
    # Conflict if dot > 2.
    # As analyzed, this only happens if support intersection is 3 and signs match on intersection.
    # We build a sparse adjacency list or matrix.
    # N_C = 5280. Matrix is 5280x5280 ~ 27M bools. 27MB. Acceptable.
    
    # To speed up, we can just compute dots in chunks or use the structure.
    # But 5280^2 dot products is ~27M ops. NumPy can do this fast.
    
    # However, we only need to know if dot > 2.
    # Let's create the matrix.
    conflicts_C = np.zeros((N_C, N_C), dtype=bool)
    
    # Chunked computation to avoid memory spikes if needed, but 27M is fine.
    chunk_size = 500
    for i_start in range(0, N_C, chunk_size):
        i_end = min(i_start + chunk_size, N_C)
        dots = vectors_C[i_start:i_end] @ vectors_C.T
        # We only care about j > i to fill upper triangle, or just full and mask diagonal
        mask = dots > 2
        np.fill_diagonal(mask, False)
        conflicts_C[i_start:i_end, :] = mask

    # Symmetrize (just in case, though dot is symmetric)
    conflicts_C = conflicts_C | conflicts_C.T

    # 5. Find Independent Set in conflicts_C
    # Use Greedy Min-Degree then SA
    
    def greedy_min_degree(conflicts, seed=0):
        n = len(conflicts)
        rng = np.random.default_rng(seed)
        in_set = np.zeros(n, dtype=bool)
        alive = np.ones(n, dtype=bool)
        degrees = conflicts.sum(axis=1).astype(np.int32)
        
        while np.any(alive):
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0: break
            degs = degrees[alive_indices]
            min_deg = np.min(degs)
            cands = alive_indices[degs == min_deg]
            chosen = cands[rng.integers(len(cands))]
            in_set[chosen] = True
            nbrs = np.where(conflicts[chosen] & alive)[0]
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            alive[to_remove] = False
            rem = np.where(alive)[0]
            if len(rem) > 0:
                degrees[rem] -= conflicts[rem][:, to_remove].sum(axis=1)
                
        return in_set

    def sa_local_search(conflicts, init_set, max_time=5.0):
        n = len(conflicts)
        rng = np.random.default_rng(42)
        in_set = init_set.copy()
        current_size = int(in_set.sum())
        
        # Precompute neighbors for faster access
        neighbors = [np.where(conflicts[i])[0] for i in range(n)]
        
        T = 1.0
        cooling = 0.999
        min_T = 0.01
        sa_deadline = time.time() + max_time
        stuck = 0
        
        # Initialize conflict counts
        count_conflicts = np.zeros(n, dtype=np.int32)
        in_set_indices = np.where(in_set)[0]
        for i in in_set_indices:
            count_conflicts[neighbors[i]] += 1
            
        local_best_size = current_size
        local_best_in_set = in_set.copy()
        
        while time.time() < sa_deadline:
            not_in_set = np.where(~in_set)[0]
            if len(not_in_set) == 0: break
            
            if rng.random() < 0.5:
                # Add
                counts = count_conflicts[not_in_set]
                min_c = np.min(counts)
                cands = not_in_set[counts == min_c]
                v = cands[rng.integers(len(cands))]
                k = int(min_c)
                delta = 1 - k
                if delta > 0 or rng.random() < np.exp(delta/T):
                    to_remove = np.where(conflicts[v] & in_set)[0]
                    if len(to_remove) > 0:
                        in_set[to_remove] = False
                        for u in to_remove:
                            count_conflicts[neighbors[u]] -= 1
                    in_set[v] = True
                    current_size += delta
                    count_conflicts[neighbors[v]] += 1
            else:
                # Remove
                in_set_indices = np.where(in_set)[0]
                if len(in_set_indices) == 0: break
                v = in_set_indices[rng.integers(len(in_set_indices))]
                delta = -1
                if rng.random() < np.exp(delta/T):
                    in_set[v] = False
                    current_size += delta
                    count_conflicts[neighbors[v]] -= 1
                    
            if current_size > local_best_size:
                local_best_size = current_size
                local_best_in_set = in_set.copy()
                stuck = 0
            else:
                stuck += 1
            if stuck > 1000:
                T = max(T, 1.0)
                stuck = 0
            T = max(min_T, T * cooling)
            
        return local_best_in_set, local_best_size

    # Run Greedy
    best_C_set = greedy_min_degree(conflicts_C, seed=0)
    best_C_size = int(best_C_set.sum())
    
    # Run SA to improve
    if time.time() < deadline - 5:
        sa_set, sa_size = sa_local_search(conflicts_C, best_C_set, max_time=min(10.0, deadline - time.time() - 5))
        if sa_size > best_C_size:
            best_C_size = sa_size
            best_C_set = sa_set

    # Combine A and C
    if best_C_size > 0:
        final_C = vectors_C[best_C_set]
        result_AC = np.vstack([vectors_A, final_C])
        k_AC = len(result_AC)
    else:
        result_AC = vectors_A
        k_AC = k_A

    # Compare with B
    if k_AC > k_B:
        return result_AC
    else:
        return vectors_B
