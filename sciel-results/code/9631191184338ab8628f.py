"""Kissing number in dimension 11 - Combined Spectral Init + SA Local Search."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin

    # Generate candidates: norm squared 4
    # Type A: weight 4, entries in {-1, 1}
    # Type B: weight 1, entries in {-2, 2}
    
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
            
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)

    all_candidates = np.array(v4_candidates + v1_candidates, dtype=np.int64)
    N = len(all_candidates)

    if N == 0:
        return np.array([], dtype=np.int64)

    # Compute conflict matrix
    # Conflict if dot > 2
    dots = all_candidates @ all_candidates.T
    conflicts = dots > 2
    np.fill_diagonal(conflicts, False)
    
    # Convert to list of lists for faster iteration
    neighbors = [np.where(conflicts[i])[0] for i in range(N)]
    degrees = np.array([len(n) for n in neighbors], dtype=np.int32)

    rng = np.random.default_rng(42)

    def spectral_init():
        """
        Spectral initialization:
        1. Compute principal eigenvector of conflict adjacency matrix.
        2. Rank vertices by absolute component value.
        3. Greedily select vertices in this order, skipping conflicts.
        """
        # Use numpy's eigh for symmetric matrix
        # conflicts is boolean, convert to float for eigh
        A = conflicts.astype(np.float64)
        
        # For large N, full eigh might be slow, but N is ~3330 here.
        # Let's try to be efficient.
        try:
            # We only need the top few eigenvectors. 
            # numpy.linalg.eigh returns all. 
            # scipy.sparse.linalg.eigsh would be better but let's stick to numpy if possible 
            # or use scipy if available. The prompt says SciPy is available.
            from scipy.sparse.linalg import eigsh
            import scipy.sparse as sp
            
            # Convert to sparse for eigsh
            A_sparse = sp.csr_matrix(A)
            # Find top 20 eigenvectors
            # k=20, which='LM' (largest magnitude)
            vals, vecs = eigsh(A_sparse, k=20, which='LM', return_eigenvectors=True)
            
            # Combine top eigenvectors to get a robust ranking
            # Weight by eigenvalue magnitude? Or just take the top one?
            # The principal eigenvector (largest eigenvalue) is the standard choice.
            # Let's take the eigenvector corresponding to the largest eigenvalue.
            idx_max = np.argmax(vals)
            v_main = vecs[:, idx_max]
            
            # Abs values for ranking
            scores = np.abs(v_main)
            
        except Exception:
            # Fallback to numpy if scipy fails or is too slow/unavailable
            # For N=3330, numpy.linalg.eigh takes ~1-2s. Acceptable.
            vals, vecs = np.linalg.eigh(A)
            # Largest eigenvalue is the last one
            v_main = vecs[:, -1]
            scores = np.abs(v_main)
        
        # Rank vertices by score
        order = np.argsort(-scores)
        
        in_set = np.zeros(N, dtype=bool)
        count = 0
        for v in order:
            if not in_set[v] and not np.any(in_set[neighbors[v]]):
                in_set[v] = True
                count += 1
            if time.time() > deadline - 10:
                break
                
        return in_set

    def greedy_init():
        """Greedy min-degree initialization."""
        in_set = np.zeros(N, dtype=bool)
        alive = np.ones(N, dtype=bool)
        deg = degrees.copy()
        
        while np.any(alive) and time.time() < deadline - 5:
            alive_indices = np.where(alive)[0]
            if len(alive_indices) == 0: break
            min_deg = np.min(deg[alive_indices])
            cands = alive_indices[deg[alive_indices] == min_deg]
            chosen = cands[rng.integers(len(cands))]
            in_set[chosen] = True
            nbrs = neighbors[chosen]
            to_remove = np.unique(np.concatenate(([chosen], nbrs)))
            alive[to_remove] = False
            rem = np.where(alive)[0]
            if len(rem) > 0:
                deg[rem] -= conflicts[rem][:, to_remove].sum(axis=1)
                
        return in_set

    def local_search_sa(in_set, max_time=10.0):
        """Simulated Annealing Local Search with improved moves."""
        in_set = in_set.copy()
        current_size = in_set.sum()
        best_size = current_size
        best_in_set = in_set.copy()
        
        # conflict_count[i]: number of neighbors of i that are in in_set
        conflict_count = np.zeros(N, dtype=np.int32)
        for i in np.where(in_set)[0]:
            conflict_count[neighbors[i]] += 1
            
        ls_deadline = time.time() + max_time
        T = 8.0
        cooling = 0.999
        min_T = 0.05
        stuck = 0
        
        while time.time() < ls_deadline:
            improved = False
            
            # 1. Try to add a vertex with 0 conflicts (Greedy improvement)
            not_in_set = np.where(~in_set)[0]
            if len(not_in_set) > 0:
                counts = conflict_count[not_in_set]
                zero_conf = not_in_set[counts == 0]
                if len(zero_conf) > 0:
                    v = zero_conf[rng.integers(len(zero_conf))]
                    in_set[v] = True
                    current_size += 1
                    conflict_count[neighbors[v]] += 1
                    improved = True
                    if current_size > best_size:
                        best_size = current_size
                        best_in_set = in_set.copy()
                    stuck = 0
                    continue
            
            # 2. SA Moves
            if len(not_in_set) > 0 and np.any(in_set):
                in_set_indices = np.where(in_set)[0]
                move_type = rng.random()
                
                if move_type < 0.5:
                    # ADD move (with potential removals)
                    counts = conflict_count[not_in_set]
                    min_count = np.min(counts)
                    min_cands = not_in_set[counts == min_count]
                    v = min_cands[rng.integers(len(min_cands))]
                    
                    k = min_count
                    delta = 1 - k
                    
                    if delta > 0 or (rng.random() < np.exp(delta / T)):
                        to_remove = np.where(conflicts[v] & in_set)[0]
                        if len(to_remove) > 0:
                            in_set[to_remove] = False
                            for u in to_remove:
                                conflict_count[neighbors[u]] -= 1
                        in_set[v] = True
                        current_size += delta
                        conflict_count[neighbors[v]] += 1
                        
                        if current_size > best_size:
                            best_size = current_size
                            best_in_set = in_set.copy()
                            stuck = 0
                        else:
                            stuck += 1
                        improved = True
                            
                elif move_type < 0.7:
                    # SWAP move: remove one, add one
                    if len(in_set_indices) > 0:
                        v_remove = in_set_indices[rng.integers(len(in_set_indices))]
                        v_add = not_in_set[rng.integers(len(not_in_set))]
                        
                        conflicts_with = np.where(conflicts[v_add] & in_set)[0]
                        other_conflicts = conflicts_with[conflicts_with != v_remove]
                        
                        if len(other_conflicts) == 0:
                            # Clean swap
                            in_set[v_remove] = False
                            in_set[v_add] = True
                            conflict_count[neighbors[v_remove]] -= 1
                            conflict_count[neighbors[v_add]] += 1
                            stuck += 1
                            improved = True
                        else:
                            # Complex swap
                            to_remove = np.unique(np.concatenate(([v_remove], other_conflicts)))
                            delta = 1 - len(to_remove)
                            if delta > 0 or (rng.random() < np.exp(delta / T)):
                                in_set[to_remove] = False
                                for u in to_remove:
                                    conflict_count[neighbors[u]] -= 1
                                in_set[v_add] = True
                                current_size += delta
                                conflict_count[neighbors[v_add]] += 1
                                
                                if current_size > best_size:
                                    best_size = current_size
                                    best_in_set = in_set.copy()
                                    stuck = 0
                                else:
                                    stuck += 1
                                improved = True

                elif move_type < 0.8:
                    # REMOVE move
                    if len(in_set_indices) > 0:
                        v = in_set_indices[rng.integers(len(in_set_indices))]
                        delta = -1
                        if rng.random() < np.exp(delta / T):
                            in_set[v] = False
                            current_size += delta
                            conflict_count[neighbors[v]] -= 1
                            stuck += 1
                            improved = True

                elif move_type < 0.9:
                    # 2-for-1 swap: remove 1, add 2 non-conflicting vertices
                    if stuck > 5:
                         counts = conflict_count[not_in_set]
                         one_conf = not_in_set[counts == 1]
                         if len(one_conf) > 0:
                             groups = {}
                             for v in one_conf:
                                 nbrs_in_set = np.where(conflicts[v] & in_set)[0]
                                 if len(nbrs_in_set) == 1:
                                     u = nbrs_in_set[0]
                                     if u not in groups:
                                         groups[u] = []
                                     groups[u].append(v)
                             
                             swap_found = False
                             for u, vs in groups.items():
                                 if len(vs) >= 2:
                                     for i in range(len(vs)):
                                         for j in range(i+1, len(vs)):
                                             v1, v2 = vs[i], vs[j]
                                             if not conflicts[v1, v2]:
                                                 in_set[u] = False
                                                 in_set[v1] = True
                                                 in_set[v2] = True
                                                 
                                                 conflict_count[neighbors[u]] -= 1
                                                 conflict_count[neighbors[v1]] += 1
                                                 conflict_count[neighbors[v2]] += 1
                                                 
                                                 current_size += 1
                                                 improved = True
                                                 swap_found = True
                                                 
                                                 if current_size > best_size:
                                                     best_size = current_size
                                                     best_in_set = in_set.copy()
                                                 break
                                         if swap_found:
                                             break
                                     if swap_found:
                                         break
                             if swap_found:
                                 stuck = 0
                                 continue

                else:
                    # 3-for-2 swap: remove 2, add 3
                    if stuck > 15:
                        counts = conflict_count[not_in_set]
                        two_conf = not_in_set[(counts == 2)]
                        if len(two_conf) > 2:
                            # Try to find 3 vertices that together conflict with only 2 in-set vertices
                            sample_size = min(50, len(two_conf))
                            sample = two_conf[rng.choice(len(two_conf), size=sample_size, replace=False)]
                            found = False
                            for i in range(len(sample)):
                                if found:
                                    break
                                v1 = sample[i]
                                c1 = np.where(conflicts[v1] & in_set)[0]
                                if len(c1) != 2:
                                    continue
                                for j in range(i+1, len(sample)):
                                    if found:
                                        break
                                    v2 = sample[j]
                                    if conflicts[v1, v2]:
                                        continue
                                    c2 = np.where(conflicts[v2] & in_set)[0]
                                    if len(c2) != 2:
                                        continue
                                    # Check if c1 and c2 share exactly 1 common vertex
                                    common = np.intersect1d(c1, c2)
                                    if len(common) != 1:
                                        continue
                                    for k in range(j+1, len(sample)):
                                        if found:
                                            break
                                        v3 = sample[k]
                                        if conflicts[v1, v3] or conflicts[v2, v3]:
                                            continue
                                        c3 = np.where(conflicts[v3] & in_set)[0]
                                        if len(c3) != 2:
                                            continue
                                        # All three should conflict with the same 2 vertices
                                        if not np.array_equal(np.sort(c1), np.sort(c3)):
                                            continue
                                        # Remove the 2 conflicting vertices, add 3
                                        to_remove = np.unique(np.concatenate((c1, c2, c3)))
                                        if len(to_remove) == 2:
                                            in_set[to_remove] = False
                                            for u in to_remove:
                                                conflict_count[neighbors[u]] -= 1
                                            in_set[v1] = True
                                            in_set[v2] = True
                                            in_set[v3] = True
                                            conflict_count[neighbors[v1]] += 1
                                            conflict_count[neighbors[v2]] += 1
                                            conflict_count[neighbors[v3]] += 1
                                            current_size += 1
                                            improved = True
                                            found = True
                                            if current_size > best_size:
                                                best_size = current_size
                                                best_in_set = in_set.copy()
                                            break
                            if found:
                                stuck = 0
                                continue

            if not improved:
                stuck += 1
                if stuck > 15:
                    # Random perturbation: remove 1-2 vertices
                    in_set_indices = np.where(in_set)[0]
                    if len(in_set_indices) > 0:
                        n_remove = 1 if rng.random() < 0.5 else 2
                        n_remove = min(n_remove, len(in_set_indices))
                        to_remove = in_set_indices[rng.choice(len(in_set_indices), size=n_remove, replace=False)]
                        for u in to_remove:
                            in_set[u] = False
                            current_size -= 1
                            conflict_count[neighbors[u]] -= 1
                        stuck = 0
            
            T *= cooling
            if T < min_T:
                T = min_T

        return best_in_set, best_size

    # Run multiple restarts
    best_size = 0
    best_in_set = np.zeros(N, dtype=bool)
    
    # 1. Spectral Init
    in_set = spectral_init()
    size = in_set.sum()
    if size > best_size:
        best_size = size
        best_in_set = in_set.copy()
        
    # Local Search on spectral init - give it more time
    remaining_time = deadline - 10 - time.time()
    if remaining_time > 2:
        ls_time = min(remaining_time * 0.5, 20.0)
        ls_result, ls_size = local_search_sa(best_in_set, max_time=ls_time)
        if ls_size > best_size:
            best_size = ls_size
            best_in_set = ls_result.copy()
            
    # 2. Greedy Init (multiple restarts)
    n_restarts = 0
    while time.time() < deadline - 5 and n_restarts < 30:
        n_restarts += 1
        in_set = greedy_init()
        size = in_set.sum()
        if size > best_size:
            best_size = size
            best_in_set = in_set.copy()
            
        remaining_time = deadline - 5 - time.time()
        if remaining_time < 2:
            break
            
        ls_time = min(remaining_time * 0.4, 10.0)
        ls_result, ls_size = local_search_sa(in_set, max_time=ls_time)
        if ls_size > best_size:
            best_size = ls_size
            best_in_set = ls_result.copy()

    # Final polish: try to add any remaining vertices with 0 conflicts
    if time.time() < deadline - 2:
        conflict_count = np.zeros(N, dtype=np.int32)
        for i in np.where(best_in_set)[0]:
            conflict_count[neighbors[i]] += 1
        
        not_in_set = np.where(~best_in_set)[0]
        if len(not_in_set) > 0:
            counts = conflict_count[not_in_set]
            zero_conf = not_in_set[counts == 0]
            for v in zero_conf:
                best_in_set[v] = True
                conflict_count[neighbors[v]] += 1
                remaining = np.where(~best_in_set)[0]
                if len(remaining) > 0:
                    conflict_count[remaining] += conflicts[remaining, v].astype(np.int32)
    
    # Second polish pass: try 2-for-1 swaps one more time
    if time.time() < deadline - 1:
        conflict_count = np.zeros(N, dtype=np.int32)
        for i in np.where(best_in_set)[0]:
            conflict_count[neighbors[i]] += 1
        
        not_in_set = np.where(~best_in_set)[0]
        if len(not_in_set) > 0:
            counts = conflict_count[not_in_set]
            one_conf = not_in_set[counts == 1]
            if len(one_conf) > 0:
                groups = {}
                for v in one_conf:
                    nbrs_in_set = np.where(conflicts[v] & best_in_set)[0]
                    if len(nbrs_in_set) == 1:
                        u = nbrs_in_set[0]
                        if u not in groups:
                            groups[u] = []
                        groups[u].append(v)
                
                for u, vs in groups.items():
                    if len(vs) >= 2:
                        for i in range(len(vs)):
                            for j in range(i+1, len(vs)):
                                v1, v2 = vs[i], vs[j]
                                if not conflicts[v1, v2]:
                                    best_in_set[u] = False
                                    best_in_set[v1] = True
                                    best_in_set[v2] = True
                                    break
                            else:
                                continue
                            break

    final_indices = np.where(best_in_set)[0]
    result_vectors = all_candidates[final_indices]
    return result_vectors

# EVOLVE-BLOCK-END
