"""Kissing number in dimension 11: construct nonzero integer vectors in R^11
whose minimum pairwise distance is at least the maximum vector norm.
Interface: construct() -> integer array of shape (k, d)."""

import itertools
import time
import numpy as np

D = 11
DEADLINE_S = 145.0  # hard wall-clock safety margin before the 150 s limit


def construct():
    t0 = time.perf_counter()

    def elapsed():
        return time.perf_counter() - t0

    def is_deadline():
        return elapsed() > DEADLINE_S

    # ------------------------------------------------------------------
    # 1) Fast deterministic baseline: the full "2-weight" shell of norm^2=2
    #    There are exactly C(11,2)*2^2 = 220 such vectors, and they satisfy
    #    min pairwise distance (sqrt(2) for shared coordinate) == max norm (sqrt(2)).
    # ------------------------------------------------------------------
    base = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i], v[j] = si, sj
                base.append(v)
    base_arr = np.array(base, dtype=np.int64)
    best = base_arr
    best_k = base_arr.shape[0]

    def update_best(arr):
        nonlocal best, best_k
        if arr is None or arr.shape[0] == 0:
            return
        k = arr.shape[0]
        if k > best_k:
            best = arr
            best_k = k

    # ------------------------------------------------------------------
    # 2) Generate candidate vectors from shells with norm^2 = m
    # ------------------------------------------------------------------

    def gen_weight_vectors(m, max_terms, hard_stop=30.0):
        """Generate integer vectors in R^D with L2^2 exactly = m and <= max_terms
        nonzero coordinates, values in {1,-1}."""
        if m < 1:
            return []
        out = []
        
        if m > max_terms or m > D:
            return []
        
        for positions in itertools.combinations(range(D), m):
            if is_deadline() or elapsed() > hard_stop:
                return out
            for signs in itertools.product([-1, 1], repeat=m):
                v = np.zeros(D, dtype=np.int64)
                for pos, s in zip(positions, signs):
                    v[pos] = s
                out.append(v)
        
        return out

    # ------------------------------------------------------------------
    # 3) Multi-start greedy with random ordering for fixed-norm shells
    # ------------------------------------------------------------------

    def multi_start_greedy(vectors, m, n_restarts=50):
        """Multi-start greedy for fixed-norm^2 = m shell."""
        max_ip = m // 2
        V = np.asarray(vectors, dtype=np.int64)
        n = V.shape[0]
        if n == 0:
            return np.zeros((0, D), dtype=np.int64)
        
        # Compute Gram matrix
        G = V.astype(np.int64) @ V.astype(np.int64).T
        G = np.ascontiguousarray(G)
        
        best_subset = None
        best_k_local = 0
        
        for restart in range(n_restarts):
            if is_deadline():
                break
            
            # Random ordering
            order = np.random.permutation(n)
            
            chosen = []
            chosen_set = set()
            
            # Track max inner product with chosen for each candidate
            max_ip_with_chosen = np.full(n, -999999, dtype=np.int64)
            
            for idx in order:
                idx = int(idx)
                if idx in chosen_set:
                    continue
                
                # Check if this vector can be added
                if max_ip_with_chosen[idx] <= max_ip:
                    chosen.append(idx)
                    chosen_set.add(idx)
                    # Update max_ip_with_chosen for all other candidates
                    max_ip_with_chosen = np.maximum(max_ip_with_chosen, G[idx])
                    max_ip_with_chosen[idx] = -999999
            
            if len(chosen) > best_k_local:
                best_k_local = len(chosen)
                best_subset = chosen
        
        if best_subset is None:
            return np.zeros((0, D), dtype=np.int64)
        
        return V[best_subset]

    # ------------------------------------------------------------------
    # 4) Mixed-norm multi-start greedy
    # ------------------------------------------------------------------

    def mixed_multi_start_greedy(shell_vecs, shell_norms, max_ip_func, n_restarts=30):
        """Multi-start greedy for mixed shells."""
        all_vecs = []
        all_norms = []
        for vecs, norm2 in zip(shell_vecs, shell_norms):
            if vecs is not None and len(vecs) > 0:
                all_vecs.append(vecs)
                all_norms.append(norm2)
        if not all_vecs:
            return np.zeros((0, D), dtype=np.int64)

        V = np.vstack(all_vecs)
        n = V.shape[0]
        if n == 0:
            return np.zeros((0, D), dtype=np.int64)

        norm_arr = np.concatenate([np.full(len(v), n2) for v, n2 in zip(all_vecs, all_norms)])

        # Compute Gram
        G = V.astype(np.int64) @ V.astype(np.int64).T
        G = np.ascontiguousarray(G)

        # Precompute allowed IP matrix
        unique_norms = np.unique(norm_arr)
        allowed_matrix = np.zeros((n, n), dtype=np.int64)
        for un in unique_norms:
            mask = norm_arr == un
            for un2 in unique_norms:
                mask2 = norm_arr == un2
                allowed_matrix[np.ix_(mask, mask2)] = max_ip_func(un, un2)

        best_subset = None
        best_k_local = 0
        
        for restart in range(n_restarts):
            if is_deadline():
                break
            
            order = np.random.permutation(n)
            chosen = []
            chosen_set = set()
            
            max_ip_with_chosen = np.full(n, -999999, dtype=np.int64)
            min_allowed = np.full(n, 999999, dtype=np.int64)
            
            for idx in order:
                idx = int(idx)
                if idx in chosen_set:
                    continue
                
                # Check if this vector can be added
                if max_ip_with_chosen[idx] <= min_allowed[idx]:
                    chosen.append(idx)
                    chosen_set.add(idx)
                    max_ip_with_chosen = np.maximum(max_ip_with_chosen, G[idx])
                    max_ip_with_chosen[idx] = -999999
                    min_allowed = np.minimum(min_allowed, allowed_matrix[idx])
                    min_allowed[idx] = -999999
            
            if len(chosen) > best_k_local:
                best_k_local = len(chosen)
                best_subset = chosen
        
        if best_subset is None:
            return np.zeros((0, D), dtype=np.int64)
        
        return V[best_subset]

    # ------------------------------------------------------------------
    # 5) Try various shells and combinations
    # ------------------------------------------------------------------

    # m = 3 shell
    vecs3 = []
    try:
        vecs3 = gen_weight_vectors(3, max_terms=3, hard_stop=10.0)
        if len(vecs3) > 0 and not is_deadline():
            sub3 = multi_start_greedy(vecs3, 3, n_restarts=30)
            update_best(sub3)
    except Exception:
        pass

    # m = 4 shell
    vecs4 = []
    try:
        vecs4 = gen_weight_vectors(4, max_terms=4, hard_stop=15.0)
        if len(vecs4) > 0 and not is_deadline():
            sub4 = multi_start_greedy(vecs4, 4, n_restarts=30)
            update_best(sub4)
    except Exception:
        pass

    # m = 5 shell
    vecs5 = []
    try:
        vecs5 = gen_weight_vectors(5, max_terms=5, hard_stop=20.0)
        if len(vecs5) > 0 and not is_deadline():
            sub5 = multi_start_greedy(vecs5, 5, n_restarts=20)
            update_best(sub5)
    except Exception:
        pass

    # Mixed m=2 and m=3
    def max_ip_func_m23(m1, m2):
        return int(min(m1, m2) / 2)

    try:
        if not is_deadline():
            mixed23 = mixed_multi_start_greedy([base_arr, np.array(vecs3, dtype=np.int64) if vecs3 else None], [2, 3], max_ip_func_m23, n_restarts=20)
            update_best(mixed23)
    except Exception:
        pass

    # Mixed m=2 and m=4
    def max_ip_func_m24(m1, m2):
        return int(min(m1, m2) / 2)

    try:
        if not is_deadline():
            mixed24 = mixed_multi_start_greedy([base_arr, np.array(vecs4, dtype=np.int64) if vecs4 else None], [2, 4], max_ip_func_m24, n_restarts=20)
            update_best(mixed24)
    except Exception:
        pass

    # Mixed m=2, m=3, m=4
    try:
        if not is_deadline():
            mixed234 = mixed_multi_start_greedy(
                [base_arr, np.array(vecs3, dtype=np.int64) if vecs3 else None, np.array(vecs4, dtype=np.int64) if vecs4 else None],
                [2, 3, 4],
                max_ip_func_m24,
                n_restarts=15
            )
            update_best(mixed234)
    except Exception:
        pass

    # ------------------------------------------------------------------
    # 6) Local search improvement using bitmask-based MIS
    # ------------------------------------------------------------------
    
    def local_search_with_mis(subset_indices, all_vecs, norm_arr, G, allowed_matrix, max_iters=50):
        """Use bitmask-based maximum independent set search to improve the subset."""
        V = np.asarray(all_vecs, dtype=np.int64)
        n = V.shape[0]
        
        if subset_indices is None or len(subset_indices) == 0:
            return np.zeros((0, D), dtype=np.int64)
        
        chosen_set = set(subset_indices)
        k = len(chosen_set)
        
        # Build adjacency bitmask for the induced subgraph on non-chosen nodes
        # We want to add as many nodes as possible while maintaining independence
        
        for it in range(max_iters):
            if is_deadline():
                break
            
            improved = False
            
            # Find all nodes that can be added
            addable = []
            for idx in range(n):
                if idx in chosen_set:
                    continue
                
                # Check if idx can be added
                can_add = True
                for c in chosen_set:
                    if G[idx, c] > allowed_matrix[idx, c]:
                        can_add = False
                        break
                
                if can_add:
                    addable.append(idx)
            
            if not addable:
                break
            
            # Greedily add the most compatible node
            best_add = None
            best_score = -1
            for idx in addable:
                # Score: how many other addable nodes it conflicts with (we want low)
                conflicts = 0
                for other in addable:
                    if other == idx:
                        continue
                    if G[idx, other] > allowed_matrix[idx, other]:
                        conflicts += 1
                score = -conflicts  # minimize conflicts
                if score > best_score:
                    best_score = score
                    best_add = idx
            
            if best_add is not None:
                chosen_set.add(best_add)
                k += 1
                improved = True
        
        return V[list(chosen_set)]

    # Apply local search to the best result
    try:
        if best is not None and best.shape[0] > 0 and not is_deadline():
            if elapsed() < 120.0:
                # Build combined set from all shells
                combined_vecs = []
                combined_norms = []
                
                if base_arr is not None:
                    combined_vecs.append(base_arr)
                    combined_norms.append(2)
                
                if vecs3:
                    combined_vecs.append(np.array(vecs3, dtype=np.int64))
                    combined_norms.append(3)
                
                if vecs4:
                    combined_vecs.append(np.array(vecs4, dtype=np.int64))
                    combined_norms.append(4)
                
                if vecs5:
                    combined_vecs.append(np.array(vecs5, dtype=np.int64))
                    combined_norms.append(5)
                
                if combined_vecs:
                    V_combined = np.vstack(combined_vecs)
                    n_combined = V_combined.shape[0]
                    norm_arr_combined = np.concatenate([np.full(len(v), n2) for v, n2 in zip(combined_vecs, combined_norms)])
                    
                    G_combined = V_combined.astype(np.int64) @ V_combined.astype(np.int64).T
                    G_combined = np.ascontiguousarray(G_combined)
                    
                    unique_norms = np.unique(norm_arr_combined)
                    allowed_matrix_combined = np.zeros((n_combined, n_combined), dtype=np.int64)
                    for un in unique_norms:
                        mask = norm_arr_combined == un
                        for un2 in unique_norms:
                            mask2 = norm_arr_combined == un2
                            allowed_matrix_combined[np.ix_(mask, mask2)] = max_ip_func_m24(un, un2)
                    
                    # Find indices of best in combined set
                    # Create a lookup
                    best_set = set(map(tuple, best))
                    best_indices = []
                    for i in range(n_combined):
                        if tuple(V_combined[i]) in best_set:
                            best_indices.append(i)
                    
                    if best_indices:
                        improved = local_search_with_mis(best_indices, V_combined, norm_arr_combined, G_combined, allowed_matrix_combined, max_iters=30)
                        update_best(improved)
    except Exception:
        pass

    # ------------------------------------------------------------------
    # 7) Safety: return the best found.
    # ------------------------------------------------------------------
    if best is None or best.shape[0] == 0:
        return base_arr
    return best
