import time
import numpy as np

DIMENSIONS = [2, 2, 9]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=9
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin

    def verify_decomp(U, V, W, n=2, m=2, p=9):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        R = U_arr.shape[1]
        U_r = U_arr.reshape(n, m, R)
        V_r = V_arr.reshape(m, p, R)
        W_r = W_arr.reshape(p, n, R)
        T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        expected = np.ones((n, m, p))
        return np.allclose(T, expected, atol=1e-8)

    def build_strassen_rank32():
        terms = []
        for block_idx in range(4):
            k0 = 2 * block_idx
            k1 = k0 + 1
            terms.append(({0: 1, 3: 1}, {k0: 1, 9+k1: 1}, {2*k0: 1, 2*k1+1: 1}))
            terms.append(({2: 1, 3: 1}, {k0: 1}, {2*k0+1: 1, 2*k1+1: -1}))
            terms.append(({0: 1}, {k1: 1, 9+k1: -1}, {2*k1: 1, 2*k1+1: 1}))
            terms.append(({3: 1}, {9+k0: 1, k0: -1}, {2*k0: 1, 2*k0+1: 1}))
            terms.append(({0: 1, 1: 1}, {9+k1: 1}, {2*k0: -1, 2*k1: 1}))
            terms.append(({2: 1, 0: -1}, {k0: 1, k1: 1}, {2*k1+1: 1}))
            terms.append(({1: 1, 3: -1}, {9+k0: 1, 9+k1: 1}, {2*k0: 1}))
        k = 8
        terms.append(({0: 1}, {k: 1}, {2*k: 1}))
        terms.append(({1: 1}, {9+k: 1}, {2*k: 1}))
        terms.append(({2: 1}, {k: 1}, {2*k+1: 1}))
        terms.append(({3: 1}, {9+k: 1}, {2*k+1: 1}))
        return terms

    def terms_to_matrices(terms, n=2, m=2, p=9):
        rank = len(terms)
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for t, (Ud, Vd, Wd) in enumerate(terms):
            for idx, val in Ud.items():
                U[idx][t] = val
            for idx, val in Vd.items():
                V[idx][t] = val
            for idx, val in Wd.items():
                W[idx][t] = val
        return U, V, W

    def term_tensor(Ud, Vd, Wd, n=2, m=2, p=9):
        t = np.zeros((n, m, p))
        for i_u, u_val in Ud.items():
            for j_v, v_val in Vd.items():
                for k_w, w_val in Wd.items():
                    i, j = divmod(i_u, m)
                    j2, k = divmod(j_v, p)
                    k2, i2 = divmod(k_w, n)
                    if j == j2 and k == k2 and i == i2:
                        t[i, j, k] += u_val * v_val * w_val
        return t

    def try_reduce_rank(terms, max_pairs=6):
        n, m, p = 2, 2, 9
        current_terms = list(terms)
        rank = len(current_terms)
        target = np.ones((n, m, p))

        def compute_full_tensor(term_list):
            R = len(term_list)
            U = np.zeros((n*m, R))
            V = np.zeros((m*p, R))
            W = np.zeros((p*n, R))
            for t, (Ud, Vd, Wd) in enumerate(term_list):
                for idx, val in Ud.items():
                    U[idx, t] = val
                for idx, val in Vd.items():
                    V[idx, t] = val
                for idx, val in Wd.items():
                    W[idx, t] = val
            U_r = U.reshape(n, m, R)
            V_r = V.reshape(m, p, R)
            W_r = W.reshape(p, n, R)
            return np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)

        term_tensors = [term_tensor(Ud, Vd, Wd) for Ud, Vd, Wd in current_terms]

        # Single-term removal
        for remove_idx in range(rank):
            if time.time() > deadline - 30:
                break
            remaining = [t for t in range(rank) if t != remove_idx]
            if len(remaining) == 0:
                continue
            removed_flat = term_tensors[remove_idx].flatten()
            remaining_matrix = np.array([term_tensors[r].flatten() for r in remaining])
            try:
                x, res, rk, sv = np.linalg.lstsq(remaining_matrix, removed_flat, rcond=1e-12)
                recon = remaining_matrix @ x
                if np.allclose(recon, removed_flat, atol=1e-10):
                    new_terms = [current_terms[t] for t in remaining]
                    U, V, W = terms_to_matrices(new_terms)
                    if verify_decomp(U, V, W):
                        current_terms = new_terms
                        rank = len(current_terms)
                        term_tensors = [term_tensor(Ud, Vd, Wd) for Ud, Vd, Wd in current_terms]
                        break
            except:
                continue

        # Pair-term removal
        rank = len(current_terms)
        term_tensors = [term_tensor(Ud, Vd, Wd) for Ud, Vd, Wd in current_terms]
        pair_found = False
        for i in range(rank):
            if time.time() > deadline - 20:
                break
            for j in range(i + 1, rank):
                if time.time() > deadline - 20:
                    break
                remaining = [t for t in range(rank) if t != i and t != j]
                if len(remaining) < 2:
                    continue
                combined_flat = (term_tensors[i] + term_tensors[j]).flatten()
                remaining_matrix = np.array([term_tensors[r].flatten() for r in remaining])
                try:
                    x, res, rk, sv = np.linalg.lstsq(remaining_matrix, combined_flat, rcond=1e-12)
                    recon = remaining_matrix @ x
                    if np.allclose(recon, combined_flat, atol=1e-10):
                        new_terms = [current_terms[t] for t in remaining]
                        U, V, W = terms_to_matrices(new_terms)
                        if verify_decomp(U, V, W):
                            current_terms = new_terms
                            pair_found = True
                            break
                except:
                    continue
            if pair_found:
                break

        # Triple-term removal
        rank = len(current_terms)
        term_tensors = [term_tensor(Ud, Vd, Wd) for Ud, Vd, Wd in current_terms]
        triple_found = False
        for i in range(rank):
            if time.time() > deadline - 10:
                break
            for j in range(i + 1, rank):
                if time.time() > deadline - 10:
                    break
                for k in range(j + 1, rank):
                    if time.time() > deadline - 10:
                        break
                    remaining = [t for t in range(rank) if t != i and t != j and t != k]
                    if len(remaining) < 3:
                        continue
                    combined_flat = (term_tensors[i] + term_tensors[j] + term_tensors[k]).flatten()
                    remaining_matrix = np.array([term_tensors[r].flatten() for r in remaining])
                    try:
                        x, res, rk, sv = np.linalg.lstsq(remaining_matrix, combined_flat, rcond=1e-12)
                        recon = remaining_matrix @ x
                        if np.allclose(recon, combined_flat, atol=1e-10):
                            new_terms = [current_terms[t] for t in remaining]
                            U, V, W = terms_to_matrices(new_terms)
                            if verify_decomp(U, V, W):
                                current_terms = new_terms
                                triple_found = True
                                break
                    except:
                        continue
                if triple_found:
                    break
            if triple_found:
                break

        return current_terms

    def try_rank28_embedding():
        if time.time() > deadline - 5:
            return None
        return None

    def try_cross_block_optimization(terms):
        """
        Try to find a lower-rank decomposition by allowing cross-block mixing.
        The idea: instead of 4 independent Strassen blocks + 1 schoolbook,
        we allow the U, V, W vectors to mix across blocks.
        
        We formulate this as: find matrices U (4xR), V (18xR), W (18xR) with
        R < 32 such that sum_r U[:,r] V[:,r]^T W[:,r]^T = T (the multiplication tensor).
        
        This is a tensor rank minimization problem. We use a heuristic:
        Start with the rank-32 solution and try to "merge" terms by finding
        linear dependencies in the tensor space.
        
        Specifically, we look for a set of R vectors (u_r, v_r, w_r) such that
        the tensor T = sum_r u_r ⊗ v_r ⊗ w_r.
        
        We use a greedy approach: start with all 32 terms, and try to replace
        pairs of terms with a single term (or remove one) while maintaining
        the tensor equality.
        """
        if time.time() > deadline - 60:
            return None
        
        n, m, p = 2, 2, 9
        current_terms = list(terms)
        
        # Build the tensor for each term
        term_tensors = [term_tensor(Ud, Vd, Wd) for Ud, Vd, Wd in current_terms]
        
        # Try to find a rank-reduction by solving a system of equations
        # For each pair of terms (i, j), check if term_i + term_j can be
        # represented as a linear combination of the remaining terms.
        
        # Actually, let's try a different approach: use the fact that the
        # multiplication tensor has a specific structure. For 2x2x9, we can
        # try to find a decomposition with rank 28, 29, 30, or 31.
        
        # One approach: use the "border rank" concept. The border rank of a
        # tensor is the minimum rank of a tensor in the closure of the rank-r
        # variety. If the border rank is less than the rank, we might be able
        # to find a lower-rank approximation that is "close" to the true tensor.
        
        # However, we need an EXACT decomposition, so border rank doesn't directly help.
        
        # Let's try a more direct approach: use numerical optimization to find
        # a rank-R decomposition. We'll use a greedy algorithm that starts with
        # a random initialization and iteratively refines the vectors.
        
        # For now, let's just return the input terms (no improvement).
        return current_terms

    # Build the base rank-32 decomposition
    terms = build_strassen_rank32()
    base_rank = len(terms)

    # Try rank-28 embedding approach
    if time.time() < deadline - 60:
        r28 = try_rank28_embedding()
        if r28 is not None:
            U, V, W = terms_to_matrices(r28)
            if verify_decomp(U, V, W):
                return {"U": U, "V": V, "W": W}

    # Try to reduce the rank using enhanced reduction
    reduced_terms = try_reduce_rank(terms)

    if len(reduced_terms) < len(terms):
        terms = reduced_terms

    # Try cross-block optimization
    if time.time() < deadline - 30:
        optimized_terms = try_cross_block_optimization(terms)
        if optimized_terms is not None and len(optimized_terms) < len(terms):
            terms = optimized_terms

    # Verify final result
    U, V, W = terms_to_matrices(terms)
    if not verify_decomp(U, V, W):
        # Ultimate fallback: schoolbook rank 36
        terms36 = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    terms36.append(({i*m+j: 1}, {j*p+k: 1}, {k*n+i: 1}))
        U, V, W = terms_to_matrices(terms36)

    return {"U": U, "V": V, "W": W}
