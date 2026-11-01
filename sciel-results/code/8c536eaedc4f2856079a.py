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
        """
        Try to construct a rank-28 decomposition by embedding 2x2x9 into 2x2x12.
        The 2x2x12 tensor has known rank-28 decompositions. We try to find one
        where the last 3 columns of B contribute zero to the output.
        
        Strategy: Build a rank-28 decomposition for 2x2x12 using 3 groups of 4
        Strassen-like blocks (each handling 4 columns with rank ~9.33... not integer).
        
        Alternative: Use 2x2x4 blocks. Rank(2,2,4) is known to be 14 (from literature).
        2x2x9 = 2x2x4 + 2x2x4 + 2x2x1. Rank = 14 + 14 + 4 = 32. That's our current.
        
        Better: 2x2x3 has rank 8. So 3 blocks of 2x2x3 gives 24. But we need to
        verify rank(2,2,3)=8 is achievable with our indexing.
        
        Let me try: 3 blocks of 2x2x3, each with rank 8, total rank 24.
        """
        if time.time() > deadline - 5:
            return None
        
        # Build rank-8 decomposition for 2x2x3
        # A is 2x2 (indices 0-3), B is 2x3 (indices 0-5), C is 2x3 (indices 0-5)
        # C[i,j] = sum_k A[i,k] * B[k,j]
        # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
        # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
        # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
        # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
        # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
        # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
        
        # Let me try to construct rank-8 for 2x2x3 by searching.
        # Use the known fact that rank(2,2,3) = 8.
        
        # One approach: start from schoolbook rank 6 and add 2 terms to get rank 8?
        # No, schoolbook for 2x2x3 is rank 6 (2*2*3/2 = 6? No, it's 2*3=6 terms).
        # Actually schoolbook: for each (i,j) in C, we need sum over k. 
        # C has 6 entries, each is sum of 2 products. Schoolbook uses 6 terms.
        # So rank(2,2,3) <= 6? No wait, that can't be right.
        
        # Schoolbook: C[i,j] = A[i,0]*B[0,j] + A[i,1]*B[1,j]
        # Term for (i,j): U has A[i,0], V has B[0,j], W has C[i,j]
        # That's 6 terms for the B[0,j] part and 6 for B[1,j] part = 12 terms.
        # Wait no. For each (i,j), we have 2 terms. Total 6*2 = 12 terms.
        # But we can share: for fixed i, the two terms for j=0,1,2 share U[i,0] and U[i,1].
        # Actually in tensor decomposition, each term is a rank-1 slice.
        # Term r: U[:,r] * V[:,r] * W[:,r]
        # For schoolbook: term (i,j,0): U[i,0]=1, V[0,j]=1, W[i,j]=1 -> that's 6 terms
        #                term (i,j,1): U[i,1]=1, V[1,j]=1, W[i,j]=1 -> that's 6 terms
        # Total schoolbook rank = 12.
        
        # So rank(2,2,3) <= 8 means we can save 4 terms.
        
        # Let me just try to verify if a rank-8 decomposition exists by construction.
        # I'll use a known construction from the literature.
        
        # Actually, let me try a different approach: use the rank-32 base and try
        # more aggressive reduction, including the embedding idea.
        
        # For the embedding: treat columns 0-8 as part of a 12-column problem.
        # If we have a rank-28 decomposition of 2x2x12, we need the terms involving
        # columns 9,10,11 to cancel out when restricted to columns 0-8.
        
        # This is complex. Let me instead focus on enhancing the reduction.
        return None

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
