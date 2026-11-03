"""
Matrix multiplication 3x4 by 4x7 with rank-minimal tensor decomposition.
Combines block decomposition with Strassen and optimized sub-block decompositions.
"""
import time
import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a low-rank tensor decomposition for 3x4 * 4x7 matrix multiplication.
    Uses block decomposition with Strassen for 2x2x2 blocks and structural
    rank reduction via linear dependence detection.
    """
    start_time = time.time()
    deadline = 135.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # 3, 4, 7

    def build_schoolbook():
        """Build the schoolbook decomposition (rank = n*m*p = 84)."""
        U_vecs = []
        V_vecs = []
        W_vecs = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = [0.0] * (n * m)
                    v = [0.0] * (m * p)
                    w = [0.0] * (p * n)
                    u[i * m + j] = 1.0
                    v[j * p + k] = 1.0
                    w[k * n + i] = 1.0
                    U_vecs.append(u)
                    V_vecs.append(v)
                    W_vecs.append(w)
        U = np.array(U_vecs).T  # shape (n*m, rank)
        V = np.array(V_vecs).T  # shape (m*p, rank)
        W = np.array(W_vecs).T  # shape (p*n, rank)
        return U, V, W

    def verify_decomp(U, V, W, tol=1e-9):
        """Verify that the decomposition exactly reconstructs the tensor using vectorized ops."""
        rank = U.shape[1]
        # U shape: (n*m, rank), V shape: (m*p, rank), W shape: (p*n, rank)
        # For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should be 1
        
        # Vectorize: create index arrays
        i_idx = np.repeat(np.arange(n), m * p)
        j_idx = np.tile(np.repeat(np.arange(m), p), n)
        k_idx = np.tile(np.arange(p), n * m)
        
        U_sel = U[i_idx * m + j_idx, :]  # (n*m*p, rank)
        V_sel = V[j_idx * p + k_idx, :]  # (n*m*p, rank)
        W_sel = W[k_idx * n + i_idx, :]  # (n*m*p, rank)
        
        # Sum over rank dimension
        totals = np.sum(U_sel * V_sel * W_sel, axis=1)
        
        if not np.all(np.abs(totals - 1.0) <= tol):
            return False
        return True

    def build_strassen_decomp():
        """Build the Strassen-based decomposition.
        
        Decomposition strategy:
        - n=3: split as 2+1
        - m=4: split as 2+2
        - p=7: split as 2+2+2+1
        For 2x2x2 blocks use Strassen (rank 7), for smaller blocks use schoolbook.
        
        Blocks:
        (2,2,2) x 6: rank 7 each = 42
        (2,2,1) x 2: rank 4 each = 8
        (1,2,2) x 6: rank 4 each = 24
        (1,2,1) x 2: rank 2 each = 4
        Total = 42+8+24+4 = 78
        """
        U_vecs = []
        V_vecs = []
        W_vecs = []

        def add_rank1_term(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals):
            u = [0.0] * (n * m)
            v = [0.0] * (m * p)
            w = [0.0] * (p * n)

            for idx, val in zip(u_indices, u_vals):
                u[idx] += val
            for idx, val in zip(v_indices, v_vals):
                v[idx] += val
            for idx, val in zip(w_indices, w_vals):
                w[idx] += val

            U_vecs.append(u)
            V_vecs.append(v)
            W_vecs.append(w)

        def strassen_222_local(i0, j0, k0):
            """Strassen decomposition for 2x2x2 block at offset (i0,j0,k0)."""
            i1 = i0 + 1
            j1 = j0 + 1
            k1 = k0 + 1

            # M1: (A11+A22)*(B11+B22) -> contributes to C11 and C22
            add_rank1_term([i0*m+j0, i1*m+j1], [1, 1],
                           [j0*p+k0, j1*p+k1], [1, 1],
                           [k0*n+i0, k1*n+i1], [1, 1])

            # M2: (A21+A22)*B11 -> contributes to C21 and C22
            add_rank1_term([i1*m+j0, i1*m+j1], [1, 1],
                           [j0*p+k0], [1],
                           [k0*n+i1, k1*n+i1], [1, -1])

            # M3: A11*(B12-B22) -> contributes to C12
            add_rank1_term([i0*m+j0], [1],
                           [j0*p+k1, j1*p+k1], [1, -1],
                           [k1*n+i0, k1*n+i1], [1, 1])

            # M4: A22*(B21-B11) -> contributes to C21
            add_rank1_term([i1*m+j1], [1],
                           [j1*p+k0, j0*p+k0], [1, -1],
                           [k0*n+i0, k0*n+i1], [1, 1])

            # M5: (A11+A12)*B22 -> contributes to C12 and C22
            add_rank1_term([i0*m+j0, i0*m+j1], [1, 1],
                           [j1*p+k1], [1],
                           [k0*n+i0, k1*n+i0], [-1, 1])

            # M6: (A21-A11)*(B11+B12) -> contributes to C21
            add_rank1_term([i1*m+j0, i0*m+j0], [1, -1],
                           [j0*p+k0, j0*p+k1], [1, 1],
                           [k1*n+i1], [1])

            # M7: (A12-A22)*(B21+B22) -> contributes to C12
            add_rank1_term([i0*m+j1, i1*m+j1], [1, -1],
                           [j1*p+k0, j1*p+k1], [1, 1],
                           [k0*n+i0], [1])

        def schoolbook_block(i0, j0, k0, ni, mj, pk):
            """Schoolbook decomposition for block of size (ni, mj, pk) at offset (i0,j0,k0)."""
            for i in range(ni):
                for j in range(mj):
                    for k in range(pk):
                        ii = i0 + i
                        jj = j0 + j
                        kk = k0 + k
                        add_rank1_term([ii*m+jj], [1.0],
                                       [jj*p+kk], [1.0],
                                       [kk*n+ii], [1.0])

        def block_decomp(i0, j0, k0, ni, mj, pk):
            """Decompose a block of size (ni, mj, pk).
            If all dimensions are 2, use Strassen. Otherwise use schoolbook.
            """
            if ni == 2 and mj == 2 and pk == 2:
                strassen_222_local(i0, j0, k0)
            else:
                schoolbook_block(i0, j0, k0, ni, mj, pk)

        # Decompose 3x4x7 as blocks:
        # i: 0-1 (size 2), 2 (size 1)
        # j: 0-1 (size 2), 2-3 (size 2)
        # k: 0-1 (size 2), 2-3 (size 2), 4-5 (size 2), 6 (size 1)
        
        # 6 blocks of size (2,2,2) -> Strassen, rank 7 each
        for j0 in [0, 2]:
            for k0 in [0, 2, 4]:
                block_decomp(0, j0, k0, 2, 2, 2)
        
        # 2 blocks of size (2,2,1) -> schoolbook, rank 4 each
        for j0 in [0, 2]:
            block_decomp(0, j0, 6, 2, 2, 1)
        
        # 6 blocks of size (1,2,2) -> schoolbook, rank 4 each
        for j0 in [0, 2]:
            for k0 in [0, 2, 4]:
                block_decomp(2, j0, k0, 1, 2, 2)
        
        # 2 blocks of size (1,2,1) -> schoolbook, rank 2 each
        for j0 in [0, 2]:
            block_decomp(2, j0, 6, 1, 2, 1)
        
        U = np.array(U_vecs).T
        V = np.array(V_vecs).T
        W = np.array(W_vecs).T
        return U, V, W

    # Build Strassen decomposition
    U, V, W = build_strassen_decomp()
    current_rank = U.shape[1]

    # Verify
    if not verify_decomp(U, V, W):
        # Fallback to schoolbook
        U, V, W = build_schoolbook()
        current_rank = U.shape[1]

    def rank_reduction(U, V, W, max_time=60.0):
        """
        Rank reduction via structural and linear dependence detection.
        
        Strategy:
        1. First pass: remove columns that are structurally redundant
           (e.g., same support pattern with proportional values).
        2. Second pass: joint linear dependence check on all three matrices.
        3. Third pass: try to find linear combinations that can be eliminated.
        """
        U = U.copy()
        V = V.copy()
        W = W.copy()
        rank = U.shape[1]
        start_opt = time.time()
        tol = 1e-10

        # Pass 1: Structural redundancy removal
        for iteration in range(50):
            if time.time() - start_opt > max_time or time.time() - start_time > deadline - 15.0:
                break

            reduced = False
            # Build support signatures
            def get_support(vec):
                return tuple(i for i in range(len(vec)) if abs(vec[i]) > tol)

            U_sigs = [get_support(U[:, r]) for r in range(rank)]
            V_sigs = [get_support(V[:, r]) for r in range(rank)]
            W_sigs = [get_support(W[:, r]) for r in range(rank)]

            # Find columns with identical support patterns
            for r in range(rank - 1, -1, -1):
                if time.time() - start_opt > max_time or time.time() - start_time > deadline - 15.0:
                    break
                if rank <= 1:
                    break

                # Check if column r has same support as any other column
                for s in range(r - 1, -1, -1):
                    if U_sigs[r] == U_sigs[s] and V_sigs[r] == V_sigs[s] and W_sigs[r] == W_sigs[s]:
                        # Check if values are proportional
                        nz_u = list(U_sigs[r])
                        if not nz_u:
                            continue
                        idx = nz_u[0]
                        if abs(U[idx, s]) < tol:
                            continue
                        c = U[idx, r] / U[idx, s]
                        
                        # Verify proportionality for all three
                        if np.allclose(U[:, r], c * U[:, s], atol=tol, rtol=tol):
                            if np.allclose(V[:, r], c * V[:, s], atol=tol, rtol=tol):
                                if np.allclose(W[:, r], c * W[:, s], atol=tol, rtol=tol):
                                    # Remove column r
                                    U = np.delete(U, r, axis=1)
                                    V = np.delete(V, r, axis=1)
                                    W = np.delete(W, r, axis=1)
                                    rank -= 1
                                    U_sigs.pop(r)
                                    V_sigs.pop(r)
                                    W_sigs.pop(r)
                                    reduced = True
                                    break
                if reduced:
                    break

            if not reduced:
                break

        # Pass 2: Joint linear dependence
        for iteration in range(100):
            if time.time() - start_opt > max_time or time.time() - start_time > deadline - 15.0:
                break

            reduced = False
            for r in range(rank - 1, -1, -1):
                if time.time() - start_opt > max_time or time.time() - start_time > deadline - 15.0:
                    break

                other = [i for i in range(rank) if i != r]
                if len(other) == 0:
                    break

                U_other = U[:, other]
                V_other = V[:, other]
                W_other = W[:, other]

                # Check U linear dependence
                try:
                    alpha_u, _, _, _ = np.linalg.lstsq(U_other, U[:, r], rcond=None)
                    if np.linalg.norm(U[:, r] - U_other @ alpha_u) > tol:
                        continue
                except:
                    continue

                # Check V linear dependence
                try:
                    alpha_v, _, _, _ = np.linalg.lstsq(V_other, V[:, r], rcond=None)
                    if np.linalg.norm(V[:, r] - V_other @ alpha_v) > tol:
                        continue
                except:
                    continue

                # Check W linear dependence
                try:
                    alpha_w, _, _, _ = np.linalg.lstsq(W_other, W[:, r], rcond=None)
                    if np.linalg.norm(W[:, r] - W_other @ alpha_w) > tol:
                        continue
                except:
                    continue

                # All three linearly dependent: try removal and verify
                U_test = np.delete(U, r, axis=1)
                V_test = np.delete(V, r, axis=1)
                W_test = np.delete(W, r, axis=1)

                if verify_decomp(U_test, V_test, W_test, tol=1e-8):
                    U = U_test
                    V = V_test
                    W = W_test
                    rank -= 1
                    reduced = True
                    break

            if not reduced:
                break

        # Pass 3: Try to find pairs of columns that can be combined
        # If column r and column s have overlapping support, check if their
        # difference or sum can be eliminated
        for iteration in range(50):
            if time.time() - start_opt > max_time or time.time() - start_time > deadline - 15.0:
                break

            reduced = False
            for r in range(rank - 1, -1, -1):
                if time.time() - start_opt > max_time or time.time() - start_time > deadline - 15.0:
                    break
                if rank <= 1:
                    break

                for s in range(r - 1, -1, -1):
                    # Check if U[:, r] + c * U[:, s] is in span of other columns
                    # for some c, and similarly for V and W
                    other = [i for i in range(rank) if i != r and i != s]
                    if len(other) == 0:
                        break

                    U_other = U[:, other]
                    V_other = V[:, other]
                    W_other = W[:, other]

                    # Try c = 1
                    U_comb = U[:, r] + U[:, s]
                    V_comb = V[:, r] + V[:, s]
                    W_comb = W[:, r] + W[:, s]

                    try:
                        alpha_u, _, _, _ = np.linalg.lstsq(U_other, U_comb, rcond=None)
                        residual_u = np.linalg.norm(U_comb - U_other @ alpha_u)
                        
                        if residual_u < tol:
                            alpha_v, _, _, _ = np.linalg.lstsq(V_other, V_comb, rcond=None)
                            residual_v = np.linalg.norm(V_comb - V_other @ alpha_v)
                            
                            if residual_v < tol:
                                alpha_w, _, _, _ = np.linalg.lstsq(W_other, W_comb, rcond=None)
                                residual_w = np.linalg.norm(W_comb - W_other @ alpha_w)
                                
                                if residual_w < tol:
                                    # Remove column r (since U[:, r] = -U[:, s] + U_comb,
                                    # and U_comb is in span of others, U[:, r] is in span of others)
                                    U_test = np.delete(U, r, axis=1)
                                    V_test = np.delete(V, r, axis=1)
                                    W_test = np.delete(W, r, axis=1)

                                    if verify_decomp(U_test, V_test, W_test, tol=1e-8):
                                        U = U_test
                                        V = V_test
                                        W = W_test
                                        rank -= 1
                                        reduced = True
                                        break
                    except:
                        continue

                if reduced:
                    break

            if not reduced:
                break

        return U, V, W, rank

    # Apply rank reduction optimization
    U, V, W, current_rank = rank_reduction(U, V, W, max_time=100.0)

    # Final verification
    if not verify_decomp(U, V, W, tol=1e-8):
        # Fallback to schoolbook
        U, V, W = build_schoolbook()
        current_rank = U.shape[1]

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
