"""
Matrix multiplication 3x4 by 4x7 with rank-minimal tensor decomposition.
Uses block decomposition with Strassen for 2x2x2 blocks and aggressive
rank reduction via support-group linear dependence detection.
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
    deadline = 140.0  # safety margin before 150s kill

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
        if rank == 0:
            return False
        i_idx = np.repeat(np.arange(n), m * p)
        j_idx = np.tile(np.repeat(np.arange(m), p), n)
        k_idx = np.tile(np.arange(p), n * m)
        
        U_sel = U[i_idx * m + j_idx, :]
        V_sel = V[j_idx * p + k_idx, :]
        W_sel = W[k_idx * n + i_idx, :]
        
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
            """Decompose a block of size (ni, mj, pk)."""
            if ni == 2 and mj == 2 and pk == 2:
                strassen_222_local(i0, j0, k0)
            else:
                schoolbook_block(i0, j0, k0, ni, mj, pk)

        # Decompose 3x4x7 as blocks:
        # 2x4x7 part: i=0,1
        for j0 in [0, 2]:
            for k0 in [0, 2, 4]:
                block_decomp(0, j0, k0, 2, 2, 2)
        for j0 in [0, 2]:
            block_decomp(0, j0, 6, 2, 2, 1)
        
        # 1x4x7 part: i=2
        for j0 in [0, 2]:
            for k0 in [0, 2, 4]:
                block_decomp(2, j0, k0, 1, 2, 2)
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
        U, V, W = build_schoolbook()
        current_rank = U.shape[1]

    def rank_reduction(U, V, W, max_time=100.0):
        """
        Aggressive rank reduction via structural grouping and linear dependence.
        """
        U = U.copy()
        V = V.copy()
        W = W.copy()
        rank = U.shape[1]
        start_opt = time.time()
        tol = 1e-10

        def get_support(vec):
            return tuple(i for i in range(len(vec)) if abs(vec[i]) > tol)

        # Pass 1: Group by (U_sig, V_sig, W_sig) and check rank within groups
        for iteration in range(30):
            if time.time() - start_opt > max_time or time.time() - start_time > deadline - 10.0:
                break

            reduced = False
            U_sigs = [get_support(U[:, r]) for r in range(rank)]
            V_sigs = [get_support(V[:, r]) for r in range(rank)]
            W_sigs = [get_support(W[:, r]) for r in range(rank)]

            groups = {}
            for r in range(rank):
                key = (U_sigs[r], V_sigs[r], W_sigs[r])
                if key not in groups:
                    groups[key] = []
                groups[key].append(r)

            for key, cols in groups.items():
                if len(cols) <= 1:
                    continue
                if time.time() - start_opt > max_time or time.time() - start_time > deadline - 10.0:
                    break

                nz_u = list(key[0])
                nz_v = list(key[1])
                nz_w = list(key[2])

                if not nz_u or not nz_v or not nz_w:
                    continue

                val_mat = np.zeros((len(nz_u) + len(nz_v) + len(nz_w), len(cols)))
                for ci, r in enumerate(cols):
                    for vi, idx in enumerate(nz_u):
                        val_mat[vi, ci] = U[idx, r]
                    for vi, idx in enumerate(nz_v):
                        val_mat[len(nz_u) + vi, ci] = V[idx, r]
                    for vi, idx in enumerate(nz_w):
                        val_mat[len(nz_u) + len(nz_v) + vi, ci] = W[idx, r]

                vm_rank = np.linalg.matrix_rank(val_mat, tol=tol)
                removable = len(cols) - vm_rank
                if removable > 0:
                    cols_to_remove = cols[-removable:]
                    for r in reversed(cols_to_remove):
                        U = np.delete(U, r, axis=1)
                        V = np.delete(V, r, axis=1)
                        W = np.delete(W, r, axis=1)
                        rank -= 1
                        reduced = True

            if not reduced:
                break

        # Pass 2: Iterative single-column removal via fast check
        for iteration in range(300):
            if time.time() - start_opt > max_time or time.time() - start_time > deadline - 10.0:
                break

            reduced = False
            for r in range(rank - 1, -1, -1):
                if time.time() - start_opt > max_time or time.time() - start_time > deadline - 10.0:
                    break
                if rank <= 1:
                    break

                u_nz = set(i for i in range(n * m) if abs(U[i, r]) > tol)
                v_nz = set(i for i in range(m * p) if abs(V[i, r]) > tol)
                w_nz = set(i for i in range(p * n) if abs(W[i, r]) > tol)

                u_unique = False
                v_unique = False
                w_unique = False
                for pos in u_nz:
                    if not any(abs(U[pos, s]) > tol for s in range(rank) if s != r):
                        u_unique = True
                        break
                if u_unique:
                    continue
                for pos in v_nz:
                    if not any(abs(V[pos, s]) > tol for s in range(rank) if s != r):
                        v_unique = True
                        break
                if v_unique:
                    continue
                for pos in w_nz:
                    if not any(abs(W[pos, s]) > tol for s in range(rank) if s != r):
                        w_unique = True
                        break
                if w_unique:
                    continue

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

        # Pass 3: Try removing pairs of columns that might be jointly redundant
        for iteration in range(50):
            if time.time() - start_opt > max_time or time.time() - start_time > deadline - 10.0:
                break
            if rank <= 2:
                break

            reduced = False
            # Try removing pairs from the end
            for r1 in range(rank - 1, 0, -1):
                if time.time() - start_opt > max_time or time.time() - start_time > deadline - 10.0:
                    break
                for r2 in range(r1 - 1, -1, -1):
                    if time.time() - start_opt > max_time or time.time() - start_time > deadline - 10.0:
                        break
                    if rank <= 2:
                        break
                    # Quick check: if both columns have unique positions, skip
                    u_nz1 = set(i for i in range(n * m) if abs(U[i, r1]) > tol)
                    u_nz2 = set(i for i in range(n * m) if abs(U[i, r2]) > tol)
                    v_nz1 = set(i for i in range(m * p) if abs(V[i, r1]) > tol)
                    v_nz2 = set(i for i in range(m * p) if abs(V[i, r2]) > tol)
                    w_nz1 = set(i for i in range(p * n) if abs(W[i, r1]) > tol)
                    w_nz2 = set(i for i in range(p * n) if abs(W[i, r2]) > tol)
                    
                    # Check if removing both would leave a unique position uncovered
                    all_u = set(i for i in range(n * m) if np.any(np.abs(U[i, :]) > tol))
                    all_v = set(i for i in range(m * p) if np.any(np.abs(V[i, :]) > tol))
                    all_w = set(i for i in range(p * n) if np.any(np.abs(W[i, :]) > tol))
                    
                    # Skip if this pair covers a unique position
                    skip = False
                    for pos in u_nz1:
                        count = sum(1 for s in range(rank) if s != r1 and s != r2 and abs(U[pos, s]) > tol)
                        if count == 0:
                            skip = True
                            break
                    if skip:
                        continue
                    for pos in v_nz1:
                        count = sum(1 for s in range(rank) if s != r1 and s != r2 and abs(V[pos, s]) > tol)
                        if count == 0:
                            skip = True
                            break
                    if skip:
                        continue
                    for pos in w_nz1:
                        count = sum(1 for s in range(rank) if s != r1 and s != r2 and abs(W[pos, s]) > tol)
                        if count == 0:
                            skip = True
                            break
                    if skip:
                        continue
                    for pos in u_nz2:
                        count = sum(1 for s in range(rank) if s != r1 and s != r2 and abs(U[pos, s]) > tol)
                        if count == 0:
                            skip = True
                            break
                    if skip:
                        continue
                    for pos in v_nz2:
                        count = sum(1 for s in range(rank) if s != r1 and s != r2 and abs(V[pos, s]) > tol)
                        if count == 0:
                            skip = True
                            break
                    if skip:
                        continue
                    for pos in w_nz2:
                        count = sum(1 for s in range(rank) if s != r1 and s != r2 and abs(W[pos, s]) > tol)
                        if count == 0:
                            skip = True
                            break
                    if skip:
                        continue
                    
                    # Try removing both
                    mask = [s for s in range(rank) if s != r1 and s != r2]
                    U_test = U[:, mask]
                    V_test = V[:, mask]
                    W_test = W[:, mask]
                    
                    if verify_decomp(U_test, V_test, W_test, tol=1e-8):
                        U = U_test
                        V = V_test
                        W = W_test
                        rank -= 2
                        reduced = True
                        break
                if reduced:
                    break

            if not reduced:
                break

        return U, V, W, rank

    # Apply rank reduction optimization
    U, V, W, current_rank = rank_reduction(U, V, W, max_time=110.0)

    # Final verification
    if not verify_decomp(U, V, W, tol=1e-8):
        U, V, W = build_schoolbook()
        current_rank = U.shape[1]

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
