"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    import numpy as np
    import time

    _deadline = time.time() + 140
    n, m, p = DIMENSIONS  # 3, 4, 6

    def verify(U_mat, V_mat, W_mat, r):
        u = U_mat.reshape(n, m, r)
        v = V_mat.reshape(m, p, r)
        w = W_mat.reshape(p, n, r)
        T_reconstructed = np.einsum('ijc,jkc,kic->ijk', u, v, w)
        target = np.ones((n, m, p))
        return np.allclose(T_reconstructed, target, atol=1e-9)

    def build_decomposition():
        """Build a block-structured rank-66 decomposition."""
        U66 = [[0.0] * 66 for _ in range(n * m)]
        V66 = [[0.0] * 66 for _ in range(m * p)]
        W66 = [[0.0] * 66 for _ in range(p * n)]
        col = 0

        def add_strassen(a_r, a_c, b_r, b_c, c_r, c_c):
            nonlocal col
            A00 = a_r * m + a_c
            A01 = a_r * m + a_c + 1
            A10 = (a_r + 1) * m + a_c
            A11 = (a_r + 1) * m + a_c + 1
            B00 = b_r * p + b_c
            B01 = b_r * p + b_c + 1
            B10 = (b_r + 1) * p + b_c
            B11 = (b_r + 1) * p + b_c + 1
            C00 = c_c * n + c_r
            C01 = (c_c + 1) * n + c_r
            C10 = c_c * n + (c_r + 1)
            C11 = (c_c + 1) * n + (c_r + 1)

            # M1 = (A00+A11)*(B00+B11) -> C00 += M1, C11 += M1
            U66[A00][col] += 1; U66[A11][col] += 1
            V66[B00][col] += 1; V66[B11][col] += 1
            W66[C00][col] += 1; W66[C11][col] += 1
            col += 1

            # M2 = (A10+A11)*B00 -> C10 += M2, C11 -= M2
            U66[A10][col] += 1; U66[A11][col] += 1
            V66[B00][col] += 1
            W66[C10][col] += 1; W66[C11][col] -= 1
            col += 1

            # M3 = A00*(B01-B11) -> C01 += M3, C11 += M3
            U66[A00][col] += 1
            V66[B01][col] += 1; V66[B11][col] -= 1
            W66[C01][col] += 1; W66[C11][col] += 1
            col += 1

            # M4 = A11*(B10-B00) -> C00 += M4, C10 += M4
            U66[A11][col] += 1
            V66[B10][col] += 1; V66[B00][col] -= 1
            W66[C00][col] += 1; W66[C10][col] += 1
            col += 1

            # M5 = (A00+A01)*B11 -> C01 += M5, C00 -= M5
            U66[A00][col] += 1; U66[A01][col] += 1
            V66[B11][col] += 1
            W66[C00][col] -= 1; W66[C01][col] += 1
            col += 1

            # M6 = (A10-A00)*(B00+B01) -> C11 += M6
            U66[A10][col] += 1; U66[A00][col] -= 1
            V66[B00][col] += 1; V66[B01][col] += 1
            W66[C11][col] += 1
            col += 1

            # M7 = (A01-A11)*(B10+B11) -> C00 += M7
            U66[A01][col] += 1; U66[A11][col] -= 1
            V66[B10][col] += 1; V66[B11][col] += 1
            W66[C00][col] += 1
            col += 1

        def add_1x2_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
            nonlocal col
            A0 = a_r * m + a_c
            A1 = a_r * m + a_c + 1
            B00 = b_r * p + b_c
            B01 = b_r * p + b_c + 1
            B10 = (b_r + 1) * p + b_c
            B11 = (b_r + 1) * p + b_c + 1
            C0 = c_c * n + c_r
            C1 = (c_c + 1) * n + c_r

            # Direct: A0*B00 -> C0, A1*B10 -> C0
            U66[A0][col] += 1; V66[B00][col] += 1; W66[C0][col] += 1
            col += 1
            U66[A1][col] += 1; V66[B10][col] += 1; W66[C0][col] += 1
            col += 1
            # Direct: A0*B01 -> C1, A1*B11 -> C1
            U66[A0][col] += 1; V66[B01][col] += 1; W66[C1][col] += 1
            col += 1
            U66[A1][col] += 1; V66[B11][col] += 1; W66[C1][col] += 1
            col += 1

        # Build the decomposition
        # Row 0: direct terms for all 6 column positions (3 groups of 2)
        for t0 in [0, 2, 4]:
            add_1x2_2x2(0, 0, 0, t0, 0, t0)
            add_1x2_2x2(0, 2, 2, t0, 0, t0)

        # Rows 1-2: Strassen for all 6 column positions (3 groups of 2)
        for t0 in [0, 2, 4]:
            add_strassen(1, 0, 0, t0, 1, t0)
            add_strassen(1, 2, 2, t0, 1, t0)

        U = np.array(U66, dtype=float)
        V = np.array(V66, dtype=float)
        W = np.array(W66, dtype=float)
        return U, V, W

    def tensor_flatten(U, V, W, r):
        """Flatten the tensor into a matrix of shape (n*m*p, r)"""
        nmp = n * m * p
        M = np.zeros((nmp, r))
        for l in range(r):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        M[i * m * p + j * p + k, l] = (
                            U[i * m + j, l] * 
                            V[j * p + k, l] * 
                            W[k * n + i, l]
                        )
        return M

    def try_reduce_rank(U, V, W, r, max_reductions=10):
        """Try to reduce rank by finding linear dependencies via SVD nullspace and pairwise elimination."""
        best_rank = r
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for _ in range(max_reductions):
            if time.time() > _deadline - 20:
                break
                
            nmp = n * m * p
            M = tensor_flatten(best_U, best_V, best_W, best_rank)
            
            try:
                U_svd, s, Vt = np.linalg.svd(M, full_matrices=False)
                tol = 1e-10
                rank_M = np.sum(s > tol)
                
                if rank_M < best_rank:
                    null_space = Vt[rank_M:, :].T
                    
                    found = False
                    for nv in null_space:
                        if time.time() > _deadline - 20:
                            break
                        max_idx = np.argmax(np.abs(nv))
                        if abs(nv[max_idx]) < tol:
                            continue
                        
                        U_new = np.delete(best_U, max_idx, axis=1)
                        V_new = np.delete(best_V, max_idx, axis=1)
                        W_new = np.delete(best_W, max_idx, axis=1)
                        
                        if verify(U_new, V_new, W_new, best_rank - 1):
                            best_U, best_V, best_W = U_new, V_new, W_new
                            best_rank -= 1
                            found = True
                            break
                    
                    if not found:
                        # Try pairwise elimination: remove two columns at once
                        found = False
                        for i in range(best_rank):
                            if time.time() > _deadline - 20:
                                break
                            for j in range(i+1, best_rank):
                                if time.time() > _deadline - 20:
                                    break
                                U_new = np.delete(best_U, [i, j], axis=1)
                                V_new = np.delete(best_V, [i, j], axis=1)
                                W_new = np.delete(best_W, [i, j], axis=1)
                                
                                if verify(U_new, V_new, W_new, best_rank - 2):
                                    best_U, best_V, best_W = U_new, V_new, W_new
                                    best_rank -= 2
                                    found = True
                                    break
                            if found:
                                break
                        if not found:
                            break
                else:
                    # No nullspace, but try pairwise elimination anyway
                    found = False
                    for i in range(best_rank):
                        if time.time() > _deadline - 20:
                            break
                        for j in range(i+1, best_rank):
                            if time.time() > _deadline - 20:
                                break
                            U_new = np.delete(best_U, [i, j], axis=1)
                            V_new = np.delete(best_V, [i, j], axis=1)
                            W_new = np.delete(best_W, [i, j], axis=1)
                            
                            if verify(U_new, V_new, W_new, best_rank - 2):
                                best_U, best_V, best_W = U_new, V_new, W_new
                                best_rank -= 2
                                found = True
                                break
                        if found:
                            break
                    if not found:
                        break
            except:
                break
        
        return best_rank, best_U, best_V, best_W

    def try_gaussian_elimination(U, V, W, r):
        """Use Gaussian elimination on the coefficient matrix to find dependencies."""
        best_rank = r
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for _ in range(5):
            if time.time() > _deadline - 20:
                break
                
            M = tensor_flatten(best_U, best_V, best_W, best_rank)
            
            try:
                # Perform rank-revealing QR decomposition
                Q, R = np.linalg.qr(M)
                tol = 1e-10
                # Count significant diagonal entries of R
                diag = np.abs(np.diag(R))
                rank_M = np.sum(diag > tol)
                
                if rank_M < best_rank:
                    # The nullspace is spanned by the last (best_rank - rank_M) rows of Q^T
                    # Actually, for QR, the nullspace vectors are in the last columns of Q
                    # Let's use SVD instead for clarity
                    U_svd, s, Vt = np.linalg.svd(M, full_matrices=False)
                    null_space = Vt[rank_M:, :].T
                    
                    found = False
                    for nv in null_space:
                        if time.time() > _deadline - 20:
                            break
                        max_idx = np.argmax(np.abs(nv))
                        if abs(nv[max_idx]) < tol:
                            continue
                        
                        U_new = np.delete(best_U, max_idx, axis=1)
                        V_new = np.delete(best_V, max_idx, axis=1)
                        W_new = np.delete(best_W, max_idx, axis=1)
                        
                        if verify(U_new, V_new, W_new, best_rank - 1):
                            best_U, best_V, best_W = U_new, V_new, W_new
                            best_rank -= 1
                            found = True
                            break
                    
                    if not found:
                        break
                else:
                    break
            except:
                break
        
        return best_rank, best_U, best_V, best_W

    def try_random_mixing(U, V, W, r, n_trials=8):
        """Try random mixing of columns to find better basis for SVD reduction."""
        best_rank = r
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        rng = np.random.default_rng(42)
        
        for trial in range(n_trials):
            if time.time() > _deadline - 20:
                break
            
            # Generate random invertible matrices
            P = rng.standard_normal((r, r))
            Q = rng.standard_normal((r, r))
            R = rng.standard_normal((r, r))
            
            # Check if invertible
            if abs(np.linalg.det(P)) < 1e-10 or abs(np.linalg.det(Q)) < 1e-10 or abs(np.linalg.det(R)) < 1e-10:
                continue
            
            U_mixed = U @ P
            V_mixed = V @ Q
            W_mixed = W @ R
            
            if not verify(U_mixed, V_mixed, W_mixed, r):
                continue
            
            # Try to reduce
            rank_mixed, U_reduced, V_reduced, W_reduced = try_reduce_rank(
                U_mixed, V_mixed, W_mixed, r, max_reductions=3
            )
            
            if rank_mixed < best_rank:
                best_rank = rank_mixed
                best_U, best_V, best_W = U_reduced, V_reduced, W_reduced
        
        return best_rank, best_U, best_V, best_W

    def try_structural_mixing(U, V, W, r):
        """Try mixing columns within Strassen blocks to expose dependencies."""
        best_rank = r
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        # The 66 columns are organized as:
        # 18 direct terms (row 0) + 48 Strassen terms (rows 1-2)
        # Each Strassen block has 7 terms
        # Try mixing within each Strassen block
        
        # Identify Strassen block boundaries
        # Row 0: columns 0-17 (18 direct terms)
        # Rows 1-2: columns 18-65 (48 Strassen terms, 6 blocks of 7 each)
        
        # Try mixing within each Strassen block
        for block_start in range(18, 66, 7):
            if time.time() > _deadline - 20:
                break
            
            block_end = min(block_start + 7, 66)
            block_size = block_end - block_start
            
            if block_size < 2:
                continue
            
            # Generate random mixing within this block
            rng = np.random.default_rng(hash((block_start, block_end)) % (2**32))
            P_block = rng.standard_normal((block_size, block_size))
            
            if abs(np.linalg.det(P_block)) < 1e-10:
                continue
            
            # Apply mixing
            U_mixed = best_U.copy()
            V_mixed = best_V.copy()
            W_mixed = best_W.copy()
            
            U_mixed[:, block_start:block_end] = U_mixed[:, block_start:block_end] @ P_block
            V_mixed[:, block_start:block_end] = V_mixed[:, block_start:block_end] @ P_block
            W_mixed[:, block_start:block_end] = W_mixed[:, block_start:block_end] @ P_block
            
            if not verify(U_mixed, V_mixed, W_mixed, r):
                continue
            
            # Try to reduce
            rank_mixed, U_reduced, V_reduced, W_reduced = try_reduce_rank(
                U_mixed, V_mixed, W_mixed, r, max_reductions=2
            )
            
            if rank_mixed < best_rank:
                best_rank = rank_mixed
                best_U, best_V, best_W = U_reduced, V_reduced, W_reduced
        
        return best_rank, best_U, best_V, best_W

    # Build initial decomposition
    U, V, W = build_decomposition()
    
    # Verify rank-66
    if not verify(U, V, W, 66):
        # Fallback to schoolbook
        rank_school = n * m * p
        U = np.zeros((n * m, rank_school))
        V = np.zeros((m * p, rank_school))
        W = np.zeros((p * n, rank_school))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to reduce rank using multiple strategies
    best_rank, best_U, best_V, best_W = try_reduce_rank(U, V, W, 66, max_reductions=5)
    
    if time.time() > _deadline - 30:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Try Gaussian elimination approach
    if best_rank > 60:
        rank_ge, U_ge, V_ge, W_ge = try_gaussian_elimination(
            best_U, best_V, best_W, best_rank
        )
        if rank_ge < best_rank:
            best_rank, best_U, best_V, best_W = rank_ge, U_ge, V_ge, W_ge
    
    if time.time() > _deadline - 25:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Try structural mixing within Strassen blocks
    if best_rank > 60:
        rank_struct, U_struct, V_struct, W_struct = try_structural_mixing(
            best_U, best_V, best_W, best_rank
        )
        if rank_struct < best_rank:
            best_rank, best_U, best_V, best_W = rank_struct, U_struct, V_struct, W_struct
    
    if time.time() > _deadline - 20:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Try random mixing
    if best_rank > 60:
        rank_mixed, U_mixed, V_mixed, W_mixed = try_random_mixing(
            best_U, best_V, best_W, best_rank, n_trials=5
        )
        if rank_mixed < best_rank:
            best_rank, best_U, best_V, best_W = rank_mixed, U_mixed, V_mixed, W_mixed
    
    if time.time() > _deadline - 10:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Final verification and return
    if verify(best_U, best_V, best_W, best_rank):
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        # Fallback to original if something went wrong
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
