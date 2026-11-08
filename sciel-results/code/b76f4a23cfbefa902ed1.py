"""Tensor decomposition for matrix multiplication 3x3 * 3x8.
Uses Strassen's rank-7 decomposition for 2x2x2 blocks combined with
direct terms for remaining dimensions, with aggressive rank reduction.
"""

DIMENSIONS = [3, 3, 8]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    def strassen_decomp(row_off_a, col_off_a, row_off_b, col_off_b, U_t, V_t, W_t):
        """Add Strassen decomposition for 2x2*2x2 block at given offsets."""
        a = (row_off_a, col_off_a)
        b = (row_off_a, col_off_a + 1)
        c = (row_off_a + 1, col_off_a)
        d = (row_off_a + 1, col_off_a + 1)
        e = (row_off_b, col_off_b)
        f = (row_off_b, col_off_b + 1)
        g = (row_off_b + 1, col_off_b)
        h = (row_off_b + 1, col_off_b + 1)
        
        C00 = (row_off_a, col_off_b)
        C01 = (row_off_a, col_off_b + 1)
        C10 = (row_off_a + 1, col_off_b)
        C11 = (row_off_a + 1, col_off_b + 1)
        
        # m1 = (a+d)(e+h) -> C00 += 1, C11 += 1
        U_t.append([(a, 1), (d, 1)])
        V_t.append([(e, 1), (h, 1)])
        W_t.append([(C00, 1), (C11, 1)])
        
        # m2 = (c+d)*e -> C10 += 1, C11 -= 1
        U_t.append([(c, 1), (d, 1)])
        V_t.append([(e, 1)])
        W_t.append([(C10, 1), (C11, -1)])
        
        # m3 = a*(f-h) -> C01 += 1, C11 += 1
        U_t.append([(a, 1)])
        V_t.append([(f, 1), (h, -1)])
        W_t.append([(C01, 1), (C11, 1)])
        
        # m4 = d*(g-e) -> C00 += 1, C10 += 1
        U_t.append([(d, 1)])
        V_t.append([(g, 1), (e, -1)])
        W_t.append([(C00, 1), (C10, 1)])
        
        # m5 = (a+b)*h -> C00 -= 1, C01 += 1
        U_t.append([(a, 1), (b, 1)])
        V_t.append([(h, 1)])
        W_t.append([(C00, -1), (C01, 1)])
        
        # m6 = (c-a)*(e+f) -> C11 += 1
        U_t.append([(c, 1), (a, -1)])
        V_t.append([(e, 1), (f, 1)])
        W_t.append([(C11, 1)])
        
        # m7 = (b-d)*(g+h) -> C00 += 1
        U_t.append([(b, 1), (d, -1)])
        V_t.append([(g, 1), (h, 1)])
        W_t.append([(C00, 1)])
    
    def direct_term(i, j, k, U_t, V_t, W_t):
        """Add a direct term A[i,j]*B[j,k] to C[i,k]."""
        U_t.append([((i, j), 1)])
        V_t.append([((j, k), 1)])
        W_t.append([((i, k), 1)])
    
    U_t = []
    V_t = []
    W_t = []
    
    # A[0:2, 0:2] * B[0:2, 0:8]: 4 Strassen blocks
    for k_off in range(0, 8, 2):
        strassen_decomp(0, 0, 0, k_off, U_t, V_t, W_t)
    
    # A[0:2, 2] * B[2, 0:8]: direct (16 terms)
    for i in range(2):
        for k in range(8):
            direct_term(i, 2, k, U_t, V_t, W_t)
    
    # A[2, 0:2] * B[0:2, 0:8]: direct (16 terms)
    for j in range(2):
        for k in range(8):
            direct_term(2, j, k, U_t, V_t, W_t)
    
    # A[2, 2] * B[2, 0:8]: direct (8 terms)
    for k in range(8):
        direct_term(2, 2, k, U_t, V_t, W_t)
    
    rank = len(W_t)
    
    # Build the factor matrices
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    for idx in range(rank):
        for (i, j), coeff in U_t[idx]:
            U[i * m + j, idx] = coeff
        for (j, k), coeff in V_t[idx]:
            V[j * p + k, idx] = coeff
        for (i, k), coeff in W_t[idx]:
            W[k * n + i, idx] = coeff
    
    def verify_decomp(Ua, Va, Wa):
        """Verify that the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(Ua[i*m+j] * Va[j*p+k] * Wa[k*n+i])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    if not verify_decomp(U, V, W):
        # Fallback to schoolbook
        rank_fb = n * m * p
        U_fb = np.zeros((n * m, rank_fb), dtype=float)
        V_fb = np.zeros((m * p, rank_fb), dtype=float)
        W_fb = np.zeros((p * n, rank_fb), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, col] = 1
                    V_fb[j * p + k, col] = 1
                    W_fb[k * n + i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    # Convert to nice integer/half-integer values
    def to_nice_list(arr):
        """Convert numpy array to list of integer/half-integer values."""
        result = []
        for row in arr:
            nice_row = []
            for val in row:
                val2 = val * 2
                if abs(val2 - round(val2)) < 1e-6:
                    nice_val = round(val2) / 2.0
                    if abs(nice_val - round(nice_val)) < 1e-9:
                        nice_val = int(round(nice_val))
                    nice_row.append(nice_val)
                else:
                    nice_row.append(float(val))
            result.append(nice_row)
        return result
    
    # Try to reduce rank by finding linear dependencies
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank
    
    # Check for zero columns
    for r in range(rank - 1, -1, -1):
        if time.time() - start_time > DEADLINE:
            break
        if np.all(np.abs(U[:, r]) < 1e-12) and np.all(np.abs(V[:, r]) < 1e-12) and np.all(np.abs(W[:, r]) < 1e-12):
            U = np.delete(U, r, axis=1)
            V = np.delete(V, r, axis=1)
            W = np.delete(W, r, axis=1)
            rank -= 1
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            best_rank = rank
    
    # Try to find if any term is in the span of others - efficient approach
    remaining_time = DEADLINE - (time.time() - start_time)
    if remaining_time > 5 and rank > 0:
        # Build the full tensor matrix: shape (n*m*p, rank)
        tensor_mat = np.zeros((n*m*p, rank))
        for r in range(rank):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        tensor_mat[i*m*p + j*p + k, r] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # Use SVD to find the rank and identify which columns to keep
        try:
            U_svd, s, Vt_svd = np.linalg.svd(tensor_mat, full_matrices=False)
            # s is the list of singular values
            if len(s) > 0:
                tol = 1e-10 * s[0]
                significant_sv = np.where(s > tol)[0]
                current_rank = len(significant_sv)
                
                if current_rank < rank:
                    # Keep only the first 'current_rank' right singular vectors
                    # These correspond to the most important directions
                    keep_V = Vt_svd[:current_rank].T  # shape (rank, current_rank)
                    # We need to express the original columns in terms of these
                    # The kept columns are a basis for the column space
                    
                    # Actually, simpler: just zero out the dependent columns
                    # Use the fact that if rank is less than rank, some columns are dependent
                    # Find which columns to keep using QR with pivoting on the reduced matrix
                    
                    # Alternative: use the left singular vectors to project
                    # The column space is spanned by U_svd[:, :current_rank]
                    # Project each original column onto this space and check residual
                    
                    # Simpler approach: iteratively remove columns that are in span of others
                    # Start from the end and check
                    for r in range(rank - 1, -1, -1):
                        if time.time() - start_time > DEADLINE:
                            break
                        if r >= tensor_mat.shape[1]:
                            continue
                        
                        other_cols = [c for c in range(tensor_mat.shape[1]) if c != r]
                        if not other_cols:
                            continue
                        
                        # Check if column r is in span of others using least squares
                        others_mat = tensor_mat[:, other_cols]
                        try:
                            coeffs, residuals, rk, sv = np.linalg.lstsq(others_mat, tensor_mat[:, r], rcond=1e-10)
                            reconstruction = others_mat @ coeffs
                            if np.allclose(reconstruction, tensor_mat[:, r], atol=1e-8):
                                U = np.delete(U, r, axis=1)
                                V = np.delete(V, r, axis=1)
                                W = np.delete(W, r, axis=1)
                                tensor_mat = np.delete(tensor_mat, r, axis=1)
                                rank -= 1
                                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                                best_rank = rank
                        except:
                            pass
        except:
            pass
    
    # Final verification
    if not verify_decomp(best_U, best_V, best_W):
        # Rebuild from scratch
        U_t2 = []
        V_t2 = []
        W_t2 = []
        
        for k_off in range(0, 8, 2):
            strassen_decomp(0, 0, 0, k_off, U_t2, V_t2, W_t2)
        
        for i in range(2):
            for k in range(8):
                direct_term(i, 2, k, U_t2, V_t2, W_t2)
        
        for j in range(2):
            for k in range(8):
                direct_term(2, j, k, U_t2, V_t2, W_t2)
        
        for k in range(8):
            direct_term(2, 2, k, U_t2, V_t2, W_t2)
        
        rank2 = len(W_t2)
        U2 = np.zeros((n * m, rank2), dtype=float)
        V2 = np.zeros((m * p, rank2), dtype=float)
        W2 = np.zeros((p * n, rank2), dtype=float)
        
        for idx in range(rank2):
            for (i, j), coeff in U_t2[idx]:
                U2[i * m + j, idx] = coeff
            for (j, k), coeff in V_t2[idx]:
                V2[j * p + k, idx] = coeff
            for (i, k), coeff in W_t2[idx]:
                W2[k * n + i, idx] = coeff
        
        return {"U": to_nice_list(U2), "V": to_nice_list(V2), "W": to_nice_list(W2)}
    
    return {"U": to_nice_list(best_U), "V": to_nice_list(best_V), "W": to_nice_list(best_W)}
# EVOLVE-BLOCK-END
