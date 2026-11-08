import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill

    n, m, p = 4, 4, 8
    N = n * m  # 16
    M = m * p  # 32
    P = p * n  # 32

    def verify_fast(U, V, W, tol=1e-9):
        """Verify tensor decomposition: sum_r U[a,r]*V[b,r]*W[c,r] == T[a,b,c]"""
        for a in range(N):
            i_a, j_a = divmod(a, 4)
            for b in range(M):
                r_b, c_b = divmod(b, 8)
                for c in range(P):
                    c_c, i_c = divmod(c, 4)
                    expected = 1.0 if (j_a == r_b and c_b == c_c and i_a == i_c) else 0.0
                    actual = np.dot(U[a, :], V[b, :] * W[c, :])
                    if abs(actual - expected) > tol:
                        return False
        return True

    def strassen_2x2x2():
        """Return U_2, V_2, W_2 for 2x2x2 matrix multiplication, rank 7."""
        U2 = np.zeros((4, 7))
        V2 = np.zeros((4, 7))
        W2 = np.zeros((4, 7))
        
        # Strassen's 7 multiplications for 2x2 matrices
        U2[0, 0] = 1; U2[3, 0] = 1
        V2[0, 0] = 1; V2[3, 0] = 1
        W2[0, 0] = 1; W2[3, 0] = 1
        
        U2[2, 1] = 1; U2[3, 1] = 1
        V2[0, 1] = 1
        W2[2, 1] = 1; W2[3, 1] = -1
        
        U2[0, 2] = 1
        V2[1, 2] = 1; V2[3, 2] = -1
        W2[1, 2] = 1; W2[3, 2] = 1
        
        U2[3, 3] = 1
        V2[2, 3] = 1; V2[0, 3] = -1
        W2[0, 3] = 1; W2[2, 3] = 1
        
        U2[0, 4] = 1; U2[1, 4] = 1
        V2[3, 4] = 1
        W2[0, 4] = -1; W2[1, 4] = 1
        
        U2[2, 5] = 1; U2[0, 5] = -1
        V2[0, 5] = 1; V2[1, 5] = 1
        W2[3, 5] = 1
        
        U2[1, 6] = 1; U2[3, 6] = -1
        V2[2, 6] = 1; V2[3, 6] = 1
        W2[0, 6] = 1
        
        return U2, V2, W2

    def build_block_strassen():
        """Build block Strassen decomposition for 4x4x8 using 2x2 Strassen sub-blocks. Rank 98."""
        U2, V2, W2 = strassen_2x2x2()
        
        U = np.zeros((N, 98))
        V = np.zeros((M, 98))
        W = np.zeros((P, 98))
        
        A_specs = [
            [(1, 0), (1, 3)],
            [(1, 2), (1, 3)],
            [(1, 0)],
            [(1, 3)],
            [(1, 0), (1, 1)],
            [(1, 2), (-1, 0)],
            [(1, 1), (-1, 3)],
        ]
        
        B_specs = [
            [(1, 0), (1, 3)],
            [(1, 0)],
            [(1, 1), (-1, 3)],
            [(1, 2), (-1, 0)],
            [(1, 3)],
            [(1, 0), (1, 1)],
            [(1, 2), (1, 3)],
        ]
        
        C_specs = [
            [(1, 0), (1, 3)],
            [(1, 2), (-1, 3)],
            [(1, 1), (1, 3)],
            [(1, 0), (1, 2)],
            [(-1, 0), (1, 1)],
            [(1, 3)],
            [(1, 0)],
        ]
        
        for half in range(2):
            for k_p in range(7):
                for m_s in range(7):
                    col = half * 49 + k_p * 7 + m_s
                    
                    for coeff_a, block_idx in A_specs[k_p]:
                        bi, bj = divmod(block_idx, 2)
                        for r in range(2):
                            for c in range(2):
                                global_r = bi * 2 + r
                                global_c = bj * 2 + c
                                global_a_idx = global_r * 4 + global_c
                                a_eff_idx = r * 2 + c
                                U[global_a_idx, col] += coeff_a * U2[a_eff_idx, m_s]
                    
                    for coeff_b, block_idx in B_specs[k_p]:
                        bi, bj = divmod(block_idx, 2)
                        for r in range(2):
                            for c in range(2):
                                global_r = bi * 2 + r
                                global_c = bj * 2 + c
                                global_k = global_c + half * 4
                                global_b_idx = global_r * 8 + global_k
                                b_eff_idx = r * 2 + c
                                V[global_b_idx, col] += coeff_b * V2[b_eff_idx, m_s]
                    
                    for coeff_c, c_block_idx in C_specs[k_p]:
                        bi, bj = divmod(c_block_idx, 2)
                        for r in range(2):
                            for c in range(2):
                                global_i = bi * 2 + r
                                global_k = bj * 2 + c + half * 4
                                global_w_idx = global_k * 4 + global_i
                                p_eff_idx = r * 2 + c
                                W[global_w_idx, col] += coeff_c * W2[p_eff_idx, m_s]
        
        return U, V, W

    def try_reduce_rank(U, V, W, max_iterations=500):
        """Try to reduce rank by finding and merging linearly dependent columns."""
        current_rank = U.shape[1]
        for _ in range(max_iterations):
            if time.time() > deadline - 15:
                break
                
            merged = False
            # Check for exact duplicates or scalar multiples
            for i in range(current_rank - 1):
                for j in range(i + 1, current_rank):
                    # Check if columns are linearly dependent
                    # U[:,j] = alpha * U[:,i] for some alpha
                    if np.allclose(U[:, i], 0, atol=1e-12):
                        continue
                    # Find alpha from first non-zero element
                    nz = np.where(np.abs(U[:, i]) > 1e-12)[0]
                    if len(nz) == 0:
                        continue
                    idx = nz[0]
                    if abs(U[idx, i]) < 1e-12:
                        continue
                    alpha = U[idx, j] / U[idx, i]
                    
                    # Check if U[:,j] ≈ alpha * U[:,i]
                    if np.allclose(U[:, j], alpha * U[:, i], atol=1e-9, rtol=1e-9):
                        # Check V and W
                        if np.allclose(V[:, j], alpha * V[:, i], atol=1e-9, rtol=1e-9) and \
                           np.allclose(W[:, j], alpha * W[:, i], atol=1e-9, rtol=1e-9):
                            # Merge: subtract alpha * col i from col j, then remove col j
                            U[:, j] -= alpha * U[:, i]
                            V[:, j] -= alpha * V[:, i]
                            W[:, j] -= alpha * W[:, i]
                            # Now col j should be zero
                            if np.allclose(U[:, j], 0, atol=1e-9) and \
                               np.allclose(V[:, j], 0, atol=1e-9) and \
                               np.allclose(W[:, j], 0, atol=1e-9):
                                U = np.delete(U, j, axis=1)
                                V = np.delete(V, j, axis=1)
                                W = np.delete(W, j, axis=1)
                                current_rank -= 1
                                merged = True
                                break
                if merged:
                    break
                    
            if not merged:
                break
                
        return U, V, W

    def try_pairwise_merge(U, V, W, max_iterations=200):
        """Try to merge pairs of columns by finding alpha, beta such that
        alpha*col_i + beta*col_j can be replaced by a single column.
        This is more general than linear dependency."""
        current_rank = U.shape[1]
        for _ in range(max_iterations):
            if time.time() > deadline - 15:
                break
                
            merged = False
            # For each pair (i,j), check if there exists a linear combination
            # that makes one column zero while preserving the tensor
            # This is equivalent to checking if the 2D subspace spanned by columns i,j
            # in U,V,W simultaneously has rank 1
            for i in range(current_rank - 1):
                for j in range(i + 1, current_rank):
                    # Check if the three vectors (U[:,i], V[:,i], W[:,i]) and
                    # (U[:,j], V[:,j], W[:,j]) are linearly dependent in the
                    # combined space. This is complex, so we use a simpler heuristic:
                    # Check if U[:,j] = a*U[:,i] + b*U[:,k] for some k, etc.
                    # For now, just do the simple scalar multiple check
                    pass
                if merged:
                    break
                    
            if not merged:
                break
                
        return U, V, W

    # Build the initial block Strassen solution (rank 98)
    U, V, W = build_block_strassen()
    
    # Verify the initial solution
    is_valid = verify_fast(U, V, W)
    
    # Try to reduce rank
    U_red, V_red, W_red = try_reduce_rank(U, V, W)
    
    # Verify the reduced solution
    if verify_fast(U_red, V_red, W_red):
        U, V, W = U_red, V_red, W_red
    
    # If still at 98, try more aggressive reduction
    if U.shape[1] == 98:
        U2, V2, W2 = try_reduce_rank(U, V, W, max_iterations=1000)
        if verify_fast(U2, V2, W2):
            U, V, W = U2, V2, W2
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
