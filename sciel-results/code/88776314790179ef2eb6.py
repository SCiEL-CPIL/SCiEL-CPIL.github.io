import numpy as np
import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n

    # Schoolbook 3x3 decomposition (rank 27) as base kernel
    def get_3x3_schoolbook():
        n3 = 3
        rank3 = 27
        U3 = np.zeros((9, rank3))
        V3 = np.zeros((9, rank3))
        W3 = np.zeros((9, rank3))
        for i in range(n3):
            for j in range(n3):
                for k in range(n3):
                    col = i * 9 + j * 3 + k
                    U3[i * 3 + j, col] = 1.0
                    V3[j * 3 + k, col] = 1.0
                    W3[k * 3 + i, col] = 1.0
        return U3, V3, W3

    U3, V3, W3 = get_3x3_schoolbook()
    r3 = 27

    # 2x2 Strassen with 3x3 blocks: total rank = 7 * 27 = 189
    total_rank = 7 * r3
    U6 = np.zeros((36, total_rank))
    V6 = np.zeros((36, total_rank))
    W6 = np.zeros((36, total_rank))

    def add_strassen_term(start_col, a_blocks, b_blocks, c_blocks):
        """Add one Strassen product term to the global factor matrices."""
        for r in range(r3):
            u_mat = U3[:, r].reshape(3, 3)
            v_mat = V3[:, r].reshape(3, 3)
            w_mat = W3[:, r].reshape(3, 3)

            # U part: A-side
            for bi, coeff in a_blocks:
                rb, cb = bi // 2, bi % 2
                for i in range(3):
                    for j in range(3):
                        val = coeff * u_mat[i, j]
                        if val != 0:
                            gi, gj = rb * 3 + i, cb * 3 + j
                            U6[gi * 6 + gj, start_col + r] += val

            # V part: B-side
            for bi, coeff in b_blocks:
                rb, cb = bi // 2, bi % 2
                for j in range(3):
                    for k in range(3):
                        val = coeff * v_mat[j, k]
                        if val != 0:
                            gj, gk = rb * 3 + j, cb * 3 + k
                            V6[gj * 6 + gk, start_col + r] += val

            # W part: C-side (index k*n + i)
            for bi, coeff in c_blocks:
                rb, cb = bi // 2, bi % 2
                for k in range(3):
                    for i in range(3):
                        val = coeff * w_mat[k, i]
                        if val != 0:
                            gk, gi = cb * 3 + k, rb * 3 + i
                            W6[gk * 6 + gi, start_col + r] += val

    # Strassen formulas for 2x2 blocks: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # P1 = (A11 + A22) * (B11 + B22) -> C11, C22
    add_strassen_term(0,
                      a_blocks=[(0, 1), (3, 1)],
                      b_blocks=[(0, 1), (3, 1)],
                      c_blocks=[(0, 1), (3, 1)])

    # P2 = (A21 + A22) * B11 -> C21 (+1), C22 (-1)
    add_strassen_term(r3,
                      a_blocks=[(2, 1), (3, 1)],
                      b_blocks=[(0, 1)],
                      c_blocks=[(2, 1), (3, -1)])

    # P3 = A11 * (B12 - B22) -> C12 (+1), C22 (+1)
    add_strassen_term(2 * r3,
                      a_blocks=[(0, 1)],
                      b_blocks=[(1, 1), (3, -1)],
                      c_blocks=[(1, 1), (3, 1)])

    # P4 = A22 * (B21 - B11) -> C11 (+1), C21 (+1)
    add_strassen_term(3 * r3,
                      a_blocks=[(3, 1)],
                      b_blocks=[(2, 1), (0, -1)],
                      c_blocks=[(0, 1), (2, 1)])

    # P5 = (A11 + A12) * B22 -> C12 (+1), C11 (-1)
    add_strassen_term(4 * r3,
                      a_blocks=[(0, 1), (1, 1)],
                      b_blocks=[(3, 1)],
                      c_blocks=[(1, 1), (0, -1)])

    # P6 = (A21 - A11) * (B11 + B12) -> C22 (+1)
    add_strassen_term(5 * r3,
                      a_blocks=[(2, 1), (0, -1)],
                      b_blocks=[(0, 1), (1, 1)],
                      c_blocks=[(3, 1)])

    # P7 = (A12 - A22) * (B21 + B22) -> C11 (+1)
    add_strassen_term(6 * r3,
                      a_blocks=[(1, 1), (3, -1)],
                      b_blocks=[(2, 1), (3, 1)],
                      c_blocks=[(0, 1)])

    # Verify decomposition using vectorized matrix multiplication
    if time.time() < deadline:
        try:
            U3g = U6.reshape(6, 6, -1)  # U3g[i,j,r] = U6[i*6+j, r]
            V3g = V6.reshape(6, 6, -1)  # V3g[j,k,r] = V6[j*6+k, r]
            W3g = W6.reshape(6, 6, -1)  # W3g[k,i,r] = W6[k*6+i, r]
            
            # T[i,j,k] = sum_r U3g[i,j,r] * V3g[j,k,r] * W3g[k,i,r]
            # Compute via: for each r, outer product of U3g[:,:,r] (6x6) with W3g[:,:,r].T (6x6)
            # Actually: T[i,j,k] = sum_r U3g[i,j,r] * (V3g[j,k,r] * W3g[k,i,r])
            # Let M_r[j,k,i] = V3g[j,k,r] * W3g[k,i,r], then T[i,j,k] = sum_r U3g[i,j,r] * M_r[j,k,i]
            # Equivalently: T = sum_r U3g[:,:,r] ⊙ (V3g[:,:,r] * W3g[:,:,r].transpose(0,2,1))
            # But easier: use einsum or batched matmul
            # T[i,j,k] = sum_r U3g[i,j,r] * V3g[j,k,r] * W3g[k,i,r]
            # = sum_r (U3g[i,j,r] * W3g[k,i,r]) * V3g[j,k,r]
            # Let A_r[i,j,k] = U3g[i,j,r] * W3g[k,i,r], then T[i,j,k] = sum_r A_r[i,j,k] * V3g[j,k,r]
            # This is a batched contraction. Use einsum for clarity:
            reconstructed = np.einsum('ijr,jkr,kir->ijk', U3g, V3g, W3g)
            if np.allclose(reconstructed, np.ones((6, 6, 6)), atol=1e-9):
                # Attempt rank reduction: find linearly dependent rank terms
                if time.time() < deadline - 20.0:
                    try:
                        # Check if any rank term is a linear combination of others
                        # We can try to identify and remove redundant terms
                        # For each term r, check if removing it still gives correct result
                        # This is expensive, so only try a heuristic approach
                        # Check rank of U6, V6, W6 column spaces
                        rank_U = np.linalg.matrix_rank(U6, tol=1e-10)
                        rank_V = np.linalg.matrix_rank(V6, tol=1e-10)
                        rank_W = np.linalg.matrix_rank(W6, tol=1e-10)
                        min_rank = min(rank_U, rank_V, rank_W)
                        
                        if min_rank < total_rank:
                            # There's linear dependence; try to find minimal rank subset
                            # Use greedy approach: try removing terms one by one
                            # This is O(n^2) which is 189^2 ~ 35721, feasible
                            keep = list(range(total_rank))
                            for r in range(total_rank):
                                if r not in keep:
                                    continue
                                if len(keep) <= min_rank:
                                    break
                                # Try removing term r
                                new_keep = [k for k in keep if k != r]
                                if len(new_keep) == 0:
                                    continue
                                U_test = U6[:, new_keep]
                                V_test = V6[:, new_keep]
                                W_test = W6[:, new_keep]
                                U3t = U_test.reshape(6, 6, -1)
                                V3t = V_test.reshape(6, 6, -1)
                                W3t = W_test.reshape(6, 6, -1)
                                rec_test = np.einsum('ijr,jkr,kir->ijk', U3t, V3t, W3t)
                                if np.allclose(rec_test, np.ones((6, 6, 6)), atol=1e-9):
                                    keep = new_keep
                            
                            if len(keep) < total_rank:
                                U_final = U6[:, keep]
                                V_final = V6[:, keep]
                                W_final = W6[:, keep]
                                return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
                    except Exception:
                        pass
                return {"U": U6.tolist(), "V": V6.tolist(), "W": W6.tolist()}
            else:
                max_err = np.max(np.abs(reconstructed - 1.0))
                if max_err < 1e-6:
                    return {"U": U6.tolist(), "V": V6.tolist(), "W": W6.tolist()}
        except Exception:
            pass

    # Fallback: schoolbook rank 216
    rank = 216
    U_f = np.zeros((36, rank))
    V_f = np.zeros((36, rank))
    W_f = np.zeros((36, rank))
    for i in range(6):
        for j in range(6):
            for k in range(6):
                col = i * 36 + j * 6 + k
                U_f[i * 6 + j, col] = 1.0
                V_f[j * 6 + k, col] = 1.0
                W_f[k * 6 + i, col] = 1.0
    return {"U": U_f.tolist(), "V": V_f.tolist(), "W": W_f.tolist()}

# EVOLVE-BLOCK-END
