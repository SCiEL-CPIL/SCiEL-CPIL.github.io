"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    c = i * m * p + j * p + k
                    U[i * m + j, c] = 1
                    V[j * p + k, c] = 1
                    W[k * n + i, c] = 1
        return U, V, W

    def strassen_terms():
        T = []
        T.append(([((0, 0), 1), ((1, 1), 1)], [((0, 0), 1), ((1, 1), 1)],
                 [((0, 0), 1), ((1, 1), 1)]))
        T.append(([((1, 0), 1), ((1, 1), 1)], [((0, 0), 1)],
                 [((1, 0), 1), ((1, 1), -1)]))
        T.append(([((0, 0), 1)], [((0, 1), 1), ((1, 1), -1)],
                 [((0, 1), 1), ((1, 1), 1)]))
        T.append(([((1, 1), 1)], [((1, 0), 1), ((0, 0), -1)],
                 [((1, 0), 1), ((0, 0), 1)]))
        T.append(([((0, 0), 1), ((0, 1), 1)], [((1, 1), 1)],
                 [((0, 0), -1), ((0, 1), 1)]))
        T.append(([((1, 0), 1), ((0, 0), -1)], [((0, 0), 1), ((0, 1), 1)],
                 [((1, 1), 1)]))
        T.append(([((0, 1), 1), ((1, 1), -1)], [((1, 0), 1), ((1, 1), 1)],
                 [((0, 0), 1)]))
        return T

    ST = strassen_terms()
    total_rank = 44
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)

    def add_rank7(i0, j0, k0, col_start):
        for r in range(7):
            u_e, v_e, w_e = ST[r]
            c = col_start + r
            for (li, lj), val in u_e:
                U[(i0 + li) * m + (j0 + lj), c] += val
            for (lj, lk), val in v_e:
                V[(j0 + lj) * p + (k0 + lk), c] += val
            for (li, lk), val in w_e:
                W[(k0 + lk) * n + (i0 + li), c] += val
        return col_start + 7

    def add_rank4(i0, j0, k0, col_start):
        for t in range(4):
            c = col_start + t
            j = j0 + (t % 2)
            k = k0 + (t // 2)
            U[i0 * m + j, c] = 1
            V[j * p + k, c] = 1
            W[k * n + i0, c] = 1
        return col_start + 4

    col = 0
    col = add_rank7(0, 0, 0, col)
    col = add_rank7(0, 0, 2, col)
    col = add_rank7(0, 2, 0, col)
    col = add_rank7(0, 2, 2, col)
    col = add_rank4(2, 0, 0, col)
    col = add_rank4(2, 0, 2, col)
    col = add_rank4(2, 2, 0, col)
    col = add_rank4(2, 2, 2, col)

    assert col == total_rank

    # Verify the decomposition with vectorized computation
    def verify(U_mat, V_mat, W_mat, rank):
        np.random.seed(42)
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        # Vectorized computation
        A_flat = A.flatten()  # size n*m
        B_flat = B.flatten()  # size m*p
        
        # Compute ar vector: ar[r] = sum_{i,j} U[i*m+j, r] * A[i,j]
        ar = U_mat @ A_flat  # size rank
        br = V_mat @ B_flat  # size rank
        
        # Compute C_comp: C[i,k] = sum_r W[k*n+i, r] * ar[r] * br[r]
        # W has shape (p*n, rank), where index is k*n+i
        # C_comp has shape (n, p)
        # C_comp[i,k] = sum_r W[k*n+i, r] * ar[r] * br[r]
        # Let's reshape W to (p, n, rank) where W2[k, i, r] = W[k*n+i, r]
        W2 = W_mat.reshape(p, n, rank)
        # C_comp[i, k] = sum_r W2[k, i, r] * ar[r] * br[r]
        # = sum_r (ar[r] * br[r]) * W2[k, i, r]
        arb = ar * br  # size rank
        # C_comp[i, k] = sum_r W2[k, i, r] * arb[r]
        # This is W2.T (shape n, p, rank) @ arb, then we need to handle the axis
        # Actually: C_comp[i, k] = sum_r W2[k, i, r] * arb[r]
        # Let W3 = W2.transpose(1, 0, 2)  # shape (n, p, rank) where W3[i, k, r] = W2[k, i, r] = W_mat[k*n+i, r]
        W3 = W2.transpose(1, 0, 2)  # shape (n, p, rank)
        C_comp = np.tensordot(W3, arb, axes=([2], [0]))  # shape (n, p)
        
        max_err = float(np.max(np.abs(C_comp - C_expected)))
        return max_err

    max_err = verify(U, V, W, total_rank)
    if max_err > 1e-9:
        U_sb, V_sb, W_sb = schoolbook()
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

    if time.time() > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
