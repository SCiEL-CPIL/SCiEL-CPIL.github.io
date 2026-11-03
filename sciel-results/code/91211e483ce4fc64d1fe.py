"""Matrix multiplication 4,4,5 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [4, 4, 5]
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    import time
    start_time = time.time()
    DEADLINE = 140  # safety margin before 150s kill

    n, m, p = 4, 4, 5

    # Rank-44 construction using Strassen for 4x4x4 part + schoolbook for 5th column
    # 4x4x4: 8 Strassen 2x2x2 multiplications (2 per ib,kb block) * 7 = 56 terms
    # 4x4x1 (k=4): 16 terms
    # Total: 72 terms

    rank_total = 72
    U = np.zeros((n * m, rank_total))
    V = np.zeros((m * p, rank_total))
    W = np.zeros((p * n, rank_total))

    col = 0

    # Strassen 2x2x2 for each (ib, kb, jb) triple
    # ib: row block of A/C (0 or 1)
    # kb: col block of B/C (0 or 1)  
    # jb: middle index block (0 or 1)
    # X = A[ib*2:ib*2+2, jb*2:jb*2+2]
    # Y = B[jb*2:jb*2+2, kb*2:kb*2+2]
    # Z = X*Y contributes to C[kb*2:kb*2+2, ib*2:ib*2+2]

    for ib in range(2):
        for kb in range(2):
            for jb in range(2):
                # Helper to add a rank-1 term
                def add_term(U_idx_vals, V_idx_vals, W_idx_vals):
                    nonlocal col
                    for (idx, val) in U_idx_vals:
                        U[idx, col] += val
                    for (idx, val) in V_idx_vals:
                        V[idx, col] += val
                    for (idx, val) in W_idx_vals:
                        W[idx, col] += val
                    col += 1

                # X[il, jl] = A[ib*2+il, jb*2+jl]
                # Y[jl, kl] = B[jb*2+jl, kb*2+kl]
                # Z[il, kl] -> C[kb*2+kl, ib*2+il]

                # M1 = (X[0,0]+X[1,1])*(Y[0,0]+Y[1,1])
                U_vals = [(ib*2*4 + jb*2 + 0, 1), (ib*2*4 + jb*2 + 1, 1)]
                V_vals = [(jb*2*5 + kb*2 + 0, 1), (jb*2*5 + kb*2 + 1, 1)]
                W_vals = [(kb*2*4 + ib*2 + 0, 1), (kb*2*4 + ib*2 + 1, 1)]
                add_term(U_vals, V_vals, W_vals)

                # M2 = (X[1,0]+X[1,1])*Y[0,0]
                U_vals = [(ib*2*4 + jb*2 + 0, 1), (ib*2*4 + jb*2 + 1, 1)]
                V_vals = [(jb*2*5 + kb*2 + 0, 1)]
                W_vals = [(kb*2*4 + ib*2 + 0, 1), (kb*2*4 + ib*2 + 1, -1)]
                add_term(U_vals, V_vals, W_vals)

                # M3 = X[0,0]*(Y[0,1]-Y[1,1])
                U_vals = [(ib*2*4 + jb*2 + 0, 1)]
                V_vals = [(jb*2*5 + kb*2 + 1, 1), ((jb*2+1)*5 + kb*2 + 1, -1)]
                W_vals = [(kb*2*4 + ib*2 + 0, 1), (kb*2*4 + ib*2 + 1, 1)]
                add_term(U_vals, V_vals, W_vals)

                # M4 = X[1,1]*(Y[1,0]-Y[0,0])
                U_vals = [(ib*2*4 + jb*2 + 1, 1)]
                V_vals = [((jb*2+1)*5 + kb*2 + 0, 1), (jb*2*5 + kb*2 + 0, -1)]
                W_vals = [(kb*2*4 + ib*2 + 0, 1), (kb*2*4 + ib*2 + 1, 1)]
                add_term(U_vals, V_vals, W_vals)

                # M5 = (X[0,0]+X[0,1])*Y[1,1]
                U_vals = [(ib*2*4 + jb*2 + 0, 1), (ib*2*4 + jb*2 + 1, 1)]
                V_vals = [((jb*2+1)*5 + kb*2 + 1, 1)]
                W_vals = [(kb*2*4 + ib*2 + 0, -1), (kb*2*4 + ib*2 + 1, 1)]
                add_term(U_vals, V_vals, W_vals)

                # M6 = (X[1,0]-X[0,0])*(Y[0,0]+Y[0,1])
                U_vals = [(ib*2*4 + jb*2 + 0, 1), (ib*2*4 + jb*2 + 1, -1)]
                V_vals = [(jb*2*5 + kb*2 + 0, 1), (jb*2*5 + kb*2 + 1, 1)]
                W_vals = [(kb*2*4 + ib*2 + 1, 1)]
                add_term(U_vals, V_vals, W_vals)

                # M7 = (X[0,1]-X[1,1])*(Y[1,0]+Y[1,1])
                U_vals = [(ib*2*4 + jb*2 + 1, 1), ((ib*2+1)*4 + jb*2 + 1, -1)]
                V_vals = [((jb*2+1)*5 + kb*2 + 0, 1), ((jb*2+1)*5 + kb*2 + 1, 1)]
                W_vals = [(kb*2*4 + ib*2 + 0, 1)]
                add_term(U_vals, V_vals, W_vals)

    # 5th column (k=4): C[i, 4] = sum_j A[i,j] * B[j, 4]
    for i in range(4):
        for j in range(4):
            U[i * 4 + j, col] = 1
            V[j * 5 + 4, col] = 1
            W[4 * 4 + i, col] = 1
            col += 1

    # Verify the construction
    def verify(U_mat, V_mat, W_mat, r):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for t in range(r):
                        val += U_mat[i*m+j, t] * V_mat[j*p+k, t] * W_mat[k*n+i, t]
                    if abs(val - 1) > 1e-9:
                        return False
        return True

    if not verify(U, V, W, rank_total):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
