"""Matrix multiplication 2x4 x 4x4 tensor decomposition (rank 28 via Strassen on 2x2 blocks)."""

import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START

def _build_strassen_rank28():
    """Build the rank-28 Strassen decomposition deterministically.
    
    A (2x4) = [A1 | A2] where A1, A2 are 2x2
    B (4x4) = [[B11, B12], [B21, B22]] where each Bi is 2x2
    C (2x4) = [C1 | C2] where C1 = A1*B11 + A2*B21, C2 = A1*B12 + A2*B22
    
    Each 2x2*2x2 product uses Strassen (rank 7), so total rank = 4*7 = 28.
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    rank = 28
    U = np.zeros((n * m, rank), dtype=float)  # 8 x 28
    V = np.zeros((m * p, rank), dtype=float)  # 16 x 28
    W = np.zeros((p * n, rank), dtype=float)  # 8 x 28

    # Strassen 2x2x2 decomposition:
    # m1 = (a11+a22)(b11+b22)
    # m2 = (a21+a22)b11
    # m3 = a11(b12-b22)
    # m4 = a22(b21-b11)
    # m5 = (a11+a12)b22
    # m6 = (a21-a11)(b11+b12)
    # m7 = (a12-a22)(b21+b22)
    # c11 = m1+m4-m5+m7
    # c12 = m3+m5
    # c21 = m2+m4
    # c22 = m1-m2+m3+m6
    
    col = 0
    for b_col_blk in range(2):  # B column block: 0 -> cols 0,1; 1 -> cols 2,3
        for a_blk in range(2):  # A column block / B row block: 0 -> cols/rows 0,1; 1 -> cols/rows 2,3
            for s in range(7):
                c = col
                j0 = 2 * a_blk
                j1 = 2 * a_blk + 1
                k0 = 2 * b_col_blk
                k1 = 2 * b_col_blk + 1
                
                # U contributions: A[i, j] where i in {0,1}, j in {j0, j1}
                # U index: i * 4 + j
                U_contrib = {
                    0: [(0, j0, 1), (1, j1, 1)],       # a11+a22
                    1: [(1, j0, 1), (1, j1, 1)],       # a21+a22
                    2: [(0, j0, 1)],                    # a11
                    3: [(1, j1, 1)],                    # a22
                    4: [(0, j0, 1), (0, j1, 1)],       # a11+a12
                    5: [(1, j0, 1), (0, j0, -1)],      # a21-a11
                    6: [(0, j1, 1), (1, j1, -1)],      # a12-a22
                }
                for (i, j, coeff) in U_contrib[s]:
                    U[i * 4 + j, c] += coeff

                # V contributions: B[j, k] where j in {j0, j1}, k in {k0, k1}
                # V index: j * 4 + k
                V_contrib = {
                    0: [(j0, k0, 1), (j1, k1, 1)],     # b11+b22
                    1: [(j0, k0, 1)],                   # b11
                    2: [(j0, k1, 1), (j1, k1, -1)],    # b12-b22
                    3: [(j1, k0, 1), (j0, k0, -1)],    # b21-b11
                    4: [(j1, k1, 1)],                   # b22
                    5: [(j0, k0, 1), (j0, k1, 1)],     # b11+b12
                    6: [(j1, k0, 1), (j1, k1, 1)],     # b21+b22
                }
                for (j, k, coeff) in V_contrib[s]:
                    V[j * 4 + k, c] += coeff

                # W contributions: C[i, k] where i in {0,1}, k in {k0, k1}
                # W index: k * 2 + i (trace convention W[k*n+i, r])
                W_contrib = {
                    0: {(0, k0): 1, (1, k1): 1},       # c11 gets m1, c22 gets m1
                    1: {(1, k0): 1, (1, k1): -1},      # c21 gets m2, c22 gets -m2
                    2: {(0, k1): 1, (1, k1): 1},       # c12 gets m3, c22 gets m3
                    3: {(0, k0): 1, (1, k0): 1},       # c11 gets m4, c21 gets m4
                    4: {(0, k0): -1, (0, k1): 1},      # c11 gets -m5, c12 gets m5
                    5: {(1, k1): 1},                   # c22 gets m6
                    6: {(0, k0): 1},                   # c11 gets m7
                }
                for (i_loc, k_loc), coeff in W_contrib[s].items():
                    W[k_loc * 2 + i_loc, c] += coeff

                col += 1

    return U, V, W


def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Check if U, V, W correctly decompose the matmul tensor.
    
    T[i,k,j] = sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r] should equal 1 for all (i,k,j).
    """
    rank = U.shape[1]
    for i in range(n):
        for k in range(m):
            for j in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
                if abs(val - 1.0) > tol:
                    return False
    return True


def _try_reduce_rank(U, V, W, n, m, p, tol=1e-10):
    """Try to reduce rank by checking if any column can be removed.
    
    For each column r, check if the rank-1 tensor formed by U[:,r], V[:,r], W[:,r]
    is linearly dependent on the others. If so, we can remove it.
    """
    rank = U.shape[1]
    n_rows_u = U.shape[0]
    n_rows_v = V.shape[0]
    n_rows_w = W.shape[0]
    
    # For each candidate column to remove, check if the remaining columns
    # can still span the same tensor.
    for r_remove in range(rank):
        # Keep all columns except r_remove
        cols_keep = [c for c in range(rank) if c != r_remove]
        U_keep = U[:, cols_keep]
        V_keep = V[:, cols_keep]
        W_keep = W[:, cols_keep]
        
        # Verify the reduced decomposition
        if _verify_decomposition(U_keep, V_keep, W_keep, n, m, p, tol):
            return U_keep, V_keep, W_keep
    
    return U, V, W


def construct():
    """
    Return a rank-28 exact decomposition of the 2x4 * 4x4 matrix multiplication tensor.
    Uses Strassen's algorithm on 2x2 blocks.
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    start_time = time.time()
    deadline = start_time + 110  # 110s limit, keep margin before 150s
    
    # Build the base rank-28 Strassen decomposition
    U, V, W = _build_strassen_rank28()
    
    # Verify it's correct
    assert _verify_decomposition(U, V, W, n, m, p), "Base rank-28 decomposition failed verification"
    
    # Try to reduce rank
    U, V, W = _try_reduce_rank(U, V, W, n, m, p)
    
    # Verify final result
    assert _verify_decomposition(U, V, W, n, m, p), "Final decomposition failed verification"
    
    # Return the result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
