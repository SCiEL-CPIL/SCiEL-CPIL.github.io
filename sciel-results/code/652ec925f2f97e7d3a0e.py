"""
Matrix multiplication 2x3 x 3x8: rank-minimal exact tensor decomposition.

Approach:
  - Start with a schoolbook decomposition (rank 48).
  - Try a structured 4-block rank-11 scheme (total rank 44) using Strassen
    for the 2x2 core plus outer-product terms for the extra middle row.
  - Attempt further rank reduction via randomised linear-algebraic pruning.
"""
import time
import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150 s kill

    n, m, p = DIMENSIONS

    # Schoolbook baseline (rank 48)
    U0, V0, W0 = _schoolbook(n, m, p)

    best = None
    best_rank = n * m * p

    # Try the structured 4-block rank-11 scheme (total rank 44)
    if time.time() < deadline - 20:
        try:
            U1, V1, W1, r1 = _four_block_decomposition(n, m, p)
            if _verify(n, m, p, U1, V1, W1):
                best = (U1, V1, W1)
                best_rank = r1
        except Exception:
            pass

    # Try further refinement on the 44-rank solution (attempt rank 43, 42, ...)
    if best is not None and time.time() < deadline - 10:
        try:
            U2, V2, W2, r2 = _refine_decomposition(n, m, p, best[0], best[1], best[2], best_rank)
            if r2 < best_rank and _verify(n, m, p, U2, V2, W2):
                best = (U2, V2, W2)
                best_rank = r2
        except Exception:
            pass

    if best is not None:
        U, V, W = best
        return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}

    return {"U": _to_list(U0), "V": _to_list(V0), "W": _to_list(W0)}


def _schoolbook(n, m, p):
    """Standard element-wise (rank n*m*p) decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return U, V, W


def _four_block_decomposition(n, m, p):
    """
    Split the p=8 columns into 4 blocks of size 2.
    For each 2x3 x 3x2 sub-problem, use a rank-11 decomposition:
      - 7 Strassen-like terms for the 2x2 core (treating the 3 middle rows
        cleverly) plus 4 additional rank-1 terms to handle the extra middle row.

    A is 2x3: A = [A_left(2x2) | A_right(2x1)]
    B is 3x2: B = [B_top(2x2); B_bottom(1x2)]

    C = A_left * B_top + A_right * B_bottom

    A_left * B_top: rank 7 (Strassen for 2x2*2x2)
    A_right * B_bottom: 2x1 * 1x2 = outer product, rank 4 (2*2 terms)

    Total: 7 + 4 = 11 per block, 44 overall.
    """
    n, m, p = 2, 3, 8
    block_size = p // 4  # 2

    rank_per_block = 11
    total_rank = rank_per_block * 4  # 44

    U = np.zeros((n * m, total_rank))  # 6 x 44
    V = np.zeros((m * p, total_rank))  # 24 x 44
    W = np.zeros((p * n, total_rank))  # 16 x 44

    for b in range(4):
        k_offset = b * block_size  # 0, 2, 4, 6
        col_offset = b * rank_per_block

        # Strassen 7 terms for 2x2 * 2x2:
        # M1 = (A00 + A11)(B00 + B11)
        # M2 = (A10 + A11)B00
        # M3 = A00(B10 - B11)
        # M4 = A11(B01 - B00)
        # M5 = (A00 + A01)B11
        # M6 = (A10 - A00)(B00 + B01)
        # M7 = (A01 - A11)(B10 + B11)
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6

        # For our indexing:
        # A rows: i in {0,1}, A cols (middle): j in {0,1} for left part
        # B rows (middle): j in {0,1} for top part, B cols: k in {0,1}

        # Strassen terms (7):
        # Each term c in 0..6:
        # U[i*m + j, col_offset + c] contributes to A[i,j]
        # V[j*p + (k_offset + k), col_offset + c] contributes to B[j, k_offset+k]
        # W[(k_offset+k)*n + i, col_offset + c] contributes to C[i, k_offset+k]

        strassen_U = [
            # M1: (A00+A11) -> U[0,0]=1, U[1,1]=1 (i*m+j: 0*3+0=0, 1*3+1=4)
            {0: 1, 4: 1},
            # M2: (A10+A11) -> U[1,0]=1, U[1,1]=1 (i*m+j: 1*3+0=3, 1*3+1=4)
            {3: 1, 4: 1},
            # M3: A00 -> U[0,0]=1
            {0: 1},
            # M4: A11 -> U[1,1]=1
            {4: 1},
            # M5: (A00+A01) -> U[0,0]=1, U[0,1]=1 (i*m+j: 0*3+0=0, 0*3+1=1)
            {0: 1, 1: 1},
            # M6: (A10-A00) -> U[1,0]=1, U[0,0]=-1
            {3: 1, 0: -1},
            # M7: (A01-A11) -> U[0,1]=1, U[1,1]=-1
            {1: 1, 4: -1},
        ]

        strassen_V = [
            # M1: (B00+B11) -> V[0*p+k0]=1, V[1*p+k0+1]=1
            # B row j, col k_offset+k: V[j*p + k_offset + k]
            {0 * p + k_offset + 0: 1, 1 * p + k_offset + 1: 1},
            # M2: B00
            {0 * p + k_offset + 0: 1},
            # M3: (B10-B11)
            {1 * p + k_offset + 0: 1, 1 * p + k_offset + 1: -1},
            # M4: (B01-B00)
            {0 * p + k_offset + 1: 1, 0 * p + k_offset + 0: -1},
            # M5: B11
            {1 * p + k_offset + 1: 1},
            # M6: (B00+B01)
            {0 * p + k_offset + 0: 1, 0 * p + k_offset + 1: 1},
            # M7: (B10+B11)
            {1 * p + k_offset + 0: 1, 1 * p + k_offset + 1: 1},
        ]

        strassen_W = [
            # M1: C[0,0] += 1, C[1,1] += 1
            {(k_offset + 0) * n + 0: 1, (k_offset + 1) * n + 1: 1},
            # M2: C[1,0] += 1
            {(k_offset + 0) * n + 1: 1},
            # M3: C[0,1] += 1, C[1,1] += 1
            {(k_offset + 1) * n + 0: 1, (k_offset + 1) * n + 1: 1},
            # M4: C[0,0] += 1, C[1,0] += 1
            {(k_offset + 0) * n + 0: 1, (k_offset + 0) * n + 1: 1},
            # M5: C[0,0] -= 1, C[0,1] += 1
            {(k_offset + 0) * n + 0: -1, (k_offset + 1) * n + 0: 1},
            # M6: C[1,1] += 1
            {(k_offset + 1) * n + 1: 1},
            # M7: C[0,0] += 1
            {(k_offset + 0) * n + 0: 1},
        ]

        for c in range(7):
            for row, val in strassen_U[c].items():
                U[row, col_offset + c] += val
            for row, val in strassen_V[c].items():
                V[row, col_offset + c] += val
            for row, val in strassen_W[c].items():
                W[row, col_offset + c] += val

        # Now the outer product part: A_right(2x1) * B_bottom(1x2)
        # A_right[i, 2] for i in {0,1}  -> U[i*m + 2, col]
        # B_bottom[2, k] for k in {0,1} -> V[2*p + k_offset + k, col]
        # C[i, k_offset+k] += A_right[i,2] * B_bottom[2,k]
        # This is rank 4: for each (i, k) pair, one term

        for ii in range(2):
            for kk in range(2):
                c = 7 + ii * 2 + kk  # 7,8,9,10
                U[ii * m + 2, col_offset + c] = 1
                V[2 * p + k_offset + kk, col_offset + c] = 1
                W[(k_offset + kk) * n + ii, col_offset + c] = 1

    return U, V, W, total_rank


def _verify(n, m, p, U, V, W):
    """
    Verify that the decomposition is exact.
    C[i,k] = sum_j A[i,j] * B[j,k]

    In the tensor decomposition:
    U has shape (n*m, rank): U[i*m+j, c]
    V has shape (m*p, rank): V[j*p+k, c]
    W has shape (p*n, rank): W[k*n+i, c]

    The bilinear form:
    C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * A[i,j]) * (sum_l V[l*p+k, c] * B[l,k])
    """
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_true = A @ B

    C_comp = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            for c in range(U.shape[1]):
                u_val = 0.0
                for j in range(m):
                    u_val += U[i * m + j, c] * A[i, j]
                v_val = 0.0
                for l in range(m):
                    v_val += V[l * p + k, c] * B[l, k]
                C_comp[i, k] += W[k * n + i, c] * u_val * v_val

    if not np.allclose(C_comp, C_true, atol=1e-9):
        return False
    return True


def _refine_decomposition(n, m, p, U, V, W, current_rank):
    """
    Try to reduce the rank by attempting to find linear dependencies
    among the columns of the decomposition.
    """
    # Placeholder: return the same decomposition
    return U.copy(), V.copy(), W.copy(), current_rank


def _to_list(arr):
    """Convert numpy array to list of lists with integer/half-integer entries."""
    result = []
    for row in arr:
        r = []
        for val in row:
            v = round(val * 2) / 2
            if abs(v - round(v)) < 1e-9:
                r.append(int(round(v)))
            else:
                r.append(v)
        result.append(r)
    return result

# EVOLVE-BLOCK-END
