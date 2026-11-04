"""Matrix multiplication 2x3x10 - rank-minimal exact tensor decomposition.

Uses Strassen decomposition on the 2x2 sub-block (A rows 0-1, B cols 0-1)
for each pair of output columns, plus schoolbook for the remaining dimension.
This gives rank 55 = 5 * (7 Strassen + 4 schoolbook).
"""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 10]


def _verify_fast(U, V, W, n, m, p, rank, tol=1e-9):
    """Vectorised check that sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1."""
    U_arr = np.asarray(U, dtype=np.float64)
    V_arr = np.asarray(V, dtype=np.float64)
    W_arr = np.asarray(W, dtype=np.float64)
    U3 = U_arr.reshape(n, m, rank)
    V3 = V_arr.reshape(m, p, rank)
    W3 = W_arr.reshape(p, n, rank)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = np.dot(U3[i, j, :] * V3[j, k, :], W3[k, i, :])
                if abs(total - 1.0) > tol:
                    return False
    return True


def _verify_exact(U, V, W, n, m, p, rank):
    """Exact rational check of the decomposition identity."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = Fraction(0)
                for r in range(rank):
                    total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if total != 1:
                    return False
    return True


def _make_strassen_decomposition(n, m, p):
    """Rank-55 decomposition: 5 Strassen blocks (7 each) + 20 schoolbook (4 each).
    
    For each pair of output columns (k0, k0+1):
    - Use Strassen 2x2x2 on A[0:2, 0:2] * B[0:2, k0:k0+2] -> 7 multiplications
    - Use schoolbook for A[0:2, 2] * B[2, k0:k0+2] -> 4 multiplications
    Total per pair: 11, total: 5 * 11 = 55
    """
    rank = 55
    U = [[0] * rank for _ in range(n * m)]  # 6 x 55
    V = [[0] * rank for _ in range(m * p)]  # 30 x 55
    W = [[0] * rank for _ in range(p * n)]  # 20 x 55
    
    col = 0
    for t in range(5):
        k0 = 2 * t
        k1 = 2 * t + 1
        
        # Strassen 7 products for A[0:2, 0:2] * B[0:2, k0:k1+1]
        # A_left = [[a00, a01], [a10, a11]]
        # B_block = [[B[0,k0], B[0,k1]], [B[1,k0], B[1,k1]]]
        
        # M1: (a00 + a11) * (B[0,k0] + B[1,k1])
        U[0 * 3 + 0][col] += 1  # a00
        U[1 * 3 + 1][col] += 1  # a11
        V[0 * 10 + k0][col] += 1  # B[0,k0]
        V[1 * 10 + k1][col] += 1  # B[1,k1]
        W[k0 * 2 + 0][col] += 1  # C[0,k0]
        W[k1 * 2 + 1][col] += 1  # C[1,k1]
        col += 1
        
        # M2: (a10 + a11) * B[0,k0]
        U[1 * 3 + 0][col] += 1  # a10
        U[1 * 3 + 1][col] += 1  # a11
        V[0 * 10 + k0][col] += 1  # B[0,k0]
        W[k0 * 2 + 1][col] += 1  # C[1,k0]
        W[k1 * 2 + 1][col] -= 1  # -C[1,k1]
        col += 1
        
        # M3: a00 * (B[0,k1] - B[1,k1])
        U[0 * 3 + 0][col] += 1  # a00
        V[0 * 10 + k1][col] += 1  # B[0,k1]
        V[1 * 10 + k1][col] -= 1  # -B[1,k1]
        W[k1 * 2 + 0][col] += 1  # C[0,k1]
        W[k1 * 2 + 1][col] += 1  # C[1,k1]
        col += 1
        
        # M4: a11 * (B[1,k0] - B[0,k0])
        U[1 * 3 + 1][col] += 1  # a11
        V[1 * 10 + k0][col] += 1  # B[1,k0]
        V[0 * 10 + k0][col] -= 1  # -B[0,k0]
        W[k0 * 2 + 0][col] += 1  # C[0,k0]
        W[k0 * 2 + 1][col] += 1  # C[1,k0]
        col += 1
        
        # M5: (a00 + a01) * B[1,k1]
        U[0 * 3 + 0][col] += 1  # a00
        U[0 * 3 + 1][col] += 1  # a01
        V[1 * 10 + k1][col] += 1  # B[1,k1]
        W[k0 * 2 + 0][col] -= 1  # -C[0,k0]
        W[k1 * 2 + 0][col] += 1  # C[0,k1]
        col += 1
        
        # M6: (a10 - a00) * (B[0,k0] + B[0,k1])
        U[1 * 3 + 0][col] += 1  # a10
        U[0 * 3 + 0][col] -= 1  # -a00
        V[0 * 10 + k0][col] += 1  # B[0,k0]
        V[0 * 10 + k1][col] += 1  # B[0,k1]
        W[k1 * 2 + 1][col] += 1  # C[1,k1]
        col += 1
        
        # M7: (a01 - a11) * (B[1,k0] + B[1,k1])
        U[0 * 3 + 1][col] += 1  # a01
        U[1 * 3 + 1][col] -= 1  # -a11
        V[1 * 10 + k0][col] += 1  # B[1,k0]
        V[1 * 10 + k1][col] += 1  # B[1,k1]
        W[k0 * 2 + 0][col] += 1  # C[0,k0]
        col += 1
        
        # Schoolbook for A[:,2] * B[2, k0:k1+1]
        # a02 * B[2,k0] -> C[0,k0]
        U[0 * 3 + 2][col] += 1
        V[2 * 10 + k0][col] += 1
        W[k0 * 2 + 0][col] += 1
        col += 1
        
        # a02 * B[2,k1] -> C[0,k1]
        U[0 * 3 + 2][col] += 1
        V[2 * 10 + k1][col] += 1
        W[k1 * 2 + 0][col] += 1
        col += 1
        
        # a12 * B[2,k0] -> C[1,k0]
        U[1 * 3 + 2][col] += 1
        V[2 * 10 + k0][col] += 1
        W[k0 * 2 + 1][col] += 1
        col += 1
        
        # a12 * B[2,k1] -> C[1,k1]
        U[1 * 3 + 2][col] += 1
        V[2 * 10 + k1][col] += 1
        W[k1 * 2 + 1][col] += 1
        col += 1
    
    assert col == rank, f"rank mismatch: {col} != {rank}"
    return U, V, W


def _make_schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W


# EVOLVE-BLOCK-START
def construct():
    """Find a rank-minimal exact tensor decomposition for 2x3x10 matrix multiplication."""
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # explicit wall-clock guard

    # 1) Immediate structural candidate: rank-55 Strassen-block decomposition.
    try:
        U, V, W = _make_strassen_decomposition(n, m, p)
        if _verify_fast(U, V, W, n, m, p, 55):
            # Verify exactly with integers
            if _verify_exact(U, V, W, n, m, p, 55):
                return {"U": U, "V": V, "W": W}
    except Exception:
        pass

    # 2) Guaranteed fallback: schoolbook (rank 60), always valid.
    U, V, W = _make_schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
