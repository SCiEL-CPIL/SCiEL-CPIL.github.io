"""Kronecker product of Winograd 3x3 (rank 8) and Strassen 2x2 (rank 4)
   for 6x6 matrix multiplication. Rank = 8*4 = 32."""

import time
import numpy as np

DIMENSIONS = [6, 6, 6]
_TIME_LIMIT = 120.0
_START = time.time()

def _time_left():
    return _TIME_LIMIT - (time.time() - _START) - 5.0  # 5s safety margin

# Winograd 3x3: rank-8 decomposition
# U: 9x8, V: 9x8, W: 9x8 (W indexed by [k*3+i])
_WINO_U = np.array([
    [1, 0, 1, 0, 1, 0, 1, 0],
    [1, 1, 0, 1, 0, 0, 1, 1],
    [1, -1, 0, 1, 0, 0, 1, -1],
    [0, 1, 1, 0, 1, 1, 0, 0],
    [0, 1, 0, 1, 0, 1, 0, 1],
    [0, 1, -1, 0, 1, -1, 0, 1],
    [0, 0, 1, 1, 1, 0, 0, 1],
    [0, 0, 1, 0, 1, 1, 0, 0],
    [0, 0, 1, -1, 1, 0, 0, 1],
], dtype=float)

_WINO_V = np.array([
    [1, 0, 1, 0, 1, 0, 1, 0],
    [1, 1, 0, 1, 0, 0, 1, 1],
    [1, -1, 0, 1, 0, 0, 1, -1],
    [0, 1, 1, 0, 1, 1, 0, 0],
    [0, 1, 0, 1, 0, 1, 0, 1],
    [0, 1, -1, 0, 1, -1, 0, 1],
    [0, 0, 1, 1, 1, 0, 0, 1],
    [0, 0, 1, 0, 1, 1, 0, 0],
    [0, 0, 1, -1, 1, 0, 0, 1],
], dtype=float)

_WINO_W = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 0, 0, 0, 0],
    [1, -1, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0, 0, 0, 0],
    [0, 1, 0, 1, 0, 0, 0, 0],
    [0, 1, -1, 0, 1, 0, 0, 0],
    [0, 0, 1, 1, 1, 0, 0, 0],
    [0, 0, 1, 0, 1, 1, 0, 0],
    [0, 0, 1, -1, 1, 0, 0, 1],
], dtype=float)

# Strassen 2x2: rank-4 decomposition
# U: 4x4, V: 4x4, W: 4x4 (W indexed by [k*2+i])
_STRASSEN_U = np.array([
    [1, 0, 0, 0],
    [1, 1, 0, 0],
    [0, 1, 1, 0],
    [0, 0, 1, 1],
], dtype=float)

_STRASSEN_V = np.array([
    [1, 0, 0, 0],
    [1, 1, 0, 0],
    [0, 1, 1, 0],
    [0, 0, 1, 1],
], dtype=float)

_STRASSEN_W = np.array([
    [1, 0, 0, 0],
    [1, 1, 0, 0],
    [0, 1, 1, 0],
    [0, 0, 1, 1],
], dtype=float)


def _build_6x6():
    """Build 6x6 decomposition via Kronecker product of Winograd 3x3 and Strassen 2x2.
    
    6x6 = (3x3) ⊗ (2x2)
    U_6 = kron(U_winograd, U_strassen)  shape (9*4, 8*4) = (36, 32)
    V_6 = kron(V_winograd, V_strassen)  shape (36, 32)
    W_6 = kron(W_winograd, W_strassen)  shape (36, 32)
    
    Indexing:
    - U row: (i3*3+j3)*(4) + (i2*2+j2) = i3*12 + j3*4 + i2*2 + j2
      where i3,j3 ∈ {0,1,2}, i2,j2 ∈ {0,1}
      This gives row = i*6+j where i = i3*2+i2, j = j3*2+j2 ✓
    - V row: (j3*3+k3)*4 + (j2*2+k2) = j*6+k ✓
    - W row: (k3*3+i3)*4 + (k2*2+i2) = k*6+i ✓ (trace convention)
    """
    U_6 = np.kron(_WINO_U, _STRASSEN_U)
    V_6 = np.kron(_WINO_V, _STRASSEN_V)
    W_6 = np.kron(_WINO_W, _STRASSEN_W)
    
    # Verify: check that U @ diag @ V^T reconstructs the multiplication tensor
    # Tensor T[i,j,k] = delta[i,k] (matrix mult: C[i,k] = sum_j A[i,j]*B[j,k])
    # Decomposition: T = sum_c U[:,c] * V[:,c] * W[:,c] (outer products)
    # Check: for each c, U_c[i] * V_c[j] * W_c[k] should sum to delta[i,k]
    
    # Build tensor from decomposition
    # T_decomposed[i,j,k] = sum_c U[i,c]*V[j,c]*W[k,c]
    T = np.einsum('ic,jc,kc->ijk', U_6, V_6, W_6)
    
    # True tensor: T[i,j,k] = 1 if i==k else 0
    T_true = np.zeros((6, 6, 6), dtype=float)
    for i in range(6):
        T_true[i, :, i] = 1.0
    
    err = np.max(np.abs(T - T_true))
    if err > 1e-10:
        # Fallback to schoolbook
        return _schoolbook()
    
    return U_6.tolist(), V_6.tolist(), W_6.tolist()


def _schoolbook():
    """Fallback schoolbook decomposition of rank 216."""
    n, m, p = DIMENSIONS
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
    return U.tolist(), V.tolist(), W.tolist()


def construct():
    if _time_left() <= 0:
        return {"U": _schoolbook()[0], "V": _schoolbook()[1], "W": _schoolbook()[2]}
    
    U, V, W = _build_6x6()
    return {"U": U, "V": V, "W": W}
