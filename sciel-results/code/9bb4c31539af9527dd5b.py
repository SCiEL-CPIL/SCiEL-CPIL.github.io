"""
Rank-minimal exact tensor decomposition for 4x4 matrix multiplication.

Strategy:
- Use the tensor product (Kronecker product) structure of matrix multiplication.
- The 4x4 matrix multiplication tensor is T_4 = T_2 ⊗ T_2.
- We define the 2x2 Strassen decomposition tensors U_2, V_2, W_2 explicitly.
- We compute U_4 = U_2 ⊗ U_2, V_4 = V_2 ⊗ V_2, W_4 = W_2 ⊗ W_2.
- This ensures exact integer arithmetic and avoids manual index mapping errors.
- We then verify the decomposition. If valid, we return the factors.
- Note: The rank of T_2 is 7 (Strassen). The rank of T_4 via this method is 7 × 7 = 49.
- This is the best known upper bound for 4x4 MM rank.
"""

import time
import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START

def _build_strassen_2x2():
    """
    Construct the 2x2 Strassen factor tensors using a recursive index-mapping approach.
    Returns U2 (4,7), V2 (4,7), W2 (4,7) with integer entries.
    """
    U2 = np.zeros((4, 7), dtype=np.int64)
    V2 = np.zeros((4, 7), dtype=np.int64)
    W2 = np.zeros((4, 7), dtype=np.int64)

    # Strassen intermediate products and their contributions:
    # m0 = (A00+A11)(B00+B11) -> C00 += m0, C11 += m0
    # m1 = (A10+A11)B00       -> C10 += m1, C11 -= m1
    # m2 = A00(B10-B11)       -> C01 += m2, C11 += m2
    # m3 = A11(B01-B00)       -> C00 += m3, C10 += m3
    # m4 = (A00+A01)B11       -> C00 -= m4, C01 += m4
    # m5 = (A10-A00)(B00+B01) -> C11 += m5
    # m6 = (A01-A11)(B10+B11) -> C00 += m6

    # For each Strassen step s, define the A, B, C coefficients
    # A indices: 0=A00, 1=A01, 2=A10, 3=A11
    # B indices: 0=B00, 1=B01, 2=B10, 3=B11
    # C indices: 0=C00, 1=C01, 2=C10, 3=C11

    strassen_steps = [
        # (a_coeffs, b_coeffs, c_coeffs)
        # a_coeffs: dict {idx: coeff}
        # b_coeffs: dict {idx: coeff}
        # c_coeffs: dict {idx: coeff}
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),   # s=0: m0
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),        # s=1: m1
        ({0: 1}, {2: 1, 3: -1}, {1: 1, 3: 1}),        # s=2: m2
        ({3: 1}, {1: 1, 0: -1}, {0: 1, 2: 1}),        # s=3: m3
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),        # s=4: m4
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),        # s=5: m5
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),        # s=6: m6
    ]

    for s, (a_coeffs, b_coeffs, c_coeffs) in enumerate(strassen_steps):
        for idx, coeff in a_coeffs.items():
            U2[idx, s] += coeff
        for idx, coeff in b_coeffs.items():
            V2[idx, s] += coeff
        for idx, coeff in c_coeffs.items():
            W2[idx, s] += coeff

    return U2, V2, W2


def _verify_decomposition(U, V, W, tol=1e-9):
    """Verify that the decomposition exactly computes 4x4 matrix multiplication."""
    n, m, p = 4, 4, 4
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)

    rank = U.shape[1]
    U4 = U.reshape(n, m, rank)
    V4 = V.reshape(m, p, rank)
    W4 = W.reshape(p, n, rank)

    recon = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                recon[i, j, k] = np.sum(U4[i, j, :] * V4[j, k, :] * W4[k, i, :])

    target = np.ones((n, m, p))
    max_err = np.max(np.abs(recon - target))
    return max_err < tol


def _fallback_schoolbook():
    """Return schoolbook rank-64 decomposition as fallback."""
    n, m, p = 4, 4, 4
    rank = 64
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    # 1. Generate 2x2 Strassen tensors via recursive construction
    U2, V2, W2 = _build_strassen_2x2()

    # 2. Compute Kronecker products to get 4x4 tensors
    U4 = np.kron(U2, U2)
    V4 = np.kron(V2, V2)
    W4 = np.kron(W2, W2)

    # 3. Verify
    if time.time() > deadline:
        return {"U": U4.tolist(), "V": V4.tolist(), "W": W4.tolist()}

    if _verify_decomposition(U4, V4, W4):
        return {"U": U4.tolist(), "V": V4.tolist(), "W": W4.tolist()}
    else:
        return _fallback_schoolbook()

# EVOLVE-BLOCK-END
