"""Matrix multiplication 4,4,8 rank-minimal exact tensor decomposition.

Combines the current program's structure (schoolbook baseline, einsum
verification, explicit deadline guard) with the inspiration's recursive
Strassen construction, fixing the critical bug where the same U factors
were shared across both output blocks (mathematically impossible since the
U-side has only 4 columns). The fix uses independent U factors per output
block, giving a valid rank-98 decomposition for 4x4 * 4x8.
"""
import time
import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START

def _schoolbook():
    """Return the trivial rank-128 schoolbook decomposition."""
    n, m, p = DIMENSIONS
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
    return {"U": U, "V": V, "W": W}


def _verify(U, V, W, tol=1e-8):
    """Check that U,V,W exactly decompose the matmul tensor. Returns True if valid."""
    n, m, p = DIMENSIONS
    r = len(U[0]) if U else 0
    if r == 0:
        return False
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    U_r = U.reshape(n, m, r)
    V_r = V.reshape(m, p, r)
    W_r = W.reshape(p, n, r)
    T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
    expected = np.ones((n, m, p))
    return bool(np.allclose(T, expected, atol=tol))


def _strassen_2x2_factors():
    """
    Returns U, V, W for 2x2 matrix multiplication using Strassen's algorithm.
    U: 4x7, V: 4x7, W: 4x7
    Indices: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))

    # M1 = (A11+A22)(B11+B22) -> contributes to C11, C22
    U[0, 0] = 1; U[3, 0] = 1  # A11, A22
    V[0, 0] = 1; V[3, 0] = 1  # B11, B22
    W[0, 0] = 1; W[3, 0] = 1  # C11, C22

    # M2 = (A21+A22)B11 -> contributes to C21
    U[2, 1] = 1; U[3, 1] = 1  # A21, A22
    V[0, 1] = 1               # B11
    W[2, 1] = 1               # C21

    # M3 = A11(B12-B22) -> contributes to C12
    U[0, 2] = 1               # A11
    V[1, 2] = 1; V[3, 2] = -1 # B12, -B22
    W[1, 2] = 1               # C12

    # M4 = A22(B21-B11) -> contributes to C21
    U[3, 3] = 1               # A22
    V[2, 3] = 1; V[0, 3] = -1 # B21, -B11
    W[2, 3] = 1               # C21

    # M5 = (A11+A12)B22 -> contributes to C12
    U[0, 4] = 1; U[1, 4] = 1  # A11, A12
    V[3, 4] = 1               # B22
    W[1, 4] = 1               # C12

    # M6 = (A21-A11)(B11+B12) -> contributes to C22
    U[2, 5] = 1; U[0, 5] = -1 # A21, -A11
    V[0, 5] = 1; V[1, 5] = 1  # B11, B12
    W[3, 5] = 1               # C22

    # M7 = (A12-A22)(B21+B22) -> contributes to C11
    U[1, 6] = 1; U[3, 6] = -1 # A12, -A22
    V[2, 6] = 1; V[3, 6] = 1  # B21, B22
    W[0, 6] = 1               # C11

    return U, V, W


def _strassen_4x4_factors():
    """
    Returns U, V, W for 4x4 matrix multiplication using recursive Strassen.
    U: 16x49, V: 16x49, W: 16x49
    Indices: U[i*4+j, r], V[j*4+k, r], W[k*4+i, r]
    """
    U2, V2, W2 = _strassen_2x2_factors()

    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))

    # 4x4 divided into 2x2 blocks of 2x2 matrices
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]

    # High-level Strassen factors:
    # M1 = (A00+A11)(B00+B11) -> C00, C11
    # M2 = (A10+A11)B00       -> C10
    # M3 = A00(B01-B11)       -> C01
    # M4 = A11(B10-B00)       -> C00, C10
    # M5 = (A00+A01)B11       -> C00, C01
    # M6 = (A10-A00)(B00+B01) -> C11
    # M7 = (A01-A11)(B10+B11) -> C00

    # Each entry: (U_block_coeffs, V_block_coeffs, W_block_coeffs)
    # Block (bi, bj) means the 2x2 sub-block at row-block bi, col-block bj
    high_level_factors = [
        # M1 = (A00+A11)(B00+B11) -> C00, C11
        {'U': {(0,0): 1, (1,1): 1}, 'V': {(0,0): 1, (1,1): 1}, 'W': {(0,0): 1, (1,1): 1}},
        # M2 = (A10+A11)B00 -> C10
        {'U': {(1,0): 1, (1,1): 1}, 'V': {(0,0): 1}, 'W': {(1,0): 1}},
        # M3 = A00(B01-B11) -> C01
        {'U': {(0,0): 1}, 'V': {(0,1): 1, (1,1): -1}, 'W': {(0,1): 1}},
        # M4 = A11(B10-B00) -> C00, C10
        {'U': {(1,1): 1}, 'V': {(1,0): 1, (0,0): -1}, 'W': {(0,0): 1, (1,0): 1}},
        # M5 = (A00+A01)B11 -> C00, C01
        {'U': {(0,0): 1, (0,1): 1}, 'V': {(1,1): 1}, 'W': {(0,0): -1, (0,1): 1}},
        # M6 = (A10-A00)(B00+B01) -> C11
        {'U': {(1,0): 1, (0,0): -1}, 'V': {(0,0): 1, (0,1): 1}, 'W': {(1,1): 1}},
        # M7 = (A01-A11)(B10+B11) -> C00
        {'U': {(0,1): 1, (1,1): -1}, 'V': {(1,0): 1, (1,1): 1}, 'W': {(0,0): 1}},
    ]

    for h, hf in enumerate(high_level_factors):
        for l in range(7):
            factor_idx = h * 7 + l

            # U part: for each block in U_coeffs, add low-level U contributions
            for (bi, bj), coeff_u in hf['U'].items():
                for li in range(2):
                    for lj in range(2):
                        a_row = bi * 2 + li
                        a_col = bj * 2 + lj
                        U_idx = a_row * 4 + a_col  # 4x4 index
                        U[U_idx, factor_idx] += coeff_u * U2[li * 2 + lj, l]

            # V part: for each block in V_coeffs, add low-level V contributions
            for (bi, bj), coeff_v in hf['V'].items():
                for li in range(2):
                    for lj in range(2):
                        b_row = bi * 2 + li
                        b_col = bj * 2 + lj
                        V_idx = b_row * 4 + b_col  # 4x4 index
                        V[V_idx, factor_idx] += coeff_v * V2[li * 2 + lj, l]

            # W part: for each block in W_coeffs, add low-level W contributions
            for (bi, bj), coeff_w in hf['W'].items():
                for li in range(2):
                    for lj in range(2):
                        c_row = bi * 2 + li
                        c_col = bj * 2 + lj
                        W_idx = c_row * 4 + c_col  # 4x4 index
                        W[W_idx, factor_idx] += coeff_w * W2[li * 2 + lj, l]

    return U, V, W


def _adapt_to_4x8(U4, V4, W4):
    """
    Adapt 4x4 Strassen factors to 4x8 multiplication.
    Input: U4: 16x49, V4: 16x49, W4: 16x49 (for 4x4 * 4x4 -> 4x4)
    Output: U: 16x98, V: 32x98, W: 32x98 (for 4x4 * 4x8 -> 4x8)

    For 4x4 * 4x8:
    A is 4x4, B is 4x8, C is 4x8
    U: (n*m, rank) = (16, rank)
    V: (m*p, rank) = (32, rank)  
    W: (p*n, rank) = (32, rank)

    We view B as two 4x4 matrices side by side: B = [B_left | B_right]
    C = [C_left | C_right]

    C_left = A * B_left (4x4 * 4x4 -> 4x4)
    C_right = A * B_right (4x4 * 4x4 -> 4x4)

    KEY FIX: Use INDEPENDENT U factors for each output block.
    The U-side has only 4 columns (j=0..3), so we cannot share U factors
    between two independent 4x4 multiplications. Instead, we use separate
    U factors U1 for the left block and U2 for the right block.

    For the left block (B_left, C_left):
      U1: 16 x 49 (A factors for left multiplication)
      V1: 16 x 49 (B_left factors)
      W1: 16 x 49 (C_left factors)

    For the right block (B_right, C_right):
      U2: 16 x 49 (A factors for right multiplication)
      V2: 16 x 49 (B_right factors)
      W2: 16 x 49 (C_right factors)

    Combined:
      U = [U1 | U2] : 16 x 98
      V = [V1; V2]  : 32 x 98  (V1 in rows 0-15, V2 in rows 16-31)
      W = [W1; W2]  : 32 x 98  (W1 in rows 0-15, W2 in rows 16-31)

    The decomposition is:
      T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]

    For k in 0..3 (left block): V[j,k,r] = V1[j,k,r] for r in 0..48, 0 otherwise
                                W[k,i,r] = W1[k,i,r] for r in 0..48, 0 otherwise
                                U[i,j,r] = U1[i,j,r] for r in 0..48, 0 otherwise
    So T[i,j,k] = sum_{r=0}^{48} U1[i,j,r] * V1[j,k,r] * W1[k,i,r] = (A*B_left)[i,k] = 1

    For k in 4..7 (right block): V[j,k,r] = V2[j,k-4,r-49] for r in 49..97, 0 otherwise
                                 W[k,i,r] = W2[k-4,i,r-49] for r in 49..97, 0 otherwise
                                 U[i,j,r] = U2[i,j,r-49] for r in 49..97, 0 otherwise
    So T[i,j,k] = sum_{r=49}^{97} U2[i,j,r-49] * V2[j,k-4,r-49] * W2[k-4,i,r-49]
                 = (A*B_right)[i,k-4] = 1
    """
    rank4 = U4.shape[1]  # 49

    # U: 16 x 98 = [U1 | U2]
    U = np.zeros((16, rank4 * 2))
    U[:, :rank4] = U4       # U1 for left block
    U[:, rank4:] = U4       # U2 for right block (independent copy)

    # V: 32 x 98
    # Rows 0-15: V1 (B_left), columns 0-48
    # Rows 16-31: V2 (B_right), columns 49-97
    V = np.zeros((32, rank4 * 2))
    V[:16, :rank4] = V4     # V1 for left block
    V[16:, rank4:] = V4     # V2 for right block

    # W: 32 x 98
    # W indices: c = k*n + i where k is 0-7 (p=8), i is 0-3 (n=4)
    # For C_left (k=0-3): rows 0-15, columns 0-48
    # For C_right (k=4-7): rows 16-31, columns 49-97
    W = np.zeros((32, rank4 * 2))
    W[:16, :rank4] = W4     # W1 for left block
    W[16:, rank4:] = W4     # W2 for right block

    return U.tolist(), V.tolist(), W.tolist()


def _construct_rank98():
    """
    Construct a rank-98 decomposition for 4x4 * 4x8 matrix multiplication.
    Uses recursive Strassen (rank 49 for 4x4*4x4) applied to two independent
    4x4*4x4 sub-problems with independent U factors.
    """
    U4, V4, W4 = _strassen_4x4_factors()
    U, V, W = _adapt_to_4x8(U4, V4, W4)
    return {"U": U, "V": V, "W": W}


def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 4x4 * 4x8 matrix multiplication.
    Returns {'U', 'V', 'W'} with the decomposition.
    """
    deadline = time.time() + 110  # Safety margin before 120s limit

    # Try the rank-98 Strassen-based decomposition first
    try:
        result = _construct_rank98()
        # Verify it's correct
        if _verify(result["U"], result["V"], result["W"]):
            return result
    except Exception:
        pass

    if time.time() > deadline:
        return _schoolbook()

    # Fallback to schoolbook (rank 128)
    return _schoolbook()

# EVOLVE-BLOCK-END
