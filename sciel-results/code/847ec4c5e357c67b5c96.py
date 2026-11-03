"""Tensor decomposition for 4x4 x 4x9 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """
    Returns U, V, W for M_{2,2,2} using Strassen's algorithm.
    Rank 7.
    U: (4, 7), V: (4, 7), W: (4, 7)
    """
    rank = 7
    U = np.zeros((4, rank))
    V = np.zeros((4, rank))
    W = np.zeros((4, rank))

    # Strassen formulas:
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)

    # A indices: 0=A00, 1=A01, 2=A10, 3=A11
    # B indices: 0=B00, 1=B01, 2=B10, 3=B11
    # C indices: 0=C00, 1=C01, 2=C10, 3=C11

    # M1: A00+A11, B00+B11
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1

    # M2: A10+A11, B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1

    # M3: A00, B01-B11
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1

    # M4: A11, B10-B00
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = -1; W[2, 3] = 1

    # M5: A00+A01, B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[1, 4] = 1

    # M6: A10-A00, B00+B01
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1

    # M7: A01-A11, B10+B11
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1

    return U, V, W

def _decomp_4x4x4():
    """
    Decomposition of M_{4,4,4} using 2-level Strassen.
    Rank 49.
    U: (16, 49), V: (16, 49), W: (16, 49)
    """
    n, m, p = 4, 4, 4
    rank = 49
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))

    inner_U, inner_V, inner_W = _strassen_2x2()

    # Outer Strassen coefficients
    outer_A = [
        [1, 0, 0, 1],   # M1: A00+A11
        [0, 0, 1, 1],   # M2: A10+A11
        [1, 0, 0, 0],   # M3: A00
        [0, 0, 0, 1],   # M4: A11
        [1, 1, 0, 0],   # M5: A00+A01
        [0, 0, 1, -1],  # M6: A10-A00
        [0, 1, 0, -1]   # M7: A01-A11
    ]

    outer_B = [
        [1, 0, 0, 1],   # M1: B00+B11
        [1, 0, 0, 0],   # M2: B00
        [0, 1, 0, -1],  # M3: B01-B11
        [0, 0, 1, -1],  # M4: B10-B00
        [0, 0, 0, 1],   # M5: B11
        [1, 1, 0, 0],   # M6: B00+B01
        [0, 0, 1, 1]    # M7: B10+B11
    ]

    outer_C = [
        [1, 0, 0, 1],   # M1 -> C00, C11
        [0, 0, 1, -1],  # M2 -> C10, C11(-1)
        [0, 1, 0, 1],   # M3 -> C01, C11
        [-1, 0, 1, 0],  # M4 -> C00(-1), C10
        [1, 1, 0, 0],   # M5 -> C00, C01
        [0, 0, 0, 1],   # M6 -> C11
        [1, 0, 0, 0]    # M7 -> C00
    ]

    for t in range(7):
        col_offset = t * 7
        for r in range(7):
            col = col_offset + r
            for li in range(2):
                for lj in range(2):
                    b = li * 2 + lj
                    coeff_U = outer_A[t][b] * inner_U[li*2 + lj, r]
                    coeff_V = outer_B[t][b] * inner_V[li*2 + lj, r]
                    coeff_W = outer_C[t][b] * inner_W[li*2 + lj, r]

                    for ei in range(2):
                        for ej in range(2):
                            g_i = 2 * (b // 2) + ei
                            g_j = 2 * (b % 2) + ej
                            U[g_i * 4 + g_j, col] = coeff_U
                            V[g_j * 4 + g_i, col] = coeff_V
                            W[g_i * 4 + g_j, col] = coeff_W

    return U, V, W

def _schoolbook_nm_p(n, m, p):
    """
    Schoolbook decomposition for M_{n,m,p}.
    Rank n*m*p.
    """
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _decomp_4x4x9_combined():
    """
    Decomposition of M_{4,4,9} by splitting into M_{4,4,4} + M_{4,4,4} + M_{4,4,1}.
    Total rank: 49 + 49 + 16 = 114.
    """
    n, m, p = 4, 4, 9
    total_rank = 49 + 49 + 16

    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))

    U444, V444, W444 = _decomp_4x4x4()
    U441, V441, W441 = _schoolbook_nm_p(4, 4, 1)

    # First 49 cols: 4x4x4 for k=0..3
    U[:, :49] = U444
    for j in range(m):
        for k in range(4):
            V[j * 9 + k, :49] = V444[j * 4 + k, :]
    for k in range(4):
        for i in range(n):
            W[k * 4 + i, :49] = W444[k * 4 + i, :]

    # Next 49 cols: 4x4x4 for k=4..7
    U[:, 49:98] = U444
    for j in range(m):
        for k in range(4):
            V[j * 9 + (4 + k), 49:98] = V444[j * 4 + k, :]
    for k in range(4):
        for i in range(n):
            W[(4 + k) * 4 + i, 49:98] = W444[k * 4 + i, :]

    # Last 16 cols: 4x4x1 for k=8
    U[:, 98:114] = U441
    for j in range(m):
        V[j * 9 + 8, 98:114] = V441[j * 1 + 0, :]
    for i in range(n):
        W[8 * 4 + i, 98:114] = W441[0 * 4 + i, :]

    return U, V, W

def _decomp_4x4x9_block_strassen():
    """
    Alternative: use Strassen on the 4x4 blocks of A and B, but treat the 9 columns
    of B in a block structure. Split B into three 4x3 blocks (columns 0-2, 3-5, 6-8).
    
    For each 4x3 block of B, we need to compute C_block = A * B_block (4x3).
    We can use a 4x4x3 decomposition.
    
    For 4x4x3: split into 4x4x2 + 4x4x1.
    4x4x2: use 2-level Strassen on the 2x2 blocks, with B being 4x2.
    Rank for 4x4x2: 4 subproblems of 2x2x2 (rank 7) = 28.
    4x4x1: schoolbook rank 16.
    So 4x4x3 = 28 + 16 = 44.
    Three blocks: 3 * 44 = 132. Worse than 114.
    
    Let's try a different approach: use the 4x4x4 decomposition for the first 8 columns
    (two 4x4x4 blocks = 98) and schoolbook for the last column (16). Total 114.
    This is the same as _decomp_4x4x9_combined.
    
    Another idea: can we use a rank-21 decomposition for 4x4x2?
    The tensor M_{4,4,2} can be seen as 2 copies of M_{4,4,1} coupled.
    Actually, M_{4,4,2} = M_{4,4,1} \oplus M_{4,4,1} if we decouple, rank 32.
    But with Strassen on the 2x2 blocks: 4 subproblems of 2x2x2 rank 7 = 28.
    Can we do better? The asymptotic rank of M_{4,4,2} is at most 4*7=28.
    I don't know of a rank < 28 for 4x4x2.
    
    Let's try to use a different block structure for the 9 columns.
    Split B into 4x4 + 4x4 + 4x1 (same as before, rank 114).
    Or split into 4x5 + 4x4: 
    4x4x5: split into 4x4x4 + 4x4x1 = 49 + 16 = 65.
    4x4x4: 49.
    Total: 65 + 49 = 114. Same.
    
    What about 4x4x9 directly using a higher-level Strassen?
    If we treat the 4x4 matrix A and the 4x9 matrix B, we can think of this as
    multiplying a 4x4 matrix by a 4x9 matrix. The result is 4x9.
    
    We can use the fact that 9 = 4 + 4 + 1, and use Strassen for the 4x4 parts.
    This gives rank 49 + 49 + 16 = 114.
    
    Is there a way to share computations between the two 4x4x4 blocks?
    The two 4x4x4 blocks use the same U (A-side) but different V and W.
    So we can't directly share, but the U part is identical.
    However, the rank is still additive: 49 + 49 + 16 = 114.
    
    Let's try a completely different approach: use the 2x2x9 decomposition.
    M_{2,2,9} can be decomposed using Strassen for the 2x2x2 part and schoolbook for the rest.
    Rank of M_{2,2,9}: 7 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 = 7 + 32 = 39? No.
    Actually, M_{2,2,9} = sum over k=0..8 of M_{2,2,1} for each k, but that's rank 4*9=36.
    Or use Strassen for 2x2x2 and schoolbook for 2x2x7: 7 + 4*7 = 35.
    
    For M_{4,4,9}, we can split into four M_{2,2,9} subproblems (2x2 blocks of A, and B, C).
    Each M_{2,2,9} has rank 35 (using Strassen for first 2 cols, schoolbook for rest).
    Total rank: 4 * 35 = 140. Worse.
    
    Or use a better M_{2,2,9} decomposition.
    M_{2,2,9}: split B into 2x4 + 2x4 + 2x1.
    2x2x4: use Strassen twice? 2x2x4 = 2x2x2 + 2x2x2 = 7 + 7 = 14.
    2x2x1: rank 4.
    Total: 14 + 14 + 4 = 32.
    Four subproblems: 4 * 32 = 128. Still worse than 114.
    
    So 114 seems to be the best we can get with these simple block decompositions.
    """
    # For now, return the same as _decomp_4x4x9_combined
    return _decomp_4x4x9_combined()

def _verify_decomp(U, V, W, n, m, p, atol=1e-9):
    """Vectorized verification of the decomposition."""
    rank = U.shape[1]
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)

    # Compute sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] for all (i,j,k)
    # Use einsum for efficiency
    # U_r: (n,m,r), V_r: (m,p,r), W_r: (p,n,r)
    # We want: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # = einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    result = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r, optimize=True)
    return np.allclose(result, 1.0, atol=atol)

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin: 150s limit, keep 20s buffer

    # Primary approach: 4x4x4 + 4x4x4 + 4x4x1 = rank 49 + 49 + 16 = 114
    U, V, W = _decomp_4x4x9_combined()

    elapsed = time.time() - start_time
    if elapsed < deadline - 5:
        if _verify_decomp(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: schoolbook (always valid, rank n*m*p = 144)
    Us, Vs, Ws = _schoolbook_nm_p(n, m, p)
    return {"U": Us.tolist(), "V": Vs.tolist(), "W": Ws.tolist()}

# EVOLVE-BLOCK-END
