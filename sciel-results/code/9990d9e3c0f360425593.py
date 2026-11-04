"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p, tol=1e-7):
    """Verify that U, V, W give a valid tensor decomposition of matrix multiplication."""
    r = len(U[0]) if U else 0
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    # Reshape to tensors
    Ut = Ua.reshape(n*m, r)
    Vt = Va.reshape(m*p, r)
    Wt = Wa.reshape(p*n, r)
    # The tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # We need T[i,j,k] = 1 for all (i,j,k)
    # Compute T
    T = np.zeros((n, m, p))
    for c in range(r):
        uc = Ut[:, c].reshape(n, m)
        vc = Vt[:, c].reshape(m, p)
        wc = Wt[:, c].reshape(p, n)
        for i in range(n):
            for j in range(m):
                if uc[i, j] != 0:
                    for k in range(p):
                        T[i, j, k] += uc[i, j] * vc[j, k] * wc[k, i]
    # Check T equals 1
    if not np.allclose(T, 1.0, atol=tol):
        return False
    return True

def _schoolbook(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
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

def _try_strassen_2x2x2():
    """
    Try to find a rank-7 decomposition for 2x2x2 matrix multiplication.
    The Strassen decomposition has:
    7 products:
    M1 = (A00+A11)*(B00+B11)
    M2 = (A10+A11)*B00
    M3 = A00*(B01-B11)
    M4 = A11*(B10-B00)
    M5 = (A00+A01)*B11
    M6 = (A10-A00)*(B00+B01)
    M7 = (A01-A11)*(B10+B11)
    
    Then:
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Product 1: (A00+A11)*(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # Product 2: (A10+A11)*B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # Product 3: A00*(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # Product 4: A11*(B10-B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[2][3] = 1; W[0][3] = 1
    
    # Product 5: (A00+A01)*B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # Product 6: (A10-A00)*(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[2][5] = 1; W[3][5] = 1
    
    # Product 7: (A01-A11)*(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[1][6] = 1; W[0][6] = -1
    
    if _verify(U, V, W, n, m, p, tol=1e-6):
        return U, V, W
    return None

def _try_2x2x3_rank_9():
    """
    Try to construct a rank-9 decomposition for 2x2x3 matrix multiplication.
    Uses Strassen for the 2x2x2 part (rank 7) and 2 additional terms for the third column.
    """
    n, m, p = 2, 2, 3
    rank = 9
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # First 7 terms: Strassen for 2x2x2
    s222 = _try_strassen_2x2x2()
    if s222 is None:
        return None
    U_s, V_s, W_s = s222
    for i in range(4):
        for c in range(7):
            U[i][c] = U_s[i][c]
    for j in range(2):
        for k in range(2):
            V[j*2+k][c if False else 0] = 0  # placeholder
    # Copy properly
    for i in range(4):
        for c in range(7):
            U[i][c] = U_s[i][c]
    for idx in range(4):
        j = idx // 2
        k = idx % 2
        for c in range(7):
            V[j*3+k][c] = V_s[idx][c]  # V is (m*p, rank) = (6, rank)
    for idx in range(4):
        k = idx // 2
        i = idx % 2
        for c in range(7):
            W[k*2+i][c] = W_s[idx][c]  # W is (p*n, rank) = (6, rank)
    
    # Terms 8 and 9 for the third column (k=2)
    # C02 = A00*B02 + A01*B12
    # C12 = A10*B02 + A11*B12
    
    # Term 8: A00*B02
    U[0][7] = 1
    V[2][7] = 1  # j=0, k=2 -> index 0*3+2 = 2
    W[4][7] = 1  # k=2, i=0 -> index 2*2+0 = 4
    
    # Term 9: A01*B12
    U[1][8] = 1
    V[5][8] = 1  # j=1, k=2 -> index 1*3+2 = 5
    W[4][8] = 1  # k=2, i=0 -> index 2*2+0 = 4
    
    # But we also need C12 = A10*B02 + A11*B12
    # This approach doesn't work cleanly. Let me try a different approach.
    
    # Actually, let me just use the schoolbook for 2x2x3 and see if we can reduce.
    # For now, return None and use a safer approach.
    return None

def _strassen_2x2x2():
    """Return Strassen 2x2x2 decomposition as (U, V, W) lists with rank 7."""
    n, m, p = 2, 2, 2
    rank = 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Product 1: (A00+A11)*(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # Product 2: (A10+A11)*B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # Product 3: A00*(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # Product 4: A11*(B10-B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[2][3] = 1; W[0][3] = 1
    
    # Product 5: (A00+A01)*B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # Product 6: (A10-A00)*(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[2][5] = 1; W[3][5] = 1
    
    # Product 7: (A01-A11)*(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[1][6] = 1; W[0][6] = -1
    
    return U, V, W

def _build_block_decomposition():
    """
    Build a decomposition using Strassen for 2x2x2 blocks and schoolbook for the rest.
    
    A is 3x4, B is 4x5, C is 3x5.
    
    We split the computation into blocks:
    - C[0, 0:2] = A[0, 0:2] * B[0:2, 0:2] + A[0, 2:4] * B[2:4, 0:2]
    - C[0, 2:5] = A[0, 0:2] * B[0:2, 2:5] + A[0, 2:4] * B[2:4, 2:5]
    - C[1:3, 0:2] = A[1:3, 0:2] * B[0:2, 0:2] + A[1:3, 2:4] * B[2:4, 0:2]
    - C[1:3, 2:5] = A[1:3, 0:2] * B[0:2, 2:5] + A[1:3, 2:4] * B[2:4, 2:5]
    
    This is getting complex. Let me use a simpler block structure.
    """
    n, m, p = 3, 4, 5
    
    # Use a block decomposition where we split:
    # A into A_top (1x4) and A_bottom (2x4)
    # B into B_left (4x2) and B_right (4x3)
    
    # C_top_left = A_top * B_left: 1x2 output
    # C_top_right = A_top * B_right: 1x3 output
    # C_bottom_left = A_bottom * B_left: 2x2 output -> use Strassen (rank 7)
    # C_bottom_right = A_bottom * B_right: 2x3 output -> schoolbook (rank 6)
    
    # Total rank: 2 + 3 + 7 + 6 = 18? No, that's not right either.
    # Each "block" computation still requires the full m=4 dimension in A and the full p=5 in B.
    
    # Let me think differently. The tensor is T[i,j,k] for i in 0..2, j in 0..3, k in 0..4.
    # T[i,j,k] = 1 means C[i,k] = sum_j A[i,j] * B[j,k].
    
    # A simple improvement: use Strassen for the 2x2x2 sub-tensor where i in {1,2}, j in {0,1}, k in {0,1}.
    # This sub-tensor has rank 7 instead of 8.
    # The remaining terms use schoolbook.
    
    # Total rank: 60 - 8 + 7 = 59
    
    # Actually, let's just try to build a rank-59 decomposition.
    
    rank = 59
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # For each (i, j, k), we need a term U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # when this is the only term for that (i,j,k), or we use Strassen for the 2x2x2 block.
    
    # Strassen block: i in {1,2}, j in {0,1}, k in {0,1}
    # These 8 (i,j,k) combinations are handled by 7 Strassen terms.
    
    sU, sV, sW = _strassen_2x2x2()
    
    # Map Strassen indices to global indices
    # Strassen U index: i_s * 2 + j_s where i_s in {0,1} maps to i in {1,2}, j_s in {0,1} maps to j in {0,1}
    # Strassen V index: j_s * 2 + k_s where j_s in {0,1} maps to j in {0,1}, k_s in {0,1} maps to k in {0,1}
    # Strassen W index: k_s * 2 + i_s where k_s in {0,1} maps to k in {0,1}, i_s in {0,1} maps to i in {1,2}
    
    for c in range(7):
        for i_s in range(2):
            for j_s in range(2):
                global_i = 1 + i_s
                global_j = 0 + j_s
                U[global_i * m + global_j][col + c] = sU[i_s * 2 + j_s][c]
        for j_s in range(2):
            for k_s in range(2):
                global_j = 0 + j_s
                global_k = 0 + k_s
                V[global_j * p + global_k][col + c] = sV[j_s * 2 + k_s][c]
        for k_s in range(2):
            for i_s in range(2):
                global_k = 0 + k_s
                global_i = 1 + i_s
                W[global_k * n + global_i][col + c] = sW[k_s * 2 + i_s][c]
    col += 7
    
    # Now add schoolbook terms for all other (i,j,k) combinations
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Skip if this (i,j,k) is in the Strassen block
                if i in [1, 2] and j in [0, 1] and k in [0, 1]:
                    continue
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # col should equal rank
    assert col == rank, f"Column count {col} != rank {rank}"
    
    return U, V, W

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x4 * 4x5 matrix multiplication.
    Uses Strassen for 2x2x2 sub-blocks to reduce rank below schoolbook.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Try the block decomposition with Strassen
    U, V, W = _build_block_decomposition()
    
    if time.time() > deadline - 5:
        # Fallback to schoolbook
        U, V, W = _schoolbook(n, m, p)
        return {"U": U, "V": V, "W": W}
    
    # Verify the decomposition
    if _verify(U, V, W, n, m, p, tol=1e-6):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    U, V, W = _schoolbook(n, m, p)
    assert _verify(U, V, W, n, m, p), "Schoolbook decomposition failed verification"
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
