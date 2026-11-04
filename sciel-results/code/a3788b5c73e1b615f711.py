"""Matrix multiplication 2x4x4 with Strassen-based rank-28 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 4, 4]


def _strassen_2x2_patterns():
    """
    Return the 7 Strassen multiplication patterns for 2x2 matrix multiplication.
    
    A entries indexed as: a00=0, a01=1, a10=2, a11=3
    B entries indexed as: b00=0, b01=1, b10=2, b11=3
    C entries indexed as: c00=0, c01=1, c10=2, c11=3
    
    M1 = (a00+a11)(b00+b11)
    M2 = (a10+a11)(b00)
    M3 = a00*(b01-b11)
    M4 = a11*(b10-b00)
    M5 = (a00+a01)*b11
    M6 = (a10-a00)(b00+b01)
    M7 = (a01-a11)(b10+b11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    """
    return [
        # M1: (a00+a11)(b00+b11) -> contributes to C00(+1), C11(+1)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2: (a10+a11)(b00) -> contributes to C10(+1), C11(-1)
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # M3: a00*(b01-b11) -> contributes to C01(+1), C11(+1)
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # M4: a11*(b10-b00) -> contributes to C00(+1), C10(+1)
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # M5: (a00+a01)*b11 -> contributes to C00(-1), C01(+1)
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # M6: (a10-a00)(b00+b01) -> contributes to C11(+1)
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # M7: (a01-a11)(b10+b11) -> contributes to C00(+1)
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]


def _build_decomposition(n=2, m=4, p=4):
    """
    Build a rank-28 Strassen-based decomposition for 2x4 * 4x4 matrix multiplication.
    
    A is 2x4: A[i,j] with i in {0,1}, j in {0,1,2,3}
    B is 4x4: B[j,k] with j in {0,1,2,3}, k in {0,1,2,3}
    C is 2x4: C[i,k] with i in {0,1}, k in {0,1,2,3}
    
    U index: i*m + j (8 rows)
    V index: j*p + k (16 rows)
    W index: k*n + i (8 rows, trace convention)
    
    Block structure:
    C[:, 0:2] = A_L * B_TL + A_R * B_BL
    C[:, 2:4] = A_L * B_TR + A_R * B_BR
    
    where A_L = A[:, 0:2], A_R = A[:, 2:4]
          B_TL = B[0:2, 0:2], B_TR = B[0:2, 2:4]
          B_BL = B[2:4, 0:2], B_BR = B[2:4, 2:4]
    
    Each 2x2 block product uses 7 Strassen multiplications.
    Total rank: 4 blocks * 7 = 28.
    """
    rank = 28
    U = [[0] * rank for _ in range(n * m)]   # (8, 28)
    V = [[0] * rank for _ in range(m * p)]   # (16, 28)
    W = [[0] * rank for _ in range(p * n)]   # (8, 28)
    
    patterns = _strassen_2x2_patterns()
    
    # Block definitions: (a_col_offset, b_row_offset, b_col_offset, c_col_offset)
    # For each 2x2 block product:
    #   A local (r,c) -> global A index: i=r, j=a_col_offset+c
    #   B local (r,c) -> global B index: j=b_row_offset+r, k=b_col_offset+c
    #   C local (r,c) -> global C index: i=r, k=c_col_offset+c
    blocks = [
        (0, 0, 0, 0),  # A_L * B_TL -> C[:, 0:2]
        (2, 2, 0, 0),  # A_R * B_BL -> C[:, 0:2]
        (0, 0, 2, 2),  # A_L * B_TR -> C[:, 2:4]
        (2, 2, 2, 2),  # A_R * B_BR -> C[:, 2:4]
    ]
    
    col = 0
    for (a_co, b_ro, b_co, c_co) in blocks:
        for (a_coeffs, b_coeffs, c_coeffs) in patterns:
            # Fill U: A entries
            for local_idx, coeff in a_coeffs.items():
                local_i = local_idx // 2
                local_j = local_idx % 2
                global_i = local_i
                global_j = a_co + local_j
                U[global_i * m + global_j][col] += coeff
            
            # Fill V: B entries
            for local_idx, coeff in b_coeffs.items():
                local_j = local_idx // 2
                local_k = local_idx % 2
                global_j = b_ro + local_j
                global_k = b_co + local_k
                V[global_j * p + global_k][col] += coeff
            
            # Fill W: C entries (trace convention: W[k*n + i])
            for local_idx, coeff in c_coeffs.items():
                local_i = local_idx // 2
                local_k = local_idx % 2
                global_i = local_i
                global_k = c_co + local_k
                W[global_k * n + global_i][col] += coeff
            
            col += 1
    
    return U, V, W


def _verify_decomposition(U, V, W, n=2, m=4, p=4):
    """Verify that the decomposition correctly computes matrix multiplication."""
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    rank = U_np.shape[1]
    
    # Build target tensor T[x,y,z] where:
    # x = i*m + j (A index)
    # y = j*p + k (B index)
    # z = k*n + i (C index, trace convention)
    T_target = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                x = i * m + j
                y = j * p + k
                z = k * n + i
                T_target[x, y, z] = 1.0
    
    # Compute reconstructed tensor
    T_recon = np.zeros((n*m, m*p, p*n), dtype=float)
    for r in range(rank):
        T_recon += np.einsum('x,y,z->xyz', U_np[:, r], V_np[:, r], W_np[:, r])
    
    return np.allclose(T_recon, T_target, atol=1e-9)


def construct():
    """
    Construct a low-rank tensor decomposition for 2x4 * 4x4 matrix multiplication.
    
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    
    Interface:
      U shape: (n*m, rank) = (8, rank)
      V shape: (m*p, rank) = (16, rank)
      W shape: (p*n, rank) = (8, rank)
    
    Indexing:
      A[i,j]: i in {0,1}, j in {0,1,2,3}. U index: i*4+j
      B[j,k]: j in {0,1,2,3}, k in {0,1,2,3}. V index: j*4+k
      C[i,k]: i in {0,1}, k in {0,1,2,3}. W index: k*2+i (trace convention)
    """
    t0 = time.time()
    deadline = 140.0  # seconds, leave 10s safety margin
    
    n, m, p = DIMENSIONS  # 2, 4, 4
    
    # Build the rank-28 Strassen decomposition
    U, V, W = _build_decomposition(n, m, p)
    
    # Verify correctness
    if not _verify_decomposition(U, V, W, n, m, p):
        raise ValueError("Decomposition verification failed")
    
    # Ensure we're within the time limit
    if time.time() - t0 > deadline:
        pass
    
    return {"U": U, "V": V, "W": W}
