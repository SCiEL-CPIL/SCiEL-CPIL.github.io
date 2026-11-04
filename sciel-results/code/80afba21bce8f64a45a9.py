"""Matrix multiplication 4x4x6 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """Return U, V, W for 2x2x2 Strassen multiplication (rank 7).
    
    Indexing convention:
    - U has shape (n*m, rank) = (4, 7), indexed by [i*m + j, r]
    - V has shape (m*p, rank) = (4, 7), indexed by [j*p + k, r]
    - W has shape (p*n, rank) = (4, 7), indexed by [k*n + i, r]
    """
    n = m = p = 2
    rank = 7
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # M1 = (a00 + a11)(b00 + b11)
    U[0, 0] = 1  # a00
    U[3, 0] = 1  # a11
    V[0, 0] = 1  # b00
    V[3, 0] = 1  # b11
    W[0, 0] = 1  # c00
    W[3, 0] = 1  # c11
    
    # M2 = (a10 + a11)b00
    U[2, 1] = 1  # a10
    U[3, 1] = 1  # a11
    V[0, 1] = 1  # b00
    W[2, 1] = 1  # c10
    W[3, 1] = -1 # c11
    
    # M3 = a00(b01 - b11)
    U[0, 2] = 1  # a00
    V[1, 2] = 1  # b01
    V[3, 2] = -1 # b11
    W[1, 2] = 1  # c01
    W[3, 2] = 1  # c11
    
    # M4 = a11(b10 - b00)
    U[3, 3] = 1  # a11
    V[2, 3] = 1  # b10
    V[0, 3] = -1 # b00
    W[0, 3] = 1  # c00
    W[2, 3] = 1  # c10
    
    # M5 = (a00 + a01)b11
    U[0, 4] = 1  # a00
    U[1, 4] = 1  # a01
    V[3, 4] = 1  # b11
    W[0, 4] = -1 # c00
    W[1, 4] = 1  # c01
    
    # M6 = (a10 - a00)(b00 + b01)
    U[2, 5] = 1  # a10
    U[0, 5] = -1 # a00
    V[0, 5] = 1  # b00
    V[1, 5] = 1  # b01
    W[3, 5] = 1  # c11
    
    # M7 = (a11 - a10)(b10 + b11)
    U[3, 6] = 1  # a11
    U[2, 6] = -1 # a10
    V[2, 6] = 1  # b10
    V[3, 6] = 1  # b11
    W[0, 6] = 1  # c00
    
    return U, V, W


def _build_4x4x4_strassen():
    """Build a rank-49 decomposition for 4x4x4 matrix multiplication using Strassen.
    
    Uses block decomposition: 4x4 = 2x2 blocks of 2x2 matrices.
    Applies Strassen at block level (7 terms), each using 2x2x2 Strassen (7 terms).
    Total: 7 * 7 = 49 multiplications.
    
    Returns U (16, 49), V (16, 49), W (16, 49).
    """
    n = m = p = 4
    U2, V2, W2 = _strassen_2x2()
    
    U44 = np.zeros((16, 49))
    V44 = np.zeros((16, 49))
    W44 = np.zeros((16, 49))
    
    for r in range(7):  # Block-level Strassen term
        for s in range(7):  # Scalar-level Strassen term
            col = r * 7 + s
            
            # A_linear_r[a, b] = sum_{i, j} U2[i*2+j, r] * A[2*i+a, 2*j+b]
            # B_linear_r[c, d] = sum_{j2, k} V2[j2*2+k, r] * B[2*j2+c, 2*k+d]
            
            # M_s = (sum_{a, b} U2[a*2+b, s] * A_linear_r[a, b]) *
            #       (sum_{c, d} V2[c*2+d, s] * B_linear_r[c, d])
            
            # Expand:
            # M_s = (sum_{a, b, i, j} U2[a*2+b, s] * U2[i*2+j, r] * A[2*i+a, 2*j+b]) *
            #       (sum_{c, d, j2, k} V2[c*2+d, s] * V2[j2*2+k, r] * B[2*j2+c, 2*k+d])
            
            # Coefficient of A[2*i+a, 2*j+b] is U2[a*2+b, s] * U2[i*2+j, r]
            # Coefficient of B[2*j2+c, 2*k+d] is V2[c*2+d, s] * V2[j2*2+k, r]
            
            # Contribution to C[2*f+a, 2*e+b] is W2[e*2+f, s] * (product of coefficients)
            # But wait, the scalar Strassen computes a scalar M_s, and W2[e*2+f, s]
            # tells us which element of the 2x2 result matrix M_r gets this contribution.
            # M_r[f, e] += W2[e*2+f, s] * M_s
            # And C[2*f+a, 2*e+b] = M_r[f, e][a, b]
            
            # So for each (a, b) pair (the position within the 2x2 block):
            for a in range(2):
                for b in range(2):
                    for i in range(2):
                        for j in range(2):
                            u_coeff = U2[a*2+b, s] * U2[i*2+j, r]
                            if u_coeff != 0:
                                A_idx = (2*i + a) * 4 + (2*j + b)
                                U44[A_idx, col] += u_coeff
                    
                    for c in range(2):
                        for d in range(2):
                            for j2 in range(2):
                                for k in range(2):
                                    v_coeff = V2[c*2+d, s] * V2[j2*2+k, r]
                                    if v_coeff != 0:
                                        B_idx = (2*j2 + c) * 4 + (2*k + d)
                                        V44[B_idx, col] += v_coeff
                    
                    for f in range(2):
                        for e in range(2):
                            w_coeff = W2[e*2+f, s]
                            if w_coeff != 0:
                                C_idx = (2*e + b) * 4 + (2*f + a)
                                W44[C_idx, col] += w_coeff
    
    return U44, V44, W44


def _build_4x4x2_schoolbook():
    """Build a rank-32 decomposition for 4x4x2 matrix multiplication (schoolbook).
    
    Returns U (16, 32), V (8, 32), W (8, 32).
    """
    n = m = 4
    p = 2
    rank = n * m * p  # 32
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    
    return U, V, W


def construct():
    """Construct a rank-81 exact tensor decomposition for 4x4x6 matrix multiplication.
    
    Strategy: Split the 4x6 output into two parts:
    - First 4 columns: 4x4 * 4x4 multiplication (Strassen-based rank 49)
    - Last 2 columns: 4x4 * 4x2 multiplication (schoolbook rank 32)
    
    Total rank: 49 + 32 = 81
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # n=4, m=4, p=6
    
    # Part 1: 4x4 * 4x4 using Strassen (rank 49)
    U1, V1, W1 = _build_4x4x4_strassen()
    
    # Part 2: 4x4 * 4x2 using schoolbook (rank 32)
    U2, V2, W2 = _build_4x4x2_schoolbook()
    
    # Combine: total rank = 49 + 32 = 81
    total_rank = 49 + 32
    
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    # First 49 columns: 4x4 * 4x4 part
    # A is 4x4, B is 4x4 (first 4 cols), C is 4x4 (first 4 cols)
    U[:, :49] = U1
    V[:, :49] = V1  # V1 has shape (16, 49), but we need (m*p, 49) = (24, 49)
    # Wait, V should have shape (m*p, rank) = (4*6, 81) = (24, 81)
    # For the first part, B is 4x4, so only the first 4 columns of B are used.
    # V1 has shape (16, 49) where index is j*4 + k for j in 0..3, k in 0..3
    # We need to map this to V with shape (24, 81) where index is j*6 + k for j in 0..3, k in 0..5
    # For the first 4 columns of B, k is 0..3, so V1[j*4+k, col] -> V[j*6+k, col]
    
    # Let me redo this more carefully.
    
    # For Part 1 (4x4 * 4x4):
    # U has shape (16, 49): A[i*4+j, col] for i,j in 0..3
    # V has shape (16, 49): B[j*4+k, col] for j in 0..3, k in 0..3
    # W has shape (16, 49): C[k*4+i, col] for k in 0..3, i in 0..3
    
    # For the full problem (4x4 * 4x6):
    # U has shape (16, 81): A[i*4+j, col] for i,j in 0..3
    # V has shape (24, 81): B[j*6+k, col] for j in 0..3, k in 0..5
    # W has shape (24, 81): C[k*4+i, col] for k in 0..5, i in 0..3
    
    # Part 1 maps to columns 0..3 of B and C:
    # U[:, :49] = U1  (same)
    # V[j*6+k, :49] = V1[j*4+k, col] for j in 0..3, k in 0..3
    # W[k*4+i, :49] = W1[k*4+i, col] for k in 0..3, i in 0..3
    
    U[:, :49] = U1
    
    # Map V1 (16, 49) to V[:, :49] (24, 49)
    for j in range(4):
        for k in range(4):
            V[j*6 + k, :49] = V1[j*4 + k, :]
    
    # Map W1 (16, 49) to W[:, :49] (24, 49)
    for k in range(4):
        for i in range(4):
            W[k*4 + i, :49] = W1[k*4 + i, :]
    
    # Part 2 maps to columns 4..5 of B and C:
    # U[:, 49:] = U2  (same, A is 4x4)
    # V[j*6+k, 49:] = V2[j*2+k, col] for j in 0..3, k in 0..1
    # W[k*4+i, 49:] = W2[k*2+i, col] for k in 0..1, i in 0..3
    
    U[:, 49:] = U2
    
    # Map V2 (8, 32) to V[:, 49:] (24, 32)
    for j in range(4):
        for k in range(2):
            V[j*6 + 4 + k, 49:] = V2[j*2 + k, :]
    
    # Map W2 (8, 32) to W[:, 49:] (24, 32)
    for k in range(2):
        for i in range(4):
            W[4 + k*4 + i, 49:] = W2[k*2 + i, :]
    
    # Verify shapes
    assert U.shape == (16, 81), f"U shape: {U.shape}"
    assert V.shape == (24, 81), f"V shape: {V.shape}"
    assert W.shape == (24, 81), f"W shape: {W.shape}"
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
