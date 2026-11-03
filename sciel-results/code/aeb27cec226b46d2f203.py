import numpy as np
import time

DIMENSIONS = [4, 4, 8]

def _strassen_2x2_factors():
    """
    Return the 7 rank-1 factors for 2x2*2x2 matrix multiplication.
    Each factor is (u_vec, v_vec, w_vec) where:
      u_vec: 4-element vector for A (row-major: A00, A01, A10, A11)
      v_vec: 4-element vector for B (row-major: B00, B01, B10, B11)
      w_vec: 4-element vector for C (row-major: C00, C01, C10, C11)
    """
    factors = []
    
    # p1 = (A00+A11)(B00+B11)
    u = [1, 0, 0, 1]
    v = [1, 0, 0, 1]
    w = [1, 0, 0, 1]
    factors.append((u, v, w))
    
    # p2 = (A10+A11) * B00
    u = [0, 0, 1, 1]
    v = [1, 0, 0, 0]
    w = [0, 0, 1, -1]
    factors.append((u, v, w))
    
    # p3 = A00 * (B01-B11)
    u = [1, 0, 0, 0]
    v = [0, 1, 0, -1]
    w = [0, 1, 0, 1]
    factors.append((u, v, w))
    
    # p4 = A11 * (B10-B00)
    u = [0, 0, 0, 1]
    v = [-1, 0, 1, 0]
    w = [1, 0, 1, 0]
    factors.append((u, v, w))
    
    # p5 = (A00+A01) * B11
    u = [1, 1, 0, 0]
    v = [0, 0, 0, 1]
    w = [-1, 1, 0, 0]
    factors.append((u, v, w))
    
    # p6 = (A10-A00) * (B00+B01)
    u = [-1, 0, 1, 0]
    v = [1, 1, 0, 0]
    w = [0, 0, 0, 1]
    factors.append((u, v, w))
    
    # p7 = (A01-A11) * (B10+B11)
    u = [0, 1, 0, -1]
    v = [0, 0, 1, 1]
    w = [1, 0, 0, 0]
    factors.append((u, v, w))
    
    return factors

def construct():
    n, m, p = DIMENSIONS
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140  # 10s margin before 150s limit
    
    # Get the 2x2 Strassen factors
    s2_factors = _strassen_2x2_factors()
    
    # We need to build 4x4x4 factors first.
    # Rank of 4x4x4 is 49.
    rank_444 = 49
    U4 = np.zeros((4, 4, rank_444))
    V4 = np.zeros((4, 4, rank_444))
    W4 = np.zeros((4, 4, rank_444))
    
    # Recursive Strassen for 4x4x4:
    # Split A, B, C into 2x2 blocks of 2x2 matrices.
    # A = [[A00, A01], [A10, A11]] where each Ai is 2x2.
    # The 7 products at the top level (s=0..6) are combinations of these blocks.
    # Each block multiplication (2x2 * 2x2) is done using the 7 scalar Strassen factors (r=0..6).
    
    for s in range(7):
        # s2_factors[s] gives the coefficients for the top-level blocks
        u_block = s2_factors[s][0] # [A00, A01, A10, A11] coeffs
        v_block = s2_factors[s][1] # [B00, B01, B10, B11] coeffs
        w_block = s2_factors[s][2] # [C00, C01, C10, C11] coeffs
        
        for r in range(7):
            col_444 = s * 7 + r
            # s2_factors[r] gives the coefficients for the scalar elements within the 2x2 blocks
            u_scalar = s2_factors[r][0]
            v_scalar = s2_factors[r][1]
            w_scalar = s2_factors[r][2]
            
            # Fill U4
            for bi in range(2):
                for bj in range(2):
                    c_block = u_block[bi * 2 + bj]
                    if c_block == 0: continue
                    for ei in range(2):
                        for ej in range(2):
                            c_scalar = u_scalar[ei * 2 + ej]
                            if c_scalar == 0: continue
                            i = 2 * bi + ei
                            j = 2 * bj + ej
                            U4[i, j, col_444] = c_block * c_scalar
                            
            # Fill V4
            for bi in range(2):
                for bj in range(2):
                    c_block = v_block[bi * 2 + bj]
                    if c_block == 0: continue
                    for ei in range(2):
                        for ej in range(2):
                            c_scalar = v_scalar[ei * 2 + ej]
                            if c_scalar == 0: continue
                            j = 2 * bi + ei
                            k = 2 * bj + ej
                            V4[j, k, col_444] = c_block * c_scalar
                            
            # Fill W4
            for bi in range(2):
                for bj in range(2):
                    c_block = w_block[bi * 2 + bj]
                    if c_block == 0: continue
                    for ei in range(2):
                        for ej in range(2):
                            c_scalar = w_scalar[ei * 2 + ej]
                            if c_scalar == 0: continue
                            k = 2 * bi + ei
                            i = 2 * bj + ej
                            W4[k, i, col_444] = c_block * c_scalar

    # Now build 4x4x8 factors.
    # Total rank = 98.
    # First 49 columns: A * B[:, 0:4]
    # Last 49 columns: A * B[:, 4:8]
    
    total_rank = 98
    U = np.zeros((n * m, total_rank)) # 16 x 98
    V = np.zeros((m * p, total_rank)) # 32 x 98
    W = np.zeros((p * n, total_rank)) # 32 x 98
    
    # Block 1: k in 0..3
    for i in range(4):
        for j in range(4):
            for R in range(49):
                U[i * m + j, R] = U4[i, j, R]
                
    for j in range(4):
        for k in range(4):
            for R in range(49):
                V[j * p + k, R] = V4[j, k, R]
                
    for k in range(4):
        for i in range(4):
            for R in range(49):
                W[k * n + i, R] = W4[k, i, R]

    # Block 2: k in 4..7
    # Map k to k' = k - 4
    for i in range(4):
        for j in range(4):
            for R in range(49):
                U[i * m + j, 49 + R] = U4[i, j, R]
                
    for j in range(4):
        for kp in range(4):
            k = 4 + kp
            for R in range(49):
                V[j * p + k, 49 + R] = V4[j, kp, R]
                
    for kp in range(4):
        k = 4 + kp
        for i in range(4):
            for R in range(49):
                W[k * n + i, 49 + R] = W4[kp, i, R]

    # Check time
    if time.time() > deadline:
        # Fallback to schoolbook if something went wrong with time (unlikely)
        # But we are well within limits.
        pass

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
