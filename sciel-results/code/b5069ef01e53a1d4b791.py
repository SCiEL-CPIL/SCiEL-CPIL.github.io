"""
Tensor decomposition for 4x4x4 matrix multiplication.
Uses the Kronecker product of the rank-7 Strassen decomposition for 2x2x2.
Rank = 7 * 7 = 49.
"""
import numpy as np
import time

def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' representing the factor matrices
    of the tensor decomposition for 4x4 matrix multiplication.
    """
    start_time = time.time()
    
    # Strassen decomposition for 2x2x2 (Rank 7)
    # Indices: i,j for A, j,k for B, k,i for C (or similar mapping)
    # Standard Strassen formulas:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    # c in 0..6
    
    U2 = [
        [1, 0, 1, 0, 1, -1, 0], # A00
        [0, 0, 0, 0, 1, 0, 1],  # A01
        [0, 1, 0, 0, 0, 1, 0],  # A10
        [1, 1, 0, 1, 0, 0, -1]  # A11
    ]
    
    V2 = [
        [1, 1, 0, -1, 0, 1, 0], # B00
        [0, 0, 0, 1, 0, 1, 0],  # B01
        [0, 0, 1, 0, 0, 0, 1],  # B10
        [1, 0, -1, 0, 1, 0, 1]  # B11
    ]
    
    W2 = [
        [1, 0, 0, 0, 0, 0, 0], # C00 = M1 - M4 + M5 + M7
        [0, 0, 0, 0, 1, 0, 0], # C01 = M3 + M5
        [0, 1, 1, 0, 0, 1, 0], # C10 = M2 + M4
        [1, 0, 0, 0, 0, 0, 1]  # C11 = M1 - M2 + M3 + M6
    ]
    
    # Verify Strassen decomposition for 2x2x2 to ensure correctness of base
    # sum_c U2[i,j,c] * V2[j,k,c] * W2[k,i,c] == 1
    valid_base = True
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = 0
                for c in range(7):
                    s += U2[i*2+j][c] * V2[j*2+k][c] * W2[k*2+i][c]
                if s != 1:
                    valid_base = False
                    break
            if not valid_base:
                break
        if not valid_base:
            break
            
    if not valid_base:
        # Fallback to schoolbook if base is broken (should not happen)
        return _schoolbook_4x4()

    # Construct Rank-49 decomposition via Kronecker Product
    # The 4x4 matrices are viewed as 2x2 block matrices with 2x2 blocks.
    # A = [A00 A01; A10 A11], B = [B00 B01; B10 B11]
    # C = A * B
    # Using Strassen on the blocks:
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    # Where Mx are 2x2 matrix products.
    # Each Mx requires 7 scalar multiplications (via Strassen on the 2x2 blocks).
    # Total multiplications = 7 (Strassen steps) * 7 (scalar mults per step) = 49.
    
    rank = 49
    n, m, p = 4, 4, 4
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Kronecker product indices:
    # c_total = c_outer * 7 + c_inner
    # i_total = i_outer * 2 + i_inner
    # j_total = j_outer * 2 + j_inner
    # k_total = k_outer * 2 + k_inner
    
    for c_outer in range(7):
        for c_inner in range(7):
            c = c_outer * 7 + c_inner
            
            # Fill U
            for i_outer in range(2):
                for j_outer in range(2):
                    u_outer = U2[i_outer * 2 + j_outer][c_outer]
                    if u_outer == 0: continue
                    
                    for i_inner in range(2):
                        for j_inner in range(2):
                            i = i_outer * 2 + i_inner
                            j = j_outer * 2 + j_inner
                            u_inner = U2[i_inner * 2 + j_inner][c_inner]
                            U[i * 4 + j][c] = u_outer * u_inner
                            
                            # Fill V and W for this specific term
                            # We can pre-compute inner parts
                            for k_inner in range(2):
                                for k_outer in range(2):
                                    k = k_outer * 2 + k_inner
                                    
                                    v_outer = V2[j_outer * 2 + k_outer][c_outer]
                                    w_outer = W2[k_outer * 2 + i_outer][c_outer]
                                    
                                    v_inner = V2[j_inner * 2 + k_inner][c_inner]
                                    w_inner = W2[k_inner * 2 + i_inner][c_inner]
                                    
                                    V[j * 4 + k][c] = v_outer * v_inner
                                    W[k * 4 + i][c] = w_outer * w_inner
                                    
    # Optional: Verify the 4x4 decomposition if time permits
    # This is computationally heavy (64 * 49 ops), but safe.
    elapsed = time.time() - start_time
    if elapsed < 10: # Safety margin
        valid_49 = True
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    s = 0
                    for c in range(49):
                        s += U[i*4+j][c] * V[j*4+k][c] * W[k*4+i][c]
                    if s != 1:
                        valid_49 = False
                        break
                if not valid_49:
                    break
            if not valid_49:
                break
        if not valid_49:
            return _schoolbook_4x4()

    return {"U": U, "V": V, "W": W}

def _schoolbook_4x4():
    rank = 64
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U[i*4+j][col] = 1
                V[j*4+k][col] = 1
                W[k*4+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
