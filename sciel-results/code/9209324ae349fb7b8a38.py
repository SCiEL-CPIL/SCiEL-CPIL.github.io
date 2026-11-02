import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 2, 16]

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit

    n, m, p = DIMENSIONS  # n=2, m=2, p=16

    # Try rank 55 using one block of rank-6 (if it exists) or all rank-7
    # For now, use the known rank-56 Strassen decomposition
    # But let's check if we can do better by mixing blocks
    
    # Actually, let's try a different approach: use the fact that the problem is
    # matrix multiplication 2x2 * 2x16, which can be seen as 8 independent 2x2*2x2
    # multiplications (one for each pair of columns), each of rank 7 by Strassen.
    # Total rank = 8 * 7 = 56.
    
    # Can we do better? The trivial bound is 4*16 = 64. Strassen gives 56.
    # Is there a known rank-55 decomposition? Let's try to use one block with rank 6.
    # Actually, for 2x2 matrix multiplication, rank 7 is minimal (known result).
    # So 8 blocks * 7 = 56 is likely minimal.
    
    # But wait - can we share computations between blocks? Since A is the same for all blocks,
    # maybe we can combine. Let me think about this differently.
    
    # The tensor is: T(A, B) = C where C[i,k] = sum_j A[i,j] * B[j,k]
    # A is 2x2 (4 entries), B is 2x16 (32 entries), C is 2x16 (32 entries)
    
    # This is equivalent to 8 independent 2x2 matrix multiplications sharing A.
    # The minimal rank for this combined problem might be less than 8*7=56 if we can
    # exploit the sharing of A.
    
    # Let's try: for each of the 8 blocks, we have Strassen rank-7.
    # The U factors for all blocks use the same A indices (0,1,2,3).
    # Can we merge them? No, because each block has its own V and W.
    
    # Actually, let me try a rank-55 decomposition by finding if one block can be done with rank 6.
    # Since rank 7 is minimal for 2x2, we can't do a single block with rank 6.
    
    # So rank 56 is likely optimal. Let's use it.
    
    num_blocks = p // 2  # 8 blocks
    R = num_blocks * 7   # 56
    
    U = np.zeros((n * m, R), dtype=int)
    V = np.zeros((m * p, R), dtype=int)
    W = np.zeros((p * n, R), dtype=int)
    
    # Strassen coefficients
    # A: 0(A00), 1(A01), 2(A10), 3(A11)
    # B local: 0(B00), 1(B01), 2(B10), 3(B11)
    # C local: 0(C00), 1(C01), 2(C10), 3(C11)
    
    U_coeffs = [
        {0: 1, 3: 1},      # M1
        {2: 1, 3: 1},      # M2
        {0: 1},            # M3
        {3: 1},            # M4
        {0: 1, 1: 1},      # M5
        {2: 1, 0: -1},     # M6
        {1: 1, 3: -1}      # M7
    ]
    
    V_coeffs = [
        {0: 1, 3: 1},      # M1
        {0: 1},            # M2
        {1: 1, 3: -1},     # M3
        {2: 1, 0: -1},     # M4
        {3: 1},            # M5
        {0: 1, 1: 1},      # M6
        {2: 1, 3: 1}       # M7
    ]
    
    W_contribs = [
        {0: 1, 3: 1},      # M1 -> C00, C11
        {2: 1, 3: -1},     # M2 -> C10, C11
        {1: 1, 3: 1},      # M3 -> C01, C11
        {0: 1, 2: 1},      # M4 -> C00, C10
        {0: -1, 1: 1},     # M5 -> C00, C01
        {3: 1},            # M6 -> C11
        {0: 1}             # M7 -> C00
    ]
    
    for q in range(num_blocks):
        b_00 = 0 * p + 2 * q
        b_01 = 0 * p + (2 * q + 1)
        b_10 = 1 * p + 2 * q
        b_11 = 1 * p + (2 * q + 1)
        
        c_00 = (2 * q) * n + 0
        c_01 = (2 * q + 1) * n + 0
        c_10 = (2 * q) * n + 1
        c_11 = (2 * q + 1) * n + 1
        
        b_map = {0: b_00, 1: b_01, 2: b_10, 3: b_11}
        c_map = {0: c_00, 1: c_01, 2: c_10, 3: c_11}
        
        for s in range(7):
            r = q * 7 + s
            
            for a_idx, coeff in U_coeffs[s].items():
                U[a_idx, r] = coeff
                
            for local_idx, coeff in V_coeffs[s].items():
                V[b_map[local_idx], r] = coeff
                
            for local_out_idx, coeff in W_contribs[s].items():
                W[c_map[local_out_idx], r] = coeff

    # Verify the decomposition
    if time.time() < deadline - 10:
        np.random.seed(12345)
        A_test = np.random.randint(-3, 3, size=(n, m)).astype(int)
        B_test = np.random.randint(-3, 3, size=(m, p)).astype(int)
        C_test = A_test @ B_test
        
        A_flat = A_test.flatten()
        B_flat = B_test.flatten()
        
        UA = U.T @ A_flat
        VB = V.T @ B_flat
        terms = UA * VB
        C_decomp = W @ terms
        
        C_test_flat = C_test.T.flatten()
        
        if not np.allclose(C_decomp, C_test_flat, rtol=1e-9, atol=1e-9):
            # Fallback: schoolbook rank 64
            rank = n * m * p
            U = np.zeros((n * m, rank), dtype=int)
            V = np.zeros((m * p, rank), dtype=int)
            W = np.zeros((p * n, rank), dtype=int)
            
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j, col] = 1
                        V[j * p + k, col] = 1
                        W[k * n + i, col] = 1
                        col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
