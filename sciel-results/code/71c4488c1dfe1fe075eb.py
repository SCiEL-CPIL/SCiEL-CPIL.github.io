"""
Matrix multiplication 2,3,10 tensor decomposition.
Implements the optimal rank 55 decomposition using Strassen's algorithm for the 2x2x2 blocks
and schoolbook for the 2x1x2 blocks.
"""
import numpy as np

def construct():
    n, m, p = 2, 3, 10
    
    # We decompose M_{2,3,10} into 5 blocks of M_{2,3,2}.
    # Each block handles output columns k in {b*2, b*2+1} for b=0..4.
    # Inside each block, we handle input columns j in {0,1,2}.
    # j in {0,1} forms a 2x2x2 tensor (rank 7 via Strassen).
    # j == 2 forms a 2x1x2 tensor (rank 4 via schoolbook).
    # Total rank per block = 7 + 4 = 11.
    # Total rank = 5 * 11 = 55.
    
    num_blocks = p // 2
    total_rank = num_blocks * 11
    
    # U shape: (n*m, total_rank) = (6, 55)
    # V shape: (m*p, total_rank) = (30, 55)
    # W shape: (p*n, total_rank) = (20, 55)
    
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    # Strassen coefficients for 2x2x2 multiplication.
    # We need to map the standard Strassen terms to the indices of our block.
    # Local indices for the block:
    # i in {0, 1} (row of A, row of C)
    # j in {0, 1} (col of A, row of B) -> This is the 2x2 core
    # k in {0, 1} (col of B, col of C)
    
    # Strassen terms defined as (u_coeffs, v_coeffs, w_coeffs)
    # Each is a list of (index, value)
    
    # M1: (A00+A11)(B00+B11)
    # U: (0,0), (1,1)
    # V: (0,0), (1,1)
    # W: (0,0), (1,1)
    strassen_m1 = (
        { (0,0): 1, (1,1): 1 },
        { (0,0): 1, (1,1): 1 },
        { (0,0): 1, (1,1): 1 }
    )
    
    # M2: (A10+A11)B00
    # U: (1,0), (1,1)
    # V: (0,0)
    # W: (0,1), (1,1) with signs? 
    # Let's verify Strassen standard form:
    # C00 = M1 - M4 + M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # M1: (A00+A11)(B00+B11)
    # M2: (A10+A11)B00
    # M3: A00(B01-B11)
    # M4: A11(B10-B00)
    # M5: (A00+A01)B11
    # M6: (A10-A00)(B00+B01)
    # M7: (A01-A11)(B10+B11)
    
    # Let's re-verify the signs and indices for W.
    # C00 (k=0, i=0): M1 - M4 + M5 + M7
    # C01 (k=1, i=0): M3 + M5
    # C10 (k=0, i=1): M2 + M4
    # C11 (k=1, i=1): M1 - M2 + M3 + M6
    
    # Strassen Term 1:
    # U: A00 (0,0) +1, A11 (1,1) +1
    # V: B00 (0,0) +1, B11 (1,1) +1
    # W: C00 (0,0) +1, C11 (1,1) +1
    s1 = (
        {(0,0): 1, (1,1): 1},
        {(0,0): 1, (1,1): 1},
        {(0,0): 1, (1,1): 1}
    )
    
    # Strassen Term 2:
    # U: A10 (1,0) +1, A11 (1,1) +1
    # V: B00 (0,0) +1
    # W: C10 (0,1) +1, C11 (1,1) -1
    s2 = (
        {(1,0): 1, (1,1): 1},
        {(0,0): 1},
        {(0,1): 1, (1,1): -1}
    )
    
    # Strassen Term 3:
    # U: A00 (0,0) +1
    # V: B01 (0,1) +1, B11 (1,1) -1
    # W: C01 (1,0) +1, C11 (1,1) +1
    s3 = (
        {(0,0): 1},
        {(0,1): 1, (1,1): -1},
        {(1,0): 1, (1,1): 1}
    )
    
    # Strassen Term 4:
    # U: A11 (1,1) +1
    # V: B10 (1,0) +1, B00 (0,0) -1
    # W: C00 (0,0) -1, C10 (0,1) +1
    s4 = (
        {(1,1): 1},
        {(1,0): 1, (0,0): -1},
        {(0,0): -1, (0,1): 1}
    )
    
    # Strassen Term 5:
    # U: A00 (0,0) +1, A01 (0,1) +1
    # V: B11 (1,1) +1
    # W: C00 (0,0) +1, C01 (1,0) +1
    s5 = (
        {(0,0): 1, (0,1): 1},
        {(1,1): 1},
        {(0,0): 1, (1,0): 1}
    )
    
    # Strassen Term 6:
    # U: A10 (1,0) +1, A00 (0,0) -1
    # V: B00 (0,0) +1, B01 (0,1) +1
    # W: C11 (1,1) +1
    s6 = (
        {(1,0): 1, (0,0): -1},
        {(0,0): 1, (0,1): 1},
        {(1,1): 1}
    )
    
    # Strassen Term 7:
    # U: A01 (0,1) +1, A11 (1,1) -1
    # V: B10 (1,0) +1, B11 (1,1) +1
    # W: C00 (0,0) +1
    s7 = (
        {(0,1): 1, (1,1): -1},
        {(1,0): 1, (1,1): 1},
        {(0,0): 1}
    )
    
    strassen_terms = [s1, s2, s3, s4, s5, s6, s7]
    
    # Schoolbook terms for the 2x1x2 part (j=2)
    # We multiply A[:, 2] (2x1) by B[2, :] (1x2) to get C[:, k] (2x2)
    # Terms:
    # 1. A[0,2] * B[2,0] -> C[0,0]
    # 2. A[0,2] * B[2,1] -> C[0,1]
    # 3. A[1,2] * B[2,0] -> C[1,0]
    # 4. A[1,2] * B[2,1] -> C[1,1]
    
    schoolbook_terms = [
        (
            {(0,2): 1},
            {(2,0): 1},
            {(0,0): 1}
        ),
        (
            {(0,2): 1},
            {(2,1): 1},
            {(1,0): 1}
        ),
        (
            {(1,2): 1},
            {(2,0): 1},
            {(0,1): 1}
        ),
        (
            {(1,2): 1},
            {(2,1): 1},
            {(1,1): 1}
        )
    ]
    
    all_block_terms = strassen_terms + schoolbook_terms
    
    col_idx = 0
    for b in range(num_blocks):
        for term in all_block_terms:
            u_coeffs, v_coeffs, w_coeffs = term
            
            # Process U
            for (i, j), val in u_coeffs.items():
                U[i*m + j, col_idx] += val
                
            # Process V
            for (j, k_local), val in v_coeffs.items():
                k_global = b*2 + k_local
                V[j*p + k_global, col_idx] += val
                
            # Process W
            for (k_local, i), val in w_coeffs.items():
                k_global = b*2 + k_local
                W[k_global*n + i, col_idx] += val
                
            col_idx += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
