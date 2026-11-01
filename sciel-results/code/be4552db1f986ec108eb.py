"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-21 tensor decomposition for multiplying a 2x2 matrix 
    by a 2x6 matrix. This is the minimal known rank (and proven minimal 
    in many contexts for this specific dimension) using Strassen's algorithm 
    applied to three 2x2 blocks of the second matrix.
    
    The matrices U, V, W have shapes (n*m, rank), (m*p, rank), (p*n, rank)
    where n=2, m=2, p=6.
    U: 4 x 21
    V: 12 x 21
    W: 12 x 21
    """
    n, m, p = 2, 2, 6
    rank = 21
    
    # Initialize matrices with zeros
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Strassen's 7 products for 2x2 * 2x2:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 - B11)
    
    # We apply this to 3 blocks of the 2x6 matrix B.
    # B is 2x6. We split it into three 2x2 blocks: B0 (cols 0-1), B1 (cols 2-3), B2 (cols 4-5).
    # For block r (0, 1, 2), the columns of B are 2*r, 2*r+1.
    
    # U coefficients for the 7 Strassen products (indices 0..6)
    # U maps (i, j) to product index.
    # A is 2x2. Indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # M1: A00(1), A11(1) -> U[0][k]=1, U[3][k]=1
    # M2: A10(1), A11(1) -> U[2][k]=1, U[3][k]=1
    # M3: A00(1) -> U[0][k]=1
    # M4: A11(1) -> U[3][k]=1
    # M5: A00(1), A01(1) -> U[0][k]=1, U[1][k]=1
    # M6: A10(1), A00(-1) -> U[2][k]=1, U[0][k]=-1
    # M7: A01(1), A11(-1) -> U[1][k]=1, U[3][k]=-1
    
    U_coeffs = [
        {0: 1, 3: 1},   # M1
        {2: 1, 3: 1},   # M2
        {0: 1},         # M3
        {3: 1},         # M4
        {0: 1, 1: 1},   # M5
        {2: 1, 0: -1},  # M6
        {1: 1, 3: -1}   # M7
    ]
    
    # V coefficients for the 7 Strassen products.
    # V maps (j, k) to product index.
    # B block is 2x2. Indices in block: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Global k index: 2*r + local_k
    # V index: j * p + global_k
    
    # M1: B00(1), B11(1) -> V[j=0, k=0]=1, V[j=1, k=1]=1
    # M2: B00(1) -> V[j=0, k=0]=1
    # M3: B01(1), B11(-1) -> V[j=0, k=1]=1, V[j=1, k=1]=-1
    # M4: B10(1), B00(-1) -> V[j=1, k=0]=1, V[j=0, k=0]=-1
    # M5: B11(1) -> V[j=1, k=1]=1
    # M6: B00(1), B01(1) -> V[j=0, k=0]=1, V[j=0, k=1]=1
    # M7: B10(1), B11(-1) -> V[j=1, k=0]=1, V[j=1, k=1]=-1
    
    # Note: The V coefficients are relative to the local block.
    # We define them as (local_j, local_k, value)
    V_coeffs = [
        [(0, 0, 1), (1, 1, 1)], # M1
        [(0, 0, 1)],            # M2
        [(0, 1, 1), (1, 1, -1)],# M3
        [(1, 0, 1), (0, 0, -1)],# M4
        [(1, 1, 1)],            # M5
        [(0, 0, 1), (0, 1, 1)], # M6
        [(1, 0, 1), (1, 1, -1)] # M7
    ]
    
    # W coefficients for the 7 Strassen products.
    # W maps (k, i) to product index.
    # Output C is 2x6. C[i, global_k].
    # W index: global_k * n + i
    
    # C00 = M1 - M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M6 - M7
    
    # Let's map W entries (global_k, i, value) for each product.
    # For a specific block r, global_k is 2*r + local_k.
    # i is row index of C (0 or 1).
    
    # M1: C00(+1), C11(+1) -> W[2r+0, 0]=1, W[2r+1, 1]=1
    # M2: C10(+1) -> W[2r+0, 1]=1  (Wait, C10 is row 1, col 0. So i=1, k=0)
    #      Let's re-verify Strassen formulas.
    #      C = [C00 C01; C10 C11]
    #      C00 = M1 - M4 - M5 + M7
    #      C01 = M3 + M5
    #      C10 = M2 + M4
    #      C11 = M1 - M2 + M6 - M7
      
    #      M1 contributes to C00 and C11.
    #      M2 contributes to C10.
    #      M3 contributes to C01.
    #      M4 contributes to C00 and C10.
    #      M5 contributes to C00 and C01.
    #      M6 contributes to C11.
    #      M7 contributes to C00 and C11.
      
    #      Indices for W: row = k * n + i.
    #      For block r:
    #      M1: (0,0)->1, (1,1)->1. W[2r+0, 0]=1, W[2r+1, 1]=1.
    #      M2: (0,1)->1. W[2r+0, 1]=1. (C10: i=1, k=0)
    #      M3: (1,0)->1. W[2r+1, 0]=1. (C01: i=0, k=1)
    #      M4: (0,0)->-1, (0,1)->1. W[2r+0, 0]=-1, W[2r+0, 1]=1. (C00: i=0,k=0; C10: i=1,k=0)
    #      M5: (0,0)->-1, (1,0)->1. W[2r+0, 0]=-1, W[2r+1, 0]=1. (C00: i=0,k=0; C01: i=0,k=1)
    #      M6: (1,1)->1. W[2r+1, 1]=1. (C11: i=1, k=1)
    #      M7: (0,0)->1, (1,1)->-1. W[2r+0, 0]=1, W[2r+1, 1]=-1. (C00: i=0,k=0; C11: i=1,k=1)
    
    # Let's double check M4.
    # M4 = A11 * (B10 - B00).
    # C00 = M1 - M4 - M5 + M7. Term -M4.
    # C10 = M2 + M4. Term +M4.
    # So M4 contributes -1 to C00 and +1 to C10.
    # C00 is i=0, k=0. C10 is i=1, k=0.
    # So W[2r+0, 0] = -1, W[2r+0, 1] = 1. Correct.
    
    # Let's double check M5.
    # M5 = (A00 + A01) * B11.
    # C00 = M1 - M4 - M5 + M7. Term -M5.
    # C01 = M3 + M5. Term +M5.
    # So M5 contributes -1 to C00 and +1 to C01.
    # C00 is i=0, k=0. C01 is i=0, k=1.
    # So W[2r+0, 0] = -1, W[2r+1, 0] = 1. Correct.
    
    # Let's double check M7.
    # M7 = (A01 - A11) * (B10 - B11).
    # C00 = M1 - M4 - M5 + M7. Term +M7.
    # C11 = M1 - M2 + M6 - M7. Term -M7.
    # So M7 contributes +1 to C00 and -1 to C11.
    # C00 is i=0, k=0. C11 is i=1, k=1.
    # So W[2r+0, 0] = 1, W[2r+1, 1] = -1. Correct.
    
    W_coeffs = [
        [(0, 0, 1), (1, 1, 1)],     # M1: (k=0,i=0), (k=1,i=1)
        [(0, 1, 1)],                # M2: (k=0,i=1)
        [(1, 0, 1)],                # M3: (k=1,i=0)
        [(0, 0, -1), (0, 1, 1)],    # M4: (k=0,i=0), (k=0,i=1)
        [(0, 0, -1), (1, 0, 1)],    # M5: (k=0,i=0), (k=1,i=0)
        [(1, 1, 1)],                # M6: (k=1,i=1)
        [(0, 0, 1), (1, 1, -1)]     # M7: (k=0,i=0), (k=1,i=1)
    ]
    
    # Populate U, V, W
    for r in range(3):
        for k_local in range(7):
            col = r * 7 + k_local
            
            # U
            for idx, val in U_coeffs[k_local].items():
                U[idx][col] = val
            
            # V
            for j_local, k_local_b, val in V_coeffs[k_local]:
                global_k = 2 * r + k_local_b
                v_idx = j_local * p + global_k
                V[v_idx][col] = val
            
            # W
            for k_local_w, i_local, val in W_coeffs[k_local]:
                global_k = 2 * r + k_local_w
                w_idx = global_k * n + i_local
                W[w_idx][col] = val
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
