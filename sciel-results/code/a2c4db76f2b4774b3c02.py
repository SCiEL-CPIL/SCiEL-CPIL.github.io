"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=16
    # Rank 56 using Strassen for 8 blocks of 2x2x2
    rank = 56
    U = [[0] * rank for _ in range(n * m)] # 4 x 56
    V = [[0] * rank for _ in range(m * p)] # 32 x 56
    W = [[0] * rank for _ in range(p * n)] # 32 x 56
    
    # Strassen factors for U (A)
    # Indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    U_factors = [
        [1, 0, 0, 1], # s=0: A00 + A11
        [0, 0, 1, 1], # s=1: A10 + A11
        [1, 0, 0, 0], # s=2: A00
        [0, 0, 0, 1], # s=3: A11
        [1, 1, 0, 0], # s=4: A00 + A01
        [-1, 0, 1, 0],# s=5: A10 - A00
        [0, 1, 0, -1] # s=6: A01 - A11
    ]
    
    # Strassen factors for V (B) and W (C)
    # For each block l (0..7), we have specific indices in V and W
    # V indices: j*p+k. j=0,1. k=2l, 2l+1.
    # W indices: k*n+i. k=2l, 2l+1. i=0,1.
    
    # V factors (coefficients for B_0,2l, B_0,2l+1, B_1,2l, B_1,2l+1)
    # Order: (0,2l), (0,2l+1), (1,2l), (1,2l+1)
    V_coeffs = [
        [1, 0, 0, 1], # s=0: B00 + B11
        [1, 0, 0, 0], # s=1: B00
        [0, 1, 0, -1],# s=2: B01 - B11
        [-1, 0, 1, 0],# s=3: B10 - B00
        [0, 0, 0, 1], # s=4: B11
        [1, 1, 0, 0], # s=5: B00 + B01
        [0, 0, 1, 1]  # s=6: B10 + B11
    ]
    
    # W factors (coefficients for C_0,2l, C_1,2l, C_0,2l+1, C_1,2l+1)
    # Order: (0,2l), (1,2l), (0,2l+1), (1,2l+1)
    # Note: W index is k*n+i.
    # k=2l, i=0 -> 4l
    # k=2l, i=1 -> 4l+1
    # k=2l+1, i=0 -> 4l+2
    # k=2l+1, i=1 -> 4l+3
    W_coeffs = [
        [1, 0, 0, 1], # s=0: C00 + C11
        [0, 1, 0, -1],# s=1: C10 - C11
        [0, 0, 1, 1], # s=2: C01 + C11
        [1, 1, 0, 0], # s=3: C00 + C10
        [-1, 0, 1, 0],# s=4: -C00 + C01
        [0, 0, 0, 1], # s=5: C11
        [1, 0, 0, 0]  # s=6: C00
    ]
    
    for l in range(8):
        for s in range(7):
            col = l * 7 + s
            
            # Fill U
            for i in range(4):
                U[i][col] = U_factors[s][i]
                
            # Fill V
            # Indices: 
            # 0: 2l
            # 1: 2l+1
            # 2: 16+2l
            # 3: 17+2l
            idxs_v = [2*l, 2*l+1, 16+2*l, 17+2*l]
            for idx, val in zip(idxs_v, V_coeffs[s]):
                V[idx][col] = val
                
            # Fill W
            # Indices:
            # 0: 4l
            # 1: 4l+1
            # 2: 4l+2
            # 3: 4l+3
            idxs_w = [4*l, 4*l+1, 4*l+2, 4*l+3]
            for idx, val in zip(idxs_w, W_coeffs[s]):
                W[idx][col] = val
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
