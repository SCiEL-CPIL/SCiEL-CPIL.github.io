"""Matrix multiplication 2x2 by 2x14: rank-minimal exact tensor decomposition."""
import time

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-49 decomposition using Strassen's algorithm on 2x2 blocks of the second matrix.
    
    The tensor is decomposed into 7 blocks, each corresponding to a 2x2 by 2x2 multiplication.
    Each block uses the 7-term Strassen decomposition.
    Total rank = 7 * 7 = 49.
    """
    n, m, p = DIMENSIONS
    # p is 14, so we have 7 blocks of size 2
    num_blocks = p // 2
    rank = num_blocks * 7
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Strassen linear forms for A (2x2)
    # Indices: A00=0, A01=1, A10=2, A11=3
    # s=0: M1 = (A00+A11)
    # s=1: M2 = (A10+A11)
    # s=2: M3 = A00
    # s=3: M4 = A11
    # s=4: M5 = (A00+A01)
    # s=5: M6 = (A10-A00)
    # s=6: M7 = (A01-A11)
    U_forms = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    
    for r in range(num_blocks):
        # Block r uses B columns 2r and 2r+1
        # B indices: B0k, B1k
        # e = B[0, 2r] -> V index 2r
        # f = B[0, 2r+1] -> V index 2r+1
        # g = B[1, 2r] -> V index 14+2r
        # h = B[1, 2r+1] -> V index 15+2r
        
        idx_e = 2 * r
        idx_f = 2 * r + 1
        idx_g = 14 + 2 * r
        idx_h = 15 + 2 * r
        
        # W indices for C in block r
        # C[0, 2r] -> W index 2*(2r) + 0 = 4r
        # C[1, 2r] -> W index 2*(2r) + 1 = 4r+1
        # C[0, 2r+1] -> W index 2*(2r+1) + 0 = 4r+2
        # C[1, 2r+1] -> W index 2*(2r+1) + 1 = 4r+3
        
        w_idx_c00 = 4 * r
        w_idx_c10 = 4 * r + 1
        w_idx_c01 = 4 * r + 2
        w_idx_c11 = 4 * r + 3
        
        for s in range(7):
            c = 7 * r + s
            
            # Fill U
            for i in range(4):
                U[i][c] = U_forms[s][i]
                
            # Fill V
            # s=0: e+h
            # s=1: e
            # s=2: f-h
            # s=3: g-e
            # s=4: h
            # s=5: e+f
            # s=6: g+h
            if s == 0:
                V[idx_e][c] = 1.0
                V[idx_h][c] = 1.0
            elif s == 1:
                V[idx_e][c] = 1.0
            elif s == 2:
                V[idx_f][c] = 1.0
                V[idx_h][c] = -1.0
            elif s == 3:
                V[idx_e][c] = -1.0
                V[idx_g][c] = 1.0
            elif s == 4:
                V[idx_h][c] = 1.0
            elif s == 5:
                V[idx_e][c] = 1.0
                V[idx_f][c] = 1.0
            elif s == 6:
                V[idx_g][c] = 1.0
                V[idx_h][c] = 1.0
                
            # Fill W
            # s=0 (M1): C00+1, C11+1
            # s=1 (M2): C10+1, C11-1
            # s=2 (M3): C01+1, C11+1
            # s=3 (M4): C00+1, C10+1
            # s=4 (M5): C00-1, C01+1
            # s=5 (M6): C11+1
            # s=6 (M7): C00+1
            
            if s == 0:
                W[w_idx_c00][c] = 1.0
                W[w_idx_c11][c] = 1.0
            elif s == 1:
                W[w_idx_c10][c] = 1.0
                W[w_idx_c11][c] = -1.0
            elif s == 2:
                W[w_idx_c01][c] = 1.0
                W[w_idx_c11][c] = 1.0
            elif s == 3:
                W[w_idx_c00][c] = 1.0
                W[w_idx_c10][c] = 1.0
            elif s == 4:
                W[w_idx_c00][c] = -1.0
                W[w_idx_c01][c] = 1.0
            elif s == 5:
                W[w_idx_c11][c] = 1.0
            elif s == 6:
                W[w_idx_c00][c] = 1.0
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
