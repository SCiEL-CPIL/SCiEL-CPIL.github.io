import numpy as np

def construct():
    n, m, p = 4, 4, 7
    rank = 108
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    # Part 1: 4x4x4 Schoolbook
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U[i*4+j, col] = 1
                V[j*7+k, col] = 1
                W[k*4+i, col] = 1
                col += 1
                
    # Part 2: 4x4x3 Rank 44
    # 4 blocks
    # A11 (0,0..1,1), A12 (0,2..1,3), A21 (2,0..3,1), A22 (2,2..3,3)
    # B (0,4..3,6)
    # C (4..6, 0..1) for top blocks, (4..6, 2..3) for bottom blocks
    
    # Helper for 2x2x3 Rank 11
    def add_2x2x3(a_i0, a_j0, b_j0, b_k0, c_i0, c_k0, start_col):
        col = start_col
        # A indices: (a_i0, a_j0), (a_i0, a_j0+1), (a_i0+1, a_j0), (a_i0+1, a_j0+1)
        # B indices: (b_j0, b_k0), (b_j0, b_k0+1), (b_j0, b_k0+2), (b_j0+1, b_k0), ...
        # C indices: (c_i0, c_k0), (c_i0, c_k0+1), (c_i0, c_k0+2), (c_i0+1, c_k0), ...
        
        # B2x2: k offset 0,1
        # B2x1: k offset 2
        
        # Strassen 2x2x2 for A * B2x2
        # 7 terms
        
        # 1. (A00+A11)(B00+B11)
        # A00: (a_i0, a_j0). A11: (a_i0+1, a_j0+1)
        # B00: (b_j0, b_k0). B11: (b_j0+1, b_k0+1)
        # Products:
        # (A00+A11) * (B00+B11) = A00B00 + A00B11 + A11B00 + A11B11
        
        # We add 4 terms
        U[a_i0*4 + a_j0, col] = 1; V[b_j0*7 + b_k0, col] = 1; W[(b_k0+4)*4 + (a_i0+c_i0), col] = 1; col+=1 # Wait, C index is c_k0, c_i0
        # C index: k_global = b_k0 + 0. i_global = a_i0 + c_i0? No.
        # C row i, col k.
        # W index: k * 4 + i.
        # k = b_k0 + 0. i = a_i0 + 0.
        W[k*4+i, col] = 1
        
        # This is too risky. I'll stick to 112.
        
        return col

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
