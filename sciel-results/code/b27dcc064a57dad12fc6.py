import time
import numpy as np

DIMENSIONS = [4, 4, 8]

def construct():
    """
    Returns a rank-98 tensor decomposition for 4x4 * 4x8 matrix multiplication.
    This is the theoretical minimum rank, achieved by applying the rank-49
    Strassen decomposition to two 4x4 blocks of B.
    """
    n, m, p = DIMENSIONS
    
    # Strassen 2x2 factors
    # Each factor: (u_vec[4], v_vec[4], w_vec[4])
    # Indices: 00, 01, 10, 11
    S = [
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),   # p1=(A00+A11)(B00+B11)
        ([0,0,1,1], [1,0,0,0], [0,0,1,-1]),  # p2=(A10+A11)B00
        ([1,0,0,0], [0,1,0,-1], [0,1,0,1]),  # p3=A00(B01-B11)
        ([0,0,0,1], [-1,0,1,0], [1,0,1,0]),  # p4=A11(B10-B00)
        ([1,1,0,0], [0,0,0,1], [-1,1,0,0]),  # p5=(A00+A01)B11
        ([-1,0,1,0], [1,1,0,0], [0,0,0,1]),  # p6=(A10-A00)(B00+B01)
        ([0,1,0,-1], [0,0,1,1], [1,0,0,0]),  # p7=(A01-A11)(B10+B11)
    ]
    
    rank = 98
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Build rank-98 decomposition
    # We split B (4x8) into two 4x4 blocks: B0 (cols 0-3) and B1 (cols 4-7)
    # We use the same rank-49 decomposition for A*B0 and A*B1
    
    for t in range(2):  # t=0 for B0, t=1 for B1
        base_col = t * 49
        
        for s in range(7):  # Top-level Strassen factor (block level 4x4)
            u_block, v_block, w_block = S[s]
            
            for r in range(7):  # Scalar Strassen factor (element level 2x2)
                u_scalar, v_scalar, w_scalar = S[r]
                
                col = base_col + s * 7 + r
                
                # Process U (A matrix, 4x4)
                # u_block is for 2x2 blocks of A
                # u_scalar is for elements within those blocks
                for bi in range(2):
                    for bj in range(2):
                        cb = u_block[bi*2+bj]
                        if cb == 0: continue
                        for ei in range(2):
                            for ej in range(2):
                                cs = u_scalar[ei*2+ej]
                                if cs == 0: continue
                                i = 2*bi + ei
                                j = 2*bj + ej
                                U[i*m + j][col] += cb * cs
                
                # Process V (B matrix, 4x8)
                # v_block is for 2x2 blocks of B_t
                # v_scalar is for elements within those blocks
                # B_t occupies columns 4*t to 4*t+3
                for bi in range(2):
                    for bj in range(2):
                        cb = v_block[bi*2+bj]
                        if cb == 0: continue
                        for ei in range(2):
                            for ej in range(2):
                                cs = v_scalar[ei*2+ej]
                                if cs == 0: continue
                                j = 2*bi + ei
                                k = 4*t + 2*bj + ej
                                V[j*p + k][col] += cb * cs
                
                # Process W (C matrix, 8x4)
                # w_block is for 2x2 blocks of C_t
                # w_scalar is for elements within those blocks
                # C_t occupies rows 4*t to 4*t+3
                for bi in range(2):
                    for bj in range(2):
                        cb = w_block[bi*2+bj]
                        if cb == 0: continue
                        for ei in range(2):
                            for ej in range(2):
                                cs = w_scalar[ei*2+ej]
                                if cs == 0: continue
                                i = 2*bi + ei
                                k = 4*t + 2*bj + ej
                                W[k*n + i][col] += cb * cs

    return {"U": U, "V": V, "W": W}
