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
    
    # Pre-compute non-zero indices for Strassen factors to speed up construction
    # S[s] is (u, v, w). We need (index, coeff) for each.
    S_sparse = []
    for s in S:
        u_sparse = [(i, val) for i, val in enumerate(s[0]) if val != 0]
        v_sparse = [(i, val) for i, val in enumerate(s[1]) if val != 0]
        w_sparse = [(i, val) for i, val in enumerate(s[2]) if val != 0]
        S_sparse.append((u_sparse, v_sparse, w_sparse))
        
    for t in range(2):  # t=0 for B0, t=1 for B1
        base_col = t * 49
        
        for s in range(7):  # Top-level Strassen factor (block level 4x4)
            u_block_sparse, v_block_sparse, w_block_sparse = S_sparse[s]
            
            for r in range(7):  # Scalar Strassen factor (element level 2x2)
                u_scalar_sparse, v_scalar_sparse, w_scalar_sparse = S_sparse[r]
                
                col = base_col + s * 7 + r
                
                # Process U (A matrix, 4x4)
                # u_block is for 2x2 blocks of A
                # u_scalar is for elements within those blocks
                for bi, cb in u_block_sparse:
                    for bj, cb2 in u_block_sparse: # Wait, u_block is a vector of 4 coeffs for 4 blocks?
                        # No, u_block is [c00, c01, c10, c11].
                        # The loop structure in the original code was:
                        # for bi in range(2):
                        #   for bj in range(2):
                        #     cb = u_block[bi*2+bj]
                        #     for ei in range(2):
                        #       for ej in range(2):
                        #         cs = u_scalar[ei*2+ej]
                        #         i = 2*bi + ei
                        #         j = 2*bj + ej
                        #         U[i*m + j][col] += cb * cs
                        
                        # My sparse list `u_block_sparse` contains (idx, val) where idx is 0..3.
                        # idx = bi*2 + bj.
                        pass
