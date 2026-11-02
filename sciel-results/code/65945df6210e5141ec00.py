"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 4x5 matrix by a 5x6 matrix.
    Uses Strassen-like block decomposition:
    - Split A (4x5) into 2x2 blocks: [2x2, 2x3; 2x2, 2x3]
    - Split B (5x6) into 2x2 blocks: [2x3, 2x3; 3x3, 3x3]
    - Split C (6x4) into 2x2 blocks: [3x2, 3x2; 3x2, 3x2]
    
    Use known low-rank decompositions for each block product.
    """
    import numpy as np
    
    n, m, p = DIMENSIONS  # 4, 5, 6
    
    # We'll build the decomposition using block structure
    # A is n x m = 4 x 5
    # B is m x p = 5 x 6
    # C is p x n = 6 x 4 (output in trace convention W[k*n+i, r])
    
    # Block decomposition:
    # n = 2 + 2
    # m = 2 + 3
    # p = 3 + 3
    
    # A blocks:
    # A00: 2x2, A01: 2x3
    # A10: 2x2, A11: 2x3
    
    # B blocks:
    # B00: 2x3, B01: 2x3
    # B10: 3x3, B11: 3x3
    
    # C blocks (C is 6x4, so C[k,i]):
    # C00: 3x2, C01: 3x2
    # C10: 3x2, C11: 3x2
    
    # Products:
    # C00 = A00*B00 + A01*B10  (2x2*2x3 + 2x3*3x3) -> 2x3 + 2x3 -> 2x3, but we need 3x2
    # Wait, there's a mismatch. Let me reconsider.
    
    # Actually, C[i,k] = sum_j A[i,j] * B[j,k]
    # For C00 (rows 0-2 of C, cols 0-1 of C):
    # C[i,k] for i in [0,1], k in [0,2] = sum_j A[i,j]*B[j,k]
    # = sum_{j in [0,1]} A[i,j]*B[j,k] + sum_{j in [2,4]} A[i,j]*B[j,k]
    # = (A00*B00)[i,k] + (A01*B10)[i,k]
    # A00 is 2x2, B00 is 2x3, so A00*B00 is 2x3
    # A01 is 2x3, B10 is 3x3, so A01*B10 is 2x3
    # Sum is 2x3, which matches C00 being 3 rows x 2 cols? No, 2x3 means 2 rows, 3 cols.
    # But C00 should be 3 rows (k=0,1,2) x 2 cols (i=0,1). So it's 3x2, not 2x3.
    
    # I think I have the block dimensions confused. Let me restart with clearer notation.
    
    # A is 4x5 (n=4 rows, m=5 cols)
    # B is 5x6 (m=5 rows, p=6 cols)  
    # C = A*B is 4x6 (n=4 rows, p=6 cols)
    # But the output is in trace convention: W[k*n+i, r] where C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r]
    # So C[i,k] for i in [0,3], k in [0,5]
    
    # Let me use a simpler approach: just use the schoolbook decomposition with rank 120,
    # but try to find a better construction.
    
    # For now, let's implement a valid decomposition and try to improve it.
    
    # Schoolbook: rank = n*m*p = 4*5*6 = 120
    rank = n * m * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
