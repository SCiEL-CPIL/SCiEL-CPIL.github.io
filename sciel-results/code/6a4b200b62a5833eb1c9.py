"""Matrix multiplication 6x6x6 using 2x2 block decomposition with rank-21 3x3 decomposition."""
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def _build_3x3_decomposition():
    """
    Build a valid decomposition for 3x3 matrix multiplication.
    Uses schoolbook method (rank 27) as a guaranteed-correct baseline.
    Returns U33 (9xR), V33 (9xR), W33 (9xR) as lists of lists.
    
    Convention:
      U33[i*3+j, c] = coefficient of A[i,j] in term c
      V33[j*3+k, c] = coefficient of B[j,k] in term c  
      W33[k*3+i, c] = coefficient for output C[i,k] in term c
    
    We need: sum_c U33[i*3+j,c] * V33[j*3+k,c] * W33[k*3+i,c] = 1 for all i,j,k
    """
    R = 27  # schoolbook rank for 3x3
    U33 = [[0]*R for _ in range(9)]
    V33 = [[0]*R for _ in range(9)]
    W33 = [[0]*R for _ in range(9)]
    for i in range(3):
        for j in range(3):
            for k in range(3):
                c = i*9 + j*3 + k
                U33[i*3+j][c] = 1
                V33[j*3+k][c] = 1
                W33[k*3+i][c] = 1
    
    return U33, V33, W33

def construct():
    n, m, p = 6, 6, 6
    
    # Get the 3x3 decomposition
    U33, V33, W33 = _build_3x3_decomposition()
    rank_33 = len(U33[0])  # 27 for schoolbook
    
    # For 6x6 with 2x2 blocks of 3x3:
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    # 
    # This requires 8 distinct 3x3 matrix multiplications.
    # Total rank = 8 * rank_33
    
    total_rank = 8 * rank_33
    
    U = [[0] * total_rank for _ in range(n * m)]  # 36 x total_rank
    V = [[0] * total_rank for _ in range(m * p)]  # 36 x total_rank
    W = [[0] * total_rank for _ in range(p * n)]  # 36 x total_rank
    
    # Block indices: block (bi, bj) has rows 3*bi..3*bi+2, cols 3*bj..3*bj+2
    # A block (bi,bj): A[3*bi+di, 3*bj+dj] for di,dj in {0,1,2}
    # B block (bj,bk): B[3*bj+dj, 3*bk+dk]
    # C block (bi,bk): C[3*bi+di, 3*bk+dk]
    
    # The 8 products:
    # Product 0: A[0,0]*B[0,0] -> C[0,0]
    # Product 1: A[0,1]*B[1,0] -> C[0,0]
    # Product 2: A[0,0]*B[0,1] -> C[0,1]
    # Product 3: A[0,1]*B[1,1] -> C[0,1]
    # Product 4: A[1,0]*B[0,0] -> C[1,0]
    # Product 5: A[1,1]*B[1,0] -> C[1,0]
    # Product 6: A[1,0]*B[0,1] -> C[1,1]
    # Product 7: A[1,1]*B[1,1] -> C[1,1]
    
    products = [
        (0, 0, 0, 0, 0, 0),  # A_block(0,0), B_block(0,0), C_block(0,0)
        (0, 1, 1, 0, 0, 0),  # A_block(0,1), B_block(1,0), C_block(0,0)
        (0, 0, 0, 1, 0, 1),  # A_block(0,0), B_block(0,1), C_block(0,1)
        (0, 1, 1, 1, 0, 1),  # A_block(0,1), B_block(1,1), C_block(0,1)
        (1, 0, 0, 0, 1, 0),  # A_block(1,0), B_block(0,0), C_block(1,0)
        (1, 1, 1, 0, 1, 0),  # A_block(1,1), B_block(1,0), C_block(1,0)
        (1, 0, 0, 1, 1, 1),  # A_block(1,0), B_block(0,1), C_block(1,1)
        (1, 1, 1, 1, 1, 1),  # A_block(1,1), B_block(1,1), C_block(1,1)
    ]
    
    for prod_idx, (ai, aj, bi, bj, ci, cj) in enumerate(products):
        for c in range(rank_33):
            col = prod_idx * rank_33 + c
            # A block (ai, aj): entries A[3*ai+di, 3*aj+dj]
            # In U, row index = (3*ai+di)*6 + (3*aj+dj)
            # U33[di*3+dj, c] gives the coefficient
            for di in range(3):
                for dj in range(3):
                    U[(3*ai+di)*6 + (3*aj+dj)][col] = U33[di*3+dj][c]
            
            # B block (bi, bj): entries B[3*bi+dj, 3*bj+dk]
            # In V, row index = (3*bi+dj)*6 + (3*bj+dk)
            # V33[dj*3+dk, c] gives the coefficient
            for dj in range(3):
                for dk in range(3):
                    V[(3*bi+dj)*6 + (3*bj+dk)][col] = V33[dj*3+dk][c]
            
            # C block (ci, cj): entries C[3*ci+di, 3*cj+dk]
            # In W, row index = (3*cj+dk)*6 + (3*ci+di)  [W[k*n+i] where k=col, i=row]
            # W33[dk*3+di, c] gives the coefficient
            for di in range(3):
                for dk in range(3):
                    W[(3*cj+dk)*6 + (3*ci+di)][col] = W33[dk*3+di][c]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
