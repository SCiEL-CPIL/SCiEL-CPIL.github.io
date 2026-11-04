"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=9
    
    # Use block decomposition: split p=9 into three blocks of size 3
    # For each 2x3 x 3x3 block, use a rank-7 decomposition
    # Total rank = 3 * 7 = 21
    
    rank_per_block = 7
    num_blocks = p // m  # 9 // 3 = 3
    rank = rank_per_block * num_blocks
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Rank-7 decomposition for 2x3 x 3x3 matrix multiplication
    # This uses the known minimal rank for this case
    # The decomposition is based on the algebraic structure of the tensor
    
    # For 2x3 x 3x3, we can use the following rank-7 scheme:
    # Each term corresponds to a product of three vectors
    
    # Define the 7 terms for 2x3 x 3x3 multiplication
    # u_term[i,j] for i in 0..1, j in 0..2 (2x3 matrix)
    # v_term[j,k] for j in 0..2, k in 0..2 (3x3 matrix)  
    # w_term[k,i] for k in 0..2, i in 0..1 (3x2 output, but we need 3x2 for C which is 2x3)
    
    # Actually for C = AB where A is 2x3, B is 3x3, C is 2x3
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor has entries T[i,j,k] = e_i ⊗ e_j ⊗ e_k
    
    # Rank-7 decomposition exists for this case
    # Let me use a known construction
    
    # The 7 terms can be defined as follows (using the fact that 
    # the border rank is 6 but actual rank is 7 for 2x3x3):
    
    terms_2x3x3 = [
        # (u_indices, v_indices, w_indices, coefficient)
        # u: 2x3, v: 3x3, w: 3x2 (output transposed for trace convention)
        
        # Term 0: diagonal-like
        ({(0,0): 1, (0,1): 1, (0,2): 1}, 
         {(0,0): 1, (1,1): 1, (2,2): 1}, 
         {(0,0): 1, (1,1): 1, (2,2): 1}, 1),
        
        # Term 1
        ({(0,0): 1, (1,0): 1}, 
         {(0,0): 1, (0,1): 1}, 
         {(0,0): 1, (1,0): 1}, 1),
        
        # Term 2
        ({(0,1): 1, (1,1): 1}, 
         {(1,0): 1, (1,1): 1}, 
         {(0,1): 1, (1,1): 1}, 1),
        
        # Term 3
        ({(0,2): 1, (1,2): 1}, 
         {(2,0): 1, (2,1): 1}, 
         {(0,2): 1, (1,2): 1}, 1),
        
        # Term 4
        ({(0,0): 1, (0,1): 1, (1,2): 1}, 
         {(0,2): 1, (1,2): 1}, 
         {(2,0): 1, (2,1): 1}, 1),
        
        # Term 5
        ({(0,2): 1, (1,0): 1}, 
         {(2,2): 1, (0,0): 1}, 
         {(2,2): 1, (0,0): 1}, 1),
        
        # Term 6
        ({(1,1): 1, (0,2): 1}, 
         {(1,2): 1, (2,1): 1}, 
         {(1,2): 1, (2,1): 1}, 1),
    ]
    
    # This is getting complicated. Let me use a simpler but still improved approach.
    # Use the standard decomposition but with some optimization.
    
    # Actually, let me just use the trivial decomposition for now but with a cleaner structure
    # and verify it's correct. The trivial rank is 54.
    
    # Reset to trivial
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx = i * m * p + j * p + k
                U[i * m + j][idx] = 1
                V[j * p + k][idx] = 1
                W[k * n + i][idx] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
