import numpy as np

DIMENSIONS = [5, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # We use a 6x6 algorithm with rank 63
    # 6x6 is decomposed into 2x2 blocks of 3x3
    # Strassen on 2x2 blocks uses 7 multiplications of 3x3 matrices
    # Schoolbook on 3x3 uses 9 multiplications
    # Total rank = 7 * 9 = 63
    
    rank = 63
    # U, V, W for 6x6
    # Size 36 x 63
    U6 = [[0] * rank for _ in range(36)]
    V6 = [[0] * rank for _ in range(36)]
    W6 = [[0] * rank for _ in range(36)]
    
    # We need to map the 3x3 schoolbook to the 63 columns
    # Each 3x3 multiplication uses 9 columns
    # There are 7 such multiplications
    # So column c corresponds to (block_mult_index, schoolbook_index)
    # block_mult_index in 0..6
    # schoolbook_index in 0..8
    
    def get_col(bm, si):
        return bm * 9 + si
        
    # Strassen formulas for 2x2 blocks of size 3x3
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # Indices for 3x3 blocks in 6x6 matrix (row-major)
    # Block (i, j) of size 3x3 starts at row i*3, col j*3
    # Index in 36-vector: r*6 + c
    
    def block_indices(i, j):
        # Returns list of 9 indices for block (i, j)
        idxs = []
        for dr in range(3):
            for dc in range(3):
                idxs.append((i*3 + dr)*6 + (j*3 + dc))
        return idxs
    
    # Define the 7 block multiplications
    # Each block multiplication is a linear combination of A blocks times a linear combination of B blocks
    # We need to expand these into scalar multiplications
    
    # M1: (A11 + A22) (B11 + B22)
    # A_part = A11 + A22
    # B_part = B11 + B22
    # We need to decompose (A11 + A22) (B11 + B22) into 9 scalar multiplications?
    # No, (A11 + A22) is a 3x3 matrix. (B11 + B22) is a 3x3 matrix.
    # Their product is a 3x3 matrix.
    # We use schoolbook for this 3x3 product.
    # So for each of the 9 scalar products in the 3x3 product, we have one rank term.
    # The scalar product is sum_k A'_ik B'_kj.
    # We need to express A'_ik and B'_kj in terms of the original A and B entries.
    
    # Let's define the linear combinations for each M_i
    
    # M1: A' = A11 + A22, B' = B11 + B22
    # A11 indices: block_indices(0,0)
    # A22 indices: block_indices(1,1)
    # B11 indices: block_indices(0,0)
    # B22 indices: block_indices(1,1)
    
    # For each of the 9 scalar products in M1:
    # C'_ij = sum_k A'_ik B'_kj
    # A'_ik = A11_ik + A22_ik
    # B'_kj = B11_kj + B22_kj
    # So the term is (A11_ik + A22_ik) * (B11_kj + B22_kj)
    # This is a single scalar multiplication in the rank decomposition?
    # No, the rank decomposition uses scalar multiplications of linear forms.
    # The term is (u_i) * (v_j) where u_i is a linear form in A and v_j is a linear form in B.
    # Here u_i = A11_ik + A22_ik and v_j = B11_kj + B22_kj.
    # So for each (i, j, k) in the 3x3 product, we have one rank term.
    # Wait, the schoolbook decomposition of 3x3 multiplication has rank 9?
    # No, schoolbook for 3x3 has rank 27?
    # $C_{ij} = \sum_{k=0}^2 A_{ik} B_{kj}$.
    # There are 9 outputs, each sum of 3 products. Total 27 products.
    # So the rank of 3x3 schoolbook is 27.
    # My previous calculation $7 \times 9 = 63$ was wrong.
    # It should be $7 \times 27 = 189$.
    # 189 > 125. So this is worse.
    
    # I need a better algorithm for 3x3.
    # Rank of 3x3 is 23 (Winograd).
    # 7 * 23 = 161. Still > 125.
    
    # So the 6x6 approach with Strassen is not better than 125.
    
    # What if I use a 5x5 algorithm directly?
    # Rank 49 is the target.
    
    # I will stick to the 125 rank algorithm to be safe.
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
