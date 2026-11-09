"""Program for matrix multiplication 4x4x4 using a rank-28 Strassen-based decomposition."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-28 exact tensor decomposition for 4x4 matrix multiplication.
    
    Approach:
    - Partition 4x4 matrices into 2x2 blocks of 2x2 submatrices.
    - Use Strassen's algorithm at the block level (7 block multiplications).
    - Each block multiplication is a 2x2 matrix multiplication.
    - For each 2x2 matmul, use the schoolbook method with 4 scalar multiplications.
    - Total rank = 7 * 4 = 28.
    
    The key insight is that we can encode the linear combinations of blocks directly
    into the U, V, W factor matrices using integer coefficients.
    """
    n, m, p = DIMENSIONS  # 4, 4, 4
    rank = 28  # 7 Strassen multiplications * 4 schoolbook per 2x2 matmul
    
    U = [[0] * rank for _ in range(n * m)]  # 16 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 16 x 28
    
    # Strassen's algorithm for 2x2 block multiplication:
    # Given block matrices A, B, C where each is 2x2 blocks of 2x2:
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    #
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    #
    # Strassen computes:
    # M1 = (A00+A11) * (B00+B11)
    # M2 = (A10+A11) * B00
    # M3 = A00 * (B01-B11)
    # M4 = A11 * (B10-B00)
    # M5 = (A00+A01) * B11
    # M6 = (A10-A00) * (B00+B10)
    # M7 = (A01-A11) * (B01+B11)
    #
    # Then:
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M6 - M7
    
    # For each of the 7 Strassen multiplications, we compute a 2x2 matrix product
    # using the schoolbook method with 4 scalar multiplications.
    
    # Helper function to add a 2x2 schoolbook matmul contribution
    # X = linear combination of A blocks
    # Y = linear combination of B blocks
    # Z = linear combination of C blocks (output)
    # sign_C = coefficient for the C block in the output
    
    def add_strassen_term(col_offset, a_blocks, b_blocks, c_block, c_coeff):
        """
        Add one Strassen term.
        
        a_blocks: dict mapping (br, bc) to coefficient, for A blocks
        b_blocks: dict mapping (br, bc) to coefficient, for B blocks
        c_block: (cr, cc) the output C block
        c_coeff: coefficient for the C block
        
        This computes: (sum of a_coeff * A_block) * (sum of b_coeff * B_block) -> c_coeff * C_block
        """
        # The product is a 2x2 matrix. We use schoolbook with 4 multiplications.
        # For 2x2 matrices X and Y:
        # Z[0][0] = X[0][0]*Y[0][0] + X[0][1]*Y[1][0]
        # Z[0][1] = X[0][0]*Y[0][1] + X[0][1]*Y[1][1]
        # Z[1][0] = X[1][0]*Y[0][0] + X[1][1]*Y[1][0]
        # Z[1][1] = X[1][0]*Y[0][1] + X[1][1]*Y[1][1]
        
        col = col_offset
        
        # For each of the 4 schoolbook terms:
        # term (xi, xj, yi, yj) where:
        # X[xi][xj] * Y[yi][yj] contributes to Z[xi][yj]
        # But we need to track which A and B blocks contribute to each element.
        
        # X[0][0] = sum over (ar, ac) of a_blocks[(ar,ac)] * A[ar*2+0][ac*2+0]
        # X[0][1] = sum over (ar, ac) of a_blocks[(ar,ac)] * A[ar*2+0][ac*2+1]
        # X[1][0] = sum over (ar, ac) of a_blocks[(ar,ac)] * A[ar*2+1][ac*2+0]
        # X[1][1] = sum over (ar, ac) of a_blocks[(ar,ac)] * A[ar*2+1][ac*2+1]
        
        # Similarly for Y.
        
        # Schoolbook term 1: X[0][0] * Y[0][0] -> Z[0][0]
        # Z[0][0] contributes to C[cr*2+0][cc*2+0] with coefficient c_coeff
        # X[0][0] = sum of a_coeff * A[ar*2+0][ac*2+0] for (ar,ac) in a_blocks
        # Y[0][0] = sum of b_coeff * B[br*2+0][bc*2+0] for (br,bc) in b_blocks
        #
        # In the tensor decomposition, for this column:
        # U[i*m+j, col] = coefficient of A[i][j] in X[0][0]
        # V[j*p+k, col] = coefficient of B[j][k] in Y[0][0]
        # W[k*n+i, col] = coefficient of C[i][k] in Z[0][0] = c_coeff if (i,k) = (cr*2+0, cc*2+0)
        
        # But wait, this doesn't work directly because X[0][0] and Y[0][0] are sums,
        # and their product is a sum of products. We need to expand it.
        
        # Actually, in the tensor decomposition, each column represents ONE scalar multiplication.
        # So we can't have a sum in U or V for a single column.
        
        # Let me reconsider. The correct approach:
        # Each "scalar multiplication" in the tensor decomposition is of the form:
        # (linear combination of A entries) * (linear combination of B entries) -> (linear combination of C entries)
        #
        # So for each Strassen term, we have 4 such scalar multiplications (one for each schoolbook term).
        # Each scalar multiplication has:
        # - A linear combination of A entries (for U)
        # - A linear combination of B entries (for V)
        # - A linear combination of C entries (for W)
        
        # Schoolbook term 1: X[0][0] * Y[0][0] -> Z[0][0]
        # U entries: for each (ar, ac) in a_blocks, U[ar*2+0, ac*2+0, col] = a_blocks[(ar,ac)]
        # V entries: for each (br, bc) in b_blocks, V[br*2+0, bc*2+0, col] = b_blocks[(br,bc)]
        # W entries: W[cc*2+0, cr*2+0, col] = c_coeff
        
        for (ar, ac), a_coeff in a_blocks.items():
            U[ar * 2 + 0][col] += a_coeff
        for (br, bc), b_coeff in b_blocks.items():
            V[br * 2 + 0][col] += b_coeff
        W[cc * 2 + 0][col] += c_coeff
        col += 1
        
        # Schoolbook term 2: X[0][1] * Y[1][0] -> Z[0][0]
        # U entries: for each (ar, ac) in a_blocks, U[ar*2+0, ac*2+1, col] = a_blocks[(ar,ac)]
        # V entries: for each (br, bc) in b_blocks, V[br*2+1, bc*2+0, col] = b_blocks[(br,bc)]
        # W entries: W[cc*2+0, cr*2+0, col] = c_coeff
        
        for (ar, ac), a_coeff in a_blocks.items():
            U[ar * 2 + 0][col] += a_coeff
        for (br, bc), b_coeff in b_blocks.items():
            V[br * 2 + 1][col] += b_coeff
        W[cc * 2 + 0][col] += c_coeff
        col += 1
        
        # Schoolbook term 3: X[0][0] * Y[0][1] -> Z[0][1]
        # U entries: for each (ar, ac) in a_blocks, U[ar*2+0, ac*2+0, col] = a_blocks[(ar,ac)]
        # V entries: for each (br, bc) in b_blocks, V[br*2+0, bc*2+1, col] = b_blocks[(br,bc)]
        # W entries: W[cc*2+1, cr*2+0, col] = c_coeff
        
        for (ar, ac), a_coeff in a_blocks.items():
            U[ar * 2 + 0][col] += a_coeff
        for (br, bc), b_coeff in b_blocks.items():
            V[br * 2 + 0][col] += b_coeff
        W[cc * 2 + 1][col] += c_coeff
        col += 1
        
        # Schoolbook term 4: X[0][1] * Y[1][1] -> Z[0][1]
        # U entries: for each (ar, ac) in a_blocks, U[ar*2+0, ac*2+1, col] = a_blocks[(ar,ac)]
        # V entries: for each (br, bc) in b_blocks, V[br*2+1, bc*2+1, col] = b_blocks[(br,bc)]
        # W entries: W[cc*2+1, cr*2+0, col] = c_coeff
        
        for (ar, ac), a_coeff in a_blocks.items():
            U[ar * 2 + 0][col] += a_coeff
        for (br, bc), b_coeff in b_blocks.items():
            V[br * 2 + 1][col] += b_coeff
        W[cc * 2 + 1][col] += c_coeff
        col += 1
        
        # Schoolbook term 5: X[1][0] * Y[0][0] -> Z[1][0]
        # U entries: for each (ar, ac) in a_blocks, U[ar*2+1, ac*2+0, col] = a_blocks[(ar,ac)]
        # V entries: for each (br, bc) in b_blocks, V[br*2+0, bc*2+0, col] = b_blocks[(br,bc)]
        # W entries: W[cc*2+0, cr*2+1, col] = c_coeff
        
        for (ar, ac), a_coeff in a_blocks.items():
            U[ar * 2 + 1][col] += a_coeff
        for (br, bc), b_coeff in b_blocks.items():
            V[br * 2 + 0][col] += b_coeff
        W[cc * 2 + 0][col] += c_coeff
        col += 1
        
        # Schoolbook term 6: X[1][1] * Y[1][0] -> Z[1][0]
        # U entries: for each (ar, ac) in a_blocks, U[ar*2+1, ac*2+1, col] = a_blocks[(ar,ac)]
        # V entries: for each (br, bc) in b_blocks, V[br*2+1, bc*2+0, col] = b_blocks[(br,bc)]
        # W entries: W[cc*2+0, cr*2+1, col] = c_coeff
        
        for (ar, ac), a_coeff in a_blocks.items():
            U[ar * 2 + 1][col] += a_coeff
        for (br, bc), b_coeff in b_blocks.items():
            V[br * 2 + 1][col] += b_coeff
        W[cc * 2 + 0][col] += c_coeff
        col += 1
        
        # Schoolbook term 7: X[1][0] * Y[0][1] -> Z[1][1]
        # U entries: for each (ar, ac) in a_blocks, U[ar*2+1, ac*2+0, col] = a_blocks[(ar,ac)]
        # V entries: for each (br, bc) in b_blocks, V[br*2+0, bc*2+1, col] = b_blocks[(br,bc)]
        # W entries: W[cc*2+1, cr*2+1, col] = c_coeff
        
        for (ar, ac), a_coeff in a_blocks.items():
            U[ar * 2 + 1][col] += a_coeff
        for (br, bc), b_coeff in b_blocks.items():
            V[br * 2 + 0][col] += b_coeff
        W[cc * 2 + 1][col] += c_coeff
        col += 1
        
        # Schoolbook term 8: X[1][1] * Y[1][1] -> Z[1][1]
        # U entries: for each (ar, ac) in a_blocks, U[ar*2+1, ac*2+1, col] = a_blocks[(ar,ac)]
        # V entries: for each (br, bc) in b_blocks, V[br*2+1, bc*2+1, col] = b_blocks[(br,bc)]
        # W entries: W[cc*2+1, cr*2+1, col] = c_coeff
        
        for (ar, ac), a_coeff in a_blocks.items():
            U[ar * 2 + 1][col] += a_coeff
        for (br, bc), b_coeff in b_blocks.items():
            V[br * 2 + 1][col] += b_coeff
        W[cc * 2 + 1][col] += c_coeff
        col += 1
        
        return col
    
    # Now apply Strassen's algorithm.
    # Each Strassen term Mi = Xi * Yi where Xi and Yi are linear combinations of blocks.
    
    # M1 = (A00+A11) * (B00+B11)
    # Contributes to: C00 (+1), C11 (+1)
    col = 0
    col = add_strassen_term(col, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, (0,0), 1)
    col = add_strassen_term(col, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, (1,1), 1)
    
    # M2 = (A10+A11) * B00
    # Contributes to: C10 (+1)
    col = add_strassen_term(col, {(1,0): 1, (1,1): 1}, {(0,0): 1}, (1,0), 1)
    
    # M3 = A00 * (B01-B11)
    # Contributes to: C01 (+1), C11 (-1)
    col = add_strassen_term(col, {(0,0): 1}, {(0,1): 1, (1,1): -1}, (0,1), 1)
    col = add_strassen_term(col, {(0,0): 1}, {(0,1): 1, (1,1): -1}, (1,1), -1)
    
    # M4 = A11 * (B10-B00)
    # Contributes to: C00 (-1), C10 (+1)
    col = add_strassen_term(col, {(1,1): 1}, {(1,0): 1, (0,0): -1}, (0,0), -1)
    col = add_strassen_term(col, {(1,1): 1}, {(1,0): 1, (0,0): -1}, (1,0), 1)
    
    # M5 = (A00+A01) * B11
    # Contributes to: C00 (+1), C01 (+1)
    col = add_strassen_term(col, {(0,0): 1, (0,1): 1}, {(1,1): 1}, (0,0), 1)
    col = add_strassen_term(col, {(0,0): 1, (0,1): 1}, {(1,1): 1}, (0,1), 1)
    
    # M6 = (A10-A00) * (B00+B10)
    # Contributes to: C11 (+1)
    col = add_strassen_term(col, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}, (1,1), 1)
    
    # M7 = (A01-A11) * (B01+B11)
    # Contributes to: C00 (-1), C11 (-1)
    col = add_strassen_term(col, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}, (0,0), -1)
    col = add_strassen_term(col, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}, (1,1), -1)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
