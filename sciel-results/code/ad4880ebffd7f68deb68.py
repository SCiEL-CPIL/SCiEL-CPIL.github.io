"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x4 * 4x8 matrix multiplication.
    
    Strategy: Use a block decomposition that exploits the structure of the dimensions.
    - Split A (3x4) into a 2x4 top part and a 1x4 bottom part.
    - Split B (4x8) into 4 blocks of 4x2.
    
    For each 4x2 block of B:
    - The top 2x4 part of A multiplied by the 4x2 block can be decomposed using Strassen-like ideas.
      Specifically, split the 4x2 block into two 2x2 blocks (rows 0-1 and 2-3 of B).
      Each 2x2 * 2x2 multiplication can use Strassen (rank 7).
      But we need to be careful about the row structure.
      
    Actually, let me think more carefully.
    
    A is 3x4, B is 4x8.
    
    Let's partition:
    - A rows: {0,1} and {2}
    - B rows: {0,1} and {2,3}
    - B columns: 4 blocks of 2 columns each
    
    For a fixed column block (2 columns) of B:
    - Top part: A[0:2, :] (2x4) * B[:, c:c+2] (4x2) = (2x2) * (2x2) + (2x2) * (2x2)
      where we split B's rows into two groups of 2.
      
    Each (2x2) * (2x2) can be done with Strassen rank 7.
    So for the top part: 7 + 7 = 14 rank per column block.
    
    - Bottom part: A[2:3, :] (1x4) * B[:, c:c+2] (4x2)
      This is a 1x4 * 4x2 multiplication. The schoolbook rank is 1*4*2 = 8.
      But we can do better: since it's just a row vector times a matrix, 
      C[2, c] = sum_j A[2,j] * B[j, c]
      C[2, c+1] = sum_j A[2,j] * B[j, c+1]
      
      This can be decomposed with rank 4: for each j, we have A[2,j] * B[j,c] and A[2,j] * B[j,c+1].
      Actually, the rank is exactly 4 (one for each j, with the output being the two entries).
      
      Wait, let me think again. For 1x4 * 4x2:
      The tensor has components T[i,j,k] where i=0 (only one row), j in 0..3, k in 0..1.
      T[0,j,k] = 1 if we're computing the product.
      
      A rank-1 term is u ⊗ v ⊗ w where u is in R^1, v in R^4, w in R^2.
      u can only be [1] (up to scaling).
      So we need sum_r v_r ⊗ w_r to give the 4x2 identity-like structure.
      
      Actually, the tensor is just the 4x2 matrix viewed as a vector. The rank is 4
      (since we need 4 terms to cover the 4 rows, each contributing to both output columns).
      
      So bottom part: rank 4 per column block.
    
    Total rank per column block: 14 + 4 = 18
    Total rank: 4 * 18 = 72
    
    Let me implement this.
    """
    n, m, p = DIMENSIONS
    # n=3, m=4, p=8
    
    # We'll build U, V, W as lists of lists
    # U shape: (n*m, rank) = (12, rank)
    # V shape: (m*p, rank) = (32, rank)
    # W shape: (p*n, rank) = (24, rank)
    
    # First, let's figure out the total rank
    # Per column block: 14 (Strassen for top) + 4 (bottom) = 18
    # 4 column blocks: 4 * 18 = 72
    rank = 72
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def strassen_2x2x2(a_row_start, b_row_start, c_col_start):
        """
        Add Strassen decomposition for A[a_row_start:a_row_start+2, b_row_start:b_row_start+2]
        * B[b_row_start:b_row_start+2, c_col_start:c_col_start+2]
        
        Strassen for 2x2 * 2x2:
        Let A = [[a, b], [c, d]], B = [[e, f], [g, h]]
        
        m1 = (a+d)(e+h)
        m2 = (c+d)e
        m3 = a(f-h)
        m4 = d(g-e)
        m5 = (a+b)h
        m6 = (c-a)(e+f)
        m7 = (b-d)(g+h)
        
        C = [[m1+m4-m5+m7, m3+m5],
             [m2+m4, m1-m2+m3+m6]]
        
        Returns the number of columns used.
        """
        global col
        # Strassen factors for 2x2 * 2x2
        # U part: indices for A rows/cols
        # V part: indices for B rows/cols
        # W part: indices for C rows/cols
        
        # a = A[0,0] -> U index: a_row_start * m + b_row_start
        # b = A[0,1] -> U index: a_row_start * m + b_row_start + 1
        # c = A[1,0] -> U index: (a_row_start+1) * m + b_row_start
        # d = A[1,1] -> U index: (a_row_start+1) * m + b_row_start + 1
        
        # e = B[0,0] -> V index: b_row_start * p + c_col_start
        # f = B[0,1] -> V index: b_row_start * p + c_col_start + 1
        # g = B[1,0] -> V index: (b_row_start+1) * p + c_col_start
        # h = B[1,1] -> V index: (b_row_start+1) * p + c_col_start + 1
        
        # C[0,0] -> W index: c_col_start * n + a_row_start
        # C[0,1] -> W index: (c_col_start+1) * n + a_row_start
        # C[1,0] -> W index: c_col_start * n + (a_row_start+1)
        # C[1,1] -> W index: (c_col_start+1) * n + (a_row_start+1)
        
        ua = a_row_start * m + b_row_start
        ub = a_row_start * m + b_row_start + 1
        uc = (a_row_start+1) * m + b_row_start
        ud = (a_row_start+1) * m + b_row_start + 1
        
        ve = b_row_start * p + c_col_start
        vf = b_row_start * p + c_col_start + 1
        vg = (b_row_start+1) * p + c_col_start
        vh = (b_row_start+1) * p + c_col_start + 1
        
        w00 = c_col_start * n + a_row_start
        w01 = (c_col_start+1) * n + a_row_start
        w10 = c_col_start * n + (a_row_start+1)
        w11 = (c_col_start+1) * n + (a_row_start+1)
        
        # m1 = (a+d)(e+h)
        U[ua][col] = 1
        U[ud][col] = 1
        V[ve][col] = 1
        V[vh][col] = 1
        W[w00][col] = 1
        W[w10][col] = 1
        W[w11][col] = 1
        col += 1
        
        # m2 = (c+d)e
        U[uc][col] = 1
        U[ud][col] = 1
        V[ve][col] = 1
        W[w10][col] = 1
        W[w11][col] = -1
        col += 1
        
        # m3 = a(f-h)
        U[ua][col] = 1
        V[vf][col] = 1
        V[vh][col] = -1
        W[w01][col] = 1
        W[w11][col] = 1
        col += 1
        
        # m4 = d(g-e)
        U[ud][col] = 1
        V[vg][col] = 1
        V[ve][col] = -1
        W[w00][col] = 1
        W[w10][col] = 1
        col += 1
        
        # m5 = (a+b)h
        U[ua][col] = 1
        U[ub][col] = 1
        V[vh][col] = 1
        W[w00][col] = -1
        W[w01][col] = 1
        col += 1
        
        # m6 = (c-a)(e+f)
        U[uc][col] = 1
        U[ua][col] = -1
        V[ve][col] = 1
        V[vf][col] = 1
        W[w11][col] = 1
        col += 1
        
        # m7 = (b-d)(g+h)
        U[ub][col] = 1
        U[ud][col] = -1
        V[vg][col] = 1
        V[vh][col] = 1
        W[w00][col] = 1
        col += 1
        
        return 7
    
    def rank4_1x4x2(a_row, b_row_start, c_col_start):
        """
        Add rank-4 decomposition for A[a_row, :] (1x4) * B[:, c_col_start:c_col_start+2] (4x2)
        
        For each j in 0..3:
        U[a_row * m + j] = 1
        V[j * p + c_col_start] = 1
        V[j * p + c_col_start + 1] = 1
        W[c_col_start * n + a_row] = 1
        W[(c_col_start+1) * n + a_row] = 1
        """
        global col
        for j in range(4):
            U[a_row * m + j][col] = 1
            V[j * p + c_col_start][col] = 1
            V[j * p + c_col_start + 1][col] = 1
            W[c_col_start * n + a_row][col] = 1
            W[(c_col_start+1) * n + a_row][col] = 1
            col += 1
        return 4
    
    # Process each of the 4 column-blocks of B (width 2 each)
    for block in range(4):
        c_col_start = block * 2
        
        # Top part: rows 0,1 of A (2 rows) x 4 rows of B
        # Split B's 4 rows into two groups of 2: rows 0-1 and rows 2-3
        # Sub-block 1: A[0:2, 0:2] x B[0:2, c_col_start:c_col_start+2]
        strassen_2x2x2(0, 0, c_col_start)
        # Sub-block 2: A[0:2, 2:4] x B[2:4, c_col_start:c_col_start+2]
        strassen_2x2x2(0, 2, c_col_start)
        
        # Bottom part: row 2 of A (1 row) x 4 rows of B
        rank4_1x4x2(2, 0, c_col_start)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
