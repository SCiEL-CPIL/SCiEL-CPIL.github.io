"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-44 decomposition for 2x3 x 3x8 matrix multiplication.
    
    Strategy: Apply Strassen's algorithm to the 2x2 sub-block within the 2x3 dimension,
    combined with the 8 columns. This reduces rank from 48 (schoolbook) to 44.
    
    The 2x3 matrix A is split as A = [A1 | A2] where A1 is 2x2 and A2 is 2x1.
    The 3x8 matrix B is split as B = [B1; B2] where B1 is 2x8 and B2 is 1x8.
    
    C = A1*B1 + A2*B2
    
    A1*B1 is 2x2 times 2x8: can use Strassen on the 2x2 part per column,
    but more efficiently, we treat the 2x2x8 tensor.
    
    Actually, let's use a direct construction:
    - The 2x2x2 tensor has Strassen rank 7 (vs 8 schoolbook).
    - For 2x2x8: we can use 7*4 = 28 terms? No, that's not right either.
    
    Let me use the known result: rank(2x3x8) = 44 is achievable.
    Here's an explicit construction using Strassen on the 2x2 block:
    
    For the 2x2 part of A (rows) times 2x8 part of B (cols):
    Use Strassen's 7 products for each 2x2x2 block, but since B has 8 cols,
    we process 4 blocks of 2 columns each: 7*4 = 28 terms for the 2x2x8 part.
    
    For the remaining 2x1 times 1x8: 2*8 = 16 terms.
    
    Wait, that gives 28+16 = 44. Yes!
    
    So the decomposition is:
    - 4 Strassen blocks of size 2x2x2 (using columns 0-1, 2-3, 4-5, 6-7 of B)
      Each uses 7 rank-1 terms: 4*7 = 28 terms
    - 8 direct products for the A2 (2x1) times B2 (1x8): 16 terms
      Wait, A2 is 2x1, B2 is 1x8, so A2*B2 gives a 2x8 result with 2*8=16 products.
    
    Total: 28 + 16 = 44.
    """
    n, m, p = DIMENSIONS  # n=2, m=3, p=8
    rank = 44
    U = [[0] * rank for _ in range(n * m)]  # 6 x 44
    V = [[0] * rank for _ in range(m * p)]  # 24 x 44
    W = [[0] * rank for _ in range(p * n)]  # 16 x 44
    
    col = 0
    
    # Part 1: Strassen decomposition of A1 (2x2) * B1 (2x8)
    # A1 consists of rows 0,1 of A (i=0,1; j=0,1) -> indices i*m+j = i*3+0, i*3+1
    # B1 consists of rows 0,1 of B (j=0,1; k=0..7) -> indices j*p+k = 0*8+k, 1*8+k
    # 
    # Split B1 into 4 blocks of 2 columns each: cols (0,1), (2,3), (4,5), (6,7)
    # For each 2x2x2 block, use Strassen's 7 products.
    
    for block in range(4):
        k0 = block * 2
        k1 = block * 2 + 1
        
        # Strassen's 7 products for 2x2 * 2x2:
        # Let A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
        # M1 = (a11+a22)(b11+b22)
        # M2 = (a21+a22)b11
        # M3 = a11(b12-b22)
        # M4 = a22(b21-b11)
        # M5 = (a11+a12)b22
        # M6 = (a21-a11)(b11+b12)
        # M7 = (a12-a22)(b21+b22)
        #
        # C11 = M1+M4-M5+M7
        # C12 = M3+M5
        # C21 = M2+M4
        # C22 = M1-M2+M3+M6
        #
        # Each Mi is a product of (linear combo of A entries) * (linear combo of B entries)
        # So each Mi contributes one rank-1 term to the tensor.
        
        # A1 entries: a11=A[0,0]=U index 0*3+0=0, a12=A[0,1]=U index 0*3+1=1
        #             a21=A[1,0]=U index 1*3+0=3, a22=A[1,1]=U index 1*3+1=4
        # B1 block entries: b11=B[0,k0]=V index 0*8+k0, b12=B[0,k1]=V index 0*8+k1
        #                   b21=B[1,k0]=V index 1*8+k0, b22=B[1,k1]=V index 1*8+k1
        
        # For each Strassen product, define the U, V, W contributions.
        # W index: k*n+i. For block, k is k0 or k1, i is 0 or 1.
        
        strassen_products = [
            # (coeff_a, a_indices, coeff_b, b_indices, c_row, c_col, coeff)
            # M1 = (a11+a22)(b11+b22)
            # contributes to C11: +1, C12: 0, C21: 0, C22: +1 (wait, need to be careful)
        ]
        
        # Let me directly enumerate the 7 Strassen terms and their contributions to W.
        # The output C has entries C[i,k] where i=0,1 (rows) and k=k0,k1 (cols in this block).
        # W index = k*n+i.
        
        # M1: U part = a11+a22 -> U[0]=1, U[4]=1; V part = b11+b22 -> V[0*8+k0]=1, V[1*8+k1]=1
        # W contribution: C[0,k0]+=M1, C[0,k1]+=0, C[1,k0]+=0, C[1,k1]+=M1
        # Wait, that's not right. Let me redo.
        
        # Actually, each Mi is a scalar product. The tensor decomposition is:
        # C[i,k] = sum_over_terms U[i,j_i] * V[j_i,k'] * W[k,i]
        # But here we're decomposing the 2x2x2 sub-tensor, so:
        # C_local[i_local, k_local] = sum_s U_local[i_local, s] * V_local[k_local, s] * W_local[i_local, k_local, s]
        # where W_local[i_local, k_local, s] is the scalar coefficient.
        
        # Let me just directly construct the 44 rank-1 terms.
        
        # For Strassen on 2x2*2x2 (rows 0,1 of A; cols k0,k1 of B):
        # Term 1: (a11+a22)*(b11+b22) -> affects C[0,k0] and C[1,k1]
        #   U: row0=1, row4=1 (a11 at idx 0, a22 at idx 4)
        #   V: col k0 row0=1, col k1 row1=1 -> V[0*8+k0]=1, V[1*8+k1]=1
        #   W: W[k0*2+0]=1, W[k1*2+1]=1
        
        # Term 2: (a21+a22)*b11 -> affects C[1,k0]
        #   U: row3=1, row4=1
        #   V: V[0*8+k0]=1
        #   W: W[k0*2+1]=1
        
        # Term 3: a11*(b12-b22) -> affects C[0,k1]
        #   U: row0=1
        #   V: V[0*8+k1]=1, V[1*8+k1]=-1
        #   W: W[k1*2+0]=1
        
        # Term 4: a22*(b21-b11) -> affects C[1,k0]
        #   U: row4=1
        #   V: V[1*8+k0]=1, V[0*8+k0]=-1
        #   W: W[k0*2+1]=1
        
        # Term 5: (a11+a12)*b22 -> affects C[0,k1] and C[1,k1]
        #   U: row0=1, row1=1
        #   V: V[1*8+k1]=1
        #   W: W[k1*2+0]=1, W[k1*2+1]=1
        
        # Term 6: (a21-a11)*(b11+b12) -> affects C[1,k0] and C[1,k1]
        #   U: row3=1, row0=-1
        #   V: V[0*8+k0]=1, V[0*8+k1]=1
        #   W: W[k0*2+1]=1, W[k1*2+1]=1
        
        # Term 7: (a12-a22)*(b21+b22) -> affects C[0,k1] and C[1,k1]
        #   U: row1=1, row4=-1
        #   V: V[1*8+k0]=1, V[1*8+k1]=1
        #   W: W[k1*2+0]=1, W[k1*2+1]=1
        
        # Now let's add each term:
        terms = [
            # (U_offsets, V_offsets, W_offsets)
            # U indices: a11=0, a12=1, a21=3, a22=4
            # V indices: b11=0*8+k0, b12=0*8+k1, b21=1*8+k0, b22=1*8+k1
            # W indices: C[i,k] = k*2+i, so C[0,k0]=k0*2+0, C[0,k1]=k1*2+0, C[1,k0]=k0*2+1, C[1,k1]=k1*2+1
            (
                [(0, 1), (4, 1)],  # U
                [(0*8+k0, 1), (1*8+k1, 1)],  # V
                [(k0*2+0, 1), (k1*2+1, 1)]  # W
            ),
            (
                [(3, 1), (4, 1)],
                [(0*8+k0, 1)],
                [(k0*2+1, 1)]
            ),
            (
                [(0, 1)],
                [(0*8+k1, 1), (1*8+k1, -1)],
                [(k1*2+0, 1)]
            ),
            (
                [(4, 1)],
                [(1*8+k0, 1), (0*8+k0, -1)],
                [(k0*2+1, 1)]
            ),
            (
                [(0, 1), (1, 1)],
                [(1*8+k1, 1)],
                [(k1*2+0, 1), (k1*2+1, 1)]
            ),
            (
                [(3, 1), (0, -1)],
                [(0*8+k0, 1), (0*8+k1, 1)],
                [(k0*2+1, 1), (k1*2+1, 1)]
            ),
            (
                [(1, 1), (4, -1)],
                [(1*8+k0, 1), (1*8+k1, 1)],
                [(k1*2+0, 1), (k1*2+1, 1)]
            ),
        ]
        
        for u_terms, v_terms, w_terms in terms:
            for idx, val in u_terms:
                U[idx][col] += val
            for idx, val in v_terms:
                V[idx][col] += val
            for idx, val in w_terms:
                W[idx][col] += val
            col += 1
    
    # Part 2: A2 (2x1, rows 0,1 of A; col j=2) * B2 (1x8, row j=2 of B; cols 0..7)
    # A2 entries: A[0,2]=U index 0*3+2=2, A[1,2]=U index 1*3+2=5
    # B2 entries: B[2,k]=V index 2*8+k for k=0..7
    # C[i,k] = A[i,2]*B[2,k]
    # This is 2*8 = 16 rank-1 terms (one for each (i,k) pair)
    
    for i in range(2):
        for k in range(8):
            U[i*3+2][col] = 1
            V[2*8+k][col] = 1
            W[k*2+i][col] = 1
            col += 1
    
    assert col == rank, f"Expected {rank} terms, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
