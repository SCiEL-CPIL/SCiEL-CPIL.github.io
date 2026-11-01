"""
Program to find the rank-minimal exact tensor decomposition for multiplying
a 2 by 2 matrix by a 2 by 15 matrix.

Uses a block-based approach:
- For the first 8 columns (p=0..7), use Strassen's algorithm on 2x2 blocks (4 blocks of 2 columns each).
- For the last 7 columns (p=8..14), use schoolbook (7 columns).

Total rank = 4 * 7 + 7 = 35.

The trace convention is W[k*n+i, r] where n=2, so W has shape (p*n, rank) = (30, rank).
"""

DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the factor matrices U, V, W for the matrix multiplication tensor
    of dimensions (2, 2, 15) using a block Strassen + schoolbook hybrid.
    
    Returns:
        dict: A dictionary containing the keys 'U', 'V', 'W' with the factor matrices.
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=15
    
    # We'll use 4 Strassen blocks of 2 columns each (covering p=0..7)
    # and 1 schoolbook block of 7 columns (covering p=8..14)
    # Total rank = 4*7 + 7 = 35
    
    total_rank = 35
    U = [[0] * total_rank for _ in range(n * m)]  # shape (4, 35)
    V = [[0] * total_rank for _ in range(m * p)]  # shape (30, 35)
    W = [[0] * total_rank for _ in range(p * n)]  # shape (30, 35)
    
    col = 0
    
    # Strassen blocks: each block covers 2 consecutive p-values
    # For block starting at p_start, with p_start in {0, 2, 4, 6}
    # The 2x2 sub-problem is A (2x2) * B_sub (2x2) where B_sub has columns p_start, p_start+1
    # Strassen's 7 products for 2x2 matrix multiplication:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # 
    # Where A is 2x2: A00, A01; A10, A11
    # B is 2x2: B00, B01; B10, B11 (row 0, col 0; row 0, col 1; row 1, col 0; row 1, col 1)
    # C = A*B, so C00 = A00*B00 + A01*B10, C01 = A00*B01 + A01*B11
    #    C10 = A10*B00 + A11*B10, C11 = A10*B01 + A11*B11
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M5 - M7
    # C10 = M2 - M4 + M6
    # C11 = M1 - M2 + M3 - M6
    
    # U indices: i*m + j where i is row of A (0 or 1), j is col of A (0 or 1)
    # So U[0] = A00, U[1] = A01, U[2] = A10, U[3] = A11
    
    # V indices: j*p + k where j is row of B (0 or 1), k is col of B (0..14)
    # For a block at p_start, B00 corresponds to V[0*p + p_start] = V[p_start]
    # B01 corresponds to V[0*p + (p_start+1)] = V[p_start+1]
    # B10 corresponds to V[1*p + p_start] = V[p + p_start]
    # B11 corresponds to V[1*p + (p_start+1)] = V[p + p_start + 1]
    
    # W indices: k*n + i where k is col of C (0..14), i is row of C (0 or 1)
    # For a block at p_start, C00 corresponds to W[p_start*n + 0] = W[2*p_start]
    # C01 corresponds to W[(p_start+1)*n + 0] = W[2*(p_start+1)]
    # C10 corresponds to W[p_start*n + 1] = W[2*p_start + 1]
    # C11 corresponds to W[(p_start+1)*n + 1] = W[2*(p_start+1) + 1]
    
    strassen_blocks = [0, 2, 4, 6]
    
    for p_start in strassen_blocks:
        # V indices for this block
        v_b00 = p_start           # B[0][p_start]
        v_b01 = p_start + 1       # B[0][p_start+1]
        v_b10 = p + p_start       # B[1][p_start]
        v_b11 = p + p_start + 1   # B[1][p_start+1]
        
        # W indices for this block
        w_c00 = 2 * p_start       # C[0][p_start]
        w_c01 = 2 * (p_start + 1) # C[0][p_start+1]
        w_c10 = 2 * p_start + 1   # C[1][p_start]
        w_c11 = 2 * (p_start + 1) + 1  # C[1][p_start+1]
        
        # M1 = (A00+A11)(B00+B11)
        # U: A00=1, A11=1 -> U[0]=1, U[3]=1
        # V: B00=1, B11=1 -> V[v_b00]=1, V[v_b11]=1
        # C contributions: C00+=1, C11+=1
        U[0][col] = 1
        U[3][col] = 1
        V[v_b00][col] = 1
        V[v_b11][col] = 1
        W[w_c00][col] = 1
        W[w_c11][col] = 1
        col += 1
        
        # M2 = (A10+A11)B00
        # U: A10=1, A11=1 -> U[2]=1, U[3]=1
        # V: B00=1 -> V[v_b00]=1
        # C contributions: C10+=1, C11+=-1
        U[2][col] = 1
        U[3][col] = 1
        V[v_b00][col] = 1
        W[w_c10][col] = 1
        W[w_c11][col] = -1
        col += 1
        
        # M3 = A00(B10-B11)
        # U: A00=1 -> U[0]=1
        # V: B10=1, B11=-1 -> V[v_b10]=1, V[v_b11]=-1
        # C contributions: C11+=1
        U[0][col] = 1
        V[v_b10][col] = 1
        V[v_b11][col] = -1
        W[w_c11][col] = 1
        col += 1
        
        # M4 = A11(B01-B00)
        # U: A11=1 -> U[3]=1
        # V: B01=1, B00=-1 -> V[v_b01]=1, V[v_b00]=-1
        # C contributions: C00+=1, C10+=-1
        U[3][col] = 1
        V[v_b01][col] = 1
        V[v_b00][col] = -1
        W[w_c00][col] = 1
        W[w_c10][col] = -1
        col += 1
        
        # M5 = (A00+A01)B11
        # U: A00=1, A01=1 -> U[0]=1, U[1]=1
        # V: B11=1 -> V[v_b11]=1
        # C contributions: C00+=-1, C01+=1
        U[0][col] = 1
        U[1][col] = 1
        V[v_b11][col] = 1
        W[w_c00][col] = -1
        W[w_c01][col] = 1
        col += 1
        
        # M6 = (A10-A00)(B00+B10)
        # U: A10=1, A00=-1 -> U[2]=1, U[0]=-1
        # V: B00=1, B10=1 -> V[v_b00]=1, V[v_b10]=1
        # C contributions: C10+=1, C11+=-1
        U[2][col] = 1
        U[0][col] = -1
        V[v_b00][col] = 1
        V[v_b10][col] = 1
        W[w_c10][col] = 1
        W[w_c11][col] = -1
        col += 1
        
        # M7 = (A01-A11)(B01+B11)
        # U: A01=1, A11=-1 -> U[1]=1, U[3]=-1
        # V: B01=1, B11=1 -> V[v_b01]=1, V[v_b11]=1
        # C contributions: C00+=1, C01+=-1
        U[1][col] = 1
        U[3][col] = -1
        V[v_b01][col] = 1
        V[v_b11][col] = 1
        W[w_c00][col] = 1
        W[w_c01][col] = -1
        col += 1
    
    # Schoolbook for remaining 7 columns (p=8..14)
    for k in range(8, p):
        for i in range(n):
            for j in range(m):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
