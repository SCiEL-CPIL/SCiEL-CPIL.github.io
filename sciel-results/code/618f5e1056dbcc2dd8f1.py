"""Matrix multiplication 2x4x5 with reduced rank decomposition."""
import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 5
    # Split middle dim into two blocks of size 2
    # A is 2x4 = [A1 | A2] where A1, A2 are 2x2
    # B is 4x5 = [B1; B2] where B1, B2 are 2x5
    # C = A1*B1 + A2*B2, each term is 2x2*2x5 = 2x5
    
    # For each 2x2x5 block, use Strassen on first 2 cols (rank 7) + schoolbook on last 3 cols (rank 6)
    # Total per block: 13, total rank: 26
    
    rank = 26
    U = [[0] * rank for _ in range(n * m)]  # 8 x 26
    V = [[0] * rank for _ in range(m * p)]  # 20 x 26
    W = [[0] * rank for _ in range(p * n)]  # 10 x 26
    
    def add_term(u_row, v_row, w_row, val):
        U[u_row][col] = val
        V[v_row][col] = val
        W[w_row][col] = val
    
    col = 0
    for block in range(2):
        # A_block is 2x2, B_block is 2x5
        # A rows: block*2 + 0, block*2 + 1 (but A is 2x4, so row i of A, col j)
        # A[i, j] where i in {0,1}, j in {block*2, block*2+1}
        # B[j, k] where j in {block*2, block*2+1}, k in {0..4}
        
        # Strassen on first 2 columns of B_block (cols 0,1 of the 2x5)
        # Let A_b = [[a00,a01],[a10,a11]], B_b first 2 cols = [[b00,b01],[b10,b11]]
        # M1 = (a00+a10)*(b00+b10)
        # M2 = (a10+a11)*b00
        # M3 = a00*(b01-b11)
        # M4 = a11*(b10+b00)  -- wait standard Strassen
        # Standard: M1=(A11+A22)(B11+B22), M2=(A21+A22)B11, M3=A11(B21-B22), M4=A22(B12-B11), M5=(A11+A12)B22, M6=(A21-A11)(B11+B12), M7=(A12-A22)(B21+B22)
        
        # For block, A indices: rows 0,1; cols block*2, block*2+1
        # B indices: rows block*2, block*2+1; cols 0,1 (first 2 of the 5)
        
        a00_row = 0  # A row 0
        a01_row = 1  # A col 1 within block
        a10_row = 2  # A row 1, col 0 within block -> but A is stored as [i*m+j, ...] = [i*4+j, ...]
        
        # U index: i*m + j where i is A row, j is A col
        # A is 2x4, so U rows are 0..7
        # B is 4x5, V rows are 0..19, index j*p+k
        # W is 5x2, index k*n+i
        
        # Strassen terms for block
        # M1 = (A[0,block*2] + A[1,block*2+1]) * (B[block*2,0] + B[block*2+1,1])
        # This is getting complex. Let me just enumerate all 13 terms per block explicitly.
        
        pass
    
    # Let me just build it more directly
    U = [[0] * rank for _ in range(8)]
    V = [[0] * rank for _ in range(20)]
    W = [[0] * rank for _ in range(10)]
    
    col = 0
    for block in range(2):
        # Strassen 7 terms for 2x2x2 part
        # A block: rows {0,1}, cols {block*2, block*2+1}
        # B block first 2 cols: rows {block*2, block*2+1}, cols {0,1}
        
        # M1 = (A[0,c0]+A[1,c1])*(B[r0,0]+B[r1,1])
        # where c0=block*2, c1=block*2+1, r0=block*2, r1=block*2+1
        c0 = block * 2
        c1 = block * 2 + 1
        r0 = block * 2
        r1 = block * 2 + 1
        
        # M1
        U[0*4+c0][col] += 1
        U[1*4+c1][col] += 1
        V[r0*5+0][col] += 1
        V[r1*5+1][col] += 1
        # Result contributes to C[0,0], C[0,1], C[1,0], C[1,1] with appropriate signs
        # C[0,0] += M1, C[0,1] -= M1... this is getting messy.
        
        col += 1
    
    # This approach is error-prone. Let me use a cleaner construction.
    # I'll return a valid but higher rank solution first, then try to improve.
    
    # Fallback: schoolbook rank 40
    rank_fb = n * m * p
    U_fb = [[0] * rank_fb for _ in range(n * m)]
    V_fb = [[0] * rank_fb for _ in range(m * p)]
    W_fb = [[0] * rank_fb for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][column] = 1
                V_fb[j * p + k][column] = 1
                W_fb[k * n + i][column] = 1
                column += 1
    return {"U": U_fb, "V": V_fb, "W": W_fb}
# EVOLVE-BLOCK-END
