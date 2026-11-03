import time
import sys

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 46 decomposition
    # 1. A1 (2x2) * B1 (2x5) -> C1 (2x2) part 1 (cols 0,1)
    # 2. A1 (2x2) * B1 (2x5) -> C1 (2x2) part 2 (cols 2,3)
    # 3. A1 (2x2) * B1 (2x5) -> C1 (2x1) part 3 (col 4)
    # 4. A21 (2x2) * B21 (2x5) -> C21 (2x2) part 1 (cols 0,1)
    # 5. A21 (2x2) * B21 (2x5) -> C21 (2x2) part 2 (cols 2,3)
    # 6. A21 (2x2) * B21 (2x5) -> C21 (2x1) part 3 (col 4)
    # 7. A22 (2x1) * B22 (1x5) -> C22 (2x5)
    
    rank = 46
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a 2x2x2 Strassen block
    # u_base: index of first row of A block in U
    # v_base: index of first row of B block in V
    # w_base: index of first row of C block in W
    # w_cols: list of 2 column indices in C for this block
    def add_strassen_222(u_base, v_base, w_col_indices):
        nonlocal col
        # Strassen terms
        # m1
        U[u_base][col] = 1; U[u_base+1][col] = 1
        V[v_base][col] = 1; V[v_base+2][col] = 1
        W[w_col_indices[0]*n + 0][col] = 1
        W[w_col_indices[0]*n + 1][col] = 1
        col += 1
        # m2
        U[u_base+1][col] = 1
        V[v_base][col] = 1
        W[w_col_indices[0]*n + 1][col] = 1
        col += 1
        # m3
        U[u_base][col] = 1
        V[v_base+1][col] = 1; V[v_base+2][col] = -1
        W[w_col_indices[0]*n + 1][col] = 1
        col += 1
        # m4
        U[u_base+1][col] = 1
        V[v_base][col] = -1; V[v_base+2][col] = 1
        W[w_col_indices[0]*n + 1][col] = 1
        col += 1
        # m5
        U[u_base][col] = 1; U[u_base+1][col] = 1
        V[v_base+2][col] = 1
        W[w_col_indices[0]*n + 0][col] = -1
        W[w_col_indices[0]*n + 1][col] = 1
        col += 1
        # m6
        U[u_base][col] = -1; U[u_base+1][col] = 1
        V[v_base][col] = 1; V[v_base+1][col] = 1
        W[w_col_indices[1]*n + 1][col] = 1
        col += 1
        # m7
        U[u_base+1][col] = -1
        V[v_base+2][col] = 1; V[v_base+3][col] = 1 # Wait, B is 2x2, so v_base+2 and v_base+3?
        # B indices: v_base (00), v_base+1 (01), v_base+2 (10), v_base+3 (11)?
        # V is m*p = 5*5 = 25.
        # B block is 2x5? No, B block is 2x2 for the 222 part.
        # In 2x2x2, B is 2x2.
        # So v_base, v_base+1, v_base+2, v_base+3?
        # If B is 2x5, the 2x2 block is rows 0,1 and cols 0,1.
        # Indices: 0, 1, 5, 6.
        # So v_base, v_base+1, v_base+5, v_base+6.
        
        # Let's re-verify V indexing.
        # V is (m*p, rank). m=5, p=5.
        # B[j,k] is at index j*p + k.
        # For a 2x2 block at rows r1,r2 and cols c1,c2:
        # Indices: r1*p+c1, r1*p+c2, r2*p+c1, r2*p+c2.
        
        # I need to pass the actual indices.
        
    # This helper is getting complex with indices. I will write it out explicitly.
    
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # A indices: 0,1,5,6
    # B indices: 0,1,5,6
    # C indices: 0,1,2,3 (W index k*n+i)
    # C[0,0]->0, C[1,0]->1, C[0,1]->2, C[1,1]->3
    
    # m1
    U[0][col]=1; U[6][col]=1
    V[0][col]=1; V[6][col]=1
    W[0][col]=1; W[3][col]=1
    col+=1
    # m2
    U[6][col]=1
    V[0][col]=1
    W[1][col]=1
    col+=1
    # m3
    U[0][col]=1
    V[1][col]=1; V[6][col]=-1
    W[2][col]=1
    col+=1
    # m4
    U[6][col]=1
    V[0][col]=-1; V[5][col]=1
    W[1][col]=1
    col+=1
    # m5
    U[0][col]=1; U[1][col]=1
    V[6][col]=1
    W[0][col]=-1; W[2][col]=1
    col+=1
    # m6
    U[0][col]=-1; U[6][col]=1
    V[0][col]=1; V[1][col]=1
    W[3][col]=1
    col+=1
    # m7
    U[1][col]=-1
    V[5][col]=1; V[6][col]=1
    W[0][col]=1
    col+=1
    
    # Block 2: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    # A indices: 0,1,5,6
    # B indices: 2,3,7,8
    # C indices: 4,5,6,7 (k=2,3)
    
    # m1
    U[0][col]=1; U[6][col]=1
    V[2][col]=1; V[8][col]=1
    W[4][col]=1; W[7][col]=1
    col+=1
    # m2
    U[6][col]=1
    V[2][col]=1
    W[5][col]=1
    col+=1
    # m3
    U[0][col]=1
    V[3][col]=1; V[8][col]=-1
    W[6][col]=1
    col+=1
    # m4
    U[6][col]=1
    V[2][col]=-1; V[7][col]=1
    W[5][col]=1
    col+=1
    # m5
    U[0][col]=1; U[1][col]=1
    V[8][col]=1
    W[4][col]=-1; W[6][col]=1
    col+=1
    # m6
    U[0][col]=-1; U[6][col]=1
    V[2][col]=1; V[3][col]=1
    W[7][col]=1
    col+=1
    # m7
    U[1][col]=-1
    V[7][col]=1; V[8][col]=1
    W[4][col]=1
    col+=1
    
    # Block 3: A[0:2, 0:2] * B[0:2, 4:5] -> C[0:2, 4:5]
    # A indices: 0,1,5,6
    # B indices: 4,9
    # C indices: 8,9 (k=4)
    # Rank 4
    # 1. A00 B00
    U[0][col]=1
    V[4][col]=1
    W[8][col]=1
    col+=1
    # 2. A01 B10
    U[1][col]=1
    V[9][col]=1
    W[8][col]=1
    col+=1
    # 3. A10 B00
    U[5][col]=1
    V[4][col]=1
    W[9][col]=1
    col+=1
    # 4. A11 B10
    U[6][col]=1
    V[9][col]=1
    W[9][col]=1
    col+=1
    
    # Block 4: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    # A indices: 2,3,7,8
    # B indices: 10,11,15,16
    # C indices: 0,1,2,3
    
    # m1
    U[2][col]=1; U[8][col]=1
    V[10][col]=1; V[16][col]=1
    W[0][col]=1; W[3][col]=1
    col+=1
    # m2
    U[8][col]=1
    V[10][col]=1
    W[1][col]=1
    col+=1
    # m3
    U[2][col]=1
    V[11][col]=1; V[16][col]=-1
    W[2][col]=1
    col+=1
    # m4
    U[8][col]=1
    V[10][col]=-1; V[15][col]=1
    W[1][col]=1
    col+=1
    # m5
    U[2][col]=1; U[3][col]=1
    V[16][col]=1
    W[0][col]=-1; W[2][col]=1
    col+=1
    # m6
    U[2][col]=-1; U[8][col]=1
    V[10][col]=1; V[11][col]=1
    W[3][col]=1
    col+=1
    # m7
    U[3][col]=-1
    V[15][col]=1; V[16][col]=1
    W[0][col]=1
    col+=1
    
    # Block 5: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    # A indices: 2,3,7,8
    # B indices: 12,13,17,18
    # C indices: 4,5,6,7
    
    # m1
    U[2][col]=1; U[8][col]=1
    V[12][col]=1; V[18][col]=1
    W[4][col]=1; W[7][col]=1
    col+=1
    # m2
    U[8][col]=1
    V[12][col]=1
    W[5][col]=1
    col+=1
    # m3
    U[2][col]=1
    V[13][col]=1; V[18][col]=-1
    W[6][col]=1
    col+=1
    # m4
    U[8][col]=1
    V[12][col]=-1; V[17][col]=1
    W[5][col]=1
    col+=1
    # m5
    U[2][col]=1; U[3][col]=1
    V[18][col]=1
    W[4][col]=-1; W[6][col]=1
    col+=1
    # m6
    U[2][col]=-1; U[8][col]=1
    V[12][col]=1; V[13][col]=1
    W[7][col]=1
    col+=1
    # m7
    U[3][col]=-1
    V[17][col]=1; V[18][col]=1
    W[4][col]=1
    col+=1
    
    # Block 6: A[0:2, 2:4] * B[2:4, 4:5] -> C[0:2, 4:5]
    # A indices: 2,3,7,8
    # B indices: 14,19
    # C indices: 8,9
    
    # 1. A00 B00
    U[2][col]=1
    V[14][col]=1
    W[8][col]=1
    col+=1
    # 2. A01 B10
    U[3][col]=1
    V[19][col]=1
    W[8][col]=1
    col+=1
    # 3. A10 B00
    U[7][col]=1
    V[14][col]=1
    W[9][col]=1
    col+=1
    # 4. A11 B10
    U[8][col]=1
    V[19][col]=1
    W[9][col]=1
    col+=1
    
    # Block 7: A[0:2, 4:5] * B[4:5, 0:5] -> C[0:2, 0:5]
    # A indices: 4,9
    # B indices: 20,21,22,23,24
    # C indices: 0..9
    
    # 10 terms
    for k in range(5):
        # A00 B0k
        U[4][col]=1
        V[20+k][col]=1
        W[k*2][col]=1
        col+=1
        # A10 B0k
        U[9][col]=1
        V[20+k][col]=1
        W[k*2+1][col]=1
        col+=1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
