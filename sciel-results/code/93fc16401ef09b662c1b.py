"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=3, m=4, p=5
    # U shape: (12, rank)
    # V shape: (20, rank)
    # W shape: (15, rank)
    
    rank = 58
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Define the sets of (i, j, k) covered by Strassen blocks
    # Block 1: i in {0,1}, j in {0,1}, k in {0,1}
    # Block 2: i in {1,2}, j in {2,3}, k in {2,3}
    
    strassen_block_1_indices = set()
    for i in [0, 1]:
        for j in [0, 1]:
            for k in [0, 1]:
                strassen_block_1_indices.add((i, j, k))
                
    strassen_block_2_indices = set()
    for i in [1, 2]:
        for j in [2, 3]:
            for k in [2, 3]:
                strassen_block_2_indices.add((i, j, k))
                
    # 1. Fill Schoolbook for remaining indices
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) in strassen_block_1_indices:
                    continue
                if (i, j, k) in strassen_block_2_indices:
                    continue
                
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
                
    # 2. Fill Strassen Block 1
    # i in {0,1}, j in {0,1}, k in {0,1}
    # A indices: (0,0), (0,1), (1,0), (1,1)
    # B indices: (0,0), (0,1), (1,0), (1,1)
    # C indices: (0,0), (0,1), (1,0), (1,1)
    
    # Local indices for U, V, W
    # U: (i, j) -> i*4+j
    # V: (j, k) -> j*5+k
    # W: (k, i) -> k*3+i
    
    # M1 = (A00 + A11)(B00 + B11) -> C00 += M1, C11 += M1
    # U: (0,0)=1, (1,1)=1
    # V: (0,0)=1, (1,1)=1
    # W: (0,0)=1, (1,1)=1
    U[0*4+0][col] = 1; U[1*4+1][col] = 1
    V[0*5+0][col] = 1; V[1*5+1][col] = 1
    W[0*3+0][col] = 1; W[1*3+1][col] = 1
    col += 1
    
    # M2 = (A10 + A11)(B00) -> C10 += M2, C11 -= M2
    # U: (1,0)=1, (1,1)=1
    # V: (0,0)=1
    # W: (0,1)=1, (1,1)=-1
    U[1*4+0][col] = 1; U[1*4+1][col] = 1
    V[0*5+0][col] = 1
    W[0*3+1][col] = 1; W[1*3+1][col] = -1
    col += 1
    
    # M3 = (A00)(B01 - B11) -> C01 += M3, C11 -= M3
    # U: (0,0)=1
    # V: (0,1)=1, (1,1)=-1
    # W: (1,0)=1, (1,1)=-1
    U[0*4+0][col] = 1
    V[0*5+1][col] = 1; V[1*5+1][col] = -1
    W[1*3+0][col] = 1; W[1*3+1][col] = -1
    col += 1
    
    # M4 = (A11)(B10 - B00) -> C00 += M4, C10 += M4
    # U: (1,1)=1
    # V: (1,0)=1, (0,0)=-1
    # W: (0,0)=1, (0,1)=1
    U[1*4+1][col] = 1
    V[1*5+0][col] = 1; V[0*5+0][col] = -1
    W[0*3+0][col] = 1; W[0*3+1][col] = 1
    col += 1
    
    # M5 = (A00 + A01)(B11) -> C01 += M5, C11 += M5
    # U: (0,0)=1, (0,1)=1
    # V: (1,1)=1
    # W: (1,0)=1, (1,1)=1
    U[0*4+0][col] = 1; U[0*4+1][col] = 1
    V[1*5+1][col] = 1
    W[1*3+0][col] = 1; W[1*3+1][col] = 1
    col += 1
    
    # M6 = (A10 - A00)(B00 + B01) -> C00 += M6, C10 -= M6
    # U: (1,0)=1, (0,0)=-1
    # V: (0,0)=1, (0,1)=1
    # W: (0,0)=1, (0,1)=-1
    U[1*4+0][col] = 1; U[0*4+0][col] = -1
    V[0*5+0][col] = 1; V[0*5+1][col] = 1
    W[0*3+0][col] = 1; W[0*3+1][col] = -1
    col += 1
    
    # M7 = (A01 - A11)(B10 + B11) -> C01 -= M7, C11 += M7
    # U: (0,1)=1, (1,1)=-1
    # V: (1,0)=1, (1,1)=1
    # W: (1,0)=-1, (1,1)=1
    U[0*4+1][col] = 1; U[1*4+1][col] = -1
    V[1*5+0][col] = 1; V[1*5+1][col] = 1
    W[1*3+0][col] = -1; W[1*3+1][col] = 1
    col += 1
    
    # 3. Fill Strassen Block 2
    # i in {1,2}, j in {2,3}, k in {2,3}
    # A indices: (1,2), (1,3), (2,2), (2,3)
    # B indices: (2,2), (2,3), (3,2), (3,3)
    # C indices: (1,2), (1,3), (2,2), (2,3)
    
    # Mapping to local 2x2 indices:
    # a = A[1,2], b = A[1,3], c = A[2,2], d = A[2,3]
    # e = B[2,2], f = B[2,3], g = B[3,2], h = B[3,3]
    # y1 = C[1,2], y2 = C[1,3], y3 = C[2,2], y4 = C[2,3]
    
    # M1' = (a+d)(e+h) -> y1 += M1', y4 += M1'
    # U: (1,2)=1, (2,3)=1
    # V: (2,2)=1, (3,3)=1
    # W: (2,1)=1, (3,2)=1  (k=2,i=1 -> 2*3+1=7; k=3,i=2 -> 3*3+2=11)
    U[1*4+2][col] = 1; U[2*4+3][col] = 1
    V[2*5+2][col] = 1; V[3*5+3][col] = 1
    W[2*3+1][col] = 1; W[3*3+2][col] = 1
    col += 1
    
    # M2' = (c+d)(e) -> y3 += M2', y4 -= M2'
    # U: (2,2)=1, (2,3)=1
    # V: (2,2)=1
    # W: (2,2)=1, (3,2)=-1  (k=2,i=2 -> 8; k=3,i=2 -> 11)
    U[2*4+2][col] = 1; U[2*4+3][col] = 1
    V[2*5+2][col] = 1
    W[2*3+2][col] = 1; W[3*3+2][col] = -1
    col += 1
    
    # M3' = (a)(f-h) -> y2 += M3', y4 -= M3'
    # U: (1,2)=1
    # V: (2,3)=1, (3,3)=-1
    # W: (3,1)=1, (3,2)=-1  (k=3,i=1 -> 10; k=3,i=2 -> 11)
    U[1*4+2][col] = 1
    V[2*5+3][col] = 1; V[3*5+3][col] = -1
    W[3*3+1][col] = 1; W[3*3+2][col] = -1
    col += 1
    
    # M4' = (d)(g-c) -> y1 += M4', y3 += M4'
    # U: (2,3)=1
    # V: (3,2)=1, (2,2)=-1
    # W: (2,1)=1, (2,2)=1  (k=2,i=1 -> 7; k=2,i=2 -> 8)
    U[2*4+3][col] = 1
    V[3*5+2][col] = 1; V[2*5+2][col] = -1
    W[2*3+1][col] = 1; W[2*3+2][col] = 1
    col += 1
    
    # M5' = (a+b)(h) -> y2 += M5', y4 += M5'
    # U: (1,2)=1, (1,3)=1
    # V: (3,3)=1
    # W: (3,1)=1, (3,2)=1  (k=3,i=1 -> 10; k=3,i=2 -> 11)
    U[1*4+2][col] = 1; U[1*4+3][col] = 1
    V[3*5+3][col] = 1
    W[3*3+1][col] = 1; W[3*3+2][col] = 1
    col += 1
    
    # M6' = (c-a)(e+f) -> y1 += M6', y3 -= M6'
    # U: (2,2)=1, (1,2)=-1
    # V: (2,2)=1, (2,3)=1
    # W: (2,1)=1, (2,2)=-1  (k=2,i=1 -> 7; k=2,i=2 -> 8)
    U[2*4+2][col] = 1; U[1*4+2][col] = -1
    V[2*5+2][col] = 1; V[2*5+3][col] = 1
    W[2*3+1][col] = 1; W[2*3+2][col] = -1
    col += 1
    
    # M7' = (b-d)(g+h) -> y2 -= M7', y4 += M7'
    # U: (1,3)=1, (2,3)=-1
    # V: (3,2)=1, (3,3)=1
    # W: (3,1)=-1, (3,2)=1  (k=3,i=1 -> 10; k=3,i=2 -> 11)
    U[1*4+3][col] = 1; U[2*4+3][col] = -1
    V[3*5+2][col] = 1; V[3*5+3][col] = 1
    W[3*3+1][col] = -1; W[3*3+2][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

