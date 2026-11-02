import time
import numpy as np

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][c] = 1
                V_school[j * p + k][c] = 1
                W_school[k * n + i][c] = 1
                c += 1
                
    # Try to construct the rank 66 decomposition
    try:
        # Helper to create Strassen decomposition for 2x2x2
        # Returns U, V, W as numpy arrays of shape (4, 7)
        def strassen_222():
            # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
            U = np.zeros((4, 7), dtype=int)
            V = np.zeros((4, 7), dtype=int)
            W = np.zeros((4, 7), dtype=int)
            
            # M1 = (A00+A11)(B00+B11)
            # U: A00, A11. V: B00, B11. W: C00, C11
            # C00 += M1, C11 += M1
            # U[0,0]=1 (A00), U[3,0]=1 (A11)
            # V[0,0]=1 (B00), V[3,0]=1 (B11)
            # W[0,0]=1 (C00), W[3,0]=1 (C11)
            U[0, 0] = 1; U[3, 0] = 1
            V[0, 0] = 1; V[3, 0] = 1
            W[0, 0] = 1; W[3, 0] = 1
            
            # M2 = (A10+A11)B00
            # U: A10, A11. V: B00. W: C10, C11
            # C10 += M2, C11 -= M2
            U[2, 1] = 1; U[3, 1] = 1
            V[0, 1] = 1
            W[2, 1] = 1; W[3, 1] = -1
            
            # M3 = A00(B10-B11)
            # U: A00. V: B10, B11. W: C01, C11
            # C01 += M3, C11 += M3
            U[0, 2] = 1
            V[1, 2] = 1; V[3, 2] = -1
            W[1, 2] = 1; W[3, 2] = 1
            
            # M4 = A11(B01-B00)
            # U: A11. V: B01, B00. W: C00, C10
            # C00 += M4, C10 += M4
            U[3, 3] = 1
            V[1, 3] = 1; V[0, 3] = -1
            W[0, 3] = 1; W[2, 3] = 1
            
            # M5 = (A00+A01)B11
            # U: A00, A01. V: B11. W: C00, C01
            # C00 -= M5, C01 += M5
            U[0, 4] = 1; U[1, 4] = 1
            V[3, 4] = 1
            W[0, 4] = -1; W[1, 4] = 1
            
            # M6 = (A10-A00)(B00+B01)
            # U: A10, A00. V: B00, B01. W: C11
            # C11 += M6
            U[2, 5] = 1; U[0, 5] = -1
            V[0, 5] = 1; V[1, 5] = 1
            W[3, 5] = 1
            
            # M7 = (A01-A11)(B10+B11)
            # U: A01, A11. V: B10, B11. W: C00
            # C00 += M7
            U[1, 6] = 1; U[3, 6] = -1
            V[1, 6] = 1; V[3, 6] = 1
            W[0, 6] = 1
            
            return U, V, W

        # Helper to create Schoolbook decomposition for 1x2x2
        # U shape (2, 4), V shape (4, 4), W shape (2, 4)
        # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
        def schoolbook_122():
            U = np.zeros((2, 4), dtype=int)
            V = np.zeros((4, 4), dtype=int)
            W = np.zeros((2, 4), dtype=int)
            # i=0 always.
            # j=0, k=0: U[0,0]=1, V[0,0]=1, W[0,0]=1
            # j=0, k=1: U[0,1]=1, V[1,1]=1, W[1,1]=1
            # j=1, k=0: U[1,2]=1, V[2,2]=1, W[0,2]=1
            # j=1, k=1: U[1,3]=1, V[3,3]=1, W[1,3]=1
            U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
            U[0, 1] = 1; V[1, 1] = 1; W[1, 1] = 1
            U[1, 2] = 1; V[2, 2] = 1; W[0, 2] = 1
            U[1, 3] = 1; V[3, 3] = 1; W[1, 3] = 1
            return U, V, W

        # Initialize global matrices with rank 66
        rank = 66
        U = np.zeros((12, rank), dtype=int)
        V = np.zeros((24, rank), dtype=int)
        W = np.zeros((18, rank), dtype=int)
        
        col = 0
        
        # Part 1: i=0 (1x4x6)
        # Split j into [0,2) and [2,4)
        # Split k into [0,2), [2,4), [4,6)
        
        # Block 1.1: i=0, j in [0,2), k in [0,2) -> 1x2x2
        U11, V11, W11 = schoolbook_122()
        # Map indices
        # U row: i=0, j=0,1 -> 0, 1
        # V row: j=0,1; k=0,1 -> 0, 1, 2, 3
        # W row: k=0,1; i=0 -> 0, 1
        
        U[0:2, col:col+4] = U11
        V[0:4, col:col+4] = V11
        W[0:2, col:col+4] = W11
        col += 4
        
        # Block 1.2: i=0, j in [0,2), k in [2,4) -> 1x2x2
        # V rows: j=0,1; k=2,3 -> 2, 3, 4, 5
        # W rows: k=2,3; i=0 -> 2, 3
        U[0:2, col:col+4] = U11
        V[2:6, col:col+4] = V11
        W[2:4, col:col+4] = W11
        col += 4
        
        # Block 1.3: i=0, j in [0,2), k in [4,6) -> 1x2x2
        # V rows: j=0,1; k=4,5 -> 4, 5, 6, 7
        # W rows: k=4,5; i=0 -> 4, 5
        U[0:2, col:col+4] = U11
        V[4:8, col:col+4] = V11
        W[4:6, col:col+4] = W11
        col += 4
        
        # Block 1.4: i=0, j in [2,4), k in [0,2) -> 1x2x2
        # U rows: i=0, j=2,3 -> 2, 3
        # V rows: j=2,3; k=0,1 -> 8, 9, 10, 11
        # W rows: k=0,1; i=0 -> 0, 1
        U[2:4, col:col+4] = U11
        V[8:12, col:col+4] = V11
        W[0:2, col:col+4] = W11
        col += 4
        
        # Block 1.5: i=0, j in [2,4), k in [2,4) -> 1x2x2
        # V rows: j=2,3; k=2,3 -> 10, 11, 12, 13
        # W rows: k=2,3; i=0 -> 2, 3
        U[2:4, col:col+4] = U11
        V[10:14, col:col+4] = V11
        W[2:4, col:col+4] = W11
        col += 4
        
        # Block 1.6: i=0, j in [2,4), k in [4,6) -> 1x2x2
        # V rows: j=2,3; k=4,5 -> 12, 13, 14, 15
        # W rows: k=4,5; i=0 -> 4, 5
        U[2:4, col:col+4] = U11
        V[12:16, col:col+4] = V11
        W[4:6, col:col+4] = W11
        col += 4
        
        # Part 2: i in [1,3) (2x4x6)
        # i' in [0,2)
        # Split j into [0,2) and [2,4)
        # Split k into [0,2), [2,4), [4,6)
        
        U22, V22, W22 = strassen_222()
        
        # Block 2.1: i' in [0,2), j in [0,2), k in [0,2) -> 2x2x2
        # U rows: i=1,2; j=0,1 -> 4, 5, 6, 7
        # V rows: j=0,1; k=0,1 -> 0, 1, 2, 3
        # W rows: k=0,1; i=1,2 -> 0*2+1=1? No.
        # W index: k*n + i. n=3.
        # k=0, i=1 -> 1. k=0, i=2 -> 2.
        # k=1, i=1 -> 4. k=1, i=2 -> 5.
        # W rows: 1, 2, 4, 5.
        
        U[4:8, col:col+7] = U22
        V[0:4, col:col+7] = V22
        # W mapping:
        # W22 row 0 (k=0, i'=0) -> global i=1, k=0 -> idx 1
        # W22 row 1 (k=0, i'=1) -> global i=2, k=0 -> idx 2
        # W22 row 2 (k=1, i'=0) -> global i=1, k=1 -> idx 4
        # W22 row 3 (k=1, i'=1) -> global i=2, k=1 -> idx 5
        W[1, col:col+7] = W22[0, :]
        W[2, col:col+7] = W22[1, :]
        W[4, col:col+7] = W22[2, :]
        W[5, col:col+7] = W22[3, :]
        col += 7
        
        # Block 2.2: i' in [0,2), j in [0,2), k in [2,4) -> 2x2x2
        # V rows: j=0,1; k=2,3 -> 2, 3, 4, 5
        # W rows: k=2,3; i=1,2 -> 2*3+1=7, 2*3+2=8, 3*3+1=10, 3*3+2=11
        U[4:8, col:col+7] = U22
        V[2:6, col:col+7] = V22
        W[7, col:col+7] = W22[0, :]
        W[8, col:col+7] = W22[1, :]
        W[10, col:col+7] = W22[2, :]
        W[11, col:col+7] = W22[3, :]
        col += 7
        
        # Block 2.3: i' in [0,2), j in [0,2), k in [4,6) -> 2x2x2
        # V rows: j=0,1; k=4,5 -> 4, 5, 6, 7
        # W rows: k=4,5; i=1,2 -> 4*3+1=13, 4*3+2=14, 5*3+1=16, 5*3+2=17
        U[4:8, col:col+7] = U22
        V[4:8, col:col+7] = V22
        W[13, col:col+7] = W22[0, :]
        W[14, col:col+7] = W22[1, :]
        W[16, col:col+7] = W22[2, :]
        W[17, col:col+7] = W22[3, :]
        col += 7
        
        # Block 2.4: i' in [0,2), j in [2,4), k in [0,2) -> 2x2x2
        # U rows: i=1,2; j=2,3 -> 6, 7, 8, 9
        # V rows: j=2,3; k=0,1 -> 8, 9, 10, 11
        # W rows: k=0,1; i=1,2 -> 1, 2, 4, 5
        U[6:10, col:col+7] = U22
        V[8:12, col:col+7] = V22
        W[1, col:col+7] = W22[0, :]
        W[2, col:col+7] = W22[1, :]
        W[4, col:col+7] = W22[2, :]
        W[5, col:col+7] = W22[3, :]
        col += 7
        
        # Block 2.5: i' in [0,2), j in [2,4), k in [2,4) -> 2x2x2
        # V rows: j=2,3; k=2,3 -> 10, 11, 12, 13
        # W rows: k=2,3; i=1,2 -> 7, 8, 10, 11
        U[6:10, col:col+7] = U22
        V[10:14, col:col+7] = V22
        W[7, col:col+7] = W22[0, :]
        W[8, col:col+7] = W22[1, :]
        W[10, col:col+7] = W22[2, :]
        W[11, col:col+7] = W22[3, :]
        col += 7
        
        # Block 2.6: i' in [0,2), j in [2,4), k in [4,6) -> 2x2x2
        # V rows: j=2,3; k=4,5 -> 12, 13, 14, 15
        # W rows: k=4,5; i=1,2 -> 13, 14, 16, 17
        U[6:10, col:col+7] = U22
        V[12:16, col:col+7] = V22
        W[13, col:col+7] = W22[0, :]
        W[14, col:col+7] = W22[1, :]
        W[16, col:col+7] = W22[2, :]
        W[17, col:col+7] = W22[3, :]
        col += 7
        
        if col == rank:
            # Convert to lists
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            return {"U": U_list, "V": V_list, "W": W_list}
            
    except Exception:
        pass
        
    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
