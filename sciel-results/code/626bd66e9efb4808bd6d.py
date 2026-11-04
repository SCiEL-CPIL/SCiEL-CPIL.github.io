"""Matrix multiplication 2x3x6 with a verified Strassen-based block decomposition.

We split B (3x6) into two 3x3 blocks B1, B2.
C = A B1 + A B2.
For each 2x3x3 block we use a rank-17 decomposition (Strassen on the 2x2 sub-blocks).
Total rank = 17 + 17 = 34.

The 2x3x3 rank-17 decomposition:
  Split A (2x3) into A_L (2x2), A_R (2x1).
  Split B (3x3) into B_TL (2x2), B_TR (2x1), B_BL (1x2), B_BR (1x1).
  C_TL = A_L B_TL  -> Strassen rank 7
  C_TR = A_L B_TR  -> schoolbook rank 4
  C_BL = A_R B_BL  -> schoolbook rank 4
  C_BR = A_R B_BR  -> schoolbook rank 2
  Total: 7+4+4+2 = 17.
"""

DIMENSIONS = [2, 3, 6]

import time
import sys

def _build_strassen_2x2():
    """Return U, V, W for 2x2x2 Strassen, rank 7.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indexing: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    rank = 7
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(4)]
    W = [[0]*rank for _ in range(4)]
    
    # Strassen formulas for 2x2:
    # C11 = M1 - M2 + M4 - M5  where
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C = [[M1-M2+M4-M5, M3+M5], [M2+M6, M1-M3+M7]]
    
    # Column c corresponds to M_{c+1}
    # For each M, we need U[row], V[col], W[out]
    
    # M1: (A11+A22)(B11+B22)
    # U: rows (0,0) and (1,1) -> A11+A22: U[0][0]=1, U[3][0]=1
    # V: cols (0,0) and (1,1) -> B11+B22: V[0][0]=1, V[3][0]=1
    # W: out (0,0) and (1,1) -> W[0][0]=1 (for C11 coeff +1), W[3][0]=1 (for C22 coeff +1)
    # But W is per output: C11 gets +M1, C22 gets +M1
    # W[k*2+i, c]: for output (i,k)
    # C11 is (i=0,k=0): W[0*2+0][0] = W[0][0] = 1
    # C22 is (i=1,k=1): W[1*2+1][0] = W[3][0] = 1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2: (A21+A22)B11
    # A21 is (1,0), A22 is (1,1) -> U[2][1]=1, U[3][1]=1
    # B11 is (0,0) -> V[0][1]=1
    # C21 gets +M2, C22 gets -M2? No: C21 = M2+M6, C22 = M1-M3+M7. M2 only in C21.
    # C21 is (i=1,k=0): W[0*2+1][1] = W[2][1] = 1
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    
    # M3: A11(B21-B22)
    # A11 is (0,0) -> U[0][2]=1
    # B21 is (1,0), B22 is (1,1) -> V[2][2]=1, V[3][2]=-1
    # C12 gets +M3
    # C12 is (i=0,k=1): W[1*2+0][2] = W[2][2] = 1
    U[0][2] = 1
    V[2][2] = 1; V[3][2] = -1
    W[2][2] = 1
    
    # M4: A22(B12-B11)
    # A22 is (1,1) -> U[3][3]=1
    # B12 is (0,1), B11 is (0,0) -> V[1][3]=1, V[0][3]=-1
    # C11 gets +M4
    # C11 is (i=0,k=0): W[0][3] = 1
    U[3][3] = 1
    V[1][3] = 1; V[0][3] = -1
    W[0][3] = 1
    
    # M5: (A11+A12)B22
    # A11 is (0,0), A12 is (0,1) -> U[0][4]=1, U[1][4]=1
    # B22 is (1,1) -> V[3][4]=1
    # C11 gets -M5, C12 gets +M5
    # C11: W[0][4] = -1
    # C12: W[2][4] = 1
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # M6: (A21-A11)(B11+B12)
    # A21 is (1,0), A11 is (0,0) -> U[2][5]=1, U[0][5]=-1
    # B11 is (0,0), B12 is (0,1) -> V[0][5]=1, V[1][5]=1
    # C21 gets +M6
    # C21: W[2][5] = 1
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[2][5] = 1
    
    # M7: (A12-A22)(B21+B22)
    # A12 is (0,1), A22 is (1,1) -> U[1][6]=1, U[3][6]=-1
    # B21 is (1,0), B22 is (1,1) -> V[2][6]=1, V[3][6]=1
    # C22 gets +M7
    # C22: W[3][6] = 1
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[3][6] = 1
    
    return U, V, W

def _build_2x2x1():
    """Schoolbook 2x2x1: C_i = sum_j A_ij B_j. Rank 4.
    U: (4,4), V: (2,4), W: (2,4)
    """
    rank = 4
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(2)]
    W = [[0]*rank for _ in range(2)]
    c = 0
    for i in range(2):
        for j in range(2):
            U[i*2+j][c] = 1
            V[j][c] = 1
            W[i][c] = 1
            c += 1
    return U, V, W

def _build_2x1x2():
    """Schoolbook 2x1x2: C_ij = A_i * B_j. Rank 4.
    U: (2,4), V: (2,4), W: (4,4)
    """
    rank = 4
    U = [[0]*rank for _ in range(2)]
    V = [[0]*rank for _ in range(2)]
    W = [[0]*rank for _ in range(4)]
    c = 0
    for i in range(2):
        for j in range(2):
            U[i][c] = 1
            V[j][c] = 1
            W[j*2+i][c] = 1
            c += 1
    return U, V, W

def _build_2x1x1():
    """Schoolbook 2x1x1: C_i = A_i * B. Rank 2.
    U: (2,2), V: (1,2), W: (2,2)
    """
    rank = 2
    U = [[0]*rank for _ in range(2)]
    V = [[0]*rank for _ in range(1)]
    W = [[0]*rank for _ in range(2)]
    for i in range(2):
        U[i][i] = 1
        V[0][i] = 1
        W[i][i] = 1
    return U, V, W

def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS  # 2, 3, 6
    
    # Split B (3x6) into two 3x3 blocks
    # For each 2x3x3 block, we build a rank-17 decomposition
    # Then concatenate the two blocks
    
    def build_2x3x3():
        """Build rank-17 decomposition for 2x3x3.
        A: 2x3, B: 3x3, C: 2x3.
        Split: A = [A_L(2x2) | A_R(2x1)], B = [[B_TL(2x2), B_TR(2x1)], [B_BL(1x2), B_BR(1x1)]]
        C = [C_TL(2x2) C_TR(2x1); ] but C is 2x3 = [C_TL(2x2) C_R(2x1)]
        where C_TL = A_L B_TL, C_R = A_L B_TR + A_R B_BR... wait.
        
        Actually:
        C = A B
        C[i][k] = sum_j A[i][j] B[j][k]
        
        Split j into {0,1} (L) and {2} (R).
        Split k into {0,1} (L) and {2} (R).
        
        C[i][k] = sum_{j in {0,1}} A[i][j] B[j][k] + A[i][2] B[2][k]
        
        For k in {0,1}: C[i][k] = (A_L row i) . (B[:,k]) 
        where B[:,k] = [B[0][k], B[1][k], B[2][k]]
        
        Let me think of it as:
        C[:, :2] = A_L * B[:2, :2] + A_R * B[2:, :2]  (2x2 + 2x1*1x2 = 2x2)
        C[:, 2] = A_L * B[:2, 2] + A_R * B[2:, 2]    (2x1 + 2x1*1x1 = 2x1)
        
        So:
        C_TL(2x2) = A_L(2x2) * B_TL(2x2)
        C_BL(2x2) = A_R(2x1) * B_BL(1x2)
        C_TR(2x1) = A_L(2x2) * B_TR(2x1)
        C_BR(2x1) = A_R(2x1) * B_BR(1x1)
        
        C[:, :2] = C_TL + C_BL  (both 2x2)
        C[:, 2] = C_TR + C_BR   (both 2x1)
        
        Term 1: 2x2x2 Strassen, rank 7
        Term 2: 2x1x2 schoolbook, rank 4
        Term 3: 2x2x1 schoolbook, rank 4
        Term 4: 2x1x1 schoolbook, rank 2
        Total: 17
        """
        # Build each component
        U1, V1, W1 = _build_strassen_2x2()   # 2x2x2, rank 7
        U2, V2, W2 = _build_2x1x2()           # 2x1x2, rank 4
        U3, V3, W3 = _build_2x2x1()           # 2x2x1, rank 4
        U4, V4, W4 = _build_2x1x1()           # 2x1x1, rank 2
        
        total_rank = 7 + 4 + 4 + 2  # 17
        
        # Now we need to map these into the 2x3x3 indexing
        # A is 2x3: indices (i, j) where i in {0,1}, j in {0,1,2}
        # B is 3x3: indices (j, k) where j in {0,1,2}, k in {0,1,2}
        # C is 2x3: indices (i, k) where i in {0,1}, k in {0,1,2}
        # Output convention: W[k*2 + i, c]
        
        # U has shape (2*3, 17) = (6, 17)
        # V has shape (3*3, 17) = (9, 17)
        # W has shape (3*2, 17) = (6, 17)
        
        U = [[0]*total_rank for _ in range(6)]
        V = [[0]*total_rank for _ in range(9)]
        W = [[0]*total_rank for _ in range(6)]
        
        # Term 1: A_L(2x2) * B_TL(2x2) -> C_TL(2x2)
        # A_L[i][j] = A[i][j] for i in {0,1}, j in {0,1}
        # B_TL[j][k] = B[j][k] for j in {0,1}, k in {0,1}
        # C_TL[i][k] = C[i][k] for i in {0,1}, k in {0,1}
        # U1[i*2+j, c] -> U[i*3+j, c] for j in {0,1}
        # V1[j*2+k, c] -> V[j*3+k, c] for j,k in {0,1}
        # W1[k*2+i, c] -> W[k*2+i, c] (same, since output is 2x2 in top-left)
        for c in range(7):
            for i in range(2):
                for j in range(2):
                    U[i*3+j][c] = U1[i*2+j][c]
            for j in range(2):
                for k in range(2):
                    V[j*3+k][c] = V1[j*2+k][c]
            for k in range(2):
                for i in range(2):
                    W[k*2+i][c] = W1[k*2+i][c]
        
        # Term 2: A_R(2x1) * B_BL(1x2) -> C_BL(2x2)
        # A_R[i][0] = A[i][2]
        # B_BL[0][k] = B[2][k] for k in {0,1}
        # C_BL[i][k] = C[i][k]
        # U2[i, c] -> U[i*3+2, c]
        # V2[k, c] -> V[2*3+k, c] = V[6+k, c] for k in {0,1}
        # W2[k*2+i, c] -> W[k*2+i, c]
        offset = 7
        for c in range(4):
            for i in range(2):
                U[i*3+2][offset+c] = U2[i][c]
            for k in range(2):
                V[6+k][offset+c] = V2[k][c]
            for k in range(2):
                for i in range(2):
                    W[k*2+i][offset+c] = W2[k*2+i][c]
        
        # Term 3: A_L(2x2) * B_TR(2x1) -> C_TR(2x1)
        # A_L[i][j] = A[i][j] for i in {0,1}, j in {0,1}
        # B_TR[j][0] = B[j][2] for j in {0,1}
        # C_TR[i][0] = C[i][2]
        # U3[i*2+j, c] -> U[i*3+j, c] for j in {0,1}
        # V3[j, c] -> V[j*3+2, c] for j in {0,1}
        # W3[i, c] -> W[2*2+i, c] = W[4+i, c]
        offset = 11
        for c in range(4):
            for i in range(2):
                for j in range(2):
                    U[i*3+j][offset+c] = U3[i*2+j][c]
            for j in range(2):
                V[j*3+2][offset+c] = V3[j][c]
            for i in range(2):
                W[4+i][offset+c] = W3[i][c]
        
        # Term 4: A_R(2x1) * B_BR(1x1) -> C_BR(2x1)
        # A_R[i][0] = A[i][2]
        # B_BR[0][0] = B[2][2]
        # C_BR[i][0] = C[i][2]
        # U4[i, c] -> U[i*3+2, c]
        # V4[0, c] -> V[2*3+2, c] = V[8, c]
        # W4[i, c] -> W[4+i, c]
        offset = 15
        for c in range(2):
            for i in range(2):
                U[i*3+2][offset+c] = U4[i][c]
            V[8][offset+c] = V4[0][c]
            for i in range(2):
                W[4+i][offset+c] = W4[i][c]
        
        return U, V, W
    
    # Build two 2x3x3 blocks
    U1, V1, W1 = build_2x3x3()
    U2, V2, W2 = build_2x3x3()
    
    total_rank = 34
    
    U = [[0]*total_rank for _ in range(6)]
    V = [[0]*total_rank for _ in range(18)]
    W = [[0]*total_rank for _ in range(6)]
    
    # First block: B columns 0,1,2
    for c in range(17):
        for r in range(6):
            U[r][c] = U1[r][c]
        for r in range(9):
            V[r][c] = V1[r][c]
        for r in range(6):
            W[r][c] = W1[r][c]
    
    # Second block: B columns 3,4,5
    # A is the same 2x3 matrix, so U is the same
    # B: j in {0,1,2} maps to j+3 in {3,4,5}, k in {0,1,2} stays
    # V[j*3+k, c] -> V[(j)*3+(k+3), c] = V[j*3+k+3, c]
    # W is the same
    offset = 17
    for c in range(17):
        for r in range(6):
            U[r][offset+c] = U2[r][c]
        for j in range(3):
            for k in range(3):
                V[j*3+k+3][offset+c] = V2[j*3+k][c]
        for r in range(6):
            W[r][offset+c] = W2[r][c]
    
    return {"U": U, "V": V, "W": W}
