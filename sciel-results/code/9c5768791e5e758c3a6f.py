import numpy as np

DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Rank 32 decomposition
    # 4 blocks of 2x2 (Strassen, rank 7 each) + 1 block of 2x1 (Schoolbook, rank 4)
    rank = 32
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen factors for 2x2 by 2x2
    # U_S (4x7), V_S (4x7), W_S (4x7)
    # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    
    # Strassen products:
    # M1 = A11 * (B11 + B22)
    # M2 = (A11 + A12) * B22
    # M3 = (A21 + A22) * B11
    # M4 = A22 * (B11 - B21)
    # M5 = (A11 + A22) * (B22 - B21)
    # M6 = (A12 - A21) * (B11 + B21)
    # M7 = (A11 - A12) * (B22 + B11)
    
    # We define the vectors for each product l=0..6
    # u_l, v_l, w_l for local 2x2 block
    
    # Local indices:
    # A: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B: 00, 01, 10, 11 -> 0, 1, 2, 3
    # C: 00, 01, 10, 11 -> 0, 1, 2, 3 (k*2+i)
    
    strassen_U = [
        [1, 0, 0, 0, 0, 0, 0], # A11
        [0, 1, 0, 0, 0, 0, 0], # A12 (Wait, M2 uses A11+A12)
        [0, 0, 1, 0, 0, 0, 0], # A21
        [0, 0, 0, 1, 0, 0, 0]  # A22
    ]
    # This is just basis. We need the specific linear forms.
    
    # Let's build the columns directly.
    # l=0: M1
    # A part: A11 -> [1, 0, 0, 0]
    # B part: B11 + B22 -> [1, 0, 0, 1]
    # C part: C11 += M1, C22 += M1
    # C11 (0,0) -> idx 0. C22 (1,1) -> idx 3.
    
    # l=1: M2
    # A part: A11 + A12 -> [1, 1, 0, 0]
    # B part: B22 -> [0, 0, 0, 1]
    # C part: C12 += M2, C22 -= M2
    # C12 (0,1) -> idx 1. C22 (1,1) -> idx 3.
    
    # l=2: M3
    # A part: A21 + A22 -> [0, 0, 1, 1]
    # B part: B11 -> [1, 0, 0, 0]
    # C part: C21 += M3, C11 -= M3
    # C21 (1,0) -> idx 2. C11 (0,0) -> idx 0.
    
    # l=3: M4
    # A part: A22 -> [0, 0, 0, 1]
    # B part: B11 - B21 -> [1, 0, -1, 0]
    # C part: C22 += M4, C21 -= M4
    # C22 (1,1) -> idx 3. C21 (1,0) -> idx 2.
    
    # l=4: M5
    # A part: A11 + A22 -> [1, 0, 0, 1]
    # B part: B22 - B21 -> [0, 0, -1, 1]
    # C part: C11 += M5, C22 -= M5
    # C11 (0,0) -> idx 0. C22 (1,1) -> idx 3.
    
    # l=5: M6
    # A part: A12 - A21 -> [0, 1, -1, 0]
    # B part: B11 + B21 -> [1, 0, 1, 0]
    # C part: C12 += M6
    # C12 (0,1) -> idx 1.
    
    # l=6: M7
    # A part: A11 - A12 -> [1, -1, 0, 0]
    # B part: B22 + B11 -> [1, 0, 0, 1]
    # C part: C21 += M7
    # C21 (1,0) -> idx 2.

    # Store these in lists
    sU = [
        [1, 1, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, -1],
        [0, 0, 1, 0, 0, -1, 0],
        [0, 0, 1, 1, 1, 0, 0]
    ]
    # Note: sU[i][l] is the coefficient of A_i in product l.
    
    sV = [
        [1, 0, 1, 1, 0, 1, 1], # B11
        [0, 0, 0, 0, 0, 1, 0], # B21
        [0, 0, 0, 0, 0, 0, 0], # B12 (Not used in standard Strassen? Wait.)
        [1, 1, 0, 0, 1, 0, 1]  # B22
    ]
    # Standard Strassen for 2x2:
    # M1 = A11(B11+B22)
    # M2 = (A11+A12)B22
    # M3 = (A21+A22)B11
    # M4 = A22(B11-B21)
    # M5 = (A11+A22)(B22-B21)
    # M6 = (A12-A21)(B11+B21)
    # M7 = (A11-A12)(B22+B11)
    
    # Let's re-verify sV.
    # l=0: B11+B22 -> [1, 0, 0, 1] (B11, B21, B12, B22). My sV row 0 is B11. Row 3 is B22.
    # So sV[0][0]=1, sV[3][0]=1.
    # l=1: B22 -> [0, 0, 0, 1]. sV[3][1]=1.
    # l=2: B11 -> [1, 0, 0, 0]. sV[0][2]=1.
    # l=3: B11-B21 -> [1, -1, 0, 0]. sV[0][3]=1, sV[1][3]=-1.
    # l=4: B22-B21 -> [0, -1, 0, 1]. sV[1][4]=-1, sV[3][4]=1.
    # l=5: B11+B21 -> [1, 1, 0, 0]. sV[0][5]=1, sV[1][5]=1.
    # l=6: B22+B11 -> [1, 0, 0, 1]. sV[0][6]=1, sV[3][6]=1.
    
    sV = [
        [1, 0, 1, 1, 0, 1, 1], # B11
        [0, 0, 0, -1, -1, 1, 0], # B21
        [0, 0, 0, 0, 0, 0, 0], # B12
        [1, 1, 0, 0, 1, 0, 1]  # B22
    ]
    
    # sW
    # l=0: C11+=1, C22+=1 -> [1, 0, 0, 1] (C11, C12, C21, C22)
    # l=1: C12+=1, C22-=1 -> [0, 1, 0, -1]
    # l=2: C21+=1, C11-=1 -> [-1, 0, 1, 0]
    # l=3: C22+=1, C21-=1 -> [0, 0, -1, 1]
    # l=4: C11+=1, C22-=1 -> [1, 0, 0, -1]
    # l=5: C12+=1 -> [0, 1, 0, 0]
    # l=6: C21+=1 -> [0, 0, 1, 0]
    
    sW = [
        [1, 0, -1, 0, 1, 0, 0], # C11
        [0, 1, 0, 0, 0, 1, 0], # C12
        [0, 0, 1, -1, 0, 0, 1], # C21
        [1, -1, 0, 1, -1, 0, 0] # C22
    ]
    
    # Fill U, V, W for the 4 blocks of size 2
    for b in range(4):
        # Block b covers columns 2*b and 2*b+1
        for l in range(7):
            # U is same for all blocks
            for i in range(4):
                U[i][col] = sU[i][l]
            
            # V: map local B indices to global
            # Local B indices: 0:00, 1:10, 2:01, 3:11 (j*2+k)
            # Global B indices: j*9+k
            for lj in range(2):
                for lk in range(2):
                    lidx = lj * 2 + lk
                    gidx = lj * 9 + (2 * b + lk)
                    V[gidx][col] = sV[lidx][l]
                    
            # W: map local C indices to global
            # Local C indices: 0:00, 1:01, 2:10, 3:11 (k*2+i)
            # Global C indices: k*2+i
            for lk in range(2):
                for li in range(2):
                    lidx = lk * 2 + li
                    gidx = (2 * b + lk) * 2 + li
                    W[gidx][col] = sW[lidx][l]
            
            col += 1
    
    # Last block: size 1 (column 8)
    # Schoolbook rank 4
    # Terms:
    # 1. A11 * B11 -> C11
    # 2. A12 * B21 -> C11
    # 3. A21 * B11 -> C21
    # 4. A22 * B21 -> C21
    
    # 1. A11 * B11 -> C11
    U[0][col] = 1
    V[0 * 9 + 8][col] = 1
    W[8 * 2 + 0][col] = 1
    col += 1
    
    # 2. A12 * B21 -> C11
    U[1][col] = 1
    V[1 * 9 + 8][col] = 1
    W[8 * 2 + 0][col] = 1
    col += 1
    
    # 3. A21 * B11 -> C21
    U[2][col] = 1
    V[0 * 9 + 8][col] = 1
    W[8 * 2 + 1][col] = 1
    col += 1
    
    # 4. A22 * B21 -> C21
    U[3][col] = 1
    V[1 * 9 + 8][col] = 1
    W[8 * 2 + 1][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
