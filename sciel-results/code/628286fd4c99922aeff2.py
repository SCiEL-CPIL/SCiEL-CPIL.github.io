"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 2x4 * 4x4 matrix multiplication.
    
    Strategy: Split the 2x4 * 4x4 multiplication into 4 independent 2x2 * 2x2 multiplications
    using the block structure, and apply Strassen's rank-7 algorithm to each.
    
    A is 2x4, B is 4x4, C = A*B is 2x4.
    Split j (middle dim) into {0,1} and {2,3}:
      A = [A_L | A_R]  where A_L, A_R are 2x2
      B = [B_T; B_B]  where B_T, B_B are 2x4 (top/bottom rows)
    C = A_L * B_T + A_R * B_B
    Each 2x2 * 2x4 is two 2x2 * 2x2 multiplications:
      B_T = [B_TL | B_TR], B_B = [B_BL | B_BR]
    C = A_L*B_TL + A_L*B_TR + A_R*B_BL + A_R*B_BR
    Each 2x2*2x2 uses Strassen (rank 7), total rank = 4*7 = 28.
    """
    n, m, p = DIMENSIONS  # 2, 4, 4
    
    # Strassen decomposition for 2x2 * 2x2:
    # C[0,0] = M1 - M4 + M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M3 + M6 + M7
    # where:
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)*b11
    # M3 = a11*(b21-b11)
    # M4 = a22*(b12-b22)
    # M5 = (a11+a12)*b22
    # M6 = (a21-a11)*(b11+b12)
    # M7 = (a12-a22)*(b21+b22)
    
    # For each 2x2*2x2 block, we have 7 rank-1 terms.
    # U factor: depends on (i, j) within the 2x2 A block
    # V factor: depends on (j, k) within the 2x2 B block  
    # W factor: depends on (k, i) within the 2x2 C block
    
    # Strassen U coefficients (for A elements a[i,j]):
    # M1: a11=+1, a22=+1
    # M2: a21=+1, a22=+1
    # M3: a11=+1
    # M4: a22=+1
    # M5: a11=+1, a12=+1
    # M6: a21=+1, a11=-1
    # M7: a12=+1, a22=-1
    
    # Strassen V coefficients (for B elements b[j,k]):
    # M1: b11=+1, b22=+1
    # M2: b11=+1
    # M3: b21=+1, b11=-1
    # M4: b12=+1, b22=-1
    # M5: b22=+1
    # M6: b11=+1, b12=+1
    # M7: b21=+1, b22=+1
    
    # Strassen W coefficients (for C elements c[i,k]):
    # M1: c11=+1, c22=+1
    # M2: c21=+1
    # M3: c12=+1
    # M4: c21=+1
    # M5: c12=+1
    # M6: c22=+1
    # M7: c11=+1, c22=+1
    
    # Wait, let me re-derive carefully.
    # C[0,0] = M1 - M4 + M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M3 + M6 + M7
    
    # W coefficients (contribution to C[i,k] from each M):
    # M1: C[0,0]+=1, C[1,1]+=1
    # M2: C[1,0]+=1
    # M3: C[0,1]+=1
    # M4: C[1,0]+=1, C[0,0]-=1
    # M5: C[0,1]+=1, C[0,0]+=1
    # M6: C[1,1]+=1
    # M7: C[0,0]+=1, C[1,1]+=1
    
    # Let me define the 4 blocks:
    # Block (block_j, block_k) where block_j in {0,1} (j offset 0 or 2), block_k in {0,1} (k offset 0 or 2)
    # Block (0,0): A_L * B_TL  -> C[0:2, 0:2]
    # Block (0,1): A_L * B_TR  -> C[0:2, 2:4]
    # Block (1,0): A_R * B_BL  -> C[0:2, 0:2]
    # Block (1,1): A_R * B_BR  -> C[0:2, 2:4]
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # 8 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 8 x 28
    
    # U indices: U[i*m + j, r] where i in [0,2), j in [0,4)
    # V indices: V[j*p + k, r] where j in [0,4), k in [0,4)
    # W indices: W[k*n + i, r] where k in [0,4), i in [0,2)
    
    # Strassen coefficients
    # U_strassen[s][i][j] for s in 0..6, i in 0..1, j in 0..1
    U_strassen = [
        [[1, 0], [0, 1]],   # M1: a11, a22
        [[0, 0], [1, 1]],   # M2: a21, a22
        [[1, 0], [0, 0]],   # M3: a11
        [[0, 0], [0, 1]],   # M4: a22
        [[1, 1], [0, 0]],   # M5: a11, a12
        [[-1, 0], [1, 0]],  # M6: a21-a11
        [[0, 1], [0, -1]],  # M7: a12-a22
    ]
    
    # V_strassen[s][j][k] for s in 0..6, j in 0..1, k in 0..1
    V_strassen = [
        [[1, 0], [0, 1]],   # M1: b11, b22
        [[1, 0], [0, 0]],   # M2: b11
        [[-1, 0], [1, 0]],  # M3: b21-b11
        [[0, 1], [-1, 0]],  # M4: b12-b22  (wait, b12 is [0,1], b22 is [1,1])
        # Let me re-check: M4 = a22*(b12-b22), so V for M4: b12=+1, b22=-1
        # b12 is B[0,1], b22 is B[1,1]
        # So V_strassen[4] should be: [[0,1],[-1,0]]? No.
        # b[j][k]: j=0,1 (rows), k=0,1 (cols)
        # b11 = B[0,0], b12 = B[0,1], b21 = B[1,0], b22 = B[1,1]
        # M4 = a22*(b12-b22): V[0,1]=+1, V[1,1]=-1
        # So V_strassen[4] = [[0,1],[-1,0]]? That's b[0,1]=1, b[1,1]=-1. Yes.
        [[0, 1], [-1, 0]],  # M4: b12-b22
        [[0, 0], [0, 1]],   # M5: b22
        [[1, 1], [0, 0]],   # M6: b11+b12
        [[0, 1], [0, 1]],   # M7: b21+b22
    ]
    
    # W_strassen[s][k][i] for s in 0..6, k in 0..1, i in 0..1
    # W[k*n + i, r] where k is output col, i is output row
    # C[i][k]: i=0,1 (rows), k=0,1 (cols)
    # C[0,0] = M1 - M4 + M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M3 + M6 + M7
    # W[s][i][k] = coefficient of C[i,k] from M_s
    W_strassen = [
        [[1, 0], [0, 1]],   # M1: C[0,0]+=1, C[1,1]+=1
        [[0, 0], [1, 0]],   # M2: C[1,0]+=1
        [[0, 1], [0, 0]],   # M3: C[0,1]+=1
        [[-1, 0], [1, 0]],  # M4: C[0,0]-=1, C[1,0]+=1
        [[1, 1], [0, 0]],   # M5: C[0,0]+=1, C[0,1]+=1
        [[0, 0], [0, 1]],   # M6: C[1,1]+=1
        [[1, 0], [0, 1]],   # M7: C[0,0]+=1, C[1,1]+=1
    ]
    
    # Fill in the decomposition
    # For each of the 4 blocks and 7 Strassen terms:
    col = 0
    for block_j in range(2):  # 0: j in {0,1}, 1: j in {2,3}
        for block_k in range(2):  # 0: k in {0,1}, 1: k in {2,3}
            j_offset = block_j * 2
            k_offset = block_k * 2
            
            for s in range(7):
                # U: A elements at (i, j_offset + j_local) for i in 0,1; j_local in 0,1
                for i in range(2):
                    for j_local in range(2):
                        j_global = j_offset + j_local
                        val = U_strassen[s][i][j_local]
                        if val != 0:
                            U[i * m + j_global][col] += val
                
                # V: B elements at (j_offset + j_local, k_offset + k_local) for j_local, k_local in 0,1
                for j_local in range(2):
                    for k_local in range(2):
                        j_global = j_offset + j_local
                        k_global = k_offset + k_local
                        val = V_strassen[s][j_local][k_local]
                        if val != 0:
                            V[j_global * p + k_global][col] += val
                
                # W: C elements at (k_offset + k_local, i) for k_local in 0,1; i in 0,1
                # W[k*n + i, r]
                for k_local in range(2):
                    for i in range(2):
                        k_global = k_offset + k_local
                        val = W_strassen[s][i][k_local]
                        if val != 0:
                            W[k_global * n + i][col] += val
                
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
