"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 7
    
    # We will construct the decomposition block by block.
    # Block 1: Cols 1-2 (indices 0-1 in p)
    # Block 2: Cols 3-4 (indices 2-3 in p)
    # Block 3: Cols 5-6 (indices 4-5 in p)
    # Block 4: Col 7 (index 6 in p)
    
    U_blocks = []
    V_blocks = []
    W_blocks = []
    
    def get_strassen_222():
        """
        Returns U, V, W for 2x2x2 Strassen.
        U shape: (4, 7)
        V shape: (4, 7)
        W shape: (4, 7)
        Indices are 0-3 for each matrix (row-major).
        """
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # M1 = (a+d)(e+h)
        # a=0, d=3, e=0, h=3
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1 # C11, C22 get +1
        
        # M2 = (c+d)e
        # c=2, d=3, e=0
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1 # C21 gets +1
        
        # M3 = a(f-h)
        # a=0, f=1, h=3
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[1][2] = 1; W[3][2] = 1 # C12, C22 get +1
        
        # M4 = d(g-e)
        # d=3, g=2, e=0
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[2][3] = 1 # C11, C21 get +1
        
        # M5 = (a+b)h
        # a=0, b=1, h=3
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[1][4] = 1 # C11 -1, C12 +1
        
        # M6 = (c-a)(e+f)
        # c=2, a=0, e=0, f=1
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1 # C22 +1
        
        # M7 = (b-d)(g+h)
        # b=1, d=3, g=2, h=3
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1 # C11 +1
        
        return U, V, W

    def get_block_242(col_offset):
        """
        Returns U, V, W for 2x4x2 block.
        col_offset: starting column index in the global p dimension (0, 2, 4).
        U shape: (8, 14)
        V shape: (8, 14)
        W shape: (4, 14)
        """
        U = [[0]*14 for _ in range(8)]
        V = [[0]*14 for _ in range(8)]
        W = [[0]*14 for _ in range(4)]
        
        # Strassen 1: A_L (0-3) * B_T (0-3)
        U1, V1, W1 = get_strassen_222()
        # Map to U, V, W
        for i in range(4):
            for k in range(7):
                U[i][k] = U1[i][k]
                V[i][k] = V1[i][k]
                W[i][k] = W1[i][k]
                
        # Strassen 2: A_R (4-7) * B_B (4-7)
        U2, V2, W2 = get_strassen_222()
        for i in range(4):
            for k in range(7):
                # A_R indices are 4-7
                U[i+4][k+7] = U2[i][k]
                # B_B indices are 4-7
                V[i+4][k+7] = V2[i][k]
                # C is same 2x2
                W[i][k+7] = W2[i][k]
                
        return U, V, W

    def get_block_241(col_offset):
        """
        Returns U, V, W for 2x4x1 block.
        col_offset: starting column index in the global p dimension (6).
        U shape: (8, 8)
        V shape: (4, 8)
        W shape: (2, 8)
        """
        U = [[0]*8 for _ in range(8)]
        V = [[0]*8 for _ in range(4)]
        W = [[0]*8 for _ in range(2)]
        
        # Schoolbook for 2x4x1
        # C_i = sum_j A_ij B_j
        # A is 2x4 (0-7), B is 4x1 (0-3), C is 2x1 (0-1)
        col = 0
        for i in range(2):
            for j in range(4):
                U[i*4 + j][col] = 1
                V[j][col] = 1
                W[i][col] = 1
                col += 1
                
        return U, V, W

    # Assemble blocks
    # Block 1: Cols 0-1
    U1, V1, W1 = get_block_242(0)
    U_blocks.append(U1)
    V_blocks.append(V1)
    W_blocks.append(W1)
    
    # Block 2: Cols 2-3
    U2, V2, W2 = get_block_242(2)
    U_blocks.append(U2)
    V_blocks.append(V2)
    W_blocks.append(W2)
    
    # Block 3: Cols 4-5
    U3, V3, W3 = get_block_242(4)
    U_blocks.append(U3)
    V_blocks.append(V3)
    W_blocks.append(W3)
    
    # Block 4: Col 6
    U4, V4, W4 = get_block_241(6)
    U_blocks.append(U4)
    V_blocks.append(V4)
    W_blocks.append(W4)
    
    # Concatenate
    U_final = []
    for block in U_blocks:
        U_final.extend(block)
        
    V_final = []
    for block in V_blocks:
        V_final.extend(block)
        
    W_final = []
    for block in W_blocks:
        W_final.extend(block)
        
    return {"U": U_final, "V": V_final, "W": W_final}
# EVOLVE-BLOCK-END
