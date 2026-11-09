import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # We decompose the multiplication into blocks.
    # A is 2x4, B is 4x7.
    # Split B into three 4x2 blocks and one 4x1 block.
    # B1: cols 0-1, B2: cols 2-3, B3: cols 4-5, B4: col 6.
    
    # For each 4x2 block, we use two Strassen decompositions.
    # For the 4x1 block, we use schoolbook.
    
    # Helper to generate Strassen U, V, W for 2x2 by 2x2 multiplication.
    # Indices: U (2*2, rank), V (2*2, rank), W (2*2, rank)
    # Note: The global indices need to be mapped.
    
    # We will build global U, V, W.
    # Total rank = 3 * (7 + 7) + 8 = 50.
    rank = 50
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Function to add a Strassen decomposition for 2x2 matrices
    # A is 2x2, B is 2x2.
    # We need to map local indices (i, j, k) to global indices.
    # Local i in [0, 1], j in [0, 1], k in [0, 1].
    # Global i_offset, j_offset, k_offset.
    
    def add_strassen(i_off, j_off, k_off):
        nonlocal col
        # Strassen's 7 multiplications for 2x2.
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B21 - B22)
        # M4 = A22(B12 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        
        # We define the coefficients for U, V, W for each of the 7 terms.
        # U is indexed by (i, j). V by (j, k). W by (k, i).
        
        # Term 1: (A11 + A22) * (B11 + B22)
        # U: (0,0)=1, (1,1)=1
        # V: (0,0)=1, (1,1)=1
        # W: (0,0)=1, (1,1)=1 ? No.
        # C11 = M1 - M4 + M5 - M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M7
        
        # This direct mapping is tricky because W depends on k and i.
        # Let's use the standard rank-1 decomposition of the Strassen tensor.
        
        # The 7 rank-1 tensors are:
        # 1. U: (0,0)=1, (1,1)=1; V: (0,0)=1, (1,1)=1; W: (0,0)=1, (1,1)=1 -> M1
        # Wait, W is (k, i).
        # For M1, it contributes to C11, C12, C21, C22?
        # No, M1 is a scalar. It is multiplied by 1 in the reconstruction.
        
        # Let's just use the known decomposition matrices.
        # U_strassen (4x7), V_strassen (4x7), W_strassen (4x7)
        
        U_s = [
            [1, 0, 1, 0, 1, -1, 0], # A11
            [0, 0, 0, 0, 0, 0, 0],   # A12 (placeholder)
            [0, 1, 0, 1, 0, 1, 0],   # A21
            [1, 1, 0, 1, 1, 0, 1]    # A22
        ]
        # Correct Strassen U, V, W:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # U (A coefficients):
        # M1: A11=1, A22=1
        # M2: A21=1, A22=1
        # M3: A11=1
        # M4: A22=1
        # M5: A11=1, A12=1
        # M6: A21=1, A11=-1
        # M7: A12=1, A22=-1
        
        U_s = [
            [1, 0, 1, 0, 1, -1, 0], # i=0, j=0 (A11)
            [0, 0, 0, 0, 1, 0, 1],  # i=0, j=1 (A12)
            [0, 1, 0, 0, 0, 1, 0],  # i=1, j=0 (A21)
            [1, 1, 0, 1, 0, 0, -1]  # i=1, j=1 (A22)
        ]
        
        # V (B coefficients):
        # M1: B11=1, B22=1
        # M2: B11=1
        # M3: B21=1, B22=-1
        # M4: B12=1, B11=-1
        # M5: B22=1
        # M6: B11=1, B12=1
        # M7: B21=1, B22=1
        
        V_s = [
            [1, 1, 0, -1, 0, 1, 0], # j=0, k=0 (B11)
            [0, 0, 0, 1, 0, 1, 0],  # j=0, k=1 (B12)
            [0, 0, 1, 0, 1, 0, 1],  # j=1, k=0 (B21)
            [1, 0, -1, 0, 1, 0, 1]  # j=1, k=1 (B22)
        ]
        
        # W (C coefficients):
        # C11 = M1 - M4 + M5 - M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M7
        
        # W is indexed by (k, i).
        # k=0, i=0 (C11): [1, 0, 0, -1, 1, 0, -1]
        # k=0, i=1 (C21): [0, 1, 0, 1, 0, 0, 0]
        # k=1, i=0 (C12): [0, 0, 1, 0, 1, 0, 0]
        # k=1, i=1 (C22): [1, -1, 1, 0, 0, 0, 1]
        
        W_s = [
            [1, 0, 0, -1, 1, 0, -1], # k=0, i=0
            [0, 1, 0, 1, 0, 0, 0],   # k=0, i=1
            [0, 0, 1, 0, 1, 0, 0],   # k=1, i=0
            [1, -1, 1, 0, 0, 0, 1]   # k=1, i=1
        ]
        
        for c_local in range(7):
            c_global = col + c_local
            for i_local in range(2):
                for j_local in range(2):
                    val = U_s[i_local * 2 + j_local][c_local]
                    if val != 0:
                        i_g = i_off + i_local
                        j_g = j_off + j_local
                        U[i_g * m + j_g][c_global] += val
            
            for j_local in range(2):
                for k_local in range(2):
                    val = V_s[j_local * 2 + k_local][c_local]
                    if val != 0:
                        j_g = j_off + j_local
                        k_g = k_off + k_local
                        V[j_g * p + k_g][c_global] += val
            
            for k_local in range(2):
                for i_local in range(2):
                    val = W_s[k_local * 2 + i_local][c_local]
                    if val != 0:
                        k_g = k_off + k_local
                        i_g = i_off + i_local
                        W[k_g * n + i_g][c_global] += val
            
            col += 7
    
    # Block 1: B cols 0-1
    # A_L: rows 0-1, cols 0-1 (i_off=0, j_off=0)
    # B_T: rows 0-1, cols 0-1 (j_off=0, k_off=0)
    add_strassen(0, 0, 0)
    
    # A_R: rows 0-1, cols 2-3 (i_off=0, j_off=2)
    # B_B: rows 2-3, cols 0-1 (j_off=2, k_off=0)
    add_strassen(0, 2, 0)
    
    # Block 2: B cols 2-3
    # A_L: i_off=0, j_off=0
    # B_T: j_off=0, k_off=2
    add_strassen(0, 0, 2)
    
    # A_R: i_off=0, j_off=2
    # B_B: j_off=2, k_off=2
    add_strassen(0, 2, 2)
    
    # Block 3: B cols 4-5
    # A_L: i_off=0, j_off=0
    # B_T: j_off=0, k_off=4
    add_strassen(0, 0, 4)
    
    # A_R: i_off=0, j_off=2
    # B_B: j_off=2, k_off=4
    add_strassen(0, 2, 4)
    
    # Block 4: B col 6 (4x1)
    # Schoolbook: 2x4 by 4x1. Rank 8.
    # For each i in 0..1, j in 0..3, k in 0..0 (only k=6)
    for i in range(n):
        for j in range(m):
            k = 6
            U[i * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print(res['U'])
