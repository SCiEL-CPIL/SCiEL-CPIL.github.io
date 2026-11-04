import numpy as np
import time

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank is n*m*p = 50
    # We will use Strassen for two 2x2x2 blocks to reduce rank to 48
    
    rank = 48
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_block(offset_r, i_offset, j_offset, k_offset):
        # i_offset, j_offset, k_offset are the starting indices for i, j, k
        # i ranges i_offset to i_offset+1
        # j ranges j_offset to j_offset+1
        # k ranges k_offset to k_offset+1
        
        # Local indices 0,1 for i, j, k
        # Global I = i*m + j
        # Global J = j*p + k
        # Global K = k*n + i
        
        # Strassen terms
        # M1: (A00+A11)(B00+B11) -> C00, C11
        # M2: (A10+A11)B00 -> C10
        # M3: A00(B01-B11) -> C01
        # M4: A11(B10-B00) -> C00, C10
        # M5: (A00+A01)B11 -> C01, C00
        # M6: (A10-A00)(B00+B01) -> C11
        # M7: (A01-A11)(B10+B11) -> C00
        
        def get_I(i_loc, j_loc):
            i = i_offset + i_loc
            j = j_offset + j_loc
            return i * m + j
            
        def get_J(j_loc, k_loc):
            j = j_offset + j_loc
            k = k_offset + k_loc
            return j * p + k
            
        def get_K(k_loc, i_loc):
            k = k_offset + k_loc
            i = i_offset + i_loc
            return k * n + i

        # M1
        r = offset_r + 0
        U[get_I(0,0)][r] += 1
        U[get_I(1,1)][r] += 1
        V[get_J(0,0)][r] += 1
        V[get_J(1,1)][r] += 1
        W[get_K(0,0)][r] += 1
        W[get_K(1,1)][r] += 1

        # M2
        r = offset_r + 1
        U[get_I(1,0)][r] += 1
        U[get_I(1,1)][r] += 1
        V[get_J(0,0)][r] += 1
        W[get_K(0,1)][r] += 1 # C10 is k=0, i=1? No.
        # C10: i=1, k=0. K = 0*n + 1 = 1.
        # My get_K(k_loc, i_loc) returns k*n+i.
        # For C10, k_loc=0, i_loc=1. K=1. Correct.

        # M3
        r = offset_r + 2
        U[get_I(0,0)][r] += 1
        V[get_J(0,1)][r] += 1
        V[get_J(1,1)][r] += -1
        W[get_K(1,0)][r] += 1 # C01: k=1, i=0. K=2.

        # M4
        r = offset_r + 3
        U[get_I(1,1)][r] += 1
        V[get_J(1,0)][r] += 1
        V[get_J(0,0)][r] += -1
        W[get_K(0,0)][r] += 1 # C00
        W[get_K(0,1)][r] += 1 # C10

        # M5
        r = offset_r + 4
        U[get_I(0,0)][r] += 1
        U[get_I(0,1)][r] += 1
        V[get_J(1,1)][r] += 1
        W[get_K(1,0)][r] += 1 # C01
        W[get_K(0,0)][r] += -1 # C00

        # M6
        r = offset_r + 5
        U[get_I(1,0)][r] += 1
        U[get_I(0,0)][r] += -1
        V[get_J(0,0)][r] += 1
        V[get_J(0,1)][r] += 1
        W[get_K(1,1)][r] += 1 # C11

        # M7
        r = offset_r + 6
        U[get_I(0,1)][r] += 1
        U[get_I(1,1)][r] += -1
        V[get_J(1,0)][r] += 1
        V[get_J(1,1)][r] += 1
        W[get_K(0,0)][r] += 1 # C00

    # Block 1: i in 0,1; j in 0,1; k in 0,1
    add_strassen_block(0, 0, 0, 0)
    
    # Block 2: i in 0,1; j in 2,3; k in 2,3
    add_strassen_block(7, 0, 2, 2)
    
    # Remaining terms
    # Schoolbook for all (i,j,k) not in block 1 or 2
    col = 14
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check if in block 1
                if (i in [0,1] and j in [0,1] and k in [0,1]):
                    continue
                # Check if in block 2
                if (i in [0,1] and j in [2,3] and k in [2,3]):
                    continue
                
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
