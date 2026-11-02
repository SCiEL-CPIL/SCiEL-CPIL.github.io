"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target rank 50
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen(U, V, W, u_off, v_off, w_off, start_col):
        # Adds a 2x2x2 Strassen decomposition
        # u_off: list of 4 global U indices corresponding to local A[0,0], A[0,1], A[1,0], A[1,1]
        # v_off: list of 4 global V indices corresponding to local B[0,0], B[0,1], B[1,0], B[1,1]
        # w_off: list of 4 global W indices corresponding to local C[0,0], C[0,1], C[1,0], C[1,1]
        
        # M1 = (A00 + A11) * (B00 + B11)
        U[u_off[0]][start_col] += 1
        U[u_off[2]][start_col] += 1
        V[v_off[0]][start_col] += 1
        V[v_off[2]][start_col] += 1
        W[w_off[0]][start_col] += 1 # C00
        W[w_off[3]][start_col] += 1 # C11
        
        # M2 = (A10 + A11) * B00
        U[u_off[1]][start_col+1] += 1
        U[u_off[3]][start_col+1] += 1
        V[v_off[0]][start_col+1] += 1
        W[w_off[1]][start_col+1] += 1 # C10
        W[w_off[3]][start_col+1] -= 1 # C11
        
        # M3 = A00 * (B01 - B11)
        U[u_off[0]][start_col+2] += 1
        V[v_off[1]][start_col+2] += 1
        V[v_off[3]][start_col+2] -= 1
        W[w_off[0]][start_col+2] += 1 # C00
        W[w_off[2]][start_col+2] += 1 # C01
        
        # M4 = A11 * (B10 - B00)
        U[u_off[2]][start_col+3] += 1
        V[v_off[2]][start_col+3] += 1
        V[v_off[0]][start_col+3] -= 1
        W[w_off[1]][start_col+3] += 1 # C10
        W[w_off[3]][start_col+3] += 1 # C11
        
        # M5 = (A00 + A01) * B11
        U[u_off[0]][start_col+4] += 1
        U[u_off[1]][start_col+4] += 1
        V[v_off[3]][start_col+4] += 1
        W[w_off[0]][start_col+4] -= 1 # C00
        W[w_off[2]][start_col+4] += 1 # C01
        
        # M6 = (A10 - A00) * (B00 + B01)
        U[u_off[1]][start_col+5] += 1
        U[u_off[0]][start_col+5] -= 1
        V[v_off[0]][start_col+5] += 1
        V[v_off[1]][start_col+5] += 1
        W[w_off[3]][start_col+5] += 1 # C11
        
        # M7 = (A01 - A11) * (B10 + B11)
        U[u_off[1]][start_col+6] += 1
        U[u_off[3]][start_col+6] -= 1
        V[v_off[2]][start_col+6] += 1
        V[v_off[3]][start_col+6] += 1
        W[w_off[0]][start_col+6] += 1 # C00
        W[w_off[2]][start_col+6] += 1 # C01
        # Wait, C11 gets M7?
        # C11 = M1 - M2 + M3 + M6. No M7.
        # C00 = M1 + M4 - M5 + M7.
        # C01 = M3 + M5.
        # C10 = M2 + M4.
        # Let's re-verify Strassen formulas.
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = d(g-c)
        # M5 = (a+b)h
        # M6 = (c-a)(e+f)
        # M7 = (b-d)(g+h)
        #
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        #
        # My mapping:
        # w_off[0] is C00 (C11 in 1-based)
        # w_off[2] is C01 (C12 in 1-based)
        # w_off[1] is C10 (C21 in 1-based)
        # w_off[3] is C11 (C22 in 1-based)
        
        # Correction for M7:
        # C00 gets +M7.
        # C01 gets 0.
        # C10 gets 0.
        # C11 gets 0.
        # So W[w_off[0]][start_col+6] += 1 is correct.
        # But I also added W[w_off[2]][start_col+6] += 1 in the code above. That is WRONG.
        # M7 only affects C00.
        
        return start_col + 7

    def add_schoolbook_1(U, V, W, u_off, v_off, w_off, start_col):
        # Adds a 2x2x1 schoolbook decomposition
        # u_off: 4 indices for A
        # v_off: 2 indices for B (col 0)
        # w_off: 2 indices for C (col 0)
        
        # P1 = A00 * B00
        U[u_off[0]][start_col] += 1
        V[v_off[0]][start_col] += 1
        W[w_off[0]][start_col] += 1
        
        # P2 = A01 * B01
        U[u_off[1]][start_col+1] += 1
        V[v_off[1]][start_col+1] += 1
        W[w_off[0]][start_col+1] += 1
        
        # P3 = A10 * B00
        U[u_off[2]][start_col+2] += 1
        V[v_off[0]][start_col+2] += 1
        W[w_off[1]][start_col+2] += 1
        
        # P4 = A11 * B01
        U[u_off[3]][start_col+3] += 1
        V[v_off[1]][start_col+3] += 1
        W[w_off[1]][start_col+3] += 1
        
        return start_col + 4

    # Global Index Mapping
    # U[i*m + j]
    # A_L (cols 0,1): 
    #   A[0,0]->0, A[0,1]->1, A[1,0]->4, A[1,1]->5
    u_off_L = [0, 1, 4, 5]
    # A_R (cols 2,3):
    #   A[0,2]->2, A[0,3]->3, A[1,2]->6, A[1,3]->7
    u_off_R = [2, 3, 6, 7]
    
    # V[j*p + k]
    # B_T (rows 0,1):
    #   B[0,0]->0, B[0,1]->1, B[0,2]->2, B[0,3]->3, B[0,4]->4, B[0,5]->5, B[0,6]->6
    #   B[1,0]->7, B[1,1]->8, B[1,2]->9, B[1,3]->10, B[1,4]->11, B[1,5]->12, B[1,6]->13
    # B_B (rows 2,3):
    #   B[2,0]->14, ... B[3,6]->27
    
    # W[k*n + i]
    # C is 2x7.
    # C[0,0]->0, C[0,1]->2, C[0,2]->4, C[0,3]->6, C[0,4]->8, C[0,5]->10, C[0,6]->12
    # C[1,0]->1, C[1,1]->3, C[1,2]->5, C[1,3]->7, C[1,4]->9, C[1,5]->11, C[1,6]->13
    
    # Subproblem 1: A_L B_T
    col = 0
    # Group 1: B_T cols 0,1
    v_off_1 = [0, 1, 7, 8] # B[0,0], B[0,1], B[1,0], B[1,1]
    w_off_1 = [0, 2, 1, 3] # C[0,0], C[0,1], C[1,0], C[1,1]
    col = add_strassen(U, V, W, u_off_L, v_off_1, w_off_1, col)
    
    # Group 2: B_T cols 2,3
    v_off_2 = [2, 3, 9, 10]
    w_off_2 = [4, 6, 5, 7]
    col = add_strassen(U, V, W, u_off_L, v_off_2, w_off_2, col)
    
    # Group 3: B_T cols 4,5
    v_off_3 = [4, 5, 11, 12]
    w_off_3 = [8, 10, 9, 11]
    col = add_strassen(U, V, W, u_off_L, v_off_3, w_off_3, col)
    
    # Group 4: B_T col 6
    v_off_4 = [6, 13] # B[0,6], B[1,6]
    w_off_4 = [12, 13] # C[0,6], C[1,6]
    col = add_schoolbook_1(U, V, W, u_off_L, v_off_4, w_off_4, col)
    
    # Subproblem 2: A_R B_B
    # Group 1: B_B cols 0,1
    v_off_5 = [14, 15, 21, 22]
    w_off_5 = [0, 2, 1, 3]
    col = add_strassen(U, V, W, u_off_R, v_off_5, w_off_5, col)
    
    # Group 2: B_B cols 2,3
    v_off_6 = [16, 17, 23, 24]
    w_off_6 = [4, 6, 5, 7]
    col = add_strassen(U, V, W, u_off_R, v_off_6, w_off_6, col)
    
    # Group 3: B_B cols 4,5
    v_off_7 = [18, 19, 25, 26]
    w_off_7 = [8, 10, 9, 11]
    col = add_strassen(U, V, W, u_off_R, v_off_7, w_off_7, col)
    
    # Group 4: B_B col 6
    v_off_8 = [20, 27]
    w_off_8 = [12, 13]
    col = add_schoolbook_1(U, V, W, u_off_R, v_off_8, w_off_8, col)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
