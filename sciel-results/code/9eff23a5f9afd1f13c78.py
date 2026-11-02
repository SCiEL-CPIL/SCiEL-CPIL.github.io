import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 120
    
    n, m, p = 2, 4, 6
    
    # Rank 42 decomposition using Strassen on 2x2x2 blocks
    rank_42 = 42
    U = np.zeros((8, rank_42))
    V = np.zeros((24, rank_42))
    W = np.zeros((12, rank_42))
    
    def add_strassen(A_idx, B_idx, C_idx, start_col):
        a, b, c, d = A_idx
        e, f, g, h = B_idx
        c11, c12, c21, c22 = C_idx
        
        # M1 = (a+d)(e+h)
        U[a, start_col] += 1; U[d, start_col] += 1
        V[e, start_col] += 1; V[h, start_col] += 1
        W[c11, start_col] += 1; W[c22, start_col] += 1
        start_col += 1
        
        # M2 = (c+d)e
        U[c, start_col] += 1; U[d, start_col] += 1
        V[e, start_col] += 1
        W[c21, start_col] += 1; W[c22, start_col] -= 1
        start_col += 1
        
        # M3 = a(f-h)
        U[a, start_col] += 1
        V[f, start_col] += 1; V[h, start_col] -= 1
        W[c12, start_col] += 1; W[c22, start_col] += 1
        start_col += 1
        
        # M4 = d(g-e)
        U[d, start_col] += 1
        V[g, start_col] += 1; V[e, start_col] -= 1
        W[c11, start_col] += 1; W[c21, start_col] += 1
        start_col += 1
        
        # M5 = (a+b)h
        U[a, start_col] += 1; U[b, start_col] += 1
        V[h, start_col] += 1
        W[c11, start_col] -= 1; W[c12, start_col] += 1
        start_col += 1
        
        # M6 = (c-a)(e+f)
        U[c, start_col] += 1; U[a, start_col] -= 1
        V[e, start_col] += 1; V[f, start_col] += 1
        W[c22, start_col] += 1
        start_col += 1
        
        # M7 = (b-d)(g+h)
        U[b, start_col] += 1; U[d, start_col] -= 1
        V[g, start_col] += 1; V[h, start_col] += 1
        W[c11, start_col] += 1
        start_col += 1
        
        return start_col
    
    col = 0
    # Left block: A[0:2, 0:2] * B[0:2, 0:6]
    # Split B into 3 blocks of 2x2
    for k in range(3):
        A_L = [0, 1, 4, 5] # a,b,c,d for A[0,0], A[0,1], A[1,0], A[1,1]
        B_T = [k*2, k*2+1, 2+k*2, 2+k*2+1] # e,f,g,h for B block
        C_idx = [k*2, k*2+1, 2+k*2, 2+k*2+1] # c11, c12, c21, c22
        
        col = add_strassen(A_L, B_T, C_idx, col)
        
        A_R = [2, 3, 6, 7] # a,b,c,d for A[0,2], A[0,3], A[1,2], A[1,3]
        B_B = [2+k*2, 2+k*2+1, 4+k*2, 4+k*2+1] # e,f,g,h for B block (rows 2,3)
        # Wait, B is 4x6.
        # B_T should be rows 0,1. B_B should be rows 2,3.
        # B indices in V are 0..23.
        # B row 0: 0,1,2,3,4,5
        # B row 1: 6,7,8,9,10,11
        # B row 2: 12,13,14,15,16,17
        # B row 3: 18,19,20,21,22,23
        
        # Correction:
        # For block k (cols 2k, 2k+1):
        # B_T (rows 0,1): indices 2k, 2k+1 (row 0), 6+2k, 6+2k+1 (row 1)
        # B_B (rows 2,3): indices 12+2k, 12+2k+1 (row 2), 18+2k, 18+2k+1 (row 3)
        
        B_T = [2*k, 2*k+1, 6+2*k, 6+2*k+1]
        B_B = [12+2*k, 12+2*k+1, 18+2*k, 18+2*k+1]
        
        # Note: C_idx is the same for both because they add to the same C?
        # No, A_L B_T and A_R B_B both contribute to the same C columns.
        # So we just add them to the same W indices.
        
    # The loop above is incorrect in logic because I defined B_T inside the loop but used it for A_L, and then redefined for A_R?
    # Let's rewrite the loop clearly.
    
    U = np.zeros((8, 42))
    V = np.zeros((24, 42))
    W = np.zeros((12, 42))
    col = 0
    for k in range(3):
        # Block k of B (cols 2k, 2k+1)
        # A_L * B_T (top part of B)
        A_L = [0, 1, 4, 5]
        B_T = [2*k, 2*k+1, 6+2*k, 6+2*k+1]
        C_idx = [2*k, 2*k+1, 2+2*k, 2+2*k+1]
        col = add_strassen(A_L, B_T, C_idx, col)
        
        # A_R * B_B (bottom part of B)
        A_R = [2, 3, 6, 7]
        B_B = [12+2*k, 12+2*k+1, 18+2*k, 18+2*k+1]
        C_idx_same = [2*k, 2*k+1, 2+2*k, 2+2*k+1]
        col = add_strassen(A_R, B_B, C_idx_same, col)
        
    if time.time() > deadline - 5:
         # Safety margin
         return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
