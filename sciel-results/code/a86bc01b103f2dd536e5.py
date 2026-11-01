"""Matrix multiplication 2,2,13 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 2, 13]

def construct():
    """Return a valid tensor decomposition for 2x2 * 2x13 matrix multiplication."""
    n, m, p = DIMENSIONS  # n=2, m=2, p=13
    
    # We use a hybrid decomposition:
    # Strassen for the first 2 columns (rank 7)
    # Schoolbook for the remaining p-2 columns (rank 4 each)
    # Total rank = 7 + 4 * (13 - 2) = 7 + 44 = 51
    
    rank = 7 + 4 * (p - 2)
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # --- Strassen Part (Columns 0 and 1) ---
    # Indices:
    # U: i*m + j
    # V: j*p + k
    # W: k*n + i
    
    # M1 = (a+d)(e+h)
    U[0*m+0, 0] = 1; U[1*m+1, 0] = 1
    V[0*p+0, 0] = 1; V[0*p+1, 0] = 1; V[1*p+0, 0] = 1; V[1*p+1, 0] = 1
    W[0*n+0, 0] = 1; W[1*n+0, 0] = -1; W[0*n+1, 0] = 1; W[1*n+1, 0] = 1
    # Wait, W indices are k*n+i.
    # C_00 = M1 + M4 - M5 + M7
    # C_01 = M3 + M5
    # C_10 = M2 + M4
    # C_11 = M1 - M2 + M3 + M6
    
    # Let's fill W carefully.
    # W[k*n + i, r] is the coefficient of M_r in C_{i,k}.
    
    # M1
    W[0*n+0, 0] = 1  # C_00
    W[1*n+0, 0] = 0  # C_01? No, k=0 is col 0.
    # My indices:
    # k is column index of B/C.
    # i is row index of A/C.
    # W[k*n + i, r]
    
    # C_00 (i=0, k=0)
    # C_01 (i=0, k=1)
    # C_10 (i=1, k=0)
    # C_11 (i=1, k=1)
    
    # M1 contributes to C_00 and C_11.
    W[0*n+0, 0] = 1
    W[1*n+1, 0] = 1
    
    # M2 = (c+d)e
    U[1*m+0, 1] = 1; U[1*m+1, 1] = 1
    V[0*p+0, 1] = 1
    # M2 contributes to C_10 and C_11 (with -1)
    W[0*n+1, 1] = 1
    W[1*n+1, 1] = -1
    
    # M3 = a(f-h)
    U[0*m+0, 2] = 1
    V[0*p+1, 2] = 1; V[1*p+1, 2] = -1
    # M3 contributes to C_01 and C_11
    W[1*n+0, 2] = 1
    W[1*n+1, 2] = 1
    
    # M4 = d(g-e)
    U[1*m+1, 3] = 1
    V[1*p+0, 3] = 1; V[0*p+0, 3] = -1
    # M4 contributes to C_00 and C_10
    W[0*n+0, 3] = 1
    W[0*n+1, 3] = 1
    
    # M5 = (a+b)h
    U[0*m+0, 4] = 1; U[0*m+1, 4] = 1
    V[1*p+1, 4] = 1
    # M5 contributes to C_00 (-1) and C_01
    W[0*n+0, 4] = -1
    W[1*n+0, 4] = 1
    
    # M6 = (c-a)(e+f)
    U[1*m+0, 5] = 1; U[0*m+0, 5] = -1
    V[0*p+0, 5] = 1; V[0*p+1, 5] = 1
    # M6 contributes to C_11
    W[1*n+1, 5] = 1
    
    # M7 = (b-d)(g+h)
    U[0*m+1, 6] = 1; U[1*m+1, 6] = -1
    V[1*p+0, 6] = 1; V[1*p+1, 6] = 1
    # M7 contributes to C_00
    W[0*n+0, 6] = 1

    # --- Schoolbook Part (Columns 2 to p-1) ---
    col_offset = 7
    for k in range(2, p):
        # 4 terms for column k
        # C_i,k = A_i0 B_0k + A_i1 B_1k
        
        # Term 1: A_00 * B_0k
        U[0*m+0, col_offset] = 1
        V[0*p+k, col_offset] = 1
        W[k*n+0, col_offset] = 1
        col_offset += 1
        
        # Term 2: A_01 * B_1k
        U[0*m+1, col_offset] = 1
        V[1*p+k, col_offset] = 1
        W[k*n+0, col_offset] = 1
        col_offset += 1
        
        # Term 3: A_10 * B_0k
        U[1*m+0, col_offset] = 1
        V[0*p+k, col_offset] = 1
        W[k*n+1, col_offset] = 1
        col_offset += 1
        
        # Term 4: A_11 * B_1k
        U[1*m+1, col_offset] = 1
        V[1*p+k, col_offset] = 1
        W[k*n+1, col_offset] = 1
        col_offset += 1

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
