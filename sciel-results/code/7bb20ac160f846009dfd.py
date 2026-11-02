import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=8
    # Rank 56 decomposition
    rank = 56
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    # Strassen coefficients for 2x2x2
    # A = [[a, b], [c, d]]
    # B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # Output:
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M4
    # C10 = M2 + M5
    # C11 = M1 - M2 + M3 + M6
    
    # We have two groups of j: (0,1) and (2,3)
    # And four blocks of k: (0,1), (2,3), (4,5), (6,7)
    
    # Group 1: j in {0, 1}
    # A indices: 0, 1, 2, 3 correspond to A[0,0], A[0,1], A[1,0], A[1,1]
    # U rows: 0..3 for this group? No, U is n*m = 2*4 = 8 rows.
    # U[i*m + j, col]
    # For j in {0,1}, U rows are 0,1 (i=0) and 2,3 (i=1)?
    # i=0, j=0 -> idx 0
    # i=0, j=1 -> idx 1
    # i=1, j=0 -> idx 2
    # i=1, j=1 -> idx 3
    
    # V rows: j*p + k
    # For j in {0,1}, V rows are 0..7 (j=0) and 8..15 (j=1)?
    # j=0, k=0..7 -> 0..7
    # j=1, k=0..7 -> 8..15
    
    # W rows: k*n + i
    # W[k*n + i, col]
    
    # We will iterate over the 4 k-blocks for the first j-group
    for block_idx in range(4):
        k_start = block_idx * 2
        
        # Strassen M1..M7
        # M1 = (A00 + A11) * (B_e + B_h)
        # A00 is U[0], A11 is U[3] (for j=0,1 group, A11 is i=1, j=1 -> idx 3)
        # Wait, A indices:
        # i=0, j=0 -> 0
        # i=0, j=1 -> 1
        # i=1, j=0 -> 2
        # i=1, j=1 -> 3
        
        # B indices for this block:
        # B_e is B[0, k_start] -> V[0*p + k_start]
        # B_f is B[0, k_start+1] -> V[0*p + k_start+1]
        # B_g is B[1, k_start] -> V[1*p + k_start]
        # B_h is B[1, k_start+1] -> V[1*p + k_start+1]
        
        # M1
        U[0, col] = 1; U[3, col] = 1
        V[k_start, col] = 1; V[p + k_start+1, col] = 1
        W[0, col] = 1; W[n + 0, col] = 1 # C00, C11? No, W is for C.
        # W[k*n + i]
        # C00 is k=0, i=0 -> W[0]
        # C11 is k=1, i=1 -> W[n+1]
        # M1 contributes to C00 and C11
        W[k_start*n + 0, col] = 1
        W[(k_start+1)*n + 1, col] = 1
        col += 1
        
        # M2 = (A10 + A11) * B_e
        # A10 is U[2], A11 is U[3]
        U[2, col] = 1; U[3, col] = 1
        V[k_start, col] = 1
        W[k_start*n + 1, col] = 1 # C10
        col += 1
        
        # M3 = A00 * (B_f - B_h)
        U[0, col] = 1
        V[p + k_start+1, col] = 1; V[p + k_start+1, col] = -1 # B_f is V[p+k+1]? No.
        # B_f is B[0, k+1] -> V[0*p + k+1]
        # B_h is B[1, k+1] -> V[1*p + k+1]
        V[k_start+1, col] = 1
        V[p + k_start+1, col] = -1
        W[k_start*n + 0, col] = 1 # C01? No, C01 is k+1, i=0.
        # C01 is W[(k+1)*n + 0]
        W[(k_start+1)*n + 0, col] = 1
        col += 1
        
        # M4 = A11 * (B_g - B_e)
        U[3, col] = 1
        V[p + k_start, col] = 1
        V[k_start, col] = -1
        W[k_start*n + 0, col] = -1 # C00
        W[(k_start+1)*n + 0, col] = 1 # C01
        col += 1
        
        # M5 = (A00 + A01) * B_h
        U[0, col] = 1; U[1, col] = 1
        V[p + k_start+1, col] = 1
        W[k_start*n + 0, col] = 1 # C00
        W[k_start*n + 1, col] = 1 # C10
        col += 1
        
        # M6 = (A10 - A00) * (B_e + B_f)
        U[2, col] = 1; U[0, col] = -1
        V[k_start, col] = 1
        V[k_start+1, col] = 1
        W[(k_start+1)*n + 1, col] = 1 # C11
        col += 1
        
        # M7 = (A01 - A11) * (B_g + B_h)
        U[1, col] = 1; U[3, col] = -1
        V[p + k_start, col] = 1
        V[p + k_start+1, col] = 1
        W[k_start*n + 0, col] = -1 # C00
        col += 1

    # Group 2: j in {2, 3}
    # A indices: 4, 5, 6, 7
    # i=0, j=2 -> 4
    # i=0, j=3 -> 5
    # i=1, j=2 -> 6
    # i=1, j=3 -> 7
    
    # B indices for j=2,3:
    # B[2, k] -> V[2*p + k]
    # B[3, k] -> V[3*p + k]
    
    for block_idx in range(4):
        k_start = block_idx * 2
        
        # M1
        U[4, col] = 1; U[7, col] = 1
        V[2*p + k_start, col] = 1; V[3*p + k_start+1, col] = 1
        W[k_start*n + 0, col] = 1
        W[(k_start+1)*n + 1, col] = 1
        col += 1
        
        # M2
        U[6, col] = 1; U[7, col] = 1
        V[2*p + k_start, col] = 1
        W[k_start*n + 1, col] = 1
        col += 1
        
        # M3
        U[4, col] = 1
        V[2*p + k_start+1, col] = 1
        V[3*p + k_start+1, col] = -1
        W[(k_start+1)*n + 0, col] = 1
        col += 1
        
        # M4
        U[7, col] = 1
        V[3*p + k_start, col] = 1
        V[2*p + k_start, col] = -1
        W[k_start*n + 0, col] = -1
        W[(k_start+1)*n + 0, col] = 1
        col += 1
        
        # M5
        U[4, col] = 1; U[5, col] = 1
        V[3*p + k_start+1, col] = 1
        W[k_start*n + 0, col] = 1
        W[k_start*n + 1, col] = 1
        col += 1
        
        # M6
        U[6, col] = 1; U[4, col] = -1
        V[2*p + k_start, col] = 1
        V[2*p + k_start+1, col] = 1
        W[(k_start+1)*n + 1, col] = 1
        col += 1
        
        # M7
        U[5, col] = 1; U[7, col] = -1
        V[3*p + k_start, col] = 1
        V[3*p + k_start+1, col] = 1
        W[k_start*n + 0, col] = -1
        col += 1

    return {"U": U, "V": V, "W": W}
