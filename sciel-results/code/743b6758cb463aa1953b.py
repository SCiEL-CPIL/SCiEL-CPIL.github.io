"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 2, 2, 5
    # Rank 18 solution:
    # 7 terms for first 2 columns of B (Strassen)
    # 7 terms for next 2 columns of B (Strassen)
    # 4 terms for last column of B (Schoolbook)
    rank = 18
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Helper to add a Strassen block for a specific column pair of B
    # col_offset: 0 for cols 0,1; 2 for cols 2,3
    def add_strassen_block(col_offset, base_rank):
        # B indices:
        # B[0, col_offset] -> row col_offset
        # B[0, col_offset+1] -> row col_offset+1
        # B[1, col_offset] -> row p + col_offset
        # B[1, col_offset+1] -> row p + col_offset + 1
        
        # C indices:
        # C[0, col_offset] -> row col_offset * n + 0
        # C[0, col_offset+1] -> row (col_offset+1) * n + 0
        # C[1, col_offset] -> row col_offset * n + 1
        # C[1, col_offset+1] -> row (col_offset+1) * n + 1
        
        # A indices:
        # A[0,0] -> 0
        # A[0,1] -> 1
        # A[1,0] -> 2
        # A[1,1] -> 3
        
        # M1 = (a+d)(e+g)
        r = base_rank
        U[0, r] = 1; U[3, r] = 1
        V[col_offset, r] = 1; V[p + col_offset, r] = 1
        W[col_offset * n, r] = 1; W[(col_offset + 1) * n + 1, r] = 1
        
        # M2 = (c+d)e
        r = base_rank + 1
        U[2, r] = 1; U[3, r] = 1
        V[col_offset, r] = 1
        W[col_offset * n + 1, r] = 1; W[(col_offset + 1) * n + 1, r] = -1
        
        # M3 = a(f-g)
        r = base_rank + 2
        U[0, r] = 1
        V[col_offset + 1, r] = 1; V[p + col_offset + 1, r] = -1
        W[(col_offset + 1) * n, r] = 1; W[(col_offset + 1) * n + 1, r] = 1
        
        # M4 = d(h-f)
        r = base_rank + 3
        U[3, r] = 1
        V[p + col_offset, r] = 1; V[col_offset, r] = -1
        W[col_offset * n, r] = 1; W[col_offset * n + 1, r] = 1
        
        # M5 = (a+b)h
        r = base_rank + 4
        U[0, r] = 1; U[1, r] = 1
        V[p + col_offset + 1, r] = 1
        W[col_offset * n, r] = -1; W[(col_offset + 1) * n, r] = 1
        
        # M6 = (c-a)(e+f)
        r = base_rank + 5
        U[2, r] = 1; U[0, r] = -1
        V[col_offset, r] = 1; V[col_offset + 1, r] = 1
        W[(col_offset + 1) * n + 1, r] = 1
        
        # M7 = (b-d)(g+h)
        r = base_rank + 6
        U[1, r] = 1; U[3, r] = -1
        V[p + col_offset, r] = 1; V[p + col_offset + 1, r] = 1
        W[col_offset * n, r] = 1
        
    # Add Strassen for cols 0,1 (base 0)
    add_strassen_block(0, 0)
    
    # Add Strassen for cols 2,3 (base 7)
    add_strassen_block(2, 7)
    
    # Schoolbook for col 4 (base 14)
    # C[0,4] = A[0,0]B[0,4] + A[0,1]B[1,4]
    # C[1,4] = A[1,0]B[0,4] + A[1,1]B[1,4]
    
    # Term 14: A[0,0] B[0,4] -> C[0,4]
    r = 14
    U[0, r] = 1
    V[4, r] = 1  # B[0,4] is row 4
    W[4 * n + 0, r] = 1 # C[0,4] is row 8? 
    # Wait, W index is k*n + i. k=4, i=0 -> 8.
    # Let's re-verify W index.
    # Prompt: W[k*n+i, r].
    # My W array is (p*n, rank). p=5, n=2. Size 10.
    # C[0,4] -> k=4, i=0 -> index 8.
    # C[1,4] -> k=4, i=1 -> index 9.
    
    W[8, r] = 1
    
    # Term 15: A[0,1] B[1,4] -> C[0,4]
    r = 15
    U[1, r] = 1
    V[9, r] = 1 # B[1,4] is row 5+4=9
    W[8, r] = 1
    
    # Term 16: A[1,0] B[0,4] -> C[1,4]
    r = 16
    U[2, r] = 1
    V[4, r] = 1
    W[9, r] = 1
    
    # Term 17: A[1,1] B[1,4] -> C[1,4]
    r = 17
    U[3, r] = 1
    V[9, r] = 1
    W[9, r] = 1
    
    # Verify
    def check(U, V, W):
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        target = 1.0 if j == jp else 0.0
                        val = np.dot(U[i*m + j, :], V[jp*p + k, :] * W[k*n + i, :])
                        if abs(val - target) > 1e-9:
                            return False
        return True

    if check(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook rank 20 if verification fails (should not happen)
        rank = 20
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
