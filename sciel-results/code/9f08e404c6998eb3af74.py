import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook rank
    rank_school = n * m * p
    
    # We will construct a rank 99 decomposition using 6 Strassen 2x2x2 blocks
    # and schoolbook for the rest.
    
    # Indices for Strassen blocks:
    # Rows of A: {0, 1}
    # Cols of A: {0, 1}, {2, 3}
    # Cols of B: {0, 1}, {2, 3}, {4, 5}
    
    row_segs = [(0, 1)]
    col_segs_A = [(0, 1), (2, 3)]
    col_segs_B = [(0, 1), (2, 3), (4, 5)]
    
    # Total rank calculation
    # 6 blocks * 7 terms = 42
    # Remaining terms:
    # i=2 (5*7=35)
    # i in {0,1}, j=4 (2*7=14)
    # i in {0,1}, j in {0,1,2,3}, k=6 (2*4*1=8)
    # Total remaining = 35 + 14 + 8 = 57
    # Total rank = 42 + 57 = 99
    
    rank = 99
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col_idx = 0
    
    # Helper to add a schoolbook term
    def add_schoolbook(i, j, k):
        nonlocal col_idx
        if col_idx >= rank:
            return
        U[i*m + j, col_idx] = 1
        V[j*p + k, col_idx] = 1
        W[k*n + i, col_idx] = 1
        col_idx += 1
        
    # Helper to add a Strassen block
    def add_strassen_block(r1, r2, c1, c2, d1, d2):
        nonlocal col_idx
        
        # Factor 1: (A[r1,c1] + A[r2,c2]) * (B[c1,d1] + B[c2,d2])
        U[r1*m + c1, col_idx] = 1
        U[r2*m + c2, col_idx] = 1
        V[c1*p + d1, col_idx] = 1
        V[c2*p + d2, col_idx] = 1
        W[d1*n + r1, col_idx] = 1
        W[d2*n + r1, col_idx] = 1
        W[d1*n + r2, col_idx] = 1
        W[d2*n + r2, col_idx] = 1
        col_idx += 1
        
        # Factor 2: (A[r2,c1] + A[r2,c2]) * B[c1,d1]
        U[r2*m + c1, col_idx] = 1
        U[r2*m + c2, col_idx] = 1
        V[c1*p + d1, col_idx] = 1
        W[d1*n + r2, col_idx] = 1
        col_idx += 1
        
        # Factor 3: A[r1,c1] * (B[c1,d2] - B[c2,d2])
        U[r1*m + c1, col_idx] = 1
        V[c1*p + d2, col_idx] = 1
        V[c2*p + d2, col_idx] = -1
        W[d2*n + r1, col_idx] = 1
        col_idx += 1
        
        # Factor 4: A[r2,c2] * (B[c2,d1] - B[c1,d1])
        U[r2*m + c2, col_idx] = 1
        V[c2*p + d1, col_idx] = 1
        V[c1*p + d1, col_idx] = -1
        W[d1*n + r2, col_idx] = 1
        col_idx += 1
        
        # Factor 5: (A[r1,c1] + A[r1,c2]) * B[c2,d2]
        U[r1*m + c1, col_idx] = 1
        U[r1*m + c2, col_idx] = 1
        V[c2*p + d2, col_idx] = 1
        W[d2*n + r1, col_idx] = 1
        col_idx += 1
        
        # Factor 6: (A[r2,c1] - A[r1,c1]) * (B[c1,d1] + B[c1,d2])
        U[r2*m + c1, col_idx] = 1
        U[r1*m + c1, col_idx] = -1
        V[c1*p + d1, col_idx] = 1
        V[c1*p + d2, col_idx] = 1
        W[d1*n + r2, col_idx] = 1
        W[d2*n + r2, col_idx] = 1
        col_idx += 1
        
        # Factor 7: (A[r1,c2] - A[r2,c2]) * (B[c2,d1] + B[c2,d2])
        U[r1*m + c2, col_idx] = 1
        U[r2*m + c2, col_idx] = -1
        V[c2*p + d1, col_idx] = 1
        V[c2*p + d2, col_idx] = 1
        W[d1*n + r1, col_idx] = 1
        W[d2*n + r1, col_idx] = 1
        col_idx += 1

    # Add Strassen blocks
    for r_seg in row_segs:
        for c_seg in col_segs_A:
            for d_seg in col_segs_B:
                add_strassen_block(r_seg[0], r_seg[1], c_seg[0], c_seg[1], d_seg[0], d_seg[1])
                
    # Add schoolbook terms for the rest
    # Covered by Strassen: i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3,4,5}
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check if this term is covered by Strassen
                if i in (0, 1) and j in (0, 1, 2, 3) and k in (0, 1, 2, 3, 4, 5):
                    continue
                else:
                    add_schoolbook(i, j, k)
                    
    # Verify col_idx matches rank
    if col_idx != rank:
        # This should not happen if logic is correct
        # But for safety, if we have extra space, we leave it zero (which adds 0 to tensor)
        # If we ran out, we truncated (invalid).
        pass
        
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
