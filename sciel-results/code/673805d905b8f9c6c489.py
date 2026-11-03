# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=10
    # Split A (2x3) into A1 (2x2) and A2 (2x1)
    # Split B (3x10) into B1 (2x10) and B2 (1x10)
    # C = A1 B1 + A2 B2
    
    # Rank of A1 B1 (2x2 by 2x10) using 5 Strassen blocks (2x2 by 2x2)
    # Each Strassen block has rank 7. Total 35.
    # Rank of A2 B2 (2x1 by 1x10) is trivial 20.
    # Total rank = 35 + 20 = 55.
    
    rank = 55
    U = np.zeros((n * m, rank)) # 6 x 55
    V = np.zeros((m * p, rank)) # 30 x 55
    W = np.zeros((p * n, rank)) # 20 x 55
    
    # Helper to add a Strassen block
    # A1 is rows 0,1 of A. Indices 0,1 (row 0), 3,4 (row 1) in U?
    # A indices:
    # 0: A[0,0]
    # 1: A[0,1]
    # 2: A[0,2]
    # 3: A[1,0]
    # 4: A[1,1]
    # 5: A[1,2]
    
    # B indices:
    # B[j, k] is at j*p + k.
    # B1 is rows 0,1.
    # B[0, 2t] is 2t.
    # B[0, 2t+1] is 2t+1.
    # B[1, 2t] is 10+2t.
    # B[1, 2t+1] is 10+2t+1.
    
    # C indices:
    # C[i, k] is i*p + k.
    # C[0, 2t] is 2t.
    # C[0, 2t+1] is 2t+1.
    # C[1, 2t] is 10+2t.
    # C[1, 2t+1] is 10+2t+1.
    
    for t in range(5):
        base_col = t * 7
        # A1 elements
        a00_idx = 0
        a01_idx = 1
        a10_idx = 3
        a11_idx = 4
        
        # B1 elements for block t
        b00_idx = 2*t
        b01_idx = 2*t + 1
        b10_idx = 10 + 2*t
        b11_idx = 10 + 2*t + 1
        
        # C elements for block t
        c00_idx = 2*t
        c01_idx = 2*t + 1
        c10_idx = 10 + 2*t
        c11_idx = 10 + 2*t + 1
        
        # M1 = (a00 + a11)(b00 + b11)
        col = base_col
        U[a00_idx, col] += 1
        U[a11_idx, col] += 1
        V[b00_idx, col] += 1
        V[b11_idx, col] += 1
        W[c00_idx, col] += 1
        W[c11_idx, col] += 1
        
        # M2 = (a10 + a11)b00
        col = base_col + 1
        U[a10_idx, col] += 1
        U[a11_idx, col] += 1
        V[b00_idx, col] += 1
        W[c10_idx, col] += 1
        W[c11_idx, col] += -1
        
        # M3 = a00(b10 - b11)
        col = base_col + 2
        U[a00_idx, col] += 1
        V[b10_idx, col] += 1
        V[b11_idx, col] += -1
        W[c01_idx, col] += 1
        W[c11_idx, col] += 1
        
        # M4 = a11(b01 - b00)
        col = base_col + 3
        U[a11_idx, col] += 1
        V[b01_idx, col] += 1
        V[b00_idx, col] += -1
        W[c00_idx, col] += 1
        W[c10_idx, col] += 1
        
        # M5 = (a00 + a01)b11
        col = base_col + 4
        U[a00_idx, col] += 1
        U[a01_idx, col] += 1
        V[b11_idx, col] += 1
        W[c00_idx, col] += -1
        W[c01_idx, col] += 1
        
        # M6 = (a10 - a00)(b00 + b01)
        col = base_col + 5
        U[a10_idx, col] += 1
        U[a00_idx, col] += -1
        V[b00_idx, col] += 1
        V[b01_idx, col] += 1
        W[c11_idx, col] += 1
        
        # M7 = (a01 - a11)(b10 + b11)
        col = base_col + 6
        U[a01_idx, col] += 1
        U[a11_idx, col] += -1
        V[b10_idx, col] += 1
        V[b11_idx, col] += 1
        W[c00_idx, col] += 1
        
    # Trivial part for A2 B2
    # A2 is col 2 of A. Indices 2 (A[0,2]) and 5 (A[1,2]).
    # B2 is row 2 of B. Indices 20+k (B[2,k]).
    # C is all cols.
    
    for k in range(10):
        # C[0,k] = A[0,2] * B[2,k]
        col = 35 + k
        U[2, col] = 1
        V[20 + k, col] = 1
        W[k, col] = 1
        
        # C[1,k] = A[1,2] * B[2,k]
        col = 45 + k
        U[5, col] = 1
        V[20 + k, col] = 1
        W[10 + k, col] = 1
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
