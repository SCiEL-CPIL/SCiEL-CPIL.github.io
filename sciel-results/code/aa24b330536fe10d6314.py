import time
import numpy as np

DIMENSIONS = [2, 2, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    
    rank = 32
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen 2x2 by 2x2
    # M1 = (A22 - A11)(B11 - B22)
    # M2 = (A11 + A22)B22
    # M3 = (A12 - A22)B21
    # M4 = (A11 + A12)B11
    # M5 = (A21 + A11)B12
    # M6 = (A11 - A21)(B12 - B11)
    # M7 = (A12 + A22)(B21 - B11)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M2 + M6
    
    for b in range(4):
        col_offset = b * 2
        base = b * 7
        
        # M1
        U[1, base] = 1  # A22
        U[0, base] = -1 # A11
        V[col_offset, base] = 1 # B11
        V[col_offset+1, base] = -1 # B22
        W[col_offset, base] = 1 # C11 (k=0, i=0) -> W[0]
        W[col_offset+1, base] = 1 # C22 (k=1, i=1) -> W[3]
        
        # M2
        U[0, base+1] = 1 # A11
        U[1, base+1] = 1 # A22
        V[col_offset+1, base+1] = 1 # B22
        W[col_offset+1, base+1] = 1 # C21 (k=1, i=0) -> W[2]
        W[col_offset+1, base+1] = 1 # C22 (k=1, i=1) -> W[3]
        # Wait, C21 is k=1, i=0 -> W[1*2+0] = W[2]
        # C22 is k=1, i=1 -> W[1*2+1] = W[3]
        # So M2 contributes to W[2] and W[3]
        W[col_offset*2 + 2, base+1] = 1 # C21
        W[col_offset*2 + 3, base+1] = 1 # C22
        
        # M3
        U[1, base+2] = 1 # A12
        U[1, base+2] = -1 # A22 (wait, A12 is index 1, A22 is index 3)
        # A is 2x2, indices 0,1,2,3
        # A11=0, A12=1, A21=2, A22=3
        U[1, base+2] = 1 # A12
        U[3, base+2] = -1 # A22
        V[col_offset+1, base+2] = 1 # B21
        W[col_offset, base+2] = 1 # C12 (k=0, i=1) -> W[1]
        
        # M4
        U[0, base+3] = 1 # A11
        U[1, base+3] = 1 # A12
        V[col_offset, base+3] = 1 # B11
        W[col_offset, base+3] = 1 # C11 (k=0, i=0) -> W[0]
        W[col_offset*2 + 2, base+3] = 1 # C21 (k=1, i=0) -> W[2]
        
        # M5
        U[2, base+4] = 1 # A21
        U[0, base+4] = 1 # A11
        V[col_offset+1, base+4] = 1 # B12
        W[col_offset, base+4] = 1 # C12 (k=0, i=1) -> W[1]
        W[col_offset, base+4] = -1 # C11 (k=0, i=0) -> W[0]
        
        # M6
        U[0, base+5] = 1 # A11
        U[2, base+5] = -1 # A21
        V[col_offset+1, base+5] = 1 # B12
        V[col_offset, base+5] = -1 # B11
        W[col_offset+1, base+5] = 1 # C22 (k=1, i=1) -> W[3]
        
        # M7
        U[1, base+6] = 1 # A12
        U[3, base+6] = 1 # A22
        V[col_offset+1, base+6] = 1 # B21
        V[col_offset, base+6] = -1 # B11
        W[col_offset, base+6] = 1 # C11 (k=0, i=0) -> W[0]
        
    # Block 5: column 8
    # 2x1 by 2x1
    # C1 = A11 B1 + A12 B2
    # C2 = A21 B1 + A22 B2
    base = 28
    # M1 = A11 B1
    U[0, base] = 1
    V[8, base] = 1
    W[8*2+0, base] = 1 # C1 (k=8, i=0) -> W[16]
    
    # M2 = A12 B2
    U[1, base+1] = 1
    V[8, base+1] = 1 # Wait, B is 2x9, so B1 is B[0,8], B2 is B[1,8]
    # V[j*p+k, r]
    # B1 is j=0, k=8 -> V[0*9+8] = V[8]
    # B2 is j=1, k=8 -> V[1*9+8] = V[17]
    V[17, base+1] = 1
    W[8*2+0, base+1] = 1 # C1
    
    # M3 = A21 B1
    U[2, base+2] = 1
    V[8, base+2] = 1
    W[8*2+1, base+2] = 1 # C2 (k=8, i=1) -> W[17]
    
    # M4 = A22 B2
    U[3, base+3] = 1
    V[17, base+3] = 1
    W[8*2+1, base+3] = 1 # C2
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
