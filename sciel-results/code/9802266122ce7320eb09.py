import numpy as np

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = 2, 3, 3
    rank = 9
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # Strassen decomposition for the M_{2,2,2} part (top-left 2x2 of B, all of A for j=0,1, k=0,1)
    # Local indices: i in {0,1}, j in {0,1}, k in {0,1}
    # Strassen factors for 2x2 matrices:
    # U rows correspond to (i,j) local -> global (i*3+j)
    # V rows correspond to (j,k) local -> global (j*3+k)
    # W rows correspond to (k,i) local -> global (k*2+i)
    
    strassen_U = [
        # term 0: (i=0,j=0)
        [1, 0, 0, 0, 0, 0, 0],
        # term 1: (i=1,j=0)
        [0, 0, 0, 0, 0, 0, 0],
        # term 2: (i=1,j=1)
        [0, 0, 0, 0, 0, 0, 0],
        # Wait, U is 2x2 local. Let me re-derive properly.
    ]
    
    # Proper Strassen 2x2x2 factors:
    # Local A is 2x2, local B is 2x2, local C is 2x2.
    # M1 = A[0,0] * B[0,0]  (using full row/col for simplicity in tensor terms)
    # Actually, the standard Strassen decomposition uses:
    # M1 = (A00) * (B00)
    # M2 = (A10) * (B00)
    # M3 = (A00 - A10) * (B10 + B11)
    # M4 = (A10 + A11) * (B00 - B10)
    # M5 = (A00 + A01) * (B11 - B01)
    # M6 = (A01 - A00) * (B00 + B10)
    # M7 = (A11 - A10) * (B01 + B11)
    # Wait, the standard decomposition is:
    # C00 = M1 + M4 - M5 - M7
    # C01 = M2 + M3 + M5
    # C10 = M2 + M4 - M6 - M7
    # C11 = M1 - M3 + M6 + M7
    
    # Let's build the factor matrices for the 2x2x2 sub-problem (j=0,1; k=0,1; i=0,1)
    # U_sub: shape (2*2, 7) = (4, 7) for local (i,j)
    # V_sub: shape (2*2, 7) = (4, 7) for local (j,k)
    # W_sub: shape (2*2, 7) = (4, 7) for local (k,i)
    
    U_sub = np.zeros((4, 7))
    V_sub = np.zeros((4, 7))
    W_sub = np.zeros((4, 7))
    
    # Local indexing: U_sub[i*2+j], V_sub[j*2+k], W_sub[k*2+i]
    # Term 0 (M1): A00*B00
    # A00: i=0,j=0 -> row 0. B00: j=0,k=0 -> row 0. C00: k=0,i=0 -> row 0. C11: k=1,i=1 -> row 3.
    # M1 = A00 * B00. U has 1 at (0,0). V has 1 at (0,0).
    # Contribution to C00 (+1) and C11 (-1)? No, C00 = M1 + ..., C11 = M1 - ...
    # So W has +1 at C00 (k=0,i=0 -> 0*2+0=0) and -1 at C11 (k=1,i=1 -> 1*2+1=3).
    U_sub[0, 0] = 1
    V_sub[0, 0] = 1
    W_sub[0, 0] = 1
    W_sub[3, 0] = -1
    
    # Term 1 (M2): A10*B00. A10: i=1,j=0 -> row 2. B00: j=0,k=0 -> row 0.
    # C01 = ... + M2 + ... (k=1,i=0 -> 1*2+0=2)
    # C10 = ... + M2 + ... (k=0,i=1 -> 0*2+1=1)
    U_sub[2, 1] = 1
    V_sub[0, 1] = 1
    W_sub[2, 1] = 1
    W_sub[1, 1] = 1
    
    # Term 2 (M3): (A00-A10)*(B10+B11).
    # A00: row 0, A10: row 2. So U has 1 at row 0, -1 at row 2.
    # B10: j=1,k=0 -> row 2. B11: j=1,k=1 -> row 3. So V has 1 at row 2, 1 at row 3.
    # C01 = ... + M3 + ... (row 2)
    # C11 = ... - M3 + ... (row 3)
    U_sub[0, 2] = 1
    U_sub[2, 2] = -1
    V_sub[2, 2] = 1
    V_sub[3, 2] = 1
    W_sub[2, 2] = 1
    W_sub[3, 2] = -1
    
    # Term 3 (M4): (A10+A11)*(B00-B10).
    # A10: row 2, A11: i=1,j=1 -> row 3. U has 1 at 2, 1 at 3.
    # B00: row 0, B10: row 2. V has 1 at 0, -1 at 2.
    # C00 = ... + M4 + ... (row 0)
    # C10 = ... + M4 + ... (row 1)
    U_sub[2, 3] = 1
    U_sub[3, 3] = 1
    V_sub[0, 3] = 1
    V_sub[2, 3] = -1
    W_sub[0, 3] = 1
    W_sub[1, 3] = 1
    
    # Term 4 (M5): (A00+A01)*(B11-B01).
    # A00: row 0, A01: i=0,j=1 -> row 1. U has 1 at 0, 1 at 1.
    # B11: row 3, B01: j=0,k=1 -> row 1. V has 1 at 3, -1 at 1.
    # C00 = ... - M5 + ... (row 0)
    # C01 = ... + M5 + ... (row 2)
    U_sub[0, 4] = 1
    U_sub[1, 4] = 1
    V_sub[3, 4] = 1
    V_sub[1, 4] = -1
    W_sub[0, 4] = -1
    W_sub[2, 4] = 1
    
    # Term 5 (M6): (A01-A00)*(B00+B10).
    # A01: row 1, A00: row 0. U has 1 at 1, -1 at 0.
    # B00: row 0, B10: row 2. V has 1 at 0, 1 at 2.
    # C10 = ... - M6 + ... (row 1)
    # C11 = ... + M6 + ... (row 3)
    U_sub[1, 5] = 1
    U_sub[0, 5] = -1
    V_sub[0, 5] = 1
    V_sub[2, 5] = 1
    W_sub[1, 5] = -1
    W_sub[3, 5] = 1
    
    # Term 6 (M7): (A11-A10)*(B01+B11).
    # A11: row 3, A10: row 2. U has 1 at 3, -1 at 2.
    # B01: row 1, B11: row 3. V has 1 at 1, 1 at 3.
    # C00 = ... - M7 + ... (row 0)
    # C10 = ... - M7 + ... (row 1)
    # C11 = ... + M7 + ... (row 3)
    U_sub[3, 6] = 1
    U_sub[2, 6] = -1
    V_sub[1, 6] = 1
    V_sub[3, 6] = 1
    W_sub[0, 6] = -1
    W_sub[1, 6] = -1
    W_sub[3, 6] = 1
    
    # Map Strassen factors to global indices
    # Global U row: i*3 + j. Local i in {0,1}, j in {0,1}.
    # Global V row: j*3 + k. Local j in {0,1}, k in {0,1}.
    # Global W row: k*2 + i. Local k in {0,1}, i in {0,1}.
    
    # U_sub rows: 0:(i=0,j=0), 1:(i=0,j=1), 2:(i=1,j=0), 3:(i=1,j=1)
    # Global U rows: 0*3+0=0, 0*3+1=1, 1*3+0=3, 1*3+1=4
    U_global_rows = [0, 1, 3, 4]
    for local_idx, global_idx in enumerate(U_global_rows):
        U[global_idx, 0:7] = U_sub[local_idx, :]
        
    # V_sub rows: 0:(j=0,k=0), 1:(j=0,k=1), 2:(j=1,k=0), 3:(j=1,k=1)
    # Global V rows: 0*3+0=0, 0*3+1=1, 1*3+0=3, 1*3+1=4
    V_global_rows = [0, 1, 3, 4]
    for local_idx, global_idx in enumerate(V_global_rows):
        V[global_idx, 0:7] = V_sub[local_idx, :]
        
    # W_sub rows: 0:(k=0,i=0), 1:(k=0,i=1), 2:(k=1,i=0), 3:(k=1,i=1)
    # Global W rows: 0*2+0=0, 0*2+1=1, 1*2+0=2, 1*2+1=3
    W_global_rows = [0, 1, 2, 3]
    for local_idx, global_idx in enumerate(W_global_rows):
        W[global_idx, 0:7] = W_sub[local_idx, :]
        
    # Term 7: Handle j=2 (the third column of A)
    # M_{2,3,1} part: A[:, 2] * B[2, :2]
    # Local: i in {0,1}, j=2, k in {0,1}
    # U row: i*3+2 -> 2, 5
    # V row: 2*3+k -> 6, 7
    # W row: k*2+i -> 0, 1, 2, 3
    for i in range(2):
        U[i*3+2, 7] = 1
    for k in range(2):
        V[2*3+k, 7] = 1
    for k in range(2):
        for i in range(2):
            W[k*2+i, 7] = 1
            
    # Term 8: Handle k=2 (the third row of B)
    # M_{2,2,1} part: A[:, 0:2] * B[0:2, 2]
    # Local: i in {0,1}, j in {0,1}, k=2
    # U row: i*3+j -> 0,1,3,4
    # V row: j*3+2 -> 2, 5
    # W row: 2*2+i -> 4, 5
    for i in range(2):
        for j in range(2):
            U[i*3+j, 8] = 1
    for j in range(2):
        V[j*3+2, 8] = 1
    for i in range(2):
        W[2*2+i, 8] = 1
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
