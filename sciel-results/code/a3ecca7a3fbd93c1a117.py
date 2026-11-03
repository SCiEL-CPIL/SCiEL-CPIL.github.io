import numpy as np

def construct():
    n, m, p = 2, 3, 6
    rank = 33
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for kb in range(3):
        # kb is the block index for p (0,1,2), each block has 2 columns
        k_offset = kb * 2
        # Split m=3 into m1=2, m2=1
        # For m1=2: Strassen rank 7
        # For m2=1: schoolbook rank 4
        # Total per block: 11
        
        # Strassen for 2x2x2 (m1=2, n=2, p_block=2)
        # A is 2x2 (rows i=0,1; cols j=0,1)
        # B is 2x2 (rows j=0,1; cols k=k_offset, k_offset+1)
        # C is 2x2 (rows i=0,1; cols k=k_offset, k_offset+1)
        
        # Strassen's 7 products:
        # M1 = (A11 + A22) * (B11 + B22)
        # M2 = (A21 + A22) * B11
        # M3 = A11 * (B12 - B22)
        # M4 = A22 * (B21 - B11)
        # M5 = (A11 + A12) * B22
        # M6 = (A21 - A11) * (B11 + B12)
        # M7 = (A12 - A22) * (B21 + B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # U indices for A: i*m + j, where i in {0,1}, j in {0,1}
        # V indices for B: j*p + k, where j in {0,1}, k in {k_offset, k_offset+1}
        # W indices for C: k*n + i, where k in {k_offset, k_offset+1}, i in {0,1}
        
        # M1: (A11 + A22) * (B11 + B22)
        # A11: i=0,j=0 -> U[0]; A22: i=1,j=1 -> U[4]
        # B11: j=0,k=k_offset -> V[k_offset]; B22: j=1,k=k_offset+1 -> V[6+k_offset+1]
        # C11: k=k_offset,i=0 -> W[k_offset*2+0]; C22: k=k_offset+1,i=1 -> W[(k_offset+1)*2+1]
        U[0, col] = 1.0; U[4, col] = 1.0
        V[k_offset, col] = 1.0; V[6+k_offset+1, col] = 1.0
        W[k_offset*2+0, col] = 1.0; W[(k_offset+1)*2+1, col] = 1.0
        col += 1
        
        # M2: (A21 + A22) * B11
        # A21: i=1,j=0 -> U[3]; A22: i=1,j=1 -> U[4]
        # B11: j=0,k=k_offset -> V[k_offset]
        # C21: k=k_offset,i=1 -> W[k_offset*2+1]
        U[3, col] = 1.0; U[4, col] = 1.0
        V[k_offset, col] = 1.0
        W[k_offset*2+1, col] = 1.0
        col += 1
        
        # M3: A11 * (B12 - B22)
        # A11: i=0,j=0 -> U[0]
        # B12: j=0,k=k_offset+1 -> V[k_offset+1]; B22: j=1,k=k_offset+1 -> V[6+k_offset+1]
        # C12: k=k_offset+1,i=0 -> W[(k_offset+1)*2+0]
        U[0, col] = 1.0
        V[k_offset+1, col] = 1.0; V[6+k_offset+1, col] = -1.0
        W[(k_offset+1)*2+0, col] = 1.0
        col += 1
        
        # M4: A22 * (B21 - B11)
        # A22: i=1,j=1 -> U[4]
        # B21: j=1,k=k_offset -> V[6+k_offset]; B11: j=0,k=k_offset -> V[k_offset]
        # C11: k=k_offset,i=0 -> W[k_offset*2+0]; C21: k=k_offset,i=1 -> W[k_offset*2+1]
        U[4, col] = 1.0
        V[6+k_offset, col] = 1.0; V[k_offset, col] = -1.0
        W[k_offset*2+0, col] = 1.0; W[k_offset*2+1, col] = 1.0
        col += 1
        
        # M5: (A11 + A12) * B22
        # A11: i=0,j=0 -> U[0]; A12: i=0,j=1 -> U[1]
        # B22: j=1,k=k_offset+1 -> V[6+k_offset+1]
        # C11: k=k_offset,i=0 -> W[k_offset*2+0]; C12: k=k_offset+1,i=0 -> W[(k_offset+1)*2+0]
        U[0, col] = 1.0; U[1, col] = 1.0
        V[6+k_offset+1, col] = 1.0
        W[k_offset*2+0, col] = 1.0; W[(k_offset+1)*2+0, col] = 1.0
        col += 1
        
        # M6: (A21 - A11) * (B11 + B12)
        # A21: i=1,j=0 -> U[3]; A11: i=0,j=0 -> U[0]
        # B11: j=0,k=k_offset -> V[k_offset]; B12: j=0,k=k_offset+1 -> V[k_offset+1]
        # C22: k=k_offset+1,i=1 -> W[(k_offset+1)*2+1]
        U[3, col] = 1.0; U[0, col] = -1.0
        V[k_offset, col] = 1.0; V[k_offset+1, col] = 1.0
        W[(k_offset+1)*2+1, col] = 1.0
        col += 1
        
        # M7: (A12 - A22) * (B21 + B22)
        # A12: i=0,j=1 -> U[1]; A22: i=1,j=1 -> U[4]
        # B21: j=1,k=k_offset -> V[6+k_offset]; B22: j=1,k=k_offset+1 -> V[6+k_offset+1]
        # C11: k=k_offset,i=0 -> W[k_offset*2+0]; C22: k=k_offset+1,i=1 -> W[(k_offset+1)*2+1]
        U[1, col] = 1.0; U[4, col] = -1.0
        V[6+k_offset, col] = 1.0; V[6+k_offset+1, col] = 1.0
        W[k_offset*2+0, col] = 1.0; W[(k_offset+1)*2+1, col] = 1.0
        col += 1
        
        # Schoolbook for 2x1x2 (m2=1, n=2, p_block=2)
        # A is 2x1 (rows i=0,1; col j=2)
        # B is 1x2 (row j=2; cols k=k_offset, k_offset+1)
        # C is 2x2 (rows i=0,1; cols k=k_offset, k_offset+1)
        # C[i,k] = A[i,2] * B[2,k]
        
        # 4 terms: (i,k) in {(0,k_offset), (0,k_offset+1), (1,k_offset), (1,k_offset+1)}
        for i in range(2):
            for kk in range(2):
                k_abs = k_offset + kk
                U[i*m+2, col] = 1.0
                V[2*p+k_abs, col] = 1.0
                W[k_abs*n+i, col] = 1.0
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
