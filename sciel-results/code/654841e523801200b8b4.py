<<<<<<< SEARCH
import time
import numpy as np

DIMENSIONS = [2, 3, 8]

def construct():
    deadline = time.time() + 110
    n, m, p = DIMENSIONS  # 2, 3, 8
    
    def verify(U, V, W):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r, optimize=True)
        return np.allclose(T, np.ones((n, m, p)), atol=1e-10)
    
    # Schoolbook: rank = n*m*p = 48
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Should not reach here
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
=======
import time
import numpy as np

DIMENSIONS = [2, 3, 8]

def construct():
    deadline = time.time() + 110
    n, m, p = DIMENSIONS  # 2, 3, 8
    
    def verify(U, V, W):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r, optimize=True)
        return np.allclose(T, np.ones((n, m, p)), atol=1e-10)
    
    # Hybrid decomposition:
    # Partition m=3 into m1=1 (first row of B) and m2=2 (rows 2-3 of B)
    # For the 2x2x4 block (rows 2-3 of B), use Strassen-like algorithm
    # For the 2x1x4 block (row 1 of B), use direct computation
    
    # Strassen for 2x2x2 (with p=2):
    # M1 = (A11+A22)(B11+B22)
    # M2 = A22(B21-B11)
    # M3 = (A11-A22)(B12+B22)
    # M4 = A11(B12-B22)
    # M5 = (A11+A21)B22
    # M6 = (A22-A12)(B11+B21)
    # M7 = (A12+A22)(B21-B11)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # For 2x2x4, we split the 4 columns into two 2x2 blocks
    # Block 1: columns 0-1, Block 2: columns 2-3
    
    # Total rank terms:
    # 2x1x4 direct: 2*1*4 = 8 terms
    # 2x2x4 with Strassen: 2 blocks * 7 = 14 terms
    # Total: 8 + 14 = 22 terms
    
    rank = 22
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    col = 0
    
    # Direct computation for 2x1x4 (i in 0..1, j=0, k in 0..3)
    for i in range(n):
        for k in range(p):
            U[i*m+0, col] = 1
            V[0*p+k, col] = 1
            W[k*n+i, col] = 1
            col += 1
    
    # Strassen-like for 2x2x4 (i in 0..1, j in 1..2, k in 0..3)
    # Split k into two groups: k1 = 0,1 and k2 = 2,3
    
    for block in range(2):
        k_offset = block * 2
        
        # M1 = (A11+A22)(B11+B22)
        # A indices: i=0,j=1 and i=1,j=2 (0-indexed: j=1,2)
        # B indices: j=1,k=k_offset+0 and j=2,k=k_offset+1
        U[0*m+1, col] = 1  # A11
        U[1*m+2, col] = 1  # A22
        V[1*p+(k_offset+0), col] = 1  # B11
        V[2*p+(k_offset+1), col] = 1  # B22
        W[(k_offset+0)*n+0, col] = 1  # C11 part
        W[(k_offset+1)*n+1, col] = 1  # C22 part
        col += 1
        
        # M2 = A22(B21-B11)
        U[1*m+2, col] = 1  # A22
        V[2*p+(k_offset+0), col] = 1  # B21
        V[1*p+(k_offset+0), col] = -1  # B11
        W[(k_offset+0)*n+1, col] = 1  # C21
        W[(k_offset+1)*n+1, col] = -1  # C22 part
        col += 1
        
        # M3 = (A11-A22)(B12+B22)
        U[0*m+1, col] = 1  # A11
        U[1*m+2, col] = -1  # A22
        V[1*p+(k_offset+1), col] = 1  # B12
        V[2*p+(k_offset+1), col] = 1  # B22
        W[(k_offset+1)*n+0, col] = 1  # C12
        W[(k_offset+1)*n+1, col] = 1  # C22 part
        col += 1
        
        # M4 = A11(B12-B22)
        U[0*m+1, col] = 1  # A11
        V[1*p+(k_offset+1), col] = 1  # B12
        V[2*p+(k_offset+1), col] = -1  # B22
        W[(k_offset+0)*n+0, col] = 1  # C11 part
        W[(k_offset+0)*n+1, col] = 1  # C21
        col += 1
        
        # M5 = (A11+A21)B22
        U[0*m+1, col] = 1  # A11
        U[0*m+2, col] = 1  # A21 (i=0,j=2)
        V[2*p+(k_offset+1), col] = 1  # B22
        W[(k_offset+0)*n+0, col] = -1  # C11 part
        W[(k_offset+1)*n+0, col] = 1  # C12
        col += 1
        
        # M6 = (A22-A12)(B11+B21)
        U[1*m+2, col] = 1  # A22
        U[1*m+1, col] = -1  # A12 (i=1,j=1)
        V[1*p+(k_offset+0), col] = 1  # B11
        V[2*p+(k_offset+0), col] = 1  # B21
        W[(k_offset+1)*n+1, col] = 1  # C22 part
        col += 1
        
        # M7 = (A12+A22)(B21-B11)
        U[1*m+1, col] = 1  # A12
        U[1*m+2, col] = 1  # A22
        V[2*p+(k_offset+0), col] = 1  # B21
        V[1*p+(k_offset+0), col] = -1  # B11
        W[(k_offset+0)*n+0, col] = 1  # C11 part
        col += 1
    
    if verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook if verification fails
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
>>>>>>> REPLACE
