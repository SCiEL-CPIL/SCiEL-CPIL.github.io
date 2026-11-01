"""Tensor decomposition for 2x2 * 2x13 matrix multiplication."""
DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # safety margin before 150s kill
    
    n, m, p = 2, 2, 13
    
    # Build the target tensor T of shape (n*m, m*p, p*n)
    # T[(i*m+k), (k*p+j), (j*n+i)] = 1
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for k in range(m):
            for j in range(p):
                T[i*m+k, k*p+j, j*n+i] = 1.0
    
    # Strategy: Use Strassen's rank-7 decomposition for 2-column blocks
    # 6 blocks of 2 columns (12 cols) + 1 block of 1 column
    # For each 2-column block: rank 7 (Strassen) instead of rank 8 (schoolbook)
    # For the 1-column block: rank 4 (schoolbook)
    # Total: 6*7 + 4 = 46
    
    # Strassen decomposition for 2x2x2 sub-tensor
    # Local U indices: 0=(i=0,k=0), 1=(i=0,k=1), 2=(i=1,k=0), 3=(i=1,k=1)
    # Local V indices: 0=(k=0,j=k0), 1=(k=0,j=k1), 2=(k=1,j=k0), 3=(k=1,j=k1)
    # Local W indices: 0=(j=k0,i=0), 1=(j=k0,i=1), 2=(j=k1,i=0), 3=(j=k1,i=1)
    
    # Standard Strassen mapping:
    # M1 = (A11+A22)(B11+B22) -> C11 += M1, C22 += M1
    # M2 = (A21+A22)B11       -> C21 += M2
    # M3 = A11(B21-B11)       -> C11 += M3, C21 -= M3
    # M4 = A22(B12-B22)       -> C11 -= M4, C21 += M4
    # M5 = (A11+A12)B22       -> C12 += M5, C22 += M5
    # M6 = (A21-A11+A22)(B11+B12) -> C22 += M6
    # M7 = (A12-A21)(B12+B22) -> C11 -= M7, C12 += M7
    
    # A11=U[0], A12=U[1], A21=U[2], A22=U[3]
    # B11=V[0], B12=V[1], B21=V[2], B22=V[3]
    # C11=W[0], C12=W[1], C21=W[2], C22=W[3]
    
    U_sub = np.array([
        [1, 0, 1, 0, 1, -1, 0],   # A11
        [0, 0, 0, 0, 1,  0, 1],   # A12
        [0, 1, 0, 0, 0,  1,-1],   # A21
        [1, 1, 0, 1, 0,  1, 0],   # A22
    ])
    V_sub = np.array([
        [1, 1, 0, 0, 0, 1, 0],   # B11
        [0, 0, 0, 1, 0, 1, 1],   # B12
        [0, 0,-1, 0, 0, 0, 0],   # B21
        [1, 0, 1,-1, 1, 0, 1],   # B22
    ])
    W_sub = np.array([
        [ 1, 0, 1,-1, 0, 0,-1],  # C11
        [ 0, 0, 0, 0, 1, 0, 1],  # C12
        [ 0, 1,-1, 1, 0, 0, 0],  # C21
        [ 1, 0, 0, 0, 1, 1, 0],  # C22
    ])
    
    # Verify Strassen decomposition for the 2x2x2 sub-tensor
    S_computed = np.zeros((4, 4, 4))
    for t in range(7):
        S_computed += np.einsum('a,b,c->abc', U_sub[:,t], V_sub[:,t], W_sub[:,t])
    
    S_target = np.zeros((4, 4, 4))
    S_target[0,0,0] = 1; S_target[0,1,2] = 1
    S_target[1,2,0] = 1; S_target[1,3,2] = 1
    S_target[2,0,1] = 1; S_target[2,1,3] = 1
    S_target[3,2,1] = 1; S_target[3,3,3] = 1
    
    strassen_valid = np.allclose(S_computed, S_target, atol=1e-10)
    
    if not strassen_valid:
        # Fall back to schoolbook for all blocks
        rank = n * m * p  # 52
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        column = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    U[i * m + k, column] = 1
                    V[k * p + j, column] = 1
                    W[j * n + i, column] = 1
                    column += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Build the full decomposition with Strassen blocks
    rank = 6 * 7 + 4  # 46
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for block in range(6):
        k0 = block * 2
        k1 = block * 2 + 1
        
        # V local to global mapping:
        # local 0 -> global k0    (B row 0, col k0)
        # local 1 -> global k1    (B row 0, col k1)
        # local 2 -> global m+k0  (B row 1, col k0)
        # local 3 -> global m+k1  (B row 1, col k1)
        v_map = [k0, k1, m + k0, m + k1]
        
        # W local to global mapping:
        # local 0 -> global k0*n+0  (C col k0, row 0)
        # local 1 -> global k0*n+1  (C col k0, row 1)
        # local 2 -> global k1*n+0  (C col k1, row 0)
        # local 3 -> global k1*n+1  (C col k1, row 1)
        w_map = [k0 * n, k0 * n + 1, k1 * n, k1 * n + 1]
        
        for t in range(7):
            # U: local index a maps directly to global index a (since U indices are i*m+k)
            for a in range(4):
                if U_sub[a, t] != 0:
                    U[a, col + t] = U_sub[a, t]
            
            # V
            for b in range(4):
                if V_sub[b, t] != 0:
                    V[v_map[b], col + t] = V_sub[b, t]
            
            # W
            for c in range(4):
                if W_sub[c, t] != 0:
                    W[w_map[c], col + t] = W_sub[c, t]
        
        col += 7
    
    # Last single column: k = 12
    k_single = 12
    # Schoolbook for single column: 4 terms (i,k) pairs
    for t in range(4):
        i = t // 2
        k = t % 2
        U[i * m + k, col + t] = 1
        V[k * p + k_single, col + t] = 1
        W[k_single * n + i, col + t] = 1
    
    col += 4
    
    # Verify the full decomposition
    T_reconstructed = np.zeros((n*m, m*p, p*n))
    for t in range(rank):
        T_reconstructed += np.einsum('a,b,c->abc', U[:,t], V[:,t], W[:,t])
    
    if not np.allclose(T_reconstructed, T, atol=1e-10):
        # Fall back to schoolbook
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        column = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    U[i * m + k, column] = 1
                    V[k * p + j, column] = 1
                    W[j * n + i, column] = 1
                    column += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
