"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    deadline = time.time() + 140  # safety margin
    
    # We'll use 4 Strassen blocks for A[0:2,0:2] x B[0:2, 0:8] (in 2-col chunks)
    # plus schoolbook for the rest
    # Total rank: 4*7 + 16 + 24 = 28 + 40 = 68
    
    # Let's build the factor matrices
    # U: (n*m, rank) = (9, rank)
    # V: (m*p, rank) = (24, rank)  
    # W: (p*n, rank) = (24, rank)
    
    # Index conventions:
    # U[i*m + j, c] - row index i*m+j for A[i,j]
    # V[j*p + k, c] - row index j*p+k for B[j,k]
    # W[k*n + i, c] - row index k*n+i for C[i,k] (output factor follows W[k*n+i,r])
    
    rank = 68
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    # 4 Strassen blocks: A[0:2, 0:2] with B[0:2, k0:k0+2] for k0 in [0,2,4,6]
    # A rows: i1=0, i2=1; A cols: j1=0, j2=1
    # B rows: j1=0, j2=1; B cols: k1, k2
    
    i1, i2 = 0, 1
    j1, j2 = 0, 1
    
    for k0 in [0, 2, 4, 6]:
        k1, k2 = k0, k0 + 1
        
        # Strassen products (7 per block)
        # M1 = (A[i1,j1]+A[i2,j2]) * (B[j1,k1]+B[j2,k2])
        U[i1*m+j1, col] = 1
        U[i2*m+j2, col] = 1
        V[j1*p+k1, col] = 1
        V[j2*p+k2, col] = 1
        W[k1*n+i1, col] = 1  # C[i1,k1] += M1
        W[k2*n+i2, col] = 1  # C[i2,k2] += M1
        col += 1
        
        # M2 = (A[i2,j1]+A[i2,j2]) * B[j1,k1]
        U[i2*m+j1, col] = 1
        U[i2*m+j2, col] = 1
        V[j1*p+k1, col] = 1
        W[k1*n+i2, col] = 1   # C[i2,k1] += M2
        W[k2*n+i2, col] = -1  # C[i2,k2] -= M2
        col += 1
        
        # M3 = A[i1,j1] * (B[j2,k1]-B[j2,k2])
        U[i1*m+j1, col] = 1
        V[j2*p+k1, col] = 1
        V[j2*p+k2, col] = -1
        W[k1*n+i1, col] = 1   # C[i1,k1] += M3
        W[k2*n+i1, col] = -1  # C[i1,k2] -= M3
        col += 1
        
        # M4 = A[i2,j2] * (B[j1,k2]-B[j1,k1])
        U[i2*m+j2, col] = 1
        V[j1*p+k2, col] = 1
        V[j1*p+k1, col] = -1
        W[k1*n+i1, col] = 1   # C[i1,k1] += M4
        W[k1*n+i2, col] = 1   # C[i2,k1] += M4
        col += 1
        
        # M5 = (A[i1,j1]+A[i1,j2]) * B[j2,k2]
        U[i1*m+j1, col] = 1
        U[i1*m+j2, col] = 1
        V[j2*p+k2, col] = 1
        W[k1*n+i1, col] = -1  # C[i1,k1] -= M5
        W[k2*n+i1, col] = 1   # C[i1,k2] += M5
        col += 1
        
        # M6 = (A[i2,j1]-A[i1,j1]) * (B[j1,k1]+B[j1,k2])
        U[i2*m+j1, col] = 1
        U[i1*m+j1, col] = -1
        V[j1*p+k1, col] = 1
        V[j1*p+k2, col] = 1
        W[k2*n+i2, col] = 1   # C[i2,k2] += M6
        col += 1
        
        # M7 = (A[i1,j2]-A[i2,j2]) * (B[j2,k1]+B[j2,k2])
        U[i1*m+j2, col] = 1
        U[i2*m+j2, col] = -1
        V[j2*p+k1, col] = 1
        V[j2*p+k2, col] = 1
        W[k1*n+i1, col] = 1   # C[i1,k1] += M7
        W[k2*n+i2, col] = 1   # C[i2,k2] += M7
        col += 1
    
    # Schoolbook: A[0:2, 2] * B[2, :] -> C[0:2, :]
    # C[i, k] += A[i, 2] * B[2, k] for i in [0,1], k in [0..7]
    for i in range(2):
        for k in range(p):
            U[i*m+2, col] = 1
            V[2*p+k, col] = 1
            W[k*n+i, col] = 1
            col += 1
    
    # Schoolbook: A[2, :] * B[:, :] -> C[2, :]
    # C[2, k] += A[2, j] * B[j, k] for j in [0,1,2], k in [0..7]
    for j in range(m):
        for k in range(p):
            U[2*m+j, col] = 1
            V[j*p+k, col] = 1
            W[k*n+2, col] = 1
            col += 1
    
    assert col == rank, f"Column count {col} != rank {rank}"
    
    # Verify the decomposition
    # The tensor T[i,j,k] = 1 should equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # But actually, the tensor for matrix multiplication is:
    # T[i,j,k] = 1 for all i,j,k (since C[i,k] = sum_j A[i,j]*B[j,k])
    # The decomposition requires: for all i,j,k, sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    
    # Let's verify
    U_mat = U  # (9, 68)
    V_mat = V  # (24, 68)
    W_mat = W  # (24, 68)
    
    # Reshape for verification
    U_r = U_mat.reshape(n, m, rank)
    V_r = V_mat.reshape(m, p, rank)
    W_r = W_mat.reshape(p, n, rank)
    
    # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # Use einsum
    T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
    
    # T should be all 1s (shape n x m x p = 3 x 3 x 8)
    if not np.allclose(T, 1.0, atol=1e-10):
        # Fall back to schoolbook
        rank_s = n * m * p
        U_s = np.zeros((n*m, rank_s))
        V_s = np.zeros((m*p, rank_s))
        W_s = np.zeros((p*n, rank_s))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i*m+j, c] = 1
                    V_s[j*p+k, c] = 1
                    W_s[k*n+i, c] = 1
                    c += 1
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    
    return {"U": U_mat.tolist(), "V": V_mat.tolist(), "W": W_mat.tolist()}
# EVOLVE-BLOCK-END
