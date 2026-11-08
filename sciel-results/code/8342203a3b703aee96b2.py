"""Matrix multiplication 2x4x6 rank-minimal decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 4, 6]


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120

    # Strategy:
    # The matrix multiplication tensor T_{n,m,p} has entries T[i,j,k] = 1.
    # The decomposition is T[i,j,k] = sum_t U[i*m+j, t] V[j*p+k, t] W[k*n+i, t].
    # 
    # For n=2, m=4, p=6:
    # The product C = A * B where A is 2x4 and B is 4x6.
    # C[0, k] = sum_{j=0}^{3} A[0, j] B[j, k]
    # C[1, k] = sum_{j=0}^{3} A[1, j] B[j, k]
    #
    # Notice that the computation for C[0, :] and C[1, :] are independent.
    # Specifically, the term A[0, j] only contributes to C[0, k], and A[1, j] only to C[1, k].
    # In the tensor T[i,j,k], the index i determines which row of A is used, and also which row of C is produced.
    #
    # We can decompose T_{2,4,6} into two blocks based on i:
    # Block 0 (i=0): T[0, j, k] = 1 for j in 0..3, k in 0..5.
    # This is effectively the tensor for 1x4x4 multiplication? No, it's 1x4x6.
    # Wait, the sub-tensor for fixed i=0 is T_0[j,k] = 1 for all j,k.
    # This is the tensor for multiplying a 1x4 matrix by a 4x6 matrix.
    # The rank of T_{1,m,p} is 1.
    # Proof: T_{1,m,p}[j,k] = 1 = u_j v_k w? No.
    # The decomposition for T_{1,m,p} is:
    # T_{1,m,p}[j,k] = sum_t U[j, t] V[k, t] (since i is fixed, W is absorbed or just 1).
    # Actually, the full decomposition is U[i*m+j, t] V[j*p+k, t] W[k*n+i, t].
    # For i=0:
    # We need sum_t U[j, t] V[j*p+k, t] W[k, t] = 1 for all j,k.
    # We can achieve this with rank 1:
    # Let t=0.
    # U[j, 0] = 1 for all j.
    # V[j*p+k, 0] = 1 for all j,k.
    # W[k, 0] = 1 for all k.
    # Then sum = 1*1*1 = 1.
    # 
    # Similarly for i=1:
    # We need sum_t U[4+j, t] V[j*p+k, t] W[6+k, t] = 1 for all j,k.
    # Let t=1.
    # U[4+j, 1] = 1 for all j.
    # V[j*p+k, 1] = 1 for all j,k.
    # W[6+k, 1] = 1 for all k.
    # Then sum = 1*1*1 = 1.
    #
    # Total Rank = 2.
    # 
    # Let's verify the shapes.
    # U: (n*m, rank) = (8, 2)
    # V: (m*p, rank) = (24, 2)
    # W: (p*n, rank) = (12, 2)
    
    rank = 2
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Term 0: Handles i=0
    # U indices for i=0: 0*m + j = j. So U[j, 0] = 1 for j=0..3.
    for j in range(m):
        U[j, 0] = 1.0
        
    # V indices: j*p + k. For this term to work for all j,k, V[j*p+k, 0] = 1.
    for j in range(m):
        for k in range(p):
            V[j * p + k, 0] = 1.0
            
    # W indices for i=0: k*n + i = k*2 + 0 = 2k.
    # We need W[2k, 0] = 1 for k=0..5.
    for k in range(p):
        W[k * n, 0] = 1.0
        
    # Term 1: Handles i=1
    # U indices for i=1: 1*m + j = 4 + j. So U[4+j, 1] = 1 for j=0..3.
    for j in range(m):
        U[m + j, 1] = 1.0
        
    # V indices: j*p + k. For this term to work for all j,k, V[j*p+k, 1] = 1.
    for j in range(m):
        for k in range(p):
            V[j * p + k, 1] = 1.0
            
    # W indices for i=1: k*n + i = k*2 + 1.
    # We need W[2k+1, 1] = 1 for k=0..5.
    for k in range(p):
        W[k * n + 1, 1] = 1.0
        
    # Verify
    # Reconstruct T[i,j,k] = sum_t U[i*m+j, t] V[j*p+k, t] W[k*n+i, t]
    T_recon = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for t in range(rank):
                    val += U[i * m + j, t] * V[j * p + k, t] * W[k * n + i, t]
                T_recon[i, j, k] = val
                
    T_expected = np.ones((n, m, p))
    
    if not np.allclose(T_recon, T_expected):
        # Fallback to schoolbook if verification fails (should not happen)
        rank_fb = n * m * p
        U_fb = np.zeros((n * m, rank_fb), dtype=float)
        V_fb = np.zeros((m * p, rank_fb), dtype=float)
        W_fb = np.zeros((p * n, rank_fb), dtype=float)
        
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, column] = 1
                    V_fb[j * p + k, column] = 1
                    W_fb[k * n + i, column] = 1
                    column += 1
        
        return {
            "U": U_fb.tolist(),
            "V": V_fb.tolist(),
            "W": W_fb.tolist()
        }
        
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
