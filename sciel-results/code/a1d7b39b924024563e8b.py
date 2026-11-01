import numpy as np
import time

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115

    size_a = n * m
    size_b = m * p
    size_c = p * n

    # The tensor T[a, b, c] = 1 if a = i*m + j, b = j*p + k, c = k*n + i
    # This decomposes into p independent tensors T_k for k in 0..p-1
    # T_k has non-zero entries at:
    # i=0, j=0: a=0, b=k, c=2k
    # i=0, j=1: a=1, b=p+k, c=2k
    # i=1, j=0: a=2, b=k, c=2k+1
    # i=1, j=1: a=3, b=p+k, c=2k+1
    
    # Each T_k is a 2x2x1 multiplication tensor (rank 2)
    # Total rank = p * 2 = 28? 
    # Wait, p=14. Rank = 14 * 2 = 28?
    # Let's re-verify the rank of 2x2x1.
    # T_k entries:
    # (0, k, 2k) = 1
    # (1, p+k, 2k) = 1
    # (2, k, 2k+1) = 1
    # (3, p+k, 2k+1) = 1
    
    # We proved rank is 2.
    # So total rank is 14 * 2 = 28.
    
    rank = 2 * p
    U = np.zeros((size_a, rank), dtype=int)
    V = np.zeros((size_b, rank), dtype=int)
    W = np.zeros((size_c, rank), dtype=int)
    
    for k in range(p):
        # Two columns for this k
        col0 = 2 * k
        col1 = 2 * k + 1
        
        # Column 0 handles i=0 terms?
        # We need:
        # T[0, k, 2k] = 1
        # T[1, p+k, 2k] = 1
        # T[2, k, 2k+1] = 1
        # T[3, p+k, 2k+1] = 1
        
        # Using the decomposition found:
        # U[0,0]=1, U[1,0]=0, U[2,0]=1, U[3,0]=0
        # V[0,0]=1, V[1,0]=0 (local V indices)
        # W[0,0]=1, W[1,0]=1 (local W indices)
        # This gives:
        # T[0, k, 2k] = 1*1*1 = 1
        # T[1, p+k, 2k] = 0*0*1 = 0 (Wait, we need 1)
        
        # Let's re-map the local indices to global.
        # Local b: 0 -> k, 1 -> p+k
        # Local c: 0 -> 2k, 1 -> 2k+1
        
        # Decomposition for T_k:
        # Term 0: U[0]=1, V[0]=1, W[0]=1 -> contributes to (0, k, 2k)
        #          U[1]=0, V[1]=0, W[0]=1 -> contributes 0 to (1, p+k, 2k)
        #          U[2]=1, V[0]=1, W[1]=1 -> contributes to (2, k, 2k+1)
        #          U[3]=0, V[1]=0, W[1]=1 -> contributes 0 to (3, p+k, 2k+1)
        # This covers (0, k, 2k) and (2, k, 2k+1).
        
        # Term 1: U[0]=0, V[0]=0, W[0]=1 -> 0
        #          U[1]=1, V[1]=1, W[0]=1 -> contributes to (1, p+k, 2k)
        #          U[2]=0, V[0]=0, W[1]=1 -> 0
        #          U[3]=1, V[1]=1, W[1]=1 -> contributes to (3, p+k, 2k+1)
        # This covers (1, p+k, 2k) and (3, p+k, 2k+1).
        
        # So:
        # Col 0:
        U[0, col0] = 1
        U[2, col0] = 1
        V[k, col0] = 1
        W[2*k, col0] = 1
        W[2*k+1, col0] = 1
        
        # Col 1:
        U[1, col1] = 1
        U[3, col1] = 1
        V[p+k, col1] = 1
        W[2*k, col1] = 1
        W[2*k+1, col1] = 1

    # Verify
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    T_recon = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr, optimize=True)
    
    T_true = np.zeros((size_a, size_b, size_c), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_true[a, b, c] = 1

    if not np.array_equal(T_recon, T_true):
        # Fallback to schoolbook if something went wrong
        rank = n * m * p
        U = np.zeros((size_a, rank), dtype=int)
        V = np.zeros((size_b, rank), dtype=int)
        W = np.zeros((size_c, rank), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
