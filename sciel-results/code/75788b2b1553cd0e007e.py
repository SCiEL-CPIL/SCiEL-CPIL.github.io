"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # Build tensor decomposition for M_{n,m,p}
    # T[i,k,j] = delta_{i,j_a} delta_{k,j_b} ... 
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Tensor: T_{(i,k), (i,j), (j,k)} = 1 if indices match appropriately
    # Actually T is a 3-tensor with indices (output, U_input, V_input)
    # U has indices (i,j), V has indices (j,k), W has indices (k,i)
    # For each rank-1 term: U[i,j] * V[j,k] * W[k,i] should contribute to C[i,k]
    
    # Schoolbook: for each (i,j,k) with j matching, U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    for r in range(rank):
        # Decode r into (i, j, k)
        i = r // (m * p)
        j = (r // p) % m
        k = r % p
        U[i * m + j, r] = 1
        V[j * p + k, r] = 1
        W[k * n + i, r] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
