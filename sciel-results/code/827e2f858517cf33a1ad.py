import numpy as np

def construct():
    n, m, p = 4, 4, 7
    # Standard multiplication rank is n*m*p if we treat each scalar product independently?
    # No, standard rank for n x m x p is n*m*p? 
    # C[i,k] = sum_j A[i,j] B[j,k].
    # Each C[i,k] requires m multiplications.
    # Total multiplications = n * p * m.
    # Here n=4, p=7, m=4. Rank = 4 * 7 * 4 = 112.
    
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    r = 0
    for k in range(p):
        for i in range(n):
            for j in range(m):
                # A index: i*m + j
                # B index: j*p + k
                # W index: k*n + i (based on prompt "W[k*n+i,r]")
                
                U[i * m + j, r] = 1
                V[j * p + k, r] = 1
                W[k * n + i, r] = 1
                r += 1
                
    return {'U': U, 'V': V, 'W': W}
