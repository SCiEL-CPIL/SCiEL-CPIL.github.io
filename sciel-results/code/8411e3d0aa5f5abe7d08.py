import numpy as np
import time

DIMENSIONS = [3, 3, 3]

def _schoolbook():
    n, m, p = 3, 3, 3
    rank = 27
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def _rank23():
    # Placeholder for 23 terms
    # If I can't recall, return None
    return None

def _verify(U, V, W):
    n, m, p = 3, 3, 3
    # Construct target tensor
    # T[i, j, k] = 1 if i,j,k form a valid term
    # i in [0,3), j in [0,3), k in [0,3)
    # U index: i*m+j
    # V index: j*p+k
    # W index: k*n+i
    
    # Target tensor T has shape (n*m, m*p, p*n)
    # T[alpha, beta, gamma] = 1 if alpha=i*m+j, beta=j*p+k, gamma=k*n+i
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i*m+j
                beta = j*p+k
                gamma = k*n+i
                T[alpha, beta, gamma] = 1
                
    # Compute sum U_r V_r W_r
    # R = U.shape[1]
    # Sum over r of U[:,r] * V[:,r] * W[:,r]
    # This is a tensor product sum.
    # We can compute T_hat[alpha, beta, gamma] = sum_r U[alpha,r] V[beta,r] W[gamma,r]
    # Using einsum: 'ar,br,cr->abc'
    T_hat = np.einsum('ar,br,cr->abc', U, V, W)
    
    return np.allclose(T, T_hat)

def construct():
    # Try rank 23
    r23 = _rank23()
    if r23 is not None:
        U, V, W = r23
        if _verify(U, V, W):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # Fallback to schoolbook
    U, V, W = _schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
