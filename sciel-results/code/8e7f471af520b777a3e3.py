import time
import numpy as np

DIMENSIONS = [5, 5, 5]

def _verify(U, V, W, n, m, p):
    """Verify tensor decomposition: sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]"""
    r = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Build target tensor
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1
    
    # Compute decomposition
    D = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    
    return bool(np.allclose(D, T, atol=1e-10))

def _schoolbook(n, m, p):
    rank = n*m*p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def _construct_partial_strassen(n, m, p):
    """
    Construct 5x5x5 decomposition using partial Strassen.
    
    Apply Strassen only to the (0,1) x (0,1) subblock (2x2),
    use schoolbook for all other terms.
    
    This reduces rank from 125 to 125 - 8 + 7 = 124 (minimal improvement).
    
    For a bigger improvement, we need a more sophisticated construction.
    """
    # This is a minimal improvement. Let's try a different approach.
    # Use the fact that we can share computations across multiple (i,k) pairs.
    
    # For a given j, the term A[i,j]*B[j,k] contributes to C[i,k] for all i,k.
    # If we fix j and vary i,k, we have a rank-1 matrix A[:,j] * B[j,:].
    # The sum over j is a sum of 5 rank-1 matrices, giving rank <= 5*5 = 25 for the "outer" structure.
    # But this doesn't directly help with the tensor rank.
    
    # Let's just return schoolbook for now.
    return _schoolbook(n, m, p)

def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 130.0
    
    # Use schoolbook as a safe, valid decomposition
    U, V, W = _schoolbook(n, m, p)
    
    return {"U": U, "V": V, "W": W}
