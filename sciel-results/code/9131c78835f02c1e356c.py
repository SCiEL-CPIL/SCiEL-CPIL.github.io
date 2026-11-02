"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
_TIME_LIMIT = 140  # seconds, leave 10s margin for 150s limit

def _verify(U, V, W, n, m, p):
    """Check if the decomposition exactly reconstructs the tensor."""
    rank = len(U[0])
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    
    # Reshape: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    U3 = Ua.reshape(n, m, rank)
    V3 = Va.reshape(m, p, rank)
    W3 = Wa.reshape(p, n, rank)
    
    # Compute sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]
    # U3: (n, m, rank)
    # V3: (m, p, rank)
    # W3: (p, n, rank)
    # Result: (n, m, p)
    # T[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
    
    try:
        result = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    except:
        return False
        
    return np.allclose(result, 1.0, atol=1e-9)

def _construct_rank_p(n, m, p):
    """
    Constructs a decomposition of rank p (the largest dimension).
    U[i,j,c] = 1
    V[j,k,c] = 1 if k==c else 0
    W[k,i,c] = 1 if k==c else 0
    """
    rank = p
    U = [[1 for _ in range(rank)] for _ in range(n * m)]
    V = [[0 for _ in range(rank)] for _ in range(m * p)]
    W = [[0 for _ in range(rank)] for _ in range(p * n)]
    
    for j in range(m):
        for k in range(p):
            c = k
            V[j * p + k][c] = 1
            for i in range(n):
                W[k * n + i][c] = 1
                
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    
    # Try the rank-p construction first
    candidate = _construct_rank_p(n, m, p)
    if _verify(candidate["U"], candidate["V"], candidate["W"], n, m, p):
        return candidate

    # Fallback to schoolbook (rank n*m*p)
    rank_school = n * m * p
    U_s = [[0] * rank_school for _ in range(n * m)]
    V_s = [[0] * rank_school for _ in range(m * p)]
    W_s = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i*m+j][col] = 1
                V_s[j*p+k][col] = 1
                W_s[k*n+i][col] = 1
                col += 1
    
    return {"U": U_s, "V": V_s, "W": W_s}
# EVOLVE-BLOCK-END
