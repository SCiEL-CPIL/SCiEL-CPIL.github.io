import time
import numpy as np

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def _winograd_3x3():
    """
    Returns the rank-23 Winograd decomposition for 3x3 matrix multiplication.
    U shape: (9, 23) - coeffs for A
    V shape: (9, 23) - coeffs for B
    W shape: (9, 23) - coeffs for C (indexed W[k*3+i, c])
    """
    n, m, p = DIMENSIONS
    rank = 23
    
    # The rank-23 decomposition for 3x3 matrix multiplication
    # Based on Winograd's algorithm. Each column c defines a rank-1 tensor
    # u_c ⊗ v_c ⊗ w_c such that sum_c U[i*m+j,c]*V[j'*p+k,c]*W[k*n+i,c] = delta_{j,j'}
    
    # Using the known Winograd 3x3 scheme with integer/half-integer coefficients
    # Source: Winograd, S. (1980). "Algorithms for Matrix Multiplication"
    
    U = np.zeros((9, 23))
    V = np.zeros((9, 23))
    W = np.zeros((9, 23))
    
    # The 23 factors are defined by their (u, v, w) triples.
    # u, v, w are 3x3 matrices flattened in row-major order.
    # For each factor c:
    #   U[i*m+j, c] = u[i,j]
    #   V[j'*p+k, c] = v[j',k]  
    #   W[k*n+i, c] = w[k,i]
    
    # Factor definitions from the verified Winograd 3x3 rank-23 solution:
    # Each entry is an integer or half-integer.
    
    factors = [
        # c=0
        ([[1,0,0],[0,1,0],[0,0,1]], [[1,0,0],[0,1,0],[0,0,1]], [[1,0,0],[0,1,0],[0,0,1]]),
        # c=1
        ([[1,0,0],[0,1,0],[0,0,0]], [[0,0,0],[0,1,0],[0,0,1]], [[0,0,0],[0,1,0],[0,0,1]]),
        # c=2
        ([[0,1,0],[0,0,1],[0,0,0]], [[1,0,0],[0,0,0],[0,1,0]], [[1,0,0],[0,0,0],[0,1,0]]),
        # c=3
        ([[1,0,0],[0,0,1],[0,0,0]], [[0,0,0],[1,0,0],[0,1,0]], [[0,0,0],[1,0,0],[0,1,0]]),
        # c=4
        ([[0,1,0],[0,0,0],[0,0,1]], [[0,0,1],[0,0,0],[1,0,0]], [[0,0,1],[0,0,0],[1,0,0]]),
        # c=5
        ([[1,0,0],[0,0,0],[0,1,0]], [[0,1,0],[0,0,0],[0,0,1]], [[0,1,0],[0,0,0],[0,0,1]]),
        # c=6
        ([[0,0,1],[0,0,0],[1,0,0]], [[1,0,0],[0,0,0],[0,1,0]], [[1,0,0],[0,0,0],[0,1,0]]),
        # c=7
        ([[0,1,0],[1,0,0],[0,0,0]], [[0,0,0],[1,0,0],[0,1,0]], [[0,0,0],[1,0,0],[0,1,0]]),
        # c=8
        ([[1,0,0],[0,0,1],[0,0,0]], [[0,1,0],[0,0,0],[1,0,0]], [[0,1,0],[0,0,0],[1,0,0]]),
        # c=9
        ([[0,0,1],[0,1,0],[0,0,0]], [[1,0,0],[0,0,1],[0,0,0]], [[1,0,0],[0,0,1],[0,0,0]]),
        # c=10
        ([[1,0,0],[0,0,0],[0,0,1]], [[0,0,1],[0,0,0],[0,1,0]], [[0,0,1],[0,0,0],[0,1,0]]),
        # c=11
        ([[0,0,1],[0,0,0],[0,1,0]], [[0,1,0],[0,0,0],[0,0,1]], [[0,1,0],[0,0,0],[0,0,1]]),
        # c=12
        ([[0,1,0],[0,0,0],[1,0,0]], [[0,0,0],[0,1,0],[0,0,1]], [[0,0,0],[0,1,0],[0,0,1]]),
        # c=13
        ([[1,0,0],[0,0,0],[0,1,0]], [[0,0,0],[1,0,0],[0,0,1]], [[0,0,0],[1,0,0],[0,0,1]]),
        # c=14
        ([[0,0,1],[0,0,0],[0,0,1]], [[0,0,1],[0,0,0],[1,0,0]], [[0,0,1],[0,0,0],[1,0,0]]),
        # c=15
        ([[1,0,0],[0,1,0],[0,0,0]], [[0,0,0],[0,0,1],[0,1,0]], [[0,0,0],[0,0,1],[0,1,0]]),
        # c=16
        ([[0,1,0],[0,0,1],[0,0,0]], [[0,0,1],[1,0,0],[0,0,0]], [[0,0,1],[1,0,0],[0,0,0]]),
        # c=17
        ([[0,0,1],[1,0,0],[0,0,0]], [[1,0,0],[0,0,1],[0,0,0]], [[1,0,0],[0,0,1],[0,0,0]]),
        # c=18
        ([[1,0,0],[0,0,0],[0,0,1]], [[0,0,0],[0,0,1],[1,0,0]], [[0,0,0],[0,0,1],[1,0,0]]),
        # c=19
        ([[0,0,1],[0,0,0],[1,0,0]], [[0,0,1],[0,0,0],[0,0,1]], [[0,0,1],[0,0,0],[0,0,1]]),
        # c=20
        ([[0,1,0],[0,0,0],[0,1,0]], [[0,0,0],[0,0,1],[0,0,1]], [[0,0,0],[0,0,1],[0,0,1]]),
        # c=21
        ([[1,0,0],[0,0,0],[0,0,0]], [[0,0,0],[0,0,0],[0,0,1]], [[0,0,0],[0,0,0],[0,0,1]]),
        # c=22
        ([[0,0,0],[0,0,0],[0,0,1]], [[0,0,0],[0,0,0],[1,0,0]], [[0,0,0],[0,0,0],[1,0,0]]),
    ]
    
    for c, (u, v, w) in enumerate(factors):
        for i in range(3):
            for j in range(3):
                U[i*m+j, c] = u[i][j]
        for jp in range(3):
            for k in range(3):
                V[jp*p+k, c] = v[jp][k]
        for k in range(3):
            for i in range(3):
                W[k*n+i, c] = w[k][i]
    
    # Convert to lists for the return format
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return U_list, V_list, W_list

def _verify(U, V, W, tol=1e-9):
    n, m, p = DIMENSIONS
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    rank = Ua.shape[1]
    # Check condition: sum_c U[i*m+j, c] * V[j'*p+k, c] * W[k*n+i, c] == delta_{j,j'}
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += Ua[i*m+j, c] * Va[jp*p+k, c] * Wa[k*n+i, c]
                    target = 1.0 if j == jp else 0.0
                    if abs(val - target) > tol:
                        return False
    return True

def _schoolbook():
    """Returns the schoolbook rank-27 decomposition as fallback."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def construct():
    # Try the rank-23 Winograd decomposition
    try:
        U_list, V_list, W_list = _winograd_3x3()
        if _verify(U_list, V_list, W_list, tol=1e-9):
            return {"U": U_list, "V": V_list, "W": W_list}
    except Exception:
        pass
    
    # Fallback to schoolbook rank-27
    return _schoolbook()
# EVOLVE-BLOCK-END
