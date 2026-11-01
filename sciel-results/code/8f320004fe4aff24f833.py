import time
import numpy as np

DIMENSIONS = [2, 2, 10]
_DEADLINE = time.time() + 140

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-35 exact tensor decomposition for 2x2 by 2x10 matrix multiplication.
    Uses Strassen's algorithm (rank 7) for each of the 5 independent 2x2 blocks.
    """
    n, m, p = DIMENSIONS
    
    # Construct rank-35 decomposition
    U, V, W = _construct_rank35(n, m, p)
    
    # Verify correctness
    if _verify_decomposition(n, m, p, U, V, W):
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook if verification fails (should not happen)
    U40, V40, W40 = _construct_schoolbook(n, m, p)
    return {"U": U40, "V": V40, "W": W40}


def _construct_rank35(n, m, p):
    """
    Constructs the rank-35 decomposition using Strassen's algorithm for 5 blocks.
    U: (n*m, 35)
    V: (m*p, 35)
    W: (p*n, 35)
    """
    rank = 35
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen coefficients for 2x2 matmul
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # Products:
    # P1 = (a+d)(e+h)
    # P2 = (c+d)e
    # P3 = a(f-h)
    # P4 = d(g-e)
    # P5 = (a+b)h
    # P6 = (c-a)(e+f)
    # P7 = (b-d)(g+h)
    
    # U coeffs for [a, b, c, d]
    strassen_A = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    
    # V coeffs for [e, f, g, h]
    strassen_B = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    
    # W coeffs for [C00, C01, C10, C11]
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    w_contrib = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    for b in range(5):
        for r in range(7):
            col = b * 7 + r
            
            # Fill U
            U[0, col] = strassen_A[r][0]
            U[1, col] = strassen_A[r][1]
            U[2, col] = strassen_A[r][2]
            U[3, col] = strassen_A[r][3]
            
            # Fill V
            # B indices: e=B[0,2b], f=B[0,2b+1], g=B[1,2b], h=B[1,2b+1]
            # V index: k*p + l
            V[0 * p + 2 * b, col] = strassen_B[r][0]
            V[0 * p + 2 * b + 1, col] = strassen_B[r][1]
            V[1 * p + 2 * b, col] = strassen_B[r][2]
            V[1 * p + 2 * b + 1, col] = strassen_B[r][3]
            
            # Fill W
            # C indices: C00, C01, C10, C11
            # W index: l*n + i
            # C00: l=2b, i=0 -> 2b*2 + 0
            # C01: l=2b+1, i=0 -> (2b+1)*2 + 0
            # C10: l=2b, i=1 -> 2b*2 + 1
            # C11: l=2b+1, i=1 -> (2b+1)*2 + 1
            W[2 * b * n + 0, col] = w_contrib[r][0]
            W[(2 * b + 1) * n + 0, col] = w_contrib[r][1]
            W[2 * b * n + 1, col] = w_contrib[r][2]
            W[(2 * b + 1) * n + 1, col] = w_contrib[r][3]
            
    return U, V, W

def _construct_schoolbook(n, m, p):
    """Constructs the rank-40 schoolbook decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if j == k:
                    U[i * m + j, idx] = 1
                    V[j * p + k, idx] = 1
                    W[k * n + i, idx] = 1
                    idx += 1
    return U, V, W

def _verify_decomposition(n, m, p, U, V, W):
    """
    Verifies the decomposition by checking the tensor identity.
    T[i,j,k,l] = sum_r U[i*m+j, r] * V[k*p+l, r] * W[l*n+i, r]
    Should equal delta_{j,k}.
    """
    rank = U.shape[1]
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    
    # Check a few random points or all points?
    # n=2, m=2, p=10. Total points = 2*2*2*10 = 80.
    # 80 * 35 operations is trivial.
    for i in range(n):
        for j in range(m):
            for k in range(m):
                for l in range(p):
                    expected = 1.0 if j == k else 0.0
                    computed = 0.0
                    for r in range(rank):
                        computed += U[i * m + j, r] * V[k * p + l, r] * W[l * n + i, r]
                    if not np.isclose(computed, expected, atol=1e-10):
                        return False
    return True
# EVOLVE-BLOCK-END
