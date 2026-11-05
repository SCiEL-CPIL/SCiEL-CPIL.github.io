"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 7]

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W form a valid decomposition."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Check tensor equation: for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Sum over rank: U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should be 1
                s = 0.0
                for r in range(rank):
                    s += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    
    return True

def _schoolbook():
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

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    time_limit = 120.0
    end_time = start_time + time_limit
    
    n, m, p = DIMENSIONS  # n=2, m=3, p=7
    
    # Winograd frequency-domain approach
    # Use 5 evaluation points for the m=3 dimension (product is degree 4)
    # Rank = 5 * p = 5 * 7 = 35
    
    rank = 5 * p
    U = [[0] * rank for _ in range(n * m)]  # 6 x 35
    V = [[0] * rank for _ in range(m * p)]  # 21 x 35
    W = [[0] * rank for _ in range(p * n)]  # 14 x 35
    
    # Use 5 evaluation points: -2, -1, 0, 1, 2
    alpha = [-2, -1, 0, 1, 2]
    
    # Precompute Lagrange basis polynomials at x=0
    lagrange_at_zero = []
    for k in range(len(alpha)):
        num = 1.0
        den = 1.0
        for l in range(len(alpha)):
            if l != k:
                num *= (0 - alpha[l])
                den *= (alpha[k] - alpha[l])
        lagrange_at_zero.append(num / den)
    
    for k_idx, a in enumerate(alpha):
        for c in range(p):
            col = k_idx * p + c
            
            # U: U[i*m+j, col] = a^j
            for i in range(n):
                for j in range(m):
                    U[i * m + j][col] = a ** j
            
            # V: V[j*p+c, col] = a^j
            for j in range(m):
                V[j * p + c][col] = a ** j
            
            # W: W[c*n+i, col] = lagrange_at_zero[k_idx]
            for i in range(n):
                W[c * n + i][col] = lagrange_at_zero[k_idx]
    
    # Verify
    if time.time() < end_time - 5.0:
        if not _verify(U, V, W, n, m, p):
            return _schoolbook()
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
