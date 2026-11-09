"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

_DIM_START = time.time()
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    _t0 = time.time()
    _DEADLINE = _t0 + 140.0  # safety margin before 150s kill
    n, m, p = DIMENSIONS
    
    # Strategy: Decompose M_{2,3,10} as direct sum of 10 copies of M_{2,3,1}
    # Each M_{2,3,1} has rank 3 (one term per input dimension j)
    # Total rank = p * m = 10 * 3 = 30
    
    total_rank = p * m  # 30
    
    U = [[0] * total_rank for _ in range(n * m)]  # shape (6, 30)
    V = [[0] * total_rank for _ in range(m * p)]  # shape (30, 30)
    W = [[0] * total_rank for _ in range(p * n)]  # shape (20, 30)
    
    col = 0
    for k in range(p):
        for j in range(m):
            # U: for each i, U[i*m + j][col] = 1 if this term corresponds to input j
            # Term (k, j): U_{(i,j), col} = 1 for all i
            for i in range(n):
                U[i * m + j][col] = 1
            
            # V: V[j*p + k][col] = 1 (only for this specific j and k)
            V[j * p + k][col] = 1
            
            # W: W[k*n + i][col] = 1 for all i
            for i in range(n):
                W[k * n + i][col] = 1
            
            col += 1
    
    # Verify the decomposition
    try:
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Check on-entries: T_{(i,j),(j,k),(k,i)} should be 1
        valid = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.einsum('r,r,r->', U_np[i*m+j], V_np[j*p+k], W_np[k*n+i])
                    if abs(s - 1.0) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        # Check off-entries should be 0
        if valid:
            T = np.einsum('ar,br,cr->abc', U_np, V_np, W_np)
            for a in range(n*m):
                for b in range(m*p):
                    for c in range(p*n):
                        # Determine if this is an on-entry
                        i = a // m
                        j = a % m
                        j2 = b // p
                        k = b % p
                        k2 = c // n
                        i2 = c % n
                        if j == j2 and k == k2 and i == i2:
                            if abs(T[a,b,c] - 1.0) > 1e-9:
                                valid = False
                                break
                        else:
                            if abs(T[a,b,c]) > 1e-9:
                                valid = False
                                break
                    if not valid:
                        break
                if not valid:
                    break
        
        if not valid:
            # Fallback to schoolbook
            rank = n * m * p
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j][col] = 1
                        V[j * p + k][col] = 1
                        W[k * n + i][col] = 1
                        col += 1
    except Exception:
        # Fallback to schoolbook on any error
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
