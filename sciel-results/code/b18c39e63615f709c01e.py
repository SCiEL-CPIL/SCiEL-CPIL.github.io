"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # Schoolbook fallback
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j][col] = 1
                V_sb[j*p+k][col] = 1
                W_sb[k*n+i][col] = 1
                col += 1
    best = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    # Build tensor
    T = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[(a, b, c)] = 1
    
    # Try to find lower rank using random search with rounding
    for rank in range(24, rank_sb):
        if time.time() > deadline - 10:
            break
        found = False
        for attempt in range(50):
            if time.time() > deadline - 10:
                break
            # Random initialization with half-integers
            U = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(n*m, rank))
            V = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(m*p, rank))
            W = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(p*n, rank))
            
            # Check
            T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
            ok = True
            for (a, b, c) in T:
                if abs(T_decomp[a,b,c] - 1) > 1e-9:
                    ok = False
                    break
            if ok:
                mask = np.ones((n*m, m*p, p*n), dtype=bool)
                for (a, b, c) in T:
                    mask[a,b,c] = False
                if np.any(np.abs(T_decomp[mask]) > 1e-9):
                    ok = False
            
            if ok:
                best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                found = True
                break
        
        if found:
            break
    
    return best
# EVOLVE-BLOCK-END
