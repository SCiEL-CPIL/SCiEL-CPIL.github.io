"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import combinations

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Schoolbook initialization
    rank0 = n * m * p
    
    U0 = np.zeros((nm, rank0))
    V0 = np.zeros((mp, rank0))
    W0 = np.zeros((pn, rank0))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i*m + j, col] = 1
                V0[j*p + k, col] = 1
                W0[k*n + i, col] = 1
                col += 1
    
    def verify(U, V, W):
        """Check if decomposition is exact. Returns True if valid."""
        r = U.shape[1]
        # Check all (i,j,k) triples
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def verify_fast(U, V, W):
        """Faster verification using vectorized operations."""
        r = U.shape[1]
        # For each (i,j,k), check sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def to_list(U, V, W):
        return {
            "U": [[float(x) for x in row] for row in U],
            "V": [[float(x) for x in row] for row in V],
            "W": [[float(x) for x in row] for row in W]
        }
    
    # Verify schoolbook
    if verify_fast(U0, V0, W0):
        best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()
        best_rank = rank0
    else:
        best_U, best_V, best_W = U0, V0, W0
        best_rank = rank0
    
    # Local search: try removing columns one at a time
    current_U, current_V, current_W = best_U.copy(), best_V.copy(), best_W.copy()
    current_rank = best_rank
    
    # Try removing single columns
    for attempt in range(current_rank):
        if time.time() > deadline - 10:
            break
        
        removed = False
        for c in range(current_rank):
            if time.time() > deadline - 10:
                break
            
            # Try removing column c
            U_test = np.delete(current_U, c, axis=1)
            V_test = np.delete(current_V, c, axis=1)
            W_test = np.delete(current_W, c, axis=1)
            
            if verify_fast(U_test, V_test, W_test):
                current_U, current_V, current_W = U_test, V_test, W_test
                current_rank -= 1
                removed = True
                break
        
        if not removed:
            break
    
    if current_rank < best_rank:
        best_U, best_V, best_W = current_U, current_V, current_W
        best_rank = current_rank
    
    # Try removing pairs of columns
    if time.time() < deadline - 20:
        for c1 in range(best_rank):
            if time.time() > deadline - 20:
                break
            for c2 in range(c1 + 1, best_rank):
                if time.time() > deadline - 20:
                    break
                
                cols = sorted([c1, c2], reverse=True)
                U_test = np.delete(best_U, cols, axis=1)
                V_test = np.delete(best_V, cols, axis=1)
                W_test = np.delete(best_W, cols, axis=1)
                
                if verify_fast(U_test, V_test, W_test):
                    best_U, best_V, best_W = U_test, V_test, W_test
                    best_rank -= 2
                    break
            if best_rank < current_rank:
                break
    
    return to_list(best_U, best_V, best_W)
# EVOLVE-BLOCK-END
