"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    if not U or not U[0]:
        return False
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    T_decomp = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for c in range(rank):
        T_decomp += np.einsum('a,b,c->abc', U_arr[:,c], V_arr[:,c], W_arr[:,c])
    
    T_target = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i*m + j, j*p + k, k*n + i] = 1.0
    
    if not np.allclose(T_decomp, T_target, atol=1e-8):
        return False
    return True

def _schoolbook(n, m, p):
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
    return {"U": U, "V": V, "W": W, "rank": rank}

def _build_target_tensor(n, m, p):
    """Build the matrix multiplication tensor."""
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    return T

def _try_search(n, m, p, deadline):
    """
    Search for lower-rank decompositions using random sparse initialization.
    """
    T_target = _build_target_tensor(n, m, p)
    nm = n * m
    mp = m * p
    pn = p * n
    
    best_result = None
    
    # Try ranks from 25 to 44
    for target_rank in range(25, 45):
        if time.time() > deadline:
            break
        
        max_attempts = 50
        found = False
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Initialize with random sparse vectors
            U = np.zeros((nm, target_rank), dtype=np.float64)
            V = np.zeros((mp, target_rank), dtype=np.float64)
            W = np.zeros((pn, target_rank), dtype=np.float64)
            
            for c in range(target_rank):
                # Choose random sparse supports with values from {-1, 0, 1}
                u_mask = np.random.random(nm) < 0.2
                v_mask = np.random.random(mp) < 0.2
                w_mask = np.random.random(pn) < 0.2
                
                U[:, c] = np.where(u_mask, np.random.choice([-1, 1], nm), 0)
                V[:, c] = np.where(v_mask, np.random.choice([-1, 1], mp), 0)
                W[:, c] = np.where(w_mask, np.random.choice([-1, 1], pn), 0)
            
            # Verify using einsum for speed
            T_decomp = np.zeros((nm, mp, pn), dtype=np.float64)
            for c in range(target_rank):
                T_decomp += np.einsum('a,b,c->abc', U[:,c], V[:,c], W[:,c])
            
            if np.allclose(T_decomp, T_target, atol=1e-8):
                U_list = [[float(x) for x in row] for row in U]
                V_list = [[float(x) for x in row] for row in V]
                W_list = [[float(x) for x in row] for row in W]
                result = {"U": U_list, "V": V_list, "W": W_list, "rank": target_rank}
                if best_result is None or target_rank < best_result["rank"]:
                    best_result = result
                found = True
                break
        
        if found:
            break
    
    return best_result

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 135
    
    best_result = _schoolbook(n, m, p)
    
    # Try search for lower rank
    search_result = _try_search(n, m, p, deadline)
    if search_result and search_result["rank"] < best_result["rank"]:
        best_result = search_result
    
    return best_result
# EVOLVE-BLOCK-END
