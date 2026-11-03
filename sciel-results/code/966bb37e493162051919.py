import numpy as np
import time
import random

DIMENSIONS = [3, 3, 8]

def _validate_numpy(U, V, W, n, m, p):
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    rank = U.shape[1]
    
    U_3d = U.reshape(n, m, rank)
    V_3d = V.reshape(m, p, rank)
    W_3d = W.reshape(p, n, rank)
    
    T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    
    if not np.allclose(T, 1.0, atol=1e-9):
        return False
    return True

def _validate(U, V, W, n, m, p):
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if len(U[0]) != len(V[0]) or len(U[0]) != len(W[0]):
        return False
    return _validate_numpy(U, V, W, n, m, p)

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
    return {"U": U, "V": V, "W": W}

def _find_332(rank, time_limit=10.0):
    n, m, p = 3, 3, 2
    start_time = time.time()
    
    # Number of equations: n*m*p = 18
    # Unknowns in W: p*n*rank = 6*rank
    num_eqs = n * m * p
    
    for _ in range(2000):
        if time.time() - start_time > time_limit:
            break
            
        # Random U, V
        # Use small integers to keep numbers manageable
        U = np.array([[random.randint(-2, 2) for _ in range(rank)] for _ in range(n * m)], dtype=np.float64)
        V = np.array([[random.randint(-2, 2) for _ in range(rank)] for _ in range(m * p)], dtype=np.float64)
        
        # Construct A matrix for W
        # A is (num_eqs, p*n*rank)
        A = np.zeros((num_eqs, p * n * rank))
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    eq_idx = i * m * p + j * p + k
                    for r in range(rank):
                        # Coefficient for W[k, i, r]
                        coeff = U[i * m + j, r] * V[j * p + k, r]
                        col_idx = k * n * rank + i * rank + r
                        A[eq_idx, col_idx] = coeff
        
        b = np.ones(num_eqs)
        
        # Solve least squares
        W_sol, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
        
        # Round to nearest half-integer
        W_rounded = np.round(W_sol * 2) / 2
        
        # Check if exact
        if np.allclose(A @ W_rounded, b, atol=1e-9):
            # Convert to list of lists
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W_rounded.reshape(p * n, rank).tolist()
            
            # Final validation
            if _validate(U_list, V_list, W_list, n, m, p):
                return U_list, V_list, W_list
                
    return None

def _construct_from_332(U_332, V_332, W_332):
    n, m, p = 3, 3, 8
    rank_332 = len(U_332[0])
    total_rank = 4 * rank_332
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    for i in range(n * m):
        for r in range(rank_332):
            for block in range(4):
                U[i][block * rank_332 + r] = U_332[i][r]
                
    for j in range(m):
        for k in range(p):
            block = k // 2
            k_local = k % 2
            for r in range(rank_332):
                V[j * p + k][block * rank_332 + r] = V_332[j * 2 + k_local][r]
                
    for k in range(p):
        block = k // 2
        k_local = k % 2
        for i in range(n):
            for r in range(rank_332):
                W[k * n + i][block * rank_332 + r] = W_332[k_local * 3 + i][r]
                
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115
    
    schoolbook = _schoolbook(n, m, p)
    
    # Try to find better 3x3x2 decomposition
    # Start from 17 down to 14
    for r_332 in range(17, 13, -1):
        if time.time() > deadline - 20:
            break
        result_332 = _find_332(r_332, time_limit=15)
        if result_332:
            U, V, W = result_332
            decomp = _construct_from_332(U, V, W)
            if _validate(decomp["U"], decomp["V"], decomp["W"], n, m, p):
                return decomp
                
    # Fallback
    return schoolbook
