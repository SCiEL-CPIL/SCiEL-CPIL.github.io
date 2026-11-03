import numpy as np
import time
import itertools
import random

DIMENSIONS = [2, 3, 6]

def _schoolbook():
    n, m, p = DIMENSIONS
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

def _verify_decomposition(U, V, W, n, m, p):
    rank = len(U[0])
    T_reconstructed = np.zeros((n*m, m*p, p*n))
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for r in range(rank):
        T_reconstructed += np.einsum('a,b,c->abc', U_arr[:,r], V_arr[:,r], W_arr[:,r])
        
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                if abs(T_reconstructed[a,b,c] - 1) > 1e-9:
                    return False
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                i_a = a // m
                j_a = a % m
                j_b = b // p
                k_b = b % p
                k_c = c // n
                i_c = c % n
                
                is_valid = (j_a == j_b) and (k_b == k_c) and (i_a == i_c)
                if not is_valid:
                    if abs(T_reconstructed[a,b,c]) > 1e-9:
                        return False
    return True

def _build_2x3x6_from_2x3x3(U_233, V_233, W_233):
    """
    Constructs 2x3x6 decomposition from 2x3x3 decomposition.
    n=2, m=3, p=6.
    The 2x3x3 tensor has n=2, m=3, p=3.
    U_233 shape: (6, rank)
    V_233 shape: (9, rank)
    W_233 shape: (6, rank)
    
    For 2x3x6:
    U shape: (6, 2*rank)
    V shape: (18, 2*rank)
    W shape: (12, 2*rank)
    
    Block 1: k=0,1,2.
    U[:, 0:rank] = U_233
    V[0:9, 0:rank] = V_233
    W[0:6, 0:rank] = W_233
    
    Block 2: k=3,4,5.
    U[:, rank:2*rank] = U_233
    V[9:18, rank:2*rank] = V_233
    W[6:12, rank:2*rank] = W_233
    """
    rank_233 = U_233[0].__len__()
    total_rank = 2 * rank_233
    
    U = [[0] * total_rank for _ in range(6)]
    V = [[0] * total_rank for _ in range(18)]
    W = [[0] * total_rank for _ in range(12)]
    
    # Copy block 1
    for i in range(6):
        U[i][:rank_233] = U_233[i]
    for i in range(9):
        V[i][:rank_233] = V_233[i]
    for i in range(6):
        W[i][:rank_233] = W_233[i]
        
    # Copy block 2
    for i in range(6):
        U[i][rank_233:] = U_233[i]
    for i in range(9):
        V[i+9][rank_233:] = V_233[i]
    for i in range(6):
        W[i+6][rank_233:] = W_233[i]
        
    return {"U": U, "V": V, "W": W}

def _search_2x3x3_rank(target_rank, time_limit=10):
    """
    Attempts to find a rank decomposition for 2x3x3.
    Returns U, V, W or None.
    """
    n, m, p = 2, 3, 3
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    
    start_time = time.time()
    
    # Target tensor
    T = np.zeros((dim_U, dim_V, dim_W))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
                
    # Random search with local improvement
    # We will try to find a solution with small integer coefficients.
    # Initialize with random sparse matrices.
    
    best_err = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    num_restarts = 100
    steps_per_restart = 1000
    
    for r in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
            
        U = np.random.randint(-1, 2, size=(dim_U, target_rank))
        V = np.random.randint(-1, 2, size=(dim_V, target_rank))
        W = np.random.randint(-1, 2, size=(dim_W, target_rank))
        
        # Calculate error
        err = 0.0
        T_rec = np.zeros((dim_U, dim_V, dim_W))
        for col in range(target_rank):
            T_rec += np.einsum('a,b,c->abc', U[:,col], V[:,col], W[:,col])
        err = np.sum((T_rec - T)**2)
        
        if err < 1e-9:
            return U.tolist(), V.tolist(), W.tolist()
            
        if err < best_err:
            best_err = err
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
            
        # Local search: flip random bits
        for step in range(steps_per_restart):
            if time.time() - start_time > time_limit:
                break
                
            # Flip a random entry in U, V, or W
            mat = np.random.choice(['U', 'V', 'W'])
            if mat == 'U':
                idx = np.random.randint(0, dim_U)
                col = np.random.randint(0, target_rank)
                old_val = U[idx, col]
                U[idx, col] = -old_val if old_val != 0 else (1 if np.random.rand() > 0.5 else -1)
                # Recalculate error efficiently? 
                # For simplicity, recalculate full error. 
                # 6*9*6 = 324 entries. 15 cols. 4860 ops. Fast enough.
                T_rec = np.zeros((dim_U, dim_V, dim_W))
                for c in range(target_rank):
                    T_rec += np.einsum('a,b,c->abc', U[:,c], V[:,c], W[:,c])
                new_err = np.sum((T_rec - T)**2)
                
                if new_err < err:
                    err = new_err
                    if err < 1e-9:
                        return U.tolist(), V.tolist(), W.tolist()
                    if err < best_err:
                        best_err = err
                        best_U = U.copy()
                        best_V = V.copy()
                        best_W = W.copy()
                else:
                    U[idx, col] = old_val
                    
            elif mat == 'V':
                idx = np.random.randint(0, dim_V)
                col = np.random.randint(0, target_rank)
                old_val = V[idx, col]
                V[idx, col] = -old_val if old_val != 0 else (1 if np.random.rand() > 0.5 else -1)
                T_rec = np.zeros((dim_U, dim_V, dim_W))
                for c in range(target_rank):
                    T_rec += np.einsum('a,b,c->abc', U[:,c], V[:,c], W[:,c])
                new_err = np.sum((T_rec - T)**2)
                
                if new_err < err:
                    err = new_err
                    if err < 1e-9:
                        return U.tolist(), V.tolist(), W.tolist()
                    if err < best_err:
                        best_err = err
                        best_U = U.copy()
                        best_V = V.copy()
                        best_W = W.copy()
                else:
                    V[idx, col] = old_val
            else:
                idx = np.random.randint(0, dim_W)
                col = np.random.randint(0, target_rank)
                old_val = W[idx, col]
                W[idx, col] = -old_val if old_val != 0 else (1 if np.random.rand() > 0.5 else -1)
                T_rec = np.zeros((dim_U, dim_V, dim_W))
                for c in range(target_rank):
                    T_rec += np.einsum('a,b,c->abc', U[:,c], V[:,c], W[:,c])
                new_err = np.sum((T_rec - T)**2)
                
                if new_err < err:
                    err = new_err
                    if err < 1e-9:
                        return U.tolist(), V.tolist(), W.tolist()
                    if err < best_err:
                        best_err = err
                        best_U = U.copy()
                        best_V = V.copy()
                        best_W = W.copy()
                else:
                    W[idx, col] = old_val

    return None

def construct():
    # Try to find better rank
    # Target rank for 2x3x3 is 15.
    # If found, total rank is 30.
    
    result_233 = _search_2x3x3_rank(15, time_limit=10)
    if result_233:
        U, V, W = result_233
        # Verify
        if _verify_decomposition(U, V, W, 2, 3, 3):
            res_236 = _build_2x3x6_from_2x3x3(U, V, W)
            # Verify 2x3x6
            if _verify_decomposition(res_236['U'], res_236['V'], res_236['W'], 2, 3, 6):
                return res_236
            else:
                # Should not happen
                pass
    else:
        # Try rank 16?
        result_233 = _search_2x3x3_rank(16, time_limit=5)
        if result_233:
            U, V, W = result_233
            if _verify_decomposition(U, V, W, 2, 3, 3):
                res_236 = _build_2x3x6_from_2x3x3(U, V, W)
                if _verify_decomposition(res_236['U'], res_236['V'], res_236['W'], 2, 3, 6):
                    return res_236

    return _schoolbook()
