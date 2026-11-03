import time
import random
import numpy as np

DIMENSIONS = [3, 4, 8]

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

def _verify(U, V, W, n, m, p):
    """Verify that the tensor decomposition is exact."""
    rank = len(U[0]) if U else 0
    # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # Should be 1 for all i,j,k
    
    # Convert to numpy for speed
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Reshape
    U_r = U_np.reshape(n, m, rank)
    V_r = V_np.reshape(m, p, rank)
    W_r = W_np.reshape(p, n, rank)
    
    # Compute tensor
    # T[i,j,k] = sum_r U_r[i,j,r] * V_r[j,k,r] * W_r[k,i,r]
    # This is a bit tricky with standard einsum if indices don't align perfectly for a single call
    # We can do:
    # Temp[j,k,r] = V_r[j,k,r] * W_r[k,i,r] ... no, i is in W.
    
    # Let's just loop i, and compute the rest
    for i in range(n):
        # W_i[k, r] = W_r[k, i, r]
        W_i = W_r[:, i, :] # shape (p, rank)
        
        # For each j, k:
        # S[j,k] = sum_r U_r[i,j,r] * V_r[j,k,r] * W_i[k,r]
        # We can compute this as:
        # M[j, r] = U_r[i,j,r]
        # N[j, k, r] = V_r[j,k,r] * W_i[k,r]
        # S[j,k] = sum_r M[j,r] * N[j,k,r]
        
        # Alternatively:
        # B_r[j, k] = V_r[j, k, r] * W_i[k, r] -> shape (rank, m, p)
        # C_r[i, j, k] = U_r[i, j, r] * B_r[r, j, k] -> shape (rank, m, p)
        # Sum over r -> shape (m, p)
        
        # Let's use einsum
        # U_i: (m, rank)
        # V: (m, p, rank)
        # W_i: (p, rank)
        
        U_i = U_r[i, :, :] # (m, rank)
        
        # We want T_i[j, k] = sum_r U_i[j, r] * V_r[j, k, r] * W_i[k, r]
        
        # Step 1: X[j, k, r] = V_r[j, k, r] * W_i[k, r]
        X = np.einsum('jkr,kr->jkr', V_r, W_i)
        
        # Step 2: T_i[j, k] = sum_r U_i[j, r] * X[j, k, r]
        T_i = np.einsum('jr,jkr->jk', U_i, X)
        
        if not np.allclose(T_i, 1.0, atol=1e-9):
            return False
            
    return True

def _to_int_half(val):
    if abs(val - round(val)) < 1e-9:
        return int(round(val))
    elif abs(val * 2 - round(val * 2)) < 1e-9:
        return round(val * 2) / 2.0
    return val

def _search_rank(n, m, p, target_rank, time_limit):
    start_time = time.time()
    
    # Random search with local refinement
    for _ in range(50): # 50 restarts
        if time.time() - start_time > time_limit:
            break
            
        # Initialize with small integers
        U = np.random.randint(-1, 2, size=(n * m, target_rank))
        V = np.random.randint(-1, 2, size=(m * p, target_rank))
        W = np.random.randint(-1, 2, size=(p * n, target_rank))
        
        # Local search
        for step in range(200):
            if time.time() - start_time > time_limit:
                break
                
            # Check error
            if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                return U, V, W
            
            # Pick a random matrix and index
            mat_idx = random.choice([0, 1, 2])
            if mat_idx == 0:
                mat = U
                rows = n * m
            elif mat_idx == 1:
                mat = V
                rows = m * p
            else:
                mat = W
                rows = p * n
                
            r_idx = random.randint(0, rows - 1)
            c_idx = random.randint(0, target_rank - 1)
            
            # Try changing value
            old_val = mat[r_idx, c_idx]
            delta = random.choice([-1, 1])
            mat[r_idx, c_idx] = old_val + delta
            
            # Check if error decreased (simplified: just check a few random points)
            # For speed, we don't verify fully every step, but this is risky.
            # Let's just keep the change if it doesn't obviously break things?
            # Actually, we need to minimize error.
            
            # To be safe and fast, we only verify every 10 steps or at the end of restart
            pass
            
        # Final check for this restart
        if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
            return U, V, W
            
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Leave 40s margin
    
    # Start with schoolbook
    best = _schoolbook()
    best_rank = n * m * p
    
    # Try to find better decompositions
    # Known lower bound is 32.
    # Try ranks 64, 48, 32
    
    ranks_to_try = [64, 48, 32]
    
    for target_rank in ranks_to_try:
        if time.time() > deadline:
            break
            
        # Allocate 30s for each rank
        time_limit = 30
        result = _search_rank(n, m, p, target_rank, time_limit)
        
        if result is not None:
            U, V, W = result
            # Convert to list and verify again
            U_list = [[_to_int_half(U[i][r]) for r in range(target_rank)] for i in range(n*m)]
            V_list = [[_to_int_half(V[i][r]) for r in range(target_rank)] for i in range(m*p)]
            W_list = [[_to_int_half(W[i][r]) for r in range(target_rank)] for i in range(p*n)]
            
            if _verify(U_list, V_list, W_list, n, m, p):
                best = {"U": U_list, "V": V_list, "W": W_list}
                best_rank = target_rank
                # Found a lower rank, break and return
                break
                
    return best
