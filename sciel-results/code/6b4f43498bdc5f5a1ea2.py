"""Rank-reduced tensor decomposition for 3x4x8 matrix multiplication."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 8]

def _verify(U, V, W, n=3, m=4, p=8):
    """Verify exact decomposition."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = sum(U[i*m+j, r_idx] * V[j*p+k, r_idx] * W[k*n+i, r_idx] 
                        for r_idx in range(r))
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _verify_fast(U, V, W, n=3, m=4, p=8):
    """Faster vectorized verification."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    r = U.shape[1]
    errors = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                errors += (s - 1.0) ** 2
    return errors < 1e-18

def _schoolbook(n=3, m=4, p=8):
    """Return schoolbook decomposition (rank n*m*p)."""
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

def _try_random_decomposition(n, m, p, rank, seed=42, max_iter=2000):
    """Try to find a valid rank-r decomposition via random initialization + refinement."""
    rng = np.random.RandomState(seed)
    
    # Initialize with random values
    U = rng.randn(n * m, rank)
    V = rng.randn(m * p, rank)
    W = rng.randn(p * n, rank)
    
    best_err = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        # Compute all residuals
        residuals = {}
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    residuals[(i,j,k)] = s - 1.0
        
        total_err = sum(r**2 for r in residuals.values())
        if total_err < best_err:
            best_err = total_err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            if total_err < 1e-18:
                break
        
        if total_err < 1e-18:
            break
        
        # Pick the worst residual
        worst_key = max(residuals, key=lambda k: abs(residuals[k]))
        i, j, k = worst_key
        resid = residuals[worst_key]
        
        v_w = V[j*p+k, :] * W[k*n+i, :]
        v_w_norm_sq = np.dot(v_w, v_w)
        if v_w_norm_sq > 1e-12:
            U[i*m+j, :] -= resid * v_w / v_w_norm_sq
    
    if _verify_fast(best_U, best_V, best_W, n, m, p):
        U_list = [[round(x * 2) / 2 for x in row] for row in best_U]
        V_list = [[round(x * 2) / 2 for x in row] for row in best_V]
        W_list = [[round(x * 2) / 2 for x in row] for row in best_W]
        if _verify(U_list, V_list, W_list, n, m, p):
            return U_list, V_list, W_list
        return best_U.tolist(), best_V.tolist(), best_W.tolist()
    
    return None

def _construct_block_decomposition(n, m, p_block, rank_block, seed=0):
    """
    Construct a rank-rank_block decomposition for n x m x p_block.
    Returns (U, V, W) or None.
    """
    result = _try_random_decomposition(n, m, p_block, rank_block, seed=seed, max_iter=500)
    return result

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 3x4x8 matrix multiplication.
    Returns {'U': (n*m, rank), 'V': (m*p, rank), 'W': (p*n, rank)}
    """
    deadline = time.time() + 115
    n, m, p = 3, 4, 8
    
    # Start with schoolbook
    school = _schoolbook(n, m, p)
    best_result = school
    best_rank = n * m * p
    
    # Strategy: Try to construct a low-rank decomposition for 3x4x4,
    # then extend to 3x4x8 by treating it as two independent 3x4x4 blocks.
    
    # Target rank for 3x4x4 block. 
    # Schoolbook for 3x4x4 is 48.
    # Let's try to find a rank significantly lower than 48 for the block.
    # If we find rank 24 for 3x4x4, total rank for 3x4x8 would be 48.
    # If we find rank 15 for 3x4x4, total rank for 3x4x8 would be 30.
    
    # Let's try to find a decomposition for 3x4x4 with rank 15.
    # This is a big jump, so it might not exist or be hard to find.
    # Let's start with a more conservative target and work down.
    
    # The theoretical lower bound for 3x4x4 is max(12, 16, 12) = 16.
    # So rank 15 is impossible. Rank 16 is the minimum possible.
    # Let's try rank 16 for 3x4x4.
    
    # If we find rank 16 for 3x4x4, we can use it for both blocks of 3x4x8.
    # Total rank = 16 + 16 = 32.
    # This is better than schoolbook 96.
    
    # Let's try to find a rank 16 decomposition for 3x4x4.
    
    p_block = 4
    target_rank_block = 16
    
    # Try multiple seeds to find a valid decomposition
    found_block = None
    for seed in range(50):
        if time.time() > deadline - 30:
            break
        
        result = _construct_block_decomposition(n, m, p_block, target_rank_block, seed=seed)
        if result is not None:
            U_block, V_block, W_block = result
            found_block = (U_block, V_block, W_block)
            break
    
    if found_block is not None:
        U_block, V_block, W_block = found_block
        r_block = len(U_block[0])
        
        # Construct the full 3x4x8 decomposition by concatenating two blocks
        # Block 1: k in [0, 4)
        # Block 2: k in [4, 8)
        
        U_full = [[0] * (2 * r_block) for _ in range(n * m)]
        V_full = [[0] * (2 * r_block) for _ in range(m * p)]
        W_full = [[0] * (2 * r_block) for _ in range(p * n)]
        
        # Fill Block 1
        for i in range(n):
            for j in range(m):
                for c in range(r_block):
                    U_full[i*m+j][c] = U_block[i*m+j][c]
        
        for j in range(m):
            for k in range(p_block):
                for c in range(r_block):
                    V_full[j*p+k][c] = V_block[j*p_block+k][c]
        
        for k in range(p_block):
            for i in range(n):
                for c in range(r_block):
                    W_full[k*n+i][c] = W_block[k*n+i][c]
        
        # Fill Block 2
        offset = r_block
        for i in range(n):
            for j in range(m):
                for c in range(r_block):
                    U_full[i*m+j][offset+c] = U_block[i*m+j][c]
        
        for j in range(m):
            for k in range(p_block):
                for c in range(r_block):
                    V_full[j*p + (k+p_block)][offset+c] = V_block[j*p_block+k][c]
        
        for k in range(p_block):
            for i in range(n):
                for c in range(r_block):
                    W_full[(k+p_block)*n+i][offset+c] = W_block[k*n+i][c]
        
        # Verify the full decomposition
        if _verify(U_full, V_full, W_full, n, m, p):
            best_result = {"U": U_full, "V": V_full, "W": W_full}
            best_rank = 2 * r_block
            return best_result
        else:
            # If verification failed, fall back to schoolbook
            return school
    
    # If we couldn't find a block decomposition, try the original random approach
    for target_rank in range(best_rank - 1, 59, -1):
        if time.time() > deadline - 30:
            break
        
        found = False
        for seed in range(20):
            if time.time() > deadline - 30:
                break
            
            result = _try_random_decomposition(n, m, p, target_rank, seed=seed, max_iter=500)
            if result is not None:
                U_list, V_list, W_list = result
                if _verify(U_list, V_list, W_list, n, m, p):
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = target_rank
                    found = True
                    break
        
        if found:
            continue
        else:
            break
    
    return best_result
