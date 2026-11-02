"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]

import time
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _verify_fast(U, V, W, n, m, p, tol=1e-8):
    """Vectorized verification."""
    rank = U.shape[1]
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    return np.all(np.abs(T - 1.0) < tol)

def _round_to_half(x):
    """Round to nearest half-integer."""
    return np.round(x * 2) / 2.0

def _try_structured_decomposition(n, m, p, time_limit=30):
    """
    Try to find a low-rank decomposition using structured approaches.
    Strategy: start from schoolbook, then try to merge terms using 
    algebraic identities and local search.
    """
    start_time = time.time()
    
    # Schoolbook baseline
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_sb
    
    # Try to find a better decomposition using random restarts with 
    # structured initialization
    rng = np.random.RandomState(123)
    
    for target_rank in range(80, 60, -2):
        if time.time() - start_time > time_limit:
            break
        
        for seed in range(30):
            if time.time() - start_time > time_limit:
                break
            
            rng = np.random.RandomState(seed)
            
            # Initialize with sparse random patterns
            U = np.zeros((n*m, target_rank))
            V = np.zeros((m*p, target_rank))
            W = np.zeros((p*n, target_rank))
            
            # Each rank component should be "meaningful" - assign random 
            # but structured patterns
            for r in range(target_rank):
                # Randomly choose which (i,j) indices are active in U
                u_active = rng.choice(n*m, size=rng.randint(2, min(6, n*m)), replace=False)
                v_active = rng.choice(m*p, size=rng.randint(2, min(6, m*p)), replace=False)
                w_active = rng.choice(p*n, size=rng.randint(2, min(6, p*n)), replace=False)
                
                # Assign half-integer values
                for idx in u_active:
                    U[idx, r] = rng.choice([-1.5, -1.0, -0.5, 0.5, 1.0, 1.5])
                for idx in v_active:
                    V[idx, r] = rng.choice([-1.5, -1.0, -0.5, 0.5, 1.0, 1.5])
                for idx in w_active:
                    W[idx, r] = rng.choice([-1.5, -1.0, -0.5, 0.5, 1.0, 1.5])
            
            # Local search: try to fix the decomposition
            for iteration in range(200):
                if time.time() - start_time > time_limit:
                    break
                
                # Compute current error
                if not _verify_fast(U, V, W, n, m, p, tol=1e-6):
                    # Find the worst (i,j,k) and try to fix it
                    U_r = U.reshape(n, m, target_rank)
                    V_r = V.reshape(m, p, target_rank)
                    W_r = W.reshape(p, n, target_rank)
                    T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
                    errors = np.abs(T - 1.0)
                    
                    # Find worst index
                    worst_idx = np.unravel_index(np.argmax(errors), errors.shape)
                    wi, wj, wk = worst_idx
                    
                    # Try to adjust one of the rank components
                    # Find which rank component contributes most to this (i,j,k)
                    contribs = U[wi*m+wj, :] * V[wj*p+wk, :] * W[wk*n+wi, :]
                    max_r = np.argmax(np.abs(contribs))
                    
                    # Try flipping or adjusting the value
                    if U[wi*m+wj, max_r] != 0:
                        old_val = U[wi*m+wj, max_r]
                        U[wi*m+wj, max_r] = -old_val
                        if _verify_fast(U, V, W, n, m, p, tol=1e-6):
                            break
                        U[wi*m+wj, max_r] = old_val
                    
                    if V[wj*p+wk, max_r] != 0:
                        old_val = V[wj*p+wk, max_r]
                        V[wj*p+wk, max_r] = -old_val
                        if _verify_fast(U, V, W, n, m, p, tol=1e-6):
                            break
                        V[wj*p+wk, max_r] = old_val
                    
                    if W[wk*n+wi, max_r] != 0:
                        old_val = W[wk*n+wi, max_r]
                        W[wk*n+wi, max_r] = -old_val
                        if _verify_fast(U, V, W, n, m, p, tol=1e-6):
                            break
                        W[wk*n+wi, max_r] = old_val
                else:
                    break
            
            if _verify_fast(U, V, W, n, m, p, tol=1e-6):
                # Round to half-integers and verify
                U_r = _round_to_half(U)
                V_r = _round_to_half(V)
                W_r = _round_to_half(W)
                if _verify_fast(U_r, V_r, W_r, n, m, p, tol=1e-6):
                    best_U, best_V, best_W = U_r, V_r, W_r
                    best_rank = target_rank
                    break
        
        if best_rank < rank_sb:
            break
    
    return best_U, best_V, best_W, best_rank

def _try_block_decomposition(n, m, p, time_limit=30):
    """
    Try a block-based decomposition. Split the matrices into blocks 
    and use known low-rank decompositions for each block.
    """
    start_time = time.time()
    
    # For 4x5 * 5x5, we can think of it as:
    # A is 4x5, B is 5x5, C is 4x5
    # Split A into 2x2 blocks of size 2x2 (with some overlap/padding)
    # Actually, let's try a different approach: use the fact that 
    # 4 = 2+2, 5 = 2+2+1
    
    # Try: decompose using 2x2 Strassen-like blocks
    # This is complex, so let's try a simpler structured approach
    
    # Use a "fan-out" decomposition where we share computations
    rank = 72  # Try a specific target
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Create a structured pattern based on modular arithmetic
    for r in range(rank):
        # Each rank component handles a "diagonal" or "block" of the tensor
        block_size = 5
        for i in range(n):
            for j in range(m):
                # U contribution
                if (i + j + r) % block_size == 0:
                    U[i*m+j, r] = 1.0
                elif (i + 2*j + r) % block_size == 0:
                    U[i*m+j, r] = 0.5
        
        for j in range(m):
            for k in range(p):
                if (j + k + r) % block_size == 0:
                    V[j*p+k, r] = 1.0
                elif (2*j + k + r) % block_size == 0:
                    V[j*p+k, r] = 0.5
        
        for k in range(p):
            for i in range(n):
                if (k + i + r) % block_size == 0:
                    W[k*n+i, r] = 1.0
                elif (k + 2*i + r) % block_size == 0:
                    W[k*n+i, r] = 0.5
    
    if _verify_fast(U, V, W, n, m, p, tol=1e-6):
        return U, V, W, rank
    
    return None, None, None, rank

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 145  # Safety margin
    
    # Schoolbook fallback
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n*m, rank_schoolbook))
    V_sb = np.zeros((m*p, rank_schoolbook))
    W_sb = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_schoolbook
    
    # Try structured decomposition
    if time.time() < deadline - 10:
        U_t, V_t, W_t, rank_t = _try_structured_decomposition(n, m, p, time_limit=30)
        if rank_t < best_rank and _verify_fast(U_t, V_t, W_t, n, m, p, tol=1e-6):
            best_U, best_V, best_W = U_t, V_t, W_t
            best_rank = rank_t
    
    # Try block decomposition
    if time.time() < deadline - 5:
        U_b, V_b, W_b, rank_b = _try_block_decomposition(n, m, p, time_limit=15)
        if U_b is not None and rank_b < best_rank and _verify_fast(U_b, V_b, W_b, n, m, p, tol=1e-6):
            best_U, best_V, best_W = U_b, V_b, W_b
            best_rank = rank_b
    
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
