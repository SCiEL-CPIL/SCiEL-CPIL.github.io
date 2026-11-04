"""Matrix multiplication tensor decomposition for 3x3 by 3x8."""
DIMENSIONS = [3, 3, 8]

import time
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition correctly represents matrix multiplication."""
    rank = U.shape[1]
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Using einsum for speed
    T_computed = np.einsum('ir,jkr,kr->ijk', 
                            U.reshape(n, m, rank),
                            V.reshape(m, p, rank),
                            W.reshape(p, n, rank))
    T_expected = np.ones((n, m, p))
    return np.allclose(T_computed, T_expected, atol=tol)

def _alternating_minimization(U, V, W, n, m, p, max_iter=300, tol=1e-10):
    """Alternating minimization to refine U, V, W."""
    rank = U.shape[1]
    
    for it in range(max_iter):
        # Fix V, W, solve for U
        U_new = np.zeros_like(U)
        for i in range(n):
            for j in range(m):
                # For fixed i,j: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all k
                # Let C[k, r] = V[j*p+k, r] * W[k*n+i, r]
                # Then C^T @ u = ones(p)
                C = np.zeros((p, rank))
                for k in range(p):
                    C[k, :] = V[j * p + k, :] * W[k * n + i, :]
                try:
                    u, residuals, rnk, sv = np.linalg.lstsq(C.T, np.ones(p), rcond=None)
                    U_new[i * m + j, :] = u
                except np.linalg.LinAlgError:
                    return None, None, None

        # Fix U, W, solve for V
        V_new = np.zeros_like(V)
        for j in range(m):
            for k in range(p):
                # For fixed j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i
                # Let B[i, r] = U[i*m+j, r] * W[k*n+i, r]
                # Then B^T @ v = ones(n)
                B = np.zeros((n, rank))
                for i in range(n):
                    B[i, :] = U_new[i * m + j, :] * W[k * n + i, :]
                try:
                    v, residuals, rnk, sv = np.linalg.lstsq(B.T, np.ones(n), rcond=None)
                    V_new[j * p + k, :] = v
                except np.linalg.LinAlgError:
                    return None, None, None

        # Fix U, V, solve for W
        W_new = np.zeros_like(W)
        for k in range(p):
            for i in range(n):
                # For fixed k,i: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all j
                # Let A[j, r] = U[i*m+j, r] * V[j*p+k, r]
                # Then A^T @ w = ones(m)
                A = np.zeros((m, rank))
                for j in range(m):
                    A[j, :] = U_new[i * m + j, :] * V_new[j * p + k, :]
                try:
                    w, residuals, rnk, sv = np.linalg.lstsq(A.T, np.ones(m), rcond=None)
                    W_new[k * n + i, :] = w
                except np.linalg.LinAlgError:
                    return None, None, None

        U, V, W = U_new, V_new, W_new
        
        # Check convergence
        if _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
            return U, V, W
            
    if _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
        return U, V, W
    return None, None, None

def _block_init(n, m, p, rank, seed=42):
    """Initialize U, V, W with block structure for 3+3+2 split of p=8."""
    rng = np.random.RandomState(seed)
    
    # Split p=8 into p1=3, p2=3, p3=2
    p1, p2, p3 = 3, 3, 2
    
    # Distribute rank among blocks
    # Try different distributions
    if rank >= 64:
        r1, r2, r3 = 23, 23, 18
    elif rank >= 63:
        r1, r2, r3 = 23, 22, 18
    elif rank >= 62:
        r1, r2, r3 = 22, 22, 18
    elif rank >= 61:
        r1, r2, r3 = 22, 21, 18
    elif rank >= 60:
        r1, r2, r3 = 20, 20, 20
    else:
        r1 = rank // 3
        r2 = rank // 3
        r3 = rank - r1 - r2
    
    # Ensure ranks sum to target
    while r1 + r2 + r3 > rank:
        if r1 > 1:
            r1 -= 1
        elif r2 > 1:
            r2 -= 1
        else:
            r3 -= 1
    while r1 + r2 + r3 < rank:
        r3 += 1
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Block 1: k in [0, 3)
    U[:, 0:r1] = rng.randn(n * m, r1) * 0.5
    for j in range(m):
        V[j * p + 0 : j * p + 3, 0:r1] = rng.randn(3, r1) * 0.5
    for k in range(3):
        W[k * n : (k+1) * n, 0:r1] = rng.randn(n, r1) * 0.5
        
    # Block 2: k in [3, 6)
    offset2 = r1
    U[:, offset2:offset2+r2] = rng.randn(n * m, r2) * 0.5
    for j in range(m):
        V[j * p + 3 : j * p + 6, offset2:offset2+r2] = rng.randn(3, r2) * 0.5
    for k in range(3, 6):
        W[k * n : (k+1) * n, offset2:offset2+r2] = rng.randn(n, r2) * 0.5
        
    # Block 3: k in [6, 8)
    offset3 = r1 + r2
    U[:, offset3:rank] = rng.randn(n * m, r3) * 0.5
    for j in range(m):
        V[j * p + 6 : j * p + 8, offset3:rank] = rng.randn(2, r3) * 0.5
    for k in range(6, 8):
        W[k * n : (k+1) * n, offset3:rank] = rng.randn(n, r3) * 0.5
        
    return U, V, W

def _random_init(n, m, p, rank, seed=42):
    """Random initialization."""
    rng = np.random.RandomState(seed)
    U = rng.randn(n * m, rank) * 0.5
    V = rng.randn(m * p, rank) * 0.5
    W = rng.randn(p * n, rank) * 0.5
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Start with schoolbook as fallback
    rank_school = n * m * p  # 72
    U_school = np.zeros((n * m, rank_school), dtype=float)
    V_school = np.zeros((m * p, rank_school), dtype=float)
    W_school = np.zeros((p * n, rank_school), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1.0
                V_school[j * p + k, col] = 1.0
                W_school[k * n + i, col] = 1.0
                col += 1
    
    best_rank = rank_school
    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    # Try to find lower rank decompositions
    # Theoretical lower bound is around 59-64 for 3x3x8
    target_ranks = [64, 63, 62, 61, 60, 59]
    
    for target_rank in target_ranks:
        if time.time() > deadline - 20:
            break
            
        found = False
        max_attempts = 30 if target_rank >= 64 else 20
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
                
            seed = 42 + target_rank * 100 + attempt * 7
            
            # Alternate between block and random init
            if attempt % 2 == 0:
                U, V, W = _block_init(n, m, p, target_rank, seed=seed)
            else:
                U, V, W = _random_init(n, m, p, target_rank, seed=seed)
            
            U_res, V_res, W_res = _alternating_minimization(U, V, W, n, m, p, max_iter=200)
            
            if U_res is not None and _verify_decomposition(U_res, V_res, W_res, n, m, p):
                best_rank = target_rank
                best_result = {"U": U_res.tolist(), "V": V_res.tolist(), "W": W_res.tolist()}
                found = True
                break
        
        if found:
            continue  # Try next lower rank
    
    return best_result
# EVOLVE-BLOCK-END
