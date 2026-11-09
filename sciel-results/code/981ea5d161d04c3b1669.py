"""Find a rank-minimal exact tensor decomposition for 2x4 * 4x6 matrix multiplication."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 6]

def _verify_fast(U, V, W, n, m, p):
    """Fast verification using einsum."""
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    R = U.shape[1]
    
    U3 = U.reshape(n, m, R)
    V3 = V.reshape(m, p, R)
    W3 = W.reshape(p, n, R)
    
    T = np.einsum('ijr,jkr,pir->ijjkpi', U3, V3, W3, optimize=True)
    
    I, J, JP, K, KP, IP = np.meshgrid(
        range(n), range(m), range(m), range(p), range(p), range(n),
        indexing='ij'
    )
    target = ((I == IP) & (J == JP) & (K == KP)).astype(np.float64)
    
    return np.allclose(T, target, atol=1e-9, rtol=1e-9)

def _polish(U, V, W, n, m, p, max_iter=1000, tol=1e-10):
    """Local polish: adjust entries to fix small errors."""
    U = U.astype(np.float64).copy()
    V = V.astype(np.float64).copy()
    W = W.astype(np.float64).copy()
    R = U.shape[1]
    
    U3 = U.reshape(n, m, R)
    V3 = V.reshape(m, p, R)
    W3 = W.reshape(p, n, R)
    
    I, J, JP, K, KP, IP = np.meshgrid(
        range(n), range(m), range(m), range(p), range(p), range(n),
        indexing='ij'
    )
    target = ((I == IP) & (J == JP) & (K == KP)).astype(np.float64)
    
    for _ in range(max_iter):
        T = np.einsum('ijr,jkr,pir->ijjkpi', U3, V3, W3, optimize=True)
        err = T - target
        if np.max(np.abs(err)) < tol:
            break
        # Simple gradient-free: nudge entries in direction of error
        # This is heuristic; may not converge
        break  # Skip complex polish for now
    
    return U, V, W

def construct():
    start_time = time.time()
    DEADLINE = 140.0
    n, m, p = DIMENSIONS
    
    # Schoolbook
    rank_school = n * m * p
    U = np.zeros((n * m, rank_school))
    V = np.zeros((m * p, rank_school))
    W = np.zeros((p * n, rank_school))
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank_school
    
    random.seed(42)
    np.random.seed(42)
    
    # Strategy 1: Structured random search
    lb = max(n * p, m * p, n * m)
    
    for target_rank in range(lb, best_rank):
        if time.time() > DEADLINE - 15:
            break
        
        max_attempts = 150
        for attempt in range(max_attempts):
            if time.time() > DEADLINE - 10:
                break
            
            method = attempt % 5
            if method == 0:
                # Sparse ±1
                U_rand = np.zeros((n * m, target_rank), dtype=int)
                V_rand = np.zeros((m * p, target_rank), dtype=int)
                W_rand = np.zeros((p * n, target_rank), dtype=int)
                for r in range(target_rank):
                    nz_u = np.random.randint(1, 5)
                    nz_v = np.random.randint(1, 5)
                    nz_w = np.random.randint(1, 5)
                    U_rand[np.random.choice(n*m, nz_u, replace=False), r] = np.random.choice([-1, 1], nz_u)
                    V_rand[np.random.choice(m*p, nz_v, replace=False), r] = np.random.choice([-1, 1], nz_v)
                    W_rand[np.random.choice(p*n, nz_w, replace=False), r] = np.random.choice([-1, 1], nz_w)
            elif method == 1:
                # Dense small integers
                U_rand = np.random.randint(-2, 3, (n * m, target_rank))
                V_rand = np.random.randint(-2, 3, (m * p, target_rank))
                W_rand = np.random.randint(-2, 3, (p * n, target_rank))
            elif method == 2:
                # Half-integers
                U_rand = np.random.randint(-4, 5, (n * m, target_rank)) / 2.0
                V_rand = np.random.randint(-4, 5, (m * p, target_rank)) / 2.0
                W_rand = np.random.randint(-4, 5, (p * n, target_rank)) / 2.0
            elif method == 3:
                # Block structure: U has blocks per i, V per j, W per k
                U_rand = np.zeros((n * m, target_rank))
                V_rand = np.zeros((m * p, target_rank))
                W_rand = np.zeros((p * n, target_rank))
                for r in range(target_rank):
                    i0 = np.random.randint(0, n)
                    j0 = np.random.randint(0, m)
                    k0 = np.random.randint(0, p)
                    U_rand[i0*m:(i0+1)*m, r] = np.random.choice([-1, 0, 1], m)
                    V_rand[j0*p:(j0+1)*p, r] = np.random.choice([-1, 0, 1], p)
                    W_rand[k0*n:(k0+1)*n, r] = np.random.choice([-1, 0, 1], n)
            else:
                # Very sparse: 1-2 non-zeros per row per factor
                U_rand = np.zeros((n * m, target_rank), dtype=int)
                V_rand = np.zeros((m * p, target_rank), dtype=int)
                W_rand = np.zeros((p * n, target_rank), dtype=int)
                for r in range(target_rank):
                    U_rand[np.random.choice(n*m, 1, replace=False), r] = np.random.choice([-1, 1])
                    V_rand[np.random.choice(m*p, 1, replace=False), r] = np.random.choice([-1, 1])
                    W_rand[np.random.choice(p*n, 1, replace=False), r] = np.random.choice([-1, 1])
            
            if _verify_fast(U_rand, V_rand, W_rand, n, m, p):
                best_U = U_rand.copy()
                best_V = V_rand.copy()
                best_W = W_rand.copy()
                best_rank = target_rank
                break
        
        if best_rank == target_rank:
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
