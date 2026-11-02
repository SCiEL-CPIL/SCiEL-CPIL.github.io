"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = 140  # 10s safety margin before 150s kill

def _verify(U, V, W, n, m, p, tol=1e-10):
    """Check if U,V,W form a valid decomposition."""
    rank = len(U[0]) if U else 0
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # T[i,j,k] should equal sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _try_rank_structured(rank, n, m, p, max_iters=5000, seed=42):
    """Try to find a rank-rank decomposition using structured initialization."""
    rng = np.random.RandomState(seed)
    
    # Initialize with a structured pattern based on the problem
    # Use the fact that for matrix multiplication, there are known patterns
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Try different initialization strategies
    for init_strategy in range(3):
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        if init_strategy == 0:
            # Random initialization
            U = rng.randn(n*m, rank) * 0.1
            V = rng.randn(m*p, rank) * 0.1
            W = rng.randn(p*n, rank) * 0.1
        elif init_strategy == 1:
            # Structured: use basis-like initialization
            for c in range(rank):
                # Distribute the "weight" across indices
                i_idx = c % n
                j_idx = (c // n) % m
                k_idx = (c // (n*m)) % p
                
                U[i_idx * m + j_idx, c] = 1.0
                V[j_idx * p + k_idx, c] = 1.0
                W[k_idx * n + i_idx, c] = 1.0
        else:
            # Hybrid: mix of random and structured
            U = rng.randn(n*m, rank) * 0.05
            V = rng.randn(m*p, rank) * 0.05
            W = rng.randn(p*n, rank) * 0.05
            for c in range(rank):
                i_idx = c % n
                j_idx = (c // n) % m
                k_idx = (c // (n*m)) % p
                U[i_idx * m + j_idx, c] += 0.5
                V[j_idx * p + k_idx, c] += 0.5
                W[k_idx * n + i_idx, c] += 0.5
        
        # Use ALS with better convergence
        for it in range(max_iters):
            if time.time() > _DEADLINE:
                return None
            
            # Update U
            for i in range(n):
                for j in range(m):
                    A = np.zeros((p, rank))
                    for k in range(p):
                        A[k,:] = V[j*p+k,:] * W[k*n+i,:]
                    # Solve A @ u = 1
                    if rank <= p:
                        u, _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                    else:
                        # Use pseudo-inverse approach
                        A_pinv = np.linalg.pinv(A)
                        u = A_pinv @ np.ones(p)
                    U[i*m+j,:] = u
            
            # Update V
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, rank))
                    for i in range(n):
                        A[i,:] = U[i*m+j,:] * W[k*n+i,:]
                    if rank <= n:
                        v, _, _, _ = np.linalg.lstsq(A, np.ones(n), rcond=None)
                    else:
                        A_pinv = np.linalg.pinv(A)
                        v = A_pinv @ np.ones(n)
                    V[j*p+k,:] = v
            
            # Update W
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m, rank))
                    for j in range(m):
                        A[j,:] = U[i*m+j,:] * V[j*p+k,:]
                    if rank <= m:
                        w, _, _, _ = np.linalg.lstsq(A, np.ones(m), rcond=None)
                    else:
                        A_pinv = np.linalg.pinv(A)
                        w = A_pinv @ np.ones(m)
                    W[k*n+i,:] = w
            
            # Check convergence
            if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p, tol=1e-8):
                return (U.tolist(), V.tolist(), W.tolist())
    
    return None

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    schoolbook_rank = n * m * p
    
    # Lower bound
    lower_bound = max(n*m, m*p, n*p)
    
    best_U, best_V, best_W = None, None, None
    best_rank = schoolbook_rank
    
    # Try ranks from lower bound up to schoolbook
    for rank in range(lower_bound, best_rank + 1):
        if time.time() > _DEADLINE - 5:
            break
        
        # Try multiple seeds and strategies
        for seed in range(20):
            if time.time() > _DEADLINE - 5:
                break
            result = _try_rank_structured(rank, n, m, p, max_iters=2000, seed=seed)
            if result is not None:
                U, V, W = result
                if _verify(U, V, W, n, m, p):
                    best_U, best_V, best_W = U, V, W
                    best_rank = rank
                    break
    
    if best_U is None:
        # Fall back to schoolbook
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
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
