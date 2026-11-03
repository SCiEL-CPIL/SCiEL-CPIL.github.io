"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 6]
START_TIME = time.time()
TIME_LIMIT = 120 # Safety margin

def construct():
    n, m, p = DIMENSIONS
    
    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def verify(U, V, W, tol=1e-7):
        """
        Verifies if U, V, W form a valid decomposition for (n,m,p) matrix multiplication.
        """
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        rank = U.shape[1]
        
        # Target tensor T[i, j, k] = 1
        # Decomposed tensor D[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # Efficient check
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U[i * m + j, :]
                    v_val = V[j * p + k, :]
                    w_val = W[k * n + i, :]
                    s = np.sum(u_val * v_val * w_val)
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def _try_rank(n, m, p, rank, seed=0, max_iters=200, deadline=None):
        """
        Try to find a rank-r decomposition using alternating least squares (ALS).
        Returns (U, V, W, err) or (None, None, None, inf) if not found.
        """
        rng = np.random.RandomState(seed)
        U = rng.randn(n * m, rank)
        V = rng.randn(m * p, rank)
        W = rng.randn(p * n, rank)
        
        best_err = float('inf')
        for it in range(max_iters):
            if deadline is not None and time.time() > deadline - 2.0:
                break
                
            # Alternate minimization steps
            # 1. Solve for U
            for i in range(n):
                for j in range(m):
                    M = np.zeros((p, rank))
                    for k in range(p):
                        M[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    try:
                        U[i * m + j, :] = np.linalg.lstsq(M.T, np.ones(p), rcond=None)[0]
                    except:
                        pass
            
            # 2. Solve for V
            for j in range(m):
                for k in range(p):
                    M = np.zeros((n, rank))
                    for i in range(n):
                        M[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    try:
                        V[j * p + k, :] = np.linalg.lstsq(M.T, np.ones(n), rcond=None)[0]
                    except:
                        pass
            
            # 3. Solve for W
            for k in range(p):
                for i in range(n):
                    M = np.zeros((m, rank))
                    for j in range(m):
                        M[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    try:
                        W[k * n + i, :] = np.linalg.lstsq(M.T, np.ones(m), rcond=None)[0]
                    except:
                        pass
        
        # Calculate final error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                    err += (s - 1.0)**2
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_err < 1e-4:
            return best_U, best_V, best_W, best_err
        return None, None, None, best_err

    # Fallback to schoolbook
    schoolbook_sol = get_schoolbook()
    
    # Try to find a lower rank solution
    if time.time() - START_TIME < TIME_LIMIT - 10:
        deadline = START_TIME + TIME_LIMIT - 5
        school_rank = n * m * p
        
        # Try ranks from a reasonable lower bound up to schoolbook rank - 1
        # Lower bound is max(n*m, m*p, p*n) = 18 for 3x3x6
        # However, we might start higher to ensure we find something, or start lower for better score.
        # Let's try a range.
        # Note: ALS is non-convex, so we need multiple seeds.
        
        # We will try ranks in ascending order to find the minimal one.
        # But if we fail at low rank, we might waste time.
        # Given the time limit, we'll try a few specific ranks.
        
        # Known theoretical bounds or just heuristic search.
        # Let's try ranks from 18 to school_rank - 1.
        # To save time, we'll limit the number of seeds per rank if time is tight.
        
        found = False
        for rank in range(18, school_rank):
            if time.time() > deadline - 10:
                break
            
            # Try a few seeds for this rank
            seeds_to_try = 10
            if time.time() > deadline - 30:
                seeds_to_try = 2
            
            for seed in range(seeds_to_try):
                if time.time() > deadline - 5:
                    break
                
                U, V, W, err = _try_rank(n, m, p, rank, seed=seed, max_iters=100, deadline=deadline)
                
                if U is not None and err < 1e-4:
                    # Verify with tighter tolerance
                    if verify(U, V, W, tol=1e-6):
                        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    return schoolbook_sol

# EVOLVE-BLOCK-START
# The construct function is defined above to respect the interface.
# EVOLVE-BLOCK-END
