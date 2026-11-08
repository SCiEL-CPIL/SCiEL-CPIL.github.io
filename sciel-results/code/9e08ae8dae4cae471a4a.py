"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 20
    mp = m * p  # 30
    pn = p * n  # 24
    
    deadline = time.time() + 130  # safety margin before 150s kill
    
    best_rank = n * m * p  # start with schoolbook
    best_result = None
    
    # First, provide the schoolbook as fallback
    schoolbook_U = np.zeros((nm, best_rank))
    schoolbook_V = np.zeros((mp, best_rank))
    schoolbook_W = np.zeros((pn, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                schoolbook_U[i * m + j, col] = 1
                schoolbook_V[j * p + k, col] = 1
                schoolbook_W[k * n + i, col] = 1
                col += 1
    best_result = {
        "U": schoolbook_U.tolist(),
        "V": schoolbook_V.tolist(),
        "W": schoolbook_W.tolist()
    }
    
    np.random.seed(42)
    
    # Precompute index arrays for vectorized ALS
    # For U-update: for each (i,j), we need V[j*p+k,:] * W[k*n+i,:] for k=0..p-1
    # For V-update: for each (j,k), we need U[i*m+j,:] * W[k*n+i,:] for i=0..n-1
    # For W-update: for each (k,i), we need U[i*m+j,:] * V[j*p+k,:] for j=0..m-1
    
    # Precompute index lookups
    # u_idx[i,j] = i*m+j
    # v_idx[j,k] = j*p+k
    # w_idx[k,i] = k*n+i
    
    def check_error(U, V, W, tol=1e-9):
        """Check max error of tensor decomposition."""
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                    max_err = max(max_err, abs(val - 1.0))
        return max_err
    
    def try_half_integer(U, V, W, max_trials=50):
        """Try to find a half-integer solution near the given solution."""
        # First try direct rounding
        U_round = np.round(U * 2) / 2
        V_round = np.round(V * 2) / 2
        W_round = np.round(W * 2) / 2
        err = check_error(U_round, V_round, W_round)
        if err < 1e-9:
            return U_round, V_round, W_round
        
        # Try perturbed rounding
        for _ in range(max_trials):
            noise_u = np.random.randn(nm, U.shape[1]) * 0.05
            noise_v = np.random.randn(mp, V.shape[1]) * 0.05
            noise_w = np.random.randn(pn, W.shape[1]) * 0.05
            U_t = np.round((U + noise_u) * 2) / 2
            V_t = np.round((V + noise_v) * 2) / 2
            W_t = np.round((W + noise_w) * 2) / 2
            err = check_error(U_t, V_t, W_t)
            if err < 1e-9:
                return U_t, V_t, W_t
        return None, None, None
    
    # Try ranks from lower to higher (we want minimal rank)
    # Known lower bounds suggest rank >= max(n*p, n*m, m*p) but actual bounds are tighter
    # For 4x5*5x6, let's try from rank 90 upward
    for rank in range(90, 120):
        if time.time() > deadline - 40:
            break
        
        # Multiple random restarts with varying initialization scales
        for restart in range(8):
            if time.time() > deadline - 15:
                break
            
            # Random initialization with varying scale
            scale = 0.3 + 0.2 * (restart % 3)
            U = np.random.randn(nm, rank) * scale
            V = np.random.randn(mp, rank) * scale
            W = np.random.randn(pn, rank) * scale
            
            # ALS iterations
            converged = False
            for it in range(80):
                if time.time() > deadline - 8:
                    break
                
                # === Update U ===
                # For each (i,j): solve A @ x = b where A[k,c] = V[j*p+k,c]*W[k*n+i,c]
                for i in range(n):
                    for j in range(m):
                        # Build A: p x rank
                        A = np.empty((p, rank))
                        for k in range(p):
                            A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                        b = np.ones(p)
                        try:
                            x = np.linalg.lstsq(A, b, rcond=None)[0]
                            U[i * m + j, :] = x
                        except:
                            pass
                
                # === Update V ===
                # For each (j,k): solve A @ x = b where A[i,c] = U[i*m+j,c]*W[k*n+i,c]
                for j in range(m):
                    for k in range(p):
                        A = np.empty((n, rank))
                        for i in range(n):
                            A[i, :] = U[i * m + j, :] * W[k * n + i, :]
                        b = np.ones(n)
                        try:
                            x = np.linalg.lstsq(A, b, rcond=None)[0]
                            V[j * p + k, :] = x
                        except:
                            pass
                
                # === Update W ===
                # For each (k,i): solve A @ x = b where A[j,c] = U[i*m+j,c]*V[j*p+k,c]
                for k in range(p):
                    for i in range(n):
                        A = np.empty((m, rank))
                        for j in range(m):
                            A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                        b = np.ones(m)
                        try:
                            x = np.linalg.lstsq(A, b, rcond=None)[0]
                            W[k * n + i, :] = x
                        except:
                            pass
                
                # Check convergence
                max_err = check_error(U, V, W, tol=1e-10)
                if max_err < 1e-10:
                    converged = True
                    break
            
            if converged:
                # Found a solution! Try to find exact half-integer version
                U_h, V_h, W_h = try_half_integer(U, V, W, max_trials=80)
                if U_h is not None:
                    best_rank = rank
                    best_result = {
                        "U": U_h.tolist(),
                        "V": V_h.tolist(),
                        "W": W_h.tolist()
                    }
                    break  # found for this rank, go to lower ranks
        
        if best_rank < n * m * p:
            break  # found a better rank, stop
    
    if time.time() > deadline - 2:
        return best_result
    
    return best_result
# EVOLVE-BLOCK-END
