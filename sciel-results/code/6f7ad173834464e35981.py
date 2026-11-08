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
    
    deadline = time.time() + 140  # safety margin before 150s kill
    
    best_rank = n * m * p  # start with schoolbook (120)
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
    
    def check_error(U, V, W, tol=1e-9):
        """Check max error of tensor decomposition using vectorized einsum."""
        # Reshape U, V, W to 3D tensors
        # U: (n, m, rank)
        # V: (m, p, rank)
        # W: (p, n, rank)
        U_3d = U.reshape(n, m, -1)
        V_3d = V.reshape(m, p, -1)
        W_3d = W.reshape(p, n, -1)
        
        # Compute T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        # Indices: i, j, k, c
        # U: i, j, c
        # V: j, k, c
        # W: k, i, c
        T = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
        
        max_err = np.max(np.abs(T - 1.0))
        return float(max_err)
    
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
    # The strategy suggests rank 70 might be achievable via Strassen-Winograd.
    # We search from 70 up to 119.
    for rank in range(70, 120):
        if time.time() > deadline - 20:
            break
        
        # Multiple random restarts
        # We might need more restarts for lower ranks if they are harder to find
        # But we must manage time.
        restarts = 10 if rank < 90 else 5
        
        for restart in range(restarts):
            if time.time() > deadline - 10:
                break
            
            # Random initialization
            scale = 0.5
            U = np.random.randn(nm, rank) * scale
            V = np.random.randn(mp, rank) * scale
            W = np.random.randn(pn, rank) * scale
            
            # ALS iterations
            converged = False
            for it in range(100):
                if time.time() > deadline - 5:
                    break
                
                # === Update U ===
                for i in range(n):
                    for j in range(m):
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
                U_h, V_h, W_h = try_half_integer(U, V, W, max_trials=100)
                if U_h is not None:
                    best_rank = rank
                    best_result = {
                        "U": U_h.tolist(),
                        "V": V_h.tolist(),
                        "W": W_h.tolist()
                    }
                    break  # found for this rank, go to lower ranks (loop breaks)
        
        if best_rank < n * m * p:
            break  # found a better rank, stop
    
    if time.time() > deadline - 2:
        return best_result
    
    return best_result
# EVOLVE-BLOCK-END
