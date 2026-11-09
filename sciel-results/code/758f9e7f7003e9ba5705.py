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
    
    def check_error(U, V, W, tol=1e-9):
        """Check max error of tensor decomposition."""
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                    max_err = max(max_err, abs(val - 1.0))
        return max_err
    
    def try_half_integer(U, V, W, max_trials=200):
        """Try to find a half-integer solution near the given solution."""
        # First try direct rounding
        U_round = np.round(U * 2) / 2
        V_round = np.round(V * 2) / 2
        W_round = np.round(W * 2) / 2
        err = check_error(U_round, V_round, W_round)
        if err < 1e-9:
            return U_round, V_round, W_round
        
        # Try perturbed rounding with increasing noise
        for trial in range(max_trials):
            noise_scale = 0.02 + 0.03 * (trial % 10)
            noise_u = np.random.randn(nm, U.shape[1]) * noise_scale
            noise_v = np.random.randn(mp, V.shape[1]) * noise_scale
            noise_w = np.random.randn(pn, W.shape[1]) * noise_scale
            U_t = np.round((U + noise_u) * 2) / 2
            V_t = np.round((V + noise_v) * 2) / 2
            W_t = np.round((W + noise_w) * 2) / 2
            err = check_error(U_t, V_t, W_t)
            if err < 1e-9:
                return U_t, V_t, W_t
        
        # Try coordinate-wise rounding
        for _ in range(50):
            U_t = np.round(U * 2) / 2
            V_t = np.round(V * 2) / 2
            W_t = np.round(W * 2) / 2
            # Perturb random coordinates
            for _ in range(20):
                idx_u = (np.random.randint(nm), np.random.randint(U.shape[1]))
                idx_v = (np.random.randint(mp), np.random.randint(V.shape[1]))
                idx_w = (np.random.randint(pn), np.random.randint(W.shape[1]))
                U_t[idx_u] += np.random.choice([-0.5, 0.5])
                V_t[idx_v] += np.random.choice([-0.5, 0.5])
                W_t[idx_w] += np.random.choice([-0.5, 0.5])
            err = check_error(U_t, V_t, W_t)
            if err < 1e-9:
                return U_t, V_t, W_t
        
        return None, None, None
    
    # Vectorized ALS update functions
    def als_update_U(U, V, W):
        """Vectorized update for U matrix."""
        for i in range(n):
            for j in range(m):
                # Build A: p x rank where A[k, c] = V[j*p+k, c] * W[k*n+i, c]
                A = V[j*p:(j+1)*p, :] * W[i::n, :]  # shape (p, rank)
                b = np.ones(p)
                try:
                    x, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                    U[i * m + j, :] = x
                except:
                    pass
        return U
    
    def als_update_V(U, V, W):
        """Vectorized update for V matrix."""
        for j in range(m):
            for k in range(p):
                # Build A: n x rank where A[i, c] = U[i*m+j, c] * W[k*n+i, c]
                A = U[j::m, :] * W[k::p, :]  # shape (n, rank)
                b = np.ones(n)
                try:
                    x, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                    V[j * p + k, :] = x
                except:
                    pass
        return V
    
    def als_update_W(U, V, W):
        """Vectorized update for W matrix."""
        for k in range(p):
            for i in range(n):
                # Build A: m x rank where A[j, c] = U[i*m+j, c] * V[j*p+k, c]
                A = U[i*m:(i+1)*m, :] * V[k::p, :]  # shape (m, rank)
                b = np.ones(m)
                try:
                    x, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                    W[k * n + i, :] = x
                except:
                    pass
        return W
    
    # Try ranks from lower to higher (we want minimal rank)
    # For 4x5*5x6, theoretical lower bound is around 90-100
    for rank in range(90, 120):
        if time.time() > deadline - 30:
            break
        
        # Multiple random restarts with varying initialization
        for restart in range(12):
            if time.time() > deadline - 10:
                break
            
            # Different initialization strategies
            if restart % 3 == 0:
                # Standard random
                scale = 0.3 + 0.1 * (restart % 5)
                U = np.random.randn(nm, rank) * scale
                V = np.random.randn(mp, rank) * scale
                W = np.random.randn(pn, rank) * scale
            elif restart % 3 == 1:
                # Sparse-like initialization
                U = np.random.randn(nm, rank) * 0.5
                V = np.random.randn(mp, rank) * 0.5
                W = np.random.randn(pn, rank) * 0.5
                # Zero out some entries
                mask_u = np.random.rand(nm, rank) < 0.3
                mask_v = np.random.rand(mp, rank) < 0.3
                mask_w = np.random.rand(pn, rank) < 0.3
                U[mask_u] = 0
                V[mask_v] = 0
                W[mask_w] = 0
            else:
                # Structured initialization based on schoolbook pattern
                U = np.zeros((nm, rank))
                V = np.zeros((mp, rank))
                W = np.zeros((pn, rank))
                for r in range(rank):
                    i = r % n
                    j = (r // n) % m
                    k = (r // (n*m)) % p
                    U[i*m+j, r] = 1.0
                    V[j*p+k, r] = 1.0
                    W[k*n+i, r] = 1.0
                # Add noise
                U += np.random.randn(nm, rank) * 0.1
                V += np.random.randn(mp, rank) * 0.1
                W += np.random.randn(pn, rank) * 0.1
            
            # ALS iterations with convergence check
            converged = False
            prev_err = float('inf')
            for it in range(100):
                if time.time() > deadline - 5:
                    break
                
                # Update all factors
                U = als_update_U(U, V, W)
                V = als_update_V(U, V, W)
                W = als_update_W(U, V, W)
                
                # Check convergence
                max_err = check_error(U, V, W, tol=1e-10)
                if max_err < 1e-10:
                    converged = True
                    break
                # Check for stagnation
                if abs(prev_err - max_err) < 1e-12 and it > 10:
                    break
                prev_err = max_err
            
            if converged:
                # Found a solution! Try to find exact half-integer version
                U_h, V_h, W_h = try_half_integer(U, V, W, max_trials=150)
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
