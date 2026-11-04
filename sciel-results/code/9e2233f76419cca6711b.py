import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    N = n * m
    M = m * p
    P = p * n
    
    # Build the target tensor
    T = np.zeros((N, M, P))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T[a_idx, b_idx, c_idx] = 1.0
    
    # Pre-compute matricizations
    T_c = T.reshape(N * M, P)
    T_a = T.reshape(N, M * P)
    T_b = T.transpose(1, 0, 2).reshape(M, N * P)
    
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    def snap_to_half_int(X):
        """Snap values to nearest half-integer"""
        return np.round(X * 2) / 2
    
    def check_exact(U, V, W, tol=1e-8):
        """Check if U,V,W give exact decomposition after snapping to half-integers"""
        U_r = snap_to_half_int(U)
        V_r = snap_to_half_int(V)
        W_r = snap_to_half_int(W)
        T_calc = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
        if np.allclose(T_calc, T, atol=tol):
            return True, U_r, V_r, W_r
        return False, None, None, None
    
    def check_exact_raw(U, V, W, tol=1e-8):
        """Check if U,V,W give exact decomposition without snapping"""
        T_calc = np.einsum('ar,br,cr->abc', U, V, W)
        if np.allclose(T_calc, T, atol=tol):
            return True, U, V, W
        return False, None, None, None
    
    def als_iteration(U, V, W, R):
        """One ALS iteration: update W, then U, then V"""
        # Update W
        E_W = (U[:, np.newaxis, :] * V[np.newaxis, :, :]).reshape(N * M, R)
        W_T, _, _, _ = np.linalg.lstsq(E_W, T_c.T, rcond=None)
        W = W_T.T
        
        # Update U
        G_U = (V[:, np.newaxis, :] * W[np.newaxis, :, :]).reshape(M * P, R)
        U_T, _, _, _ = np.linalg.lstsq(G_U.T, T_a.T, rcond=None)
        U = U_T.T
        
        # Update V
        G_V = (U[:, np.newaxis, :] * W[np.newaxis, :, :]).reshape(N * P, R)
        V_T, _, _, _ = np.linalg.lstsq(G_V.T, T_b.T, rcond=None)
        V = V_T.T
        
        return U, V, W
    
    def compute_error(U, V, W):
        """Compute Frobenius error of current decomposition"""
        E_W = (U[:, np.newaxis, :] * V[np.newaxis, :, :]).reshape(N * M, W.shape[1])
        T_calc_c = E_W @ W.T
        return np.linalg.norm(T_calc_c - T_c)
    
    best_rank = N * M * P  # Schoolbook fallback
    best_U = None
    best_V = None
    best_W = None
    
    # Try ranks from 35 down to 20
    # Allocate more time to higher ranks first where solutions are more likely
    ranks_to_try = list(range(35, 19, -1))
    
    for R in ranks_to_try:
        if time.time() > deadline:
            break
        
        # Determine number of attempts based on rank and remaining time
        remaining = deadline - time.time()
        # Higher ranks: more attempts, lower ranks: fewer
        if R >= 33:
            attempts = 8
        elif R >= 30:
            attempts = 6
        else:
            attempts = 4
        
        for attempt in range(attempts):
            if time.time() > deadline:
                break
            
            # Diverse initialization
            seed = 10000 + R * 1000 + attempt * 7 + int(time.time() * 1000) % 1000
            np.random.seed(seed)
            
            # Mix of initialization strategies
            if attempt % 3 == 0:
                # Standard random
                U = np.random.randn(N, R) * 0.5
                V = np.random.randn(M, R) * 0.5
                W = np.random.randn(P, R) * 0.5
            elif attempt % 3 == 1:
                # Sparse-ish initialization
                U = np.random.randn(N, R) * 0.3
                V = np.random.randn(M, R) * 0.3
                W = np.random.randn(P, R) * 0.3
            else:
                # Structured initialization based on tensor structure
                U = np.zeros((N, R))
                V = np.zeros((M, R))
                W = np.zeros((P, R))
                for r in range(min(R, 36)):
                    # Distribute entries to match tensor structure
                    i = r % n
                    j = (r // n) % m
                    k = (r // (n * m)) % p
                    U[i * m + j, r] = 1.0
                    V[j * p + k, r] = 1.0
                    W[k * n + i, r] = 1.0
                # Add small random perturbation
                U += np.random.randn(N, R) * 0.1
                V += np.random.randn(M, R) * 0.1
                W += np.random.randn(P, R) * 0.1
            
            # Run ALS with convergence check
            max_iters = 50
            converged = False
            
            for it in range(max_iters):
                if time.time() > deadline:
                    break
                
                U, V, W = als_iteration(U, V, W, R)
                
                # Check convergence
                err = compute_error(U, V, W)
                if err < 1e-10:
                    # Try to snap to half-integers
                    valid, U_r, V_r, W_r = check_exact(U, V, W)
                    if valid:
                        best_rank = R
                        best_U = U_r
                        best_V = V_r
                        best_W = W_r
                        converged = True
                        break
                    
                    # If raw solution is good but not half-integer, try more iterations
                    # or accept if very close
                    if err < 1e-12:
                        # Very close - try a few more iterations to see if it snaps
                        for polish_it in range(10):
                            U, V, W = als_iteration(U, V, W, R)
                            err = compute_error(U, V, W)
                            if err < 1e-12:
                                valid, U_r, V_r, W_r = check_exact(U, V, W)
                                if valid:
                                    best_rank = R
                                    best_U = U_r
                                    best_V = V_r
                                    best_W = W_r
                                    converged = True
                                    break
                        if converged:
                            break
            
            if converged:
                break
    
    # Final check: if we found a solution, return it
    if best_U is not None:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Fallback to schoolbook
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
