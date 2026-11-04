"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import sys

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 145  # Leave 5s margin before 150s kill
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 72
    
    # Build target tensor T[i,j,k] = 1 if (i,j,k) is valid else 0
    # T has shape (n*m, m*p, p*n)
    # Index: u = i*m+j, v = j*p+k, w = k*n+i
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
    
    T_flat = T.flatten()
    NM, MP, PN = n*m, m*p, p*n
    
    def check_exact(U, V, W, rank):
        """Check if U, V, W with given rank exactly reconstruct T"""
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        T_rec = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        return np.allclose(T_rec.flatten(), T_flat, atol=1e-9)
    
    def als_decompose(rank, max_iters=80, tol=1e-12, seed=0):
        """Alternating Least Squares for 3rd order tensor decomposition with L-BFGS-B polish"""
        np.random.seed(seed)
        
        # Initialize with small random values
        U = np.random.randn(n, m, rank) * 0.3
        V = np.random.randn(m, p, rank) * 0.3
        W = np.random.randn(p, n, rank) * 0.3
        
        prev_err = np.inf
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Compute current error
            T_rec = np.einsum('ijc,jkc,kic->ijk', U, V, W)
            err = T_rec - T
            current_err = np.sum(err**2)
            
            if abs(prev_err - current_err) < tol:
                break
            prev_err = current_err
            
            # Solve for U: fix V, W
            for i in range(n):
                for j in range(m):
                    A = np.einsum('kc,kc->ck', V[j], W[:, i])
                    b = T[i, j, :]
                    try:
                        u_sol, _, _, _ = np.linalg.lstsq(A.T, b, rcond=None)
                        U[i, j, :] = u_sol
                    except:
                        pass
            
            # Solve for V: fix U, W
            for j in range(m):
                for k in range(p):
                    A = np.einsum('ic,ic->ci', U[:, j, :], W[k, :, :])
                    b = T[:, j, k]
                    try:
                        v_sol, _, _, _ = np.linalg.lstsq(A.T, b, rcond=None)
                        V[j, k, :] = v_sol
                    except:
                        pass
            
            # Solve for W: fix U, V
            for k in range(p):
                for i in range(n):
                    A = np.einsum('jc,jc->cj', U[i, :, :], V[:, k, :])
                    b = T[i, :, k]
                    try:
                        w_sol, _, _, _ = np.linalg.lstsq(A.T, b, rcond=None)
                        W[k, i, :] = w_sol
                    except:
                        pass
        
        # Final check
        T_rec = np.einsum('ijc,jkc,kic->ijk', U, V, W)
        final_err = np.sum((T_rec - T)**2)
        
        if final_err < 1e-8:
            # Try rounding to half-integers
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            if check_exact(U_rounded, V_rounded, W_rounded, rank):
                return U_rounded.flatten(axis=1).tolist(), V_rounded.flatten(axis=1).tolist(), W_rounded.flatten(axis=1).tolist()
            
            # Try rounding to integers
            U_int = np.round(U)
            V_int = np.round(V)
            W_int = np.round(W)
            if check_exact(U_int, V_int, W_int, rank):
                return U_int.flatten(axis=1).tolist(), V_int.flatten(axis=1).tolist(), W_int.flatten(axis=1).tolist()
            
            # Polish with L-BFGS-B if close but not exact
            if final_err < 1.0:
                try:
                    from scipy.optimize import minimize
                    
                    x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
                    
                    def objective(x):
                        U_x = x[:n*m*rank].reshape(n, m, rank)
                        V_x = x[n*m*rank:n*m*rank+m*p*rank].reshape(m, p, rank)
                        W_x = x[n*m*rank+m*p*rank:].reshape(p, n, rank)
                        T_rec = np.einsum('ijc,jkc,kic->ijk', U_x, V_x, W_x)
                        return np.sum((T_rec - T)**2)
                    
                    res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 50, 'ftol': 1e-15})
                    
                    if res.success:
                        x_opt = res.x
                        U_opt = x_opt[:n*m*rank].reshape(n, m, rank)
                        V_opt = x_opt[n*m*rank:n*m*rank+m*p*rank].reshape(m, p, rank)
                        W_opt = x_opt[n*m*rank+m*p*rank:].reshape(p, n, rank)
                        
                        # Round and check
                        U_rounded = np.round(U_opt * 2) / 2
                        V_rounded = np.round(V_opt * 2) / 2
                        W_rounded = np.round(W_opt * 2) / 2
                        if check_exact(U_rounded, V_rounded, W_rounded, rank):
                            return U_rounded.flatten(axis=1).tolist(), V_rounded.flatten(axis=1).tolist(), W_rounded.flatten(axis=1).tolist()
                        
                        U_int = np.round(U_opt)
                        V_int = np.round(V_opt)
                        W_int = np.round(W_opt)
                        if check_exact(U_int, V_int, W_int, rank):
                            return U_int.flatten(axis=1).tolist(), V_int.flatten(axis=1).tolist(), W_int.flatten(axis=1).tolist()
                except Exception:
                    pass
        
        return None
    
    def try_rank(rank, num_restarts=8):
        """Try to find decomposition of given rank with multiple restarts"""
        for seed in range(num_restarts):
            if time.time() > deadline - 5:
                return None
            result = als_decompose(rank, max_iters=80, tol=1e-12, seed=seed)
            if result is not None:
                return result
        return None
    
    # Search for lower rank.
    # Hypothesis: Ranks close to schoolbook are easier to find or more likely to be the true minimal rank.
    # We search downwards from 71 to 60, then jump to 50, 40.
    best_result = None
    
    search_ranks = [71, 70, 69, 68, 67, 66, 65, 64, 63, 62, 61, 60, 55, 50, 45, 40]
    
    for rank in search_ranks:
        if time.time() > deadline - 10:
            break
        
        result = try_rank(rank, num_restarts=5)
        if result is not None:
            U, V, W = result
            best_result = {"U": U, "V": V, "W": W}
            break
    
    # Fallback to schoolbook
    if best_result is None:
        rank = rank_schoolbook
        U = [[0] * rank for _ in range(NM)]
        V = [[0] * rank for _ in range(MP)]
        W = [[0] * rank for _ in range(PN)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        best_result = {"U": U, "V": V, "W": W}
    
    return best_result
# EVOLVE-BLOCK-END
