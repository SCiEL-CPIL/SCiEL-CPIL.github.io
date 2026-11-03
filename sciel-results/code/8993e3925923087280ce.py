"""Matrix multiplication 2x3x6 - Gauss-Newton search for low-rank exact decomposition."""
import time
import numpy as np
import scipy.optimize as opt

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill

    n, m, p = 2, 3, 6

    def schoolbook():
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

    best_result = schoolbook()
    best_rank = n * m * p

    def verify_full(U, V, W, n_s, m_s, p_s):
        r = U.shape[1]
        for i in range(n_s):
            for j in range(m_s):
                for k in range(p_s):
                    val = 0.0
                    for c in range(r):
                        val += U[i * m_s + j, c] * V[j * p_s + k, c] * W[k * n_s + i, c]
                    if abs(val - 1.0) > 1e-8:
                        return False
        return True

    def compute_residuals(U, V, W, n_s, m_s, p_s):
        """Returns vector of residuals for all (i,j,k) terms."""
        r = U.shape[1]
        residuals = np.zeros(n_s * m_s * p_s)
        idx = 0
        for i in range(n_s):
            for j in range(m_s):
                for k in range(p_s):
                    val = 0.0
                    for c in range(r):
                        val += U[i * m_s + j, c] * V[j * p_s + k, c] * W[k * n_s + i, c]
                    residuals[idx] = val - 1.0
                    idx += 1
        return residuals

    def try_snap_and_verify(U, V, W, rank):
        """
        Try to snap the real-valued solution to half-integers (x.0 or x.5) 
        and verify if it's an exact decomposition.
        """
        # Snap to nearest 0.5
        U_snap = np.round(U * 2) / 2
        V_snap = np.round(V * 2) / 2
        W_snap = np.round(W * 2) / 2
        
        # Verify
        if verify_full(U_snap, V_snap, W_snap, n, m, p):
            return U_snap, V_snap, W_snap
        
        # Try snapping to integers (in case half-integers are not needed)
        U_int = np.round(U)
        V_int = np.round(V)
        W_int = np.round(W)
        if verify_full(U_int, V_int, W_int, n, m, p):
            return U_int, V_int, W_int
            
        return None

    def solve_gauss_newton(rank, seed=42, max_nfev=500, tol=1e-12):
        """
        Attempts to find a rank-r decomposition using Gauss-Newton.
        Returns (U, V, W, error) or None if failed.
        """
        # Parameter vector: flatten U, V, W
        size_U = n * m * rank
        size_V = m * p * rank
        size_W = p * n * rank
        total_size = size_U + size_V + size_W
        
        # Random initialization
        np.random.seed(seed)
        x0 = np.random.randn(total_size)
        
        # Split x0 into U, V, W
        U0 = x0[:size_U].reshape(n * m, rank)
        V0 = x0[size_U:size_U + size_V].reshape(m * p, rank)
        W0 = x0[size_U + size_V:].reshape(p * n, rank)
        
        def fun(x):
            U = x[:size_U].reshape(n * m, rank)
            V = x[size_U:size_U + size_V].reshape(m * p, rank)
            W = x[size_U + size_V:].reshape(p * n, rank)
            return compute_residuals(U, V, W, n, m, p)
            
        try:
            res = opt.least_squares(fun, x0, method='lm', max_nfev=max_nfev)
            x_opt = res.x
            err = np.sum(res.fun**2)
            
            if err < tol:
                U_opt = x_opt[:size_U].reshape(n * m, rank)
                V_opt = x_opt[size_U:size_U + size_V].reshape(m * p, rank)
                W_opt = x_opt[size_U + size_V:].reshape(p * n, rank)
                return U_opt, V_opt, W_opt, err
            else:
                # Check if it's "close" enough to try snapping
                if err < 1e-4:
                    U_opt = x_opt[:size_U].reshape(n * m, rank)
                    V_opt = x_opt[size_U:size_U + size_V].reshape(m * p, rank)
                    W_opt = x_opt[size_U + size_V:].reshape(p * n, rank)
                    return U_opt, V_opt, W_opt, err
        except Exception:
            pass
        return None

    # Search for lower ranks
    # Known rank for 2x3x6 is 20. We try from 20 upwards.
    # We also try a few lower ranks just in case (though 20 is the known min).
    for rank in range(20, 36):
        if time.time() > deadline - 10:
            break
            
        # Try multiple random seeds for robustness
        found = False
        for seed in range(5):
            if time.time() > deadline - 5:
                break
                
            result = solve_gauss_newton(rank, seed=42 + rank + seed)
            if result:
                U_sol, V_sol, W_sol, err = result
                snapped = try_snap_and_verify(U_sol, V_sol, W_sol, rank)
                if snapped:
                    U_final, V_final, W_final = snapped
                    if rank < best_rank:
                        best_rank = rank
                        best_result = {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
                        found = True
                        # If we found rank 20, we can stop or continue to see if 19 exists (unlikely)
                        if rank == 20:
                            break
        if found and rank == 20:
            break

    return best_result
# EVOLVE-BLOCK-END
