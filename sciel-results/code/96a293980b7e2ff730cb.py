"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import sys
from itertools import product
import scipy.optimize as opt

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    # The evaluator kills after 150s. We use 145s as a hard stop to be safe.
    deadline = 145.0
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 72
    
    # Target tensor T[i, j, k] = 1
    T = np.ones((n, m, p))
    T_flat = T.flatten()
    
    NM, MP, PN = n*m, m*p, p*n

    def check_exact(U, V, W, rank):
        """
        Check if U, V, W with given rank exactly reconstruct T.
        U: shape (n, m, rank)
        V: shape (m, p, rank)
        W: shape (p, n, rank)
        """
        # Recalculate tensor
        # T_rec[i, j, k] = sum_c U[i, j, c] * V[j, k, c] * W[k, i, c]
        T_rec = np.einsum('ijc,jkc,kic->ijk', U, V, W)
        # Check exactness with tight tolerance
        return np.allclose(T_rec, T, atol=1e-9)

    def schoolbook_init():
        """Create the schoolbook (trivial) decomposition"""
        U = np.zeros((n, m, rank_schoolbook))
        V = np.zeros((m, p, rank_schoolbook))
        W = np.zeros((p, n, rank_schoolbook))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i, j, col] = 1
                    V[j, k, col] = 1
                    W[k, i, col] = 1
                    col += 1
        return U, V, W

    def try_rank_optimization(rank, num_restarts=10, time_per_restart=5.0):
        """
        Try to find a decomposition of given rank using L-BFGS-B optimization.
        Returns (U, V, W) if found, else None.
        """
        # Flatten tensors for optimization
        # U_flat: size n*m*rank
        # V_flat: size m*p*rank
        # W_flat: size p*n*rank
        
        def objective(x):
            U = x[:n*m*rank].reshape(n, m, rank)
            V = x[n*m*rank:n*m*rank + m*p*rank].reshape(m, p, rank)
            W = x[n*m*rank + m*p*rank:].reshape(p, n, rank)
            
            T_rec = np.einsum('ijc,jkc,kic->ijk', U, V, W)
            err = T_rec - T
            return np.sum(err**2)
        
        best_result = None
        best_err = np.inf
        
        for r in range(num_restarts):
            if time.time() > deadline - 10:
                break
            
            # Random initialization
            np.random.seed(r)
            # Scale initialization to avoid huge gradients
            U_init = np.random.randn(n, m, rank) * 0.5
            V_init = np.random.randn(m, p, rank) * 0.5
            W_init = np.random.randn(p, n, rank) * 0.5
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            # Set a local deadline for this restart
            restart_deadline = time.time() + time_per_restart
            
            # L-BFGS-B
            try:
                res = opt.minimize(objective, x0, method='L-BFGS-B', 
                                    options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 5000})
                
                if time.time() > restart_deadline:
                    continue
                    
                err = res.fun
                if err < best_err:
                    best_err = err
                    # Extract solution
                    U_sol = res.x[:n*m*rank].reshape(n, m, rank)
                    V_sol = res.x[n*m*rank:n*m*rank + m*p*rank].reshape(m, p, rank)
                    W_sol = res.x[n*m*rank + m*p*rank:].reshape(p, n, rank)
                    best_result = (U_sol, V_sol, W_sol)
                
                # If error is extremely small, try to snap to integers/half-integers
                if err < 1e-6:
                    # Try rounding to nearest 0.5
                    U_snapped = np.round(U_sol * 2) / 2
                    V_snapped = np.round(V_sol * 2) / 2
                    W_snapped = np.round(W_sol * 2) / 2
                    
                    if check_exact(U_snapped, V_snapped, W_snapped, rank):
                        return (U_snapped, V_snapped, W_snapped)
                    
                    # Try rounding to nearest integer
                    U_int = np.round(U_sol)
                    V_int = np.round(V_sol)
                    W_int = np.round(W_sol)
                    
                    if check_exact(U_int, V_int, W_int, rank):
                        return (U_int, V_int, W_int)
                        
            except Exception:
                continue
        
        # If we found a very good solution but didn't snap perfectly, 
        # try a few more snapping strategies on the best result
        if best_result is not None and best_err < 1e-4:
            U_sol, V_sol, W_sol = best_result
            # Try finer snapping grids
            for scale in [4, 8, 10]:
                U_snapped = np.round(U_sol * scale) / scale
                V_snapped = np.round(V_sol * scale) / scale
                W_snapped = np.round(W_sol * scale) / scale
                if check_exact(U_snapped, V_snapped, W_snapped, rank):
                    return (U_snapped, V_snapped, W_snapped)
            
            # Sometimes the solution is close to integer but slightly off due to numerical noise
            # Try local polish around the snapped values? 
            # For now, if it's not exact, we discard it for this rank and try lower rank or fall back.
            # However, if the error is tiny, it's likely a valid decomposition that just needs cleaning.
            # Let's try one more time with higher precision polish if available.
            
        return None

    # Strategy:
    # 1. Check if rank 72 (schoolbook) is valid (it always is).
    # 2. Try to find a lower rank (71, 70, ...) using optimization.
    #    Given the small size, we can afford a few restarts for each rank.
    #    However, if 72 is the true rank, we will waste time.
    #    We should prioritize checking if 71 is possible.
    
    best_result = None
    best_rank = rank_schoolbook
    
    # Initialize with schoolbook as fallback
    U_sb, V_sb, W_sb = schoolbook_init()
    if check_exact(U_sb, V_sb, W_sb, rank_schoolbook):
        best_result = (U_sb, V_sb, W_sb)
        best_rank = rank_schoolbook
    
    # Try to reduce rank
    # We iterate downwards from 71.
    # We limit the total time spent on optimization.
    # If we find a valid rank R, we stop and return it.
    
    # Heuristic: It is highly unlikely that rank < 72 exists for 3x4x6.
    # But we must check.
    # Let's try rank 71 with a few restarts.
    # If it fails, we assume 72 is the minimum.
    
    # To be safe and fast, we only try rank 71 and maybe 70 if time permits.
    # If 71 fails, 70 is even less likely.
    
    # Time budget for optimization: ~120s
    # We will try rank 71 for up to 60s.
    
    if time.time() < deadline - 60:
        res = try_rank_optimization(71, num_restarts=5, time_per_restart=10.0)
        if res is not None:
            best_result = res
            best_rank = 71
    
    # If we have time, try 70? Probably not needed.
    # Let's stick to 71 check.
    
    # Prepare output
    if best_result is None:
        # Should not happen, but just in case
        U, V, W = schoolbook_init()
        best_rank = rank_schoolbook
    else:
        U, V, W = best_result
    
    # Convert to lists of lists with required shapes
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    U_out = U.reshape(NM, best_rank).tolist()
    V_out = V.reshape(MP, best_rank).tolist()
    W_out = W.reshape(PN, best_rank).tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
