import time
import numpy as np
import random
import sys

# Try to import scipy for optimization
try:
    from scipy.optimize import minimize
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Precompute valid indices
    valid_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                valid_indices.append((i*m+j, j*p+k, k*n+i))
                
    # Build target tensor T
    T = np.zeros((nm, mp, pn), dtype=float)
    for a, b, c in valid_indices:
        T[a, b, c] = 1.0

    def verify(U, V, W):
        """
        Verify if U, V, W exactly decompose the tensor.
        """
        T_rec = np.einsum('ar,br,cr->abc', U, V, W)
        # Check valid entries
        for a, b, c in valid_indices:
            if abs(T_rec[a, b, c] - 1.0) > 1e-9:
                return False
        # Check invalid entries
        mask = np.zeros((nm, mp, pn), dtype=bool)
        for a, b, c in valid_indices:
            mask[a, b, c] = True
        invalid_vals = T_rec[~mask]
        if np.any(np.abs(invalid_vals) > 1e-9):
            return False
        return True

    def to_half_integers(U, V, W):
        """
        Snap to nearest half-integers and verify.
        """
        U2 = np.round(U * 2) / 2
        V2 = np.round(V * 2) / 2
        W2 = np.round(W * 2) / 2
        if verify(U2, V2, W2):
            return U2, V2, W2
        return None

    # Schoolbook fallback
    rank_fb = n * m * p
    U_fb = np.zeros((nm, rank_fb))
    V_fb = np.zeros((mp, rank_fb))
    W_fb = np.zeros((pn, rank_fb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j, col] = 1
                V_fb[j*p+k, col] = 1
                W_fb[k*n+i, col] = 1
                col += 1
                
    best_result = (U_fb, V_fb, W_fb)
    best_rank = rank_fb
    
    deadline = time.time() + 115  # Safety margin
    
    if not HAS_SCIPY:
        # Fallback to simple random search if scipy is missing (unlikely but safe)
        val_set = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
        for rank in range(12, best_rank):
            if time.time() > deadline:
                break
            for _ in range(500):
                if time.time() > deadline:
                    break
                U_try = np.random.choice(val_set, size=(nm, rank))
                V_try = np.random.choice(val_set, size=(mp, rank))
                W_try = np.random.choice(val_set, size=(pn, rank))
                if verify(U_try, V_try, W_try):
                    best_result = (U_try, V_try, W_try)
                    best_rank = rank
                    break
            if best_rank == rank:
                break
    else:
        # Use SciPy optimization
        def objective(params, rank):
            """
            params: flat array of length (nm + mp + pn) * rank
            Returns squared error.
            """
            U = params[:nm*rank].reshape(nm, rank)
            V = params[nm*rank:(nm+mp)*rank].reshape(mp, rank)
            W = params[(nm+mp)*rank:].reshape(pn, rank)
            
            T_rec = np.einsum('ar,br,cr->abc', U, V, W)
            error = T_rec - T
            return np.sum(error * error)

        def gradient(params, rank):
            """
            Compute gradient of the objective.
            """
            U = params[:nm*rank].reshape(nm, rank)
            V = params[nm*rank:(nm+mp)*rank].reshape(mp, rank)
            W = params[(nm+mp)*rank:].reshape(pn, rank)
            
            T_rec = np.einsum('ar,br,cr->abc', U, V, W)
            error = T_rec - T
            
            # Gradients:
            # dL/dU[a,r] = 2 * sum_{b,c} error[a,b,c] * V[b,r] * W[c,r]
            # dL/dV[b,r] = 2 * sum_{a,c} error[a,b,c] * U[a,r] * W[c,r]
            # dL/dW[c,r] = 2 * sum_{a,b} error[a,b,c] * U[a,r] * V[b,r]
            
            grad_U = 2 * np.einsum('abc,br,cr->ar', error, V, W)
            grad_V = 2 * np.einsum('abc,ar,cr->br', error, U, W)
            grad_W = 2 * np.einsum('abc,ar,br->cr', error, U, V)
            
            return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])

        # Search from lower bound upwards
        min_rank = max(nm, mp, pn)
        
        for rank in range(min_rank, best_rank):
            if time.time() > deadline:
                break
            
            # Try a few random initializations for this rank
            max_trials = 10
            for trial in range(max_trials):
                if time.time() > deadline:
                    break
                
                # Initialize with small random values
                np.random.seed(42 + trial + rank) # Deterministic for reproducibility in tests, but varied per trial
                params0 = np.random.uniform(-1, 1, size=(nm + mp + pn) * rank)
                
                # Run optimization
                try:
                    res = minimize(objective, params0, args=(rank,), 
                                   method='L-BFGS-B', 
                                   jac=gradient, 
                                   options={'maxiter': 50, 'ftol': 1e-12})
                    
                    if res.fun < 1e-8:
                        # Extract U, V, W
                        U_opt = res.x[:nm*rank].reshape(nm, rank)
                        V_opt = res.x[nm*rank:(nm+mp)*rank].reshape(mp, rank)
                        W_opt = res.x[(nm+mp)*rank:].reshape(pn, rank)
                        
                        # Snap to half-integers
                        snapped = to_half_integers(U_opt, V_opt, W_opt)
                        if snapped is not None:
                            U_s, V_s, W_s = snapped
                            if verify(U_s, V_s, W_s):
                                best_result = (U_s, V_s, W_s)
                                best_rank = rank
                                break # Found solution for this rank, move to next (lower) rank? No, we are going up.
                                # Wait, we want minimal rank. We start from min_rank.
                                # If we find one at rank, we break out of the trial loop.
                                # Should we break out of the rank loop? Yes, if we are going from low to high.
                                # But maybe a lower rank exists? No, we start from the theoretical lower bound.
                                # So the first valid rank we find is the minimal one (assuming we search sequentially from min_rank).
                                # However, it's possible we missed a solution at min_rank due to local minima.
                                # So we should probably allow a bit more time for lower ranks or just return the first found?
                                # Given the small size, finding rank 12 or 13 is the goal.
                                # Let's break and return immediately to save time, as lower is better.
                                return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
                
                except Exception:
                    pass
            
            # If we didn't find a solution for this rank, continue to next rank

    U_out, V_out, W_out = best_result
    return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}
# EVOLVE-BLOCK-END
