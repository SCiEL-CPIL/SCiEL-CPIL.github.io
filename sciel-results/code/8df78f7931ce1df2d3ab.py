"""Combined program: structural rank-32 base with smooth local search for rank-31."""
import time
import numpy as np

DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS  # 2, 2, 9
    nm, mp, pn = n*m, m*p, p*n
    total = n*m*p  # 36

    # Build the rank-32 base decomposition using Strassen blocks
    rank_base = 32
    U = np.zeros((nm, rank_base))
    V = np.zeros((mp, rank_base))
    W = np.zeros((pn, rank_base))

    col = 0

    def add_strassen(k0, k1):
        nonlocal col
        U[0, col] = 1; U[3, col] = 1
        V[0*p + k0, col] = 1; V[1*p + k1, col] = 1
        W[k0*n + 0, col] = 1; W[k1*n + 1, col] = 1
        col += 1
        U[2, col] = 1; U[3, col] = 1
        V[0*p + k0, col] = 1
        W[k0*n + 1, col] = 1; W[k1*n + 1, col] = -1
        col += 1
        U[0, col] = 1
        V[0*p + k1, col] = 1; V[1*p + k1, col] = -1
        W[k1*n + 0, col] = 1; W[k1*n + 1, col] = 1
        col += 1
        U[3, col] = 1
        V[1*p + k0, col] = 1; V[0*p + k0, col] = -1
        W[k0*n + 0, col] = 1; W[k0*n + 1, col] = 1
        col += 1
        U[0, col] = 1; U[1, col] = 1
        V[1*p + k1, col] = 1
        W[k0*n + 0, col] = -1; W[k1*n + 0, col] = 1
        col += 1
        U[2, col] = 1; U[0, col] = -1
        V[0*p + k0, col] = 1; V[0*p + k1, col] = 1
        W[k1*n + 1, col] = 1
        col += 1
        U[1, col] = 1; U[3, col] = -1
        V[1*p + k0, col] = 1; V[1*p + k1, col] = 1
        W[k0*n + 0, col] = 1
        col += 1

    def add_trivial(k):
        nonlocal col
        U[0, col] = 1; V[0*p + k, col] = 1; W[k*n + 0, col] = 1; col += 1
        U[1, col] = 1; V[1*p + k, col] = 1; W[k*n + 0, col] = 1; col += 1
        U[2, col] = 1; V[0*p + k, col] = 1; W[k*n + 1, col] = 1; col += 1
        U[3, col] = 1; V[1*p + k, col] = 1; W[k*n + 1, col] = 1; col += 1

    add_strassen(0, 1)
    add_strassen(2, 3)
    add_strassen(4, 5)
    add_trivial(6)
    add_strassen(7, 8)

    def verify(U_arr, V_arr, W_arr, tol=1e-9):
        T = np.einsum('ir,jr,kr->ijk', U_arr, V_arr, W_arr)
        return np.allclose(T, 1.0, atol=tol)

    def try_rank31(drop_idx, U0, V0, W0, max_iters=2000, tol=1e-10):
        """Try to find a rank-31 decomposition by dropping term drop_idx and adjusting."""
        if time.time() > deadline - 5:
            return None
        
        keep = [i for i in range(rank_base) if i != drop_idx]
        R = len(keep)
        
        U = U0[:, keep].copy()
        V = V0[:, keep].copy()
        W = W0[:, keep].copy()
        
        # Objective: minimize max |T[i,j,k] - 1|
        # Use smooth surrogate: log-sum-exp of squared residuals
        def objective(params):
            """params: flattened [U, V, W]"""
            U_flat = params[:nm*R]
            V_flat = params[nm*R:nm*R+mp*R]
            W_flat = params[nm*R+mp*R:]
            
            U_mat = U_flat.reshape(nm, R)
            V_mat = V_flat.reshape(mp, R)
            W_mat = W_flat.reshape(pn, R)
            
            T = np.einsum('ir,jr,kr->ijk', U_mat, V_mat, W_mat)
            residuals = (T - 1.0) ** 2
            # Smooth max via log-sum-exp
            max_res = np.max(residuals)
            if max_res < 1e-30:
                return 0.0
            # Scale for numerical stability
            scaled = residuals / max(1e-15, max_res)
            log_sum_exp = max_res + np.log(np.sum(np.exp(scaled - 1.0)))
            return log_sum_exp
        
        def gradient(params):
            """Compute gradient of the smooth objective."""
            U_flat = params[:nm*R]
            V_flat = params[nm*R:nm*R+mp*R]
            W_flat = params[nm*R+mp*R:]
            
            U_mat = U_flat.reshape(nm, R)
            V_mat = V_flat.reshape(mp, R)
            W_mat = W_flat.reshape(pn, R)
            
            T = np.einsum('ir,jr,kr->ijk', U_mat, V_mat, W_mat)
            residuals = (T - 1.0) ** 2
            max_res = np.max(residuals)
            if max_res < 1e-30:
                return np.zeros_like(params)
            
            scaled = residuals / max(1e-15, max_res)
            weights = np.exp(scaled - 1.0) / np.sum(np.exp(scaled - 1.0))
            # d/dT of smooth max ≈ 2*(T-1) * weights
            dT = 2.0 * (T - 1.0) * weights
            
            # dU = einsum('ijk,kr,jr->ir', dT, W_mat, V_mat)
            dU = np.einsum('ijk,kr,jr->ir', dT, W_mat, V_mat)
            dV = np.einsum('ijk,ir,kr->jr', dT, U_mat, W_mat)
            dW = np.einsum('ijk,ir,jr->kr', dT, U_mat, V_mat)
            
            return np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
        
        # Start from current solution
        params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        # Use L-BFGS-B from scipy
        from scipy.optimize import minimize
        
        # Set up bounds: allow values in [-100, 100]
        bounds = [(-100, 100)] * len(params)
        
        best_params = params.copy()
        best_obj = objective(params)
        
        # Multiple restarts with small perturbations
        rng = np.random.RandomState(42)
        for restart in range(3):
            if time.time() > deadline - 5:
                break
            
            if restart == 0:
                p = params.copy()
            else:
                p = params + rng.normal(0, 0.1, size=params.shape)
            
            result = minimize(
                objective, p, jac=gradient, method='L-BFGS-B',
                bounds=bounds, options={'maxiter': max_iters, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            
            if result.fun < best_obj:
                best_obj = result.fun
                best_params = result.x
        
        # Check if we achieved a valid decomposition
        U_final = best_params[:nm*R].reshape(nm, R)
        V_final = best_params[nm*R:nm*R+mp*R].reshape(mp, R)
        W_final = best_params[nm*R+mp*R:].reshape(pn, R)
        
        if verify(U_final, V_final, W_final, tol=1e-8):
            # Convert to list format
            U_list = U_final.tolist()
            V_list = V_final.tolist()
            W_list = W_final.tolist()
            return {"U": U_list, "V": V_list, "W": W_list}
        
        return None

    # Try dropping each term
    for drop_idx in range(rank_base):
        if time.time() > deadline - 10:
            break
        
        result = try_rank31(drop_idx, U, V, W)
        if result is not None:
            return result

    # Return the base decomposition
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
