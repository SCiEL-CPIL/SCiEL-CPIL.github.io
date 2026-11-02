"""Rewritten program for matrix multiplication 3,3,7."""
import numpy as np
import time

DIMENSIONS = [3, 3, 7]
DEADLINE = 140.0  # safety margin before 150s kill


def _trivial_decomposition(n, m, p):
    """Return the trivial rank-n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx, col] = 1.0
                V[v_idx, col] = 1.0
                W[w_idx, col] = 1.0
                col += 1
    return U, V, W


def _validate(U, V, W, n, m, p):
    """Check if the decomposition is exact."""
    rank = U.shape[1]
    if rank != V.shape[1] or rank != W.shape[1]:
        return False
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    
    # Check all (i,j,k) tuples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = np.dot(U[u_idx, :], np.multiply(V[v_idx, :], W[w_idx, :]))
                if abs(val - 1.0) > 1e-9:
                    return False
    return True


def _validate_tensor(T_calc, T_target, tol=1e-9):
    """Check if the computed tensor matches the target tensor."""
    diff = T_calc - T_target
    return np.linalg.norm(diff) < tol


def _try_rank_reduction(n, m, p, base_rank, deadline):
    """
    Try to find a lower-rank decomposition using optimization.
    """
    start_time = time.time()
    
    num_u = n * m
    num_v = m * p
    num_w = p * n
    
    # Build the target tensor T
    # T[u, v, w] = 1 if consistent, else 0
    T = np.zeros((num_u, num_v, num_w))
    for u in range(num_u):
        i = u // m
        j = u % m
        for v in range(num_v):
            j_v = v // p
            k = v % p
            if j != j_v:
                continue
            for w in range(num_w):
                k_w = w // n
                i_w = w % n
                if k == k_w and i == i_w:
                    T[u, v, w] = 1.0
    
    T_flat = T.flatten()
    
    # Lower bound for rank
    lower_bound = max(n * m, m * p, p * n)
    
    # Try ranks from lower_bound to base_rank
    # We only try a few ranks to stay within time limits
    ranks_to_try = list(range(lower_bound, base_rank + 1, 5)) # Step by 5
    if base_rank not in ranks_to_try:
        ranks_to_try.append(base_rank)
        
    for R in ranks_to_try:
        if time.time() > deadline:
            break
        
        # Try multiple restarts
        max_restarts = 3
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            np.random.seed(42 + R * 1000 + restart)
            U_init = np.random.randn(num_u, R)
            V_init = np.random.randn(num_v, R)
            W_init = np.random.randn(num_w, R)
            
            # Normalize
            U_init /= np.linalg.norm(U_init) + 1e-10
            V_init /= np.linalg.norm(V_init) + 1e-10
            W_init /= np.linalg.norm(W_init) + 1e-10
            
            # Use scipy optimization
            try:
                from scipy.optimize import minimize
                
                params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
                
                def objective(params):
                    u_size = num_u * R
                    v_size = num_v * R
                    w_size = num_w * R
                    
                    U_flat = params[:u_size]
                    V_flat = params[u_size:u_size + v_size]
                    W_flat = params[u_size + v_size:u_size + v_size + w_size]
                    
                    U = U_flat.reshape(num_u, R)
                    V = V_flat.reshape(num_v, R)
                    W = W_flat.reshape(num_w, R)
                    
                    T_calc = np.einsum('ur, vr, wr -> uvw', U, V, W)
                    diff = T_calc.flatten() - T_flat
                    return np.dot(diff, diff)
                
                # Run optimization with time limit
                res = minimize(
                    objective, 
                    params_init, 
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                
                if res.fun < 1e-8: # Stricter tolerance
                    # Check if we have a valid solution
                    U_opt = res.x[:num_u * R].reshape(num_u, R)
                    V_opt = res.x[num_u * R:num_u * R + num_v * R].reshape(num_v, R)
                    W_opt = res.x[num_u * R + num_v * R:].reshape(num_w, R)
                    
                    T_calc = np.einsum('ur, vr, wr -> uvw', U_opt, V_opt, W_opt)
                    if _validate_tensor(T_calc, T, tol=1e-8):
                        # Try to round to half-integers
                        U_round = np.round(U_opt * 2) / 2
                        V_round = np.round(V_opt * 2) / 2
                        W_round = np.round(W_opt * 2) / 2
                        
                        T_calc_round = np.einsum('ur, vr, wr -> uvw', U_round, V_round, W_round)
                        if _validate_tensor(T_calc_round, T, tol=1e-8):
                            return U_round, V_round, W_round, R
                        else:
                            # Return the unrounded solution if valid
                            return U_opt, V_opt, W_opt, R
            except Exception:
                continue
    
    # If no better solution found, return trivial
    U, V, W = _trivial_decomposition(n, m, p)
    return U, V, W, base_rank


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Start with the trivial decomposition
    U, V, W = _trivial_decomposition(n, m, p)
    best_rank = n * m * p
    
    # Validate the trivial decomposition
    if _validate(U, V, W, n, m, p):
        # Try to reduce rank (heuristic, may not succeed)
        U_red, V_red, W_red, reduced_rank = _try_rank_reduction(n, m, p, best_rank, DEADLINE)
        if reduced_rank < best_rank and _validate(U_red, V_red, W_red, n, m, p):
            U, V, W = U_red, V_red, W_red
            best_rank = reduced_rank
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        pass  # Return best found
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
