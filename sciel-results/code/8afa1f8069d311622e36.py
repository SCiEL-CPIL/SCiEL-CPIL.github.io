"""Matrix multiplication 3x4x5 - explicit rank-minimal decomposition attempt."""
DIMENSIONS = [3, 4, 5]

import time
import numpy as np
import scipy.optimize as opt

# EVOLVE-BLOCK-START

def _build_expected_tensor(n, m, p):
    """Build the expected matmul tensor T[x,y,z] where:
    x = i*m+j (A index), y = j*p+k (B index), z = k*n+i (C index, column-major)
    T[x,y,z] = 1 if x=i*m+j, y=j*p+k, z=k*n+i for some consistent (i,j,k), else 0.
    """
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _compute_decomposed_tensor(U, V, W):
    """Compute D[x,y,z] = sum_r U[x,r] * V[y,r] * W[z,r]"""
    return np.einsum('xr,yr,zr->xyz', U, V, W)

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly reproduces the matmul tensor."""
    expected = _build_expected_tensor(n, m, p)
    D = _compute_decomposed_tensor(U, V, W)
    return np.allclose(D, expected, atol=tol)

def _round_to_half_integer(x):
    """Round a value to the nearest half-integer."""
    return np.round(x * 2) / 2

def _try_rank_optimized(n, m, p, rank, num_trials=10, time_limit=5.0, seed=None):
    """Try to find a valid rank-r decomposition via optimization + rounding.
    
    Returns (U, V, W) if found, else None.
    """
    start_time = time.time()
    rng = np.random.default_rng(seed)
    T = _build_expected_tensor(n, m, p)
    
    # Target tensor for integer search: 8*T if we scale by 2
    # But we will optimize on floats and then round to half-integers.
    
    for trial in range(num_trials):
        if time.time() - start_time > time_limit:
            break
        
        # Random initialization with small values
        # Scale is important. If rank is close to schoolbook, values are likely 0, 1, -1.
        # Let's initialize with small random integers or half-integers.
        U = rng.integers(-2, 3, size=(n*m, rank)) / 2.0
        V = rng.integers(-2, 3, size=(m*p, rank)) / 2.0
        W = rng.integers(-2, 3, size=(p*n, rank)) / 2.0
        
        # Flatten for optimization
        U_flat = U.flatten()
        V_flat = V.flatten()
        W_flat = W.flatten()
        
        # Objective function: Minimize || D - T ||_F^2
        def objective(params):
            if time.time() - start_time > time_limit:
                return 1e10 # Timeout penalty
            U_curr = params[:n*m*rank].reshape(n*m, rank)
            V_curr = params[n*m*rank : n*m*rank + m*p*rank].reshape(m*p, rank)
            W_curr = params[n*m*rank + m*p*rank : ].reshape(p*n, rank)
            
            D = _compute_decomposed_tensor(U_curr, V_curr, W_curr)
            error = np.sum((D - T)**2)
            return error
        
        # Initial error
        init_err = objective(np.concatenate([U_flat, V_flat, W_flat]))
        if init_err < 1e-9:
            return U, V, W
            
        # Optimize
        # L-BFGS-B is good for this size
        try:
            result = opt.minimize(
                objective, 
                np.concatenate([U_flat, V_flat, W_flat]), 
                method='L-BFGS-B', 
                options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-12, 'maxfun': 100}
            )
        except Exception:
            continue
            
        # Extract optimized parameters
        opt_params = result.x
        U_opt = opt_params[:n*m*rank].reshape(n*m, rank)
        V_opt = opt_params[n*m*rank : n*m*rank + m*p*rank].reshape(m*p, rank)
        W_opt = opt_params[n*m*rank + m*p*rank : ].reshape(p*n, rank)
        
        # Check if the optimized solution is close to exact
        D_opt = _compute_decomposed_tensor(U_opt, V_opt, W_opt)
        err_opt = np.max(np.abs(D_opt - T))
        
        if err_opt < 1e-6:
            # Try rounding to half-integers
            U_rounded = _round_to_half_integer(U_opt)
            V_rounded = _round_to_half_integer(V_opt)
            W_rounded = _round_to_half_integer(W_opt)
            
            if _verify_decomposition(U_rounded, V_rounded, W_rounded, n, m, p):
                return U_rounded, V_rounded, W_rounded
                
            # Also try rounding to integers (subset of half-integers)
            U_int = np.round(U_opt)
            V_int = np.round(V_opt)
            W_int = np.round(W_opt)
            if _verify_decomposition(U_int, V_int, W_int, n, m, p):
                return U_int, V_int, W_int

    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 60
    
    # Build schoolbook decomposition as fallback
    U_school = np.zeros((n * m, schoolbook_rank))
    V_school = np.zeros((m * p, schoolbook_rank))
    W_school = np.zeros((p * n, schoolbook_rank))
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, column] = 1
                V_school[j * p + k, column] = 1
                W_school[k * n + i, column] = 1
                column += 1
    
    # Verify schoolbook (sanity check)
    if not _verify_decomposition(U_school, V_school, W_school, n, m, p):
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    best_result = (schoolbook_rank, U_school, V_school, W_school)
    
    # Try to find lower-rank decompositions
    # We will try ranks from schoolbook-1 down to a reasonable limit.
    # Given the computational cost, we limit the number of ranks and time per rank.
    remaining_time = deadline - time.time()
    
    # Heuristic: If rank < 60 exists, it's likely close to 60.
    # We'll try 59, 58, 57 with a tight budget.
    ranks_to_try = [59, 58, 57]
    
    for rank in ranks_to_try:
        if time.time() > deadline - 10:  # 10s safety margin
            break
        
        # Allocate time based on remaining ranks
        time_per_rank = max(1.0, (deadline - time.time() - 10) / len(ranks_to_try))
        
        result = _try_rank_optimized(n, m, p, rank, num_trials=5, time_limit=time_per_rank, seed=rank)
        
        if result is not None:
            U, V, W = result
            if _verify_decomposition(U, V, W, n, m, p):
                best_result = (rank, U, V, W)
                break  # Found a valid lower-rank decomposition, stop searching
    
    rank, U, V, W = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
