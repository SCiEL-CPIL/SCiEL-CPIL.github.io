"""Program for matrix multiplication tensor decomposition: 2x4 * 4x5."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START

def _verify_exact(U, V, W, n, m, p):
    """Check if the decomposition is exact and entries are half-integers."""
    # Check half-integers
    for mat in [U, V, W]:
        for row in mat:
            for val in row:
                # val should be integer or half-integer
                # i.e., 2*val should be integer
                if abs(val * 2 - round(val * 2)) > 1e-6:
                    return False
    
    # Check tensor reconstruction
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    rank = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    
    U3 = Ua.reshape(n, m, rank)
    V3 = Va.reshape(m, p, rank)
    W3 = Wa.reshape(p, n, rank)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U3[i, j, :] * V3[j, k, :] * W3[k, i, :])
                if abs(val - 1.0) > 1e-6:
                    return False
    return True

def _objective(params, n, m, p, r):
    """
    Objective function for optimization.
    params: flattened array of U, V, W.
    Returns: (error, gradient)
    """
    size_U = n * m * r
    size_V = m * p * r
    size_W = p * n * r
    
    U = params[:size_U].reshape(n, m, r)
    V = params[size_U:size_U+size_V].reshape(m, p, r)
    W = params[size_U+size_V:size_U+size_V+size_W].reshape(p, n, r)
    
    # Compute T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
    # We want T[i,j,k] = 1
    
    error = 0.0
    # Penalty for non-half-integers
    penalty = 0.0
    
    # We can vectorize the error calculation partially
    # But for gradient, we need to be careful.
    # Let's compute the error sum of squares
    # T[i,j,k] = sum_c U[i,j,c] * (V[j,k,c] * W[k,i,c])
    
    # Pre-compute V*W terms?
    # V[j,k,c] * W[k,i,c] -> shape (m, p, n, r)
    # This is m*p*n*r = 4*5*2*20 = 800 operations per rank step? No, r is variable.
    # 4*5*2*20 = 800. Very fast.
    
    # VW[k,i,j,c] is not quite right.
    # Let's loop over i, j, k for clarity and speed is sufficient for r<40
    err_sq = 0.0
    pen = 0.0
    
    # Gradient accumulation
    grad_U = np.zeros_like(U)
    grad_V = np.zeros_like(V)
    grad_W = np.zeros_like(W)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # val = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
                prod_VW = V[j, k, :] * W[k, i, :]
                val = np.dot(U[i, j, :], prod_VW)
                
                err = val - 1.0
                err_sq += err * err
                
                # Gradients
                # d(err^2)/dU[i,j,c] = 2 * err * V[j,k,c] * W[k,i,c]
                grad_U[i, j, :] += 2 * err * prod_VW
                
                # d(err^2)/dV[j,k,c] = 2 * err * U[i,j,c] * W[k,i,c]
                grad_V[j, k, :] += 2 * err * (U[i, j, :] * W[k, i, :])
                
                # d(err^2)/dW[k,i,c] = 2 * err * U[i,j,c] * V[j,k,c]
                grad_W[k, i, :] += 2 * err * (U[i, j, :] * V[j, k, :])
    
    # Add penalty for half-integers
    # P(x) = (2x - round(2x))^2
    # This is non-differentiable at half-integers, but we can use a smooth approximation
    # or just add a small L2 penalty towards 0 to keep numbers small, 
    # and rely on the structure to snap.
    # Actually, let's add a penalty that pushes towards integers/half-integers.
    # A smooth proxy: sin^2(2 * pi * x) is 0 at half-integers? No, sin(2*pi*x) is 0 at integers.
    # sin^2(pi * x) is 0 at integers.
    # We want 0 at integers and half-integers.
    # sin^2(2 * pi * x) is 0 at integers.
    # Let's use a penalty that encourages small values, as solutions are often sparse/small.
    # L2 penalty: lambda * (sum U^2 + sum V^2 + sum W^2)
    lambda_reg = 0.01
    pen = lambda_reg * (np.sum(U**2) + np.sum(V**2) + np.sum(W**2))
    
    grad_U += 2 * lambda_reg * U
    grad_V += 2 * lambda_reg * V
    grad_W += 2 * lambda_reg * W
    
    total_error = err_sq + pen
    grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
    
    return total_error, grad

def _try_rank(r, n, m, p, max_restarts=5, time_limit=10.0):
    """Try to find a decomposition of rank r."""
    start_time = time.time()
    size_U = n * m * r
    size_V = m * p * r
    size_W = p * n * r
    total_size = size_U + size_V + size_W
    
    np.random.seed(42) # Fixed seed for reproducibility within the function? 
    # Actually, let's vary seed per restart
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
        
        seed = 1000 + restart
        np.random.seed(seed)
        
        # Initialize with small random values
        # Sometimes initializing with 1s or -1s works better for sign symmetry
        init_U = np.random.randn(size_U) * 0.5
        init_V = np.random.randn(size_V) * 0.5
        init_W = np.random.randn(size_W) * 0.5
        
        # Try a few different initializations
        # 1. Random
        # 2. Sparse (mostly 0, some 1/-1)
        if restart % 2 == 0:
            # Sparse init
            init_U = np.zeros(size_U)
            init_V = np.zeros(size_V)
            init_W = np.zeros(size_W)
            # Set random non-zeros
            for idx in np.random.choice(size_U, size_U//4, replace=False):
                init_U[idx] = 1 if np.random.rand() > 0.5 else -1
            for idx in np.random.choice(size_V, size_V//4, replace=False):
                init_V[idx] = 1 if np.random.rand() > 0.5 else -1
            for idx in np.random.choice(size_W, size_W//4, replace=False):
                init_W[idx] = 1 if np.random.rand() > 0.5 else -1
        
        x0 = np.concatenate([init_U, init_V, init_W])
        
        try:
            res = minimize(
                _objective, 
                x0, 
                args=(n, m, p, r),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-8}
            )
            
            if res.success or res.fun < 1e-6:
                # Extract U, V, W
                U_flat = res.x[:size_U]
                V_flat = res.x[size_U:size_U+size_V]
                W_flat = res.x[size_U+size_V:]
                
                U = U_flat.reshape(n, m, r)
                V = V_flat.reshape(m, p, r)
                W = W_flat.reshape(p, n, r)
                
                # Snap to half-integers
                U_snapped = np.round(U * 2) / 2
                V_snapped = np.round(V * 2) / 2
                W_snapped = np.round(W * 2) / 2
                
                # Check if snapped solution is valid
                # Convert to lists for _verify_exact
                U_list = U_snapped.tolist()
                V_list = V_snapped.tolist()
                W_list = W_snapped.tolist()
                
                # Reshape back to (n*m, r) etc for the final output format
                # _verify_exact expects (n, m, r) internally? No, it reshapes.
                # Let's create the list format (n*m, r)
                U_final = np.array(U_list).reshape(n*m, r)
                V_final = np.array(V_list).reshape(m*p, r)
                W_final = np.array(W_list).reshape(p*n, r)
                
                if _verify_exact(U_final.tolist(), V_final.tolist(), W_final.tolist(), n, m, p):
                    return U_final.tolist(), V_final.tolist(), W_final.tolist()
                    
        except Exception as e:
            pass
            
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    
    n, m, p = DIMENSIONS
    
    # Schoolbook solution
    rank_schoolbook = n * m * p
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    best_U, best_V, best_W = U, V, W
    best_rank = rank_schoolbook
    
    # Try to find lower rank
    # Lower bound is max(n*m, m*p, p*n) = 20
    lower_bound = max(n * m, m * p, p * n)
    
    # We allocate time for each rank attempt
    # If we have 120s, and try ranks 20..39 (20 ranks), that's 6s per rank.
    # Let's be more aggressive on lower ranks.
    
    for r in range(lower_bound, best_rank):
        if time.time() > deadline - 5.0:
            break
            
        # Allocate time based on remaining budget
        remaining = deadline - time.time()
        # Give more time to lower ranks
        # Heuristic: 10s per rank, min 2s
        time_for_rank = min(10.0, remaining / (best_rank - r))
        
        if time_for_rank < 1.0:
            break
            
        result = _try_rank(r, n, m, p, max_restarts=10, time_limit=time_for_rank)
        
        if result is not None:
            U_new, V_new, W_new = result
            best_U, best_V, best_W = U_new, V_new, W_new
            best_rank = r
            # If we found a solution, we can stop if we hit the lower bound
            if r == lower_bound:
                break
                
    return {"U": best_U, "V": best_V, "W": best_W}

# EVOLVE-BLOCK-END
