"""Matrix multiplication 3,4,5 decomposition."""
import numpy as np
import time
import sys

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START

def _build_target_tensor(n, m, p):
    """
    Build the target tensor T with shape (n*m, m*p, p*n).
    T[i*m+j, j'*p+k, k'*n+i'] = 1 if i==i', j==j', k==k' else 0.
    """
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                T[idx_u, idx_v, idx_w] = 1.0
    return T

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify if the decomposition U, V, W exactly reproduces the target tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Compute the decomposed tensor D
    # D[x, y, z] = sum_r U[x, r] * V[y, r] * W[z, r]
    D = np.einsum('xr,yr,zr->xyz', U, V, W)
    
    # Build target tensor
    T = _build_target_tensor(n, m, p)
    
    return np.allclose(D, T, atol=tol, rtol=0)

def _round_to_half_integer(arr):
    """Round array values to the nearest integer or half-integer."""
    # Multiply by 2, round to nearest integer, divide by 2
    return (np.round(arr * 2) / 2)

def _search_lower_rank(n, m, p, start_rank, end_rank, time_limit):
    """
    Attempt to find a decomposition with rank between end_rank and start_rank (inclusive)
    using random initialization and local optimization.
    """
    T_target = _build_target_tensor(n, m, p)
    flat_target = T_target.flatten()
    
    def objective(params):
        # Reshape params into U, V, W
        n_u = n * m
        n_v = m * p
        n_w = p * n
        rank = params[0]  # This won't work as rank changes. We fix rank per call.
        # This function is for a fixed rank.
        pass # Placeholder, will define inside loop

    for rank in range(start_rank, end_rank - 1, -1):
        if time.time() > time_limit:
            break
            
        print(f"Trying rank {rank}...", file=sys.stderr)
        
        # Initialize parameters
        # We use a simple random initialization with small integers/halves
        # to encourage sparse/structured solutions
        np.random.seed(rank) # Deterministic for reproducibility in tests
        
        # Try a few random restarts for this rank
        for restart in range(3):
            if time.time() > time_limit:
                break
                
            U_init = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(n * m, rank))
            V_init = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(m * p, rank))
            W_init = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(p * n, rank))
            
            # Flatten
            params0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            # Define objective for this specific rank
            def obj(params):
                U_r = params[:n * m * rank].reshape(n * m, rank)
                V_r = params[n * m * rank : n * m * rank + m * p * rank].reshape(m * p, rank)
                W_r = params[n * m * rank + m * p * rank :].reshape(p * n, rank)
                
                # Compute tensor
                D = np.einsum('xr,yr,zr->xyz', U_r, V_r, W_r)
                error = np.linalg.norm(D.flatten() - flat_target)
                return error**2
            
            # Use scipy if available, otherwise simple gradient descent?
            # SciPy is available.
            try:
                from scipy.optimize import minimize
                # Nelder-Mead is robust for non-smooth-ish problems but slow.
                # Let's try a few iterations.
                res = minimize(obj, params0, method='Nelder-Mead', options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-4})
                
                if res.fun < 1e-6:
                    # Check if we can round to half-integers
                    U_opt = res.x[:n * m * rank].reshape(n * m, rank)
                    V_opt = res.x[n * m * rank : n * m * rank + m * p * rank].reshape(m * p, rank)
                    W_opt = res.x[n * m * rank + m * p * rank :].reshape(p * n, rank)
                    
                    U_rounded = _round_to_half_integer(U_opt)
                    V_rounded = _round_to_half_integer(V_opt)
                    W_rounded = _round_to_half_integer(W_opt)
                    
                    if _verify_decomposition(U_rounded, V_rounded, W_rounded, n, m, p):
                        print(f"Found valid rank {rank} decomposition!", file=sys.stderr)
                        return U_rounded, V_rounded, W_rounded
                        
            except Exception as e:
                print(f"Optimization failed for rank {rank}: {e}", file=sys.stderr)
                continue

    return None

def _schoolbook(n, m, p):
    """Generate the schoolbook decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # 1. Generate Schoolbook Decomposition (Rank 60)
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = U_school.shape[1]
    
    # Verify schoolbook (should always pass)
    if not _verify_decomposition(best_U, best_V, best_W, n, m, p):
        # Fallback to manual list construction if numpy fails (unlikely)
        pass

    # 2. Attempt to find a lower rank decomposition
    # We try ranks from 59 down to 50 (arbitrary limit, can be adjusted)
    # This is a heuristic search.
    if time.time() < deadline - 20:
        result = _search_lower_rank(n, m, p, best_rank - 1, 50, deadline - 10)
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = best_U.shape[1]
    
    # Convert to lists for JSON serialization
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }

# EVOLVE-BLOCK-END
