import numpy as np
import time
from scipy.optimize import minimize

def construct():
    n, m, p = 2, 3, 7
    start_time = time.time()
    DEADLINE = 140.0

    # Indices for the tensor T[a, b, c] = 1 if consistent (i,j,k)
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                triples.append((a, b, c))
    
    num_constraints = len(triples) # 42

    def schoolbook():
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
        return U, V, W, rank

    U_fb, V_fb, W_fb, rank_fb = schoolbook()
    best_U, best_V, best_W, best_rank = U_fb, V_fb, W_fb, rank_fb

    def objective(params, R):
        """
        params: flattened vector of size R * (nm + mp + pn)
        Order: U (nm x R), V (mp x R), W (pn x R)
        """
        size_U = n * m
        size_V = m * p
        size_W = p * n
        
        U = params[0:size_U*R].reshape(size_U, R)
        V = params[size_U*R:size_U*R + size_V*R].reshape(size_V, R)
        W = params[size_U*R + size_V*R:].reshape(size_W, R)
        
        # Compute residuals
        # res[idx] = sum_r U[a,r] * V[b,r] * W[c,r] - 1
        residuals = np.zeros(num_constraints)
        for idx, (a, b, c) in enumerate(triples):
            residuals[idx] = np.sum(U[a, :] * V[b, :] * W[c, :]) - 1.0
            
        return np.sum(residuals * residuals)

    def try_rank(R, max_restarts=10, max_iter=1000):
        if R >= best_rank:
            return None
            
        size_U = n * m
        size_V = m * p
        size_W = p * n
        total_vars = R * (size_U + size_V + size_W)
        
        best_res = np.inf
        best_params = None
        
        # We can use multiple restarts
        # Note: L-BFGS-B is fast but can get stuck. Nelder-Mead is slower but more robust for small problems.
        # Given 150s limit, we have time for a few restarts.
        
        for restart in range(max_restarts):
            if time.time() - start_time > DEADLINE:
                return None
            
            # Initialize with small random values
            # Scaling matters. If values are too large, overflow. Too small, slow convergence.
            # 1/sqrt(R) is a decent heuristic.
            scale = 1.0 / np.sqrt(R)
            params0 = np.random.randn(total_vars) * scale
            
            # Try to optimize
            try:
                res = minimize(
                    objective, 
                    params0, 
                    args=(R,),
                    method='L-BFGS-B', 
                    options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8}
                )
            except Exception:
                continue
                
            if res.success or res.fun < best_res:
                best_res = res.fun
                best_params = res.x
                
            if best_res < 1e-10:
                break
        
        if best_params is not None and best_res < 1e-8:
            # Reconstruct U, V, W
            U = best_params[0:size_U*R].reshape(size_U, R)
            V = best_params[size_U*R:size_U*R + size_V*R].reshape(size_V, R)
            W = best_params[size_U*R + size_V*R:].reshape(size_W, R)
            
            # Verify strictly
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        val = np.sum(U[a, :] * V[b, :] * W[c, :])
                        if abs(val - 1.0) > 1e-8:
                            valid = False
                            break
                    if not valid: break
                if not valid: break
            
            if valid:
                return U, V, W
        return None

    # Try ranks from 41 down to 21
    # We can also try to speed up by checking if R is feasible.
    # Lower bound for rank of matrix multiplication is max(nm, mp, pn, n+p+m-1 ... ) 
    # Actually, just try downwards.
    
    # Optimization: If we find a solution for R, we can try R-1.
    # If we fail for R, we might still succeed for R-1? Unlikely if R is the min, but possible if search failed.
    # However, if the true min is 42, we will fail for 41..21 and return 42.
    
    # To save time, we can limit the number of restarts for lower ranks if we are running out of time.
    current_rank = rank_fb - 1
    while current_rank >= 21:
        if time.time() - start_time > DEADLINE - 20: # Leave time for return
            break
            
        # Adaptive restarts: more restarts for higher ranks (closer to known solution)?
        # Actually, lower ranks are harder. But if 42 is the min, we waste time.
        # Let's do a quick check for 41 first.
        
        max_restarts = 20 if current_rank > 35 else 5
        result = try_rank(current_rank, max_restarts=max_restarts)
        
        if result is not None:
            U_try, V_try, W_try = result
            best_U, best_V, best_W, best_rank = U_try, V_try, W_try, current_rank
            # If we found a solution, we can try to go lower
            current_rank -= 1
        else:
            # If we failed, we might assume no solution exists for this rank?
            # No, search might have failed. But for efficiency, we might stop if we've tried several and failed?
            # Given the strict time limit, if 41 fails, it's likely 42 is the min.
            # But let's try 40 just in case, with fewer restarts.
            current_rank -= 1
            if current_rank < 30: # If we drop too low without success, assume min is higher
                 # We can break early to save time if we are confident
                 # But for safety, let's just break if we hit deadline.
                 pass

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
