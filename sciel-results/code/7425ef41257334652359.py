import numpy as np
import time
import sys

DIMENSIONS = [2, 3, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n, m, p = DIMENSIONS
    # Dimensions for U, V, W
    dim_u = n * m  # 6
    dim_v = m * p  # 15
    dim_w = p * n  # 10
    
    # Construct the target tensor T
    # T[u, v, w] = 1 if u=(i,j), v=(j,k), w=(k,i)
    # Indices:
    # u = i * m + j
    # v = j * p + k
    # w = k * n + i
    T = np.zeros((dim_u, dim_v, dim_w), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
    
    T_flat = T.flatten()
    
    def compute_residuals(params):
        """
        params: flat array of length r * (dim_u + dim_v + dim_w)
        Order: U (r*dim_u), V (r*dim_v), W (r*dim_w)
        """
        r = params.size // (dim_u + dim_v + dim_w)
        if r == 0:
            return np.zeros(T_flat.size)
        
        # Reshape
        U = params[0:r*dim_u].reshape(r, dim_u).T # (dim_u, r)
        V = params[r*dim_u:r*(dim_u+dim_v)].reshape(r, dim_v).T # (dim_v, r)
        W = params[r*(dim_u+dim_v):].reshape(r, dim_w).T # (dim_w, r)
        
        # Compute product tensor
        # P[u, v, w] = sum_c U[u, c] * V[v, c] * W[w, c]
        P = np.einsum('uc,vc,wc->uvw', U, V, W)
        
        return (P - T).flatten()
    
    def verify_solution(U, V, W):
        """
        Check if U, V, W (with half-integer entries) exactly decompose T.
        """
        # Round to half-integers
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        # Check if they are close to the original (optional, but good for sanity)
        # Compute product
        P = np.einsum('uc,vc,wc->uvw', U_r, V_r, W_r)
        
        # Check exact match (with tolerance for floating point)
        if np.allclose(P, T, atol=1e-9):
            return True, U_r, V_r, W_r
        return False, None, None
    
    # Try ranks from 15 upwards
    # Known lower bound is 15. Schoolbook is 30.
    # Likely rank is 17 or 18.
    ranks_to_try = [17, 18, 16, 19, 15, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
    
    # If we don't find a better rank, return schoolbook
    schoolbook_rank = n * m * p
    U_school = np.zeros((dim_u, schoolbook_rank))
    V_school = np.zeros((dim_v, schoolbook_rank))
    W_school = np.zeros((dim_w, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    # Search loop
    best_solution = None
    best_rank = schoolbook_rank
    
    for r in ranks_to_try:
        if time.time() > deadline - 5.0:
            break
            
        if r >= best_rank:
            continue
            
        # Number of restarts depends on time available
        # Let's try a few restarts for each rank
        max_restarts = 50
        # Adjust restarts if time is tight
        if time.time() > deadline - 20.0:
            max_restarts = 10
            
        found = False
        for restart in range(max_restarts):
            if time.time() > deadline - 2.0:
                break
                
            # Random initialization
            # Scale: values should be around 1/sqrt(r) or similar
            # Let's use small random values
            scale = 1.0 / np.sqrt(r)
            U_init = np.random.randn(dim_u, r) * scale
            V_init = np.random.randn(dim_v, r) * scale
            W_init = np.random.randn(dim_w, r) * scale
            
            # Flatten
            params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            # Use scipy.optimize
            try:
                from scipy.optimize import least_squares
                # Use 'trf' method for bounds or just 'lm' if no bounds
                # We don't have bounds, so 'lm' (Levenberg-Marquardt) might be faster
                # But 'lm' doesn't support bounds. We don't need bounds.
                res = least_squares(compute_residuals, params_init, method='lm', max_nfev=1000)
                
                # Extract solution
                params_sol = res.x
                U_sol = params_sol[0:r*dim_u].reshape(dim_u, r)
                V_sol = params_sol[r*dim_u:r*(dim_u+dim_v)].reshape(dim_v, r)
                W_sol = params_sol[r*(dim_u+dim_v):].reshape(dim_w, r)
                
                # Check residual
                # If residual is small, verify with rounding
                if res.cost < 1e-8:
                    is_valid, U_r, V_r, W_r = verify_solution(U_sol, V_sol, W_sol)
                    if is_valid:
                        best_solution = (U_r, V_r, W_r)
                        best_rank = r
                        found = True
                        break
            except Exception:
                continue
                
        if found:
            # Found a solution for this rank, no need to try higher ranks (since we are trying in order of likelihood/optimality)
            # Note: ranks_to_try is ordered by likelihood, not strictly ascending.
            # If we find 17, we stop. If we find 18, we might still want to try 17? 
            # My list is [17, 18, 16, ...]. So if 17 fails, we try 18. If 18 succeeds, we have a rank 18 solution.
            # Should we continue to try 16? 16 is lower.
            # So we should not break immediately if we find a solution, unless we found the absolute lowest possible (15).
            # But 15 is unlikely.
            # Let's continue the loop to see if we can find a lower rank.
            pass

    if best_solution is not None:
        U, V, W = best_solution
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
