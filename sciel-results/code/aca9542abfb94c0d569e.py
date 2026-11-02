import numpy as np
import scipy.optimize as opt
import time
import random

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # 1. Construct Schoolbook (Baseline)
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
                
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = rank_schoolbook
    
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    # Target tensor: all ones
    # T[i, j, k] = 1 for all i, j, k
    # Decomposition: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    
    def get_residual(x, r):
        U = x[:r * n * m].reshape(n * m, r)
        V = x[r * n * m : r * n * m + r * m * p].reshape(m * p, r)
        W = x[r * n * m + r * m * p :].reshape(p * n, r)
        
        # Compute tensor
        T = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
        return (T - 1).flatten()
        
    def verify_and_round(U, V, W, r):
        """
        Checks if U, V, W (floats) can be rounded to half-integers such that
        the decomposition is exact.
        Returns (True, U_rounded, V_rounded, W_rounded) or (False, None, None, None).
        """
        def is_valid(x_mat):
            x2 = x_mat * 2
            # Check if x2 is close to integer
            if np.all(np.abs(x2 - np.round(x2)) < 1e-5):
                return True
            return False
        
        if is_valid(U) and is_valid(V) and is_valid(W):
            U_int = (U * 2).round() / 2
            V_int = (V * 2).round() / 2
            W_int = (W * 2).round() / 2
            
            # Verify exact decomposition with rounded values
            T_check = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_check[i, j, k] = np.dot(U_int[i * m + j, :], V_int[j * p + k, :] * W_int[k * n + i, :])
                        
            if np.all(np.abs(T_check - 1) < 1e-9):
                return True, U_int, V_int, W_int
        return False, None, None, None

    # Strategy:
    # 1. Try a few ranks slightly below schoolbook (62, 61, 60) with random restarts.
    #    This is the most likely place to find a valid decomposition if one exists
    #    that is "close" to the schoolbook structure in the manifold of solutions.
    # 2. If time remains, try lower ranks.
    
    # We prioritize finding a valid lower rank.
    # Start with ranks likely to be solvable if the problem is "almost" schoolbook.
    # 62, 61, 60, 59...
    
    # To save time, we can try a few specific ranks first.
    # If we find rank 60, score 1.05.
    # If we find rank 55, score 1.14.
    
    # We will iterate downwards.
    ranks_to_try = list(range(rank_schoolbook - 1, 39, -1))
    
    # Optimization settings
    # We use 'trf' (Trust Region Reflective) which handles bounds well if we added them,
    # but here we use unconstrained least squares.
    # We will try multiple random initializations.
    
    for r in ranks_to_try:
        if time.time() > deadline:
            break
            
        # Try a few random initializations for this rank
        # Number of restarts decreases as time runs out
        # Start with 5, reduce to 2 if time is tight
        remaining_time = deadline - time.time()
        if remaining_time < 10:
            break
            
        # Estimate cost per restart:
        # Residual evaluation: O(n*m*p*r) ~ 63 * r.
        # Jacobian: finite differences or analytic? scipy uses finite diff by default for least_squares if jac not provided.
        # Actually, providing an analytic Jacobian is much faster.
        
        def get_jacobian(x, r):
            U = x[:r * n * m].reshape(n * m, r)
            V = x[r * n * m : r * n * m + r * m * p].reshape(m * p, r)
            W = x[r * n * m + r * m * p :].reshape(p * n, r)
            
            # T[i,j,k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
            # dT/du_a_c = V[j*p+k, c] * W[k*n+i, c]
            # dT/dv_b_c = U[i*m+j, c] * W[k*n+i, c]
            # dT/dw_c_c = U[i*m+j, c] * V[j*p+k, c]
            
            # The residual vector is flattened T - 1.
            # Index in residual: idx = i * m * p + j * p + k
            # We need to map (i,j,k) to the appropriate entries in U, V, W.
            
            # This is complex to construct as a dense matrix efficiently in Python loops.
            # For small dimensions (63 entries), dense Jacobian is fine.
            # Size of x: r * (9 + 21 + 21) = 51r.
            # Size of residual: 63.
            # Jacobian shape: (63, 51r).
            
            J = np.zeros((n * m * p, r * (n * m + m * p + p * n)))
            
            # Precompute indices for speed?
            # Let's just loop.
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # Row idx in J corresponds to residual element T[i,j,k]
                        
                        # dT/du
                        # U index: i*m+j, col c
                        # x index: c (offset 0)
                        for c in range(r):
                            val = V[j * p + k, c] * W[k * n + i, c]
                            J[idx, c] = val
                            
                        # dT/dv
                        # V index: j*p+k, col c
                        # x index: n*m + (j*p+k)*r + c
                        for c in range(r):
                            val = U[i * m + j, c] * W[k * n + i, c]
                            J[idx, n * m + (j * p + k) * r + c] = val
                            
                        # dT/dw
                        # W index: k*n+i, col c
                        # x index: n*m + m*p + (k*n+i)*r + c
                        for c in range(r):
                            val = U[i * m + j, c] * V[j * p + k, c]
                            J[idx, n * m + m * p + (k * n + i) * r + c] = val
                            
                        idx += 1
            return J

        max_restarts = 5 if remaining_time > 60 else 2
        if remaining_time < 30:
            max_restarts = 1
            
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            # Initialize with small random numbers
            # Scale: if U~1, V~1, W~1, sum is ~r. Target is 1.
            # So we need small values. 1/sqrt(r) ?
            scale = 1.0 / np.sqrt(r)
            x0 = np.random.randn(r * (n * m + m * p + p * n)) * scale
            
            # Optimize
            try:
                # Use analytic Jacobian for speed
                res = opt.least_squares(
                    get_residual, 
                    x0, 
                    args=(r,), 
                    jac=get_jacobian, 
                    args_jac=(r,),
                    method='trf', 
                    max_nfev=500, # Reduced max evals to fit in time budget
                    ftol=1e-12,
                    xtol=1e-12,
                    gtol=1e-12
                )
                
                # Check convergence
                if res.cost < 1e-8:
                    U = res.x[:r * n * m].reshape(n * m, r)
                    V = res.x[r * n * m : r * n * m + r * m * p].reshape(m * p, r)
                    W = res.x[r * n * m + r * m * p :].reshape(p * n, r)
                    
                    valid, U_int, V_int, W_int = verify_and_round(U, V, W, r)
                    if valid:
                        best_rank = r
                        best_result = {
                            "U": U_int.tolist(),
                            "V": V_int.tolist(),
                            "W": W_int.tolist()
                        }
                        # We found a valid solution for rank r.
                        # Since we are iterating downwards, this is the best so far.
                        # We can continue to try lower ranks in the outer loop.
                        # But we should probably break the restart loop for this rank
                        # to save time for lower ranks.
                        break
                        
            except Exception as e:
                pass
                
    return best_result
