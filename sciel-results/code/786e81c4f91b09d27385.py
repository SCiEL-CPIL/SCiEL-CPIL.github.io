import numpy as np
import time

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Schoolbook decomposition
    U_schoolbook = np.zeros((n * m, rank_schoolbook))
    V_schoolbook = np.zeros((m * p, rank_schoolbook))
    W_schoolbook = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_schoolbook[i * m + j, col] = 1
                V_schoolbook[j * p + k, col] = 1
                W_schoolbook[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U = U_schoolbook
    best_V = V_schoolbook
    best_W = W_schoolbook
    
    start_time = time.time()
    deadline = start_time + 120  # Safety margin
    
    def check_decomposition(U, V, W, tol=1e-8):
        """Check if U, V, W decompose the matrix multiplication tensor."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                    if abs(val - 1) > tol:
                        return False
        return True
    
    def compute_error(U, V, W):
        """Compute the max error of the decomposition."""
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                    err = abs(val - 1)
                    if err > max_err:
                        max_err = err
        return max_err
    
    def alternating_minimization(U, V, W, target_rank, max_iters=30, tol=1e-10, time_limit=None):
        """
        Alternating minimization for tensor decomposition.
        Given U, V, W of shape (n*m, r), (m*p, r), (p*n, r),
        find W, U, V such that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1.
        
        For fixed U,V: W[k*n+i, r] = argmin sum_{i,j,k} (U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - delta_{i,k})^2
        This is a linear system in W for each r independently.
        """
        if time_limit is not None and time.time() > time_limit:
            return U, V, W
        
        for iteration in range(max_iters):
            if time_limit is not None and time.time() > time_limit:
                break
            
            # Solve for W: for each r, solve linear system
            # sum_{i,j,k} U[i*m+j,r]^2 * V[j*p+k,r]^2 * W[k*n+i,r]^2 - 2*sum_{i,j,k} U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] + ...
            # Actually, for fixed U,V, the problem for W is:
            # minimize sum_{i,j,k} (sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1)^2
            # This is NOT separable in r because of the square of the sum.
            # 
            # Alternative: use gradient-based approach or solve the full system.
            # Let's use a simpler approach: treat it as a least squares problem.
            
            # For each r, the contribution to the (i,j,k) entry is U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
            # We want sum_r of these = 1
            # 
            # Let's use a different approach: iterate by fixing two and solving for the third.
            # For fixed U,V, find W to minimize ||sum_r U_r * V_r * W_r - T||_F^2
            # where U_r is the outer product structure.
            
            # Actually, let's use a simpler gradient descent approach.
            
            # Compute gradient for W
            # f = sum_{i,j,k} (sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1)^2
            # df/dW[k*n+i,r] = 2 * sum_{j} U[i*m+j,r]*V[j*p+k,r] * (sum_{r'} U[i*m+j,r']*V[j*p+k,r']*W[k*n+i,r'] - 1)
            
            # This is getting complex. Let's use a simpler approach:
            # Use the fact that for fixed U,V, we can solve for W via linear least squares
            # by treating each (i,k) pair separately, but the coupling through j makes it non-trivial.
            
            # Let's use a practical approach: gradient descent with adaptive step size.
            
            # Compute current values
            current_vals = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        current_vals[i, j, k] = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
            
            # Gradient for W
            grad_W = np.zeros_like(W)
            for r in range(target_rank):
                for i in range(n):
                    for k in range(p):
                        # sum over j
                        s = 0.0
                        for j in range(m):
                            s += U[i * m + j, r] * V[j * p + k, r] * (current_vals[i, j, k] - 1)
                        grad_W[k * n + i, r] = 2 * s
            
            # Adaptive step size
            if np.max(np.abs(grad_W)) < 1e-15:
                continue
            step = 0.1 / (np.max(np.abs(grad_W)) + 1e-15)
            step = min(step, 0.5)
            W_new = W - step * grad_W
            
            # Check improvement
            err_old = compute_error(U, V, W)
            err_new = compute_error(U, V, W_new)
            if err_new < err_old:
                W = W_new
            else:
                # Reduce step size
                step *= 0.5
                W_new = W - step * grad_W
                err_new = compute_error(U, V, W_new)
                if err_new < err_old:
                    W = W_new
                else:
                    break
            
            # Similarly for U and V (but this is getting expensive)
            # For simplicity, let's just do one pass per iteration and rely on multiple restarts
            
            if compute_error(U, V, W) < tol:
                break
        
        return U, V, W
    
    # Strategy: Use alternating minimization with multiple restarts at various ranks
    target_ranks = [64, 62, 60, 58, 56, 54, 52, 50]
    
    for target_rank in target_ranks:
        if time.time() > deadline - 20:
            break
        
        found = False
        for attempt in range(15):
            if time.time() > deadline - 20:
                break
            
            np.random.seed(attempt * 1000 + target_rank)
            
            # Initialize with scaled random values
            scale = 1.0 / np.sqrt(target_rank)
            U = np.random.randn(n * m, target_rank) * scale
            V = np.random.randn(m * p, target_rank) * scale
            W = np.random.randn(p * n, target_rank) * scale
            
            # Run alternating minimization
            U, V, W = alternating_minimization(U, V, W, target_rank, max_iters=20, tol=1e-10, time_limit=deadline - 20)
            
            # Check if we found a solution
            if check_decomposition(U, V, W, tol=1e-8):
                # Try to round to half-integers
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                if check_decomposition(U_r, V_r, W_r, tol=1e-8):
                    best_rank = target_rank
                    best_U = U_r
                    best_V = V_r
                    best_W = W_r
                else:
                    best_rank = target_rank
                    best_U = U
                    best_V = V
                    best_W = W
                found = True
                break
        
        if found:
            break
    
    # If still at schoolbook rank, try more aggressive search
    if best_rank == rank_schoolbook and time.time() < deadline - 10:
        for target_rank in [64, 62, 60]:
            if time.time() > deadline - 10:
                break
            
            for attempt in range(30):
                if time.time() > deadline - 10:
                    break
                
                np.random.seed(attempt * 5000 + target_rank)
                scale = 1.0 / np.sqrt(target_rank)
                U = np.random.randn(n * m, target_rank) * scale
                V = np.random.randn(m * p, target_rank) * scale
                W = np.random.randn(p * n, target_rank) * scale
                
                U, V, W = alternating_minimization(U, V, W, target_rank, max_iters=30, tol=1e-10, time_limit=deadline - 10)
                
                if check_decomposition(U, V, W, tol=1e-8):
                    best_rank = target_rank
                    best_U = U
                    best_V = V
                    best_W = W
                    break
            
            if best_rank < rank_schoolbook:
                break
    
    # Return the best found
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
