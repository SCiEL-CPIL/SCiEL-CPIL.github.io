"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s safety margin
    
    # Build schoolbook decomposition as fallback
    U_school = np.zeros((n * m, schoolbook_rank))
    V_school = np.zeros((m * p, schoolbook_rank))
    W_school = np.zeros((p * n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1.0
                V_school[j * p + k, col] = 1.0
                W_school[k * n + i, col] = 1.0
                col += 1
    
    best_U = U_school.copy()
    best_V = V_school.copy()
    best_W = W_school.copy()
    best_rank = schoolbook_rank
    best_err = 0.0
    
    def compute_error(U, V, W, n, m, p):
        """Compute sum of squared errors for the decomposition using einsum."""
        rank = U.shape[1]
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        err = np.sum((T - 1.0) ** 2)
        return err

    def compute_gradient(U, V, W, n, m, p):
        """Compute gradients of the error with respect to U, V, W using einsum."""
        rank = U.shape[1]
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        
        T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        R = 2.0 * (T - 1.0)
        
        dU3 = np.einsum('ijk,jkr,kir->ijr', R, V3, W3)
        dV3 = np.einsum('ijk,ijr,kir->jkr', R, U3, W3)
        dW3 = np.einsum('ijk,ijr,jkr->kir', R, U3, V3)
        
        return dU3.reshape(n*m, rank), dV3.reshape(m*p, rank), dW3.reshape(p*n, rank)
    
    # Try various ranks from low to high
    # Lower bound: max(n*p, n*m, m*p) = max(24, 20, 30) = 30
    lower_bound = max(n * p, n * m, m * p)
    
    for target_rank in range(lower_bound, best_rank):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations with different seeds
        n_tries = 10
        for trial in range(n_tries):
            if time.time() > deadline:
                break
            
            np.random.seed(12345 + trial * 7919 + target_rank * 104729)
            
            # Random initialization
            U = np.random.randn(n * m, target_rank) * 0.3
            V = np.random.randn(m * p, target_rank) * 0.3
            W = np.random.randn(p * n, target_rank) * 0.3
            
            # Gradient descent with adaptive step size
            lr = 0.01
            best_local_err = float('inf')
            best_local_U = U.copy()
            best_local_V = V.copy()
            best_local_W = W.copy()
            
            max_iters = 2000
            for it in range(max_iters):
                if time.time() > deadline:
                    break
                
                err = compute_error(U, V, W, n, m, p)
                
                if err < best_local_err:
                    best_local_err = err
                    best_local_U = U.copy()
                    best_local_V = V.copy()
                    best_local_W = W.copy()
                
                if err < 1e-12:
                    break
                
                # Compute gradient
                dU, dV, dW = compute_gradient(U, V, W, n, m, p)
                
                # Gradient norm for adaptive step
                grad_norm = np.sqrt(np.sum(dU**2) + np.sum(dV**2) + np.sum(dW**2))
                if grad_norm < 1e-15:
                    break
                
                # Adaptive learning rate
                step = lr / max(grad_norm, 1e-8)
                
                # Update
                U -= step * dU
                V -= step * dV
                W -= step * dW
                
                # If error increased, reduce learning rate
                new_err = compute_error(U, V, W, n, m, p)
                if new_err > err * 1.5:
                    U += step * dU
                    V += step * dV
                    W += step * dW
                    lr *= 0.5
                
                # Occasionally increase learning rate if stuck
                if it % 100 == 99 and new_err < best_local_err * 0.99:
                    lr *= 1.1
            
            # Check if we found a good solution
            final_err = compute_error(best_local_U, best_local_V, best_local_W, n, m, p)
            if final_err < best_err or (final_err < 1e-10 and target_rank < best_rank):
                best_err = final_err
                best_rank = target_rank
                best_U = best_local_U.copy()
                best_V = best_local_V.copy()
                best_W = best_local_W.copy()
                
                if best_err < 1e-10:
                    break
        
        if best_err < 1e-10:
            break
    
    # If we found a valid decomposition, clean it up and verify
    valid_solution = False
    if best_err < 1e-8:
        U_clean = best_U.copy()
        V_clean = best_V.copy()
        W_clean = best_W.copy()
        
        # Round to nearest half-integer
        # Values should be k/2
        U_clean = np.round(U_clean * 2) / 2.0
        V_clean = np.round(V_clean * 2) / 2.0
        W_clean = np.round(W_clean * 2) / 2.0
        
        # Verify the rounded solution is exact
        err_clean = compute_error(U_clean, V_clean, W_clean, n, m, p)
        if err_clean < 1e-6:
            valid_solution = True
            best_U = U_clean
            best_V = V_clean
            best_W = W_clean
    
    # Convert to list of lists
    def to_list(mat):
        return [[float(x) for x in row] for row in mat]
    
    if valid_solution:
        return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
    else:
        # Fallback to schoolbook decomposition which is always valid
        return {"U": to_list(U_school), "V": to_list(V_school), "W": to_list(W_school)}
# EVOLVE-BLOCK-END
