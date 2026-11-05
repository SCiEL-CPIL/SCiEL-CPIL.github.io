import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=3, m=3, p=5
    start_time = time.time()
    time_limit = 110.0  # Safety margin
    
    def verify_decomposition(U, V, W, n, m, p, num_trials=50):
        """Verify that the decomposition exactly computes matrix multiplication."""
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        rank = U.shape[1]
        
        # Check for half-integer constraint
        if not np.allclose(2 * U, np.round(2 * U), atol=1e-5):
            return False
        if not np.allclose(2 * V, np.round(2 * V), atol=1e-5):
            return False
        if not np.allclose(2 * W, np.round(2 * W), atol=1e-5):
            return False

        for _ in range(num_trials):
            A = np.random.rand(n, m)
            B = np.random.rand(m, p)
            C_expected = A @ B
            
            U_reshaped = U.reshape(n, m, rank)
            V_reshaped = V.reshape(m, p, rank)
            W_reshaped = W.reshape(p, n, rank)
            
            C_computed = np.zeros((n, p))
            for r in range(rank):
                u_r = U_reshaped[:, :, r]
                v_r = V_reshaped[:, :, r]
                w_r = W_reshaped[:, :, r]
                
                left = A @ u_r 
                right = v_r @ B 
                C_computed += np.outer(left, right) * w_r.T 
            
            if not np.allclose(C_computed, C_expected, atol=1e-9):
                return False
        return True

    def schoolbook():
        """Standard schoolbook decomposition with rank n*m*p = 45."""
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U.tolist(), V.tolist(), W.tolist()

    best_U, best_V, best_W = schoolbook()
    best_rank = n * m * p  # 45

    def objective(params, n, m, p, rank, test_As, test_Bs):
        """
        Objective function for optimization.
        params: flattened (U, V, W)
        """
        num_U = n * m * rank
        num_V = m * p * rank
        num_W = p * n * rank
        
        U_flat = params[:num_U]
        V_flat = params[num_U:num_U+num_V]
        W_flat = params[num_U+num_V:]
        
        U = U_flat.reshape(n, m, rank)
        V = V_flat.reshape(m, p, rank)
        W = W_flat.reshape(p, n, rank)
        
        error = 0.0
        for A, B in zip(test_As, test_Bs):
            C_expected = A @ B
            C_computed = np.zeros((n, p))
            for r in range(rank):
                u_r = U[:, :, r]
                v_r = V[:, :, r]
                w_r = W[:, :, r]
                left = A @ u_r
                right = v_r @ B
                C_computed += np.outer(left, right) * w_r.T
            error += np.sum((C_computed - C_expected) ** 2)
            
        return error

    # Pre-generate test matrices for speed
    np.random.seed(123)
    test_As = [np.random.rand(n, m) for _ in range(10)]
    test_Bs = [np.random.rand(m, p) for _ in range(10)]

    # Try to find lower rank
    for target_rank in range(44, 30, -1):
        if time.time() - start_time > time_limit:
            break
            
        # Multiple random initializations
        for init_trial in range(3):
            if time.time() - start_time > time_limit:
                break
                
            np.random.seed(42 + target_rank + init_trial)
            # Initialize with small random values to help convergence
            U_init = np.random.randn(n * m, target_rank) * 0.1
            V_init = np.random.randn(m * p, target_rank) * 0.1
            W_init = np.random.randn(p * n, target_rank) * 0.1
            
            params_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                res = minimize(
                    objective, 
                    params_init, 
                    args=(n, m, p, target_rank, test_As, test_Bs),
                    method='L-BFGS-B',
                    options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                if res.success and res.fun < 1e-8:
                    # Extract solution
                    num_U = n * m * target_rank
                    num_V = m * p * target_rank
                    num_W = p * n * target_rank
                    
                    U_sol = res.x[:num_U].reshape(n, m, target_rank).transpose(2, 0, 1).reshape(target_rank, n*m).T
                    # Correct reshaping:
                    U_sol = res.x[:num_U].reshape(n, m, target_rank)
                    U_sol = U_sol.transpose(2, 0, 1).reshape(target_rank, n*m).T
                    U_sol = U_sol.reshape(n*m, target_rank)
                    
                    V_sol = res.x[num_U:num_U+num_V].reshape(m, p, target_rank)
                    V_sol = V_sol.transpose(2, 0, 1).reshape(target_rank, m*p).T
                    V_sol = V_sol.reshape(m*p, target_rank)
                    
                    W_sol = res.x[num_U+num_V:].reshape(p, n, target_rank)
                    W_sol = W_sol.transpose(2, 0, 1).reshape(target_rank, p*n).T
                    W_sol = W_sol.reshape(p*n, target_rank)
                    
                    # Try to snap to half-integers
                    U_snap = np.round(2 * U_sol) / 2.0
                    V_snap = np.round(2 * V_sol) / 2.0
                    W_snap = np.round(2 * W_sol) / 2.0
                    
                    # Verify snapped solution
                    if verify_decomposition(U_snap, V_snap, W_snap, n, m, p, num_trials=20):
                        best_U, best_V, best_W = U_snap.tolist(), V_snap.tolist(), W_snap.tolist()
                        best_rank = target_rank
                        break # Found lowest for this rank, break init loop
                    else:
                        # Check if unsnapped is valid (though task requires half-integers)
                        # If not half-integer, it's invalid for the contract.
                        pass
                        
            except Exception as e:
                continue
        
        if best_rank == target_rank:
            break # Found a lower rank, continue to next lower or stop if minimal found logic applies
            
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
