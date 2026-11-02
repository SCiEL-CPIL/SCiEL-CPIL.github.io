import numpy as np
import time
import random
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Trivial decomposition as fallback
    rank_trivial = n * m * p
    U_trivial = np.zeros((n * m, rank_trivial))
    V_trivial = np.zeros((m * p, rank_trivial))
    W_trivial = np.zeros((p * n, rank_trivial))
    t = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i * m + j, t] = 1
                V_trivial[j * p + k, t] = 1
                W_trivial[k * n + i, t] = 1
                t += 1
    
    best_result = (rank_trivial, U_trivial, V_trivial, W_trivial)
    
    # Build target tensor T
    # Shape: (n*m, m*p, p*n)
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    def get_error_and_grad(x, r):
        """
        x is a flat vector containing U, V, W
        r is the rank
        Returns error and gradient
        """
        # Unpack
        U = x[:n*m*r].reshape(n*m, r)
        V = x[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
        W = x[n*m*r + m*p*r :].reshape(p*n, r)
        
        # Compute R
        R = np.einsum('at,bt,ct->abc', U, V, W)
        
        # Error
        diff = R - T
        error = np.sum(diff * diff)
        
        # Gradient
        # dE/dU = 2 * diff * (V outer W)
        # diff shape (nm, mp, pn)
        # V shape (mp, r), W shape (pn, r)
        # We need grad_U (nm, r)
        # grad_U[a, t] = 2 * sum_{b,c} diff[a,b,c] * V[b,t] * W[c,t]
        
        # Efficient computation:
        # temp = np.einsum('abc,bt->act', diff, V) -> (nm, r)
        # grad_U = 2 * np.einsum('act,ct->at', temp, W)
        
        temp1 = np.einsum('abc,bt->act', diff, V)
        grad_U = 2.0 * np.einsum('act,ct->at', temp1, W)
        
        temp2 = np.einsum('abc,ct->abt', diff, W)
        grad_V = 2.0 * np.einsum('abt,at->bt', temp2, U)
        
        temp3 = np.einsum('abc,at->bct', diff, U)
        grad_W = 2.0 * np.einsum('bct,bt->ct', temp3, V)
        
        grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        
        return error, grad

    def verify_solution(U, V, W, r):
        """
        Checks if U, V, W provide an exact decomposition with integer or half-integer entries.
        """
        # Check if entries are close to integer or half-integer
        def is_valid_matrix(M):
            # Check integer
            int_rounded = np.rint(M)
            if np.allclose(M, int_rounded, atol=1e-5):
                return int_rounded
            # Check half-integer
            half_rounded = np.rint(M * 2) / 2.0
            if np.allclose(M, half_rounded, atol=1e-5):
                return half_rounded
            return None
        
        U_int = is_valid_matrix(U)
        V_int = is_valid_matrix(V)
        W_int = is_valid_matrix(W)
        
        if U_int is None or V_int is None or W_int is None:
            return None
        
        # Verify exact decomposition with rounded values
        R = np.einsum('at,bt,ct->abc', U_int, V_int, W_int)
        if np.allclose(R, T, atol=1e-9):
            return (U_int, V_int, W_int)
        
        return None

    random.seed(42)
    
    # Try ranks from 24 up to 36
    # Rank 23 is the rank for 3x3x3. For 3x3x4, it might be higher.
    # Let's try a range.
    min_rank_to_try = 24
    max_rank_to_try = 36
    
    for target_rank in range(min_rank_to_try, max_rank_to_try + 1):
        if time.time() > deadline:
            break
            
        # Number of restarts per rank
        # Lower ranks are harder, but if 24 is impossible, we waste time.
        # Let's try 10-20 restarts per rank.
        max_restarts = 10
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            # Random initialization
            # Scale: entries should be roughly 1/sqrt(r) or similar?
            # For matmul, entries are often small integers.
            # Let's try uniform random in [-1, 1] or small integers.
            # Small integers might help convergence to integer solutions.
            
            U_init = np.random.randint(-2, 3, size=(n*m, target_rank)).astype(np.float64)
            V_init = np.random.randint(-2, 3, size=(m*p, target_rank)).astype(np.float64)
            W_init = np.random.randint(-2, 3, size=(p*n, target_rank)).astype(np.float64)
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                res = minimize(
                    get_error_and_grad, 
                    x0, 
                    method='L-BFGS-B', 
                    jac=True, # We return (f, g)
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                if res.success or res.fun < 1e-6:
                    # Unpack result
                    x_opt = res.x
                    U_opt = x_opt[:n*m*target_rank].reshape(n*m, target_rank)
                    V_opt = x_opt[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                    W_opt = x_opt[n*m*target_rank + m*p*target_rank :].reshape(p*n, target_rank)
                    
                    sol = verify_solution(U_opt, V_opt, W_opt, target_rank)
                    if sol is not None:
                        U_sol, V_sol, W_sol = sol
                        best_result = (target_rank, U_sol, V_sol, W_sol)
                        return {"U": U_sol.tolist(), "V": V_sol.tolist(), "W": W_sol.tolist()}
                        
            except Exception as e:
                pass # Ignore optimization errors and try next restart
    
    r, U, V, W = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
