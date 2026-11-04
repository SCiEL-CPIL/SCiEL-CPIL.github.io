import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    
    N = n * m  # 9
    M = m * p  # 12
    P = p * n  # 12
    
    # Build target tensor T[a, b, c]
    T = np.zeros((N, M, P))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T[a_idx, b_idx, c_idx] = 1.0
    
    start_time = time.time()
    timeout = 135.0  # Safety margin before 150s kill
    
    # Pre-compute matricizations
    T_c = T.reshape(N * M, P)
    T_a_vec = T.reshape(N, M * P)
    T_b_vec = T.transpose(1, 0, 2).reshape(M, N * P)
    
    def check_solution(U, V, W, tol=1e-8):
        """Check if solution is valid with half-integer rounding."""
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        T_calc = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
        return np.allclose(T_calc, T, atol=tol), U_r, V_r, W_r
    
    def als_iterate(U, V, W, R, max_iter=300, tol=1e-12):
        """Run ALS iterations with convergence check."""
        prev_err = np.inf
        for it in range(max_iter):
            if time.time() - start_time > timeout:
                return U, V, W, False
            
            # Update W
            E_W = np.einsum('ar,br->ab r', U, V).reshape(N * M, R)
            try:
                W_T, _, _, _ = np.linalg.lstsq(E_W, T_c.T, rcond=None)
                W = W_T.T
            except:
                return U, V, W, False
            
            # Update U
            E_U = np.einsum('br,cr->bc r', V, W).reshape(M * P, R)
            try:
                U_T, _, _, _ = np.linalg.lstsq(E_U.T, T_a_vec.T, rcond=None)
                U = U_T.T
            except:
                return U, V, W, False
            
            # Update V
            E_V = np.einsum('ar,cr->ac r', U, W).reshape(N * P, R)
            try:
                V_T, _, _, _ = np.linalg.lstsq(E_V.T, T_b_vec.T, rcond=None)
                V = V_T.T
            except:
                return U, V, W, False
            
            # Check convergence
            T_calc = np.einsum('ar,br,cr->abc', U, V, W)
            error = np.linalg.norm(T_calc - T)
            
            if abs(prev_err - error) < tol:
                return U, V, W, True
            prev_err = error
        
        return U, V, W, False
    
    def gradient_descent_polish(U, V, W, R, lr=0.01, max_iter=5000):
        """Polish with gradient descent to find near-half-integer solutions."""
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        for it in range(max_iter):
            if time.time() - start_time > timeout:
                return U, V, W, False
            
            # Compute gradient
            # dL/dU = (T - T_calc) contracted with V, W
            T_calc = np.einsum('br,cr->abc', V, W)
            residual = T - np.einsum('ar,abc->abc', U, T_calc)
            
            dU = -np.einsum('abc,br,cr->ar', residual, V, W)
            dV = -np.einsum('abc,ar,cr->br', residual, U, W)
            dW = -np.einsum('abc,ar,br->cr', residual, U, V)
            
            # Adaptive learning rate
            grad_norm = np.sqrt(np.sum(dU**2) + np.sum(dV**2) + np.sum(dW**2))
            if grad_norm < 1e-15:
                return U, V, W, True
            
            # Scale step to keep updates small
            step_scale = min(lr, 0.1 / (grad_norm + 1e-10))
            
            U += step_scale * dU
            V += step_scale * dV
            W += step_scale * dW
        
        return U, V, W, False
    
    def try_rank(R, num_attempts=20):
        """Try to find valid decomposition at rank R."""
        for attempt in range(num_attempts):
            if time.time() - start_time > timeout:
                return None
            
            # Different initialization strategies
            if attempt == 0:
                np.random.seed(42 + R * 1000 + attempt)
                U = np.random.randn(N, R)
                V = np.random.randn(M, R)
                W = np.random.randn(P, R)
            elif attempt == 1:
                np.random.seed(1000 + R * 100 + attempt)
                U = np.random.randn(N, R) * 0.5
                V = np.random.randn(M, R) * 2.0
                W = np.random.randn(P, R) * 1.0
            elif attempt == 2:
                # Sparse-ish initialization
                np.random.seed(2000 + R * 100 + attempt)
                U = np.random.randn(N, R)
                V = np.random.randn(M, R)
                W = np.random.randn(P, R)
                # Zero out most entries
                mask_U = np.random.rand(N, R) < 0.3
                mask_V = np.random.rand(M, R) < 0.3
                mask_W = np.random.rand(P, R) < 0.3
                U *= mask_U
                V *= mask_V
                W *= mask_W
                # Ensure at least some non-zero entries
                if np.sum(U != 0) < R:
                    U[np.random.choice(N, R, replace=False), np.arange(R)] = 1.0
                if np.sum(V != 0) < R:
                    V[np.random.choice(M, R, replace=False), np.arange(R)] = 1.0
                if np.sum(W != 0) < R:
                    W[np.random.choice(P, R, replace=False), np.arange(R)] = 1.0
            elif attempt == 3:
                # Structured initialization based on known patterns
                np.random.seed(3000 + R * 100 + attempt)
                U = np.zeros((N, R))
                V = np.zeros((M, R))
                W = np.zeros((P, R))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            col = (i * m + j) * p + k
                            if col < R:
                                U[i * m + j, col] = 1.0
                                V[j * p + k, col] = 1.0
                                W[k * n + i, col] = 1.0
                # Add random perturbation
                U += 0.1 * np.random.randn(N, R)
                V += 0.1 * np.random.randn(M, R)
                W += 0.1 * np.random.randn(P, R)
            elif attempt == 4:
                # Scale variation
                np.random.seed(4000 + R * 100 + attempt)
                scale = np.random.choice([0.1, 0.5, 1.0, 2.0, 5.0])
                U = np.random.randn(N, R) * scale
                V = np.random.randn(M, R) / scale
                W = np.random.randn(P, R)
            else:
                # More random seeds
                np.random.seed(5000 + R * 50 + attempt * 7)
                U = np.random.randn(N, R)
                V = np.random.randn(M, R)
                W = np.random.randn(P, R)
            
            # Run ALS
            U, V, W, converged = als_iterate(U, V, W, R, max_iter=200)
            
            if time.time() - start_time > timeout:
                return None
            
            # Check if solution is valid with rounding
            valid, U_r, V_r, W_r = check_solution(U, V, W)
            if valid:
                return (U_r, V_r, W_r)
            
            # If not valid, try polishing with gradient descent
            if attempt < 5:  # Only polish for first few attempts to save time
                U_p, V_p, W_p, _ = gradient_descent_polish(U, V, W, R, lr=0.005, max_iter=2000)
                if time.time() - start_time > timeout:
                    return None
                valid, U_r, V_r, W_r = check_solution(U_p, V_p, W_p)
                if valid:
                    return (U_r, V_r, W_r)
        
        return None
    
    # Search ranks from low to high
    # Known results: 3x3x3 = 23, 3x3x4 likely around 24-28
    best_result = None
    
    for R in range(23, 37):
        if time.time() - start_time > timeout:
            break
        
        result = try_rank(R, num_attempts=25)
        if result is not None:
            best_result = (R, result)
            break
    
    if best_result is not None:
        R, (U, V, W) = best_result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
