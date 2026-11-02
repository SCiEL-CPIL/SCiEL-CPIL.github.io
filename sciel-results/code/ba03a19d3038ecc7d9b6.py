import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0
    best_rank = n * m * p
    best_result = _schoolbook(n, m, p, best_rank)

    # Verify schoolbook
    if _verify(n, m, p, best_result):
        pass
    
    # Try to find lower rank decompositions
    # We will search for ranks lower than 90
    # Using continuous optimization with regularization towards half-integers
    
    def objective_and_grad(x, n, m, p, r):
        """
        x is a flattened array of length (n*m + m*p + p*n) * r
        Returns (loss, gradient)
        """
        # Unpack
        size_u = n * m * r
        size_v = m * p * r
        size_w = p * n * r
        
        U = x[:size_u].reshape(n * m, r)
        V = x[size_u:size_u + size_v].reshape(m * p, r)
        W = x[size_u + size_v:].reshape(p * n, r)
        
        # Calculate Tensor Error
        # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # We want T[i,j,k] == 1
        
        # Efficient calculation using einsum or loops
        # T is n x m x p
        # U is (n*m) x r
        # V is (m*p) x r
        # W is (p*n) x r
        
        # Reshape U to (n, m, r)
        U_3d = U.reshape(n, m, r)
        # Reshape V to (m, p, r)
        V_3d = V.reshape(m, p, r)
        # Reshape W to (p, n, r)
        W_3d = W.reshape(p, n, r)
        
        # T[i,j,k] = sum_c U_3d[i,j,c] * V_3d[j,k,c] * W_3d[k,i,c]
        # This is a contraction.
        # First, compute M[i,j,k] = sum_c U_3d[i,j,c] * V_3d[j,k,c]
        # M is n x m x p
        # M[i,j,k] = np.einsum('ijc,jkc->ijk', U_3d, V_3d)
        M = np.einsum('ijc,jkc->ijk', U_3d, V_3d)
        
        # Now T[i,j,k] = sum_k M[i,j,k] * W_3d[k,i,c]? No.
        # W is indexed by [k*n+i, c]. So W_3d[k,i,c] is correct.
        # T[i,j,k] = sum_c M[i,j,c] * W_3d[k,i,c] ? No.
        # M[i,j,c] is sum_c U[i,j,c]V[j,k,c].
        # So T[i,j,k] = sum_c (sum_c' U[i,j,c']V[j,k,c']) * W[k,i,c] ?? No.
        # The sum is over c.
        # T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c].
        
        # Let's compute T directly with einsum
        # 'ijc,jkc,kic->ijk'
        T = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
        
        # Error
        error = T - 1.0
        loss_tensor = np.sum(error ** 2)
        
        # Regularization towards half-integers
        # Penalize (x * 2 - round(x * 2))^2
        # This is non-differentiable at .5 boundaries, but we can use a smooth approximation
        # or just use the raw value. Since we are using gradient methods, non-differentiability is a problem.
        # Instead, we can penalize distance to nearest half-integer using a smooth function.
        # dist(x, 0.5*k) = min_k |x - 0.5k|
        # A smooth approximation: 0.5 * (1 - cos(4 * pi * x))? No, that's periodic.
        # Let's just use the tensor error primarily. If we find a solution, we snap and verify.
        # The regularization might slow down convergence if it's too strong.
        # Let's try without regularization first, then snap.
        
        return loss_tensor, None # Gradient is complex to compute analytically here, use numerical or BFGS finite diff? 
        # BFGS with finite diff is slow for 63*r variables.
        # Let's use Powell or Nelder-Mead which don't need gradients?
        # Nelder-Mead is very slow for high dimensions.
        # Let's stick to a simple stochastic gradient descent or just rely on the fact that 
        # for small r, the landscape might be navigable.
        
        # Actually, let's just use a simple local search on the discrete grid if optimization fails.
        # But the prompt asks for a stronger algorithm.
        # Let's use scipy.optimize.minimize with method='Powell' (derivative free) on a subset of variables?
        # No, let's try to implement the gradient.
        
        # Gradient of T wrt U[i,j,c] is V[j,k,c] W[k,i,c] summed over k?
        # dT[i,j,k] / dU[i,j,c] = V[j,k,c] W[k,i,c]
        # dLoss / dU[i,j,c] = 2 * error[i,j,k] * V[j,k,c] W[k,i,c]
        
        # We can compute the gradient efficiently.
        grad_U = np.zeros_like(U_3d)
        grad_V = np.zeros_like(V_3d)
        grad_W = np.zeros_like(W_3d)
        
        # dL/dU[i,j,c] = 2 * sum_k error[i,j,k] * V[j,k,c] * W[k,i,c]
        # error is n x m x p
        # V is m x p x r
        # W is p x n x r
        
        # grad_U[i,j,c] = 2 * sum_k error[i,j,k] * V[j,k,c] * W[k,i,c]
        # This is a tensor contraction.
        # Let's compute it.
        
        # Note: error is (n, m, p)
        # V is (m, p, r)
        # W is (p, n, r)
        
        # For fixed i, j, c:
        # grad_U[i,j,c] = 2 * sum_k error[i,j,k] * V[j,k,c] * W[k,i,c]
        
        # We can compute this using einsum:
        # 'ijk,jkc,kic->ijc'
        grad_U = 2.0 * np.einsum('ijk,jkc,kic->ijc', error, V_3d, W_3d)
        
        # dL/dV[j,k,c] = 2 * sum_i error[i,j,k] * U[i,j,c] * W[k,i,c]
        # 'ijk,ijc,kic->jkc'
        grad_V = 2.0 * np.einsum('ijk,ijc,kic->jkc', error, U_3d, W_3d)
        
        # dL/dW[k,i,c] = 2 * sum_j error[i,j,k] * U[i,j,c] * V[j,k,c]
        # 'ijk,ijc,jkc->kic'
        grad_W = 2.0 * np.einsum('ijk,ijc,jkc->kic', error, U_3d, V_3d)
        
        grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        return loss_tensor, grad

    # Use L-BFGS-B
    def try_rank(r, num_restarts=3, time_budget=20.0):
        nonlocal best_rank, best_result
        start_time = time.time()
        
        for restart in range(num_restarts):
            if time.time() > time_budget + start_time or time.time() > deadline:
                break
            
            # Initialize
            size_u = n * m * r
            size_v = m * p * r
            size_w = p * n * r
            x0 = np.random.normal(0, 1, size_u + size_v + size_w)
            
            # Optimize
            try:
                res = minimize(objective_and_grad, x0, args=(n, m, p, r), method='L-BFGS-B', 
                               jac=True, options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-8})
                
                if res.success or res.fun < 1e-6:
                    # Extract solution
                    U = res.x[:size_u].reshape(n * m, r)
                    V = res.x[size_u:size_u + size_v].reshape(m * p, r)
                    W = res.x[size_u + size_v:].reshape(p * n, r)
                    
                    # Snap to half-integers
                    U_snap = np.round(U * 2) / 2
                    V_snap = np.round(V * 2) / 2
                    W_snap = np.round(W * 2) / 2
                    
                    # Verify snapped version
                    if _verify_matrices(n, m, p, U_snap, V_snap, W_snap):
                        result = {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
                        if r < best_rank:
                            best_rank = r
                            best_result = result
                            return True
                            
                    # If snap fails, check if original is valid (in case evaluator allows reals)
                    # But we assume strict integer/half-integer. 
                    # We can also try to search locally around the snapped solution.
                    
            except Exception as e:
                pass
                
        return False

    # Try ranks
    # We want to find the minimal rank. 
    # Let's try a range.
    ranks_to_try = [85, 80, 75, 70, 65, 60, 55, 50, 45, 40]
    
    for r in ranks_to_try:
        if time.time() > deadline - 10:
            break
        if try_rank(r, num_restarts=2, time_budget=10.0):
            # Found a solution at rank r. 
            # We could try to go lower, but time is tight.
            # Let's break and return.
            break
            
    return best_result

def _schoolbook(n, m, p, rank):
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

def _verify(n, m, p, result):
    U = np.array(result["U"], dtype=float)
    V = np.array(result["V"], dtype=float)
    W = np.array(result["W"], dtype=float)
    r = U.shape[1]
    return _verify_matrices(n, m, p, U, V, W)

def _verify_matrices(n, m, p, U, V, W):
    r = U.shape[1]
    U_3d = U.reshape(n, m, r)
    V_3d = V.reshape(m, p, r)
    W_3d = W.reshape(p, n, r)
    T = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
    return np.allclose(T, 1.0, atol=1e-9)

