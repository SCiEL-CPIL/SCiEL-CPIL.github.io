import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    def verify(U, V, W, tol=1e-8):
        R = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r] for r in range(R))
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def objective_and_grad(params, R):
        size_U = n * m * R
        size_V = m * p * R
        size_W = p * n * R
        
        U = params[:size_U].reshape((n*m, R))
        V = params[size_U:size_U + size_V].reshape((m*p, R))
        W = params[size_U + size_V:].reshape((p*n, R))
        
        # Compute T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        # and error = sum (T[i,j,k] - 1)^2
        # Also compute gradient
        
        T = np.zeros((n, m, p))
        for r in range(R):
            Ur = U[:, r]  # (n*m,)
            Vr = V[:, r]  # (m*p,)
            Wr = W[:, r]  # (p*n,)
            
            for i in range(n):
                for j in range(m):
                    u_val = Ur[i*m + j]
                    for k in range(p):
                        T[i, j, k] += u_val * Vr[j*p + k] * Wr[k*n + i]
        
        error = np.sum((T - 1.0) ** 2)
        
        # Gradient computation
        # d(error)/d(U[i*m+j, r]) = 2 * (T[i,j,k] - 1) * V[j*p+k, r] * W[k*n+i, r]  for each k
        # Actually: error = sum_{i,j,k} (sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] - 1)^2
        # d(error)/d(U[i*m+j, r]) = sum_k 2*(T[i,j,k]-1) * V[j*p+k, r] * W[k*n+i, r]
        
        grad = np.zeros_like(params)
        size_U = n * m * R
        size_V = m * p * R
        
        for i in range(n):
            for j in range(m):
                for r in range(R):
                    g = 0.0
                    for k in range(p):
                        g += 2.0 * (T[i, j, k] - 1.0) * V[j*p + k, r] * W[k*n + i, r]
                    grad[i*m + j + r * n * m] = g  # Wait, need to be careful with indexing
        
        # Let me redo the gradient more carefully
        # U is (n*m, R), so U[i*m+j, r] is at position (i*m+j)*R + r in flat array
        # V is (m*p, R), so V[j*p+k, r] is at position size_U + (j*p+k)*R + r
        # W is (p*n, R), so W[k*n+i, r] is at position size_U + size_V + (k*n+i)*R + r
        
        grad = np.zeros_like(params)
        
        for i in range(n):
            for j in range(m):
                for r in range(R):
                    gU = 0.0
                    gV = 0.0
                    gW = 0.0
                    for k in range(p):
                        diff = T[i, j, k] - 1.0
                        gU += 2.0 * diff * V[j*p + k, r] * W[k*n + i, r]
                        gV += 2.0 * diff * U[i*m + j, r] * W[k*n + i, r]
                        gW += 2.0 * diff * U[i*m + j, r] * V[j*p + k, r]
                    grad[(i*m + j) * R + r] = gU
                    grad[size_U + (j*p + k) * R + r] = gV  # This is wrong, k is from last iteration
                    grad[size_U + size_V + (k*n + i) * R + r] = gW  # Same issue
        
        # The gradient computation above has a bug. Let me fix it.
        # For V[j*p+k, r], the gradient w.r.t. V[j*p+k, r] is:
        # sum_i 2*(T[i,j,k]-1) * U[i*m+j, r] * W[k*n+i, r]
        # For W[k*n+i, r], the gradient w.r.t. W[k*n+i, r] is:
        # sum_j 2*(T[i,j,k]-1) * U[i*m+j, r] * V[j*p+k, r]
        
        grad = np.zeros_like(params)
        
        for r in range(R):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        diff = 2.0 * (T[i, j, k] - 1.0)
                        u_val = U[i*m + j, r]
                        v_val = V[j*p + k, r]
                        w_val = W[k*n + i, r]
                        
                        grad[(i*m + j) * R + r] += diff * v_val * w_val
                        grad[size_U + (j*p + k) * R + r] += diff * u_val * w_val
                        grad[size_U + size_V + (k*n + i) * R + r] += diff * u_val * v_val
        
        return error, grad

    def search_rank(R, max_restarts=5):
        best_result = None
        best_error = float('inf')
        
        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(123 + restart * 7)
            
            # Structured initialization: start with schoolbook-like, then perturb
            init_U = np.zeros((n*m, R))
            init_V = np.zeros((m*p, R))
            init_W = np.zeros((p*n, R))
            
            # Fill with schoolbook pattern for first min(R, n*m*p) columns
            num_schoolbook = min(R, n * m * p)
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if col < num_schoolbook:
                            init_U[i*m + j, col] = 1.0
                            init_V[j*p + k, col] = 1.0
                            init_W[k*n + i, col] = 1.0
                            col += 1
            
            # Add random perturbation to all entries
            scale = 0.5
            init_U += np.random.uniform(-scale, scale, init_U.shape)
            init_V += np.random.uniform(-scale, scale, init_V.shape)
            init_W += np.random.uniform(-scale, scale, init_W.shape)
            
            params = np.concatenate([init_U.flatten(), init_V.flatten(), init_W.flatten()])
            
            # Run optimization
            try:
                result = minimize(
                    lambda x: objective_and_grad(x, R)[0],
                    params,
                    jac=lambda x: objective_and_grad(x, R)[1],
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'gtol': 1e-6}
                )
                
                if result.success or result.fun < best_error:
                    best_error = result.fun
                    if best_error < 1e-10:
                        U = result.x[:n*m*R].reshape((n*m, R))
                        V = result.x[n*m*R:n*m*R+m*p*R].reshape((m*p, R))
                        W = result.x[n*m*R+m*p*R:].reshape((p*n, R))
                        if verify(U, V, W, tol=1e-8):
                            return U, V, W
            except Exception as e:
                pass
        
        return None

    # Try ranks from 35 down to 30
    for R in [35, 34, 33, 32, 31, 30]:
        if time.time() > deadline - 20:
            break
        
        result = search_rank(R)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: Schoolbook
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
