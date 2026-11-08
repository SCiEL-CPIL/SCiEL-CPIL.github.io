"""Direct tensor decomposition for 2x3 x 3x6 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]

def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS
    # n=2, m=3, p=6
    # A is n x m = 2x3, B is m x p = 3x6, C is n x p = 2x6
    # U has shape (n*m, rank) = (6, rank)
    # V has shape (m*p, rank) = (18, rank)
    # W has shape (p*n, rank) = (12, rank)
    # Convention: W[k*n + i, r] where C = A * B, C[i,k] = sum_j A[i,j]*B[j,k]
    # So W is indexed by (k, i) -> flat index k*n + i
    
    # The tensor for matrix multiplication has entries T[i,j,k] = A[i,j]*B[j,k]
    # We need to decompose: C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor rank decomposition: C[i,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    # Trivial solution: rank = n*m*p = 36
    # Each (i,j,k) triple gets its own rank slot
    rank = n * m * p  # 36
    
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * m * p + j * p + k
                U[i*m + j, r] = 1.0
                V[j*p + k, r] = 1.0
                W[k*n + i, r] = 1.0
    
    def verify(U, V, W, n, m, p):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for r in range(len(U[0])):
                        total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(total - 1.0) > 1e-10:
                        return False
        return True
    
    best_rank = rank
    best_U, best_V, best_W = U, V, W
    
    # Try decreasing ranks
    for try_rank in range(rank - 1, 28, -1):
        if time.time() - t0 > 130:
            break
        try:
            success = _try_rank(try_rank, n, m, p)
            if success is not None:
                U_t, V_t, W_t = success
                if verify(U_t, V_t, W_t, n, m, p):
                    best_rank = try_rank
                    best_U, best_V, best_W = U_t, V_t, W_t
                    break  # Found the best, stop
        except Exception:
            continue
    
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}


def _try_rank(rank, n, m, p):
    """Try to find a rank-r decomposition using random initialization + optimization."""
    import numpy as np
    from scipy.optimize import minimize
    
    # Parameters: U (n*m x rank), V (m*p x rank), W (p*n, rank)
    num_params = rank * (n*m + m*p + p*n)
    
    def make_flat(U, V, W):
        return np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    def unpack_flat(x):
        size_U = n*m*rank
        size_V = m*p*rank
        size_W = p*n*rank
        U = x[:size_U].reshape(n*m, rank)
        V = x[size_U:size_U+size_V].reshape(m*p, rank)
        W = x[size_U+size_V:size_U+size_V+size_W].reshape(p*n, rank)
        return U, V, W
    
    def objective_and_grad(x):
        U, V, W = unpack_flat(x)
        
        # Compute residuals and gradients
        # For each (i,j,k): residual = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1
        # d/dU[i*m+j,r] = sum_k residual(i,j,k) * V[j*p+k,r] * W[k*n+i,r]
        
        loss = 0.0
        U_grad = np.zeros_like(U)
        V_grad = np.zeros_like(V)
        W_grad = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    res = 0.0
                    for r in range(rank):
                        res += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    res -= 1.0
                    loss += res * res
                    
                    for r in range(rank):
                        U_grad[i*m+j, r] += res * V[j*p+k, r] * W[k*n+i, r]
                        V_grad[j*p+k, r] += U[i*m+j, r] * res * W[k*n+i, r]
                        W_grad[k*n+i, r] += U[i*m+j, r] * V[j*p+k, r] * res
        
        grad = np.concatenate([U_grad.flatten(), V_grad.flatten(), W_grad.flatten()])
        return loss, grad
    
    rng = np.random.RandomState(42)
    
    for attempt in range(5):
        U = rng.randn(n*m, rank) * 0.1
        V = rng.randn(m*p, rank) * 0.1
        W = rng.randn(p*n, rank) * 0.1
        
        # Scale to help convergence
        U *= (n*m) ** 0.25
        V *= (m*p) ** 0.25
        W *= (p*n) ** 0.25
        
        x0 = make_flat(U, V, W)
        
        try:
            result = minimize(
                lambda x: objective_and_grad(x)[0],
                x0,
                jac=lambda x: objective_and_grad(x)[1],
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 10000}
            )
            
            if result.success or result.fun < 1e-6:
                U_final, V_final, W_final = unpack_flat(result.x)
                
                # Verify and round to half-integers if close
                max_res = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            res = 0.0
                            for r in range(rank):
                                res += U_final[i*m+j, r] * V_final[j*p+k, r] * W_final[k*n+i, r]
                            max_res = max(max_res, abs(res - 1.0))
                
                if max_res < 1e-4:
                    # Try rounding to nearest 0.5
                    U_rounded = np.round(U_final * 2) / 2
                    V_rounded = np.round(V_final * 2) / 2
                    W_rounded = np.round(W_final * 2) / 2
                    
                    max_res_rounded = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                res = 0.0
                                for r in range(rank):
                                    res += U_rounded[i*m+j, r] * V_rounded[j*p+k, r] * W_rounded[k*n+i, r]
                                max_res_rounded = max(max_res_rounded, abs(res - 1.0))
                    
                    if max_res_rounded < 1e-8:
                        return (U_rounded, V_rounded, W_rounded)
                    else:
                        return (U_final, V_final, W_final)
                
                return (U_final, V_final, W_final)
                
        except Exception:
            continue
    
    return None


def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition satisfies the tensor equation."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for r in range(len(U[0])):
                    total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(total - 1.0) > 1e-6:
                    return False
    return True
