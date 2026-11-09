"""Optimized tensor decomposition for 3x3 matrix multiplication."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 3]

# The matrix multiplication tensor T[i,j,k,l] = delta_{j,k} * delta_{i,l}
# We want rank r such that:
# T[i,j,k,l] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
# With n=m=p=3, this means:
# delta_{j,k} * delta_{i,l} = sum_c U[i*3+j, c] * V[j*3+k, c] * W[k*3+i, c]

def _verify(U, V, W, n=3, m=3, p=3, tol=1e-9):
    """Check if the decomposition exactly matches the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Reconstruct tensor
    T = np.zeros((n, m, p, n))
    for c in range(rank):
        uc = U[:, c]  # n*m entries
        vc = V[:, c]  # m*p entries
        wc = W[:, c]  # p*n entries
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k, i] += uc[i*m+j] * vc[j*p+k] * wc[k*n+i]
    
    # Target tensor: T[i,j,k,l] = delta_{j,k} * delta_{i,l}
    # Only non-zero when j==k and i==l
    errors = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0 if (j == k) else 0.0
                actual = T[i, j, k, i]
                errors += (actual - expected) ** 2
                # For l != i, should be 0
                for l in range(n):
                    if l != i:
                        # T[i,j,k,l] should be 0, but we only computed T[i,j,k,i]
                        # Actually the tensor is T[i,j,k,l] = delta_{j,k}*delta_{i,l}
                        # Our reconstruction only gives us T[i,j,k,i] via W[k*n+i]
                        # We need to check all l values
                        pass
    
    # More careful verification
    T_full = np.zeros((n, m, p, n))
    for c in range(rank):
        uc = U[:, c]
        vc = V[:, c]
        wc = W[:, c]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for l in range(n):
                        T_full[i, j, k, l] += uc[i*m+j] * vc[j*p+k] * wc[k*n+l]
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for l in range(n):
                    expected = 1.0 if (j == k and i == l) else 0.0
                    max_err = max(max_err, abs(T_full[i, j, k, l] - expected))
    
    return max_err < tol

def _schoolbook():
    """Return the schoolbook decomposition of rank 27."""
    n = m = p = 3
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    best_result = _schoolbook()
    best_rank = 27
    
    # Try to find lower rank decompositions using optimization
    # The known lower bound for 3x3 matrix multiplication rank is 19
    # Best known upper bound is 23 (Bini et al.)
    
    # We'll try to find a rank 20-23 decomposition using gradient-based optimization
    # with random restarts
    
    n = m = p = 3
    
    for target_rank in [20, 21, 22, 23]:
        if time.time() > deadline:
            break
        
        found = False
        max_restarts = 10
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            # Random initialization
            np.random.seed(42 + restart * 1000 + target_rank)
            
            U = np.random.randn(n*m, target_rank)
            V = np.random.randn(m*p, target_rank)
            W = np.random.randn(p*n, target_rank)
            
            # Objective: minimize sum of squared errors
            def objective(params):
                U_flat = params[:n*m*target_rank].reshape(n*m, target_rank)
                V_flat = params[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                W_flat = params[n*m*target_rank + m*p*target_rank :].reshape(p*n, target_rank)
                
                # Compute tensor
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for c in range(target_rank):
                                val += U_flat[i*m+j, c] * V_flat[j*p+k, c] * W_flat[k*n+i, c]
                            expected = 1.0 if j == k else 0.0
                            err += (val - expected) ** 2
                            for l in range(n):
                                if l != i:
                                    val2 = 0.0
                                    for c in range(target_rank):
                                        val2 += U_flat[i*m+j, c] * V_flat[j*p+k, c] * W_flat[k*n+l, c]
                                    err += val2 ** 2
                return err
            
            params0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            res = minimize(objective, params0, method='L-BFGS-B', options={'maxiter': 200, 'maxls': 50})
            
            if res.fun < 1e-10:
                U_opt = res.x[:n*m*target_rank].reshape(n*m, target_rank)
                V_opt = res.x[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                W_opt = res.x[n*m*target_rank + m*p*target_rank :].reshape(p*n, target_rank)
                
                # Verify
                if _verify(U_opt, V_opt, W_opt):
                    best_result = {
                        "U": U_opt.tolist(),
                        "V": V_opt.tolist(),
                        "W": W_opt.tolist()
                    }
                    best_rank = target_rank
                    found = True
                    break
            
            if time.time() > deadline:
                break
        
        if found:
            break
    
    return best_result
