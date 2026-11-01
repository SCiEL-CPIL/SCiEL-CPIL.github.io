import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Target tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    target = np.zeros((nm, mp, pn))
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx = (i*m+j, j*p+k, k*n+i)
                target[idx] = 1.0
                target_indices.append(idx)
                
    def compute_error(U, V, W):
        T_approx = np.einsum('ak,bk,ck->abc', U, V, W)
        return np.sum((T_approx - target)**2)
    
    def check_decomposition(U, V, W, atol=1e-9):
        T_approx = np.einsum('ak,bk,ck->abc', U, V, W)
        return np.allclose(T_approx, target, atol=atol)
    
    def polish_solution(U, V, W, r, max_iters=100, tol=1e-14):
        """High-precision polish using Gauss-Newton style iterations."""
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            T_approx = np.einsum('ak,bk,ck->abc', U, V, W)
            diff = T_approx - target
            err = np.sum(diff**2)
            if err < tol:
                break
            
            # Compute gradients
            dU = np.einsum('abc,bk,ck->ak', diff, V, W)
            dV = np.einsum('abc,ak,ck->bk', diff, U, W)
            dW = np.einsum('abc,ak,bk->ck', diff, U, V)
            
            # Use a simple damped Newton step
            # Approximate Hessian as 2 * (outer products of gradients)
            # For simplicity, use a fixed small step size
            step = 0.05
            U_new = U - step * dU
            V_new = V - step * dV
            W_new = W - step * dW
            
            err_new = compute_error(U_new, V_new, W_new)
            if err_new < err:
                U, V, W = U_new, V_new, W_new
                err = err_new
            else:
                # Try smaller step
                step *= 0.5
                U_new = U - step * dU
                V_new = V - step * dV
                W_new = W - step * dW
                err_new = compute_error(U_new, V_new, W_new)
                if err_new < err:
                    U, V, W = U_new, V_new, W_new
                    err = err_new
                else:
                    break
        return U, V, W, err
    
    def try_rank(r, num_restarts=20):
        if time.time() > deadline:
            return None
        
        best_err = float('inf')
        best_params = None
        
        size = r * (nm + mp + pn)
        
        for restart in range(num_restarts):
            if time.time() > deadline - 15:
                break
                
            # Diverse initialization strategies
            if restart == 0:
                # Structured: use schoolbook-like pattern with noise
                params = np.zeros(size)
                # Initialize with sparse pattern
                for t in range(r):
                    # Random assignment of non-zero entries
                    u_idx = np.random.choice(nm, size=min(3, nm), replace=False)
                    v_idx = np.random.choice(mp, size=min(3, mp), replace=False)
                    w_idx = np.random.choice(pn, size=min(3, pn), replace=False)
                    for idx in u_idx:
                        params[idx * r + t] = np.random.choice([-1, 1])
                    for idx in v_idx:
                        params[(nm + idx) * r + t] = np.random.choice([-1, 1])
                    for idx in w_idx:
                        params[(nm + mp + idx) * r + t] = np.random.choice([-1, 1])
            elif restart == 1:
                # Start from scaled schoolbook and project
                params = np.random.randn(size) * 0.1
            elif restart % 3 == 0:
                # Random with small variance
                params = np.random.randn(size) * 0.3
            elif restart % 3 == 1:
                # Sparse random
                params = np.random.randn(size)
                mask = np.random.rand(size) > 0.8
                params[mask] = 0
                params *= 0.5
            else:
                # Mixed initialization
                params = np.random.randn(size) * 0.2
                # Add some structured components
                for t in range(min(3, r)):
                    params[np.random.randint(nm) * r + t] += np.random.choice([-1, 1]) * 0.5
                    params[(nm + np.random.randint(mp)) * r + t] += np.random.choice([-1, 1]) * 0.5
                    params[(nm + mp + np.random.randint(pn)) * r + t] += np.random.choice([-1, 1]) * 0.5
            
            def objective(params):
                U = params[:r*nm].reshape(nm, r)
                V = params[r*nm:r*(nm+mp)].reshape(mp, r)
                W = params[r*(nm+mp):].reshape(pn, r)
                return compute_error(U, V, W)
            
            def gradient(params):
                U = params[:r*nm].reshape(nm, r)
                V = params[r*nm:r*(nm+mp)].reshape(mp, r)
                W = params[r*(nm+mp):].reshape(pn, r)
                
                T_approx = np.einsum('ak,bk,ck->abc', U, V, W)
                diff = T_approx - target
                
                dU = np.einsum('abc,bk,ck->ak', diff, V, W)
                dV = np.einsum('abc,ak,ck->bk', diff, U, W)
                dW = np.einsum('abc,ak,bk->ck', diff, U, V)
                
                grad = np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
                return grad
            
            try:
                res = minimize(objective, params, jac=gradient, method='L-BFGS-B', 
                              options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 1000})
                
                err = res.fun
                if err < best_err:
                    best_err = err
                    best_params = res.x.copy()
                    
                    U = res.x[:r*nm].reshape(nm, r)
                    V = res.x[r*nm:r*(nm+mp)].reshape(mp, r)
                    W = res.x[r*(nm+mp):].reshape(pn, r)
                    
                    # Polish pass
                    U_p, V_p, W_p, err_p = polish_solution(U.copy(), V.copy(), W.copy(), r)
                    if err_p < err:
                        err = err_p
                        U, V, W = U_p, V_p, W_p
                        best_params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
                    
                    if err < 1e-10:
                        # Try rounding to half-integers
                        U_rounded = np.round(U * 2) / 2
                        V_rounded = np.round(V * 2) / 2
                        W_rounded = np.round(W * 2) / 2
                        
                        if check_decomposition(U_rounded, V_rounded, W_rounded):
                            return U_rounded, V_rounded, W_rounded
                        elif check_decomposition(U, V, W):
                            return U, V, W
            except Exception:
                pass
        
        return None

    # Try ranks from low to high
    rank_configs = [
        (23, 25),
        (22, 20),
        (21, 15),
        (20, 10),
        (19, 8),
    ]
    
    for rank, restarts in rank_configs:
        if time.time() > deadline - 20:
            break
        res = try_rank(rank, num_restarts=restarts)
        if res is not None:
            U, V, W = res
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to rank 24 (Schoolbook)
    rank = n * m * p
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
