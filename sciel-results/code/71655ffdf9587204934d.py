import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Target tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    target = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m+j, j*p+k, k*n+i] = 1.0
    
    def check_decomposition(U, V, W, atol=1e-9):
        T_approx = np.einsum('ak,bk,ck->abc', U, V, W)
        return np.allclose(T_approx, target, atol=atol)
    
    def compute_error(U, V, W):
        T_approx = np.einsum('ak,bk,ck->abc', U, V, W)
        return np.sum((T_approx - target)**2)
    
    def schoolbook_init(r):
        """Initialize with schoolbook decomposition padded to rank r."""
        U = np.zeros((nm, r))
        V = np.zeros((mp, r))
        W = np.zeros((pn, r))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    col = i*m*p + j*p + k
                    if col < r:
                        U[i*m+j, col] = 1.0
                        V[j*p+k, col] = 1.0
                        W[k*n+i, col] = 1.0
        return U, V, W
    
    def try_rank_with_optimization(r, num_restarts=50):
        if time.time() > deadline:
            return None
        
        best_err = float('inf')
        best_params = None
        
        size = r * (nm + mp + pn)
        
        for restart in range(num_restarts):
            if time.time() > deadline - 10:
                break
            
            # Multiple initialization strategies
            if restart % 5 == 0:
                # Pure random
                params = np.random.randn(size) * 0.5
            elif restart % 5 == 1:
                # Schoolbook + noise
                U_init, V_init, W_init = schoolbook_init(r)
                params = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
                params += np.random.randn(size) * 0.1
            elif restart % 5 == 2:
                # Sparse random
                params = np.random.randn(size)
                mask = np.random.rand(size) > 0.6
                params[mask] = 0
                params *= 0.8
            elif restart % 5 == 3:
                # Structured: combine pairs of schoolbook terms
                U_init = np.zeros((nm, r))
                V_init = np.zeros((mp, r))
                W_init = np.zeros((pn, r))
                # Use first r-1 as schoolbook, last one as combination
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            col = i*m*p + j*p + k
                            if col < r-1:
                                U_init[i*m+j, col] = 1.0
                                V_init[j*p+k, col] = 1.0
                                W_init[k*n+i, col] = 1.0
                # Last column: random combination
                U_init[:, r-1] = np.random.randn(nm) * 0.3
                V_init[:, r-1] = np.random.randn(mp) * 0.3
                W_init[:, r-1] = np.random.randn(pn) * 0.3
                params = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            else:
                # Random with bias toward sparse
                params = np.random.randn(size) * 0.7
                params[np.abs(params) < 0.3] = 0
            
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
                              options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxfun': 2000})
            except Exception:
                continue
            
            if time.time() > deadline - 10:
                break
            
            err = res.fun
            if err < best_err:
                best_err = err
                best_params = res.x.copy()
                
                if err < 1e-12:
                    U = res.x[:r*nm].reshape(nm, r)
                    V = res.x[r*nm:r*(nm+mp)].reshape(mp, r)
                    W = res.x[r*(nm+mp):].reshape(pn, r)
                    
                    # Try rounding to nearest 0.5
                    U_rounded = np.round(U * 2) / 2
                    V_rounded = np.round(V * 2) / 2
                    W_rounded = np.round(W * 2) / 2
                    
                    if check_decomposition(U_rounded, V_rounded, W_rounded):
                        return U_rounded, V_rounded, W_rounded
                    elif check_decomposition(U, V, W):
                        return U, V, W
        
        return None
    
    def try_rank_with_differential_evolution(r, max_iter=100):
        if time.time() > deadline:
            return None
        
        size = r * (nm + mp + pn)
        
        def objective(params):
            U = params[:r*nm].reshape(nm, r)
            V = params[r*nm:r*(nm+mp)].reshape(mp, r)
            W = params[r*(nm+mp):].reshape(pn, r)
            return compute_error(U, V, W)
        
        bounds = [(-3, 3)] * size
        
        try:
            res = differential_evolution(objective, bounds, maxiter=max_iter, 
                                         tol=1e-14, seed=42, polish=True, 
                                         popsize=15, mutation=(0.5, 1.0))
        except Exception:
            return None
        
        if time.time() > deadline - 10:
            return None
        
        if res.fun < 1e-12:
            U = res.x[:r*nm].reshape(nm, r)
            V = res.x[r*nm:r*(nm+mp)].reshape(mp, r)
            W = res.x[r*(nm+mp):].reshape(pn, r)
            
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            if check_decomposition(U_rounded, V_rounded, W_rounded):
                return U_rounded, V_rounded, W_rounded
            elif check_decomposition(U, V, W):
                return U, V, W
        
        return None
    
    # Try rank 23 first
    if time.time() < deadline - 60:
        res = try_rank_with_optimization(23, num_restarts=40)
        if res is not None:
            U, V, W = res
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try differential evolution for rank 23
    if time.time() < deadline - 30:
        res = try_rank_with_differential_evolution(23, max_iter=50)
        if res is not None:
            U, V, W = res
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to rank 24
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
# EVOLVE-BLOCK-END
