import numpy as np
import time
import scipy.optimize as opt

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # safety margin before 150s limit
    
    # Build the target tensor T of shape (n*m, m*p, p*n)
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Schoolbook decomposition (rank 30)
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # Use a structured approach: try ranks from 29 down to 20
    for target_rank in range(best_rank - 1, 19, -1):
        if time.time() > deadline:
            break
        
        nU = n * m * target_rank
        nV = m * p * target_rank
        nW = p * n * target_rank
        total_vars = nU + nV + nW
        
        def objective_and_gradient(x):
            U = x[:nU].reshape(n*m, target_rank)
            V = x[nU:nU+nV].reshape(m*p, target_rank)
            W = x[nU+nV:].reshape(p*n, target_rank)
            
            S = np.einsum('ar,br,cr->abc', U, V, W)
            error = S - T
            val = np.sum(error**2)
            
            gU = 2.0 * np.einsum('abc,br,cr->ar', error, V, W)
            gV = 2.0 * np.einsum('abc,ar,cr->br', error, U, W)
            gW = 2.0 * np.einsum('abc,ar,br->cr', error, U, V)
            
            grad = np.concatenate([gU.flatten(), gV.flatten(), gW.flatten()])
            return val, grad
        
        # Number of restarts
        restarts = 20 if target_rank > 26 else 10
        
        for r in range(restarts):
            if time.time() > deadline:
                break
            
            # Smart initialization: start from schoolbook and perturb
            if target_rank < n*m*p:
                # Initialize with a subset of schoolbook terms plus random
                x0 = np.random.randn(total_vars) * 0.1
            else:
                x0 = np.random.randn(total_vars) * 0.1
            
            try:
                res = opt.minimize(
                    objective_and_gradient,
                    x0,
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'maxfun': 3000, 'ftol': 1e-14, 'gtol': 1e-10}
                )
            except Exception:
                continue
            
            if res.fun < 1e-10:
                U = res.x[:nU].reshape(n*m, target_rank)
                V = res.x[nU:nU+nV].reshape(m*p, target_rank)
                W = res.x[nU+nV:].reshape(p*n, target_rank)
                
                S = np.einsum('ar,br,cr->abc', U, V, W)
                if np.allclose(S, T, atol=1e-8):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        # Try with different random seeds
        for r in range(5):
            if time.time() > deadline:
                break
            np.random.seed(42 + r)
            x0 = np.random.randn(total_vars) * 0.05
            
            try:
                res = opt.minimize(
                    objective_and_gradient,
                    x0,
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'maxfun': 2000, 'ftol': 1e-14, 'gtol': 1e-10}
                )
            except Exception:
                continue
            
            if res.fun < 1e-10:
                U = res.x[:nU].reshape(n*m, target_rank)
                V = res.x[nU:nU+nV].reshape(m*p, target_rank)
                W = res.x[nU+nV:].reshape(p*n, target_rank)
                
                S = np.einsum('ar,br,cr->abc', U, V, W)
                if np.allclose(S, T, atol=1e-8):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return best_result
