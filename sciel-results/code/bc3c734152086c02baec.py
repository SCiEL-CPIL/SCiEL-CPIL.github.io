import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Target tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    target = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m+j, j*p+k, k*n+i] = 1.0

    def check_decomposition(U, V, W, atol=1e-9):
        """Check if U, V, W exactly decompose the target tensor."""
        T_approx = np.einsum('ak,bk,ck->abc', U, V, W)
        return np.allclose(T_approx, target, atol=atol)

    def compute_error(U, V, W):
        T_approx = np.einsum('ak,bk,ck->abc', U, V, W)
        return np.sum((T_approx - target)**2)

    def try_rank(r, num_restarts=100):
        """
        Attempt to find a decomposition of rank r.
        Returns (U, V, W) if found, else None.
        """
        if time.time() > deadline:
            return None
            
        best_err = float('inf')
        best_params = None
        
        size = r * (nm + mp + pn)
        
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                break
            
            # Initialization strategies
            if restart % 3 == 0:
                # Random sparse
                params = np.random.randn(size) * 0.5
                params[np.abs(params) < 0.2] = 0
            elif restart % 3 == 1:
                # Structured: Based on pairs of indices
                # This is a heuristic initialization for matrix multiplication
                U_init = np.zeros((nm, r))
                V_init = np.zeros((mp, r))
                W_init = np.zeros((pn, r))
                
                # Fill first r columns with random combinations of basis vectors
                for col in range(r):
                    # Pick random indices
                    i_idx = np.random.randint(0, n)
                    j_idx = np.random.randint(0, m)
                    k_idx = np.random.randint(0, p)
                    
                    # Add some noise to allow flexibility
                    U_init[i_idx * m + j_idx, col] = 1.0 + np.random.randn() * 0.1
                    V_init[j_idx * p + k_idx, col] = 1.0 + np.random.randn() * 0.1
                    W_init[k_idx * n + i_idx, col] = 1.0 + np.random.randn() * 0.1
                    
                    # Add secondary terms to handle overlaps
                    if np.random.rand() > 0.5:
                        i2 = np.random.randint(0, n)
                        j2 = np.random.randint(0, m)
                        k2 = np.random.randint(0, p)
                        U_init[i2 * m + j2, col] += np.random.randn() * 0.5
                        V_init[j2 * p + k2, col] += np.random.randn() * 0.5
                        W_init[k2 * n + i2, col] += np.random.randn() * 0.5
                        
                params = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            else:
                # Pure random
                params = np.random.randn(size) * 0.8

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
                              options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-12})
            except Exception:
                continue
            
            err = res.fun
            if err < best_err:
                best_err = err
                best_params = res.x.copy()
                
                if err < 1e-10:
                    U = res.x[:r*nm].reshape(nm, r)
                    V = res.x[r*nm:r*(nm+mp)].reshape(mp, r)
                    W = res.x[r*(nm+mp):].reshape(pn, r)
                    
                    # Try snapping to half-integers
                    # The problem states entries can be integer/half-integer
                    U_snap = np.round(U * 2) / 2
                    V_snap = np.round(V * 2) / 2
                    W_snap = np.round(W * 2) / 2
                    
                    if check_decomposition(U_snap, V_snap, W_snap):
                        return U_snap, V_snap, W_snap
                    elif check_decomposition(U, V, W):
                        # If exact float solution found, try to snap again with higher precision or return
                        # But we need integer/half-integer. If float solution is very close to half-int, snap.
                        # If it's a valid float solution but not half-integer, it's invalid for the contract.
                        # However, if the error is 0, it's a valid decomposition. 
                        # The contract says "integer/half-integer entries".
                        # So we must return half-integers.
                        # If the float solution is not close to half-integers, we might fail.
                        # But typically, optimal tensor decompositions for small matrices have simple coefficients.
                        pass
        
        return None

    # Fallback: Schoolbook decomposition
    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank), dtype=float)
        V = np.zeros((mp, rank), dtype=float)
        W = np.zeros((pn, rank), dtype=float)
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W

    # Try to find a better rank
    # Known lower bound is 12. Known rank is 18.
    # We try ranks from 18 up to 23. If we find one, we return it.
    # We prioritize lower ranks.
    for r in range(18, 24):
        if time.time() > deadline - 10:
            break
            
        result = try_rank(r)
        if result is not None:
            U, V, W = result
            # Convert to lists as per interface
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # If no better rank found, return schoolbook
    U, V, W = get_schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
