import time
import numpy as np
import scipy.optimize as opt
import itertools

DIMENSIONS = [3, 3, 3]

def _schoolbook(n, m, p):
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
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-9):
    """
    Verify if U, V, W exactly decompose the matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    T_expected[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i else 0.
    """
    r = U.shape[1]
    # Calculate T_approx
    # T_approx[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    T_approx = np.einsum('ar,br,cr->abc', U, V, W)
    
    # Build expected tensor
    T_exp = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_exp[i*m+j, j*p+k, k*n+i] = 1.0
                
    return np.allclose(T_approx, T_exp, atol=tol, rtol=0)

def _snap_to_half(x):
    """Snap values to nearest half-integer."""
    return np.round(x * 2) / 2

def _try_find_decomposition(n, m, p, target_rank, deadline, seed=0):
    dim_a, dim_b, dim_c = n*m, m*p, p*n
    T_exp = np.zeros((dim_a, dim_b, dim_c))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_exp[i*m+j, j*p+k, k*n+i] = 1.0

    rng = np.random.RandomState(seed)
    
    def objective_and_grad(params):
        # params: flattened U, V, W
        # U: dim_a x R, V: dim_b x R, W: dim_c x R
        U = params[0:dim_a*target_rank].reshape(dim_a, target_rank)
        V = params[dim_a*target_rank : dim_a*target_rank + dim_b*target_rank].reshape(dim_b, target_rank)
        W = params[dim_a*target_rank + dim_b*target_rank :].reshape(dim_c, target_rank)
        
        # T_approx = einsum('ar,br,cr->abc', U, V, W)
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        diff = T_approx - T_exp
        loss = np.sum(diff**2)
        
        # Gradient calculation
        # dL/dU = 2 * einsum('abc,b,c->ar', diff, V, W)
        # dL/dV = 2 * einsum('abc,a,c->br', diff, U, W)
        # dL/dW = 2 * einsum('abc,a,b->cr', diff, U, V)
        
        grad_U = 2 * np.einsum('abc,b,c->ar', diff, V, W)
        grad_V = 2 * np.einsum('abc,a,c->br', diff, U, W)
        grad_W = 2 * np.einsum('abc,a,b->cr', diff, U, V)
        
        grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        return loss, grad

    # Try a few restarts
    for restart in range(20):
        if time.time() > deadline - 5:
            break
            
        # Initialize
        # Strategy: Mix of random and structured init
        if restart == 0:
            # Start from schoolbook and drop columns? No, rank is fixed.
            # Just random small noise
            scale = 0.1
            U0 = rng.randn(dim_a, target_rank) * scale
            V0 = rng.randn(dim_b, target_rank) * scale
            W0 = rng.randn(dim_c, target_rank) * scale
        else:
            # Larger random init to escape local minima
            scale = 0.5
            U0 = rng.randn(dim_a, target_rank) * scale
            V0 = rng.randn(dim_b, target_rank) * scale
            W0 = rng.randn(dim_c, target_rank) * scale
            
        params0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
        
        # Optimize
        try:
            res = opt.minimize(
                objective_and_grad, 
                params0, 
                method='L-BFGS-B', 
                jac=True,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-9, 'maxfun': 1000}
            )
            
            if res.success or res.fun < 1e-8:
                params_opt = res.x
                U_opt = params_opt[0:dim_a*target_rank].reshape(dim_a, target_rank)
                V_opt = params_opt[dim_a*target_rank : dim_a*target_rank + dim_b*target_rank].reshape(dim_b, target_rank)
                W_opt = params_opt[dim_a*target_rank + dim_b*target_rank :].reshape(dim_c, target_rank)
                
                # Snap to half-integers
                U_snap = _snap_to_half(U_opt)
                V_snap = _snap_to_half(V_opt)
                W_snap = _snap_to_half(W_opt)
                
                # Verify snapped version
                if _verify(U_snap, V_snap, W_snap, n, m, p, tol=1e-9):
                    return U_snap, V_snap, W_snap
                    
                # Verify unsnapped version (in case snapping was too aggressive, though task asks for integer/half-integer)
                # The task says "entries that exactly decompose... integer/half-integer entries".
                # So we MUST return snapped if it works.
                
        except Exception:
            continue
            
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # Known minimal rank is 23
    target_rank = 23
    
    result = _try_find_decomposition(n, m, p, target_rank, deadline, seed=42)
    
    if result is not None:
        U, V, W = result
        # Final verification
        if _verify(U, V, W, n, m, p, tol=1e-9):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # Fallback to schoolbook (Rank 27)
    U, V, W = _schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
