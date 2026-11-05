import numpy as np
import time
from itertools import combinations, product
import sys
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 7]

def _build_target_tensor(n, m, p):
    """Build the target matrix multiplication tensor."""
    target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                target[a, b, c] = 1.0
    return target

def _compute_tensor(U, V, W):
    """Compute the tensor from factor matrices."""
    return np.einsum('ar,br,cr->abc', U, V, W)

def _objective_and_grad(params, n, m, p, target, r):
    """Objective function and gradient for L-BFGS-B.
    
    params: flat array of length 3 * (dim * r)
    Returns (objective_value, gradient)
    """
    U_size = n * m * r
    V_size = m * p * r
    W_size = p * n * r
    
    U = params[:U_size].reshape(n*m, r)
    V = params[U_size:U_size+V_size].reshape(m*p, r)
    W = params[U_size+V_size:].reshape(p*n, r)
    
    T = _compute_tensor(U, V, W)
    diff = T - target
    
    obj = np.sum(diff * diff)
    
    # Gradients: d/dU[a,r] = sum_{b,c} diff[a,b,c] * V[b,r] * W[c,r]
    grad_U = np.einsum('abc,br,cr->ar', diff, V, W)
    grad_V = np.einsum('abc,ar,cr->br', diff, U, W)
    grad_W = np.einsum('abc,ar,br->cr', diff, U, V)
    
    grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
    
    return obj, grad

def _try_rank_with_lbfgsb(r, n, m, p, target, max_iter=200, timeout_remaining=None):
    """Try to find a rank-r decomposition using L-BFGS-B."""
    start_time = time.time()
    
    U_size = n * m * r
    V_size = m * p * r
    W_size = p * n * r
    total_size = U_size + V_size + W_size
    
    # Multiple random initializations
    for attempt in range(3):
        if timeout_remaining is not None and time.time() - start_time > timeout_remaining:
            break
        
        # Initialize with small random values
        np.random.seed(42 + attempt)
        U = np.random.randn(n*m, r) * 0.5
        V = np.random.randn(m*p, r) * 0.5
        W = np.random.randn(p*n, r) * 0.5
        
        # Scale to help convergence
        U = U / np.linalg.norm(U) * np.sqrt(r)
        V = V / np.linalg.norm(V) * np.sqrt(r)
        W = W / np.linalg.norm(W) * np.sqrt(r)
        
        params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        # Use L-BFGS-B with the objective and gradient
        result = minimize(
            _objective_and_grad,
            params,
            args=(n, m, p, target, r),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        if timeout_remaining is not None and time.time() - start_time > timeout_remaining:
            break
        
        # Extract final parameters
        U_opt = result.x[:U_size].reshape(n*m, r)
        V_opt = result.x[U_size:U_size+V_size].reshape(m*p, r)
        W_opt = result.x[U_size+V_size:].reshape(p*n, r)
        
        # Check if we achieved a good solution
        T = _compute_tensor(U_opt, V_opt, W_opt)
        error = np.sum((T - target)**2)
        
        if error < 1e-8:
            # Try to snap to half-integers
            U_snap = np.round(U_opt * 2) / 2
            V_snap = np.round(V_opt * 2) / 2
            W_snap = np.round(W_opt * 2) / 2
            
            T_snap = _compute_tensor(U_snap, V_snap, W_snap)
            if np.sum((T_snap - target)**2) < 1e-6:
                return U_snap.tolist(), V_snap.tolist(), W_snap.tolist()
            
            # Try integer snapping
            U_int = np.round(U_opt)
            V_int = np.round(V_opt)
            W_int = np.round(W_opt)
            
            T_int = _compute_tensor(U_int, V_int, W_int)
            if np.sum((T_int - target)**2) < 1e-6:
                return U_int.tolist(), V_int.tolist(), W_int.tolist()
            
            # Return as-is if close enough
            return U_opt.tolist(), V_opt.tolist(), W_opt.tolist()
    
    return None

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    rank = len(W[0]) if W else 0
    if U_np.shape != (n*m, rank) or V_np.shape != (m*p, rank) or W_np.shape != (p*n, rank):
        return False
    
    T = _compute_tensor(U_np, V_np, W_np)
    
    # Check required entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                if abs(T[a,b,c] - 1.0) > tol:
                    return False
    
    # Check zeros elsewhere
    target = _build_target_tensor(n, m, p)
    if np.sum((T - target)**2) > tol:
        return False
    
    return True

def _schoolbook_decomposition(n, m, p):
    """Return the schoolbook rank-n*m*p decomposition."""
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

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 10 second safety margin
    
    target = _build_target_tensor(n, m, p)
    schoolbook_rank = n * m * p
    
    # Try ranks from low to high, stopping at schoolbook rank
    for r in range(20, schoolbook_rank):
        if time.time() > deadline:
            break
        
        timeout_remaining = deadline - time.time()
        if timeout_remaining < 2:
            break
        
        try:
            result = _try_rank_with_lbfgsb(r, n, m, p, target, 
                                           max_iter=100, 
                                           timeout_remaining=timeout_remaining)
            if result is not None:
                U, V, W = result
                # Verify before returning
                if _verify_decomposition(U, V, W, n, m, p):
                    return {"U": U, "V": V, "W": W}
        except Exception:
            continue
    
    # Fallback to schoolbook
    return _schoolbook_decomposition(n, m, p)
