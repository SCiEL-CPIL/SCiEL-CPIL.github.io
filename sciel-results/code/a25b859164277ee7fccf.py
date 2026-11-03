import numpy as np
from scipy.optimize import minimize
import time

DIMENSIONS = [3, 4, 5]

def _build_target_tensor():
    """Build the matrix multiplication tensor Y[a, b, c] where a=i*m+j, b=j*p+k, c=k*n+i."""
    n, m, p = DIMENSIONS
    Y = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                Y[a, b, c] = 1.0
    return Y

def _schoolbook():
    """Return the schoolbook decomposition of rank n*m*p."""
    n, m, p = DIMENSIONS
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

def _verify_decomposition(U, V, W, Y, tol=1e-9):
    """Verify that the decomposition exactly matches the target tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    T = np.einsum('ar,br,cr->abc', U, V, W)
    return np.max(np.abs(T - Y)) < tol

def _try_optimization(rank, Y, n, m, p, deadline, max_restarts=5, verbose=False):
    """Try to find a decomposition of given rank using L-BFGS-B optimization."""
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        np.random.seed(42 + rank * 100 + restart)
        U = np.random.randn(n * m, rank) * 0.1
        V = np.random.randn(m * p, rank) * 0.1
        W = np.random.randn(p * n, rank) * 0.1
        
        def objective(params):
            U_flat = params[:n * m * rank].reshape(n * m, rank)
            V_flat = params[n * m * rank : n * m * rank + m * p * rank].reshape(m * p, rank)
            W_flat = params[n * m * rank + m * p * rank :].reshape(p * n, rank)
            T = np.einsum('ar,br,cr->abc', U_flat, V_flat, W_flat)
            diff = T - Y
            return np.sum(diff * diff)
        
        def gradient(params):
            U_flat = params[:n * m * rank].reshape(n * m, rank)
            V_flat = params[n * m * rank : n * m * rank + m * p * rank].reshape(m * p, rank)
            W_flat = params[n * m * rank + m * p * rank :].reshape(p * n, rank)
            T = np.einsum('ar,br,cr->abc', U_flat, V_flat, W_flat)
            diff = T - Y
            gU = np.einsum('abc,br,cr->ar', diff, V_flat, W_flat)
            gV = np.einsum('abc,ar,cr->br', diff, U_flat, W_flat)
            gW = np.einsum('abc,ar,br->cr', diff, U_flat, V_flat)
            return np.concatenate([gU.flatten(), gV.flatten(), gW.flatten()])
        
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B', jac=gradient, 
                           options={'maxiter': 200, 'maxfun': 200, 'gtol': 1e-8})
            
            if res.fun < 1e-5:
                U_opt = res.x[:n * m * rank].reshape(n * m, rank)
                V_opt = res.x[n * m * rank : n * m * rank + m * p * rank].reshape(m * p, rank)
                W_opt = res.x[n * m * rank + m * p * rank :].reshape(p * n, rank)
                
                # Try rounding to nearest half-integer
                U_int = np.round(U_opt * 2) / 2
                V_int = np.round(V_opt * 2) / 2
                W_int = np.round(W_opt * 2) / 2
                
                if _verify_decomposition(U_int, V_int, W_int, Y):
                    return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
                
                # Try rounding to nearest integer
                U_int2 = np.round(U_opt)
                V_int2 = np.round(V_opt)
                W_int2 = np.round(W_opt)
                
                if _verify_decomposition(U_int2, V_int2, W_int2, Y):
                    return {"U": U_int2.tolist(), "V": V_int2.tolist(), "W": W_int2.tolist()}
                
                # Try quarter-integers
                U_int3 = np.round(U_opt * 4) / 4
                V_int3 = np.round(V_opt * 4) / 4
                W_int3 = np.round(W_opt * 4) / 4
                
                if _verify_decomposition(U_int3, V_int3, W_int3, Y):
                    return {"U": U_int3.tolist(), "V": V_int3.tolist(), "W": W_int3.tolist()}
                    
        except Exception:
            pass
    
    return None

def _try_strassen_block_decomposition(Y, n, m, p, deadline):
    """
    Try to exploit Strassen-like block structure.
    For 3x4x5, we can try to decompose using known efficient algorithms.
    
    Strategy: Use the fact that matrix multiplication can be decomposed 
    into blocks, and apply Strassen's algorithm where beneficial.
    
    For a 3x4x5 multiplication, we can think of it as:
    - Split the 3 rows into groups, 4 columns into groups, etc.
    
    Actually, let's try a more direct approach: search for low-rank 
    decompositions using the structure of the problem.
    
    Known results:
    - R(2,2,2) = 7 (Strassen)
    - R(3,3,3) <= 23 (Landsberg)
    
    For 3x4x5, we can try to construct a decomposition by combining
    smaller known optimal decompositions.
    """
    # Try ranks from a reasonable lower bound up to schoolbook
    # Lower bound: max(n*m, m*p, p*n) = max(12, 20, 15) = 20
    # But tensor rank is usually higher
    
    # Try a range of ranks, prioritizing lower ones
    for rank in range(25, 60):
        if time.time() > deadline:
            break
        
        result = _try_optimization(rank, Y, n, m, p, deadline, max_restarts=3)
        if result is not None:
            return result
    
    return None

def construct():
    n, m, p = DIMENSIONS
    Y = _build_target_tensor()
    
    start_time = time.time()
    deadline = start_time + 130  # Safety margin for 150s limit
    
    # First, try the structured Strassen-inspired approach
    result = _try_strassen_block_decomposition(Y, n, m, p, deadline)
    if result is not None:
        return result
    
    # If that didn't work, try more aggressive optimization with more restarts
    # for a few specific promising ranks
    for rank in [30, 35, 40, 45, 50, 55]:
        if time.time() > deadline:
            break
        
        result = _try_optimization(rank, Y, n, m, p, deadline, max_restarts=10)
        if result is not None:
            return result
    
    # Fallback to schoolbook
    return _schoolbook()
