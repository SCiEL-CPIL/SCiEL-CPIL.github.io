"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction
from itertools import product
import scipy.optimize as opt

def _build_tensor(n, m, p):
    """Return the matrix multiplication tensor T[i,j,k] = (i*m+j, j*p+k, k*n+i) as a dict."""
    T = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[(i, j, k)] = (i * m + j, j * p + k, k * n + i)
    return T

def _check_decomposition(U, V, W, n, m, p, tol=1e-10):
    """
    Check if U, V, W (lists of lists) form a valid rank-r decomposition.
    U: (n*m, r), V: (m*p, r), W: (p*n, r)
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = np.dot(U[u_idx, :], np.multiply(V[v_idx, :], W[w_idx, :]))
                if abs(val - 1.0) > tol:
                    return False
    return True

def _loss_and_grad(params, n, m, p, r):
    """
    Compute loss and gradient for the decomposition problem.
    params: flattened array of size (n*m + m*p + p*n) * r
    """
    size_U = n * m * r
    size_V = m * p * r
    size_W = p * n * r
    
    U = params[:size_U].reshape(n * m, r)
    V = params[size_U:size_U + size_V].reshape(m * p, r)
    W = params[size_U + size_V:].reshape(p * n, r)
    
    # Compute T
    T = np.zeros((n, m, p))
    for c in range(r):
        Uc = U[:, c]
        Vc = V[:, c]
        Wc = W[:, c]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += Uc[i * m + j] * Vc[j * p + k] * Wc[k * n + i]
    
    err = 1.0 - T
    loss = np.sum(err ** 2)
    
    # Gradient
    dU = np.zeros_like(U)
    dV = np.zeros_like(V)
    dW = np.zeros_like(W)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                e = err[i, j, k]
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                for c in range(r):
                    dU[u_idx, c] += -2 * e * V[v_idx, c] * W[w_idx, c]
                    dV[v_idx, c] += -2 * e * U[u_idx, c] * W[w_idx, c]
                    dW[w_idx, c] += -2 * e * U[u_idx, c] * V[v_idx, c]
    
    grad = np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
    return loss, grad

def _try_rank_scipy(r, n, m, p, timeout=10.0):
    """
    Try to find a rank-r decomposition using scipy.optimize.minimize.
    """
    start_time = time.time()
    
    n_vars = (n * m + m * p + p * n) * r
    
    best_loss = float('inf')
    best_params = None
    
    # Try a few random initializations
    for seed in range(20):
        if time.time() - start_time > timeout:
            break
            
        np.random.seed(seed)
        # Initialize with small random values
        x0 = np.random.randn(n_vars) * 0.5
        
        # Define objective function
        def objective(x):
            loss, grad = _loss_and_grad(x, n, m, p, r)
            return loss, grad
        
        try:
            result = opt.minimize(
                objective, 
                x0, 
                method='L-BFGS-B', 
                jac=True, 
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
        except Exception:
            continue
            
        loss = result.fun
        if loss < best_loss:
            best_loss = loss
            best_params = result.x
            
        if best_loss < 1e-10:
            # Found a solution
            break
            
    if best_params is None:
        return None
        
    if best_loss > 1e-6:
        return None
        
    # Extract U, V, W
    size_U = n * m * r
    size_V = m * p * r
    size_W = p * n * r
    
    U = best_params[:size_U].reshape(n * m, r)
    V = best_params[size_U:size_U + size_V].reshape(m * p, r)
    W = best_params[size_U + size_V:].reshape(p * n, r)
    
    # Convert to Fractions
    U_f = [[Fraction(x).limit_denominator(1000) for x in row] for row in U]
    V_f = [[Fraction(x).limit_denominator(1000) for x in row] for row in V]
    W_f = [[Fraction(x).limit_denominator(1000) for x in row] for row in W]
    
    if _check_decomposition(U_f, V_f, W_f, n, m, p, tol=1e-9):
        return U_f, V_f, W_f
        
    return None

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    deadline = time.time() + 100
    
    # Try to find lower rank
    # Known: rank(2x3, 3x3) is 16.
    # Try 16, then 17.
    for r in [16, 17]:
        if time.time() > deadline - 10:
            break
            
        print(f"Trying rank {r}...")
        res = _try_rank_scipy(r, n, m, p, timeout=40.0)
        if res is not None:
            U, V, W = res
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            
            if _check_decomposition(U_list, V_list, W_list, n, m, p, tol=1e-9):
                print(f"Found rank {r} decomposition!")
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: Schoolbook
    rank = rank_schoolbook
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
# EVOLVE-BLOCK-END
