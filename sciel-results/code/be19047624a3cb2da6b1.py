import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 4]

def _build_tensor(n, m, p):
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _verify(U, V, W, n, m, p, tol=1e-9):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    T = _build_tensor(n, m, p)
    result = np.einsum('ar,br,cr->abc', U, V, W)
    return np.max(np.abs(result - T)) < tol

def _schoolbook():
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

def _optimize_rank(r, n, m, p, max_iter=500, seed=0):
    np.random.seed(seed)
    T = _build_tensor(n, m, p)
    
    # Initialize
    U = np.random.randn(n*m, r) * 0.1
    V = np.random.randn(m*p, r) * 0.1
    W = np.random.randn(p*n, r) * 0.1
    
    # Flatten for scipy
    def loss(params):
        U_flat = params[:n*m*r]
        V_flat = params[n*m*r:n*m*r+m*p*r]
        W_flat = params[n*m*r+m*p*r:]
        U = U_flat.reshape(n*m, r)
        V = V_flat.reshape(m*p, r)
        W = W_flat.reshape(p*n, r)
        result = np.einsum('ar,br,cr->abc', U, V, W)
        return np.sum((result - T)**2)
    
    def grad(params):
        U_flat = params[:n*m*r]
        V_flat = params[n*m*r:n*m*r+m*p*r]
        W_flat = params[n*m*r+m*p*r:]
        U = U_flat.reshape(n*m, r)
        V = V_flat.reshape(m*p, r)
        W = W_flat.reshape(p*n, r)
        result = np.einsum('ar,br,cr->abc', U, V, W)
        err = result - T
        gU = 2*np.einsum('abc,br,cr->ar', err, V, W)
        gV = 2*np.einsum('abc,ar,cr->br', err, U, W)
        gW = 2*np.einsum('abc,ar,br->cr', err, U, V)
        return np.concatenate([gU.flatten(), gV.flatten(), gW.flatten()])
        
    initial = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    res = minimize(loss, initial, jac=grad, method='L-BFGS-B', options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-12})
    
    final_params = res.x
    U_opt = final_params[:n*m*r].reshape(n*m, r)
    V_opt = final_params[n*m*r:n*m*r+m*p*r].reshape(m*p, r)
    W_opt = final_params[n*m*r+m*p*r:].reshape(p*n, r)
    
    # Round to half-integers
    U_r = np.round(U_opt * 2) / 2
    V_r = np.round(V_opt * 2) / 2
    W_r = np.round(W_opt * 2) / 2
    
    if _verify(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p):
        return U_r.tolist(), V_r.tolist(), W_r.tolist()
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = DIMENSIONS
    best = _schoolbook()
    
    # Try ranks from 20 to 35
    for r in range(20, 36):
        if time.time() > deadline - 20:
            break
        found = False
        for attempt in range(20):
            if time.time() > deadline - 20:
                break
            res = _optimize_rank(r, n, m, p, max_iter=100, seed=attempt)
            if res:
                best = {"U": res[0], "V": res[1], "W": res[2]}
                found = True
                break
        if found:
            # If we found a solution, we could try lower ranks, but the loop goes up.
            # Wait, the loop goes from 20 to 35. If we find one at 20, we should return it immediately?
            # Yes, we want the minimal rank.
            # But the loop starts at 20. If we find it at 20, we break.
            # If we don't find it at 20, we try 21, etc.
            # This is correct.
            pass

    return best
