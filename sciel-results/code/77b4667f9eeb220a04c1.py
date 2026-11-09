"""Rewritten program: try to find a low-rank exact decomposition of the
3x3 * 3x8 matrix multiplication tensor, falling back to schoolbook."""
import time
import itertools
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 8]

# Deadline guard: 150 s wall-clock limit, use 135 s as safety margin.
DEADLINE = time.time() + 135.0

def _time_left():
    return max(0.0, DEADLINE - time.time())

def _build_tensor():
    """Build the matrix multiplication tensor for 3x3 * 3x8 -> 3x8.
    Indexing: U has size n*m = 9, V has size m*p = 24, W has size p*n = 24.
    T[a, b, c] = 1 if a=(i,j), b=(j,k), c=(k,i) with matching indices.
    Specifically: a = i*m + j, b = j*p + k, c = k*n + i.
    The product tensor T[a,b,c] = 1 when i=i', j=j', k=k'.
    """
    n, m, p = DIMENSIONS
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _verify_decomposition(U, V, W, T, tol=1e-9):
    """Check that sum_r U[:,r] V[:,r] W[:,r] == T (outer product sum)."""
    # U: (n*m, R), V: (m*p, R), W: (p*n, R)
    R = U.shape[1]
    reconstructed = np.zeros(T.shape)
    for r in range(R):
        reconstructed += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    return np.max(np.abs(reconstructed - T)) < tol

def _verify_decomposition_fast(U, V, W, T, tol=1e-9):
    """Faster verification using einsum."""
    # Sum over r of outer product
    reconstructed = np.einsum('ar,br,cr->abc', U, V, W)
    return np.max(np.abs(reconstructed - T)) < tol

def _try_random_search(T, max_rank, num_trials, seed=42):
    """Try random initialization with gradient descent for various ranks."""
    n, m, p = DIMENSIONS
    um, vp, pn = n * m, m * p, p * n
    rng = np.random.RandomState(seed)
    
    for R in range(max_rank, 23, -1):  # Try from max_rank down to 24
        if _time_left() < 5.0:
            break
        for trial in range(num_trials):
            if _time_left() < 3.0:
                break
            
            # Initialize with small random values
            U = rng.randn(um, R) * 0.5
            V = rng.randn(vp, R) * 0.5
            W = rng.randn(pn, R) * 0.5
            
            def objective(params):
                if _time_left() < 1.0:
                    return 1e10
                R_local = R
                U_l = params[:um * R_local].reshape(um, R_local)
                V_l = params[um * R_local:um * R_local + vp * R_local].reshape(vp, R_local)
                W_l = params[um * R_local + vp * R_local:].reshape(pn, R_local)
                rec = np.einsum('ar,br,cr->abc', U_l, V_l, W_l)
                return np.sum((rec - T) ** 2)
            
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            res = minimize(objective, x0, method='L-BFGS-B',
                          options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
            
            if res.success or res.fun < 1e-8:
                U_opt = res.x[:um * R].reshape(um, R)
                V_opt = res.x[um * R:um * R + vp * R].reshape(vp, R)
                W_opt = res.x[um * R + vp * R:].reshape(pn, R)
                
                if _verify_decomposition_fast(U_opt, V_opt, W_opt, T, tol=1e-9):
                    # Convert to lists with rounded values
                    U_list = [[float(U_opt[i][r]) for r in range(R)] for i in range(um)]
                    V_list = [[float(V_opt[i][r]) for r in range(R)] for i in range(vp)]
                    W_list = [[float(W_opt[i][r]) for r in range(R)] for i in range(pn)]
                    return U_list, V_list, W_list
    
    return None

def _try_greedy_support_search(T):
    """Greedy heuristic: build decomposition term by term, selecting
    rank-1 tensors that best explain the remaining tensor."""
    n, m, p = DIMENSIONS
    um, vp, pn = n * m, m * p, p * n
    residual = T.copy()
    U_cols = []
    V_cols = []
    W_cols = []
    
    max_terms = 40  # Cap on number of terms to try
    for step in range(max_terms):
        if _time_left() < 5.0:
            break
        if np.max(np.abs(residual)) < 1e-12:
            break
        
        # Find best rank-1 tensor approximating residual
        best_val = np.inf
        best_U = best_V = best_W = None
        
        # Sample candidate rank-1 tensors
        rng = np.random.RandomState(step * 1000 + 42)
        for _ in range(2000):
            u = rng.randn(um)
            v = rng.randn(vp)
            w = rng.randn(pn)
            # Normalize
            nu = np.linalg.norm(u)
            nv = np.linalg.norm(v)
            nw = np.linalg.norm(w)
            if nu < 1e-12 or nv < 1e-12 or nw < 1e-12:
                continue
            u = u / nu
            v = v / nv
            w = w / nw
            term = np.einsum('a,b,c->abc', u, v, w)
            err = np.sum((residual - term) ** 2)
            if err < best_val:
                best_val = err
                best_U = u
                best_V = v
                best_W = w
        
        if best_U is None:
            break
        
        # Try to find exact coefficients for this direction
        # We want alpha * (u,v,w) such that residual - alpha*term is minimized
        term = np.einsum('a,b,c->abc', best_U, best_V, best_W)
        # Optimal alpha = <residual, term> / <term, term>
        dot_rt = np.sum(residual * term)
        dot_tt = np.sum(term * term)
        if abs(dot_tt) < 1e-15:
            break
        alpha = dot_rt / dot_tt
        residual -= alpha * term
        
        U_cols.append(best_U * np.sqrt(abs(alpha)) if alpha >= 0 else best_U * np.sqrt(abs(alpha)))
        V_cols.append(best_V * np.sqrt(abs(alpha)) if alpha >= 0 else best_V * np.sqrt(abs(alpha)))
        if alpha < 0:
            W_cols.append(best_W * np.sqrt(abs(alpha)))
        else:
            W_cols.append(best_W * np.sqrt(abs(alpha)))
        
        if step > 0 and step % 5 == 0:
            pass  # progress check
    
    if np.max(np.abs(residual)) < 1e-9 and len(U_cols) > 0:
        U_arr = np.column_stack(U_cols)
        V_arr = np.column_stack(V_cols)
        W_arr = np.column_stack(W_cols)
        if _verify_decomposition_fast(U_arr, V_arr, W_arr, T, tol=1e-9):
            R = len(U_cols)
            U_list = [[float(U_arr[i][r]) for r in range(R)] for i in range(um)]
            V_list = [[float(V_arr[i][r]) for r in range(R)] for i in range(vp)]
            W_list = [[float(W_arr[i][r]) for r in range(R)] for i in range(pn)]
            return U_list, V_list, W_list
    
    return None

def _schoolbook():
    """Return the schoolbook (trivial) decomposition of rank n*m*p = 72."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1.0
                V[j * p + k][column] = 1.0
                W[k * n + i][column] = 1.0
                column += 1
    return {"U": U, "V": V, "W": W}

def construct():
    start = time.time()
    T = _build_tensor()
    
    # Strategy 1: Try greedy support search
    if _time_left() > 10.0:
        result = _try_greedy_support_search(T)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Strategy 2: Try random optimization search
    if _time_left() > 10.0:
        result = _try_random_search(T, max_rank=40, num_trials=3, seed=123)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook decomposition
    return _schoolbook()
