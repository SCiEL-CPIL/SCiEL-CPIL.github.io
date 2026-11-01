import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 10]

def _verify_exact(U, V, W, n, m, p):
    """Verify exact decomposition using Fraction arithmetic."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                if s != 1:
                    return False
    return True

def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Fast verification using NumPy."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                prod = U[i*m+j] * V[j*p+k] * W[k*n+i]
                s = np.sum(prod)
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _try_rank_random(n, m, p, r, time_budget, seed=42):
    """Try to find a decomposition of rank r using random ±1 and sparse patterns."""
    rng = np.random.RandomState(seed)
    deadline = time.time() + time_budget
    best = None
    
    # Try different sparsity levels
    sparsity_levels = [0.0, 0.3, 0.5, 0.7]
    
    for sparsity in sparsity_levels:
        if time.time() > deadline - 2:
            break
            
        max_attempts = 500 if sparsity == 0.0 else 200
        for attempt in range(max_attempts):
            if time.time() > deadline - 2:
                    break
                    
            U = rng.randint(0, 2, size=(n*m, r)) * 2 - 1
            V = rng.randint(0, 2, size=(m*p, r)) * 2 - 1
            W = rng.randint(0, 2, size=(p*n, r)) * 2 - 1
            
            if sparsity > 0:
                U[rng.random((n*m, r)) < sparsity] = 0
                V[rng.random((m*p, r)) < sparsity] = 0
                W[rng.random((p*n, r)) < sparsity] = 0
            
            # Quick check
            ok = True
            for i in range(n):
                if not ok:
                    break
                for j in range(m):
                    if not ok:
                        break
                    for k in range(p):
                        prod = U[i*m+j] * V[j*p+k] * W[k*n+i]
                        s = np.sum(prod)
                        if abs(s - 1.0) > 1e-10:
                            ok = False
                            break
            if ok:
                best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                return best
                
    return best

def _try_rank_optimization(n, m, p, r, time_budget, seed=42):
    """
    Try to find a decomposition of rank r using continuous optimization
    followed by rounding to integers/half-integers.
    """
    from scipy.optimize import minimize
    
    rng = np.random.RandomState(seed)
    deadline = time.time() + time_budget
    best = None
    
    # Try a few random initializations
    for attempt in range(10):
        if time.time() > deadline - 5:
            break
            
        # Initialize with random values scaled to be small
        scale = 0.1
        U = rng.randn(n * m, r) * scale
        V = rng.randn(m * p, r) * scale
        W = rng.randn(p * n, r) * scale
        
        # Flatten for optimization
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def objective(x):
            U_f = x[:n*m*r].reshape(n*m, r)
            V_f = x[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
            W_f = x[n*m*r + m*p*r :].reshape(p*n, r)
            
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.sum(U_f[i*m+j] * V_f[j*p+k] * W_f[k*n+i])
                        err += (s - 1.0) ** 2
            return err
        
        def jac(x):
            U_f = x[:n*m*r].reshape(n*m, r)
            V_f = x[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
            W_f = x[n*m*r + m*p*r :].reshape(p*n, r)
            
            gU = np.zeros_like(U_f)
            gV = np.zeros_like(V_f)
            gW = np.zeros_like(W_f)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.sum(U_f[i*m+j] * V_f[j*p+k] * W_f[k*n+i])
                        e = 2 * (s - 1.0)
                        gU[i*m+j] += e * V_f[j*p+k] * W_f[k*n+i]
                        gV[j*p+k] += e * U_f[i*m+j] * W_f[k*n+i]
                        gW[k*n+i] += e * U_f[i*m+j] * V_f[j*p+k]
            
            return np.concatenate([gU.flatten(), gV.flatten(), gW.flatten()])
        
        try:
            res = minimize(objective, x0, jac=jac, method='L-BFGS-B', 
                          options={'maxiter': 100, 'gtol': 1e-6, 'ftol': 1e-12})
            
            if res.fun < 1e-6:
                # Extract solution
                U_opt = res.x[:n*m*r].reshape(n*m, r)
                V_opt = res.x[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
                W_opt = res.x[n*m*r + m*p*r :].reshape(p*n, r)
                
                # Try rounding to integers
                for mult in [1, 2]:
                    U_round = np.round(U_opt * mult) / mult
                    V_round = np.round(V_opt * mult) / mult
                    W_round = np.round(W_opt * mult) / mult
                    
                    if _verify_fast(U_round, V_round, W_round, n, m, p):
                        # Verify exactly
                        if _verify_exact(U_round.tolist(), V_round.tolist(), W_round.tolist(), n, m, p):
                            best = {"U": U_round.tolist(), "V": V_round.tolist(), "W": W_round.tolist()}
                            return best
                            
        except Exception:
            pass
            
    return best

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Start with schoolbook as fallback
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to find lower ranks
    # Lower bound considerations:
    # For 2x3 * 3x10, the tensor rank is at least max(n*m, m*p, p*n) = 30
    # But we can try below 60
    candidate_ranks = list(range(59, 29, -1))  # Try from 59 down to 30
    
    for r in candidate_ranks:
        if time.time() > deadline - 15:
            break
            
        # First try fast random search
        result = _try_rank_random(n, m, p, r, deadline - time.time() - 10, seed=1000 + r)
        if result is not None:
            # Verify exactly
            if _verify_exact(result["U"], result["V"], result["W"], n, m, p):
                best = result
                best_rank = r
                # Continue searching for lower ranks
                continue
        
        # Try optimization-based approach
        time_left = deadline - time.time() - 5
        if time_left > 10:
            result = _try_rank_optimization(n, m, p, r, time_left, seed=2000 + r)
            if result is not None:
                best = result
                best_rank = r
                continue
                
    return best
