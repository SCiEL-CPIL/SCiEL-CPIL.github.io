import numpy as np
import time
from scipy.optimize import minimize
import sys

N = 50
TIME_LIMIT = 145.0  # safety margin before 150s kill

def _autoconv(h):
    """Compute autoconvolution of h with itself."""
    return np.convolve(h, h, mode='full')

def _score(h):
    """Compute the ratio: ||h||_2^2 / (||h||_1 * ||autoconv(h)||_inf)"""
    h = np.asarray(h, dtype=float)
    if np.any(h < -1e-12):
        return 0.0
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 < 1e-12:
        return 0.0
    l2_sq = np.sum(h**2)
    ac = _autoconv(h)
    ac_max = np.max(ac)
    if ac_max < 1e-12:
        return 0.0
    return l2_sq / (l1 * ac_max)

def construct():
    start_time = time.time()
    
    # Strategy: Maximize ||h||_2^2 subject to:
    # 1. sum(h) = 1
    # 2. h >= 0
    # 3. ||h * h||_inf <= 1
    
    # This maximizes the ratio directly because Ratio = ||h||_2^2 / (||h||_1 * ||h*h||_inf)
    # With ||h||_1=1 and ||h*h||_inf <= 1, Ratio >= ||h||_2^2.
    # Actually, if ||h*h||_inf < 1, we can scale h up to make ||h*h||_inf = 1,
    # which increases ||h||_2^2 / ||h*h||_inf?
    # Let h' = c h. Ratio(h') = (c^2 ||h||_2^2) / (c ||h||_1 * c^2 ||h*h||_inf) = Ratio(h).
    # The ratio is scale invariant.
    # So we just fix ||h||_1 = 1.
    # We want to maximize ||h||_2^2 / ||h*h||_inf.
    # This is equivalent to minimizing ||h*h||_inf subject to ||h||_2^2 >= K?
    # Or maximizing ||h||_2^2 subject to ||h*h||_inf <= 1?
    # Yes, the latter is convex-ish? No, ||h||_2^2 is convex, maximizing convex is hard.
    # But ||h*h||_inf <= 1 is a set of linear/quadratic constraints?
    # (h*h)_k = sum h_i h_{i-k}. This is quadratic.
    # So we are maximizing a convex function over a non-convex set (intersection of quadratic inequalities).
    # SLSQP can still handle this locally.
    
    def _objective(h):
        # Maximize sum(h^2) -> Minimize -sum(h^2)
        return -np.sum(h**2)
    
    def _objective_grad(h):
        return -2.0 * h
    
    # Constraints
    # Equality: sum(h) = 1
    cons_eq = ({
        'type': 'eq',
        'fun': lambda h: np.sum(h) - 1,
        'jac': lambda h: np.ones(N)
    })
    
    # Inequality: (h*h)_k <= 1 for all k
    # g_k(h) = 1 - (h*h)_k >= 0
    
    num_lags = 2 * N - 1
    
    def _ineq_fun(h):
        ac = _autoconv(h)
        return 1.0 - ac
        
    def _ineq_jac(h):
        # Jacobian of g_k with respect to h_j
        # g_k = 1 - sum_i h_i h_{i-k}
        # dg_k/dh_j = - (h_{j-k} + h_{j+k})
        # Note: indices must be within [0, N-1]
        jac = np.zeros((num_lags, N))
        # We can compute this efficiently
        # For each lag k (0 to 2N-2), the term in autoconv is sum_i h[i] h[i-(k-N+1)]?
        # np.convolve(h, h) index k corresponds to lag k - (N-1)
        # Let lag = k - (N-1).
        # c_lag = sum_i h[i] h[i-lag]
        # dc_lag/dh[j] = h[j-lag] + h[j+lag]
        
        for k in range(num_lags):
            lag = k - (N - 1)
            for j in range(N):
                val = 0.0
                if 0 <= j - lag < N:
                    val += h[j - lag]
                if 0 <= j + lag < N:
                    val += h[j + lag]
                jac[k, j] = -val # derivative of (1 - c_lag)
        return jac

    cons_ineq = ({
        'type': 'ineq',
        'fun': _ineq_fun,
        'jac': _ineq_jac
    })
    
    bounds = [(0, None)] * N
    
    best_h = np.ones(N) / N
    best_score = _score(best_h)
    
    # Initial guesses
    # 1. Uniform
    # 2. Peaked in center
    # 3. Random Dirichlet
    
    initial_guesses = [np.ones(N) / N]
    
    # Create some peaked guesses
    for width in [2, 4, 8, 16]:
        h_guess = np.zeros(N)
        center = N // 2
        start = max(0, center - width // 2)
        end = min(N, start + width)
        h_guess[start:end] = 1.0 / (end - start)
        initial_guesses.append(h_guess)
        
    # Random guesses
    np.random.seed(42)
    for _ in range(10):
        h_guess = np.random.dirichlet(np.ones(N) * 0.5)
        initial_guesses.append(h_guess)
        
    for h0 in initial_guesses:
        if time.time() - start_time > TIME_LIMIT - 10:
            break
            
        # Ensure h0 is on the simplex
        h0 = np.maximum(h0, 0)
        s = np.sum(h0)
        if s > 0:
            h0 = h0 / s
        else:
            continue
            
        try:
            # SLSQP with analytic gradients
            res = minimize(_objective, h0, method='SLSQP', 
                           jac=_objective_grad,
                           bounds=bounds,
                           constraints=[cons_eq, cons_ineq],
                           options={'maxiter': 200, 'ftol': 1e-12, 'disp': False})
            
            if res.success:
                h_opt = res.x
                # Clean up small negatives
                h_opt = np.maximum(h_opt, 0)
                s = np.sum(h_opt)
                if s > 1e-12:
                    h_opt = h_opt / s
                    s_opt = _score(h_opt)
                    if s_opt > best_score:
                        best_score = s_opt
                        best_h = h_opt.copy()
        except Exception:
            pass

    # If we have time, try to polish the best solution with more iterations
    if time.time() - start_time < TIME_LIMIT - 20:
        try:
            res = minimize(_objective, best_h, method='SLSQP', 
                           jac=_objective_grad,
                           bounds=bounds,
                           constraints=[cons_eq, cons_ineq],
                           options={'maxiter': 500, 'ftol': 1e-15, 'disp': False})
            if res.success:
                h_opt = res.x
                h_opt = np.maximum(h_opt, 0)
                s = np.sum(h_opt)
                if s > 1e-12:
                    h_opt = h_opt / s
                    s_opt = _score(h_opt)
                    if s_opt > best_score:
                        best_score = s_opt
                        best_h = h_opt.copy()
        except Exception:
            pass

    return list(best_h)

if __name__ == "__main__":
    h = construct()
    print("Score:", _score(h))
    print("Heights:", h)
