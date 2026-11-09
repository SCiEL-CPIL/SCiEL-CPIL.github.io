"""Refined Cohn-Goncalves Laguerre construction: optimize double-root locations."""
import time
import math
import numpy as np
from scipy.special import eval_laguerre, gammaln, roots_genlaguerre
from scipy.optimize import minimize

# EVOLVE-BLOCK-START

def _lag_coeffs(m, n):
    """Coefficients of the Laguerre polynomial L_m^{(n)}(x) (degree m, alpha=n).
    Returned as list c[0..m] for sum c[k] x^k."""
    alpha = float(n)
    L0 = [1.0]
    if m == 0:
        return L0
    L1 = [1.0 + alpha, -1.0]
    if m == 1:
        return L1
    Lm1 = L0
    Lm = L1
    for k in range(1, m):
        # L_{k+1} = ((2k+alpha+1 - x) L_k - (k+alpha) L_{k-1}) / (k+1)
        A = [0.0] * (len(Lm) + 1)
        for i in range(len(Lm)):
            A[i] += (2.0 * k + alpha + 1.0) * Lm[i]
            A[i + 1] += -1.0 * Lm[i]
        for i in range(len(Lm1)):
            A[i] -= (k + alpha) * Lm1[i]
        Lnew = [v / (k + 1.0) for v in A]
        Lm1, Lm = Lm, Lnew
    return Lm

def _poly_eval_vec(coeffs, x):
    """Vectorized polynomial evaluation. coeffs: list, x: numpy array."""
    # Use Horner's rule vectorized
    r = np.zeros_like(x)
    for c in reversed(coeffs):
        r = r * x + c
    return r

def _objective_log(theta):
    """
    Compute the L2 norm of P(x) = L_m^{(n)}(x) * prod (x-r_i)^2 
    with weight e^{-x} x^n.
    theta: log of roots.
    Returns the norm (lower is better).
    """
    m, n = 12, 1
    # Recover roots
    r = np.exp(theta)
    
    # Pre-compute Laguerre coefficients (constant)
    # We can cache this, but it's fast enough to compute or pass in closure.
    # For efficiency in the loop, we should pre-compute Lc outside.
    # However, to keep the function self-contained for the optimizer, we'll use a global or closure.
    # Let's use a global cache.
    if not hasattr(_objective_log, 'Lc'):
        _objective_log.Lc = _lag_coeffs(m, n)
    Lc = _objective_log.Lc
    
    # Build P(x) coefficients? No, that's slow.
    # Instead, evaluate P(x) directly at quadrature nodes.
    # P(x) = L(x) * prod (x - r_i)^2
    
    # Get Gauss-Laguerre nodes and weights for alpha=n
    # We need to integrate P(x)^2 * x^n * e^{-x}.
    # Degree of P is m + 2*12 = 36. P^2 is degree 72.
    # Standard Gauss-Laguerre integrates polynomials of degree 2N-1 exactly.
    # So we need 2N-1 >= 72 => N >= 36.5 => N=37.
    # Let's use N=50 for safety and precision.
    if not hasattr(_objective_log, 'nodes'):
        nodes, weights = roots_genlaguerre(50, n)
        _objective_log.nodes = nodes
        _objective_log.weights = weights
    nodes = _objective_log.nodes
    weights = _objective_log.weights
    
    # Evaluate L(x) at nodes
    L_vals = np.array([eval_laguerre(m, n, x) for x in nodes])
    
    # Evaluate prod (x - r_i)^2 at nodes
    # To avoid overflow/underflow and for speed, we can compute log or just multiply.
    # Roots are positive. Nodes are positive.
    # prod (x - r_i)^2 = (prod (x - r_i))^2
    prod_term = np.ones_like(nodes)
    for ri in r:
        prod_term *= (nodes - ri) ** 2
        
    P_vals = L_vals * prod_term
    
    # Integral = sum(weights * P_vals^2)
    # Note: roots_genlaguerre weights are for integral of f(x) x^alpha e^-x dx
    integral = np.sum(weights * P_vals ** 2)
    
    if integral < 0:
        integral = 0.0
        
    return math.sqrt(integral)

def construct():
    """Return 12 positive double-root locations (ascending) minimizing the C4 upper bound."""
    start = time.time()
    deadline = start + 580.0  # 20s safety margin
    
    m, n = 12, 1
    
    # Initial guess: Zeros of L_12^(1)(x)
    # These are known to be good starting points for Laguerre-related extremal problems.
    zeros, _ = roots_genlaguerre(m, n)
    x0 = np.log(zeros)
    
    # Also try the seed uniform spacing in log space?
    # x0_seed = np.log(4.0 * np.arange(1, 13))
    
    # We will run L-BFGS-B from the Laguerre zeros start.
    # If time permits, we can try other starts.
    
    best_x = x0.copy()
    best_val = _objective_log(x0)
    
    # Optimization
    try:
        res = minimize(
            _objective_log, 
            x0, 
            method='L-BFGS-B', 
            options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 5000}
        )
        if res.fun < best_val:
            best_val = res.fun
            best_x = res.x
    except Exception:
        pass
    
    if time.time() > deadline:
        r = np.sort(np.exp(best_x))
        return [float(v) for v in r]
    
    # Multi-start refinement
    # Try a few other initializations if time permits
    rng = np.random.default_rng(42)
    
    # 1. Perturbed Laguerre zeros
    for _ in range(5):
        if time.time() > deadline - 30:
            break
        x_try = x0 + rng.normal(0, 0.2, size=12)
        try:
            v_try = _objective_log(x_try)
            if v_try < best_val:
                best_val = v_try
                best_x = x_try
            # Local polish
            res2 = minimize(
                _objective_log, 
                best_x, 
                method='L-BFGS-B', 
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res2.fun < best_val:
                best_val = res2.fun
                best_x = res2.x
        except Exception:
            continue
            
    # 2. Uniform log spacing (Seed-like)
    x_seed = np.log(4.0 * np.arange(1, 13))
    try:
        v_seed = _objective_log(x_seed)
        if v_seed < best_val:
            best_val = v_seed
            best_x = x_seed
        res3 = minimize(
            _objective_log, 
            best_x, 
            method='L-BFGS-B', 
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        if res3.fun < best_val:
            best_val = res3.fun
            best_x = res3.x
    except Exception:
        pass
        
    # 3. Random restarts in log space
    for _ in range(10):
        if time.time() > deadline - 10:
            break
        # Random log-spaced roots
        r_rand = np.sort(rng.uniform(1, 50, size=12))
        x_rand = np.log(r_rand)
        try:
            v_rand = _objective_log(x_rand)
            if v_rand < best_val:
                best_val = v_rand
                best_x = x_rand
            # Polish
            res4 = minimize(
                _objective_log, 
                best_x, 
                method='L-BFGS-B', 
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-6}
            )
            if res4.fun < best_val:
                best_val = res4.fun
                best_x = res4.x
        except Exception:
            continue

    r = np.sort(np.exp(best_x))
    # Ensure strictly positive and real
    r = np.maximum(r, 1e-9)
    return [float(v) for v in r]

# EVOLVE-BLOCK-END
