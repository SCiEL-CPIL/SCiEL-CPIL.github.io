"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

def _compute_q_coefficients(c0, c1, c2):
    """Compute coefficients of Q(y) = b1 + b2*y + ... + b6*y^5 where y=x^2."""
    c3 = -(c0 + 12.0 * c1 + 168.0 * c2) / 18480.0
    b1 = -48.0 * c1 - 1344.0 * c2 - 921600.0 * c3
    b2 = 16.0 * c1 + 2688.0 * c2 + 1424640.0 * c3
    b3 = -1344.0 * c2 - 1003520.0 * c3
    b4 = 256.0 * c2 + 332160.0 * c3
    b5 = -46080.0 * c3
    b6 = 4096.0 * c3
    return np.array([b1, b2, b3, b4, b5, b6])

def _evaluate_value(c0, c1, c2):
    """Compute value = r^2 / (2*pi) where r is largest sign-changing root of q(x)=g(x)/x^2."""
    coeffs = _compute_q_coefficients(c0, c1, c2)
    # Q(y) = b1 + b2*y + b3*y^2 + b4*y^3 + b5*y^4 + b6*y^5
    # Find roots of Q(y)
    # Use highest degree first for np.roots
    roots = np.roots(coeffs[::-1])
    # Filter real positive roots with small imaginary part
    real_roots = roots[np.abs(roots.imag) < 1e-8 * np.maximum(1.0, np.abs(roots))].real
    real_roots = real_roots[real_roots > 1e-12]
    if len(real_roots) == 0:
        return np.inf
    # Evaluate Q at each root to check sign change
    # Build polynomial from coefficients (lowest degree first)
    # Q(y) = sum(coeffs[i] * y^i) for i=0..5
    def Q(y):
        return coeffs[0] + coeffs[1]*y + coeffs[2]*y**2 + coeffs[3]*y**3 + coeffs[4]*y**4 + coeffs[5]*y**5
    
    valid_roots = []
    for r in real_roots:
        eps = 1e-10 * max(1.0, abs(r))
        q_left = Q(r - eps)
        q_right = Q(r + eps)
        if q_left * q_right < 0:
            valid_roots.append(r)
    if len(valid_roots) == 0:
        return np.inf
    y_max = max(valid_roots)
    return y_max / (2.0 * np.pi)

def construct():
    start_time = time.time()
    deadline = start_time + 200.0  # Safety margin: 210s limit, keep 10s buffer
    
    seed_c = np.array([-113.0/100, 1.0/25, 1.0/3240])
    best_c = seed_c.copy()
    best_val = _evaluate_value(seed_c[0], seed_c[1], seed_c[2])
    
    def objective(x):
        elapsed = time.time() - start_time
        if elapsed > deadline - start_time:
            return best_val
        try:
            v = _evaluate_value(x[0], x[1], x[2])
            return v
        except:
            return 1e10
    
    # Phase 1: Global search with differential evolution
    # Bound the search space reasonably
    bounds = [(-10.0, 10.0), (-10.0, 10.0), (-10.0, 10.0)]
    
    try:
        result = differential_evolution(
            objective, bounds, 
            maxiter=200, 
            popsize=15,
            tol=1e-10,
            seed=42,
            polish=True,
            init='sobol'
        )
        if result.fun < best_val:
            best_c = result.x
            best_val = result.fun
    except Exception as e:
        pass
    
    # Phase 2: Local refinement from best point found
    if time.time() < deadline:
        try:
            def obj2(x):
                elapsed = time.time() - start_time
                if elapsed > deadline - start_time:
                    return best_val
                try:
                    return _evaluate_value(x[0], x[1], x[2])
                except:
                    return 1e10
            
            res2 = minimize(obj2, best_c, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-13, 'fatol': 1e-13, 'adaptive': True})
            if res2.fun < best_val:
                best_c = res2.x
                best_val = res2.fun
        except:
            pass
    
    # Phase 3: Multi-start local optimization from random points
    if time.time() < deadline - 10:
        rng = np.random.default_rng(12345)
        for i in range(50):
            if time.time() > deadline - 5:
                break
            x0 = rng.uniform(-5, 5, size=3)
            # Also try near seed
            if i % 2 == 0:
                x0 = seed_c + rng.normal(0, 0.1, size=3)
            try:
                res = minimize(objective, x0, method='Nelder-Mead',
                             options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True})
                if res.fun < best_val:
                    best_c = res.x
                    best_val = res.fun
            except:
                pass
    
    return best_c
# EVOLVE-BLOCK-END
