"""Uncertainty bound minimization via multi-start Nelder-Mead optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 195.0


def _compute_bound(a0, a4, a8):
    """
    Compute the uncertainty product sqrt(<x^2>/<1> * <xi^2>/<1>) for
    g(x) = P(x) * exp(-x^2/2) where P is a degree-12 Hermite combination
    with g(0) = 0 enforced via a12 = -(a0 + 3*a4 + 105*a8)/945.
    """
    a12 = -(a0 + 3.0 * a4 + 105.0 * a8) / 945.0

    # Polynomial coefficients p_k such that P(x) = sum_{k=0}^{6} p_k x^{2k}
    # H0 = 1
    # H4 = x^4 - 6x^2 + 3
    # H8 = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    # H12 = x^12 - 66x^10 + 1155x^8 - 6930x^6 + 13860x^4 - 7560x^2 + 945
    p0 = a0 + 3.0 * a4 + 105.0 * a8 + 945.0 * a12
    p1 = -6.0 * a4 - 420.0 * a8 - 7560.0 * a12
    p2 = a4 + 210.0 * a8 + 13860.0 * a12
    p3 = -28.0 * a8 - 6930.0 * a12
    p4 = a8 + 1155.0 * a12
    p5 = -66.0 * a12
    p6 = a12

    p = np.array([p0, p1, p2, p3, p4, p5, p6])

    # I[n] = integral x^{2n} exp(-x^2) dx = (2n-1)!! * sqrt(pi) / 2^n = (2n)! / (4^n n!) * sqrt(pi)
    # Precompute up to n=12 (need x^{24})
    sp = np.sqrt(np.pi)
    I = np.zeros(13)
    I[0] = sp
    for n in range(1, 13):
        # (2n)! / (4^n * n!) * sqrt(pi)
        # Recurrence: I[n] = I[n-1] * (2n-1) / 2
        I[n] = I[n - 1] * (2 * n - 1) / 2.0

    # m[k] = integral x^{2k} |P(x)|^2 exp(-x^2) dx for k=0,1,2
    # |P|^2 = sum_{i,j} p_i p_j x^{2(i+j)}
    # m[k] = sum_{i,j} p_i p_j I[k+i+j]
    m = np.zeros(3)
    for i in range(7):
        pi = p[i]
        for j in range(7):
            pp = pi * p[j]
            for k in range(3):
                idx = k + i + j
                if idx < 13:
                    m[k] += pp * I[idx]

    # Fourier: G(xi) = sqrt(2*pi) * P(-i*xi) * exp(-xi^2/2)
    # P(-i*xi) = sum p_k (-i)^{2k} xi^{2k} = sum p_k (-1)^k xi^{2k}
    q = p * np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])

    # n[k] = integral xi^{2k} |G(xi)|^2 dxi = 2*pi * sum_{i,j} q_i q_j I[k+i+j]
    n = np.zeros(3)
    for i in range(7):
        qi = q[i]
        for j in range(7):
            qq = qi * q[j]
            for k in range(3):
                idx = k + i + j
                if idx < 13:
                    n[k] += 2.0 * np.pi * qq * I[idx]

    m0, m2 = m[0], m[1]
    n0, n2 = n[0], n[1]

    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18

    vf = m2 / m0
    vF = n2 / n0

    if vf <= 0 or vF <= 0:
        return 1e18

    return np.sqrt(vf * vF)


def construct():
    """
    Find coefficients (a0, a4, a8) that minimize the uncertainty bound.
    Uses multi-start Nelder-Mead optimization.
    """
    # Seed values for reference
    seed = np.array([-1.13, 0.04, 0.000308642])
    
    # Objective function wrapper
    def objective(x):
        return _compute_bound(x[0], x[1], x[2])
        
    best_bound = objective(seed)
    best_params = seed.copy()
    
    rng = np.random.default_rng(42)
    
    # Define a set of starting points
    # 1. The seed
    # 2. Random points in a reasonable range
    # The seed has norm ~1.13. Let's explore up to norm 10.
    start_points = [seed.copy()]
    
    # Generate random starting points
    for _ in range(50):
        # Sample from a distribution that covers the space
        # Mix of small and large scales
        scale = 10 ** rng.uniform(-2, 2) # 0.01 to 100
        x = rng.normal(0, 1, 3) * scale
        start_points.append(x)
        
    # Also add some structured points
    for r in [0.5, 1.0, 2.0, 5.0]:
        for theta1 in np.linspace(-np.pi, np.pi, 10):
            for theta2 in np.linspace(0.1, np.pi-0.1, 10):
                x = np.array([
                    r * np.sin(theta2) * np.cos(theta1),
                    r * np.sin(theta2) * np.sin(theta1),
                    r * np.cos(theta2)
                ])
                start_points.append(x)
                
    # Run Nelder-Mead from each start point
    # We need to be careful with time. 
    # Nelder-Mead is fast for 3D.
    
    for x0 in start_points:
        if time.time() > _DEADLINE - 10:
            break
            
        # Check initial value
        val0 = objective(x0)
        if val0 < best_bound:
            best_bound = val0
            best_params = x0.copy()
            
        try:
            # Nelder-Mead options
            # maxiter: max number of iterations
            # xatol, fatol: tolerances
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
            
            if res.success or res.fun < best_bound:
                if np.isfinite(res.fun) and res.fun < best_bound:
                    best_bound = res.fun
                    best_params = res.x.copy()
        except Exception:
            continue
            
    # Final verification
    final_bound = objective(best_params)
    if not np.isfinite(final_bound) or final_bound > best_bound:
        return seed.copy()
        
    return best_params
# EVOLVE-BLOCK-END
