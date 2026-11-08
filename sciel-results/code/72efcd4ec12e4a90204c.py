import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

def _compute_bound(c):
    """
    c: array [c0, c4, c8, c12]
    Returns the uncertainty product Δx * Δp for f(x) = sum c_n H_n(x) exp(-x^2/2)
    Uses analytical Hermite-Gaussian orthogonality for O(1) computation.
    """
    # Coefficients in the orthonormal Hermite function basis:
    # psi_n(x) = (2^n n! sqrt(pi))^{-1/2} H_n(x) e^{-x^2/2}
    # f(x) = sum_n c_n H_n(x) e^{-x^2/2} = sum_n a_n psi_n(x)
    # where a_n = c_n * sqrt(2^n n! sqrt(pi))
    degrees = [0, 4, 8, 12]
    a = np.zeros(13)  # coefficients in orthonormal basis, a_n for n=0..12
    for i, deg in enumerate(degrees):
        a[deg] = c[i] * np.sqrt(2.0**deg * float(np.math.factorial(deg)) * np.sqrt(np.pi))
    
    norm = np.sum(a**2)
    if norm < 1e-30:
        return np.inf
    
    # <x>: <psi_n|x|psi_m> = (1/sqrt(2))(sqrt(m+1) delta_{n,m+1} + sqrt(m) delta_{n,m-1})
    x_mean = 0.0
    for n in range(13):
        for m in range(13):
            if n == m + 1:
                x_mean += a[n] * a[m] * (1.0/np.sqrt(2.0)) * np.sqrt(m+1)
            elif n == m - 1:
                x_mean += a[n] * a[m] * (1.0/np.sqrt(2.0)) * np.sqrt(m)
    x_mean /= norm
    
    # <x^2>: <psi_n|x^2|psi_m> = (1/2)sqrt((m+1)(m+2)) delta_{n,m+2} + (m+1/2) delta_{n,m} + (1/2)sqrt(m(m-1)) delta_{n,m-2}
    x2_mean = 0.0
    for n in range(13):
        for m in range(13):
            if n == m + 2:
                x2_mean += a[n] * a[m] * 0.5 * np.sqrt((m+1)*(m+2))
            elif n == m:
                x2_mean += a[n] * a[m] * (m + 0.5)
            elif n == m - 2:
                x2_mean += a[n] * a[m] * 0.5 * np.sqrt(m*(m-1))
    x2_mean /= norm
    
    sigma_x_sq = max(0.0, x2_mean - x_mean**2)
    
    # <p>: <psi_n|p|psi_m> = i/sqrt(2)(sqrt(m) delta_{n,m-1} - sqrt(m+1) delta_{n,m+1})
    p_mean = 0.0
    for n in range(13):
        for m in range(13):
            if n == m - 1:
                p_mean += a[n] * a[m] * (1.0/np.sqrt(2.0)) * np.sqrt(m)
            elif n == m + 1:
                p_mean -= a[n] * a[m] * (1.0/np.sqrt(2.0)) * np.sqrt(m+1)
    p_mean /= norm
    
    # <p^2>: Using p^2 = H - x^2/2 where H is the harmonic oscillator Hamiltonian
    # <psi_n|H|psi_m> = (n+1/2) delta_{nm}
    # So <p^2> = <H> - <x^2>/2
    # But we need <H> = sum (n+1/2) |a_n|^2 / norm
    H_mean = 0.0
    for n in range(13):
        H_mean += a[n]**2 * (n + 0.5)
    H_mean /= norm
    
    p2_mean = H_mean - x2_mean / 2.0
    sigma_p_sq = max(0.0, p2_mean - p_mean**2)
    
    return np.sqrt(sigma_x_sq) * np.sqrt(sigma_p_sq)

def construct():
    start_time = time.time()
    deadline = 195.0  # Safety margin before 210s kill
    
    # Constraint: f(0) = 0
    # H0(0)=1, H4(0)=12, H8(0)=1680, H12(0)=3991680
    # c0 + 12*c4 + 1680*c8 + 3991680*c12 = 0
    # c12 = -(c0 + 12*c4 + 1680*c8) / 3991680
    
    def objective(params):
        c4, c8 = params
        c0 = 1.0
        c12 = -(c0 + 12.0*c4 + 1680.0*c8) / 3991680.0
        c = np.array([c0, c4, c8, c12])
        return _compute_bound(c)
    
    best_score = np.inf
    best_params = np.array([0.0, 0.0])
    
    # Phase 1: Broad global search with differential evolution
    if time.time() - start_time < deadline - 60:
        try:
            bounds = [(-5.0, 5.0), (-1.0, 1.0)]
            de_result = differential_evolution(
                objective, bounds, seed=42, maxiter=200, 
                popsize=25, tol=1e-10, polish=True,
                init='sobol'
            )
            if de_result.fun < best_score:
                best_score = de_result.fun
                best_params = de_result.x.copy()
        except Exception:
            pass
    
    # Phase 2: Local refinement from many starting points
    # Generate diverse starting points around the best found
    rng = np.random.default_rng(123)
    start_points = [best_params.copy()]
    
    # Add points at various scales
    for scale in [0.1, 0.5, 1.0, 2.0, 5.0]:
        for _ in range(10):
            p = best_params + scale * rng.standard_normal(2)
            start_points.append(p)
    
    # Also try some structured points
    for c4 in [-2.0, -1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0, 2.0]:
        for c8 in [-0.5, -0.1, 0.0, 0.1, 0.5]:
            start_points.append(np.array([c4, c8]))
    
    for x0 in start_points:
        if time.time() - start_time > deadline - 20:
            break
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                          options={'maxiter': 2000, 'xatol': 1e-14, 'fatol': 1e-16})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 3: Final high-precision polish
    if time.time() - start_time < deadline - 5 and np.isfinite(best_score):
        try:
            res = minimize(objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-16, 'fatol': 1e-18})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 4: One more DE run in a tighter neighborhood if time allows
    if time.time() - start_time < deadline - 10 and np.isfinite(best_score):
        try:
            bounds_tight = [(best_params[0]-0.5, best_params[0]+0.5), 
                           (best_params[1]-0.1, best_params[1]+0.1)]
            de_result2 = differential_evolution(
                objective, bounds_tight, seed=99, maxiter=100, 
                popsize=15, tol=1e-12, polish=True
            )
            if de_result2.fun < best_score:
                best_score = de_result2.fun
                best_params = de_result2.x.copy()
                
                # Polish again
                res = minimize(objective, best_params, method='Nelder-Mead',
                              options={'maxiter': 3000, 'xatol': 1e-16, 'fatol': 1e-18})
                if res.fun < best_score:
                    best_score = res.fun
                    best_params = res.x.copy()
        except Exception:
            pass
    
    return np.array([1.0, best_params[0], best_params[1]])
