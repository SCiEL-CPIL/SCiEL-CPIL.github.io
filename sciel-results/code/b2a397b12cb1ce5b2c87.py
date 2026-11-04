"""Optimized program: refined multi-stage global search for uncertainty bound minimization."""
import numpy as np
import time
import scipy.optimize
import math
from scipy.integrate import quad

def _compute_bound(c0, c4, c8):
    """
    Compute the uncertainty product bound for the given coefficients.
    c12 is determined by c0, c4, c8 to satisfy g(0)=0.
    Returns the product sqrt(<x^2> * <xi^2>).
    """
    # Hermite values at 0: H0(0)=1, H4(0)=12, H8(0)=168, H12(0)=1330
    c12 = -(c0 + 12*c4 + 168*c8) / 1330.0
    
    # Coefficients of the polynomial P(x) = c0 H0 + c4 H4 + c8 H8 + c12 H12
    # P(x) = a12 x^12 + a10 x^10 + a8 x^8 + a6 x^6 + a4 x^4 + a2 x^2 + a0
    # a0 = c0 + 12c4 + 168c8 + 1330c12 = 0
    a0 = 0.0
    a2 = -48*c4 - 1344*c8 - 32256*c12
    a4 = 16*c4 + 2688*c8 + 133056*c12
    a6 = -1344*c8 - 180224*c12
    a8 = 256*c8 + 118272*c12
    a10 = -36864*c12
    a12 = 4096*c12
    
    p = np.array([a0, a2, a4, a6, a8, a10, a12])
    
    # Convolution to get coefficients of P(x)^2
    q = np.convolve(p, p)
    
    # Integrate x^{2k} e^{-x^2}
    factors = np.zeros(13)
    for k in range(13):
        factors[k] = math.sqrt(math.pi) * (2**k) * math.factorial(k)
        
    I = q * factors
    
    norm_sq = I[0]
    if norm_sq <= 1e-15:
        return 1e10
    
    x2 = I[1] / I[0]
    
    # Calculate g'(x) to get <xi^2>
    r = np.zeros(7)
    for j in range(7):
        val = 0.0
        if j <= 5:
            val += 2*(j+1)*p[j+1]
        val += -p[j]
        r[j] = val
    
    integral_R_sq = 0.0
    for j in range(7):
        for k in range(7):
            m = j + k
            integral_R_sq += r[j] * r[k] * math.sqrt(math.pi) * (2**(m+1)) * math.factorial(m+1)
    
    xi2 = integral_R_sq / (4 * math.pi**2 * norm_sq)
    
    product = math.sqrt(x2 * xi2)
    
    return product

def construct():
    start_time = time.time()
    deadline = start_time + 170.0  # Keep margin for evaluator kill at 210s
    
    # Initial guess: Seed values
    seed_coeffs = np.array([-113 / 100, 1 / 25, 1 / 3240])
    
    best_val = _compute_bound(seed_coeffs[0], seed_coeffs[1], seed_coeffs[2])
    best_coeffs = seed_coeffs.copy()
    
    def _objective(x):
        return _compute_bound(x[0], x[1], x[2])
    
    # 1. Local Optimization from Seed with tighter tolerance
    try:
        res_local = scipy.optimize.minimize(
            _objective,
            seed_coeffs,
            method='L-BFGS-B',
            options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12}
        )
        if res_local.success and res_local.fun < best_val:
            best_val = res_local.fun
            best_coeffs = res_local.x.copy()
    except Exception:
        pass
    
    # 2. Global Search using Differential Evolution with more iterations
    bounds = [(-5, 5), (-5, 5), (-5, 5)]
    
    if time.time() < deadline - 40.0:
        try:
            de_res = scipy.optimize.differential_evolution(
                _objective,
                bounds,
                seed=42,
                maxiter=200,
                tol=1e-10,
                popsize=20,
                polish=True,
                polish_options={'maxiter': 500, 'ftol': 1e-12}
            )
            
            if de_res.fun < best_val:
                best_val = de_res.fun
                best_coeffs = de_res.x.copy()
        except Exception:
            pass
    
    # 3. Multi-start Local Optimization with diverse starting points
    if time.time() < deadline - 15.0:
        np.random.seed(123)
        # Try multiple diverse starting points
        start_points = []
        # Near best
        for _ in range(5):
            noise = np.random.normal(0, 0.3, 3)
            start_points.append(best_coeffs + noise)
        # Far from best
        for _ in range(5):
            start_points.append(np.random.uniform(-2, 2, 3))
        # Specific structured points
        start_points.append(np.array([0.0, 0.0, 0.0]))
        start_points.append(np.array([1.0, -1.0, 0.1]))
        start_points.append(np.array([-1.0, 1.0, -0.1]))
        
        for sp in start_points:
            if time.time() > deadline - 5.0:
                break
            try:
                res_restart = scipy.optimize.minimize(
                    _objective,
                    sp,
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-13, 'gtol': 1e-11}
                )
                if res_restart.success and res_restart.fun < best_val:
                    best_val = res_restart.fun
                    best_coeffs = res_restart.x.copy()
            except Exception:
                pass
    
    # 4. Final High-Precision Polish with Nelder-Mead for non-gradient refinement
    if time.time() < deadline - 3.0:
        try:
            # Nelder-Mead can find slightly better solutions in the final neighborhood
            res_nm = scipy.optimize.minimize(
                _objective,
                best_coeffs,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
            )
            if res_nm.success and res_nm.fun < best_val:
                best_val = res_nm.fun
                best_coeffs = res_nm.x.copy()
        except Exception:
            pass
        
        # Final L-BFGS-B polish
        if time.time() < deadline - 1.0:
            try:
                res_final = scipy.optimize.minimize(
                    _objective,
                    best_coeffs,
                    method='L-BFGS-B',
                    options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-14}
                )
                if res_final.success and res_final.fun < best_val:
                    best_val = res_final.fun
                    best_coeffs = res_final.x.copy()
            except Exception:
                pass

    return best_coeffs
