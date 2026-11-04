"""Optimized program combining analytic exactness with global search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize
import math

# Precompute integration factors for speed
_FACTORS_13 = np.array([math.sqrt(math.pi) * (2**k) * math.factorial(k) for k in range(13)])
_FACTORS_15 = np.array([math.sqrt(math.pi) * (2**k) * math.factorial(k) for k in range(15)])

def _compute_bound(c0, c4, c8):
    """
    Compute the uncertainty product bound for the given coefficients.
    c12 is determined by c0, c4, c8 to satisfy g(0)=0.
    Returns the product sqrt(<x^2> * <xi^2>).
    """
    # Hermite values at 0: H0(0)=1, H4(0)=12, H8(0)=168, H12(0)=1330
    # c12 = -(c0 + 12*c4 + 168*c8) / 1330.0
    c12 = -(c0 + 12*c4 + 168*c8) / 1330.0
    
    # Coefficients of the polynomial P(x) = c0 H0 + c4 H4 + c8 H8 + c12 H12
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
    
    I = q * _FACTORS_13
    norm_sq = I[0]
    if norm_sq <= 1e-15:
        return 1e10
    
    x2 = I[1] / I[0]
    
    # Construct R(x) = P'' - 2xP' + (x^2 - 1)P
    r = np.zeros(8)
    for j in range(8):
        val = 0.0
        if j <= 5:
            val += 2*(j+1)*(2*j+1)*p[j+1]
        if 1 <= j <= 6:
            val += -4*j*p[j]
        if 1 <= j <= 7:
            val += p[j-1]
        if j <= 6:
            val += -p[j]
        r[j] = val
        
    s = np.convolve(r, r)
    I_R = s * _FACTORS_15
    norm_R_sq = I_R[0]
    
    xi2 = norm_R_sq / (4 * math.pi**2 * norm_sq)
    product = math.sqrt(x2 * xi2)
    
    return product

def construct():
    start_time = time.time()
    deadline = start_time + 175.0
    
    # Initial guess: Seed values
    seed_coeffs = np.array([-113 / 100, 1 / 25, 1 / 3240])
    
    best_val = _compute_bound(seed_coeffs[0], seed_coeffs[1], seed_coeffs[2])
    best_coeffs = seed_coeffs.copy()
    
    def _objective(x):
        try:
            return _compute_bound(x[0], x[1], x[2])
        except Exception:
            return 1e10
    
    # 1. Local Optimization from Seed
    try:
        res_local = scipy.optimize.minimize(
            _objective,
            seed_coeffs,
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        if res_local.success and res_local.fun < best_val:
            best_val = res_local.fun
            best_coeffs = res_local.x.copy()
    except Exception:
        pass
    
    # 2. Global Search using CMA-ES (more robust than DE for this problem)
    bounds = [(-10, 10), (-10, 10), (-10, 10)]
    
    if time.time() < deadline - 30.0:
        try:
            # Use CMA-ES via scipy's differential_evolution as a fallback,
            # but try a better global search first
            # We'll use multiple DE runs with different seeds for diversity
            for seed in [42, 123, 456]:
                if time.time() > deadline - 30.0:
                    break
                de_res = scipy.optimize.differential_evolution(
                    _objective,
                    bounds,
                    seed=seed,
                    maxiter=80,
                    tol=1e-8,
                    popsize=20,
                    polish=False
                )
                if de_res.fun < best_val:
                    best_val = de_res.fun
                    best_coeffs = de_res.x.copy()
        except Exception:
            pass
    
    # 3. Multi-start Local Optimization with diverse starting points
    if time.time() < deadline - 10.0:
        # Generate diverse starting points
        np.random.seed(123)
        start_points = []
        # Near the current best
        for _ in range(5):
            noise = np.random.normal(0, 0.5, 3)
            start_points.append(best_coeffs + noise)
        # Far from the current best (exploration)
        for _ in range(5):
            start_points.append(np.random.uniform(-5, 5, 3))
        
        for start_point in start_points:
            if time.time() > deadline - 5.0:
                break
            try:
                res_restart = scipy.optimize.minimize(
                    _objective,
                    start_point,
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-12}
                )
                if res_restart.success and res_restart.fun < best_val:
                    best_val = res_restart.fun
                    best_coeffs = res_restart.x.copy()
            except Exception:
                pass

    # 4. Final High-Precision Polish
    if time.time() < deadline - 2.0:
        try:
            res_final = scipy.optimize.minimize(
                _objective,
                best_coeffs,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-14}
            )
            if res_final.success and res_final.fun < best_val:
                best_val = res_final.fun
                best_coeffs = res_final.x.copy()
        except Exception:
            pass

    return best_coeffs
# EVOLVE-BLOCK-END
