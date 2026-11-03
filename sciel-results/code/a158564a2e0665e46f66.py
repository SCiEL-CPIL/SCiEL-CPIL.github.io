"""
Optimized program using spherical-coordinate search with high-precision polish
to minimize the uncertainty bound for the Hermite-Gaussian test function.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize
import math

def _compute_bound(c0, c4, c8):
    """
    Compute the uncertainty product bound for the given coefficients.
    c12 is determined by c0, c4, c8 to satisfy g(0)=0.
    """
    c12 = -(c0 + 12*c4 + 168*c8) / 1330.0
    
    # Coefficients of the polynomial P(x) = c0 H0 + c4 H4 + c8 H8 + c12 H12
    # H0 = 1
    # H4 = 16x^4 - 48x^2 + 12
    # H8 = 256x^8 - 1344x^6 + 2688x^4 - 1344x^2 + 168
    # H12 = 4096x^12 - 36864x^10 + 118272x^8 - 180224x^6 + 133056x^4 - 32256x^2 + 1330
    
    # P(x) = a12 x^12 + a10 x^10 + a8 x^8 + a6 x^6 + a4 x^4 + a2 x^2 + a0
    a0 = c0 + 12*c4 + 168*c8 + 1330*c12
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
        return 1e10 # Invalid
        
    x2 = I[1] / I[0]
    
    # Calculate R(x) for Fourier side
    r = np.zeros(8)
    for j in range(8):
        val = 0.0
        if j <= 5:
            val += 2*(j+1)*(2*j+1)*p[j+1]
        if j >= 1 and j <= 6:
            val += -4*j*p[j]
        if j >= 1 and j <= 7:
            val += p[j-1]
        if j <= 6:
            val += -p[j]
        r[j] = val
        
    s = np.convolve(r, r)
    
    factors_R = np.zeros(15)
    for k in range(15):
        factors_R[k] = math.sqrt(math.pi) * (2**k) * math.factorial(k)
        
    I_R = s * factors_R
    norm_R_sq = I_R[0]
    
    xi2 = norm_R_sq / (4 * math.pi**2 * norm_sq)
    
    product = math.sqrt(x2 * xi2)
    
    return product

def construct():
    start_time = time.time()
    deadline = start_time + 175.0  # Safety margin before 180s limit
    
    # Seed values
    seed_coeffs = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best_val = _compute_bound(seed_coeffs[0], seed_coeffs[1], seed_coeffs[2])
    best_coeffs = seed_coeffs.copy()
    
    # Convert to spherical coordinates: scale r, angles theta, phi
    # c0 = r * cos(theta)
    # c4 = r * sin(theta) * cos(phi)
    # c8 = r * sin(theta) * sin(phi)
    
    def from_spherical(r, theta, phi):
        c0 = r * math.cos(theta)
        c4 = r * math.sin(theta) * math.cos(phi)
        c8 = r * math.sin(theta) * math.sin(phi)
        return c0, c4, c8
    
    def spherical_objective(x):
        try:
            r, theta, phi = x[0], x[1], x[2]
            c0, c4, c8 = from_spherical(r, theta, phi)
            return _compute_bound(c0, c4, c8)
        except Exception:
            return 1e10
    
    # Bounds for spherical coordinates
    # r: scale of coefficients (seed has |c| ~ 1.13 max)
    # theta: angle from c0 axis (0 to pi)
    # phi: azimuthal angle (-pi to pi)
    spherical_bounds = [(0.1, 10.0), (0.01, math.pi - 0.01), (-math.pi + 0.01, math.pi - 0.01)]
    
    # Phase 1: Global search in spherical coordinates
    if time.time() < deadline - 30.0:
        try:
            res1 = scipy.optimize.differential_evolution(
                spherical_objective,
                bounds=spherical_bounds,
                seed=42,
                maxiter=150,
                tol=1e-10,
                polish=True,
                init='sobol',
                popsize=15
            )
            if res1.fun < best_val:
                best_val = res1.fun
                r, theta, phi = res1.x
                best_coeffs = np.array(from_spherical(r, theta, phi))
        except Exception:
            pass
    
    # Phase 2: Local refinement with tighter bounds around best solution
    if time.time() < deadline - 15.0:
        try:
            r_best = np.linalg.norm(best_coeffs)
            c0, c4, c8 = best_coeffs
            theta_best = math.acos(np.clip(c0 / r_best, -1.0, 1.0))
            sin_theta = math.sin(theta_best)
            if sin_theta > 1e-10:
                phi_best = math.atan2(c8, c4)
            else:
                phi_best = 0.0
            
            # Tight local bounds
            local_bounds = [
                (max(0.1, r_best * 0.5), r_best * 2.0),
                (max(0.01, theta_best - 0.5), min(math.pi - 0.01, theta_best + 0.5)),
                (max(-math.pi + 0.01, phi_best - 0.5), min(math.pi - 0.01, phi_best + 0.5))
            ]
            
            res2 = scipy.optimize.differential_evolution(
                spherical_objective,
                bounds=local_bounds,
                seed=123,
                maxiter=100,
                tol=1e-12,
                polish=True,
                init='sobol',
                popsize=10
            )
            if res2.fun < best_val:
                best_val = res2.fun
                r, theta, phi = res2.x
                best_coeffs = np.array(from_spherical(r, theta, phi))
        except Exception:
            pass
    
    # Phase 3: High-precision polish using L-BFGS-B in Cartesian coordinates
    if time.time() < deadline - 5.0:
        try:
            def objective_cartesian(x):
                try:
                    return _compute_bound(x[0], x[1], x[2])
                except Exception:
                    return 1e10
            
            # Use L-BFGS-B with tight tolerance for final polish
            result = scipy.optimize.minimize(
                objective_cartesian,
                best_coeffs,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if result.success and result.fun < best_val:
                best_val = result.fun
                best_coeffs = result.x
        except Exception:
            pass
    
    # Final safety check
    if time.time() > deadline - 2.0:
        # Return best found so far
        pass
    
    return best_coeffs
# EVOLVE-BLOCK-END
