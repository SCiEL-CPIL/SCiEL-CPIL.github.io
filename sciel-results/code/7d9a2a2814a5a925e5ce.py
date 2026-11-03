"""
Structural change: Root-finding on critical-point conditions instead of
black-box optimization.

The uncertainty bound B(c0, c4, c8) is a smooth function of the coefficients.
After fixing c0 = ±1 (scale invariance), the 2D objective B(c4, c8) is smooth
and its global minimum occurs where the gradient vanishes (interior) or on the
boundary. We use fsolve to find all real stationary points from many diverse
initial guesses, then evaluate B at each and pick the best.

This is a fundamentally different search paradigm: instead of iteratively
proposing new points via stochastic rules (DE, annealing) or gradient descent
(L-BFGS), we directly solve the algebraic system ∇B = 0, which is a finite
set of polynomial-like equations in 2 variables.
"""

# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize
import math
from itertools import product

def _compute_bound_and_grad(c0, c4, c8):
    """
    Compute the uncertainty bound AND its gradient w.r.t. c4, c8 analytically.
    Returns (bound, dB_dc4, dB_dc8).
    """
    c12 = -(c0 + 12.0*c4 + 168.0*c8) / 1330.0

    # Coefficients of P(x) in standard monomial basis
    # H0 = 1, H4 = 16x^4 - 48x^2 + 12, H8 = 256x^8 - 1344x^6 + 2688x^4 - 1344x^2
    # H12 = 4096x^12 - 36864x^10 + 118272x^8 - 180224x^6 + 133056x^4 - 32256x^2
    # P(x) = c0*H0 + c4*H4 + c8*H8 + c12*H12
    # Express as a0 + a2 x^2 + a4 x^4 + a6 x^6 + a8 x^8 + a10 x^10 + a12 x^12
    
    a0  = c0 + 12.0*c4 + 168.0*c8 + 1330.0*c12
    a2  = -48.0*c4 - 1344.0*c8 - 32256.0*c12
    a4  = 16.0*c4 + 2688.0*c8 + 133056.0*c12
    a6  = -1344.0*c8 - 180224.0*c12
    a8  = 256.0*c8 + 118272.0*c12
    a10 = -36864.0*c12
    a12 = 4096.0*c12

    # Derivatives of a_k w.r.t. c4 and c8
    # da0/dc4 = 12, da0/dc8 = 168, da0/dc12 = 1330
    # da2/dc4 = -48, da2/dc8 = -1344, da2/dc12 = -32256
    # da4/dc4 = 16, da4/dc8 = 2688, da4/dc12 = 133056
    # da6/dc4 = 0, da6/dc8 = -1344, da6/dc12 = -180224
    # da8/dc4 = 0, da8/dc8 = 256, da8/dc12 = 118272
    # da10/dc4 = 0, da10/dc8 = 0, da10/dc12 = -36864
    # da12/dc4 = 0, da12/dc8 = 0, da12/dc12 = 4096
    
    dc12_dc4 = -12.0/1330.0
    dc12_dc8 = -168.0/1330.0
    
    da0_dc4 = 12.0 + 1330.0*dc12_dc4
    da0_dc8 = 168.0 + 1330.0*dc12_dc8
    da2_dc4 = -48.0 + (-32256.0)*dc12_dc4
    da2_dc8 = -1344.0 + (-32256.0)*dc12_dc8
    da4_dc4 = 16.0 + 133056.0*dc12_dc4
    da4_dc8 = 2688.0 + 133056.0*dc12_dc8
    da6_dc4 = 0.0 + (-180224.0)*dc12_dc4
    da6_dc8 = -1344.0 + (-180224.0)*dc12_dc8
    da8_dc4 = 0.0 + 118272.0*dc12_dc4
    da8_dc8 = 256.0 + 118272.0*dc12_dc8
    da10_dc4 = 0.0 + (-36864.0)*dc12_dc4
    da10_dc8 = 0.0 + (-36864.0)*dc12_dc8
    da12_dc4 = 0.0 + 4096.0*dc12_dc4
    da12_dc8 = 0.0 + 4096.0*dc12_dc8

    p = np.array([a0, a2, a4, a6, a8, a10, a12])
    p_dc4 = np.array([da0_dc4, da2_dc4, da4_dc4, da6_dc4, da8_dc4, da10_dc4, da12_dc4])
    p_dc8 = np.array([da0_dc8, da2_dc8, da4_dc8, da6_dc8, da8_dc8, da10_dc8, da12_dc8])

    # q = p * p (convolution)
    q = np.convolve(p, p)
    q_dc4 = np.convolve(p_dc4, p) + np.convolve(p, p_dc4)
    q_dc8 = np.convolve(p_dc8, p) + np.convolve(p, p_dc8)

    # I[k] = q[k] * sqrt(pi) * 2^k * k!
    sqrt_pi = math.sqrt(math.pi)
    I = np.zeros(13)
    I_dc4 = np.zeros(13)
    I_dc8 = np.zeros(13)
    for k in range(13):
        factor = sqrt_pi * (2.0**k) * math.factorial(k)
        I[k] = q[k] * factor
        I_dc4[k] = q_dc4[k] * factor
        I_dc8[k] = q_dc8[k] * factor

    norm_sq = I[0]
    if norm_sq <= 1e-15:
        return 1e10, 0.0, 0.0

    x2 = I[1] / I[0]
    
    # Derivative of x2
    dx2_dc4 = (I_dc4[1]*I[0] - I[1]*I_dc4[0]) / (I[0]**2)
    dx2_dc8 = (I_dc8[1]*I[0] - I[1]*I_dc8[0]) / (I[0]**2)

    # R(x) = P'(x) - 2x*P(x) ... let's re-derive
    # Actually the bound involves ||R||^2 where R = x*P - P'/2 or similar.
    # Let me use the same R construction as the original code.
    
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
        
    # Derivatives of r[j] w.r.t. c4 and c8
    r_dc4 = np.zeros(8)
    r_dc8 = np.zeros(8)
    for j in range(8):
        val4 = 0.0
        val8 = 0.0
        if j <= 5:
            val4 += 2*(j+1)*(2*j+1)*p_dc4[j+1]
            val8 += 2*(j+1)*(2*j+1)*p_dc8[j+1]
        if j >= 1 and j <= 6:
            val4 += -4*j*p_dc4[j]
            val8 += -4*j*p_dc8[j]
        if j >= 1 and j <= 7:
            val4 += p_dc4[j-1]
            val8 += p_dc8[j-1]
        if j <= 6:
            val4 += -p_dc4[j]
            val8 += -p_dc8[j]
        r_dc4[j] = val4
        r_dc8[j] = val8

    s = np.convolve(r, r)
    s_dc4 = np.convolve(r_dc4, r) + np.convolve(r, r_dc4)
    s_dc8 = np.convolve(r_dc8, r) + np.convolve(r, r_dc8)

    I_R = np.zeros(15)
    I_R_dc4 = np.zeros(15)
    I_R_dc8 = np.zeros(15)
    for k in range(15):
        factor = sqrt_pi * (2.0**k) * math.factorial(k)
        I_R[k] = s[k] * factor
        I_R_dc4[k] = s_dc4[k] * factor
        I_R_dc8[k] = s_dc8[k] * factor

    norm_R_sq = I_R[0]
    if norm_R_sq <= 1e-15:
        return 1e10, 0.0, 0.0

    xi2 = norm_R_sq / (4.0 * math.pi**2 * norm_sq)

    # Derivative of xi2
    dnorm_R_sq_dc4 = I_R_dc4[0]
    dnorm_R_sq_dc8 = I_R_dc8[0]
    dxi2_dc4 = (dnorm_R_sq_dc4 * (4.0*math.pi**2*norm_sq) - norm_R_sq * (4.0*math.pi**2*norm_sq) * I_dc4[0] / I[0]) / ((4.0*math.pi**2*norm_sq)**2)
    # Wait, let me redo: xi2 = norm_R_sq / (4 pi^2 norm_sq)
    # dxi2/dc = dnorm_R_sq/dc / (4 pi^2 norm_sq) - norm_R_sq * dnorm_sq/dc / (4 pi^2 norm_sq^2)
    dxi2_dc4 = dnorm_R_sq_dc4 / (4.0*math.pi**2*norm_sq) - norm_R_sq * I_dc4[0] / (4.0*math.pi**2*norm_sq**2)
    dxi2_dc8 = dnorm_R_sq_dc8 / (4.0*math.pi**2*norm_sq) - norm_R_sq * I_dc8[0] / (4.0*math.pi**2*norm_sq**2)

    product = math.sqrt(max(x2 * xi2, 0.0))
    
    # d(product)/dc = 0.5 / product * (xi2 * dx2/dc + x2 * dxi2/dc)
    if product > 1e-15:
        dprod_dc4 = 0.5 / product * (xi2 * dx2_dc4 + x2 * dxi2_dc4)
        dprod_dc8 = 0.5 / product * (xi2 * dx2_dc8 + x2 * dxi2_dc8)
    else:
        dprod_dc4 = 0.0
        dprod_dc8 = 0.0

    return product, dprod_dc4, dprod_dc8

def _compute_bound_fast(c0, c4, c8):
    """Fast computation of the uncertainty product bound (no gradient)."""
    c12 = -(c0 + 12*c4 + 168*c8) / 1330.0
    
    a0 = c0 + 12*c4 + 168*c8 + 1330*c12
    a2 = -48*c4 - 1344*c8 - 32256*c12
    a4 = 16*c4 + 2688*c8 + 133056*c12
    a6 = -1344*c8 - 180224*c12
    a8 = 256*c8 + 118272*c12
    a10 = -36864*c12
    a12 = 4096*c12

    p = np.array([a0, a2, a4, a6, a8, a10, a12])
    q = np.convolve(p, p)
    
    I = np.zeros(13)
    for k in range(13):
        I[k] = q[k] * math.sqrt(math.pi) * (2**k) * math.factorial(k)
        
    norm_sq = I[0]
    if norm_sq <= 1e-15:
        return 1e10
        
    x2 = I[1] / I[0]
    
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
    I_R = np.zeros(15)
    for k in range(15):
        I_R[k] = s[k] * math.sqrt(math.pi) * (2**k) * math.factorial(k)
        
    norm_R_sq = I_R[0]
    if norm_R_sq <= 1e-15:
        return 1e10
        
    xi2 = norm_R_sq / (4 * math.pi**2 * norm_sq)
    product = math.sqrt(max(x2 * xi2, 0.0))
    return product

def construct():
    start_time = time.time()
    deadline = start_time + 175.0  # 35s safety margin before 210s kill

    seed_coeffs = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    best_val = _compute_bound_fast(seed_coeffs[0], seed_coeffs[1], seed_coeffs[2])
    best_coeffs = seed_coeffs.copy()

    def objective_2d(c4, c8):
        """Bound with c0=1."""
        return _compute_bound_fast(1.0, c4, c8)

    def gradient_2d(c4, c8):
        """Gradient of bound w.r.t. (c4, c8) with c0=1."""
        _, g4, g8 = _compute_bound_and_grad(1.0, c4, c8)
        return np.array([g4, g8])

    def gradient_system(x):
        """System of equations for fsolve: ∂B/∂c4 = 0, ∂B/∂c8 = 0."""
        return gradient_2d(x[0], x[1])

    # Generate diverse initial guesses for fsolve
    # Cover the plausible range of c4, c8
    c4_range = np.linspace(-5.0, 5.0, 11)
    c8_range = np.linspace(-1.0, 1.0, 11)
    
    initial_guesses = []
    for c4_i in c4_range:
        for c8_i in c8_range:
            initial_guesses.append([c4_i, c8_i])
    
    # Also add the seed point
    initial_guesses.append([seed_coeffs[1], seed_coeffs[2]])
    initial_guesses.append([0.0, 0.0])
    
    # Add some random points for diversity
    np.random.seed(42)
    for _ in range(30):
        initial_guesses.append([
            np.random.uniform(-5.0, 5.0),
            np.random.uniform(-1.0, 1.0)
        ])
    
    # Track found stationary points to avoid duplicates
    found_points = []
    found_vals = []
    
    for guess in initial_guesses:
        if time.time() > deadline - 15.0:
            break
        
        try:
            sol, info, ier, msg = scipy.optimize.fsolve(
                gradient_system, 
                guess, 
                full_output=True,
                maxfev=200
            )
            
            if ier != 1:
                continue
            
            c4_sol, c8_sol = sol
            
            # Check if this is a valid interior point (not on boundary)
            if abs(c4_sol) > 10.0 or abs(c8_sol) > 2.0:
                continue
            
            # Check if we've already found this point
            is_duplicate = False
            for fp in found_points:
                if abs(fp[0] - c4_sol) < 1e-4 and abs(fp[1] - c8_sol) < 1e-4:
                    is_duplicate = True
                    break
            if is_duplicate:
                continue
            
            # Evaluate the bound at this stationary point
            val = objective_2d(c4_sol, c8_sol)
            
            if val < 1e9:  # valid value
                found_points.append([c4_sol, c8_sol])
                found_vals.append(val)
                
                if val < best_val:
                    best_val = val
                    best_coeffs = np.array([1.0, c4_sol, c8_sol])
                    
        except Exception:
            continue
    
    # Also try c0 = -1 (equivalent by symmetry but let's verify)
    if time.time() < deadline - 15.0:
        def gradient_system_neg(x):
            _, g4, g8 = _compute_bound_and_grad(-1.0, x[0], x[1])
            return np.array([g4, g8])
        
        for guess in initial_guesses[:50]:  # fewer guesses for negative c0
            if time.time() > deadline - 15.0:
                break
            try:
                sol, info, ier, msg = scipy.optimize.fsolve(
                    gradient_system_neg, 
                    guess, 
                    full_output=True,
                    maxfev=200
                )
                if ier != 1:
                    continue
                c4_sol, c8_sol = sol
                if abs(c4_sol) > 10.0 or abs(c8_sol) > 2.0:
                    continue
                val = _compute_bound_fast(-1.0, c4_sol, c8_sol)
                if val < best_val and val < 1e9:
                    best_val = val
                    best_coeffs = np.array([-1.0, c4_sol, c8_sol])
            except Exception:
                continue

    # If time remains, do a final polish with L-BFGS-B from the best point found
    if time.time() < deadline - 10.0 and best_coeffs[0] != 0:
        c0_sign = best_coeffs[0]
        try:
            res_local = scipy.optimize.minimize(
                lambda x: _compute_bound_fast(c0_sign, x[0], x[1]),
                [best_coeffs[1], best_coeffs[2]],
                method='L-BFGS-B',
                bounds=[(-10.0, 10.0), (-2.0, 2.0)],
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-14}
            )
            if res_local.fun < best_val:
                best_val = res_local.fun
                best_coeffs = np.array([c0_sign, res_local.x[0], res_local.x[1]])
        except Exception:
            pass

    # Final safety: compare with seed
    seed_val = _compute_bound_fast(seed_coeffs[0], seed_coeffs[1], seed_coeffs[2])
    if best_val > seed_val:
        return seed_coeffs
    return best_coeffs

# EVOLVE-BLOCK-END
