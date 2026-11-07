"""Structural approach: search over root configurations with feasibility checking."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, brentq
from scipy.special import roots_hermite
import math

H0_0 = 1
H4_0 = 12
H8_0 = 1680
H12_0 = 2032128

def compute_value(c0, c1, c2):
    """Compute the task value: r^2 / (2*pi)."""
    c3 = -(c0 * H0_0 + c1 * H4_0 + c2 * H8_0) / H12_0
    
    a2  = -48.0*c1 - 13440.0*c2 - 1290240.0*c3
    a4  =  16.0*c1 + 11520.0*c2 + 2123328.0*c3
    a6  = -3072.0*c2 - 1548288.0*c3
    a8  =  256.0*c2 +  516096.0*c3
    a10 = -73728.0*c3
    a12 =  4096.0*c3
    
    # Check leading coefficient sign
    if a12 < 0:
        a2, a4, a6, a8, a10, a12 = -a2, -a4, -a6, -a8, -a10, -a12
    
    # Q(z) = a2 + a4*z + a6*z^2 + a8*z^3 + a10*z^4 + a12*z^5
    coeffs = np.array([a12, a10, a8, a6, a4, a2])
    
    # Remove leading zeros
    idx = 0
    while idx < len(coeffs) - 1 and abs(coeffs[idx]) < 1e-30:
        idx += 1
    coeffs = coeffs[idx:]
    
    if len(coeffs) <= 1:
        return 1e10
    
    roots_z = np.roots(coeffs)
    
    # Find positive real roots with sign change
    best_r = 0.0
    for rz in roots_z:
        if abs(rz.imag) < 1e-10 * max(1.0, abs(rz.real)):
            z = rz.real
            if z > 1e-12:
                # Check sign change
                eps = 1e-8 * max(1.0, abs(z))
                q_minus = np.polyval(coeffs, z - eps)
                q_plus = np.polyval(coeffs, z + eps)
                if q_minus * q_plus < 0:
                    r = math.sqrt(z)
                    if r > best_r:
                        best_r = r
    
    if best_r == 0.0:
        return 1e10
    
    return best_r**2 / (2.0 * np.pi)

def construct():
    start_time = time.time()
    deadline = start_time + 170.0  # Safety margin
    
    seed = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    best_val = compute_value(*seed)
    best_params = seed.copy()
    
    # Strategy 1: Coarse grid search in (c0, c1) space with c2=0, then refine
    # The seed has c2 very small, so explore c2=0 plane first
    np.random.seed(123)
    
    # Grid search
    c0_range = np.linspace(-3.0, 3.0, 50)
    c1_range = np.linspace(-1.0, 2.0, 40)
    c2_range = np.linspace(-0.1, 0.1, 10)
    
    for c2 in c2_range:
        if time.time() > deadline - 60.0:
            break
        for c1 in c1_range:
            if time.time() > deadline - 60.0:
                break
            for c0 in c0_range:
                if time.time() > deadline - 60.0:
                    break
                try:
                    v = compute_value(c0, c1, c2)
                    if v < best_val:
                        best_val = v
                        best_params = np.array([c0, c1, c2])
                except:
                    pass
    
    # Strategy 2: Local refinement from best grid point
    if time.time() < deadline - 30.0:
        try:
            result = minimize(
                lambda x: compute_value(*x),
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x.copy()
        except:
            pass
    
    # Strategy 3: Multi-start from diverse points
    if time.time() < deadline - 20.0:
        starts = [
            best_params.copy(),
            seed.copy(),
            np.array([-1.0, 0.0, 0.0]),
            np.array([0.0, 0.1, 0.0]),
            np.array([-0.5, 0.05, 0.001]),
            np.array([-2.0, 0.0, 0.0]),
            np.array([-1.13, 0.04, 0.0003]),
            np.array([-1.2, 0.03, 0.0001]),
            np.array([-1.1, 0.045, 0.0004]),
            np.array([-1.15, 0.035, 0.0002]),
        ]
        
        for start in starts:
            if time.time() > deadline - 10.0:
                break
            try:
                result = minimize(
                    lambda x: compute_value(*x),
                    start,
                    method='Nelder-Mead',
                    options={'maxiter': 3000, 'xatol': 1e-13, 'fatol': 1e-15, 'adaptive': True}
                )
                if result.fun < best_val:
                    best_val = result.fun
                    best_params = result.x.copy()
            except:
                pass
    
    # Strategy 4: High-precision polish
    if time.time() < deadline - 5.0:
        try:
            result = minimize(
                lambda x: compute_value(*x),
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 20000, 'xatol': 1e-15, 'fatol': 1e-16, 'adaptive': True}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x.copy()
        except:
            pass
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
