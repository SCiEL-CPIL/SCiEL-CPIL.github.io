import numpy as np
import time
from scipy.optimize import brentq, minimize_scalar, minimize
from fractions import Fraction
import math

def _hermite_even_coeffs():
    """Return coefficient arrays for H0, H4, H8, H12 in ascending power order."""
    H0 = np.array([1.0])
    H4 = np.array([12.0, 0.0, -48.0, 0.0, 16.0])
    H8 = np.array([1680.0, 0.0, -13440.0, 0.0, 11520.0, 0.0, -3072.0, 0.0, 256.0])
    H12 = np.array([10321920.0, 0.0, -5529600.0, 0.0, 8709120.0, 0.0, -4915200.0, 0.0, 1032192.0, 0.0, -98304.0, 0.0, 4096.0])
    return H0, H4, H8, H12

def _build_q(c0, c1, c2):
    """Build q(x) = g(x)/x^2 coefficients in ascending power order."""
    H0, H4, H8, H12 = _hermite_even_coeffs()
    # c3 from g(0)=0: c0*H0(0) + c1*H4(0) + c2*H8(0) + c3*H12(0) = 0
    # H0(0)=1, H4(0)=12, H8(0)=1680, H12(0)=10321920
    c3 = -(c0 * 1.0 + c1 * 12.0 + c2 * 1680.0) / 10321920.0
    # Build g coefficients
    g = np.zeros(13)
    g[0] += c0 * H0[0]
    for i in range(len(H4)):
        g[i] += c1 * H4[i]
    for i in range(len(H8)):
        g[i] += c2 * H8[i]
    for i in range(len(H12)):
        g[i] += c3 * H12[i]
    # q = g / x^2, so q coefficients are g[2:]
    q = g[2:]
    # Trim trailing zeros
    while len(q) > 1 and abs(q[-1]) < 1e-15:
        q = q[:-1]
    return q

def _eval_q(qcoeffs, x):
    """Evaluate polynomial q at x using Horner's method."""
    result = 0.0
    for c in reversed(qcoeffs):
        result = result * x + c
    return result

def _largest_sign_change_root(qcoeffs):
    """Find the largest real root of q where q changes sign."""
    # Find all real roots using numpy
    roots = np.roots(qcoeffs)
    real_roots = []
    for r in roots:
        if abs(r.imag) < 1e-10:
            real_roots.append(float(r.real))
    real_roots.sort()
    
    best_r = None
    for r in real_roots:
        # Check sign change with adaptive epsilon
        for eps in [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]:
            vm = _eval_q(qcoeffs, r - eps)
            vp = _eval_q(qcoeffs, r + eps)
            if vm * vp < 0:
                if best_r is None or r > best_r:
                    best_r = r
                break
    return best_r

def _value_exact(c0, c1, c2):
    """Compute value using exact rational arithmetic for higher precision."""
    # Convert to fractions for exact arithmetic
    c0_f = Fraction(c0).limit_denominator(10**12)
    c1_f = Fraction(c1).limit_denominator(10**12)
    c2_f = Fraction(c2).limit_denominator(10**12)
    
    H0_0 = 1
    H4_0 = 12
    H8_0 = 1680
    H12_0 = 10321920
    
    c3_f = -(c0_f * H0_0 + c1_f * H4_0 + c2_f * H8_0) / H12_0
    
    # Build g coefficients as fractions
    H4 = [12, 0, -48, 0, 16]
    H8 = [1680, 0, -13440, 0, 11520, 0, -3072, 0, 256]
    H12 = [10321920, 0, -5529600, 0, 8709120, 0, -4915200, 0, 1032192, 0, -98304, 0, 4096]
    
    g = [Fraction(0)] * 13
    g[0] += c0_f * 1
    for i, h4 in enumerate(H4):
        g[i] += c1_f * h4
    for i, h8 in enumerate(H8):
        g[i] += c2_f * h8
    for i, h12 in enumerate(H12):
        g[i] += c3_f * h12
    
    # q = g / x^2
    q = g[2:]
    
    # Find roots of q using numpy on float representation
    q_float = np.array([float(x) for x in q])
    roots = np.roots(q_float)
    real_roots = []
    for r in roots:
        if abs(r.imag) < 1e-10:
            real_roots.append(float(r.real))
    real_roots.sort()
    
    best_r = None
    for r in real_roots:
        for eps in [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]:
            # Evaluate q at r-eps and r+eps using exact fractions
            def eval_q_frac(x):
                result = Fraction(0)
                for coeff in reversed(q):
                    result = result * x + coeff
                return result
            
            vm = eval_q_frac(Fraction(r - eps).limit_denominator(10**12))
            vp = eval_q_frac(Fraction(r + eps).limit_denominator(10**12))
            if vm * vp < 0:
                if best_r is None or r > best_r:
                    best_r = r
                break
    
    if best_r is None:
        return 1e12
    return (best_r * best_r) / (2.0 * math.pi)

def _objective(c):
    """Compute the value r^2/(2*pi) for coefficient vector c."""
    c0, c1, c2 = c
    q = _build_q(c0, c1, c2)
    r = _largest_sign_change_root(q)
    if r is None:
        return 1e12
    return (r * r) / (2.0 * np.pi)

def _objective_smooth(c, softness=0.01):
    """Smooth surrogate: use soft-min over sign-changing roots."""
    c0, c1, c2 = c
    q = _build_q(c0, c1, c2)
    roots = np.roots(q)
    real_roots = []
    for r in roots:
        if abs(r.imag) < 1e-8:
            real_roots.append(float(r.real))
    
    if not real_roots:
        return 1e12
    
    # For each real root, check if it's a sign change and compute "signed" value
    signed_vals = []
    for r in real_roots:
        eps = 1e-6
        vm = _eval_q(q, r - eps)
        vp = _eval_q(q, r + eps)
        if vm * vp < 0:
            # Sign change: this is a valid root
            signed_vals.append(r * r)
    
    if not signed_vals:
        return 1e12
    
    # Use soft-max (we want the largest r^2)
    # soft_max = log(sum(exp(x_i / T))) * T
    vals = np.array(signed_vals)
    max_val = np.max(vals)
    soft_max = max_val + softness * np.log(np.sum(np.exp((vals - max_val) / softness)))
    return soft_max / (2.0 * np.pi)

def _local_refine(c_start, max_iter=500, tol=1e-10):
    """Local refinement using Nelder-Mead with fallback to coordinate descent."""
    c = np.array(c_start, dtype=float)
    best_val = _objective(c)
    
    # Try Nelder-Mead first
    try:
        result = minimize(_objective, c, method='Nelder-Mead', 
                        options={'maxiter': max_iter, 'xatol': tol, 'fatol': tol})
        if result.fun < best_val:
            c = result.x
            best_val = result.fun
    except Exception:
        pass
    
    # Coordinate descent polish with finer scales
    for _ in range(100):
        improved = False
        for i in range(3):
            for scale in [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]:
                for direction in [-1, 1]:
                    c_new = c.copy()
                    c_new[i] += direction * scale
                    val = _objective(c_new)
                    if val < best_val - 1e-14:
                        c = c_new
                        best_val = val
                        improved = True
                        break
                if improved:
                    break
        if not improved:
            break
    
    # Final high-precision polish using exact arithmetic
    try:
        exact_val = _value_exact(c[0], c[1], c[2])
        if exact_val < best_val:
            best_val = exact_val
    except Exception:
        pass
    
    return c, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 170.0  # 10s safety margin
    
    # Seed
    seed_c = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    best_c = seed_c.copy()
    best_val = _objective(best_c)
    
    # Phase 1: Grid exploration around seed
    # Explore a range of values for each coefficient
    base = seed_c.copy()
    ranges = [
        (base[0] - 0.5, base[0] + 0.5, 8),
        (base[1] - 0.05, base[1] + 0.05, 8),
        (base[2] - 0.002, base[2] + 0.002, 8),
    ]
    
    # Do a coarse grid search
    grid_points = []
    for d0 in np.linspace(ranges[0][0], ranges[0][1], ranges[0][2]):
        for d1 in np.linspace(ranges[1][0], ranges[1][1], ranges[1][2]):
            for d2 in np.linspace(ranges[2][0], ranges[2][1], ranges[2][2]):
                if time.time() > deadline:
                    return best_c
                grid_points.append([d0, d1, d2])
    
    # Evaluate all grid points
    grid_results = []
    for pt in grid_points:
        if time.time() > deadline:
            break
        val = _objective(pt)
        grid_results.append((val, pt))
    
    grid_results.sort(key=lambda x: x[0])
    
    # Refine top candidates
    for val, pt in grid_results[:30]:
        if time.time() > deadline:
            break
        c_ref, v_ref = _local_refine(pt, max_iter=300)
        if v_ref < best_val:
            best_val = v_ref
            best_c = c_ref
    
    # Phase 2: Random restarts from diverse points
    np.random.seed(12345)
    for i in range(150):
        if time.time() > deadline:
            break
        # Generate random points in a broader region
        c_rand = np.array([
            np.random.uniform(-2.0, 2.0),
            np.random.uniform(-0.3, 0.3),
            np.random.uniform(-0.02, 0.02),
        ])
        c_ref, v_ref = _local_refine(c_rand, max_iter=200)
        if v_ref < best_val:
            best_val = v_ref
            best_c = c_ref
    
    # Phase 3: Use smooth surrogate to guide further search
    if time.time() < deadline - 30:
        try:
            result_smooth = minimize(_objective_smooth, best_c, method='Nelder-Mead',
                                   options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10})
            if result_smooth.fun < best_val * 1.01:  # Check if smooth surrogate suggests improvement
                c_smooth = result_smooth.x
                c_ref, v_ref = _local_refine(c_smooth, max_iter=500)
                if v_ref < best_val:
                    best_val = v_ref
                    best_c = c_ref
        except Exception:
            pass
    
    # Phase 4: Final polish on best found
    if time.time() < deadline - 15:
        c_final, v_final = _local_refine(best_c, max_iter=1000, tol=1e-12)
        if v_final < best_val:
            best_val = v_final
            best_c = c_final
    
    # Phase 5: Very fine coordinate search around best
    if time.time() < deadline - 5:
        for i in range(3):
            for scale in [1e-8, 1e-9, 1e-10, 1e-11]:
                for direction in [-1, 1]:
                    if time.time() > deadline:
                        break
                    c_new = best_c.copy()
                    c_new[i] += direction * scale
                    val = _objective(c_new)
                    if val < best_val - 1e-15:
                        best_c = c_new
                        best_val = val
    
    return best_c.tolist()
