import numpy as np
import time
from scipy.optimize import brentq, minimize_scalar, minimize, approx_fprime, differential_evolution

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

def _find_all_real_roots(qcoeffs, tol=1e-10):
    """Find all real roots of polynomial with high accuracy."""
    roots = np.roots(qcoeffs)
    real_roots = []
    for r in roots:
        if abs(r.imag) < tol * max(1.0, abs(r.real)):
            real_roots.append(float(r.real))
    return sorted(real_roots)

def _is_sign_change_root(qcoeffs, r, eps=1e-8):
    """Check if r is a sign-changing root."""
    vm = _eval_q(qcoeffs, r - eps)
    vp = _eval_q(qcoeffs, r + eps)
    return vm * vp < 0

def _largest_sign_change_root(qcoeffs):
    """Find the largest real root of q where q changes sign."""
    real_roots = _find_all_real_roots(qcoeffs)
    
    best_r = None
    for r in real_roots:
        if _is_sign_change_root(qcoeffs, r):
            if best_r is None or r > best_r:
                best_r = r
    return best_r

def _objective(c):
    """Compute the value r^2/(2*pi) for coefficient vector c."""
    c0, c1, c2 = c
    q = _build_q(c0, c1, c2)
    r = _largest_sign_change_root(q)
    if r is None:
        return 1e12
    return (r * r) / (2.0 * np.pi)

def _smooth_objective(c, beta=100.0):
    """
    Smooth surrogate: instead of just the largest sign-changing root,
    use a soft-max over all sign-changing roots weighted by exp(beta * r).
    This makes the objective differentiable almost everywhere.
    """
    c0, c1, c2 = c
    q = _build_q(c0, c1, c2)
    real_roots = _find_all_real_roots(q)
    
    sign_change_roots = [r for r in real_roots if _is_sign_change_root(q, r)]
    
    if not sign_change_roots:
        return 1e12
    
    # Soft-max: log(sum(exp(beta * r))) / beta approximates max(r) smoothly
    # Use log-sum-exp for numerical stability
    max_r = max(sign_change_roots)
    shifted = [beta * (r - max_r) for r in sign_change_roots]
    log_sum = max_r + np.log(np.sum(np.exp(shifted))) / beta
    
    return (log_sum * log_sum) / (2.0 * np.pi)

def _local_refine(c_start, max_iter=2000, tol=1e-12, use_smooth=True):
    """Local refinement using smooth surrogate then exact objective polish."""
    c = np.array(c_start, dtype=float)
    
    # First, use smooth objective for gradient-based optimization
    if use_smooth:
        try:
            # Use smooth objective with multiple betas for robustness
            for beta in [50.0, 100.0, 200.0]:
                def smooth_obj_wrapper(x):
                    return _smooth_objective(x, beta=beta)
                
                def smooth_grad_wrapper(x):
                    return approx_fprime(x, smooth_obj_wrapper, 1e-8)
                
                result = minimize(smooth_obj_wrapper, c, jac=smooth_grad_wrapper, 
                                method='L-BFGS-B', 
                                options={'maxiter': max_iter // 2, 'ftol': 1e-10, 'gtol': 1e-10})
                if result.fun < _objective(c):
                    c = result.x
        except Exception:
            pass
    
    # Then polish with exact objective
    best_val = _objective(c)
    
    try:
        result = minimize(_objective, c, method='Nelder-Mead', 
                        options={'maxiter': max_iter, 'xatol': tol, 'fatol': tol})
        if result.fun < best_val:
            c = result.x
            best_val = result.fun
    except Exception:
        pass
    
    # Coordinate descent polish for final precision
    for _ in range(50):
        improved = False
        for i in range(3):
            for scale in [1e-6, 1e-7, 1e-8, 1e-9]:
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
    return c, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 195.0  # 15s safety margin before 210s limit
    
    # Seed
    seed_c = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    best_c = seed_c.copy()
    best_val = _objective(best_c)
    
    # Phase 1: Differential evolution for global search
    if time.time() < deadline - 60:
        bounds = [(-2.0, 2.0), (-0.5, 0.5), (-0.02, 0.02)]
        try:
            de_result = differential_evolution(
                _objective, bounds, 
                maxiter=100, popsize=30, 
                tol=1e-8, seed=42,
                polish=True,
                workers=1
            )
            if de_result.fun < best_val:
                best_val = de_result.fun
                best_c = de_result.x.copy()
        except Exception:
            pass
    
    # Phase 2: Multi-start local refinement with smooth surrogate
    np.random.seed(42)
    n_restarts = 30
    for i in range(n_restarts):
        if time.time() > deadline - 30:
            break
        
        # Generate diverse random points around seed and best
        if i % 3 == 0:
            # Around seed
            c_rand = seed_c + np.random.normal(0, 0.3, size=3)
        elif i % 3 == 1:
            # Around best
            c_rand = best_c + np.random.normal(0, 0.1, size=3)
        else:
            # Wider random
            c_rand = np.array([
                np.random.uniform(-1.5, 1.5),
                np.random.uniform(-0.3, 0.3),
                np.random.uniform(-0.01, 0.01),
            ])
        
        c_ref, v_ref = _local_refine(c_rand, max_iter=500, tol=1e-11, use_smooth=True)
        if v_ref < best_val:
            best_val = v_ref
            best_c = c_ref.copy()
    
    # Phase 3: Fine grid around best found
    if time.time() < deadline - 20:
        fine_ranges = [
            (best_c[0] - 0.05, best_c[0] + 0.05, 9),
            (best_c[1] - 0.005, best_c[1] + 0.005, 9),
            (best_c[2] - 0.0002, best_c[2] + 0.0002, 9),
        ]
        
        fine_results = []
        for d0 in np.linspace(fine_ranges[0][0], fine_ranges[0][1], fine_ranges[0][2]):
            for d1 in np.linspace(fine_ranges[1][0], fine_ranges[1][1], fine_ranges[1][2]):
                for d2 in np.linspace(fine_ranges[2][0], fine_ranges[2][1], fine_ranges[2][2]):
                    if time.time() > deadline - 15:
                        break
                    pt = np.array([d0, d1, d2])
                    val = _objective(pt)
                    fine_results.append((val, pt))
        
        fine_results.sort(key=lambda x: x[0])
        
        for val, pt in fine_results[:15]:
            if time.time() > deadline - 10:
                break
            c_ref, v_ref = _local_refine(pt, max_iter=1000, tol=1e-12, use_smooth=True)
            if v_ref < best_val:
                best_val = v_ref
                best_c = c_ref.copy()
    
    # Phase 4: Final high-precision polish
    if time.time() < deadline - 5:
        c_final, v_final = _local_refine(best_c, max_iter=5000, tol=1e-14, use_smooth=True)
        if v_final < best_val:
            best_val = v_final
            best_c = c_final.copy()
    
    return best_c.tolist()
