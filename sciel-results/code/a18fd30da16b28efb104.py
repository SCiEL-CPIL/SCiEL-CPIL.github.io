"""Improved: spherical parameterization exploiting scale invariance of the C4 bound."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import roots_hermite
from scipy.optimize import minimize
import time

_DEADLINE = 190

def _hermite_h(n, x):
    """Physicists' Hermite polynomials."""
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return 2.0 * x
    elif n == 2:
        return 4.0 * x**2 - 2.0
    elif n == 3:
        return 8.0 * x**3 - 12.0 * x
    elif n == 4:
        return 16.0 * x**4 - 48.0 * x**2 + 12.0
    elif n == 7:
        return 128.0*x**7 - 4480.0*x**5 + 40320.0*x**3 - 67200.0*x
    elif n == 8:
        return 256.0*x**8 - 4480.0*x**6 + 26880.0*x**4 - 67200.0*x**2 + 40320.0
    elif n == 11:
        return (2048.0*x**11 - 147456.0*x**9 + 3133440.0*x**7
                - 30198720.0*x**5 + 122305920.0*x**3 - 145152000.0*x)
    elif n == 12:
        return (4096.0*x**12 - 147456.0*x**10 + 2580480.0*x**8
                - 20321280.0*x**6 + 81729600.0*x**4
                - 172972800.0*x**2 + 103219200.0)
    else:
        raise ValueError(f"Hermite H_{n} not implemented")

_quad_data = None

def _get_quad_data():
    global _quad_data
    if _quad_data is None:
        n_quad = 256
        x_quad, w_quad = roots_hermite(n_quad)
        
        H0 = _hermite_h(0, x_quad)
        H4 = _hermite_h(4, x_quad)
        H8 = _hermite_h(8, x_quad)
        H12 = _hermite_h(12, x_quad)
        
        dH4 = 8.0 * _hermite_h(3, x_quad)
        dH8 = 16.0 * _hermite_h(7, x_quad)
        dH12 = 24.0 * _hermite_h(11, x_quad)
        
        x2 = x_quad**2
        x2w = x2 * w_quad
        
        _quad_data = (x_quad, w_quad, H0, H4, H8, H12, dH4, dH8, dH12, x2, x2w)
    return _quad_data

H0_0 = 1.0
H4_0 = 12.0
H8_0 = 40320.0
H12_0 = 103219200.0

def _evaluate_from_directions(v):
    """
    Given a 3D direction vector v (nonzero), compute the C4 bound.
    The bound is scale-invariant, so only direction matters.
    Returns (bound, log_bound) or (inf, inf) if invalid.
    """
    if time.time() > _DEADLINE:
        return float('inf'), float('inf')
    
    x_quad, w_quad, H0, H4, H8, H12, dH4, dH8, dH12, x2, x2w = _get_quad_data()
    
    a0, a4, a8 = v
    
    # Determine d from P(0) = 0: a0 + 12*a4 + 40320*a8 + d*H12(0) = 0
    d = -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0
    
    # Build P = a0*H0 + a4*H4 + a8*H8 + d*H12
    # and dP - x*P for the Fourier side
    P = a0*H0 + a4*H4 + a8*H8 + d*H12
    P2 = P * P
    
    # Q = dP - x*P
    dP = a4*dH4 + a8*dH8 + d*dH12  # dH0 = 0
    Q = dP - x_quad * P
    Q2 = Q * Q
    
    S0 = np.sum(w_quad * P2)
    if S0 <= 1e-15:
        return float('inf'), float('inf')
    
    Sx = np.sum(x2w * P2)
    if Sx <= 1e-15:
        return float('inf'), float('inf')
    
    Sd = np.sum(w_quad * Q2)
    if Sd <= 1e-15:
        return float('inf'), float('inf')
    
    var_x = Sx / S0
    var_k = Sd / S0
    
    if var_x <= 0 or var_k <= 0:
        return float('inf'), float('inf')
    
    log_bound = np.log(var_x) + np.log(var_k)
    bound = np.exp(log_bound)
    
    return bound, log_bound

def _bound_from_angles(theta, phi):
    """
    Convert spherical angles to direction and compute bound.
    theta in [0, pi], phi in [0, 2*pi)
    v = (sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta))
    """
    st = np.sin(theta)
    ct = np.cos(theta)
    v = np.array([st * np.cos(phi), st * np.sin(phi), ct])
    return _evaluate_from_directions(v)

def construct():
    start_time = time.time()
    
    seed = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    seed_norm = np.linalg.norm(seed)
    if seed_norm < 1e-15:
        seed = np.array([1.0, 0.0, 0.0])
        seed_norm = 1.0
    seed_dir = seed / seed_norm
    
    # Convert seed direction to spherical angles
    sx = seed_dir[0]
    sy = seed_dir[1]
    sz = seed_dir[2]
    
    seed_theta = np.arccos(np.clip(sz, -1.0, 1.0))
    seed_phi = np.arctan2(sy, sx)
    if seed_phi < 0:
        seed_phi += 2.0 * np.pi
    
    seed_bound, seed_log = _evaluate_from_directions(seed)
    best_bound = seed_bound
    best_v = seed.copy()
    best_log = seed_log
    
    if best_log == float('inf'):
        best_log = 999.0
        best_bound = np.exp(best_log)
    
    # === Phase 1: High-resolution grid search on the sphere ===
    # Adaptive grid: start coarse, refine near best
    grid_results = []
    n_theta = 60
    n_phi = 120
    
    thetas = np.linspace(0.0, np.pi, n_theta)
    phis = np.linspace(0.0, 2.0*np.pi, n_phi, endpoint=False)
    
    # Vectorized evaluation for grid
    best_grid_log = float('inf')
    best_grid_theta = seed_theta
    best_grid_phi = seed_phi
    
    for i_t, th in enumerate(thetas):
        for i_p, ph in enumerate(phis):
            if time.time() > _DEADLINE - 60:
                break
            st = np.sin(th)
            ct = np.cos(th)
            v = np.array([st * np.cos(ph), st * np.sin(ph), ct])
            bound, log_b = _evaluate_from_directions(v)
            if log_b < best_grid_log:
                best_grid_log = log_b
                best_grid_theta = th
                best_grid_phi = ph
                best_v = v.copy()
                best_bound = bound
                best_log = log_b
    
    # === Phase 2: Local refinement using gradient-based optimization ===
    # Use the analytic gradient on the sphere for all local optimizations
    
    # Compute best direction as unit vector
    best_v_unit = best_v / np.linalg.norm(best_v)
    
    # Phase 3: Direct 3D optimization with gradient (scale-invariant)
    def _sphere_obj_grad(params):
        v = np.array(params, dtype=float)
        vn = np.linalg.norm(v)
        if vn < 1e-10:
            return 999.0, np.zeros(3)
        v_unit = v / vn
        obj, grad_t = _sphere_objective_and_grad(v_unit)
        if obj == float('inf'):
            return 999.0, np.zeros(3)
        return obj, grad_t
    
    # Start from current best
    if time.time() < _DEADLINE - 40:
        try:
            res = minimize(_sphere_obj_grad, best_v_unit, method='L-BFGS-B',
                         jac=True, options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-12})
            if res.fun < best_log:
                best_log = res.fun
                best_bound = np.exp(res.fun)
                best_v = res.x.copy()
                best_v_unit = best_v / np.linalg.norm(best_v)
        except Exception:
            pass
    
    # === Phase 3: Multiple random restarts with gradient-based optimization ===
    np.random.seed(12345)
    n_restarts = 15
    if time.time() < _DEADLINE - 30:
        for i in range(n_restarts):
            if time.time() > _DEADLINE - 30:
                break
            
            if i < 8:
                # Uniform random direction
                z1, z2, z3 = np.random.randn(3)
                v_rand = np.array([z1, z2, z3])
                v_rand /= np.linalg.norm(v_rand)
            else:
                # Perturbation of best
                v_rand = best_v_unit + np.random.randn(3) * 0.3
                v_rand /= np.linalg.norm(v_rand)
            
            try:
                res = minimize(_sphere_obj_grad, v_rand, method='L-BFGS-B',
                             jac=True, options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-12})
                if res.fun < best_log:
                    best_log = res.fun
                    best_bound = np.exp(res.fun)
                    best_v = res.x.copy()
                    best_v_unit = best_v / np.linalg.norm(best_v)
            except Exception:
                pass
    
    # === Phase 4: Direct 3D Nelder-Mead fallback (handles gradient singularity at origin) ===
    def _obj_3d(params):
        if time.time() > _DEADLINE - 10:
            return float('inf')
        v = np.array(params, dtype=float)
        vn = np.linalg.norm(v)
        if vn < 1e-10:
            return 999.0
        v_unit = v / vn
        obj, _ = _sphere_objective_and_grad(v_unit)
        return obj if np.isfinite(obj) else 999.0
    
    if time.time() < _DEADLINE - 15:
        try:
            res = minimize(_obj_3d, best_v_unit, method='Nelder-Mead',
                         options={'maxiter': 5000, 'xatol': 1e-13, 'fatol': 1e-15, 'adaptive': True})
            if res.fun < best_log:
                best_log = res.fun
                best_bound = np.exp(res.fun)
                best_v = res.x.copy()
                best_v_unit = best_v / np.linalg.norm(best_v)
        except Exception:
            pass
    
    # === Phase 5: Final high-precision polish ===
    if time.time() < _DEADLINE - 8:
        try:
            res = minimize(_sphere_obj_grad, best_v_unit, method='L-BFGS-B',
                         jac=True, options={'maxiter': 10000, 'ftol': 1e-16, 'gtol': 1e-14})
            if res.fun < best_log:
                best_log = res.fun
                best_v = res.x.copy()
        except Exception:
            pass
    
    # Final Nelder-Mead polish
    if time.time() < _DEADLINE - 3:
        try:
            res = minimize(_obj_3d, best_v / np.linalg.norm(best_v), method='Nelder-Mead',
                         options={'maxiter': 3000, 'xatol': 1e-14, 'fatol': 1e-15, 'adaptive': True})
            if res.fun < best_log:
                best_log = res.fun
                best_v = res.x.copy()
        except Exception:
            pass
    
    # Return the three coefficients (a0, a4, a8) from best direction
    # Scale doesn't matter for the bound, but return the actual coefficient values
    # Normalize to avoid numerical issues
    bv = best_v / np.linalg.norm(best_v)
    return np.array(bv)

# EVOLVE-BLOCK-END
