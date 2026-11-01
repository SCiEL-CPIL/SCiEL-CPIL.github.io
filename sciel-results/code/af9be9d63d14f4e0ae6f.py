"""Uncertainty bound minimizer using spherical parameterization with adaptive stochastic search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 195.0  # Safety margin before 210s kill

# Precompute integration weights I_n = integral x^{2n} e^{-x^2} dx
# I_n = (2n)! / (4^n n!) * sqrt(pi)
# We need n up to 13 (since max index in variance calc is 1+6+6=13)
_N_MAX = 14
_I = np.zeros(_N_MAX)
_I[0] = np.sqrt(np.pi)
for n in range(1, _N_MAX):
    _I[n] = (2*n - 1) * _I[n-1] / 8.0

# Precompute the index matrices for efficient dot products
_idx = np.add.outer(np.arange(7), np.arange(7))  # i+j, shape (7,7), values 0..12
_idx1 = _idx + 1  # 1+i+j, values 1..13

# Precompute Hermite coefficient matrix
# P(x) = a0*H0 + a4*H4 + a8*H8 + a12*H12
# H0 = 1
# H4 = x^4 - 6x^2 + 3
# H8 = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
# H12 = x^12 - 66x^10 + 1155x^8 - 6930x^6 + 13860x^4 - 7560x^2 + 945
#
# p_k = coefficient of x^{2k} in P(x)
# p0 = a0 + 3*a4 + 105*a8 + 945*a12
# p1 = -6*a4 - 420*a8 - 7560*a12
# p2 = a4 + 210*a8 + 13860*a12
# p3 = -28*a8 - 6930*a12
# p4 = a8 + 1155*a12
# p5 = -66*a12
# p6 = a12
#
# a12 = -(a0 + 3*a4 + 105*a8) / 945
#
# So p = M @ [a0, a4, a8] where M is 7x3:
_M = np.array([
    [1.0 + 945.0*(-1.0/945.0), 3.0 + 945.0*(-3.0/945.0), 105.0 + 945.0*(-105.0/945.0)],
    [0.0 - 7560.0*(-1.0/945.0), -6.0 - 7560.0*(-3.0/945.0), -420.0 - 7560.0*(-105.0/945.0)],
    [0.0 + 13860.0*(-1.0/945.0), 1.0 + 13860.0*(-3.0/945.0), 210.0 + 13860.0*(-105.0/945.0)],
    [0.0 - 6930.0*(-1.0/945.0), 0.0 - 6930.0*(-3.0/945.0), -28.0 - 6930.0*(-105.0/945.0)],
    [0.0 + 1155.0*(-1.0/945.0), 0.0 + 1155.0*(-3.0/945.0), 1.0 + 1155.0*(-105.0/945.0)],
    [0.0 - 66.0*(-1.0/945.0), 0.0 - 66.0*(-3.0/945.0), 0.0 - 66.0*(-105.0/945.0)],
    [0.0 + 1.0*(-1.0/945.0), 0.0 + 1.0*(-3.0/945.0), 0.0 + 1.0*(-105.0/945.0)],
])

# Verify: M @ [a0,a4,a8] should give p
# p0 = a0*(1-1) + a4*(3-3) + a8*(105-105) = 0? That's wrong.
# Let me recompute. a12 = -(a0 + 3a4 + 105a8)/945
# p0 = a0 + 3a4 + 105a8 + 945*a12 = a0 + 3a4 + 105a8 + 945*(-(a0+3a4+105a8)/945) = 0
# So p0 is always 0! That's the constraint g(0)=0.
# p1 = -6a4 - 420a8 - 7560*a12 = -6a4 - 420a8 - 7560*(-(a0+3a4+105a8)/945)
#    = -6a4 - 420a8 + (7560/945)*(a0+3a4+105a8)
#    = -6a4 - 420a8 + 8*(a0+3a4+105a8)
#    = 8a0 + (-6+24)a4 + (-420+840)a8
#    = 8a0 + 18a4 + 420a8
# p2 = a4 + 210a8 + 13860*a12 = a4 + 210a8 + 13860*(-(a0+3a4+105a8)/945)
#    = a4 + 210a8 - (13860/945)*(a0+3a4+105a8)
#    = a4 + 210a8 - 14.6667*(a0+3a4+105a8)
#    = -14.6667a0 + (1-44)a4 + (210-1540)a8
#    = -44/3 a0 - 43a4 - 1330a8
# Let me just compute numerically:
_M = np.zeros((7, 3))
# For each basis vector e0, e4, e8, compute p
for c_idx, c_val in enumerate([1.0, 0.0, 0.0]):
    a0, a4, a8 = (c_val if c_idx==0 else 0.0, c_val if c_idx==1 else 0.0, c_val if c_idx==2 else 0.0)
    a12 = -(a0 + 3.0*a4 + 105.0*a8) / 945.0
    _M[0, c_idx] = a0 + 3.0*a4 + 105.0*a8 + 945.0*a12
    _M[1, c_idx] = -6.0*a4 - 420.0*a8 - 7560.0*a12
    _M[2, c_idx] = a4 + 210.0*a8 + 13860.0*a12
    _M[3, c_idx] = -28.0*a8 - 6930.0*a12
    _M[4, c_idx] = a8 + 1155.0*a12
    _M[5, c_idx] = -66.0*a12
    _M[6, c_idx] = a12

# q_k = p_k * (-1)^k
_signs = ((-1.0) ** np.arange(7))
_Mq = _M * _signs[np.newaxis, :]

def _compute_bound(c):
    """
    Compute the uncertainty bound for normalized coefficient vector c on the sphere.
    c is a 3D vector with ||c||=1.
    """
    p = _M @ c
    q = _Mq @ c
    
    P_mat = np.outer(p, p)
    Q_mat = np.outer(q, q)
    
    m0 = np.sum(P_mat * _I[_idx])
    m2 = np.sum(P_mat * _I[_idx1])
    
    n0 = 2.0 * np.pi * np.sum(Q_mat * _I[_idx])
    n2 = 2.0 * np.pi * np.sum(Q_mat * _I[_idx1])
    
    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18
        
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18
        
    return np.sqrt(vf * vF)

def _project_to_sphere(c):
    """Project c onto the unit sphere."""
    norm = np.linalg.norm(c)
    if norm < 1e-15:
        return np.array([1.0, 0.0, 0.0])
    return c / norm

def construct():
    best_c = np.array([1.0, 0.0, 0.0])
    best_val = _compute_bound(best_c)
    
    # Multiple starting points on the sphere
    rng = np.random.RandomState(42)
    
    # Generate diverse starting points
    start_points = []
    # Add the known good region (near a0=-1.13, a4=0.04, a8=0.000308642)
    # Normalize it
    known = np.array([-1.13, 0.04, 0.000308642])
    start_points.append(_project_to_sphere(known))
    
    # Add random points
    for _ in range(20):
        v = rng.randn(3)
        start_points.append(_project_to_sphere(v))
    
    # Add some structured points
    angles = np.linspace(0, 2*np.pi, 12, endpoint=False)
    for phi in angles:
        for theta in [0.3, 0.7, 1.1, 1.5]:
            pt = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
            start_points.append(pt)
    
    # Phase 1: Evaluate all starting points
    for c in start_points:
        val = _compute_bound(c)
        if val < best_val:
            best_val = val
            best_c = c
    
    # Phase 2: Adaptive stochastic local search on the sphere
    # Use a simulated annealing-like approach with adaptive step sizes
    current_c = best_c.copy()
    current_val = best_val
    step_size = 0.1
    min_step = 1e-8
    max_iter = 50000
    patience = 5000
    no_improve = 0
    
    for it in range(max_iter):
        if time.time() > _DEADLINE - 15.0:
            break
        
        # Generate a random direction on the sphere
        rand_dir = rng.randn(3)
        rand_dir = rand_dir / np.linalg.norm(rand_dir)
        
        # Move in the random direction, then project back to sphere
        new_c = _project_to_sphere(current_c + step_size * rand_dir)
        new_val = _compute_bound(new_c)
        
        if new_val < current_val:
            current_c = new_c
            current_val = new_val
            step_size = min(step_size * 1.1, 0.5)
            no_improve = 0
        else:
            # Accept with some probability (simulated annealing style)
            if np.log(rng.rand()) < (current_val - new_val) / (0.01 * (1.0 + it/max_iter)):
                current_c = new_c
                current_val = new_val
            step_size = max(step_size * 0.95, min_step)
            no_improve += 1
        
        if new_val < best_val:
            best_val = new_val
            best_c = new_c.copy()
        
        if no_improve > patience:
            # Reset to best and reduce step size
            current_c = best_c.copy()
            current_val = best_val
            step_size = max(step_size * 0.5, min_step)
            no_improve = 0
    
    # Phase 3: Local refinement using scipy minimize on the sphere
    # Use a parameterization: c = [sin(t1)cos(t2), sin(t1)sin(t2), cos(t1)]
    def _sphere_to_params(c):
        t1 = np.arccos(np.clip(c[2], -1.0, 1.0))
        if np.abs(np.sin(t1)) < 1e-10:
            t2 = 0.0
        else:
            t2 = np.arctan2(c[1], c[0])
        return np.array([t1, t2])
    
    def _params_to_sphere(params):
        t1, t2 = params
        return np.array([np.sin(t1)*np.cos(t2), np.sin(t1)*np.sin(t2), np.cos(t1)])
    
    def _obj_sphere(params):
        c = _params_to_sphere(params)
        return _compute_bound(c)
    
    # Try local optimization from the best point
    if time.time() < _DEADLINE - 5.0:
        try:
            p0 = _sphere_to_params(best_c)
            res = minimize(_obj_sphere, p0, method='Nelder-Mead', 
                          options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14})
            if res.fun < best_val:
                best_val = res.fun
                best_c = _params_to_sphere(res.x)
        except Exception:
            pass
    
    # Convert back to original parameters (a0, a4, a8)
    # c is normalized, so we need to find the scale
    # The original problem has a0, a4, a8 free. The normalization is just for search.
    # We return the unnormalized coefficients that give the same direction.
    # Actually, the bound depends only on the direction of (a0, a4, a8) since it's homogeneous?
    # Let me check: if we scale (a0,a4,a8) by s, then p scales by s, m0 scales by s^2, m2 by s^2,
    # so vf = m2/m0 is unchanged. Similarly for vF. So the bound is scale-invariant!
    # Therefore we can return any scalar multiple. Let's return c itself (unit norm).
    
    return best_c
# EVOLVE-BLOCK-END
