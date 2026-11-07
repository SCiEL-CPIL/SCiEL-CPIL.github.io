"""Search for coefficients minimizing r^2/(2*pi) where r is the largest
sign-changing real root of q = g/x^2, subject to g >= 0."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.special import hermite


def _hermite_coeffs(n):
    """Return coefficients of physicists' Hermite polynomial H_n(x) in ascending powers."""
    coeffs = np.zeros(n + 1)
    if n == 0:
        coeffs[0] = 1.0
        return coeffs
    if n == 1:
        coeffs[1] = 2.0
        return coeffs
    h_prev = np.array([1.0])
    h_curr = np.array([0.0, 2.0])
    for k in range(2, n + 1):
        h_next = np.zeros(k + 1)
        h_next[1:] += 2.0 * h_curr
        h_next[:len(h_prev)] -= 2.0 * k * h_prev
        h_prev, h_curr = h_curr, h_next
    return h_curr


# Precompute Hermite coefficients
_H0 = _hermite_coeffs(0)
_H4 = _hermite_coeffs(4)
_H8 = _hermite_coeffs(8)
_H12 = _hermite_coeffs(12)
_H0_0 = _H0[0]
_H4_0 = _H4[0]
_H8_0 = _H8[0]
_H12_0 = _H12[0]
_H12_lead = _H12[-1]  # 2^12 = 4096


def _build_q(c0, c1, c2):
    """Return coefficients of q(x)=g(x)/x^2 (ascending powers) as a float array, and c3."""
    c0 = float(c0); c1 = float(c1); c2 = float(c2)
    c3 = -(c0 * _H0_0 + c1 * _H4_0 + c2 * _H8_0) / _H12_0
    
    g = np.zeros(13)
    g[:len(_H0)] += c0 * _H0
    g[:len(_H4)] += c1 * _H4
    g[:len(_H8)] += c2 * _H8
    g[:len(_H12)] += c3 * _H12
    
    q = np.zeros(11)
    q[:] = g[2:]
    return q, c3


def _find_signchange_roots(q, tol=1e-8):
    """Find all real roots of q where q changes sign, using bisection refinement."""
    # First find candidate roots via numpy
    nz = np.nonzero(q)[0]
    if len(nz) == 0:
        return []
    q_trimmed = q[nz[0]:]
    roots = np.roots(q_trimmed)
    real_roots = [z.real for z in roots if abs(z.imag) < tol * max(1.0, abs(z))]
    
    if not real_roots:
        return []
    
    # For each real root, check if it's a sign change using small perturbation
    signchange_roots = []
    for r in real_roots:
        # Use multiple epsilon values to be robust
        eps = max(1e-6, 1e-4 * abs(r))
        vals = [np.polyval(q, r - eps), np.polyval(q, r + eps)]
        # Also check at r itself
        val_at_r = np.polyval(q, r)
        
        # A sign change means the function crosses zero
        # Check if values on both sides have opposite signs
        if len(vals) == 2:
            if vals[0] * vals[1] < 0:
                signchange_roots.append(r)
            elif abs(val_at_r) < 1e-10:
                # Check more carefully
                eps2 = eps * 0.1
                v1 = np.polyval(q, r - eps2)
                v2 = np.polyval(q, r + eps2)
                if v1 * v2 < 0:
                    signchange_roots.append(r)
    
    return signchange_roots


def _signchange_root(q):
    """Largest real root of q where q changes sign."""
    roots = _find_signchange_roots(q)
    if not roots:
        return None
    return max(roots)


def _eval_g(c0, c1, c2, xs, c3=None):
    """Evaluate g(x) at points xs."""
    if c3 is None:
        _, c3 = _build_q(c0, c1, c2)
    g = np.zeros(13)
    g[:len(_H0)] += c0 * _H0
    g[:len(_H4)] += c1 * _H4
    g[:len(_H8)] += c2 * _H8
    g[:len(_H12)] += c3 * _H12
    return np.polyval(g, xs)


def _check_nonnegativity(c0, c1, c2, c3, n_points=1000, max_x=30.0):
    """Check if g(x) >= 0 for all x >= 0. Returns (is_valid, min_value)."""
    # Check on [0, max_x] with fine grid
    xs = np.linspace(0.0, max_x, n_points)
    g = _eval_g(c0, c1, c2, xs, c3)
    
    # Also check behavior at infinity via leading coefficient
    g_lead = c3 * _H12_lead
    if g_lead < 0:
        g = -g
        g_lead = -g_lead
    
    mn = float(np.min(g))
    if mn < -1e-8:
        return False, mn
    
    # Check if g tends to +infinity (g_lead > 0 ensures this)
    if g_lead < 0:
        return False, float('-inf')
    
    return True, mn


def _objective(c, smooth_alpha=0.05):
    """Smooth objective for optimization."""
    c0, c1, c2 = c[0], c[1], c[2]
    q, c3 = _build_q(c0, c1, c2)
    
    # Find sign-changing root
    r = _signchange_root(q)
    if r is None:
        # No sign change - penalize heavily
        return 1e6
    
    obj = r * r
    
    # Check non-negativity with smooth penalty
    g_lead = c3 * _H12_lead
    xs = np.linspace(0.0, 25.0, 500)
    g = _eval_g(c0, c1, c2, xs, c3)
    if g_lead < 0:
        g = -g
    
    mn = float(np.min(g))
    if mn < 0:
        # Smooth penalty: use soft-min approximation
        # Penalize the negative part
        neg_part = -mn
        obj += 100.0 * neg_part ** 2 / max(1.0, r * r)
    
    # Penalize if leading coefficient is negative (g -> -inf)
    if g_lead < 0:
        obj += 50.0 * g_lead ** 2
    
    return obj


def _value(c):
    """Exact value = r^2/(2*pi) for the sign-changing root; None if invalid."""
    c0, c1, c2 = c[0], c[1], c[2]
    q, c3 = _build_q(c0, c1, c2)
    g_lead = c3 * _H12_lead
    
    r = _signchange_root(q)
    if r is None:
        return None
    
    # Check non-negativity
    valid, mn = _check_nonnegativity(c0, c1, c2, c3)
    if not valid:
        return None
    
    return r * r / (2.0 * np.pi)


def _local_refine(c0, c1, c2, time_budget=5.0):
    """Refine coefficients using local optimization."""
    t_start = time.time()
    c_init = np.array([c0, c1, c2], dtype=float)
    
    best_c = c_init.copy()
    best_obj = _objective(c_init)
    
    # Nelder-Mead refinement
    try:
        res = minimize(_objective, c_init, method='Nelder-Mead',
                      options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14,
                              'adaptive': True})
        if res.fun < best_obj:
            best_c = res.x
            best_obj = res.fun
    except Exception:
        pass
    
    if time.time() - t_start > time_budget:
        return best_c
    
    # L-BFGS-B polish
    try:
        res2 = minimize(_objective, best_c, method='L-BFGS-B',
                       options={'maxiter': 1000, 'gtol': 1e-15, 'ftol': 1e-15})
        if res2.fun < best_obj:
            best_c = res2.x
            best_obj = res2.fun
    except Exception:
        pass
    
    return best_c


def construct():
    t0 = time.time()
    deadline = t0 + 200.0  # Safety margin before 210s kill
    
    seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    seed_val = _value(seed)
    if seed_val is None:
        seed_val = 1.0 / (2.0 * np.pi)
    
    best_c = seed.copy()
    best_val = seed_val
    
    def _try_update(c):
        nonlocal best_c, best_val
        v = _value(c)
        if v is not None and v < best_val:
            best_c = c.copy()
            best_val = v
    
    # Phase 1: Global exploration with differential evolution
    if time.time() < deadline - 60.0:
        try:
            bounds = [(-5.0, 5.0), (-5.0, 5.0), (-0.1, 0.1)]
            de_result = differential_evolution(_objective, bounds,
                                               maxiter=50, popsize=20,
                                               seed=42, tol=1e-8,
                                               init='sobol',
                                               polish=False,
                                               update_interval=10)
            _try_update(de_result.x)
            # Local refine the DE result
            if time.time() < deadline - 30.0:
                refined = _local_refine(*de_result.x, time_budget=10.0)
                _try_update(refined)
        except Exception:
            pass
    
    # Phase 2: Multi-start local optimization
    rng = np.random.default_rng(12345)
    starts = [seed.copy()]
    
    # Add random perturbations around seed
    for scale in [0.05, 0.1, 0.2, 0.5]:
        for _ in range(5):
            starts.append(seed + rng.normal(0.0, scale, size=3))
    
    # Add structured starts
    for c1_val in [0.01, 0.02, 0.05, 0.1, 0.2]:
        for c0_val in [-0.5, -1.0, -2.0, -3.0]:
            starts.append(np.array([c0_val, c1_val, 0.01]))
    
    for x0 in starts:
        if time.time() > deadline - 15.0:
            break
        try:
            refined = _local_refine(*x0, time_budget=3.0)
            _try_update(refined)
        except Exception:
            continue
    
    # Phase 3: Final polish on best solution
    if best_val < seed_val and time.time() < deadline - 5.0:
        try:
            final_refined = _local_refine(*best_c, time_budget=10.0)
            _try_update(final_refined)
        except Exception:
            pass
    
    if best_val >= seed_val:
        return np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    return np.array(best_c, dtype=float)
# EVOLVE-BLOCK-END
