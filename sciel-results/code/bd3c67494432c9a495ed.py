"""Optimization-based program: minimize r^2/(2*pi) using smooth surrogate objective and L-BFGS-B."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

# H0(0)=1, H4(0)=-12, H8(0)=1008, H12(0)=-103320
_H0_0 = 1.0
_H4_0 = -12.0
_H8_0 = 1008.0
_H12_0 = -103320.0


def _coeffs_to_q(c0, c1, c2):
    """Return (c3, q) where g=c0 H0+c1 H4+c2 H8+c3 H12, g(0)=0, q=g/x^2.
    q is a degree-10 polynomial in x with only even powers.
    Returns q as numpy array in descending order for np.polyval.
    """
    c3 = -(_H0_0 * c0 + _H4_0 * c1 + _H8_0 * c2) / _H12_0
    a10 = 4096.0 * c3
    a8 = 256.0 * c2 + 1198080.0 * c3
    a6 = -2304.0 * c2 - 4644960.0 * c3
    a4 = 16.0 * c1 + 4320.0 * c2 + 6652800.0 * c3
    a2 = -48.0 * c1 - 1512.0 * c2 - 2126112.0 * c3
    q = np.array([a10, 0.0, a8, 0.0, a6, 0.0, a4, 0.0, a2, 0.0, 0.0])
    return c3, q


def _find_r(q):
    """Largest real root of q at which q changes sign. Returns -1 if none found."""
    try:
        roots = np.roots(q)
    except Exception:
        return -1.0
    
    best = -1.0
    for r in roots:
        if abs(r.imag) > 1e-8:
            continue
        x = r.real
        if x <= 1e-8:
            continue
        # Adaptive epsilon for sign-change detection
        eps = max(1e-6, abs(x) * 1e-6)
        try:
            qm = np.polyval(q, x - eps)
            qp = np.polyval(q, x + eps)
        except Exception:
            continue
        if qm * qp < 0:
            if x > best:
                best = x
    return best


def _smooth_objective(c):
    """
    Smooth surrogate objective: track the largest real root of q(x) = 0.
    For a sign-changing root, q(x) changes sign, so the root is where q crosses zero.
    We use the fact that for the largest sign-changing root r, q(r)=0 and q'(r) != 0.
    
    Strategy: Find all real roots, identify sign-changing ones, and smoothly track
    the largest one using a soft-max approach over the roots.
    """
    c0, c1, c2 = c
    c3, q = _coeffs_to_q(c0, c1, c2)
    
    if q[0] < 0:
        q = -q
    
    # Find all roots
    try:
        roots = np.roots(q)
    except Exception:
        return 1e10
    
    real_roots = []
    for r in roots:
        if abs(r.imag) > 1e-6:
            continue
        x = r.real
        if x <= 1e-6:
            continue
        # Check sign change
        eps = max(1e-5, abs(x) * 1e-5)
        qm = np.polyval(q, x - eps)
        qp = np.polyval(q, x + eps)
        if qm * qp < 0:
            real_roots.append(x)
    
    if not real_roots:
        return 1e10
    
    # Use soft-max to smoothly select the largest root
    # r_max = logsumexp trick for max
    real_roots = np.array(real_roots)
    # Scale for numerical stability
    scale = np.max(real_roots)
    if scale <= 0:
        return 1e10
    
    # Soft max with temperature
    temp = 0.1
    shifted = real_roots / temp
    max_shifted = np.max(shifted)
    exp_shifted = np.exp(shifted - max_shifted)
    soft_max = temp * (max_shifted + np.log(np.sum(exp_shifted)))
    
    # The soft max approximates the max, but we want the actual max for the value
    # Use the actual max for the value computation
    r = np.max(real_roots)
    
    return (r * r) / (2.0 * np.pi)


def _objective(c):
    """Compute value = r^2/(2*pi), or a large penalty."""
    c0, c1, c2 = c
    c3, q = _coeffs_to_q(c0, c1, c2)
    if q[0] < 0:
        q = -q
    r = _find_r(q)
    if r < 0:
        return 1e10
    return (r * r) / (2.0 * np.pi)


def construct():
    seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    seed_val = _objective(seed)

    best_c = seed.copy()
    best_val = seed_val

    # Phase 1: L-BFGS-B optimization from seed
    try:
        res = minimize(_objective, best_c.copy(), method='L-BFGS-B',
                       options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-15})
        if res.fun < best_val:
            best_val = res.fun
            best_c = res.x.copy()
    except Exception:
        pass

    # Phase 2: Nelder-Mead for local refinement
    try:
        res = minimize(_objective, best_c.copy(), method='Nelder-Mead',
                       options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
        if res.fun < best_val:
            best_val = res.fun
            best_c = res.x.copy()
    except Exception:
        pass

    # Phase 3: Multiple random starts with local refinement
    rng = np.random.default_rng(42)
    for trial in range(50):
        scale = 0.5 + trial * 0.05
        start = seed + rng.normal(0, scale, size=3)
        v = _objective(start)
        if v < best_val:
            best_val = v
            best_c = start.copy()
        
        # Local refinement from this point
        try:
            res = minimize(_objective, best_c.copy(), method='Nelder-Mead',
                           options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_c = res.x.copy()
        except Exception:
            pass

    # Phase 4: Fine-tuning with very small perturbations
    for trial in range(100):
        perturbation = rng.normal(0, 1e-6 * (1 - trial/100.0), size=3)
        cand = best_c + perturbation
        v = _objective(cand)
        if v < best_val:
            best_val = v
            best_c = cand.copy()

    return np.array(best_c, dtype=float)
# EVOLVE-BLOCK-END
