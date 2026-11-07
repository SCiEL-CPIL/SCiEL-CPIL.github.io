"""Optimization-based program: minimize r^2/(2*pi) using differentiable objective and scipy.optimize.minimize."""
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


def _q_derivative(q):
    """Return derivative of polynomial q (descending order)."""
    n = len(q) - 1
    dq = np.zeros(n)
    for i in range(n):
        dq[i] = q[i] * (n - i)
    return dq


def _find_r_robust(q):
    """More robust root finding: use both q roots and q' roots, verify sign change."""
    best = -1.0
    
    # Get roots of q
    try:
        roots_q = np.roots(q)
    except Exception:
        roots_q = np.array([])
    
    # Get roots of q'
    try:
        dq = _q_derivative(q)
        roots_dq = np.roots(dq)
    except Exception:
        roots_dq = np.array([])
    
    # Check all candidate roots
    candidates = []
    for r in np.concatenate([roots_q, roots_dq]):
        if abs(r.imag) > 1e-7:
            continue
        x = r.real
        if x <= 1e-6:
            continue
        candidates.append(x)
    
    for x in candidates:
        # Verify sign change with multiple epsilons
        for eps in [1e-6, 1e-5, 1e-4]:
            try:
                qm = np.polyval(q, x - eps)
                qp = np.polyval(q, x + eps)
            except Exception:
                continue
            if qm * qp < 0:
                if x > best:
                    best = x
                break
    
    return best


def _objective(c):
    """Compute value = r^2/(2*pi), or a large penalty."""
    c0, c1, c2 = c
    c3, q = _coeffs_to_q(c0, c1, c2)
    if q[0] < 0:
        q = -q
    r = _find_r_robust(q)
    if r < 0:
        return 1e10
    return (r * r) / (2.0 * np.pi)


def construct():
    seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    seed_val = _objective(seed)

    best_c = seed.copy()
    best_val = seed_val

    # Phase 1: Local optimization from seed with multiple methods
    methods_opts = [
        ('Nelder-Mead', {'maxiter': 8000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}),
        ('Powell', {'maxiter': 5000, 'xtol': 1e-14, 'ftol': 1e-16}),
        ('COBYLA', {'maxiter': 10000, 'rhobeg': 1e-3}),
    ]

    for method, opts in methods_opts:
        try:
            res = minimize(_objective, best_c.copy(), method=method, options=opts)
            if res.fun < best_val:
                best_val = res.fun
                best_c = res.x.copy()
        except Exception:
            pass

    # Phase 2: Try multiple random starting points for global exploration
    rng = np.random.default_rng(42)
    n_trials = 60
    for trial in range(n_trials):
        # Multi-scale random initialization
        if trial < 20:
            scale = 0.5
        elif trial < 40:
            scale = 1.0
        else:
            scale = 2.0
        start = seed + rng.normal(0, scale, size=3)
        v = _objective(start)
        if v < best_val:
            best_val = v
            best_c = start.copy()
        
        # Local refinement from this point
        try:
            res = minimize(_objective, best_c.copy(), method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_c = res.x.copy()
        except Exception:
            pass

    # Phase 3: Fine-tuning around the best point found
    for trial in range(100):
        # Decreasing perturbation scale
        scale = 1e-4 * (1 - trial / 100.0) + 1e-6
        perturbation = rng.normal(0, scale, size=3)
        cand = best_c + perturbation
        v = _objective(cand)
        if v < best_val:
            best_val = v
            best_c = cand.copy()
            try:
                res = minimize(_objective, best_c.copy(), method='Nelder-Mead',
                               options={'maxiter': 8000, 'xatol': 1e-15, 'fatol': 1e-17, 'adaptive': True})
                if res.fun < best_val:
                    best_val = res.fun
                    best_c = res.x.copy()
            except Exception:
                pass

    return np.array(best_c, dtype=float)
# EVOLVE-BLOCK-END
