"""Uncertainty bound minimization via unit sphere search and local refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 195.0


def _compute_bound(a0, a4, a8):
    """
    Compute the uncertainty product sqrt(<x^2>/<1> * <xi^2>/<1>) for
    g(x) = P(x) * exp(-x^2/2) where P is a degree-12 Hermite combination
    with g(0) = 0 enforced via a12 = -(a0 + 3*a4 + 105*a8)/945.
    """
    a12 = -(a0 + 3.0 * a4 + 105.0 * a8) / 945.0

    # Polynomial coefficients p_k such that P(x) = sum_{k=0}^{6} p_k x^{2k}
    # H0 = 1
    # H4 = x^4 - 6x^2 + 3
    # H8 = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    # H12 = x^12 - 66x^10 + 1155x^8 - 6930x^6 + 13860x^4 - 7560x^2 + 945
    p0 = a0 + 3.0 * a4 + 105.0 * a8 + 945.0 * a12
    p1 = -6.0 * a4 - 420.0 * a8 - 7560.0 * a12
    p2 = a4 + 210.0 * a8 + 13860.0 * a12
    p3 = -28.0 * a8 - 6930.0 * a12
    p4 = a8 + 1155.0 * a12
    p5 = -66.0 * a12
    p6 = a12

    p = np.array([p0, p1, p2, p3, p4, p5, p6])

    # I[n] = integral x^{2n} exp(-x^2) dx = (2n-1)!! * sqrt(pi) / 2^n = (2n)! / (4^n n!) * sqrt(pi)
    # Precompute up to n=12 (need x^{24})
    sp = np.sqrt(np.pi)
    I = np.zeros(13)
    I[0] = sp
    for n in range(1, 13):
        # (2n)! / (4^n * n!) * sqrt(pi)
        # Recurrence: I[n] = I[n-1] * (2n-1) / 2
        I[n] = I[n - 1] * (2 * n - 1) / 2.0

    # m[k] = integral x^{2k} |P(x)|^2 exp(-x^2) dx for k=0,1,2
    # |P|^2 = sum_{i,j} p_i p_j x^{2(i+j)}
    # m[k] = sum_{i,j} p_i p_j I[k+i+j]
    m = np.zeros(3)
    for i in range(7):
        pi = p[i]
        for j in range(7):
            pp = pi * p[j]
            for k in range(3):
                idx = k + i + j
                if idx < 13:
                    m[k] += pp * I[idx]

    # Fourier: G(xi) = sqrt(2*pi) * P(-i*xi) * exp(-xi^2/2)
    # P(-i*xi) = sum p_k (-i)^{2k} xi^{2k} = sum p_k (-1)^k xi^{2k}
    q = p * np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])

    # n[k] = integral xi^{2k} |G(xi)|^2 dxi = 2*pi * sum_{i,j} q_i q_j I[k+i+j]
    n = np.zeros(3)
    for i in range(7):
        qi = q[i]
        for j in range(7):
            qq = qi * q[j]
            for k in range(3):
                idx = k + i + j
                if idx < 13:
                    n[k] += 2.0 * np.pi * qq * I[idx]

    m0, m2 = m[0], m[1]
    n0, n2 = n[0], n[1]

    if m0 <= 0 or n0 <= 0 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18

    vf = m2 / m0
    vF = n2 / n0

    if vf <= 0 or vF <= 0:
        return 1e18

    return np.sqrt(vf * vF)


def _bound_from_angles(t1, t2):
    """
    Compute bound for coefficients on the unit sphere.
    a0 = sin(t2) * cos(t1)
    a4 = sin(t2) * sin(t1)
    a8 = cos(t2)
    """
    st2 = np.sin(t2)
    a0 = st2 * np.cos(t1)
    a4 = st2 * np.sin(t1)
    a8 = np.cos(t2)
    return _compute_bound(a0, a4, a8)


def construct():
    """
    Find coefficients (a0, a4, a8) that minimize the uncertainty bound.
    Uses unit sphere reparameterization (scale invariance) and local refinement.
    """
    # Seed values for reference
    seed = np.array([-1.13, 0.04, 0.000308642])
    # Normalize seed to unit sphere
    seed_norm = np.linalg.norm(seed)
    if seed_norm > 1e-15:
        seed_unit = seed / seed_norm
    else:
        seed_unit = np.array([1.0, 0.0, 0.0])
    
    # Convert seed to angles
    t1_seed = np.arctan2(seed_unit[1], seed_unit[0])
    t2_seed = np.arccos(np.clip(seed_unit[2], -1.0, 1.0))
    
    best_bound = _compute_bound(*seed_unit)
    best_params = seed_unit.copy()
    
    # Phase 1: Grid search on the unit sphere
    # t1 in [0, 2*pi), t2 in [0, pi]
    n_t1 = 100
    n_t2 = 50
    t1_vals = np.linspace(0, 2 * np.pi, n_t1, endpoint=False)
    t2_vals = np.linspace(0, np.pi, n_t2)
    
    candidates = []
    for t2 in t2_vals:
        for t1 in t1_vals:
            if time.time() > _DEADLINE - 30:
                break
            b = _bound_from_angles(t1, t2)
            if np.isfinite(b):
                candidates.append((b, t1, t2))
    
    if not candidates:
        return seed_unit.copy()
        
    candidates.sort(key=lambda x: x[0])
    # Take top 10 candidates for local refinement
    top_candidates = candidates[:10]
    
    # Phase 2: Local refinement using Nelder-Mead
    for b_init, t1_init, t2_init in top_candidates:
        if time.time() > _DEADLINE - 10:
            break
            
        def objective(params):
            t1, t2 = params
            return _bound_from_angles(t1, t2)
        
        try:
            res = minimize(objective, [t1_init, t2_init], method='Nelder-Mead', 
                           options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 500, 'maxfev': 500})
            if res.success or res.fun < best_bound:
                t1_opt, t2_opt = res.x
                b_opt = res.fun
                if b_opt < best_bound:
                    best_bound = b_opt
                    st2 = np.sin(t2_opt)
                    best_params = np.array([
                        st2 * np.cos(t1_opt),
                        st2 * np.sin(t1_opt),
                        np.cos(t2_opt)
                    ])
        except Exception:
            pass
            
    # Verify the result
    final_bound = _compute_bound(*best_params)
    if not np.isfinite(final_bound) or final_bound > best_bound:
        # Fall back to seed
        return seed_unit.copy()

    return best_params
# EVOLVE-BLOCK-END
