"""Refined Cohn-Goncalves Laguerre construction: direct coefficient optimization."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.optimize import minimize, Bounds
from scipy.special import eval_genlaguerre, gammaln

_DEADLINE = 600.0
_START = time.time()
_MARGIN = 30.0


def _time_left():
    return _DEADLINE - (time.time() - _START) - _MARGIN


def _laguerre_coeffs(m, alpha=1):
    """Return coefficients of L_m^(alpha)(x) in ascending power order using recurrence."""
    if m == 0:
        return np.array([1.0])
    if m == 1:
        return np.array([alpha + 1, -1.0])
    
    L0 = np.array([1.0])
    L1 = np.array([alpha + 1, -1.0])
    
    for k in range(1, m):
        # (k+1) L_{k+1} = (2k+1+alpha-x) L_k - k L_{k-1}
        # term1: (2k+1+alpha) * L_k
        term1 = (2 * k + 1 + alpha) * L1
        # term2: -x * L_k  (shift L1 by 1)
        term2 = -np.concatenate([np.zeros(1), L1])
        # term3: -k * L_{k-1}
        term3 = -k * L0
        
        max_len = max(len(term1), len(term2), len(term3))
        t1 = np.zeros(max_len)
        t2 = np.zeros(max_len)
        t3 = np.zeros(max_len)
        t1[:len(term1)] = term1
        t2[:len(term2)] = term2
        t3[:len(term3)] = term3
        
        L2 = (t1 + t2 + t3) / (k + 1)
        L0, L1 = L1, L2
    
    return L1


def _build_psi_coeff(roots, m, alpha=1):
    """Build coefficients of Psi(x) = prod (x - r_i)^2 * L_m^(alpha)(x), ascending order."""
    cL = _laguerre_coeffs(m, alpha)
    poly = cL.copy()
    for r in roots:
        q = np.array([r * r, -2.0 * r, 1.0])
        poly = np.convolve(poly, q)
    return poly


def _compute_norm_sq(poly, alpha=1):
    """Compute integral_0^inf psi(x)^2 x^alpha e^{-x} dx using moment formula."""
    deg = len(poly) - 1
    sq = np.convolve(poly, poly)
    ks = np.arange(len(sq))
    # Gamma(k + alpha + 1) for each k
    log_facts = gammaln(ks + alpha + 1)
    facts = np.exp(log_facts)
    val = np.dot(sq, facts)
    return val


def _c4_bound(roots, m, alpha=1):
    """Upper bound on C4: 1 / integral psi^2 x^alpha e^{-x}."""
    if _time_left() < 5:
        return float('inf')
    poly = _build_psi_coeff(roots, m, alpha)
    if not np.all(np.isfinite(poly)):
        return float('inf')
    norm_sq = _compute_norm_sq(poly, alpha)
    if not np.isfinite(norm_sq) or norm_sq <= 0:
        return float('inf')
    return 1.0 / norm_sq


def _neg_log_norm(roots, m, alpha=1):
    """Negative log of norm squared (for minimization)."""
    poly = _build_psi_coeff(roots, m, alpha)
    if not np.all(np.isfinite(poly)):
        return 1e300
    norm_sq = _compute_norm_sq(poly, alpha)
    if not np.isfinite(norm_sq) or norm_sq <= 0:
        return 1e300
    return -math.log(norm_sq)


def _nelder_mead_minimize(func, x0, bounds_low, bounds_high, max_iter=300):
    """Simple Nelder-Mead to minimize func. Returns best x and value."""
    n = len(x0)
    simplex = [np.array(x0, dtype=float)]
    for i in range(n):
        x = np.array(x0, dtype=float)
        step = 0.1 * (bounds_high[i] - bounds_low[i]) if bounds_high[i] > bounds_low[i] else 1.0
        x[i] += step
        x = np.clip(x, bounds_low, bounds_high)
        simplex.append(x)
    
    func_vals = [func(x) for x in simplex]
    best_x = x0[:]
    best_val = func(best_x)
    
    for it in range(max_iter):
        if _time_left() < 30:
            break
        order = np.argsort(func_vals)
        simplex = [simplex[i] for i in order]
        func_vals = [func_vals[i] for i in order]
        
        if func_vals[0] < best_val:
            best_val = func_vals[0]
            best_x = simplex[0].copy()
        
        centroid = np.mean(simplex[:-1], axis=0)
        
        xr = centroid + (centroid - simplex[-1])
        xr = np.clip(xr, bounds_low, bounds_high)
        fr = func(xr)
        
        if fr <= func_vals[0]:
            xe = centroid + 2.0 * (centroid - simplex[-1])
            xe = np.clip(xe, bounds_low, bounds_high)
            fe = func(xe)
            if fe < fr:
                simplex[-1] = xe
                func_vals[-1] = fe
            else:
                simplex[-1] = xr
                func_vals[-1] = fr
        elif fr < func_vals[-2]:
            simplex[-1] = xr
            func_vals[-1] = fr
        else:
            xc = centroid + 0.5 * (simplex[-1] - centroid)
            xc = np.clip(xc, bounds_low, bounds_high)
            fc = func(xc)
            if fc < func_vals[-1]:
                simplex[-1] = xc
                func_vals[-1] = fc
            else:
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    simplex[i] = np.clip(simplex[i], bounds_low, bounds_high)
                    func_vals[i] = func(simplex[i])
    
    return best_x, best_val


def construct():
    m = 12
    alpha = 1  # Laguerre parameter for n=1 case

    # Initial guess: linearly spaced
    best_roots = np.array([4.0 * (i + 1) for i in range(m)], dtype=float)
    best_val = _c4_bound(best_roots, m, alpha)
    if not np.isfinite(best_val):
        best_val = float('inf')

    # Multi-start optimization using scipy minimize
    # Parameterize roots as log(roots) to ensure positivity
    starts = []

    # Start 1: linear spacing
    r1 = np.linspace(1.0, 20.0, m)
    starts.append(np.log(r1))

    # Start 2: quadratic spacing
    r2 = np.linspace(1.0, 20.0, m) ** 2
    starts.append(np.log(r2))

    # Start 3: current best
    starts.append(np.log(best_roots))

    # Start 4: geometric spacing
    r4 = 1.0 * (1.5 ** np.arange(m))
    starts.append(np.log(r4))

    # Start 5: clustered near origin
    r5 = np.linspace(0.5, 5.0, m)
    starts.append(np.log(r5))

    # Start 6: wide spread
    r6 = np.linspace(0.1, 50.0, m)
    starts.append(np.log(r6))

    # Start 7: based on Laguerre node approximation
    r7 = np.array([i + 1.0 for i in range(m)]) * 1.5
    starts.append(np.log(r7))

    bounds = Bounds(np.full(m, -5.0), np.full(m, 5.0))  # roots in [e^-5, e^5] ~ [0.0067, 148]

    # Ensure distinctness via penalty in objective
    def objective(params):
        roots = np.exp(params)
        # Penalty for non-distinct roots
        diffs = np.diff(roots)
        min_diff = np.min(np.abs(diffs))
        if min_diff < 1e-6:
            return 1e300
        val = _neg_log_norm(roots, m, alpha)
        # Add small penalty for roots too close
        penalty = -0.5 * math.log(min_diff)
        return val + penalty

    best_val = float('inf')
    best_params = None

    for start in starts:
        if _time_left() < 30:
            break
        try:
            result = minimize(
                objective,
                start,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if result.success or result.fun < best_val:
                best_val = result.fun
                best_params = result.x
        except Exception:
            pass

    if best_params is not None:
        best_roots = np.exp(best_params)

    # Local refinement with Nelder-Mead on the best solution
    if _time_left() > 60:
        try:
            def obj_nm(roots):
                if np.any(roots <= 0):
                    return 1e300
                diffs = np.diff(roots)
                min_diff = np.min(np.abs(diffs))
                if min_diff < 1e-6:
                    return 1e300
                val = _neg_log_norm(roots, m, alpha)
                return val

            initial = best_roots.copy()
            result = minimize(
                obj_nm,
                initial,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            if result.fun < best_val:
                best_roots = result.x
                best_val = result.fun
        except Exception:
            pass

    # Verify final roots are positive and distinct
    if np.any(best_roots <= 0) or len(set(np.round(best_roots, 10))) != m:
        # Fallback to simple linear spacing
        best_roots = np.linspace(1.0, 20.0, m)

    return list(best_roots)
# EVOLVE-BLOCK-END
