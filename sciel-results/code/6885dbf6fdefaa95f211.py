"""Refined Cohn-Goncalves Laguerre construction: direct coefficient optimization."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint
from scipy.special import eval_genlaguerre, factorial

_DEADLINE = 600.0
_START = time.time()


def _time_left():
    return _DEADLINE - time.time()


def _laguerre_coeffs(m, alpha=1):
    """Return coefficients of L_m^(alpha)(x) in ascending power order."""
    try:
        N = 4 * m + 60
        xg = np.linspace(1e-10, 6 * m + 40, N)
        Lvals = eval_genlaguerre(m, alpha, xg)
        cL = np.polyfit(xg, Lvals, m)
        return cL[::-1]
    except Exception:
        # Fallback: build via recurrence
        if m == 0:
            return np.array([1.0])
        if m == 1:
            return np.array([alpha + 1 - 0 * 1, -1.0])  # L_1 = -x + alpha+1
        L0 = np.array([1.0])
        L1 = np.array([alpha + 1, -1.0])
        for k in range(1, m):
            # (k+1) L_{k+1} = (2k+1+alpha-x) L_k - k L_{k-1}
            # (2k+1+alpha) L_k - x L_k - k L_{k-1}
            term1 = (2 * k + 1 + alpha) * L1
            term1 = np.concatenate([term1, np.zeros(1)])  # multiply by 1
            # -x * L1: shift L1 by 1
            term2 = -np.concatenate([np.zeros(1), L1])
            term3 = -k * L0
            # Align lengths
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
    # psi^2 = sum_{i,j} c_i c_j x^{i+j}
    # integral x^{i+j} x^alpha e^{-x} dx = Gamma(i+j+alpha+1)
    deg = len(poly) - 1
    sq = np.convolve(poly, poly)
    ks = np.arange(len(sq))
    # Gamma(k + alpha + 1) for each k
    # Use log-gamma for stability
    from scipy.special import gammaln
    log_facts = gammaln(ks + alpha + 1)
    facts = np.exp(log_facts)
    val = np.dot(sq, facts)
    return val


def _c4_bound(roots, m, alpha=1):
    """Upper bound on C4: 1 / integral psi^2 x^alpha e^{-x}."""
    if _time_left() < 5:
        return float('inf')
    poly = _build_psi_coeff(roots, m, alpha)
    norm_sq = _compute_norm_sq(poly, alpha)
    if not np.isfinite(norm_sq) or norm_sq <= 0:
        return float('inf')
    return 1.0 / norm_sq


def _neg_log_norm(roots, m, alpha=1):
    """Negative log of norm squared (for minimization)."""
    poly = _build_psi_coeff(roots, m, alpha)
    norm_sq = _compute_norm_sq(poly, alpha)
    if not np.isfinite(norm_sq) or norm_sq <= 0:
        return 1e300
    return -math.log(norm_sq)


def _make_roots_from_params(params, m=12):
    """Convert optimization parameters to positive distinct roots."""
    # params: log-spaced positions
    logs = params
    roots = np.exp(logs)
    return roots


def construct():
    m = 12
    alpha = 1  # Laguerre parameter for n=1 case

    # Initial guess: evenly spaced in log space
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
    # Nodes of L_m^(1) are roughly at i + (alpha+1)/2 for large m
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
