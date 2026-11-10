"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _E_of_a(a, w0, wa):
    """H(a)/H0 for CPL model. a = 1/(1+z)."""
    Om_m = 0.3
    Om_La = 1.0 - Om_m
    a = np.asarray(a, dtype=float)
    term_m = Om_m * a**(-3.0)
    exponent_de = 3.0 * (1.0 + w0 + wa * (1.0 - a))
    term_de = Om_La * a**exponent_de
    E_sq = term_m + term_de
    return np.sqrt(np.maximum(E_sq, 1e-30))


def _true_integral_fast(z, w0, wa, n_pts=513):
    """Fast high-accuracy reference integral using Simpson's rule."""
    if z < 1e-15:
        return 0.0
    u = np.linspace(0.0, 1.0, n_pts)
    a = 1.0 / (1.0 + z * u)
    E_vals = _E_of_a(a, w0, wa)
    integrand = 1.0 / E_vals
    h = 1.0 / (n_pts - 1)
    # Simpson's rule: n_pts must be odd
    result = h / 3.0 * (integrand[0] + integrand[-1]
                        + 4.0 * np.sum(integrand[1:-1:2])
                        + 2.0 * np.sum(integrand[2:-1:2]))
    return result


def _quad_value(z, nodes, weights, w0, wa):
    """Compute quadrature approximation of dL/(c/H0)."""
    a = 1.0 / (1.0 + z * nodes)
    E_vals = _E_of_a(a, w0, wa)
    integral_approx = np.sum(weights / E_vals)
    return (1.0 + z) * z * integral_approx


def _decode_params(params):
    """Decode log-spacings and log-weights to nodes and weights."""
    s = params[:8]
    w = params[8:16]
    spacings = np.exp(s)
    total = np.sum(spacings)
    nodes = np.cumsum(spacings) / total
    w_max = np.max(w)
    exp_w = np.exp(w - w_max)
    weights = exp_w / np.sum(exp_w)
    return nodes, weights


def _is_valid(nodes, weights):
    """Check constraint validity."""
    if np.any(nodes <= 0) or np.any(nodes >= 1):
        return False
    if np.any(np.diff(nodes) <= 1e-12):
        return False
    if np.any(weights <= 0):
        return False
    return True


# Precomputed test set: diverse CPL cosmologies and redshifts
_COSMOLOGIES = [
    (-1.0, 0.0),    # Cosmological constant
    (-0.9, 0.0),    # Near CC
    (-0.5, 0.0),    # Quintessence
    (-0.5, 0.5),    # Quintessence with evolution
    (-0.8, 0.3),    # Another quintessence
    (-0.3, -0.3),   # Phantom-ish
    (0.0, -0.5),    # Phantom
    (0.0, 0.0),     # Matter-dominated-ish
    (0.0, 0.3),     # Matter-dominated with evolution
    (0.3, -0.6),    # Phantom-ish
    (0.5, 0.0),     # Matter-dominated
    (-1.0, 0.5),    # CC with evolution
    (-1.0, 1.0),    # Strong evolution
    (-1.2, 0.3),    # Phantom with evolution
    (-0.9, -0.4),   # Near -1 with negative evolution
    (0.5, -0.5),    # Strong phantom
    (-1.5, 0.0),    # Very negative w0
    (1.0, 0.0),     # Very positive w0
]

_Z_VALUES = np.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 10.0])


def _objective_minimax(params):
    """Worst-case relative error in percent across cosmologies and z values."""
    nodes, weights = _decode_params(params)
    if not _is_valid(nodes, weights):
        return 1e10
    
    max_err = 0.0
    for w0, wa in _COSMOLOGIES:
        for z in _Z_VALUES:
            true_val = _true_integral_fast(z, w0, wa)
            if true_val < 1e-15:
                continue
            quad_val = _quad_value(z, nodes, weights, w0, wa)
            true_val_full = (1.0 + z) * z * true_val
            if abs(true_val_full) < 1e-15:
                continue
            err = abs(quad_val - true_val_full) / abs(true_val_full) * 100.0
            if err > max_err:
                max_err = err
            if max_err > 200.0:
                return max_err
    return max_err


def _objective_smooth(params):
    """Smooth surrogate: sum of squared relative errors for faster optimization."""
    nodes, weights = _decode_params(params)
    if not _is_valid(nodes, weights):
        return 1e10
    
    sum_sq_err = 0.0
    for w0, wa in _COSMOLOGIES:
        for z in _Z_VALUES:
            true_val = _true_integral_fast(z, w0, wa)
            if true_val < 1e-15:
                continue
            quad_val = _quad_value(z, nodes, weights, w0, wa)
            true_val_full = (1.0 + z) * z * true_val
            if abs(true_val_full) < 1e-15:
                continue
            rel_err = (quad_val - true_val_full) / true_val_full
            sum_sq_err += rel_err**2
    return sum_sq_err


def construct():
    """
    Eight-node positive quadrature rule for dimensionless luminosity distance.
    Two-stage optimization: fast smooth surrogate (L-BFGS-B) then minimax refinement.
    """
    start_time = time.time()
    deadline = start_time + 82.0  # Safety margin before 90s limit
    
    # Start from Gauss-Legendre
    from scipy.special import roots_legendre
    x, w = roots_legendre(8)
    nodes_gl = 0.5 * (x + 1.0)
    weights_gl = 0.5 * w
    weights_gl = weights_gl / np.sum(weights_gl)
    
    # Encode GL as parameters
    spacings_gl = np.diff(np.concatenate([[0], nodes_gl]))
    s_init = np.log(spacings_gl)
    w_init = np.log(weights_gl)
    x0 = np.concatenate([s_init, w_init])
    
    best_params = x0.copy()
    best_score = _objective_minimax(x0)
    
    # Phase 1: L-BFGS-B on smooth surrogate
    if time.time() - start_time < deadline - 20:
        try:
            result = minimize(
                _objective_smooth, x0, method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if result.fun < _objective_smooth(x0):
                cand_score = _objective_minimax(result.x)
                if cand_score < best_score:
                    best_score = cand_score
                    best_params = result.x.copy()
        except Exception:
            pass
    
    # Phase 2: Nelder-Mead on minimax
    if time.time() - start_time < deadline - 15:
        try:
            result = minimize(
                _objective_minimax, best_params, method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True}
            )
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    # Phase 3: Random restarts
    rng = np.random.default_rng(12345)
    for i in range(5):
        if time.time() - start_time > deadline - 30:
            break
        x_rand = best_params + rng.normal(0, 0.08, size=16)
        try:
            res1 = minimize(
                _objective_smooth, x_rand, method='L-BFGS-B',
                options={'maxiter': 150, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            cand = res1.x
            if time.time() - start_time < deadline - 10:
                res2 = minimize(
                    _objective_minimax, cand, method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True}
                )
                if res2.fun < best_score:
                    best_score = res2.fun
                    best_params = res2.x.copy()
            else:
                if res1.fun < _objective_smooth(best_params):
                    cand_score = _objective_minimax(cand)
                    if cand_score < best_score:
                        best_score = cand_score
                        best_params = cand.copy()
        except Exception:
            pass
    
    # Phase 4: Final polish
    if time.time() - start_time < deadline - 5:
        try:
            result = minimize(
                _objective_minimax, best_params, method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    # Decode best parameters
    nodes, weights = _decode_params(best_params)
    
    # Final safety: if anything went wrong, fall back to GL
    if not (np.all(nodes > 0) and np.all(nodes < 1) and np.all(np.diff(nodes) > 0) and np.all(weights > 0)):
        nodes = nodes_gl.copy()
        weights = weights_gl.copy()
    
    # Normalize weights
    weights = weights / np.sum(weights)
    
    # Verify constraints
    assert len(nodes) == 8
    assert len(weights) == 8
    assert np.all(np.diff(nodes) > 0), "Nodes must be strictly increasing"
    assert np.all(nodes > 0) and np.all(nodes < 1), "Nodes must be in (0,1)"
    assert np.all(weights > 0), "Weights must be positive"
    assert abs(np.sum(weights) - 1.0) < 1e-12, "Weights must sum to 1"
    
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
