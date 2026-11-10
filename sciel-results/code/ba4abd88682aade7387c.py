"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _E_of_a(a, w0, wa, Om_m=0.3):
    """H(a)/H0 for CPL model. a = 1/(1+z)."""
    Om_m = 0.3
    Om_La = 1.0 - Om_m
    a = np.asarray(a, dtype=float)
    # E^2(a) = Om_m * a^(-3) + Om_La * a^(3*(1+w0+wa*(1-a)))
    term_m = Om_m * a**(-3.0)
    exponent_de = 3.0 * (1.0 + w0 + wa * (1.0 - a))
    term_de = Om_La * a**exponent_de
    E_sq = term_m + term_de
    return np.sqrt(np.maximum(E_sq, 1e-30))


def _integrand_u(u, z, w0, wa):
    """1/E(a(u)) where a = 1/(1+z*u), u in [0,1]."""
    a = 1.0 / (1.0 + z * u)
    return 1.0 / _E_of_a(a, w0, wa)


def _true_integral_dense(z, w0, wa, n_pts=20001):
    """High-accuracy reference integral using dense trapezoidal rule."""
    if z < 1e-15:
        return 0.0
    u = np.linspace(0.0, 1.0, n_pts)
    a = 1.0 / (1.0 + z * u)
    E_vals = _E_of_a(a, w0, wa)
    integrand = 1.0 / E_vals
    # Trapz
    return np.trapz(integrand, u)


def _rel_error_pct(z, nodes, weights, w0, wa, true_val):
    """Relative error in percent."""
    a = 1.0 / (1.0 + z * nodes)
    E_vals = _E_of_a(a, w0, wa)
    integral_approx = np.sum(weights / E_vals)
    quad_val = (1.0 + z) * z * integral_approx
    true_val_full = (1.0 + z) * z * true_val
    if abs(true_val_full) < 1e-15:
        return 0.0
    return abs(quad_val - true_val_full) / abs(true_val_full) * 100.0


def _objective_minimax(params):
    """Worst-case relative error across cosmologies and z values."""
    s = params[:8]
    w = params[8:16]
    
    spacings = np.exp(s)
    total = np.sum(spacings)
    nodes = np.cumsum(spacings) / total
    
    w_max = np.max(w)
    exp_w = np.exp(w - w_max)
    weights = exp_w / np.sum(exp_w)
    
    if np.any(nodes <= 0) or np.any(nodes >= 1):
        return 1e10
    if np.any(np.diff(nodes) <= 1e-15):
        return 1e10
    if np.any(weights <= 0):
        return 1e10
    
    # Diverse CPL cosmologies
    cosmologies = [
        (0.0, 0.0),      # LCDM
        (-1.0, 0.0),     # Cosmological constant
        (-0.5, 0.5),     # Quintessence
        (0.3, -0.6),     # Phantom-ish
        (-0.8, 0.3),     # Another quintessence
        (0.0, -0.5),     # Phantom
        (-0.5, 0.0),     # Simple quintessence
        (-1.0, 0.5),     # CC with evolution
        (0.0, 0.3),      # Matter-dominated-ish
        (-0.3, -0.3),    # Another phantom
        (-1.0, 1.0),     # Strong evolution
        (0.5, 0.0),      # Matter-dominated
        (-1.2, 0.3),     # Phantom with evolution
        (-0.9, -0.4),    # Near -1 with negative evolution
    ]
    
    # Range of redshifts to test
    z_values = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0])
    
    max_err = 0.0
    for w0, wa in cosmologies:
        for z in z_values:
            true_val = _true_integral_dense(z, w0, wa)
            err = _rel_error_pct(z, nodes, weights, w0, wa, true_val)
            if err > max_err:
                max_err = err
            if max_err > 100.0:
                return max_err
    return max_err


def _objective_smooth(params):
    """Smooth surrogate: sum of squared log-errors for faster optimization."""
    s = params[:8]
    w = params[8:16]
    
    spacings = np.exp(s)
    total = np.sum(spacings)
    nodes = np.cumsum(spacings) / total
    
    w_max = np.max(w)
    exp_w = np.exp(w - w_max)
    weights = exp_w / np.sum(exp_w)
    
    if np.any(nodes <= 0) or np.any(nodes >= 1):
        return 1e10
    if np.any(np.diff(nodes) <= 1e-15):
        return 1e10
    if np.any(weights <= 0):
        return 1e10
    
    # Diverse CPL cosmologies
    cosmologies = [
        (0.0, 0.0),
        (-1.0, 0.0),
        (-0.5, 0.5),
        (0.3, -0.6),
        (-0.8, 0.3),
        (0.0, -0.5),
        (-0.5, 0.0),
        (-1.0, 0.5),
        (0.0, 0.3),
        (-0.3, -0.3),
        (-1.0, 1.0),
        (0.5, 0.0),
    ]
    
    z_values = np.array([0.1, 0.3, 0.5, 1.0, 2.0, 3.0])
    
    sum_sq_err = 0.0
    for w0, wa in cosmologies:
        for z in z_values:
            true_val = _true_integral_dense(z, w0, wa)
            if true_val < 1e-15:
                continue
            a = 1.0 / (1.0 + z * nodes)
            E_vals = _E_of_a(a, w0, wa)
            integral_approx = np.sum(weights / E_vals)
            quad_val = (1.0 + z) * z * integral_approx
            true_val_full = (1.0 + z) * z * true_val
            if abs(true_val_full) < 1e-15:
                continue
            rel_err = abs(quad_val - true_val_full) / abs(true_val_full)
            sum_sq_err += rel_err**2
    return sum_sq_err


def construct():
    """
    Eight-node positive quadrature rule for dimensionless luminosity distance.
    Optimizes nodes and weights to minimize worst-case relative error across
    diverse CPL cosmologies.
    """
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
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
    
    elapsed = time.time() - start_time
    if elapsed < deadline - 20:
        # Phase 1: Smooth surrogate with L-BFGS-B for fast convergence
        try:
            result = minimize(
                _objective_smooth, x0, method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if result.fun < _objective_smooth(x0):
                candidate = result.x
                cand_score = _objective_minimax(candidate)
                if cand_score < best_score:
                    best_score = cand_score
                    best_params = candidate.copy()
        except Exception:
            pass
        
        # Phase 2: Refine with Nelder-Mead on minimax objective
        elapsed = time.time() - start_time
        if elapsed < deadline - 15:
            try:
                result = minimize(
                    _objective_minimax, best_params, method='Nelder-Mead',
                    options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
                )
                if result.fun < best_score:
                    best_score = result.fun
                    best_params = result.x.copy()
            except Exception:
                pass
        
        # Phase 3: A few random restarts if time allows
        elapsed = time.time() - start_time
        rng = np.random.default_rng(42)
        n_restarts = 3
        for i in range(n_restarts):
            if time.time() - start_time > deadline - 25:
                break
            x_rand = best_params + rng.normal(0, 0.05, size=16)
            try:
                # First smooth out
                res1 = minimize(
                    _objective_smooth, x_rand, method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                cand = res1.x
                # Then refine minimax
                res2 = minimize(
                    _objective_minimax, cand, method='Nelder-Mead',
                    options={'maxiter': 150, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True}
                )
                if res2.fun < best_score:
                    best_score = res2.fun
                    best_params = res2.x.copy()
            except Exception:
                pass
    
    # Decode best parameters
    s = best_params[:8]
    w = best_params[8:16]
    spacings = np.exp(s)
    total = np.sum(spacings)
    nodes = np.cumsum(spacings) / total
    w_max = np.max(w)
    exp_w = np.exp(w - w_max)
    weights = exp_w / np.sum(exp_w)
    
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
