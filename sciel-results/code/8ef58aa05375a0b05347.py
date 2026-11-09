"""QAOA parameter design: per-graph optimization of angles maximizing expected cut ratio.

Strategy:
  1. Fast vectorized QAOA simulation using precomputed cut sizes.
  2. Analytic gradient computation via the parameter-shift rule.
  3. Multi-restart optimization:
     a. Coarse screening with Nelder-Mead to find promising basins.
     b. High-precision refinement using L-BFGS-B with analytic gradients.
  4. Strict deadline guard to ensure termination.
"""
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = None
_TIME_LIMIT = 140.0  # Safety margin against 150s limit


def _time_left():
    if _DEADLINE is None:
        return np.inf
    return _DEADLINE - time.time()


def _precompute_cut_sizes(n, edges):
    """Precompute the cut size for all 2^n states using vectorized operations."""
    dim = 1 << n
    cut_sizes = np.zeros(dim, dtype=np.float64)
    states = np.arange(dim, dtype=np.int64)
    
    for (i, j) in edges:
        bi = (states >> i) & 1
        bj = (states >> j) & 1
        cut_sizes += (bi ^ bj).astype(np.float64)
        
    return cut_sizes


def _apply_x_layer(state, beta, n):
    """
    Apply exp(-i beta sum X) to state vector.
    Optimized bit-pair mixer.
    """
    c = np.cos(beta)
    s = np.sin(beta)
    dim = len(state)
    
    # Reshape to tensor for easy axis manipulation
    tensor = state.reshape([2] * n)
    
    for i in range(n):
        # Move axis i to the front
        moved = np.moveaxis(tensor, i, 0)
        # Reshape to (2, -1) to operate on pairs
        flat = moved.reshape(2, -1)
        
        # Apply the 2x2 rotation matrix for each pair
        # [[c, -is], [-is, c]]
        new_flat = np.empty_like(flat)
        new_flat[0] = c * flat[0] - 1j * s * flat[1]
        new_flat[1] = -1j * s * flat[0] + c * flat[1]
        
        # Reshape back and move axis back
        tensor = np.moveaxis(new_flat.reshape([2] * n), 0, i)
    
    return tensor.flatten()


def _expected_cut(n, cut_sizes, angles):
    """Compute expected cut value for given angles."""
    dim = 1 << n
    state = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    for gamma, beta in angles:
        # Apply cost operator: exp(-i gamma C)
        # Since C is diagonal, this is just element-wise multiplication
        state *= np.exp(-1j * gamma * cut_sizes)
        
        # Apply mixer: exp(-i beta sum X)
        state = _apply_x_layer(state, beta, n)
    
    probs = np.abs(state) ** 2
    return float(np.sum(probs * cut_sizes))


def _compute_gradient(n, cut_sizes, angles):
    """
    Compute the gradient of the expected cut with respect to all angles.
    Uses the parameter-shift rule, which is exact for QAOA.
    """
    p = len(angles)
    grad = np.zeros(2 * p)
    
    for k in range(p):
        # Gradient with respect to gamma_k
        angles_plus = angles.copy()
        angles_minus = angles.copy()
        angles_plus[k, 0] += np.pi / 2
        angles_minus[k, 0] -= np.pi / 2
        
        f_plus = _expected_cut(n, cut_sizes, angles_plus)
        f_minus = _expected_cut(n, cut_sizes, angles_minus)
        grad[2 * k] = (f_plus - f_minus) / 2.0
        
        # Gradient with respect to beta_k
        angles_plus = angles.copy()
        angles_minus = angles.copy()
        angles_plus[k, 1] += np.pi / 2
        angles_minus[k, 1] -= np.pi / 2
        
        f_plus = _expected_cut(n, cut_sizes, angles_plus)
        f_minus = _expected_cut(n, cut_sizes, angles_minus)
        grad[2 * k + 1] = (f_plus - f_minus) / 2.0
        
    return grad


def _objective_and_grad(params, n, cut_sizes, p):
    """Objective function and gradient for L-BFGS-B (minimization)."""
    if _time_left() < 2.0:
        raise TimeoutError()
    
    angles = params.reshape(p, 2)
    val = _expected_cut(n, cut_sizes, angles)
    grad = _compute_gradient(n, cut_sizes, angles)
    
    # We want to maximize, so minimize negative value
    return -val, -grad


def _initial_guesses_from_structure(n, edges, p):
    """Generate structure-informed initial angle guesses."""
    m = len(edges)
    density = m / (n * (n - 1) / 2) if n > 1 else 0.0
    
    degree = np.zeros(n, dtype=np.int32)
    for (i, j) in edges:
        degree[i] += 1
        degree[j] += 1
    max_deg = int(np.max(degree)) if n > 0 else 0
    avg_deg = 2.0 * m / n if n > 0 else 0.0
    
    guesses = []
    
    # Heuristic 1: based on edge density
    base_gamma = 0.15 + 0.3 * density
    base_beta = 0.2 + 0.15 * (1.0 - density)
    guesses.append(np.tile([base_gamma, base_beta], p))
    
    # Heuristic 2: degree-scaled
    gamma2 = 0.1 + 0.05 * avg_deg
    beta2 = 0.25 - 0.02 * avg_deg
    guesses.append(np.tile([max(0.05, gamma2), max(0.05, beta2)], p))
    
    # Heuristic 3: known good QAOA starting points
    guesses.append(np.array([0.35, 0.25] * p))
    guesses.append(np.array([0.4, 0.2] * p))
    guesses.append(np.array([0.25, 0.3] * p))
    
    # Heuristic 4: small angles
    guesses.append(np.tile([0.1, 0.1], p))
    
    # Heuristic 5: moderate angles
    guesses.append(np.tile([0.5, 0.35], p))
    
    # Random guesses with fixed seed
    rng = np.random.RandomState(42)
    for _ in range(5):
        gammas = rng.uniform(0.1, 0.8, p)
        betas = rng.uniform(0.05, 0.5, p)
        guesses.append(np.column_stack((gammas, betas)).flatten())
        
    return guesses


def _nelder_mead_screen(x0, n, cut_sizes, p, maxiter=100):
    """Fast Nelder-Mead screening."""
    def _obj_wrapper(params):
        if _time_left() < 5.0:
            raise TimeoutError()
        angles = params.reshape(p, 2)
        return -_expected_cut(n, cut_sizes, angles)
    
    try:
        res = minimize(
            _obj_wrapper,
            x0,
            method='Nelder-Mead',
            options={'maxiter': maxiter, 'xatol': 1e-6, 'fatol': 1e-6}
        )
        return res.x, -res.fun
    except TimeoutError:
        return x0, _expected_cut(n, cut_sizes, x0.reshape(p, 2))
    except Exception:
        return x0, _expected_cut(n, cut_sizes, x0.reshape(p, 2))


def _l_bfgs_refine(x0, n, cut_sizes, p, maxiter=50):
    """High-precision L-BFGS-B refinement with analytic gradients."""
    def _obj_grad_wrapper(params):
        if _time_left() < 5.0:
            raise TimeoutError()
        return _objective_and_grad(params, n, cut_sizes, p)
    
    try:
        res = minimize(
            _obj_grad_wrapper,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': maxiter, 'ftol': 1e-10, 'gtol': 1e-6}
        )
        return res.x, -res.fun
    except TimeoutError:
        return x0, _expected_cut(n, cut_sizes, x0.reshape(p, 2))
    except Exception:
        return x0, _expected_cut(n, cut_sizes, x0.reshape(p, 2))


def angles(n, edges, p):
    """Design graph-dependent two-layer QAOA angles."""
    global _DEADLINE
    _DEADLINE = time.time() + _TIME_LIMIT
    
    # Precompute cut sizes
    cut_sizes = _precompute_cut_sizes(n, edges)
    
    # Generate structure-informed initial guesses
    initial_guesses = _initial_guesses_from_structure(n, edges, p)
    
    best_angles = None
    best_score = -np.inf
    
    # Phase 1: Multi-restart Nelder-Mead Screening
    # Use a limited number of iterations to quickly find good basins
    for idx, x0 in enumerate(initial_guesses):
        if _time_left() < 15.0:
            break
        
        x_opt, score = _nelder_mead_screen(x0, n, cut_sizes, p, maxiter=100)
        
        if score > best_score:
            best_score = score
            best_angles = x_opt.reshape(p, 2)
            
    # Phase 2: High-precision L-BFGS-B Refinement on the best candidate
    if best_angles is not None and _time_left() > 20.0:
        x_refined, score_refined = _l_bfgs_refine(
            best_angles.flatten(), n, cut_sizes, p, maxiter=50
        )
        if score_refined > best_score:
            best_score = score_refined
            best_angles = x_refined.reshape(p, 2)
            
    # Phase 3: Final Polish with Nelder-Mead on the refined result
    if best_angles is not None and _time_left() > 10.0:
        x_polished, score_polished = _nelder_mead_screen(
            best_angles.flatten(), n, cut_sizes, p, maxiter=200
        )
        if score_polished > best_score:
            best_score = score_polished
            best_angles = x_polished.reshape(p, 2)
            
    # Fallback
    if best_angles is None:
        best_angles = np.tile([0.35, 0.25], (p, 1))
        
    return best_angles
