"""Combined QAOA parameter design: coarse-to-fine + multi-restart Nelder-Mead + gradient polish.

Strategy:
  1. Fast vectorized QAOA simulation (bit-pair mixer, precomputed cut sizes).
  2. Coarse grid search for global landscape exploration.
  3. Multi-restart Nelder-Mead from diverse initial points (including grid best).
  4. Analytic gradient refinement (parameter-shift rule) on the best candidate.
  5. High-precision Nelder-Mead polish pass with very tight tolerance.
  6. Strict deadline guard.
"""
# EVOLVE-BLOCK-START
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
    Vectorized bit-pair mixer.
    """
    c = np.cos(beta)
    s = np.sin(beta)
    
    tensor = state.reshape([2] * n)
    
    for i in range(n):
        moved = np.moveaxis(tensor, i, 0)
        flat = moved.reshape(2, -1)
        new_flat = np.empty_like(flat)
        new_flat[0] = c * flat[0] - 1j * s * flat[1]
        new_flat[1] = -1j * s * flat[0] + c * flat[1]
        tensor = np.moveaxis(new_flat.reshape([2] * n), 0, i)
    
    return tensor.flatten()


def _expected_cut(n, cut_sizes, angles):
    """Compute expected cut value for given angles."""
    dim = 1 << n
    state = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    for gamma, beta in angles:
        state *= np.exp(-1j * gamma * cut_sizes)
        state = _apply_x_layer(state, beta, n)
    
    probs = np.abs(state) ** 2
    return float(np.sum(probs * cut_sizes))


def _objective(params, n, cut_sizes, p):
    """Objective function for minimization (negative expected cut)."""
    angles = params.reshape(p, 2)
    return -_expected_cut(n, cut_sizes, angles)


def _nelder_mead(x0, n, cut_sizes, p, maxiter=300, tol=1e-10):
    """Run Nelder-Mead optimization with time guard."""
    def _obj_wrapper(params):
        if _time_left() < 5.0:
            raise TimeoutError()
        return _objective(params, n, cut_sizes, p)
    
    try:
        res = minimize(
            _obj_wrapper,
            x0,
            args=(),
            method='Nelder-Mead',
            options={
                'maxiter': maxiter,
                'xatol': tol,
                'fatol': tol,
                'adaptive': True
            }
        )
        return res.x, -res.fun
    except TimeoutError:
        return x0, _expected_cut(n, cut_sizes, x0.reshape(p, 2))
    except Exception:
        return x0, _expected_cut(n, cut_sizes, x0.reshape(p, 2))


def _gradient_refinement(x0, n, cut_sizes, p, max_steps=30, lr=0.02):
    """
    Gradient ascent using parameter-shift rule for exact gradients with momentum.
    This is a smooth refinement step after Nelder-Mead.
    """
    x = x0.copy()
    best_x = x.copy()
    best_val = _expected_cut(n, cut_sizes, x.reshape(p, 2))
    velocity = np.zeros_like(x)
    momentum = 0.9
    epsilon = 1e-8
    
    for step in range(max_steps):
        if _time_left() < 10.0:
            break
        
        # Compute gradient via parameter-shift rule
        grad = np.zeros(2 * p)
        for i in range(2 * p):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += np.pi / 2
            x_minus[i] -= np.pi / 2
            
            f_plus = _expected_cut(n, cut_sizes, x_plus.reshape(p, 2))
            f_minus = _expected_cut(n, cut_sizes, x_minus.reshape(p, 2))
            
            grad[i] = (f_plus - f_minus) / 2.0
        
        # Gradient ascent step with adaptive learning rate and momentum
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-12:
            break
        
        # Adaptive learning rate that decays
        step_size = lr / (1.0 + step * 0.05)
        
        # Momentum update
        velocity = momentum * velocity + (1 - momentum) * grad
        x_new = x + step_size * velocity
        
        # Clip to valid range [0, pi]
        x_new = np.clip(x_new, 0.0, np.pi)
        
        f_new = _expected_cut(n, cut_sizes, x_new.reshape(p, 2))
        
        if f_new > best_val + epsilon:
            best_val = f_new
            best_x = x_new.copy()
            x = x_new
        else:
            # Backtrack with line search
            found = False
            for backtrack in range(8):
                step_size *= 0.5
                x_new = x + step_size * velocity
                x_new = np.clip(x_new, 0.0, np.pi)
                f_new = _expected_cut(n, cut_sizes, x_new.reshape(p, 2))
                if f_new > best_val + epsilon:
                    best_val = f_new
                    best_x = x_new.copy()
                    x = x_new
                    found = True
                    break
            if not found:
                # Try smaller step without momentum
                velocity *= 0.5
                for backtrack in range(5):
                    step_size *= 0.5
                    x_new = x + step_size * grad
                    x_new = np.clip(x_new, 0.0, np.pi)
                    f_new = _expected_cut(n, cut_sizes, x_new.reshape(p, 2))
                    if f_new > best_val + epsilon:
                        best_val = f_new
                        best_x = x_new.copy()
                        x = x_new
                        break
                else:
                    break
    
    return best_x, best_val


def _coarse_grid_search(n, cut_sizes, p):
    """Coarse grid search to find a good starting point for local optimization."""
    best_angles = None
    best_score = -np.inf
    
    # Use a moderately fine grid (10 points per dimension for p=2 => 10^4 evaluations)
    # For n<=10, each evaluation is fast, so this is feasible
    grid_size = 10
    gamma_grid = np.linspace(0.05, 1.5, grid_size)
    beta_grid = np.linspace(0.05, 1.0, grid_size)
    
    if p == 2:
        for g1 in gamma_grid:
            for b1 in beta_grid:
                for g2 in gamma_grid:
                    for b2 in beta_grid:
                        if _time_left() < 30.0:
                            return best_angles, best_score
                        angles = np.array([[g1, b1], [g2, b2]])
                        score = _expected_cut(n, cut_sizes, angles)
                        if score > best_score:
                            best_score = score
                            best_angles = angles.copy()
    else:
        # For other p values, use a simpler approach
        for _ in range(100):
            if _time_left() < 30.0:
                return best_angles, best_score
            gammas = np.random.uniform(0.05, 1.5, p)
            betas = np.random.uniform(0.05, 1.0, p)
            angles = np.column_stack((gammas, betas))
            score = _expected_cut(n, cut_sizes, angles)
            if score > best_score:
                best_score = score
                best_angles = angles.copy()
    
    return best_angles, best_score


def _initial_guesses_from_structure(n, edges, p, grid_best=None):
    """Generate structure-informed initial angle guesses."""
    m = len(edges)
    density = m / (n * (n - 1) / 2) if n > 1 else 0.0
    
    # Degree-based heuristic
    degree = np.zeros(n, dtype=np.int32)
    for (i, j) in edges:
        degree[i] += 1
        degree[j] += 1
    max_deg = int(np.max(degree)) if n > 0 else 0
    avg_deg = 2.0 * m / n if n > 0 else 0.0
    
    guesses = []
    
    # Include grid best if available
    if grid_best is not None:
        guesses.append(grid_best.flatten())
    
    # Heuristic 1: based on edge density
    base_gamma = 0.15 + 0.3 * density
    base_beta = 0.2 + 0.15 * (1.0 - density)
    guesses.append(np.tile([base_gamma, base_beta], p))
    
    # Heuristic 2: degree-scaled
    gamma2 = 0.1 + 0.05 * avg_deg
    beta2 = 0.25 - 0.02 * avg_deg
    guesses.append(np.tile([max(0.05, gamma2), max(0.05, beta2)], p))
    
    # Heuristic 3: known good QAOA starting points from literature
    guesses.append(np.array([0.35, 0.25] * p))
    guesses.append(np.array([0.4, 0.2] * p))
    guesses.append(np.array([0.25, 0.3] * p))
    
    # Heuristic 4: small angles (near identity)
    guesses.append(np.tile([0.1, 0.1], p))
    
    # Heuristic 5: moderate angles
    guesses.append(np.tile([0.5, 0.35], p))
    
    # Heuristic 6: larger angles
    guesses.append(np.tile([0.6, 0.4], p))
    
    # Heuristic 7: asymmetric layers
    if p == 2:
        guesses.append(np.array([0.3, 0.2, 0.45, 0.35]))
        guesses.append(np.array([0.5, 0.3, 0.35, 0.4]))
        guesses.append(np.array([0.25, 0.35, 0.55, 0.25]))
        guesses.append(np.array([0.4, 0.15, 0.3, 0.45]))
    
    # A few random guesses with fixed seed for reproducibility
    rng = np.random.RandomState(42)
    for _ in range(8):
        gammas = rng.uniform(0.08, 1.0, p)
        betas = rng.uniform(0.03, 0.6, p)
        guesses.append(np.column_stack((gammas, betas)).flatten())
    
    # Structure-specific guesses based on degree distribution
    if n > 0:
        degree = np.zeros(n, dtype=np.int32)
        for (i, j) in edges:
            degree[i] += 1
            degree[j] += 1
        deg_var = np.var(degree)
        deg_max = np.max(degree)
        
        # More aggressive angles for heterogeneous graphs
        if deg_var > 0.5:
            guesses.append(np.array([0.35 + 0.05 * deg_var, 0.2, 0.5 + 0.05 * deg_var, 0.3]))
            guesses.append(np.array([0.25, 0.4, 0.45, 0.25]))
        
        # Conservative angles for homogeneous graphs
        if deg_var < 0.3:
            guesses.append(np.array([0.3, 0.3, 0.3, 0.3]))
            guesses.append(np.array([0.35, 0.25, 0.35, 0.25]))
    
    return guesses


def angles(n, edges, p):
    """Design graph-dependent two-layer QAOA angles."""
    global _DEADLINE
    _DEADLINE = time.time() + _TIME_LIMIT
    
    # Precompute cut sizes
    cut_sizes = _precompute_cut_sizes(n, edges)
    
    best_angles = None
    best_score = -np.inf
    
    # Phase 1: Coarse grid search for global landscape exploration
    grid_best, grid_score = _coarse_grid_search(n, cut_sizes, p)
    if grid_best is not None:
        best_score = grid_score
        best_angles = grid_best
    
    # Generate structure-informed initial guesses (including grid best)
    initial_guesses = _initial_guesses_from_structure(n, edges, p, grid_best)
    
    # Phase 2: Multi-restart Nelder-Mead optimization
    # Sort initial guesses by heuristic quality (put most promising first)
    # and allocate more iterations to earlier (better) starts
    sorted_guesses = sorted(
        enumerate(initial_guesses), 
        key=lambda x: -x[0]  # Later indices are typically better heuristics
    )
    
    for idx, (orig_idx, x0) in enumerate(sorted_guesses):
        if _time_left() < 20.0:
            break
        
        # Allocate iterations based on position: earlier (better) gets more
        # First 3 gets 300, next 3 gets 250, rest gets 200
        if idx < 3:
            maxiter = 300
            tol = 1e-10
        elif idx < 6:
            maxiter = 250
            tol = 1e-10
        else:
            maxiter = 200
            tol = 1e-9
        
        x_opt, score = _nelder_mead(x0, n, cut_sizes, p, maxiter=maxiter, tol=tol)
        
        if score > best_score:
            best_score = score
            best_angles = x_opt.reshape(p, 2)
    
    # Phase 3: Gradient refinement on the best result
    if best_angles is not None and _time_left() > 30.0:
        x_grad, score_grad = _gradient_refinement(
            best_angles.flatten(), n, cut_sizes, p, max_steps=25, lr=0.015
        )
        if score_grad > best_score:
            best_score = score_grad
            best_angles = x_grad.reshape(p, 2)
    
    # Phase 4: Final high-precision polish on the best result (Nelder-Mead)
    if best_angles is not None and _time_left() > 25.0:
        x_polished, score_polished = _nelder_mead(
            best_angles.flatten(), n, cut_sizes, p, maxiter=800, tol=1e-14
        )
        if score_polished > best_score:
            best_score = score_polished
            best_angles = x_polished.reshape(p, 2)
    
    # Phase 5: Gradient refinement with momentum after polish
    if best_angles is not None and _time_left() > 15.0:
        x_grad2, score_grad2 = _gradient_refinement(
            best_angles.flatten(), n, cut_sizes, p, max_steps=25, lr=0.012
        )
        if score_grad2 > best_score:
            best_score = score_grad2
            best_angles = x_grad2.reshape(p, 2)
    
    # Phase 6: Ultra-precise final polish if time permits
    if best_angles is not None and _time_left() > 8.0:
        x_final, score_final = _nelder_mead(
            best_angles.flatten(), n, cut_sizes, p, maxiter=400, tol=1e-15
        )
        if score_final > best_score:
            best_score = score_final
            best_angles = x_final.reshape(p, 2)
    
    # Fallback if no valid optimization result
    if best_angles is None:
        best_angles = np.tile([0.35, 0.25], (p, 1))
    
    return best_angles
# EVOLVE-BLOCK-END
