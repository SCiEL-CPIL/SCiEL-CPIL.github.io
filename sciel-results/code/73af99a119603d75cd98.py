"""Rewritten program with model-and-optimize approach using efficient QAOA simulation."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations

# Safety deadline: 140s to ensure we finish before the 150s kill
_DEADLINE = time.time() + 140.0


def _time_left():
    return _DEADLINE - time.time()


def _precompute_cut_sizes(n, edges):
    """Precompute the cut size for all 2^n states using vectorized operations."""
    dim = 1 << n
    cut_sizes = np.zeros(dim, dtype=np.float64)
    for (i, j) in edges:
        states = np.arange(dim, dtype=np.int64)
        bi = (states >> i) & 1
        bj = (states >> j) & 1
        cut_sizes += (bi != bj).astype(np.float64)
    return cut_sizes


def _apply_x_layer_fast(state, beta, n):
    """Apply exp(-i beta sum X) to state vector using tensor operations."""
    c = np.cos(beta)
    s = np.sin(beta)
    # X gate matrix: [[c, -1js], [-1js, c]]
    # Apply to each qubit axis
    tensor = state.reshape([2] * n)
    for i in range(n):
        # Move axis i to front
        tensor = np.moveaxis(tensor, i, 0)
        # Reshape to (2, -1)
        flat = tensor.reshape(2, -1)
        # Apply X gate: new[0] = c*old[0] - 1js*old[1], new[1] = -1js*old[0] + c*old[1]
        new0 = c * flat[0] - 1j * s * flat[1]
        new1 = -1j * s * flat[0] + c * flat[1]
        # Reconstruct
        tensor = np.empty_like(flat)
        tensor[0] = new0
        tensor[1] = new1
        tensor = tensor.reshape([2] * n)
        # Move axis back
        tensor = np.moveaxis(tensor, 0, i)
    return tensor.flatten()


def _qaoa_state(n, cut_sizes, angles):
    """Simulate QAOA circuit and return final state."""
    dim = 1 << n
    state = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    for gamma, beta in angles:
        # Cost layer: diagonal
        state *= np.exp(-1j * gamma * cut_sizes)
        # X layer
        state = _apply_x_layer_fast(state, beta, n)
        
    return state


def _expected_cut(n, cut_sizes, angles):
    """Compute expected cut value."""
    state = _qaoa_state(n, cut_sizes, angles)
    probs = np.abs(state) ** 2
    return float(np.sum(probs * cut_sizes))


def _exact_maxcut(n, edges):
    """Find exact max cut by brute force for small n."""
    dim = 1 << n
    best_cut = 0
    for s in range(dim):
        cut = 0
        for (i, j) in edges:
            if ((s >> i) & 1) != ((s >> j) & 1):
                cut += 1
        if cut > best_cut:
            best_cut = cut
    return best_cut


def _objective(params, n, cut_sizes, p):
    """Objective function for minimization (negative expected cut)."""
    angles = params.reshape(p, 2)
    return -_expected_cut(n, cut_sizes, angles)


def _nelder_mead_optimize(x0, n, cut_sizes, p, maxiter=300, tol=1e-10):
    """Run Nelder-Mead optimization with time guard."""
    try:
        res = minimize(
            _objective,
            x0,
            args=(n, cut_sizes, p),
            method='Nelder-Mead',
            options={
                'maxiter': maxiter,
                'xatol': tol,
                'fatol': tol,
                'adaptive': True
            }
        )
        return res.x, -res.fun
    except Exception:
        return x0, _expected_cut(n, cut_sizes, x0.reshape(p, 2))


def _polish(x0, n, cut_sizes, p, maxiter=500, tol=1e-12):
    """High-precision polish pass using Nelder-Mead with tighter tolerance."""
    try:
        res = minimize(
            _objective,
            x0,
            args=(n, cut_sizes, p),
            method='Nelder-Mead',
            options={
                'maxiter': maxiter,
                'xatol': tol,
                'fatol': tol,
                'adaptive': True
            }
        )
        return res.x, -res.fun
    except Exception:
        return x0, _expected_cut(n, cut_sizes, x0.reshape(p, 2))


def _gradient_descent_optimize(x0, n, cut_sizes, p, lr=0.01, max_iter=2000, tol=1e-12):
    """Gradient ascent using parameter-shift rule for exact gradients."""
    x = x0.copy()
    best_x = x.copy()
    best_score = _expected_cut(n, cut_sizes, x.reshape(p, 2))
    
    for it in range(max_iter):
        if _time_left() < 5.0:
            break
        
        # Compute gradient using parameter-shift rule
        grad = np.zeros(p * 2)
        for i in range(p * 2):
            xp = x.copy()
            xm = x.copy()
            xp[i] += np.pi / 4
            xm[i] -= np.pi / 4
            fp = _expected_cut(n, cut_sizes, xp.reshape(p, 2))
            fm = _expected_cut(n, cut_sizes, xm.reshape(p, 2))
            grad[i] = (fp - fm) / 2
        
        # Update
        x = x + lr * grad
        
        # Evaluate
        score = _expected_cut(n, cut_sizes, x.reshape(p, 2))
        if score > best_score:
            best_score = score
            best_x = x.copy()
        
        # Check convergence
        if np.max(np.abs(grad)) < tol:
            break
        
        # Adaptive learning rate
        if it % 50 == 0:
            lr = max(0.001, lr * 0.9)
    
    return best_x, best_score


def _coarse_to_fine_optimize(n, cut_sizes, p):
    """Multi-scale optimization: coarse grid then local refinement."""
    best_angles = None
    best_score = -np.inf
    
    # Phase 1: Coarse grid search
    gamma_grid = np.linspace(0.05, np.pi * 0.95, 8)
    beta_grid = np.linspace(0.05, np.pi * 0.95, 8)
    
    best_coarse = None
    best_coarse_score = -np.inf
    
    for g1 in gamma_grid:
        for b1 in beta_grid:
            for g2 in gamma_grid:
                for b2 in beta_grid:
                    angles = np.array([[g1, b1], [g2, b2]])
                    score = _expected_cut(n, cut_sizes, angles)
                    if score > best_coarse_score:
                        best_coarse_score = score
                        best_coarse = angles.copy()
    
    if best_coarse is not None:
        best_score = best_coarse_score
        best_angles = best_coarse
    
    # Phase 2: Local refinement from best coarse point
    if best_angles is not None and _time_left() > 10.0:
        x_opt, score = _nelder_mead_optimize(
            best_angles.flatten(), n, cut_sizes, p, maxiter=200, tol=1e-9
        )
        if score > best_score:
            best_score = score
            best_angles = x_opt.reshape(p, 2)
    
    # Phase 3: Gradient descent polish
    if best_angles is not None and _time_left() > 15.0:
        x_grad, score_grad = _gradient_descent_optimize(
            best_angles.flatten(), n, cut_sizes, p, lr=0.01, max_iter=1000, tol=1e-12
        )
        if score_grad > best_score:
            best_score = score_grad
            best_angles = x_grad.reshape(p, 2)
    
    # Phase 4: Final Nelder-Mead polish
    if best_angles is not None and _time_left() > 10.0:
        x_polished, score_polished = _polish(
            best_angles.flatten(), n, cut_sizes, p, maxiter=400, tol=1e-12
        )
        if score_polished > best_score:
            best_score = score_polished
            best_angles = x_polished.reshape(p, 2)
    
    return best_angles, best_score


def _multi_restart_optimize(n, cut_sizes, p, num_restarts=20):
    """Multi-restart Nelder-Mead with diverse initial guesses."""
    best_angles = None
    best_score = -np.inf
    
    # Generate diverse initial guesses
    np.random.seed(123)
    initial_guesses = []
    
    # Known good QAOA angles for p=2 (from literature and empirical testing)
    initial_guesses.append(np.array([0.35, 0.25, 0.35, 0.25]))
    initial_guesses.append(np.array([0.5, 0.3, 0.5, 0.3]))
    initial_guesses.append(np.array([0.2, 0.4, 0.2, 0.4]))
    initial_guesses.append(np.array([0.7, 0.2, 0.7, 0.2]))
    initial_guesses.append(np.array([0.4, 0.35, 0.4, 0.35]))
    initial_guesses.append(np.array([0.6, 0.4, 0.6, 0.4]))
    initial_guesses.append(np.array([0.3, 0.3, 0.3, 0.3]))
    initial_guesses.append(np.array([0.45, 0.2, 0.45, 0.2]))
    initial_guesses.append(np.array([0.55, 0.25, 0.55, 0.25]))
    initial_guesses.append(np.array([0.25, 0.35, 0.25, 0.35]))
    
    # Random guesses with better coverage
    for _ in range(num_restarts - len(initial_guesses)):
        gammas = np.random.uniform(0.1, np.pi * 0.8, p)
        betas = np.random.uniform(0.1, np.pi * 0.7, p)
        initial_guesses.append(np.column_stack((gammas, betas)).flatten())
    
    # Optimize from each initial guess
    for x0 in initial_guesses:
        if _time_left() < 10.0:
            break
        
        x_opt, score = _nelder_mead_optimize(x0, n, cut_sizes, p, maxiter=150, tol=1e-8)
        
        if score > best_score:
            best_score = score
            best_angles = x_opt.reshape(p, 2)
    
    # Polish the best result with tighter tolerance
    if best_angles is not None and _time_left() > 15.0:
        x_polished, score_polished = _polish(
            best_angles.flatten(), n, cut_sizes, p, maxiter=500, tol=1e-12
        )
        if score_polished > best_score:
            best_score = score_polished
            best_angles = x_polished.reshape(p, 2)
    
    return best_angles, best_score


def angles(n, edges, p):
    """Design graph-dependent two-layer QAOA angles."""
    global _DEADLINE
    _DEADLINE = time.time() + 140.0
    
    # Precompute cut sizes
    cut_sizes = _precompute_cut_sizes(n, edges)
    
    # Strategy 1: Coarse-to-fine optimization
    angles_cf, score_cf = _coarse_to_fine_optimize(n, cut_sizes, p)
    
    # Strategy 2: Multi-restart optimization
    angles_mr, score_mr = _multi_restart_optimize(n, cut_sizes, p, num_restarts=15)
    
    # Pick the best result
    if angles_cf is None:
        best_angles = angles_mr
    elif angles_mr is None:
        best_angles = angles_cf
    else:
        best_angles = angles_cf if score_cf >= score_mr else angles_mr
    
    # Final gradient-based polish
    if best_angles is not None and _time_left() > 15.0:
        x_grad, score_grad = _gradient_descent_optimize(
            best_angles.flatten(), n, cut_sizes, p, lr=0.005, max_iter=1500, tol=1e-13
        )
        if score_grad > (score_cf if angles_cf is not None else score_mr):
            best_angles = x_grad.reshape(p, 2)
    
    # Final Nelder-Mead polish with very tight tolerance
    if best_angles is not None and _time_left() > 10.0:
        x_polished, score_polished = _polish(
            best_angles.flatten(), n, cut_sizes, p, maxiter=600, tol=1e-13
        )
        if score_polished > (score_cf if angles_cf is not None else score_mr):
            best_angles = x_polished.reshape(p, 2)
    
    # Fallback
    if best_angles is None:
        best_angles = np.tile([0.35, 0.25], (p, 1))
    
    return best_angles
# EVOLVE-BLOCK-END
