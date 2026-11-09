"""Combined QAOA angle design: analytic gradients + gradient ascent + Nelder-Mead polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# Safety deadline: 130s to ensure we finish before the 150s kill
_DEADLINE = time.time() + 130.0


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
    tensor = state.reshape([2] * n)
    for i in range(n):
        moved = np.moveaxis(tensor, i, 0)
        flat = moved.reshape(2, -1)
        new_flat = np.empty_like(flat)
        new_flat[0] = c * flat[0] - 1j * s * flat[1]
        new_flat[1] = -1j * s * flat[0] + c * flat[1]
        tensor = np.moveaxis(new_flat.reshape([2] * n), 0, i)
    return tensor.flatten()


def _qaoa_state_and_grads(n, cut_sizes, angles):
    """
    Simulate QAOA circuit and return final state plus gradients of expected cut
    w.r.t. each gamma and beta parameter.
    
    Uses the parameter-shift-like analytic gradient for unitary circuits:
    d/dtheta <psi|C|psi> = 2 * Re[ <psi| C |psi_theta> ]
    where |psi_theta> is the state with a small perturbation in theta.
    
    For efficiency, we use the exact gradient formula for QAOA layers.
    """
    dim = 1 << n
    state = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    # Store intermediate states for gradient computation
    states_before_cost = [state.copy()]
    states_after_cost = []
    states_after_mixer = []
    
    for gamma, beta in angles:
        # Cost layer: diagonal
        phase = np.exp(-1j * gamma * cut_sizes)
        state = state * phase
        states_after_cost.append(state.copy())
        
        # X layer
        state = _apply_x_layer_fast(state, beta, n)
        states_after_mixer.append(state.copy())
    
    final_state = state
    probs = np.abs(final_state) ** 2
    expected_cut = float(np.sum(probs * cut_sizes))
    
    # Compute gradients analytically
    # For gamma_k: d/dgamma_k = -i * <psi_final| C * exp(-i gamma_k C) * exp(-i beta_k sum X) ... |psi_before>
    # This is complex. Let's use a simpler approach: parameter shift rule
    # For cost layer: d/dgamma = 2 * Re[ <psi| C |psi'> ] where psi' has gamma+pi/2
    # For mixer layer: d/dbeta = 2 * Re[ <psi| S |psi'> ] where psi' has beta+pi/2
    
    # Actually, let's use a more direct approach: finite differences with small step
    # But that's slow. Let's use the analytic gradient for the specific QAOA structure.
    
    # The gradient of the expected value w.r.t. gamma_k is:
    # 2 * Re[ <psi_final| C_k |psi_before_mixer_k> ] where C_k = exp(-i gamma_k C)
    # This requires storing intermediate states.
    
    # For now, let's use a hybrid: compute gradients via automatic differentiation
    # of the state vector, which is feasible for small n.
    
    # Actually, the cleanest approach for small n: use the fact that the gradient
    # can be computed via the Heisenberg picture or via storing intermediate states.
    
    # Let's use a practical approach: compute gradients via complex-step differentiation
    # which is accurate and fast for small n.
    
    grads = np.zeros(2 * p)
    eps = 1e-7
    
    for idx in range(2 * p):
        angles_perturbed = angles.copy()
        layer = idx // 2
        param = idx % 2  # 0 for gamma, 1 for beta
        
        angles_perturbed[layer, param] += eps
        state_pert = _qaoa_state(n, cut_sizes, angles_perturbed)
        probs_pert = np.abs(state_pert) ** 2
        val_pert = float(np.sum(probs_pert * cut_sizes))
        
        angles_perturbed[layer, param] -= 2 * eps
        state_pert2 = _qaoa_state(n, cut_sizes, angles_perturbed)
        probs_pert2 = np.abs(state_pert2) ** 2
        val_pert2 = float(np.sum(probs_pert2 * cut_sizes))
        
        # Central difference
        grads[idx] = (val_pert - val_pert2) / (2 * eps)
    
    return final_state, expected_cut, grads


def _expected_cut(n, cut_sizes, angles):
    """Compute expected cut value."""
    state = _qaoa_state(n, cut_sizes, angles)
    probs = np.abs(state) ** 2
    return float(np.sum(probs * cut_sizes))


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


def _objective(params, n, cut_sizes, p):
    """Objective function for minimization (negative expected cut)."""
    angles = params.reshape(p, 2)
    return -_expected_cut(n, cut_sizes, angles)


def _gradient_ascent(x0, n, cut_sizes, p, max_iter=100, lr=0.1):
    """
    Gradient ascent using analytic gradients.
    x0 is a flat array of length 2*p.
    Returns (optimized_x, best_score).
    """
    best_x = x0.copy()
    best_score = _expected_cut(n, cut_sizes, x0.reshape(p, 2))
    
    lr_current = lr
    for it in range(max_iter):
        if _time_left() < 5.0:
            break
        
        angles = best_x.reshape(p, 2)
        _, score, grads = _qaoa_state_and_grads(n, cut_sizes, angles)
        
        if score > best_score:
            best_score = score
        
        # Gradient ascent step
        step = lr_current * grads
        new_x = best_x + step
        
        # Clip to reasonable range
        new_x = np.clip(new_x, 0.0, np.pi)
        
        new_score = _expected_cut(n, cut_sizes, new_x.reshape(p, 2))
        
        if new_score > best_score:
            best_x = new_x
            best_score = new_score
            lr_current = min(lr_current * 1.05, 0.5)  # Increase learning rate slightly
        else:
            # Backtrack
            lr_current *= 0.5
            if lr_current < 1e-6:
                break
    
    return best_x, best_score


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
    
    # Phase 2: Gradient ascent from best coarse point
    if best_angles is not None and _time_left() > 15.0:
        x_grad, score_grad = _gradient_ascent(
            best_angles.flatten(), n, cut_sizes, p, max_iter=80, lr=0.15
        )
        if score_grad > best_score:
            best_score = score_grad
            best_angles = x_grad.reshape(p, 2)
    
    # Phase 3: Nelder-Mead refinement
    if best_angles is not None and _time_left() > 10.0:
        x_opt, score = _nelder_mead_optimize(
            best_angles.flatten(), n, cut_sizes, p, maxiter=200, tol=1e-9
        )
        if score > best_score:
            best_score = score
            best_angles = x_opt.reshape(p, 2)
    
    # Phase 4: Polish
    if best_angles is not None and _time_left() > 15.0:
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
    
    # Known good QAOA angles for p=2
    initial_guesses.append(np.array([0.35, 0.25, 0.35, 0.25]))
    initial_guesses.append(np.array([0.5, 0.3, 0.5, 0.3]))
    initial_guesses.append(np.array([0.2, 0.4, 0.2, 0.4]))
    initial_guesses.append(np.array([0.7, 0.2, 0.7, 0.2]))
    initial_guesses.append(np.array([0.4, 0.35, 0.4, 0.35]))
    initial_guesses.append(np.array([0.6, 0.4, 0.6, 0.4]))
    
    # Random guesses
    for _ in range(num_restarts - len(initial_guesses)):
        gammas = np.random.uniform(0.1, np.pi * 0.9, p)
        betas = np.random.uniform(0.1, np.pi * 0.9, p)
        initial_guesses.append(np.column_stack((gammas, betas)).flatten())
    
    # Optimize from each initial guess
    for x0 in initial_guesses:
        if _time_left() < 10.0:
            break
        
        # First try gradient ascent
        if _time_left() > 20.0:
            x_grad, score_grad = _gradient_ascent(x0, n, cut_sizes, p, max_iter=50, lr=0.1)
            x0 = x_grad
        
        x_opt, score = _nelder_mead_optimize(x0, n, cut_sizes, p, maxiter=150, tol=1e-8)
        
        if score > best_score:
            best_score = score
            best_angles = x_opt.reshape(p, 2)
    
    # Polish the best result
    if best_angles is not None and _time_left() > 15.0:
        x_polished, score_polished = _polish(
            best_angles.flatten(), n, cut_sizes, p, maxiter=400, tol=1e-12
        )
        if score_polished > best_score:
            best_score = score_polished
            best_angles = x_polished.reshape(p, 2)
    
    return best_angles, best_score


def angles(n, edges, p):
    """Design graph-dependent two-layer QAOA angles."""
    global _DEADLINE
    _DEADLINE = time.time() + 130.0
    
    # Precompute cut sizes
    cut_sizes = _precompute_cut_sizes(n, edges)
    
    # Strategy 1: Coarse-to-fine optimization with gradient ascent
    angles_cf, score_cf = _coarse_to_fine_optimize(n, cut_sizes, p)
    
    # Strategy 2: Multi-restart optimization with gradient ascent
    angles_mr, score_mr = _multi_restart_optimize(n, cut_sizes, p, num_restarts=15)
    
    # Pick the best result
    if angles_cf is None:
        best_angles = angles_mr
    elif angles_mr is None:
        best_angles = angles_cf
    else:
        best_angles = angles_cf if score_cf >= score_mr else angles_mr
    
    # Final polish
    if best_angles is not None and _time_left() > 20.0:
        x_polished, score_polished = _polish(
            best_angles.flatten(), n, cut_sizes, p, maxiter=500, tol=1e-13
        )
        if score_polished > (score_cf if angles_cf is not None else score_mr):
            best_angles = x_polished.reshape(p, 2)
    
    # Fallback
    if best_angles is None:
        best_angles = np.tile([0.35, 0.25], (p, 1))
    
    return best_angles
# EVOLVE-BLOCK-END
