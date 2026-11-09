"""Rewritten program with analytic gradient ascent for QAOA angle design."""
# EVOLVE-BLOCK-START
import numpy as np
import time

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
    tensor = state.reshape([2] * n)
    for i in range(n):
        moved = np.moveaxis(tensor, i, 0)
        flat = moved.reshape(2, -1)
        new_flat = np.empty_like(flat)
        new_flat[0] = c * flat[0] - 1j * s * flat[1]
        new_flat[1] = -1j * s * flat[0] + c * flat[1]
        tensor = np.moveaxis(new_flat.reshape([2] * n), 0, i)
    return tensor.flatten()


def _qaoa_state_with_history(n, cut_sizes, angles):
    """
    Simulate QAOA circuit and return final state plus intermediate states for gradient computation.
    
    Returns:
        final_state: complex array of shape (2^n,)
        state_history: list of states after each layer (for gradient computation)
    """
    dim = 1 << n
    state = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    state_history = [state.copy()]
    
    for gamma, beta in angles:
        # Cost layer: diagonal
        state = state * np.exp(-1j * gamma * cut_sizes)
        state_history.append(state.copy())
        # X layer
        state = _apply_x_layer_fast(state, beta, n)
        state_history.append(state.copy())
        
    return state, state_history


def _compute_gradients(n, cut_sizes, angles, state_history):
    """
    Compute exact gradients of expected cut with respect to QAOA parameters.
    
    For p=2, we have parameters: gamma1, beta1, gamma2, beta2
    
    The expected cut is E[C] = <psi|C|psi>
    
    Gradient with respect to gamma_k (cost layer k):
    dE/dgamma_k = -i * <psi| C * (C_k - <C_k>) |psi>
    where C_k is the cost operator applied at layer k
    
    Gradient with respect to beta_k (mixing layer k):
    dE/dbeta_k = -i * <psi| (sum X) * (something) |psi>
    
    For efficiency, we use the fact that the initial state is uniform and
    the cost operator is diagonal.
    """
    dim = 1 << n
    final_state = state_history[-1]
    probs = np.abs(final_state) ** 2
    expected_cut = float(np.sum(probs * cut_sizes))
    
    gradients = np.zeros(2 * len(angles), dtype=np.float64)
    
    # For each layer k (0-indexed), compute gradient wrt gamma_k and beta_k
    for k, (gamma, beta) in enumerate(angles):
        # State before cost layer k
        state_before_cost = state_history[2 * k]
        # State after cost layer k
        state_after_cost = state_history[2 * k + 1]
        # State after mixing layer k
        state_after_mix = state_history[2 * k + 2]
        
        # Gradient wrt gamma_k:
        # dE/dgamma = -2 * Im(<psi_final| U_{k+1}...U_p C (exp(-i gamma C) - <exp(-i gamma C)>) |psi_before_cost>)
        # Simplified: dE/dgamma = -2 * Im(<psi_final| U_{k+1}...U_p C * (C - <C>) * |psi_before_cost>)
        # where <C> is computed with respect to the state after cost layer
        
        # Compute <C> for state_after_cost
        probs_after_cost = np.abs(state_after_cost) ** 2
        C_mean = float(np.sum(probs_after_cost * cut_sizes))
        
        # C - <C> is diagonal
        C_deviation = cut_sizes - C_mean
        
        # Apply U_{k+1}...U_p to C_deviation * state_before_cost
        # This is equivalent to evolving the perturbed state
        perturbed_state = state_before_cost * C_deviation
        perturbed_state = perturbed_state.astype(np.complex128)
        
        # Apply remaining layers (k+1 to p-1)
        for j in range(k + 1, len(angles)):
            gamma_j, beta_j = angles[j]
            # Cost layer
            perturbed_state = perturbed_state * np.exp(-1j * gamma_j * cut_sizes)
            # Mixing layer
            perturbed_state = _apply_x_layer_fast(perturbed_state, beta_j, n)
        
        # Gradient wrt gamma_k
        grad_gamma_k = -2.0 * np.imag(np.vdot(final_state, perturbed_state)).real
        gradients[2 * k] = grad_gamma_k
        
        # Gradient wrt beta_k:
        # dE/dbeta = -2 * Im(<psi_final| U_{k+1}...U_p (sum X) * (exp(-i beta sum X) - <exp(-i beta sum X)>) |psi_after_cost>)
        # Simplified: dE/dbeta = -2 * Im(<psi_final| U_{k+1}...U_p (sum X) * (S - <S>) * |psi_after_cost>)
        # where S = sum X, and <S> is computed with respect to state_after_cost
        
        # Compute S|psi_after_cost> (sum of X operators)
        S_state = np.zeros(dim, dtype=np.complex128)
        for i in range(n):
            # Flip bit i
            mask = 1 << i
            for s in range(dim):
                flipped = s ^ mask
                S_state[s] += state_after_cost[flipped]
        
        # <S> = <psi_after_cost|S|psi_after_cost>
        S_mean = np.real(np.vdot(state_after_cost, S_state))
        
        # S - <S> applied to state_after_cost
        S_deviation_state = S_state - S_mean * state_after_cost
        
        # Apply remaining layers (k+1 to p-1)
        perturbed_state2 = S_deviation_state.copy()
        for j in range(k + 1, len(angles)):
            gamma_j, beta_j = angles[j]
            # Cost layer
            perturbed_state2 = perturbed_state2 * np.exp(-1j * gamma_j * cut_sizes)
            # Mixing layer
            perturbed_state2 = _apply_x_layer_fast(perturbed_state2, beta_j, n)
        
        # Gradient wrt beta_k
        grad_beta_k = -2.0 * np.imag(np.vdot(final_state, perturbed_state2)).real
        gradients[2 * k + 1] = grad_beta_k
    
    return gradients, expected_cut


def _line_search(n, cut_sizes, angles, gradients, max_iter=20, tol=1e-10):
    """
    Perform backtracking line search to find optimal step size.
    
    Returns updated angles and the achieved score.
    """
    current_angles = angles.copy()
    current_score = _expected_cut(n, cut_sizes, current_angles)
    
    # Initial step size
    step_size = 0.5
    
    for _ in range(max_iter):
        if _time_left() < 5.0:
            break
        
        # Try step
        new_angles = current_angles + step_size * gradients.reshape(-1, 2)
        
        # Clip to reasonable range
        new_angles = np.clip(new_angles, -np.pi, np.pi)
        
        new_score = _expected_cut(n, cut_sizes, new_angles)
        
        # Check improvement
        if new_score > current_score + 1e-12:
            return new_angles, new_score
        else:
            # Reduce step size
            step_size *= 0.5
    
    # No improvement found, return original
    return current_angles, current_score


def _expected_cut(n, cut_sizes, angles):
    """Compute expected cut value."""
    state, _ = _qaoa_state_with_history(n, cut_sizes, angles)
    probs = np.abs(state) ** 2
    return float(np.sum(probs * cut_sizes))


def _gradient_ascent_optimize(x0, n, cut_sizes, p, max_iter=100, tol=1e-12):
    """
    Perform analytic gradient ascent with line search.
    
    Args:
        x0: initial parameters, shape (2*p,)
        n: number of qubits
        cut_sizes: precomputed cut sizes
        p: number of QAOA layers
        max_iter: maximum iterations
        tol: convergence tolerance
    
    Returns:
        optimized parameters and score
    """
    current_params = x0.copy()
    current_angles = current_params.reshape(p, 2)
    current_score = _expected_cut(n, cut_sizes, current_angles)
    
    for iteration in range(max_iter):
        if _time_left() < 10.0:
            break
        
        # Compute gradients
        _, state_history = _qaoa_state_with_history(n, cut_sizes, current_angles)
        gradients, score = _compute_gradients(n, cut_sizes, current_angles, state_history)
        
        # Check convergence
        grad_norm = np.linalg.norm(gradients)
        if grad_norm < tol:
            break
        
        # Line search
        new_params, new_score = _line_search(n, cut_sizes, current_angles, gradients)
        
        # Update
        if new_score > current_score:
            current_params = new_params.flatten()
            current_angles = new_params.reshape(p, 2)
            current_score = new_score
        else:
            # No improvement, try smaller steps
            break
    
    return current_params, current_score


def _coarse_grid_search(n, cut_sizes, p):
    """
    Perform coarse grid search to find good initial points.
    
    Returns list of (angles, score) tuples.
    """
    results = []
    
    # For p=2, search over reasonable ranges
    gamma_vals = np.linspace(0.1, 1.5, 5)
    beta_vals = np.linspace(0.1, 1.0, 5)
    
    for g1 in gamma_vals:
        for b1 in beta_vals:
            for g2 in gamma_vals:
                for b2 in beta_vals:
                    if _time_left() < 5.0:
                        return results
                    angles = np.array([[g1, b1], [g2, b2]])
                    score = _expected_cut(n, cut_sizes, angles)
                    results.append((angles, score))
    
    # Sort by score and return top results
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:10]  # Return top 10


def _multi_restart_gradient_ascent(n, cut_sizes, p, num_restarts=15):
    """
    Perform multi-restart gradient ascent with diverse initial guesses.
    
    Returns best angles and score.
    """
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
    
    # Add coarse grid results
    grid_results = _coarse_grid_search(n, cut_sizes, p)
    for angles, score in grid_results[:5]:
        initial_guesses.append(angles.flatten())
    
    # Random guesses
    for _ in range(num_restarts - len(initial_guesses)):
        if len(initial_guesses) >= num_restarts:
            break
        gammas = np.random.uniform(0.1, 1.5, p)
        betas = np.random.uniform(0.1, 1.0, p)
        initial_guesses.append(np.column_stack((gammas, betas)).flatten())
    
    # Optimize from each initial guess
    for x0 in initial_guesses:
        if _time_left() < 15.0:
            break
        
        x_opt, score = _gradient_ascent_optimize(x0, n, cut_sizes, p, max_iter=50, tol=1e-10)
        
        if score > best_score:
            best_score = score
            best_angles = x_opt.reshape(p, 2)
    
    # Final high-precision polish on best result
    if best_angles is not None and _time_left() > 20.0:
        x_polished, score_polished = _gradient_ascent_optimize(
            best_angles.flatten(), n, cut_sizes, p, max_iter=100, tol=1e-14
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
    
    # Perform multi-restart gradient ascent
    best_angles, best_score = _multi_restart_gradient_ascent(n, cut_sizes, p, num_restarts=15)
    
    # Fallback
    if best_angles is None:
        best_angles = np.tile([0.35, 0.25], (p, 1))
    
    return best_angles
# EVOLVE-BLOCK-END
