"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 140.0  # 10s safety margin before 150s kill

def _time_left():
    return _DEADLINE - time.time()

def _precompute_graph(n, edges):
    """Precompute cut values and edge masks for vectorized operations."""
    dim = 1 << n
    states = np.arange(dim, dtype=np.int64)
    
    # Compute cut values vectorized
    cut_values = np.zeros(dim, dtype=np.float64)
    for (i, j) in edges:
        bits_i = (states >> i) & 1
        bits_j = (states >> j) & 1
        cut_values += (bits_i != bits_j).astype(np.float64)
    
    # Precompute edge masks for expected cut computation
    edge_masks = []
    for (i, j) in edges:
        bits_i = (states >> i) & 1
        bits_j = (states >> j) & 1
        edge_masks.append(bits_i != bits_j)
    
    # Precompute mixer permutation structure for fast application
    # For each qubit i, precompute the reshape shape
    mixer_shapes = []
    for i in range(n):
        mixer_shapes.append((dim >> (i + 1), 2, 1 << i))
    
    return cut_values, edge_masks, mixer_shapes

def _apply_mixer(state, beta, n, mixer_shapes):
    """Apply the mixer exp(-i*beta*sum X_i) to state in-place."""
    cb = np.cos(beta)
    sb = np.sin(beta)
    
    for i in range(n):
        shape = mixer_shapes[i]
        state_reshaped = state.reshape(shape)
        
        a = state_reshaped[:, 0, :]
        b = state_reshaped[:, 1, :]
        
        new_a = cb * a - 1j * sb * b
        new_b = -1j * sb * a + cb * b
        
        state_reshaped[:, 0, :] = new_a
        state_reshaped[:, 1, :] = new_b

def _qaoa_state(n, gammas, betas, precomputed):
    """Compute the final state vector of the QAOA circuit."""
    cut_values, edge_masks, mixer_shapes = precomputed
    dim = 1 << n
    
    state = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    for gamma, beta in zip(gammas, betas):
        state = state * np.exp(-1j * gamma * cut_values)
        _apply_mixer(state, beta, n, mixer_shapes)
    
    return state

def _qaoa_expected_cut(n, edges, gammas, betas, precomputed=None):
    """Compute the expected cut size for a given QAOA circuit with p layers."""
    if n > 12:
        return 0.5 * len(edges)
    
    if precomputed is None:
        precomputed = _precompute_graph(n, edges)
    else:
        cut_values, edge_masks, mixer_shapes = precomputed
    
    state = _qaoa_state(n, gammas, betas, precomputed)
    probs = np.abs(state) ** 2
    expected_cut = 0.0
    for mask in edge_masks:
        expected_cut += np.sum(probs[mask])
    
    return expected_cut

def _qaoa_gradient(n, edges, gammas, betas, precomputed):
    """
    Compute exact gradients of expected cut w.r.t. all parameters using
    the parameter-shift rule: d/dθ <C> = (f(θ+π/2) - f(θ-π/2)) / 2
    """
    p = len(gammas)
    dim = 2 * p  # total number of parameters
    grad = np.zeros(dim, dtype=np.float64)
    
    # For each parameter, compute shifted values
    for k in range(p):
        # Gradient w.r.t. gamma_k (index 2k)
        g_plus = gammas.copy()
        g_minus = gammas.copy()
        g_plus[k] += np.pi / 2
        g_minus[k] -= np.pi / 2
        
        f_plus = _qaoa_expected_cut(n, edges, g_plus, betas, precomputed)
        f_minus = _qaoa_expected_cut(n, edges, g_minus, betas, precomputed)
        grad[2 * k] = (f_plus - f_minus) / 2.0
        
        # Gradient w.r.t. beta_k (index 2k+1)
        b_plus = betas.copy()
        b_minus = betas.copy()
        b_plus[k] += np.pi / 2
        b_minus[k] -= np.pi / 2
        
        f_plus = _qaoa_expected_cut(n, edges, gammas, b_plus, precomputed)
        f_minus = _qaoa_expected_cut(n, edges, gammas, b_minus, precomputed)
        grad[2 * k + 1] = (f_plus - f_minus) / 2.0
    
    return grad

def _gradient_ascent(n, edges, p, x_init, precomputed, max_iters=200, time_budget=15.0):
    """
    Perform gradient ascent on the expected cut using exact parameter-shift gradients.
    Uses adaptive step size with backtracking.
    """
    start_time = time.time()
    x = x_init.copy()
    gammas = x[0::2].copy()
    betas = x[1::2].copy()
    
    best_x = x.copy()
    best_val = _qaoa_expected_cut(n, edges, gammas, betas, precomputed)
    
    step_size = 0.1
    min_step = 1e-6
    max_step = np.pi / 4
    
    for it in range(max_iters):
        if time.time() - start_time > time_budget:
            break
        if _time_left() < 5:
            break
        
        # Compute gradient
        grad = _qaoa_gradient(n, edges, gammas, betas, precomputed)
        
        # Check if gradient is small
        if np.linalg.norm(grad) < 1e-8:
            break
        
        # Backtracking line search
        alpha = step_size
        f_current = _qaoa_expected_cut(n, edges, gammas, betas, precomputed)
        
        improved = False
        for _ in range(10):
            x_trial = x + alpha * grad
            # Clamp to bounds
            x_trial[0::2] = np.clip(x_trial[0::2], 0, np.pi)
            x_trial[1::2] = np.clip(x_trial[1::2], 0, np.pi / 2)
            
            g_trial = x_trial[0::2]
            b_trial = x_trial[1::2]
            f_trial = _qaoa_expected_cut(n, edges, g_trial, b_trial, precomputed)
            
            if f_trial > f_current:
                x = x_trial
                gammas = g_trial
                betas = b_trial
                f_current = f_trial
                improved = True
                break
            else:
                alpha *= 0.5
        
        if not improved:
            # Try smaller step
            step_size = max(min_step, step_size * 0.5)
            if step_size < min_step:
                break
        else:
            # Increase step size if we made good progress
            if f_trial - f_current > 0.01:
                step_size = min(max_step, step_size * 1.2)
        
        if f_current > best_val:
            best_val = f_current
            best_x = x.copy()
    
    return best_x

def _exhaustive_maxcut(n, edges):
    """Compute the exact maximum cut by brute force for small n."""
    if n > 15:
        return 0.5 * len(edges)
    
    dim = 1 << n
    states = np.arange(dim, dtype=np.int64)
    best = 0
    for z in states:
        cut = 0
        for (i, j) in edges:
            bit_i = (z >> i) & 1
            bit_j = (z >> j) & 1
            if bit_i != bit_j:
                cut += 1
        if cut > best:
            best = cut
    
    return best

def _optimize_angles(n, edges, p, time_limit):
    """
    Optimize QAOA angles using DE for global search + gradient ascent for local polish.
    """
    from scipy.optimize import differential_evolution
    
    # Precompute graph data
    precomputed = _precompute_graph(n, edges)
    
    def objective(x):
        gammas = x[0::2]
        betas = x[1::2]
        return -_qaoa_expected_cut(n, edges, gammas, betas, precomputed)
    
    # Bounds: gamma in [0, pi], beta in [0, pi/2]
    bounds = [(0, np.pi)] * p + [(0, np.pi / 2)] * p
    
    # Initial guess
    best_x = np.array([0.35, 0.25] * p, dtype=np.float64)
    best_val = objective(best_x)
    
    # Phase 1: Differential Evolution for global exploration
    if _time_left() > 25:
        try:
            de_result = differential_evolution(
                objective,
                bounds,
                maxiter=40,
                popsize=12,
                tol=1e-6,
                seed=42,
                polish=False,
                init='sobol',
                updating='deferred'
            )
            if de_result.fun < best_val:
                best_val = de_result.fun
                best_x = de_result.x.copy()
        except Exception:
            pass
    
    # Phase 2: Gradient ascent from best DE result
    if _time_left() > 10:
        try:
            grad_result = _gradient_ascent(
                n, edges, p, best_x, precomputed,
                max_iters=150,
                time_budget=min(15.0, _time_left() - 8)
            )
            g_val = -_qaoa_expected_cut(n, edges, grad_result[0::2], grad_result[1::2], precomputed)
            if g_val < best_val:
                best_val = g_val
                best_x = grad_result.copy()
        except Exception:
            pass
    
    # Phase 3: Additional gradient ascent from alternative starts
    np.random.seed(42)
    for start_idx in range(3):
        if _time_left() < 8:
            break
        
        if start_idx == 0:
            x_init = best_x.copy()
        else:
            x_init = best_x + np.random.randn(2 * p) * 0.2
            x_init[0::2] = np.clip(x_init[0::2], 0, np.pi)
            x_init[1::2] = np.clip(x_init[1::2], 0, np.pi / 2)
        
        try:
            grad_result = _gradient_ascent(
                n, edges, p, x_init, precomputed,
                max_iters=100,
                time_budget=min(10.0, _time_left() - 6)
            )
            g_val = -_qaoa_expected_cut(n, edges, grad_result[0::2], grad_result[1::2], precomputed)
            if g_val < best_val:
                best_val = g_val
                best_x = grad_result.copy()
        except Exception:
            pass
    
    # Phase 4: Final high-precision gradient polish
    if _time_left() > 4:
        try:
            grad_result = _gradient_ascent(
                n, edges, p, best_x, precomputed,
                max_iters=100,
                time_budget=min(5.0, _time_left() - 3)
            )
            g_val = -_qaoa_expected_cut(n, edges, grad_result[0::2], grad_result[1::2], precomputed)
            if g_val < best_val:
                best_val = g_val
                best_x = grad_result.copy()
        except Exception:
            pass
    
    return best_x

def angles(n, edges, p):
    """
    Return QAOA angles for a given graph.
    
    Args:
        n: number of vertices
        edges: list of (i, j) edges
        p: number of QAOA layers
    
    Returns:
        np.array of shape (p, 2), rows are (gamma, beta) in radians
    """
    # Check time
    if _time_left() < 5:
        return np.tile([0.35, 0.25], (p, 1))
    
    try:
        x_opt = _optimize_angles(n, edges, p, _time_left() - 5)
        gammas = x_opt[0::2]
        betas = x_opt[1::2]
        result = np.column_stack([gammas, betas])
        return result
    except Exception:
        return np.tile([0.35, 0.25], (p, 1))
# EVOLVE-BLOCK-END
