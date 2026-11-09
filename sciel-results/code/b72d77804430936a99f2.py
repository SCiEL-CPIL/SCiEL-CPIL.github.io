"""Direct structure-derived QAOA angle design with analytic parameter-shift gradient ascent."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = time.time() + 140.0


def _time_left():
    return _DEADLINE - time.time()


def _precompute_cut_sizes(n, edges):
    """Precompute cut size for all 2^n states vectorized."""
    dim = 1 << n
    cut_sizes = np.zeros(dim, dtype=np.float64)
    for (i, j) in edges:
        states = np.arange(dim, dtype=np.int64)
        bi = (states >> i) & 1
        bj = (states >> j) & 1
        cut_sizes += (bi != bj).astype(np.float64)
    return cut_sizes


def _apply_x_layer(state, beta, n):
    """Apply exp(-i beta sum_i X_i) via tensor reshape per qubit."""
    c = np.cos(beta)
    s = np.sin(beta)
    tensor = state.reshape([2] * n)
    for i in range(n):
        moved = np.moveaxis(tensor, i, 0)
        flat = moved.reshape(2, -1)
        new_flat = np.empty_like(flat)
        new_flat[0] = c * flat[0] - 1j * s * flat[1]
        new_flat[1] -1j * s * flat[0] + c * flat[1]
        tensor = np.moveaxis(new_flat.reshape([2] * n), 0, i)
    return tensor.flatten()


def _qaoa_state(n, cut_sizes, angles):
    """Simulate p-layer QAOA circuit."""
    dim = 1 << n
    state = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    for gamma, beta in angles:
        state *= np.exp(-1j * gamma * cut_sizes)
        state = _apply_x_layer(state, beta, n)
    return state


def _expected_cut(n, cut_sizes, angles):
    """Compute E[C] = <psi|C|psi>."""
    state = _qaoa_state(n, cut_sizes, angles)
    probs = np.abs(state) ** 2
    return float(np.sum(probs * cut_sizes))


def _gradient_parameter_shift(n, cut_sizes, angles, p):
    """
    Compute exact gradient using the parameter-shift rule.
    For each parameter theta_i, grad_i = 0.5 * (E(theta_i + pi/2) - E(theta_i - pi/2)).
    This is exact for quantum circuits with Hamiltonians that have eigenvalues differing by integers (like Pauli Z / MaxCut).
    """
    grad = np.zeros(2 * p, dtype=np.float64)
    shift = np.pi / 2.0
    
    # We need to evaluate E for 2 * (2p) points.
    # To be efficient, we can batch or just loop. Given n<=10, looping is fine.
    for k in range(2 * p):
        # Determine which row and column (gamma or beta)
        row = k // 2
        col = k % 2
        
        angles_plus = angles.copy()
        angles_minus = angles.copy()
        
        angles_plus[row, col] += shift
        angles_minus[row, col] -= shift
        
        # Ensure angles are within reasonable range for numerical stability in exp/cos/sin
        # Though parameter shift is invariant to 2pi shifts, keeping them bounded helps slightly
        angles_plus = angles_plus % (2 * np.pi)
        angles_minus = angles_minus % (2 * np.pi)
        
        e_plus = _expected_cut(n, cut_sizes, angles_plus)
        e_minus = _expected_cut(n, cut_sizes, angles_minus)
        
        grad[k] = 0.5 * (e_plus - e_minus)
        
        if _time_left() < 1.0:
            break
            
    return grad


def _graph_invariant_init(n, edges, p):
    """
    Direct initialization from graph structure.
    """
    m = len(edges)
    max_edges = n * (n - 1) // 2
    density = m / max_edges if max_edges > 0 else 0.0
    
    # Heuristic: denser graphs need larger gamma
    gamma1 = np.pi * 0.35 + 0.3 * density
    gamma2 = np.pi * 0.15 + 0.2 * density
    beta1 = np.pi * 0.25 - 0.1 * density
    beta2 = np.pi * 0.15
    
    angles = np.array([
        [gamma1 % (2 * np.pi), beta1 % (2 * np.pi)],
        [gamma2 % (2 * np.pi), beta2 % (2 * np.pi)]
    ], dtype=np.float64)
    return angles


def _gradient_ascent(x0, n, cut_sizes, p, max_steps=100, lr_init=0.1):
    """
    Perform gradient ascent with backtracking line search using exact gradients.
    """
    angles = x0.reshape(p, 2).copy()
    current_val = _expected_cut(n, cut_sizes, angles)
    
    lr = lr_init
    for step in range(max_steps):
        if _time_left() < 2.0:
            break
        
        grad = _gradient_parameter_shift(n, cut_sizes, angles, p)
        
        # Check for convergence
        if np.linalg.norm(grad) < 1e-6:
            break
        
        # Gradient direction
        direction = grad.reshape(p, 2)
        
        # Line search: find step size that increases objective
        # Start with lr, reduce if it doesn't improve
        improved = False
        for ls_iter in range(20):
            step_size = lr * (0.5 ** ls_iter)
            candidate = angles + step_size * direction
            # Keep angles in [0, 2pi)
            candidate = candidate % (2 * np.pi)
            
            candidate_val = _expected_cut(n, cut_sizes, candidate)
            
            if candidate_val > current_val:
                angles = candidate
                current_val = candidate_val
                # Increase learning rate slightly if we took a full step?
                # Or keep it stable. Let's keep it stable for safety.
                improved = True
                break
            else:
                # If we didn't improve, we reduced step size. 
                # If we get to very small steps, we are stuck.
                if step_size < 1e-8:
                    break
        
        if not improved:
            # Stuck at this point, maybe try a small random nudge to escape?
            # Or just break.
            if step_size < 1e-8:
                break
            # If we failed to improve even with tiny steps, it might be a numerical issue or local max.
            # For gradient ascent, if gradient is non-zero but no step improves, it's weird.
            # Usually line search succeeds.
            break

    return angles, current_val


def angles(n, edges, p):
    """Design graph-dependent two-layer QAOA angles."""
    global _DEADLINE
    _DEADLINE = time.time() + 140.0
    
    cut_sizes = _precompute_cut_sizes(n, edges)
    
    best_angles = None
    best_score = -np.inf
    
    # Build diverse initial guesses
    initial_guesses = []
    
    # 1. Structure-derived
    init_struct = _graph_invariant_init(n, edges, p)
    initial_guesses.append(init_struct.flatten())
    
    # 2. Literature anchors
    initial_guesses.append(np.array([0.35, 0.25, 0.25, 0.15]))
    initial_guesses.append(np.array([0.5, 0.3, 0.3, 0.2]))
    initial_guesses.append(np.array([np.pi/4, np.pi/8, np.pi/8, np.pi/16]))
    
    # 3. Random inits
    rng = np.random.default_rng(42)
    for _ in range(10):
        g = rng.uniform(0, np.pi, p)
        b = rng.uniform(0, np.pi, p)
        initial_guesses.append(np.column_stack((g, b)).flatten())
    
    # 4. Perturbed structure inits
    for _ in range(5):
        noise_g = rng.normal(0, 0.2, p)
        noise_b = rng.normal(0, 0.2, p)
        perturbed = init_struct.copy()
        perturbed[:, 0] = (perturbed[:, 0] + noise_g) % (2 * np.pi)
        perturbed[:, 1] = (perturbed[:, 1] + noise_b) % (2 * np.pi)
        initial_guesses.append(perturbed.flatten())

    # Run Gradient Ascent from each start
    for idx, x0 in enumerate(initial_guesses):
        if _time_left() < 5.0:
            break
            
        # Use a moderate number of steps. Gradient ascent can be fast.
        # We can increase max_steps if time allows.
        max_steps = 50
        if _time_left() > 60.0:
            max_steps = 100
            
        x_opt, score = _gradient_ascent(x0, n, cut_sizes, p, max_steps=max_steps)
        
        if score > best_score:
            best_score = score
            best_angles = x_opt
    
    # If we have time, do a final polish with Nelder-Mead on the best result
    # This combines the global search of multi-restart gradient ascent with local precision of NM.
    if best_angles is not None and _time_left() > 10.0:
        try:
            from scipy.optimize import minimize
            # Define objective for NM (minimize negative cut)
            def obj_nm(params):
                ang = params.reshape(p, 2)
                return -_expected_cut(n, cut_sizes, ang)
            
            res = minimize(
                obj_nm, best_angles.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            if -res.fun > best_score:
                best_score = -res.fun
                best_angles = res.x.reshape(p, 2)
        except Exception:
            pass

    # Fallback
    if best_angles is None:
        best_angles = np.tile([0.35, 0.25], (p, 1))
        
    return best_angles
# EVOLVE-BLOCK-END
