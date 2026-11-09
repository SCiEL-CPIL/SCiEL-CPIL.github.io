"""Rewritten program with L-BFGS-B optimization and analytical-inspired initialization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# Safety deadline: 140s to ensure we finish before the 150s kill
_DEADLINE = time.time() + 140.0


def _time_left():
    return _DEADLINE - time.time()


def _precompute_cut_sizes(n, edges):
    """Precompute the cut size for all 2^n states using vectorized operations."""
    dim = 1 << n
    cut_sizes = np.zeros(dim, dtype=np.float64)
    states = np.arange(dim)
    for (i, j) in edges:
        bi = (states >> i) & 1
        bj = (states >> j) & 1
        cut_sizes += (bi != bj).astype(np.float64)
    return cut_sizes


def _apply_x_layer(state, beta, n):
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


def _qaoa_state(n, cut_sizes, angles):
    """Simulate QAOA circuit and return final state."""
    dim = 1 << n
    state = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    for gamma, beta in angles:
        # Cost layer: diagonal
        state *= np.exp(-1j * gamma * cut_sizes)
        # X layer
        state = _apply_x_layer(state, beta, n)
        
    return state


def _expected_cut(n, cut_sizes, angles):
    """Compute expected cut value."""
    state = _qaoa_state(n, cut_sizes, angles)
    probs = np.abs(state) ** 2
    return float(np.sum(probs * cut_sizes))


def _objective(params, n, cut_sizes, p):
    """Objective function for minimization (negative expected cut)."""
    angles = params.reshape(p, 2)
    return -_expected_cut(n, cut_sizes, angles)


def _optimize_lbfgs(x0, n, cut_sizes, p, maxiter=200, tol=1e-8):
    """Run L-BFGS-B optimization with central difference gradients."""
    try:
        res = minimize(
            _objective,
            x0,
            args=(n, cut_sizes, p),
            method='L-BFGS-B',
            options={
                'maxiter': maxiter,
                'gtol': tol,
                'ftol': 1e-12, # Tighter ftol for better convergence
                'maxls': 50,
                'eps': 1e-5   # Central difference step size
            }
        )
        return res.x, -res.fun
    except Exception:
        return x0, _expected_cut(n, cut_sizes, x0.reshape(p, 2))


def _optimize_nm(x0, n, cut_sizes, p, maxiter=100, tol=1e-8):
    """Run Nelder-Mead optimization as a backup or for specific starts."""
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


def _polish(x0, n, cut_sizes, p, maxiter=500, tol=1e-14):
    """High-precision polish pass using L-BFGS with tighter tolerance."""
    try:
        res = minimize(
            _objective,
            x0,
            args=(n, cut_sizes, p),
            method='L-BFGS-B',
            options={
                'maxiter': maxiter,
                'gtol': tol,
                'ftol': 1e-15,
                'maxls': 100,
                'eps': 1e-6
            }
        )
        return res.x, -res.fun
    except Exception:
        return x0, _expected_cut(n, cut_sizes, x0.reshape(p, 2))


def angles(n, edges, p):
    """Design graph-dependent two-layer QAOA angles."""
    global _DEADLINE
    _DEADLINE = time.time() + 135.0
    
    # 1. Precompute cut sizes
    cut_sizes = _precompute_cut_sizes(n, edges)
    
    best_angles = None
    best_score = -np.inf
    
    # 2. Generate diverse initial guesses
    # We use a mix of heuristic points and random points.
    # Heuristic points are based on common QAOA optima for small p.
    np.random.seed(123)
    initial_guesses = []
    
    # Common good starting points for p=2
    heuristics = [
        np.array([0.35, 0.25] * p),
        np.array([0.40, 0.30] * p),
        np.array([0.25, 0.35] * p),
        np.array([0.50, 0.20] * p),
        np.array([0.20, 0.40] * p),
        np.array([0.45, 0.25] * p),
    ]
    initial_guesses.extend(heuristics)
    
    # Random guesses covering the [0, pi] x [0, pi] space
    for _ in range(50):
        gammas = np.random.uniform(0, np.pi, p)
        betas = np.random.uniform(0, np.pi, p)
        initial_guesses.append(np.column_stack((gammas, betas)).flatten())
    
    # 3. Multi-restart optimization
    # Strategy: Use L-BFGS for most restarts as it converges faster per iteration
    # for smooth functions. Use NM for a few if L-BFGS seems stuck (optional, but let's stick to L-BFGS for consistency and speed).
    
    for idx, x0 in enumerate(initial_guesses):
        if _time_left() < 15.0:
            break
        
        # Use L-BFGS for the main search
        x_opt, score = _optimize_lbfgs(x0, n, cut_sizes, p, maxiter=100, tol=1e-8)
        
        if score > best_score:
            best_score = score
            best_angles = x_opt.reshape(p, 2)
            
        # Early stopping if we've likely found a good solution and time is tight
        # If we have a score > 0.95 of max possible (roughly n/4 for random graphs), we can be more aggressive with time limits.
        # Max cut is roughly n^2 * density / 4. For n=10, d=0.5, max cut ~ 12.5.
        # If best_score is high, we can reduce time per restart.

    # 4. Final high-precision polish on the best result
    if best_angles is not None and _time_left() > 10.0:
        x_polished, score_polished = _polish(
            best_angles.flatten(), n, cut_sizes, p, maxiter=500, tol=1e-14
        )
        if score_polished > best_score:
            best_score = score_polished
            best_angles = x_polished.reshape(p, 2)
    
    # 5. Additional polish rounds if time permits
    if best_angles is not None and _time_left() > 15.0:
        for _ in range(2):
            if _time_left() < 5.0:
                break
            x_polished, score_polished = _polish(
                best_angles.flatten(), n, cut_sizes, p, maxiter=300, tol=1e-15
            )
            if score_polished > best_score:
                best_score = score_polished
                best_angles = x_polished.reshape(p, 2)
            else:
                break
    
    # Fallback if no valid optimization result
    if best_angles is None:
        best_angles = np.tile([0.35, 0.25], (p, 1))
        
    return best_angles
# EVOLVE-BLOCK-END
