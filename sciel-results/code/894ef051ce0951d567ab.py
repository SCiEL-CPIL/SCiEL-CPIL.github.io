"""Vectorized QAOA with multi-restart Nelder-Mead optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 140.0


def _time_left():
    return _DEADLINE - time.time()


def _precompute_cut_sizes(n, edges):
    """Precompute cut size for each basis state."""
    dim = 1 << n
    cut_sizes = np.zeros(dim, dtype=np.float64)
    for (i, j) in edges:
        for state in range(dim):
            bi = (state >> i) & 1
            bj = (state >> j) & 1
            if bi != bj:
                cut_sizes[state] += 1.0
    return cut_sizes


def _exact_maxcut(n, edges):
    """Compute exact MaxCut value by brute force."""
    dim = 1 << n
    best = 0
    for state in range(dim):
        cut = 0
        for (i, j) in edges:
            bi = (state >> i) & 1
            bj = (state >> j) & 1
            if bi != bj:
                cut += 1
        if cut > best:
            best = cut
    return best


def _qaoa_state(n, cut_sizes, angles):
    """Simulate QAOA circuit using vectorized tensor operations.
    
    State is shaped as (2,)*n tensor.
    X-layer: product of single-qubit rotations, applied via tensor operations.
    Cost-layer: diagonal phase, applied via reshape to flat array.
    """
    dim = 1 << n
    # Start in |+>^n: all amplitudes equal
    state = np.ones((2,) * n, dtype=np.complex128) / np.sqrt(dim)
    
    flat = cut_sizes  # length dim
    
    for gamma, beta in angles:
        # Cost layer: diagonal phase
        flat_state = state.reshape(-1)
        phase = np.exp(-1j * gamma * flat)
        flat_state *= phase
        state = flat_state.reshape((2,) * n)
        
        # X layer: prod_i (cos(beta) I - i sin(beta) X_i)
        c = np.cos(beta)
        s = np.sin(beta)
        for i in range(n):
            # X_i flips bit i: swap axes i's 0 and 1 with rotation
            # new_state[..., 0, ...] = c * old[..., 0, ...] - i*s * old[..., 1, ...]
            # new_state[..., 1, ...] = c * old[..., 1, ...] - i*s * old[..., 0, ...]
            state = np.swapaxes(state, i, i)  # no-op, just for clarity
            # Actually let's do it properly:
            # We need to apply (c*I - i*s*X) along axis i
            # X swaps 0 and 1, so:
            # out[..., 0, ...] = c * in[..., 0, ...] + (-i*s) * in[..., 1, ...]
            # out[..., 1, ...] = (-i*s) * in[..., 0, ...] + c * in[..., 1, ...]
            # Let's extract slices
            sl0 = [slice(None)] * n
            sl1 = [slice(None)] * n
            sl0[i] = 0
            sl1[i] = 1
            
            a0 = state[tuple(sl0)].copy()
            a1 = state[tuple(sl1)].copy()
            
            new_a0 = c * a0 + (-1j * s) * a1
            new_a1 = (-1j * s) * a0 + c * a1
            
            state[tuple(sl0)] = new_a0
            state[tuple(sl1)] = new_a1
    
    return state.reshape(-1)


def _expected_cut(state, cut_sizes):
    """Compute expected cut value <state|C|state>."""
    probs = np.abs(state) ** 2
    return float(np.real(np.sum(probs * cut_sizes)))


def _objective(params, n, cut_sizes, p):
    """Objective: minimize negative expected cut."""
    angles_arr = params.reshape(p, 2)
    state = _qaoa_state(n, cut_sizes, angles_arr)
    return -_expected_cut(state, cut_sizes)


def _optimize_graph(n, edges, p, n_restarts=12, max_time=120.0):
    """Optimize QAOA angles with multiple restarts using Nelder-Mead."""
    cut_sizes = _precompute_cut_sizes(n, edges)
    
    # Known good initial parameters from literature for p=2
    # Typical optimal ranges: gamma in [0, pi/2], beta in [0, pi/4]
    base_init = np.array([0.35, 0.25, 0.25, 0.15])  # gamma1, beta1, gamma2, beta2
    
    best_params = base_init.copy()
    best_val = -_objective(base_init, n, cut_sizes, p)  # positive expected cut
    
    rng = np.random.RandomState(42)
    
    # Generate diverse starting points
    starts = [base_init.copy()]
    
    # Add perturbed versions
    for trial in range(20):
        if _time_left() < 5.0:
            break
        perturbation = rng.randn(2 * p) * 0.5
        x_try = base_init + perturbation
        # Wrap angles to [-pi, pi]
        x_try = np.mod(x_try + np.pi, 2 * np.pi) - np.pi
        starts.append(x_try)
    
    # Also try some structured starts
    structured = [
        np.array([0.5, 0.3, 0.3, 0.15]),
        np.array([0.4, 0.2, 0.3, 0.1]),
        np.array([0.6, 0.35, 0.2, 0.1]),
        np.array([0.25, 0.15, 0.35, 0.2]),
        np.array([0.45, 0.25, 0.25, 0.12]),
    ]
    starts.extend(structured)
    
    # Deduplicate (roughly)
    seen = []
    unique_starts = []
    for s in starts:
        is_dup = False
        for seen_s in seen:
            if np.allclose(s, seen_s, atol=0.01):
                is_dup = True
                break
        if not is_dup:
            seen.append(s)
            unique_starts.append(s)
    
    # Optimize each start
    for idx, x0 in enumerate(unique_starts):
        if _time_left() < 5.0:
            break
        
        try:
            result = minimize(
                _objective,
                x0,
                args=(n, cut_sizes, p),
                method='Nelder-Mead',
                options={
                    'maxiter': 200,
                    'xatol': 1e-6,
                    'fatol': 1e-8,
                    'adaptive': True
                }
            )
            
            val = -result.fun  # positive expected cut
            if val > best_val:
                best_val = val
                best_params = result.x.copy()
                
        except Exception:
            continue
    
    # Try L-BFGS-B from the best Nelder-Mead result for refinement
    if _time_left() > 10.0:
        try:
            result2 = minimize(
                _objective,
                best_params,
                args=(n, cut_sizes, p),
                method='L-BFGS-B',
                bounds=[(-np.pi, np.pi)] * (2 * p),
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            val2 = -result2.fun
            if val2 > best_val:
                best_val = val2
                best_params = result2.x.copy()
        except Exception:
            pass
    
    return best_params.reshape(p, 2)


def angles(n, edges, p):
    """Design graph-dependent two-layer QAOA angles."""
    global _DEADLINE
    _DEADLINE = time.time() + 140.0
    
    # Check time budget
    if _time_left() < 5.0:
        return np.array([[0.35, 0.25], [0.25, 0.15]])[:p]
    
    try:
        opt_angles = _optimize_graph(n, edges, p)
        
        # Validate: ensure angles are finite
        if not np.all(np.isfinite(opt_angles)):
            return np.array([[0.35, 0.25], [0.25, 0.15]])[:p]
        
        return opt_angles
    except Exception:
        return np.array([[0.35, 0.25], [0.25, 0.15]])[:p]
# EVOLVE-BLOCK-END
