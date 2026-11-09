"""Vectorized QAOA with analytical gradient and multi-restart optimization."""
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


def _qaoa_state(n, cut_sizes, angles):
    """Simulate QAOA circuit and return final state vector."""
    dim = 1 << n
    state = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    for gamma, beta in angles:
        # Cost layer: diagonal phase
        state *= np.exp(-1j * gamma * cut_sizes)
        
        # X layer: prod_i (cos(beta) I - i sin(beta) X_i)
        c = np.cos(beta)
        s = np.sin(beta)
        for i in range(n):
            # Apply rotation along axis i
            # Split state into even/odd indices for bit i
            mask = (np.arange(dim) >> i) & 1
            even_idx = np.where(mask == 0)[0]
            odd_idx = np.where(mask == 1)[0]
            
            a0 = state[even_idx].copy()
            a1 = state[odd_idx].copy()
            
            state[even_idx] = c * a0 - 1j * s * a1
            state[odd_idx] = -1j * s * a0 + c * a1
    
    return state


def _expected_cut(state, cut_sizes):
    """Compute expected cut value <state|C|state>."""
    probs = np.abs(state) ** 2
    return float(np.real(np.sum(probs * cut_sizes)))


def _objective_and_grad(params, n, cut_sizes, p):
    """Compute objective and gradient for QAOA optimization."""
    angles_arr = params.reshape(p, 2)
    state = _qaoa_state(n, cut_sizes, angles_arr)
    val = _expected_cut(state, cut_sizes)
    
    # Compute gradients using parameter-shift or finite differences
    # For speed, use central finite differences with small step
    eps = 1e-7
    grad = np.zeros_like(params)
    
    for k in range(len(params)):
        # Forward
        params_plus = params.copy()
        params_plus[k] += eps
        val_plus = -_expected_cut(_qaoa_state(n, cut_sizes, params_plus.reshape(p, 2)), cut_sizes)
        
        # Backward
        params_minus = params.copy()
        params_minus[k] -= eps
        val_minus = -_expected_cut(_qaoa_state(n, cut_sizes, params_minus.reshape(p, 2)), cut_sizes)
        
        grad[k] = (val_plus - val_minus) / (2 * eps)
    
    return -val, grad


def _objective_only(params, n, cut_sizes, p):
    """Objective only (no gradient)."""
    angles_arr = params.reshape(p, 2)
    state = _qaoa_state(n, cut_sizes, angles_arr)
    return -_expected_cut(state, cut_sizes)


def _optimize_graph(n, edges, p):
    """Optimize QAOA angles with gradient-based multi-restart optimization."""
    cut_sizes = _precompute_cut_sizes(n, edges)
    
    # Known good initial parameters for p=2
    base_init = np.array([0.35, 0.25, 0.25, 0.15])
    
    best_params = base_init.copy()
    best_val = -_objective_only(base_init, n, cut_sizes, p)
    
    rng = np.random.RandomState(42)
    
    # Generate diverse starting points
    starts = [base_init.copy()]
    
    # Random perturbations
    for trial in range(30):
        if _time_left() < 5.0:
            break
        perturbation = rng.randn(2 * p) * 0.4
        x_try = base_init + perturbation
        x_try = np.mod(x_try + np.pi, 2 * np.pi) - np.pi
        starts.append(x_try)
    
    # Structured starts based on known QAOA regimes
    structured = [
        np.array([0.5, 0.3, 0.3, 0.15]),
        np.array([0.4, 0.2, 0.3, 0.1]),
        np.array([0.6, 0.35, 0.2, 0.1]),
        np.array([0.25, 0.15, 0.35, 0.2]),
        np.array([0.45, 0.25, 0.25, 0.12]),
        np.array([0.3, 0.2, 0.3, 0.2]),
        np.array([0.5, 0.25, 0.2, 0.15]),
        np.array([0.4, 0.3, 0.35, 0.1]),
        np.array([0.35, 0.15, 0.4, 0.2]),
        np.array([0.55, 0.2, 0.25, 0.18]),
    ]
    starts.extend(structured)
    
    # Deduplicate
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
    
    # Optimize each start using L-BFGS-B with gradient
    for idx, x0 in enumerate(unique_starts):
        if _time_left() < 5.0:
            break
        
        try:
            result = minimize(
                _objective_and_grad,
                x0,
                args=(n, cut_sizes, p),
                method='L-BFGS-B',
                jac=True,
                bounds=[(-np.pi, np.pi)] * (2 * p),
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
            )
            
            val = -result.fun
            if val > best_val:
                best_val = val
                best_params = result.x.copy()
                
        except Exception:
            continue
    
    # Final high-precision polish with Nelder-Mead for robustness
    if _time_left() > 15.0:
        try:
            result2 = minimize(
                _objective_only,
                best_params,
                args=(n, cut_sizes, p),
                method='Nelder-Mead',
                options={
                    'maxiter': 300,
                    'xatol': 1e-10,
                    'fatol': 1e-12,
                    'adaptive': True
                }
            )
            val2 = -result2.fun
            if val2 > best_val:
                best_val = val2
                best_params = result2.x.copy()
        except Exception:
            pass
    
    # Another L-BFGS-B polish from best point
    if _time_left() > 10.0:
        try:
            result3 = minimize(
                _objective_and_grad,
                best_params,
                args=(n, cut_sizes, p),
                method='L-BFGS-B',
                jac=True,
                bounds=[(-np.pi, np.pi)] * (2 * p),
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 100}
            )
            val3 = -result3.fun
            if val3 > best_val:
                best_val = val3
                best_params = result3.x.copy()
        except Exception:
            pass
    
    return best_params.reshape(p, 2)


def angles(n, edges, p):
    """Design graph-dependent two-layer QAOA angles."""
    global _DEADLINE
    _DEADLINE = time.time() + 140.0
    
    if _time_left() < 5.0:
        return np.array([[0.35, 0.25], [0.25, 0.15]])[:p]
    
    try:
        opt_angles = _optimize_graph(n, edges, p)
        
        if not np.all(np.isfinite(opt_angles)):
            return np.array([[0.35, 0.25], [0.25, 0.15]])[:p]
        
        return opt_angles
    except Exception:
        return np.array([[0.35, 0.25], [0.25, 0.15]])[:p]
# EVOLVE-BLOCK-END
