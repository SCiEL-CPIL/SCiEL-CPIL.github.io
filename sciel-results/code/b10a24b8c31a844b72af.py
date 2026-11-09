"""Direct structure-derived QAOA angle design with graph-invariant initialization and multi-restart polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 145.0

def _time_left():
    return _DEADLINE - time.time()

def _precompute_cut_sizes(n, edges):
    """Precompute cut size for all 2^n states vectorized."""
    dim = 1 << n
    cut_sizes = np.zeros(dim, dtype=np.float64)
    for (i, j) in edges:
        # Create mask for bit i and bit j
        # We can do this efficiently by generating the bit patterns
        # But for n<=10, even a loop over edges is fine if the inner is vectorized
        states = np.arange(dim, dtype=np.int64)
        bi = (states >> i) & 1
        bj = (states >> j) & 1
        cut_sizes += (bi != bj).astype(np.float64)
    return cut_sizes

def _qaoa_state(n, cut_sizes, angles):
    """Simulate p-layer QAOA circuit with vectorized mixing."""
    dim = 1 << n
    state = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    for gamma, beta in angles:
        # Cost layer: exp(-i gamma C)
        # C is diagonal, so element-wise multiplication
        state *= np.exp(-1j * gamma * cut_sizes)
        
        # Mixing layer: exp(-i beta sum X)
        # This is cos(beta) I - i sin(beta) X^tensor n
        # X^tensor n flips all bits.
        # In the computational basis indexed by integer k, the flipped state is at index (dim - 1 - k).
        # So state_flipped[k] = state[dim - 1 - k]
        # This is equivalent to reversing the array.
        c = np.cos(beta)
        s = np.sin(beta)
        state = c * state - 1j * s * state[::-1]
        
    return state

def _expected_cut(n, cut_sizes, angles):
    """Compute E[C] = <psi|C|psi>."""
    state = _qaoa_state(n, cut_sizes, angles)
    probs = np.abs(state) ** 2
    return float(np.sum(probs * cut_sizes))

def _objective(params, n, cut_sizes, p):
    """Negative expected cut for minimization."""
    angles = params.reshape(p, 2)
    return -_expected_cut(n, cut_sizes, angles)

def _graph_invariant_init(n, edges, p):
    """
    Direct initialization from graph structure.
    """
    m = len(edges)
    max_edges = n * (n - 1) // 2
    density = m / max_edges if max_edges > 0 else 0.0
    
    # Average degree
    avg_deg = 2.0 * m / n if n > 0 else 0.0
    
    # Degree variance
    deg = np.zeros(n)
    for (i, j) in edges:
        deg[i] += 1
        deg[j] += 1
    deg_var = np.var(deg)
    
    # Heuristic initialization
    gamma1 = np.pi * 0.35 + 0.3 * density
    gamma2 = np.pi * 0.15 + 0.2 * density
    beta1 = np.pi * 0.25 - 0.1 * density
    beta2 = np.pi * 0.15
    
    scale = 1.0 + 0.1 * (deg_var / (avg_deg + 1e-9)) if avg_deg > 0 else 1.0
    gamma1 *= scale
    gamma2 *= scale
    
    angles = np.array([
        [gamma1 % (2 * np.pi), beta1 % (2 * np.pi)],
        [gamma2 % (2 * np.pi), beta2 % (2 * np.pi)]
    ], dtype=np.float64)
    return angles

def _nelder_mead(x0, n, cut_sizes, p, maxiter=200, tol=1e-10):
    """Nelder-Mead with time guard."""
    try:
        if _time_left() < 5.0:
            return x0, _expected_cut(n, cut_sizes, x0.reshape(p, 2))
        res = minimize(
            _objective, x0, args=(n, cut_sizes, p),
            method='Nelder-Mead',
            options={'maxiter': maxiter, 'xatol': tol, 'fatol': tol, 'adaptive': True}
        )
        return res.x, -res.fun
    except Exception:
        return x0, _expected_cut(n, cut_sizes, x0.reshape(p, 2))

def _l_bfgs_b(x0, n, cut_sizes, p, maxiter=100):
    """L-BFGS-B with finite difference gradients."""
    try:
        if _time_left() < 5.0:
            return x0, _expected_cut(n, cut_sizes, x0.reshape(p, 2))
        res = minimize(
            _objective, x0, args=(n, cut_sizes, p),
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        return res.x, -res.fun
    except Exception:
        return x0, _expected_cut(n, cut_sizes, x0.reshape(p, 2))

def angles(n, edges, p):
    """Design graph-dependent two-layer QAOA angles."""
    global _DEADLINE
    _DEADLINE = time.time() + 145.0
    
    cut_sizes = _precompute_cut_sizes(n, edges)
    
    best_angles = None
    best_score = -np.inf
    
    # Build diverse initial guesses
    initial_guesses = []
    
    # Structure-derived direct init
    init_struct = _graph_invariant_init(n, edges, p)
    initial_guesses.append(init_struct.flatten())
    
    # Perturbed structure-derived inits
    rng = np.random.default_rng(42)
    for _ in range(15): # Increased restarts due to speedup
        noise_g = rng.normal(0, 0.2, p)
        noise_b = rng.normal(0, 0.2, p)
        perturbed = init_struct.copy()
        perturbed[:, 0] = (perturbed[:, 0] + noise_g) % (2 * np.pi)
        perturbed[:, 1] = (perturbed[:, 1] + noise_b) % (2 * np.pi)
        initial_guesses.append(perturbed.flatten())
        
    # Standard literature-based anchors
    initial_guesses.append(np.array([0.35, 0.25, 0.25, 0.15]))
    initial_guesses.append(np.array([0.5, 0.3, 0.3, 0.2]))
    initial_guesses.append(np.array([0.7, 0.2, 0.4, 0.15]))
    initial_guesses.append(np.array([np.pi/4, np.pi/8, np.pi/8, np.pi/16]))
    
    # Random uniform inits
    for _ in range(30): # More random restarts
        g = rng.uniform(0, np.pi, p)
        b = rng.uniform(0, np.pi, p)
        initial_guesses.append(np.column_stack((g, b)).flatten())
        
    # Multi-restart Optimization
    # We alternate or just run Nelder-Mead, then polish best with L-BFGS-B
    for idx, x0 in enumerate(initial_guesses):
        if _time_left() < 15.0:
            break
            
        # Use Nelder-Mead for robustness to local minima
        x_opt, score = _nelder_mead(x0, n, cut_sizes, p, maxiter=100, tol=1e-8)
        
        if score > best_score:
            best_score = score
            best_angles = x_opt.reshape(p, 2)
            
    # Final high-precision polish with L-BFGS-B (faster convergence near optimum)
    if best_angles is not None and _time_left() > 10.0:
        try:
            # L-BFGS-B can handle the smooth landscape well
            x_polished, score_polished = _l_bfgs_b(best_angles.flatten(), n, cut_sizes, p, maxiter=200)
            if score_polished > best_score:
                best_score = score_polished
                best_angles = x_polished.reshape(p, 2)
                
            # One more Nelder-Mead pass on the polished result to ensure we didn't get stuck in a saddle
            if _time_left() > 5.0:
                x_final, score_final = _nelder_mead(best_angles.flatten(), n, cut_sizes, p, maxiter=200, tol=1e-12)
                if score_final > best_score:
                    best_score = score_final
                    best_angles = x_final.reshape(p, 2)
        except Exception:
            pass
    
    # Fallback
    if best_angles is None:
        best_angles = np.tile([0.35, 0.25], (p, 1))
        
    return best_angles
# EVOLVE-BLOCK-END
