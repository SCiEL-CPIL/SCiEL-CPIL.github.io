"""
QAOA p=2 graph-dependent angle optimizer.
Uses a global grid search followed by local polishing to maximize the expected cut value.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# Global time management
_START_TIME = time.time()
_DEADLINE = _START_TIME + 140.0  # 140s safety margin out of 150s

# Cache for computed values to handle repeated calls or internal retries efficiently
_value_cache = {}

def _get_elapsed():
    return time.time() - _START_TIME

def _build_state_data(n, edges):
    """
    Pre-compute bit patterns, cut values, and mixer indices.
    """
    dim = 1 << n
    states = np.arange(dim)
    bits = np.zeros((dim, n), dtype=np.int8)
    for j in range(n):
        bits[:, j] = (states >> j) & 1
    
    # Pre-compute cut values for each state
    cut_values = np.zeros(dim, dtype=np.float64)
    for u, v in edges:
        cut_values += (bits[:, u] != bits[:, v]).astype(np.float64)
        
    # Pre-compute indices for mixer application
    mixer_indices = []
    for i in range(n):
        mask_bit = 1 << i
        idx_0 = np.where((states & mask_bit) == 0)[0]
        idx_1 = idx_0 + mask_bit
        mixer_indices.append((idx_0, idx_1))
        
    return bits, cut_values, mixer_indices

def _compute_expected_cut(n, edges, gammas, betas, bits, cut_values, mixer_indices):
    """
    Compute the expected cut value for given QAOA angles.
    Optimized using pre-computed cut values and mixer indices.
    """
    dim = 1 << n
    psi = np.ones(dim, dtype=complex) / np.sqrt(dim)
    
    for layer in range(2):
        gamma = gammas[layer]
        beta = betas[layer]
        
        # Apply Cost Hamiltonian: exp(-i gamma * C)
        # C is diagonal with entries cut_values[s]
        # Phase is exp(-i gamma * cut_values[s])
        if gamma != 0.0:
            phases = np.exp(-1j * gamma * cut_values)
            psi *= phases
            
        # Apply Mixer Hamiltonian: exp(-i beta * sum X_i)
        c_b = np.cos(beta)
        s_b = np.sin(beta)
        
        if beta != 0.0:
            for idx_0, idx_1 in mixer_indices:
                psi_0 = psi[idx_0]
                psi_1 = psi[idx_1]
                
                new_psi_0 = c_b * psi_0 - 1j * s_b * psi_1
                new_psi_1 = -1j * s_b * psi_0 + c_b * psi_1
                
                psi[idx_0] = new_psi_0
                psi[idx_1] = new_psi_1
                
    # Calculate Expected Cut Value
    probs = np.real(np.conj(psi) * psi)
    
    # E[cut] = sum_s probs[s] * cut_values[s]
    total_cut = np.sum(probs * cut_values)
    
    return total_cut

def angles(n, edges, p):
    global _value_cache
    
    start_time = time.time()
    
    # Normalize edges for caching
    edge_set = frozenset(frozenset((u, v)) for u, v in edges)
    cache_key = (n, edge_set)
    
    if cache_key in _value_cache:
        return _value_cache[cache_key]
        
    # Pre-compute state data
    bits, cut_values, mixer_indices = _build_state_data(n, edges)
    
    # Define the objective function for the optimizer
    def objective(params):
        g1, b1, g2, b2 = params
        return -_compute_expected_cut(n, edges, [g1, g2], [b1, b2], bits, cut_values, mixer_indices)
        
    # Check time
    if time.time() > _DEADLINE - 25.0:
        # Emergency return default good params
        return np.array([[0.3, 0.1], [0.3, 0.1]])
        
    best_val = -np.inf
    best_params = None
    
    # Grid Search
    # With the speedup, we can afford a denser grid or more iterations.
    # Let's try 12x12x12x12 = 20,736 points.
    # Or stick to 10x10x10x10 = 10,000 points but ensure it finishes fast.
    
    g_range = np.linspace(0.05, 0.7, 10)
    b_range = np.linspace(0.02, 0.45, 10)
    
    g1s = g_range
    b1s = b_range
    g2s = g_range
    b2s = b_range
    
    count = 0
    for g1 in g1s:
        for b1 in b1s:
            for g2 in g2s:
                for b2 in b2s:
                    if time.time() > _DEADLINE - 15.0:
                        break
                    count += 1
                    val = _compute_expected_cut(n, edges, [g1, g2], [b1, b2], bits, cut_values, mixer_indices)
                    if val > best_val:
                        best_val = val
                        best_params = np.array([g1, b1, g2, b2])
                if time.time() > _DEADLINE - 15.0: break
            if time.time() > _DEADLINE - 15.0: break
        if time.time() > _DEADLINE - 15.0: break
        
    if best_params is None:
        # Fallback if grid didn't run or failed
        best_params = np.array([0.3, 0.1, 0.3, 0.1])
        best_val = _compute_expected_cut(n, edges, [0.3, 0.3], [0.1, 0.1], bits, cut_values, mixer_indices)
        
    # 2. Local Optimization (Polish)
    # Use BFGS starting from the best grid point
    if time.time() < _DEADLINE - 10.0:
        try:
            result = minimize(
                objective,
                best_params,
                method='BFGS',
                options={'maxiter': 100, 'gtol': 1e-6}
            )
            if result.success:
                res_val = -result.fun
                if res_val > best_val:
                    best_val = res_val
                    best_params = result.x
        except Exception:
            pass
            
    # 3. Nelder-Mead Polish (Robustness)
    if time.time() < _DEADLINE - 5.0:
        try:
            result_nm = minimize(
                objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-7, 'fatol': 1e-7}
            )
            if result_nm.success:
                res_val = -result_nm.fun
                if res_val > best_val:
                    best_val = res_val
                    best_params = result_nm.x
        except Exception:
            pass
    
    # 4. High-precision polish: L-BFGS-B with tight bounds and very tight tolerance
    if time.time() < _DEADLINE - 2.0:
        try:
            # Bounds: gammas in [0, pi/2], betas in [0, pi/2]
            bounds = [(0.0, np.pi/2), (0.0, np.pi/2), (0.0, np.pi/2), (0.0, np.pi/2)]
            result_lbfgs = minimize(
                objective,
                best_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'gtol': 1e-8, 'ftol': 1e-10}
            )
            if result_lbfgs.success:
                res_val = -result_lbfgs.fun
                if res_val > best_val:
                    best_val = res_val
                    best_params = result_lbfgs.x
        except Exception:
            pass
    
    # 5. Multi-start from neighborhood of best point to escape potential local minima
    if time.time() < _DEADLINE - 1.0:
        try:
            # Try a few small perturbations around the best point
            perturbations = [
                [0.02, 0.0, 0.02, 0.0],
                [-0.02, 0.0, -0.02, 0.0],
                [0.0, 0.02, 0.0, 0.02],
                [0.0, -0.02, 0.0, -0.02],
                [0.01, 0.01, 0.01, 0.01],
                [-0.01, -0.01, -0.01, -0.01],
            ]
            for pert in perturbations:
                if time.time() > _DEADLINE - 0.5:
                    break
                start = best_params + np.array(pert)
                # Clamp to valid range
                start = np.clip(start, 0.0, np.pi/2)
                try:
                    res = minimize(
                        objective,
                        start,
                        method='L-BFGS-B',
                        bounds=[(0.0, np.pi/2)] * 4,
                        options={'maxiter': 50, 'gtol': 1e-6}
                    )
                    if res.success:
                        res_val = -res.fun
                        if res_val > best_val:
                            best_val = res_val
                            best_params = res.x
                except Exception:
                    pass
        except Exception:
            pass

    # Format output: (p, 2) array with rows (gamma, beta)
    out = np.zeros((p, 2))
    out[0, 0] = best_params[0]  # gamma 1
    out[0, 1] = best_params[1]  # beta 1
    if p > 1:
        out[1, 0] = best_params[2]  # gamma 2
        out[1, 1] = best_params[3]  # beta 2
        
    _value_cache[cache_key] = out
    return out
# EVOLVE-BLOCK-END
