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

def _build_state_bitpatterns(n):
    """
    Pre-compute bit patterns for all 2^n states.
    Returns an array of shape (2^n, n) with 0s and 1s.
    """
    states = np.arange(1 << n)
    # Create a (2^n, n) array where state[i, j] is the j-th bit of i
    # We want bit 0 (LSB) to correspond to qubit 0
    # state >> j & 1
    # Vectorized way:
    bits = np.zeros((1 << n, n), dtype=np.int8)
    for j in range(n):
        bits[:, j] = (states >> j) & 1
    return states, bits

def _compute_expected_cut(n, edges, gammas, betas, states, bits):
    """
    Compute the expected cut value for given QAOA angles.
    Uses vectorized state evolution.
    """
    dim = 1 << n
    # Initial state |+>^n
    psi = np.ones(dim, dtype=complex) / np.sqrt(dim)
    
    # Pre-compute edge bit pairs for cost Hamiltonian phase
    # Phase for edge (u,v) is exp(-i gamma/2 * (1 - Z_u Z_v))
    # Z_u = 1 if bit u is 0, -1 if bit u is 1 -> Z_u = 1 - 2*bit_u
    # Z_u Z_v = (1-2b_u)(1-2b_v) = 1 - 2b_u - 2b_v + 4b_u b_v
    # If b_u == b_v, Z_u Z_v = 1. Else -1.
    # Phase = exp(-i gamma/2) if b_u != b_v else exp(i gamma/2)
    # Let's pre-compute a matrix or just apply edge by edge.
    # For n<=10, applying edge by edge is fast enough if vectorized over states.
    
    edge_list = list(edges)
    
    for layer in range(2):
        gamma = gammas[layer]
        beta = betas[layer]
        
        # Apply Cost Hamiltonian: exp(-i gamma * C)
        # C = sum_{(u,v) in E} (1 - Z_u Z_v) / 2
        # exp(-i gamma/2 * (1 - Z_u Z_v))
        # For each edge, multiply psi by phase
        # We can vectorize the phase calculation for all states for a given edge
        for u, v in edge_list:
            # Get bits for u and v for all states
            bu = bits[:, u]
            bv = bits[:, v]
            
            # Determine which states have bu == bv
            # If bu == bv, Z_u Z_v = 1, phase = exp(i gamma/2)
            # If bu != bv, Z_u Z_v = -1, phase = exp(-i gamma/2)
            
            # We can create a phase array
            # phase[state] = exp(i gamma/2) if bits[state, u] == bits[state, v] else exp(-i gamma/2)
            
            # Vectorized:
            # diff = bu != bv
            # phase = np.where(diff, np.exp(-1j * gamma / 2), np.exp(1j * gamma / 2))
            # psi *= phase
            
            # Optimization: Pre-compute exp factors
            exp_pos = np.exp(1j * gamma / 2)
            exp_neg = np.exp(-1j * gamma / 2)
            
            # Apply
            # Note: np.where creates a new array. For small dim, this is fine.
            # To avoid temporary array allocation, we can do:
            # mask = bu != bv
            # psi[mask] *= exp_neg
            # psi[~mask] *= exp_pos
            
            mask = (bu != bv)
            psi[mask] *= exp_neg
            psi[~mask] *= exp_pos
            
        # Apply Mixer Hamiltonian: exp(-i beta * sum X_i)
        # This is a tensor product of single-qubit rotations.
        # R_x(beta) = [[cos(beta), -i sin(beta)], [-i sin(beta), cos(beta)]]
        # We apply this to each qubit.
        # Vectorized application for all qubits:
        # For qubit i, we pair states differing only in bit i.
        # psi_new[state] = cos(beta) * psi[state] - i sin(beta) * psi[state ^ (1<<i)]
        
        c_b = np.cos(beta)
        s_b = np.sin(beta)
        
        # We can iterate over qubits. n is small.
        for i in range(n):
            mask_bit = 1 << i
            # Identify pairs
            # We need to apply the rotation matrix to pairs of amplitudes.
            # Let's create a new array to avoid in-place update issues if we weren't careful, 
            # but since each pair (s, s^mask) is disjoint, we can do it in place if we store one value.
            
            # Get indices where the bit i is 0 (state < state ^ mask)
            # Actually, simpler: iterate over all states, but only process if (state & mask_bit) == 0
            # This visits each pair once.
            
            # Vectorized approach:
            # Create two indices arrays
            idx_0 = np.where((np.arange(dim) & mask_bit) == 0)[0]
            idx_1 = idx_0 + mask_bit  # Corresponding index with bit set
            
            psi_0 = psi[idx_0]
            psi_1 = psi[idx_1]
            
            new_psi_0 = c_b * psi_0 - 1j * s_b * psi_1
            new_psi_1 = -1j * s_b * psi_0 + c_b * psi_1
            
            psi[idx_0] = new_psi_0
            psi[idx_1] = new_psi_1
            
    # Calculate Expected Cut Value
    # E[cut] = sum_{(u,v) in E} (1 - <Z_u Z_v>) / 2
    # <Z_u Z_v> = sum_state |psi[state]|^2 * Z_u(state) * Z_v(state)
    # Z_i(state) = 1 if bit_i==0 else -1
    
    total_cut = 0.0
    probs = np.real(np.conj(psi) * psi)
    
    for u, v in edge_list:
        zu = 1.0 - 2.0 * bits[:, u].astype(float)
        zv = 1.0 - 2.0 * bits[:, v].astype(float)
        zz = zu * zv
        exp_zz = np.sum(probs * zz)
        total_cut += (1.0 - exp_zz) / 2.0
        
    return total_cut

def angles(n, edges, p):
    global _value_cache
    
    start_time = time.time()
    
    # Normalize edges for caching
    edge_set = frozenset(frozenset((u, v)) for u, v in edges)
    cache_key = (n, edge_set)
    
    if cache_key in _value_cache:
        return _value_cache[cache_key]
        
    # Pre-compute state bits
    states, bits = _build_state_bitpatterns(n)
    
    # Define the objective function for the optimizer
    def objective(params):
        g1, b1, g2, b2 = params
        return -_compute_expected_cut(n, edges, [g1, g2], [b1, b2], states, bits)
        
    # 1. Global Grid Search
    # Ranges based on QAOA properties and previous knowledge.
    # Gamma typically in [0, pi/2], Beta in [0, pi/2].
    # Let's use a coarse grid first.
    
    # Check time
    if time.time() > _DEADLINE - 25.0:
        # Emergency return default good params
        return np.array([[0.3, 0.1], [0.3, 0.1]])
        
    # Grid parameters
    # For n=6-10, a 10x10x10x10 grid is 10,000 evals.
    # Each eval is ~ O(2^n * n * |E|). For n=10, 1024 * 10 * 25 ~ 250k ops. 
    # 10k evals * 250k ops = 2.5e9 ops. In Python/NumPy this might be slow if not fully vectorized.
    # My `_compute_expected_cut` is vectorized over states, but loops over edges and qubits.
    # Let's try a smaller grid or rely on multi-start local optimization if grid is too slow.
    
    # Let's estimate speed with a small test
    test_params = [0.3, 0.3, 0.1, 0.1]
    t0 = time.time()
    _compute_expected_cut(n, edges, test_params[:2], test_params[2:], states, bits)
    t1 = time.time()
    eval_time = t1 - t0
    
    # If eval is fast (< 1ms), we can do a dense grid.
    # If eval is ~1ms, 10k evals take 10s. Feasible.
    
    best_val = -np.inf
    best_params = None
    
    # Grid Search
    # Use 10 points per dimension for 10000 total evaluations
    # Ranges: Gamma [0.05, 0.7], Beta [0.02, 0.45] - slightly wider to capture more
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
                    val = _compute_expected_cut(n, edges, [g1, g2], [b1, b2], states, bits)
                    if val > best_val:
                        best_val = val
                        best_params = np.array([g1, b1, g2, b2])
                if time.time() > _DEADLINE - 15.0: break
            if time.time() > _DEADLINE - 15.0: break
        if time.time() > _DEADLINE - 15.0: break
        
    if best_params is None:
        # Fallback if grid didn't run or failed
        best_params = np.array([0.3, 0.1, 0.3, 0.1])
        best_val = _compute_expected_cut(n, edges, [0.3, 0.3], [0.1, 0.1], states, bits)
        
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
