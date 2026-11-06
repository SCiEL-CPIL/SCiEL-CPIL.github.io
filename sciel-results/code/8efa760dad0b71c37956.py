"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

# Cache for optimal angles per graph
_angle_cache = {}
_T0 = time.time()
_DEADLINE = 150.0
_SAFETY_MARGIN = 10.0

def _get_remaining_time():
    return _DEADLINE - _SAFETY_MARGIN - (time.time() - _T0)

def _build_c_diagonal(n, edges):
    """
    Compute the diagonal values of the MaxCut Hamiltonian C.
    C = sum_{(u,v) in E} 0.5 * (1 - Z_u Z_v)
    For a bitstring b, Z_u = 1 if b_u=0, -1 if b_u=1.
    Z_u Z_v = 1 if b_u == b_v, -1 if b_u != b_v.
    Term = 0.5 * (1 - (1 if same else -1))
    If same: 0.5 * (1 - 1) = 0
    If diff: 0.5 * (1 - (-1)) = 1
    So C(b) = number of edges crossing the cut defined by b.
    """
    diag = np.zeros(2**n)
    for mask in range(2**n):
        cut_val = 0
        for u, v in edges:
            # Check if bits u and v are different
            if ((mask >> u) & 1) != ((mask >> v) & 1):
                cut_val += 1
        diag[mask] = cut_val
    return diag

def _apply_mixer(state, beta, n):
    """
    Apply exp(-i beta sum X) to state.
    sum X = sum_j X_j.
    exp(-i beta sum X) = prod_j (cos(beta) I_j - i sin(beta) X_j).
    This is a tensor product of single-qubit gates.
    We can apply it efficiently or build the matrix.
    For n <= 10, building the matrix is fine, but applying sequentially is faster.
    However, the operator is global (sum), so it's a product of local gates.
    Gate on qubit j: [[cos, -i sin], [-i sin, cos]].
    """
    c = np.cos(beta)
    s = -1j * np.sin(beta)
    # The gate is U_j = [[c, s], [s, c]]
    # We apply this gate to each qubit j in the state.
    # State is a vector of length 2^n.
    # We can reshape to (2, 2, ..., 2) tensor and apply matrix multiplication.
    # Or simply iterate.
    # Let's use the tensor approach for clarity and speed.
    state_tensor = state.reshape([2] * n)
    # The gate is applied to the j-th axis.
    # Note: numpy axes order. If state is indexed by bits b_0...b_{n-1},
    # and we reshape to [2]*n, axis 0 corresponds to b_0 (LSB? or MSB?).
    # Standard: index = sum b_k 2^k.
    # Reshape [2]*n: index [b_0, b_1, ..., b_{n-1}] corresponds to flat index.
    # So axis k corresponds to bit k.
    for j in range(n):
        # Matrix [[c, s], [s, c]]
        # state_tensor[..., j, ...]
        # We need to contract along axis j.
        # new_state[..., j, ...] = c * old[..., j, ...] + s * old[..., 1-j, ...]
        # It's easier to use np.einsum or reshape.
        pass 
    
    # Alternative: Build the full unitary matrix for sum X.
    # For n=10, 1024x1024 complex matrix is ~8MB. Multiplication is fast.
    # Let's build the matrix once per beta? No, beta changes.
    # But we can compute the action directly.
    # Let's stick to the tensor method but implement it correctly.
    
    # Actually, since it's a sum of X, it's a product of identical single-qubit gates.
    # U = U_1 \otimes U_2 ... \otimes U_n
    # We can apply U to each qubit sequentially.
    
    # Let's use a helper to apply a single qubit gate to a specific axis.
    def apply_gate_to_axis(state, axis, gate):
        # gate is 2x2
        # Move axis to front
        state = np.moveaxis(state, axis, 0)
        # Reshape to (-1, 2)
        orig_shape = state.shape
        state = state.reshape(-1, 2)
        # Apply gate: state_new = gate @ state
        state = gate @ state
        # Reshape back
        state = state.reshape(orig_shape)
        # Move axis back
        state = np.moveaxis(state, 0, axis)
        return state

    gate = np.array([[c, s], [s, c]], dtype=complex)
    for j in range(n):
        state = apply_gate_to_axis(state_tensor, j, gate)
        
    return state.reshape(-1)

def _apply_cost(state, gamma, c_diag):
    """
    Apply exp(-i gamma C).
    C is diagonal with values c_diag.
    So multiply state by exp(-i gamma c_diag).
    """
    phase = np.exp(-1j * gamma * c_diag)
    return state * phase

def _expected_cut_value(n, edges, gammas, betas):
    """
    Compute expected cut for given parameters.
    gammas, betas are arrays of length p.
    Order: 
    Start |+>
    For k in range(p):
       Apply exp(-i gamma[k] C)
       Apply exp(-i beta[k] sum X)
    """
    p = len(gammas)
    c_diag = _build_c_diagonal(n, edges)
    # Initial state |+>^n
    state = np.ones(2**n, dtype=complex) / np.sqrt(2**n)
    
    for k in range(p):
        # Apply Cost
        state = _apply_cost(state, gammas[k], c_diag)
        # Apply Mixer
        state = _apply_mixer(state, betas[k], n)
        
    # Compute expectation <psi| C |psi>
    # C is diagonal, so <psi| C |psi> = sum_i |psi_i|^2 * C_ii
    prob = np.abs(state)**2
    return np.dot(prob, c_diag)

def _optimize_angles(n, edges, p):
    """
    Optimize angles for given graph.
    """
    # Initial guess
    # Heuristics: gamma ~ pi/4, beta ~ pi/8?
    # Let's use a few random starts or a grid.
    best_val = -1
    best_params = None
    
    # Bounds
    bounds = [(0, np.pi)] * (2 * p)
    
    # Objective function to minimize (negative expected cut)
    def obj(params):
        gammas = params[:p]
        betas = params[p:]
        val = _expected_cut_value(n, edges, gammas, betas)
        return -val

    # Try a few starting points
    starts = [
        [0.35, 0.25] * p, # Seed values
        [0.5, 0.2] * p,
        [1.0, 0.5] * p,
        [np.pi/4, np.pi/8] * p,
    ]
    
    # Add random starts if time permits? 
    # For now, fixed starts + Nelder-Mead.
    
    for start in starts:
        # Check time
        if _get_remaining_time() < 5.0:
            break
            
        try:
            res = minimize(obj, start, method='Nelder-Mead', 
                           options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-4})
            if res.fun < best_val: # Note: res.fun is negative expected cut
                # We want to maximize expected cut, so minimize -expected cut.
                # So lower res.fun is better.
                # Wait, best_val initialized to -1.
                # If res.fun is e.g. -5, it's better than -1? No, -5 < -1.
                # We want the minimum of -ExpectedCut.
                # So we keep the result with the lowest fun.
                if best_val == -1 or res.fun < best_val:
                    best_val = res.fun
                    best_params = res.x
        except Exception:
            continue

    if best_params is None:
        # Fallback to seed
        return np.tile([0.35, 0.25], (p, 1))
        
    # Extract gammas and betas
    gammas = best_params[:p]
    betas = best_params[p:]
    
    # Format output: (p, 2) array, rows (gamma, beta)
    angles = np.column_stack((gammas, betas))
    return angles

def angles(n, edges, p):
    # Create a cache key
    edges_tuple = tuple(sorted((u, v) if u < v else (v, u) for u, v in edges))
    key = (n, edges_tuple)
    
    if key in _angle_cache:
        return _angle_cache[key]
        
    # Check time
    if _get_remaining_time() < 1.0:
        # Not enough time to optimize, return default
        res = np.tile([0.35, 0.25], (p, 1))
        return res
        
    res = _optimize_angles(n, edges, p)
    
    # Store in cache
    _angle_cache[key] = res
    
    return res
# EVOLVE-BLOCK-END
