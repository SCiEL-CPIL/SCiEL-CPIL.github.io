"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

# Cache for exact maxcut values
_exact_maxcut_cache = {}
# Cache for best angles found per graph signature
_angles_cache = {}

def _graph_signature(n, edges):
    """Create a hashable signature for the graph."""
    edge_set = frozenset(frozenset((int(u), int(v))) for u, v in edges)
    return (n, edge_set)

def _compute_edge_bitmasks(n, edges):
    """
    Precompute bitmasks for each edge. For edge (u,v), the mask has bits u and v set.
    This allows us to check if an edge is cut by a given bitstring.
    """
    masks = []
    for u, v in edges:
        mask = (1 << int(u)) | (1 << int(v))
        masks.append(mask)
    return masks

def _cost_vector(n, edges, edge_masks):
    """
    Compute the cost vector c[x] = sum of edges cut by bitstring x.
    For MaxCut, the cost operator C is diagonal with entries c[x].
    """
    m = 1 << n
    c = np.zeros(m, dtype=np.float64)
    for x in range(m):
        cut = 0
        for mask in edge_masks:
            # Edge is cut if x_u != x_v, i.e., (x & mask) has exactly one bit set
            val = x & mask
            # Check if val has exactly one bit set
            if val & (val - 1) == 0:
                cut += 1
        c[x] = cut
    return c

def _exact_maxcut(n, edges):
    """Compute the exact maximum cut value for the graph."""
    sig = _graph_signature(n, edges)
    if sig in _exact_maxcut_cache:
        return _exact_maxcut_cache[sig]
    
    edge_masks = _compute_edge_bitmasks(n, edges)
    c = _cost_vector(n, edges, edge_masks)
    opt = float(np.max(c))
    _exact_maxcut_cache[sig] = opt
    return opt

def _simulate_qaoa(n, edges, p, gammas, betas):
    """
    Simulate the QAOA circuit and return the expected cut value.
    Uses exact state vector simulation.
    
    The circuit starts in |+>^n, then for each layer k:
      1. Apply exp(-i gamma_k C) where C is the MaxCut cost operator (diagonal)
      2. Apply exp(-i beta_k sum_i X_i) (mixer)
    
    Returns the expected value of the MaxCut cost operator.
    """
    m = 1 << n
    edge_masks = _compute_edge_bitmasks(n, edges)
    c = _cost_vector(n, edges, edge_masks)
    
    # Initial state: |+>^n = (1/sqrt(2^n)) * [1, 1, ..., 1]
    psi = np.ones(m, dtype=np.complex128) / np.sqrt(m)
    
    for k in range(p):
        # Apply cost phase: psi[x] *= exp(-i gamma_k * c[x])
        psi *= np.exp(-1j * gammas[k] * c)
        
        # Apply mixer: exp(-i beta * sum_i X_i)
        # This is a tensor product of single-qubit rotations:
        # exp(-i beta X) = cos(beta) I - i sin(beta) X
        # For n qubits, this is a Kronecker product.
        # We can apply this efficiently by iterating over each qubit.
        beta = betas[k]
        cb = np.cos(beta)
        sb = np.sin(beta)
        
        for q in range(n):
            # Apply single-qubit mixer on qubit q
            # The operation is: for each pair of basis states differing only in bit q,
            # new_psi[a] = cb * psi[a] - i*sb * psi[b]
            # new_psi[b] = -i*sb * psi[a] + cb * psi[b]
            # where a and b differ in bit q.
            bit = 1 << q
            # Create indices
            # We need to pair up basis states that differ in bit q
            # Efficient approach: iterate over all states where bit q is 0
            mask_no_bit = m - bit  # not quite, let's use a different approach
            # Actually, let's just do it with numpy operations
            # For each state x, if bit q is 0, partner is x | bit
            # if bit q is 1, partner is x & ~bit
            
            # Vectorized approach:
            # Create an array of all indices
            idx = np.arange(m)
            # Indices where bit q is 0
            mask_zero = (idx & bit) == 0
            idx_zero = idx[mask_zero]
            idx_one = idx_zero | bit
            
            # Get the values
            psi_zero = psi[idx_zero].copy()
            psi_one = psi[idx_one].copy()
            
            # Apply the rotation
            new_psi_zero = cb * psi_zero - 1j * sb * psi_one
            new_psi_one = -1j * sb * psi_zero + cb * psi_one
            
            psi[idx_zero] = new_psi_zero
            psi[idx_one] = new_psi_one
    
    # Compute expected cut: <psi| C |psi> = sum_x |psi[x]|^2 * c[x]
    probs = np.abs(psi) ** 2
    expected_cut = np.dot(probs, c)
    return expected_cut

def _objective_and_grad(params, n, edges, p, c, edge_masks):
    """
    Objective function (negative expected cut) and gradient for L-BFGS-B.
    params is a flat array of length 2p: [gamma_0, beta_0, gamma_1, beta_1, ...]
    """
    gammas = params[0::2]
    betas = params[1::2]
    
    m = 1 << n
    
    # Simulate and get expected cut
    # For gradient, we use finite differences or analytical
    # Let's use the parameter-shift rule for QAOA
    
    # Actually, let's just compute the value and use numerical gradient
    # for simplicity, but that's slow. Let's do analytical.
    
    # The gradient of <psi|C|psi> with respect to gamma_k and beta_k
    # can be computed using the parameter-shift rule.
    
    # For now, let's just return the negative expected cut and let scipy
    # use numerical gradient. This is slower but more reliable.
    
    expected_cut = _simulate_qaoa(n, edges, p, gammas, betas)
    return -expected_cut

def _objective(params, n, edges, p):
    """Negative expected cut for minimization."""
    gammas = params[0::2]
    betas = params[1::2]
    expected_cut = _simulate_qaoa(n, edges, p, gammas, betas)
    return -expected_cut

def angles(n, edges, p):
    """
    Design graph-dependent two-layer QAOA angles maximizing the mean ratio
    of the independently simulated expected cut to the exact classical optimum.
    
    Returns: real array (p, 2), rows (gamma, beta) in radians.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s
    
    sig = _graph_signature(n, edges)
    
    # Check cache
    if sig in _angles_cache:
        return _angles_cache[sig]
    
    # Compute exact maxcut
    opt_cut = _exact_maxcut(n, edges)
    
    if opt_cut == 0:
        # No edges, any angles work
        result = np.zeros((p, 2), dtype=np.float64)
        _angles_cache[sig] = result
        return result
    
    # Precompute cost vector and edge masks
    edge_masks = _compute_edge_bitmasks(n, edges)
    c = _cost_vector(n, edges, edge_masks)
    
    # Multi-start L-BFGS-B optimization
    best_params = None
    best_value = -np.inf
    
    # Initial guesses
    initial_guesses = [
        np.array([0.35, 0.25] * p, dtype=np.float64),  # Seed program
        np.array([0.5, 0.3] * p, dtype=np.float64),
        np.array([0.25, 0.15] * p, dtype=np.float64),
        np.array([0.7, 0.4] * p, dtype=np.float64),
        np.array([0.1, 0.1] * p, dtype=np.float64),
        np.array([0.9, 0.5] * p, dtype=np.float64),
    ]
    
    # Add some random starts
    np.random.seed(42)
    for _ in range(4):
        initial_guesses.append(np.random.uniform(0, 1.5, size=2*p))
    
    for guess in initial_guesses:
        if time.time() > deadline:
            break
        
        try:
            result = minimize(
                _objective,
                guess,
                args=(n, edges, p),
                method='L-BFGS-B',
                bounds=[(0, np.pi)] * (2 * p),
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            val = result.fun
            if val < best_value:
                best_value = val
                best_params = result.x.copy()
                
        except Exception:
            continue
    
    if best_params is None:
        # Fallback to seed program
        result = np.tile([0.35, 0.25], (p, 1))
    else:
        # Reshape to (p, 2)
        gammas = best_params[0::2]
        betas = best_params[1::2]
        result = np.column_stack((gammas, betas))
    
    _angles_cache[sig] = result
    return result
# EVOLVE-BLOCK-END
