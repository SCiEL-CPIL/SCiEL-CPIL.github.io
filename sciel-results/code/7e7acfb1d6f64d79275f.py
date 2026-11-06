"""QAOA angle design: graph-dependent two-layer QAOA for MaxCut."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import itertools

# Deadline guard
_START_TIME = time.time()
_DEADLINE = 150.0  # seconds
_MARGIN = 5.0      # safety margin


def _build_cost_matrix(n, edges):
    """Build the MaxCut cost matrix C where C[i,j] = 1 if (i,j) in edges else 0."""
    C = np.zeros((n, n))
    for u, v in edges:
        C[u, v] = 1.0
        C[v, u] = 1.0
    return C


def _expected_cut(n, C, gamma, beta, p):
    """
    Compute the expected cut value for a p-layer QAOA circuit.
    
    For p=2, we compute <Z_i Z_j> for each edge using the QAOA state.
    The expected cut = (1/2) * sum_{(i,j) in E} <Z_i Z_j>
    
    For a p-layer QAOA starting from |+>^n:
    - Layer k applies exp(-i gamma_k * C) then exp(-i beta_k * sum X)
    
    We use the fact that for the X-rotation, each qubit's state can be tracked
    via the rotation angles. For p=2, the state after both layers can be computed
    using the parameterized rotations.
    
    Actually, for a general graph, we need to track the full state or use
    the tensor-network / matrix approach. For n<=10, we can use the full
    state vector approach or the Pauli string expectation approach.
    
    For small n (<=10), let's use the full state vector simulation.
    """
    # For p=2 QAOA:
    # |psi> = exp(-i beta_2 sum X) exp(-i gamma_2 C) exp(-i beta_1 sum X) exp(-i gamma_1 C) |+>^n
    #
    # We compute <Z_i Z_j> for each edge (i,j).
    #
    # For n up to 10, full state vector is 2^10 = 1024 complex numbers - feasible.
    
    dim = 1 << n
    
    # Initial state: |+>^n
    psi = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    # Precompute C matrix for fast application
    # C = sum_{(i,j) in E} Z_i Z_j (unweighted MaxCut)
    
    def apply_C(psi):
        """Apply exp(-i gamma * C) where C = sum_{(i,j) in E} Z_i Z_j."""
        # For small n, we can diagonalize or use the fact that C is diagonal in Z-basis
        # C is diagonal in the computational basis: C|b> = (sum_{(i,j) in E} (-1)^{b_i+b_j})|b>
        # So exp(-i gamma * C)|b> = exp(-i gamma * c_b)|b> where c_b = sum_{(i,j) in E} (-1)^{b_i+b_j}
        result = np.zeros_like(psi)
        for b in range(dim):
            # Compute c_b
            c_b = 0
            for u, v in edges:
                bit_u = (b >> u) & 1
                bit_v = (b >> v) & 1
                if bit_u == bit_v:
                    c_b += 1
                else:
                    c_b -= 1
            result[b] = psi[b] * np.exp(-1j * gamma * c_b)
        return result
    
    def apply_X_rotation(psi, beta):
        """Apply exp(-i beta * sum X) = (exp(-i beta X))^{\otimes n}."""
        # exp(-i beta X) on a single qubit:
        # cos(beta) I - i sin(beta) X
        # So for each qubit, the rotation is:
        # |0> -> cos(beta)|0> - i sin(beta)|1>
        # |1> -> -i sin(beta)|0> + cos(beta)|1>
        #
        # This is a tensor product of single-qubit rotations.
        # We can apply it qubit by qubit.
        
        cb = np.cos(beta)
        sb = np.sin(beta)
        
        for q in range(n):
            # Apply the rotation on qubit q
            # The rotation matrix is [[cb, -i*sb], [-i*sb, cb]]
            # For each pair of basis states differing only in bit q:
            # new[b] = cb * old[b] - i*sb * old[b ^ (1<<q)]
            # new[b ^ (1<<q)] = -i*sb * old[b] + cb * old[b ^ (1<<q)]
            new_psi = np.zeros_like(psi)
            for b in range(dim):
                if (b >> q) & 1 == 0:
                    b2 = b | (1 << q)
                    new_psi[b] = cb * psi[b] - 1j * sb * psi[b2]
                    new_psi[b2] = -1j * sb * psi[b] + cb * psi[b2]
            psi = new_psi
        
        return psi
    
    # Apply layers in order: gamma_1, beta_1, gamma_2, beta_2, ...
    for k in range(p):
        # Apply exp(-i gamma_k * C)
        psi = apply_C(psi)
        # Apply exp(-i beta_k * sum X)
        psi = apply_X_rotation(psi, beta)
    
    # Compute expected cut
    # <Z_i Z_j> = sum_b (-1)^{b_i+b_j} |psi_b|^2
    probs = np.abs(psi) ** 2
    expected = 0.0
    for u, v in edges:
        # <Z_u Z_v>
        zz = 0.0
        for b in range(dim):
            bit_u = (b >> u) & 1
            bit_v = (b >> v) & 1
            zz += ((-1) ** (bit_u + bit_v)) * probs[b]
        expected += zz / 2.0
    
    return expected


def _expected_cut_fast(n, C, edges, params):
    """
    Faster expected cut computation using the structure of QAOA.
    params: array of shape (p, 2) with [gamma_k, beta_k]
    """
    p = len(params)
    dim = 1 << n
    
    # Initial state
    psi = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    # Precompute c_b for all basis states
    c_vals = np.zeros(dim)
    for b in range(dim):
        c_b = 0
        for u, v in edges:
            bit_u = (b >> u) & 1
            bit_v = (b >> v) & 1
            if bit_u == bit_v:
                c_b += 1
            else:
                c_b -= 1
        c_vals[b] = c_b
    
    # Precompute phase factors for each gamma
    # But gamma changes per layer, so we compute on the fly
    
    def apply_C_fast(psi, gamma):
        phases = np.exp(-1j * gamma * c_vals)
        return psi * phases
    
    def apply_X_rotation_fast(psi, beta):
        cb = np.cos(beta)
        sb = np.sin(beta)
        new_psi = np.zeros_like(psi)
        for q in range(n):
            mask = 1 << q
            for b in range(dim):
                if (b & mask) == 0:
                    b2 = b | mask
                    new_psi[b] = cb * psi[b] - 1j * sb * psi[b2]
                    new_psi[b2] = -1j * sb * psi[b] + cb * psi[b2]
            psi = new_psi
        return psi
    
    for k in range(p):
        gamma_k = params[k, 0]
        beta_k = params[k, 1]
        psi = apply_C_fast(psi, gamma_k)
        psi = apply_X_rotation_fast(psi, beta_k)
    
    probs = np.abs(psi) ** 2
    expected = 0.0
    for u, v in edges:
        zz = 0.0
        for b in range(dim):
            bit_u = (b >> u) & 1
            bit_v = (b >> v) & 1
            zz += ((-1) ** (bit_u + bit_v)) * probs[b]
        expected += zz / 2.0
    
    return expected


def _exact_maxcut(n, edges):
    """Brute-force exact MaxCut for small n."""
    dim = 1 << n
    best = 0
    for b in range(dim):
        cut = 0
        for u, v in edges:
            bit_u = (b >> u) & 1
            bit_v = (b >> v) & 1
            if bit_u != bit_v:
                cut += 1
        if cut > best:
            best = cut
    return best


def _objective(params, n, edges):
    """Negative expected cut (for minimization)."""
    try:
        val = _expected_cut_fast(n, None, edges, params)
        return -val
    except Exception:
        return 1e10


def _optimize_for_graph(n, edges, p):
    """Find optimal QAOA angles for a given graph."""
    # Multiple random starts
    best_params = np.array([0.35, 0.25])  # fallback
    best_val = -1e10
    
    # Graph density-based initial ranges
    m = len(edges)
    max_edges = n * (n - 1) // 2
    density = m / max_edges if max_edges > 0 else 0
    
    # Scale angle ranges based on density
    gamma_range = (0.1, 2.0)
    beta_range = (0.05, 1.5)
    
    n_starts = 8
    rng = np.random.RandomState(42)
    
    for i in range(n_starts):
        if time.time() > _DEADLINE - _MARGIN:
            break
        
        if i == 0:
            x0 = np.array([0.35, 0.25])
        else:
            x0 = np.array([
                rng.uniform(gamma_range[0], gamma_range[1]),
                rng.uniform(beta_range[0], beta_range[1])
            ])
        
        result = minimize(
            _objective,
            x0,
            args=(n, edges),
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
        )
        
        if result.fun < -best_val:
            best_val = -result.fun
            best_params = result.x
    
    return best_params, best_val


def angles(n, edges, p):
    """
    Design graph-dependent two-layer QAOA angles.
    
    Args:
        n: number of vertices
        edges: list of (u, v) tuples
        p: number of QAOA layers (fixed at 2)
    
    Returns:
        numpy array of shape (p, 2) with rows [gamma, beta] in radians
    """
    global _START_TIME
    
    # Ensure we have a valid edge list
    edges = list(edges)
    
    if n <= 2 or len(edges) == 0:
        return np.tile([0.35, 0.25], (p, 1))
    
    # Check deadline
    if time.time() > _DEADLINE - _MARGIN:
        return np.tile([0.35, 0.25], (p, 1))
    
    try:
        params, val = _optimize_for_graph(n, edges, p)
    except Exception:
        params = np.tile([0.35, 0.25], (p, 1))
    
    # Validate output
    if params is None or len(params) != p or len(params[0]) != 2:
        return np.tile([0.35, 0.25], (p, 1))
    
    return np.array(params, dtype=np.float64)
# EVOLVE-BLOCK-END
