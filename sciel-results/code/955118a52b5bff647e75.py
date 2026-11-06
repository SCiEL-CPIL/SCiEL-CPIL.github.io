import numpy as np
from scipy.optimize import minimize
import time
import hashlib
import itertools

# Cache for optimized angles keyed by graph structure
_angle_cache = {}
_start_time = None
_TIME_LIMIT = 140.0  # Safety margin before 150s kill

def _get_start_time():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return _start_time

def _time_left():
    return _TIME_LIMIT - (time.time() - _get_start_time())

def _graph_key(n, edges):
    """Create a hashable key for the graph structure."""
    edge_set = frozenset(tuple(sorted(e)) for e in edges)
    return (n, edge_set)

def _exact_maxcut(n, edges):
    """Compute exact MaxCut value by brute force."""
    if not edges:
        return 0.0
    best = 0
    for mask in range(1 << n):
        cut = 0
        for (u, v) in edges:
            if ((mask >> u) & 1) != ((mask >> v) & 1):
                cut += 1
        if cut > best:
            best = cut
    return float(best)

def _expected_cut(n, edges, gammas, betas):
    """
    Compute expected cut value for QAOA with p=2.
    gammas = [gamma1, gamma2], betas = [beta1, beta2]
    State evolution: |+>^n -> exp(-i g1 C) -> exp(-i b1 B) -> exp(-i g2 C) -> exp(-i b2 B)
    C = sum_{(i,j) in E} (I - Z_i Z_j)/2
    B = sum_i X_i
    """
    dim = 1 << n
    # Start in |+>^n
    psi = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    # Precompute cost Hamiltonian matrix (or apply it efficiently)
    # For small n, we can build the full matrix
    C_mat = np.zeros((dim, dim), dtype=np.complex128)
    for (u, v) in edges:
        # (I - Z_u Z_v) / 2
        # Z_u Z_v: diagonal with +1 if bits u,v same, -1 if different
        for state in range(dim):
            bu = (state >> u) & 1
            bv = (state >> v) & 1
            zz = 1.0 if bu == bv else -1.0
            C_mat[state, state] += (1.0 - zz) / 2.0
    
    # Mixer: B = sum X_i, which is a sum of bit-flip operators
    # Apply exp(-i * beta * B) = product of exp(-i * beta * X_i) since X_i commute
    # exp(-i * beta * X) = cos(beta) * I - i * sin(beta) * X
    # So we can apply each single-qubit rotation separately
    
    def apply_mixer(psi, beta):
        """Apply exp(-i * beta * sum X_i) to state."""
        # Since X_i operators commute, exp(-i beta sum X_i) = prod_i exp(-i beta X_i)
        # exp(-i beta X_i) = cos(beta) I_i - i sin(beta) X_i
        # This acts on qubit i: for each pair of states differing only in bit i,
        # [a, b] -> [cos(beta)*a - i*sin(beta)*b, -i*sin(beta)*a + cos(beta)*b]
        for i in range(n):
            stride = 1 << i
            for base in range(0, dim, stride * 2):
                for offset in range(stride):
                    idx0 = base + offset
                    idx1 = idx0 + stride
                    a = psi[idx0]
                    b = psi[idx1]
                    cb = np.cos(beta)
                    sb = np.sin(beta)
                    psi[idx0] = cb * a - 1j * sb * b
                    psi[idx1] = -1j * sb * a + cb * b
        return psi
    
    def apply_cost(psi, gamma):
        """Apply exp(-i * gamma * C) to state."""
        # For small n, use matrix exponential or direct matrix multiplication
        # exp(-i gamma C) @ psi
        # Since C is diagonal (it's a sum of diagonal operators), exp(-i gamma C) is also diagonal
        # exp(-i gamma * c_state) where c_state is the eigenvalue of C for that state
        # c_state = sum over edges (1 - z_u z_v)/2
        result = np.zeros(dim, dtype=np.complex128)
        for state in range(dim):
            c_val = C_mat[state, state].real  # C is diagonal
            result[state] = psi[state] * np.exp(-1j * gamma * c_val)
        return result
    
    # Apply the QAOA circuit
    psi = apply_cost(psi, gammas[0])
    psi = apply_mixer(psi, betas[0])
    psi = apply_cost(psi, gammas[1])
    psi = apply_mixer(psi, betas[1])
    
    # Expected cut value = <psi| C |psi>
    # Since C is diagonal: sum_state |psi[state]|^2 * C_mat[state,state]
    probs = np.abs(psi) ** 2
    expected = np.sum(probs * C_mat[np.arange(dim), np.arange(dim)].real)
    return float(expected)

def _optimize_angles(n, edges, max_iters=500):
    """Optimize 4 QAOA parameters for a given graph."""
    best_val = _exact_maxcut(n, edges)
    
    def neg_objective(params):
        g1, b1, g2, b2 = params
        val = _expected_cut(n, edges, [g1, g2], [b1, b2])
        return -val
    
    # Multi-start optimization
    best_params = None
    best_score = -np.inf
    
    # Initial guesses
    initial_guesses = [
        [0.35, 0.25, 0.35, 0.25],  # Seed values
        [0.5, 0.3, 0.5, 0.3],
        [0.2, 0.4, 0.2, 0.4],
        [0.6, 0.2, 0.4, 0.3],
        [0.4, 0.5, 0.3, 0.4],
        [0.1, 0.1, 0.1, 0.1],
        [0.8, 0.5, 0.8, 0.5],
        [0.3, 0.3, 0.5, 0.2],
    ]
    
    for guess in initial_guesses:
        try:
            res = minimize(neg_objective, guess, method='Nelder-Mead',
                          options={'maxiter': max_iters, 'xatol': 1e-6, 'fatol': 1e-8})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception:
            pass
    
    if best_params is None:
        # Fallback to seed values
        return np.array([[0.35, 0.25], [0.35, 0.25]])
    
    g1, b1, g2, b2 = best_params
    return np.array([[g1, b1], [g2, b2]])

def angles(n, edges, p):
    """
    Design graph-dependent two-layer QAOA angles.
    
    Args:
        n: number of vertices
        edges: list of edges (u, v)
        p: number of QAOA layers (must be 2)
    
    Returns:
        numpy array of shape (p, 2) with rows (gamma, beta) in radians
    """
    _get_start_time()
    
    if p != 2:
        # Fallback for unexpected p
        return np.tile([0.35, 0.25], (p, 1))
    
    key = _graph_key(n, edges)
    
    if key in _angle_cache:
        return _angle_cache[key].copy()
    
    # Check time
    if _time_left() < 5.0:
        # Not enough time, return fallback
        result = np.array([[0.35, 0.25], [0.35, 0.25]])
        _angle_cache[key] = result
        return result.copy()
    
    try:
        result = _optimize_angles(n, edges)
        _angle_cache[key] = result
        return result.copy()
    except Exception:
        # Fallback on any error
        result = np.array([[0.35, 0.25], [0.35, 0.25]])
        _angle_cache[key] = result
        return result.copy()
