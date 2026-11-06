"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import expm
import time

# Cache for optimal angles per graph
_angle_cache = {}
_T0 = time.time()
_DEADLINE = 150.0
_SAFETY_MARGIN = 12.0

def _get_remaining_time():
    return _DEADLINE - _SAFETY_MARGIN - (time.time() - _T0)

def _build_c_diagonal(n, edges):
    """
    Compute the diagonal values of the MaxCut Hamiltonian C.
    C(b) = number of edges crossing the cut defined by b.
    """
    diag = np.zeros(2**n)
    for mask in range(2**n):
        cut_val = 0
        for u, v in edges:
            if ((mask >> u) & 1) != ((mask >> v) & 1):
                cut_val += 1
        diag[mask] = cut_val
    return diag

def _build_mixer_matrix(n):
    """
    Build the mixer matrix M = sum_j X_j (as a dense matrix).
    X_j flips bit j in the computational basis.
    """
    dim = 2**n
    M = np.zeros((dim, dim), dtype=complex)
    for state in range(dim):
        for j in range(n):
            flipped = state ^ (1 << j)
            M[state, flipped] = 1.0
    return M

def _expected_cut_fast(n, edges, gammas, betas, c_diag, mixer_matrix):
    """
    Compute expected cut using precomputed matrices.
    Uses matrix-vector products with precomputed cost phases.
    """
    p = len(gammas)
    dim = 2**n
    
    # Precompute cost phase factors for each gamma
    # U_cost(gamma) = diag(exp(-i * gamma * c_diag))
    
    # Initial state |+>^n
    state = np.ones(dim, dtype=complex) / np.sqrt(dim)
    
    for k in range(p):
        # Apply Cost: multiply by exp(-i * gamma_k * c_diag)
        phase = np.exp(-1j * gammas[k] * c_diag)
        state = state * phase
        
        # Apply Mixer: state = exp(-i * beta_k * M) @ state
        # For efficiency, use the fact that M has a simple structure
        # M = sum_j X_j, so exp(-i beta M) can be applied efficiently
        # But for correctness, let's use matrix exponential if dim is small
        if dim <= 1024:
            U_mixer = expm(-1j * betas[k] * mixer_matrix)
            state = U_mixer @ state
        else:
            # Fallback: apply single-qubit gates
            c = np.cos(betas[k])
            s = -1j * np.sin(betas[k])
            state_tensor = state.reshape([2] * n)
            gate = np.array([[c, s], [s, c]], dtype=complex)
            for j in range(n):
                state_tensor = np.moveaxis(state_tensor, j, 0)
                orig_shape = state_tensor.shape
                state_tensor = state_tensor.reshape(-1, 2)
                state_tensor = gate @ state_tensor
                state_tensor = state_tensor.reshape(orig_shape)
                state_tensor = np.moveaxis(state_tensor, 0, j)
            state = state_tensor.reshape(-1)
    
    # Compute expectation <psi| C |psi>
    prob = np.abs(state)**2
    return float(np.dot(prob, c_diag))

def _optimize_angles(n, edges, p):
    """
    Optimize angles for given graph.
    """
    c_diag = _build_c_diagonal(n, edges)
    mixer_matrix = _build_mixer_matrix(n)
    
    dim = 2**n
    use_mexp = dim <= 1024
    
    # Generate diverse initial guesses
    starts = []
    
    # Heuristic-based starts
    heuristic_starts = [
        [0.35, 0.25],
        [0.5, 0.2],
        [1.0, 0.5],
        [np.pi/4, np.pi/8],
        [0.2, 0.4],
        [0.8, 0.3],
        [0.6, 0.15],
        [np.pi/6, np.pi/12],
        [0.4, 0.3],
        [0.7, 0.25],
        [0.9, 0.1],
        [0.15, 0.6],
        [1.2, 0.4],
        [0.25, 0.35],
        [0.55, 0.15],
    ]
    
    for gs, bs in heuristic_starts:
        starts.append([gs] * p + [bs] * p)
    
    # Random starts with fixed seed for reproducibility
    rng = np.random.RandomState(123)
    for _ in range(8):
        g_rand = rng.uniform(0.05, 1.5, p)
        b_rand = rng.uniform(0.05, 0.8, p)
        starts.append(np.concatenate([g_rand, b_rand]).tolist())
    
    # Also try scaled versions
    for scale in [0.5, 1.5, 2.0]:
        starts.append([0.35 * scale] * p + [0.25 * scale] * p)
    
    best_val = -np.inf
    best_params = None
    best_angles = np.tile([0.35, 0.25], (p, 1))
    
    # Objective function to minimize (negative expected cut)
    def obj(params):
        gammas = params[:p]
        betas = params[p:]
        val = _expected_cut_fast(n, edges, gammas, betas, c_diag, mixer_matrix)
        return -val
    
    # Bounds
    bounds = [(0, np.pi)] * (2 * p)
    
    # Track evaluation count for time management
    eval_count = 0
    
    for idx, start in enumerate(starts):
        # Check time - be conservative
        remaining = _get_remaining_time()
        if remaining < 5.0:
            break
        
        # Estimate remaining work
        num_starts_remaining = len(starts) - idx
        if num_starts_remaining > 0:
            time_per_start_estimate = remaining / (num_starts_remaining + 1)
            if time_per_start_estimate < 1.0:
                break
        
        try:
            res = minimize(obj, start, method='Nelder-Mead', 
                           options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6,
                                   'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
                gammas = best_params[:p]
                betas = best_params[p:]
                best_angles = np.column_stack((gammas, betas))
        except Exception:
            continue
    
    if best_params is None:
        # Fallback to seed
        return np.tile([0.35, 0.25], (p, 1))
        
    return best_angles

def angles(n, edges, p):
    # Create a cache key
    edges_tuple = tuple(sorted((u, v) if u < v else (v, u) for u, v in edges))
    key = (n, edges_tuple, p)
    
    if key in _angle_cache:
        return _angle_cache[key]
        
    # Check time
    if _get_remaining_time() < 2.0:
        # Not enough time to optimize, return default
        res = np.tile([0.35, 0.25], (p, 1))
        return res
        
    res = _optimize_angles(n, edges, p)
    
    # Store in cache
    _angle_cache[key] = res
    
    return res
# EVOLVE-BLOCK-END
