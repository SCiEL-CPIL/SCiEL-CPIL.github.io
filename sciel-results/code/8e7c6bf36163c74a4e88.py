"""QAOA parameter design for fixed-depth (p=2) MaxCut."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import itertools

# Deadline guard
_START_TIME = time.time()
_DEADLINE = 135.0  # seconds, leave 15s safety margin for 150s limit

def _remaining_time():
    """Return remaining time in seconds."""
    return _DEADLINE - (time.time() - _START_TIME)

def _exact_maxcut(n, edges):
    """Compute exact MaxCut value via brute force for small n."""
    m = len(edges)
    if m == 0:
        return 0
    best = 0
    for mask in range(1 << n):
        cut = 0
        for (u, v) in edges:
            if (mask >> u) & 1 != (mask >> v) & 1:
                cut += 1
        if cut > best:
            best = cut
    return best

def _build_graph_data(n, edges):
    """Precompute data needed for fast expected cut computation."""
    dim = 1 << n
    m = len(edges)
    
    # Precompute Z values for each qubit: Z_i[j] = +1 if bit i of j is 0, else -1
    Z = np.ones((n, dim), dtype=float)
    for i in range(n):
        for j in range(dim):
            if (j >> i) & 1:
                Z[i, j] = -1.0
    
    # Precompute edge ZZ products
    ZZ = np.zeros((m, dim), dtype=float)
    for k, (u, v) in enumerate(edges):
        ZZ[k] = Z[u] * Z[v]
    
    # Precompute flip masks for each edge
    flip_masks = []
    for (u, v) in edges:
        flip_masks.append((1 << u) | (1 << v))
    
    return dim, m, Z, ZZ, flip_masks

def _expected_cut_fast(n, edges, gammas, betas, graph_data=None):
    """
    Fast expected cut for p=2 QAOA using vectorized operations.
    
    Key insight: exp(-i*beta*X_sum) where X_sum = product of all X_i
    maps |j> to |j XOR (2^n - 1)>, so we can use bit-reversal.
    """
    if graph_data is None:
        graph_data = _build_graph_data(n, edges)
    dim, m, Z, ZZ, flip_masks = graph_data
    
    if m == 0:
        return 0.0
    
    state = np.ones(dim, dtype=complex) / np.sqrt(dim)
    all_ones = dim - 1
    
    for step in range(2):
        gamma = gammas[step]
        beta = betas[step]
        
        # Apply cost unitary: exp(-i*gamma*C) where C = 0.5 * sum_k (1 - ZZ_k)
        # = exp(-i*gamma*m/2) * prod_k exp(i*gamma*ZZ_k/2)
        phase = np.exp(-1j * gamma * 0.5 * m)
        for k in range(m):
            phase *= np.exp(1j * gamma * 0.5 * ZZ[k])
        state = phase * state
        
        # Apply mixer: exp(-i*beta*X_sum) = cos(beta)*I - i*sin(beta)*X_sum
        # X_sum |j> = |j XOR all_ones>
        state = np.cos(beta) * state - 1j * np.sin(beta) * state[all_ones ^ np.arange(dim)]
    
    # Compute expected cut: for each edge, E[Z_u Z_v] = sum_j conj(state[j]) * state[j XOR flip_mask]
    cut_val = 0.0
    arange = np.arange(dim)
    for fm in flip_masks:
        corr = np.sum(np.conj(state) * state[arange ^ fm]).real
        cut_val += 0.5 - 0.5 * corr
    
    return cut_val

def _expected_cut_gradient(n, edges, params, graph_data=None):
    """Compute gradient of expected cut w.r.t. [gamma1, beta1, gamma2, beta2] using finite differences."""
    if graph_data is None:
        graph_data = _build_graph_data(n, edges)
    
    dim, m, Z, ZZ, flip_masks = graph_data
    if m == 0:
        return np.zeros(4)
    
    gammas = np.array([params[0], params[2]])
    betas = np.array([params[1], params[3]])
    
    f0 = _expected_cut_fast(n, edges, gammas, betas, graph_data)
    
    grad = np.zeros(4)
    h = 1e-6
    
    # Gradient w.r.t. gamma1
    g1 = gammas.copy(); g1[0] += h
    grad[0] = (_expected_cut_fast(n, edges, g1, betas, graph_data) - f0) / h
    
    # Gradient w.r.t. beta1
    b1 = betas.copy(); b1[0] += h
    grad[1] = (_expected_cut_fast(n, edges, gammas, b1, graph_data) - f0) / h
    
    # Gradient w.r.t. gamma2
    g2 = gammas.copy(); g2[1] += h
    grad[2] = (_expected_cut_fast(n, edges, g2, betas, graph_data) - f0) / h
    
    # Gradient w.r.t. beta2
    b2 = betas.copy(); b2[1] += h
    grad[3] = (_expected_cut_fast(n, edges, gammas, b2, graph_data) - f0) / h
    
    return grad

def _objective(params, n, edges, optimum, graph_data=None):
    """Negative ratio: minimize -expected_cut/optimum."""
    if _remaining_time() < 0.5:
        return 1e10
    
    gammas = params[0::2]
    betas = params[1::2]
    
    try:
        ec = _expected_cut_fast(n, edges, gammas, betas, graph_data)
        if optimum <= 0:
            return 0.0
        return -ec / optimum
    except:
        return 1.0

def _objective_grad(params, n, edges, optimum, graph_data=None):
    """Objective and gradient for L-BFGS-B."""
    if _remaining_time() < 0.5:
        return 1e10, np.zeros(4)
    
    gammas = params[0::2]
    betas = params[1::2]
    
    try:
        ec = _expected_cut_fast(n, edges, gammas, betas, graph_data)
        if optimum <= 0:
            return 0.0, np.zeros(4)
        
        grad_ec = _expected_cut_gradient(n, edges, params, graph_data)
        obj = -ec / optimum
        grad = -grad_ec / optimum
        return obj, grad
    except:
        return 1.0, np.zeros(4)

def _optimize_angles(n, edges, p=2):
    """Find optimal QAOA angles using multi-start L-BFGS-B with gradient info."""
    if _remaining_time() < 3.0:
        return np.array([[0.35, 0.25], [0.25, 0.35]])
    
    optimum = _exact_maxcut(n, edges)
    if optimum <= 0:
        return np.zeros((p, 2))
    
    graph_data = _build_graph_data(n, edges)
    m = len(edges)
    density = m / (n * (n - 1) / 2) if n > 1 else 0
    
    best_params = None
    best_val = 1e10
    
    # Generate diverse initializations
    inits = []
    
    # Known good starting points
    base_inits = [
        [0.5, 0.3, 0.3, 0.2],
        [0.3, 0.5, 0.5, 0.3],
        [np.pi/4, np.pi/6, np.pi/6, np.pi/8],
        [0.2, 0.4, 0.4, 0.2],
        [0.4, 0.2, 0.2, 0.4],
        [0.1, 0.6, 0.6, 0.1],
        [0.6, 0.1, 0.1, 0.6],
        [0.25, 0.35, 0.35, 0.25],
        [0.4, 0.4, 0.4, 0.4],
        [0.3, 0.3, 0.3, 0.3],
        [0.5, 0.5, 0.5, 0.5],
        [0.2, 0.2, 0.2, 0.2],
        [0.7, 0.1, 0.1, 0.7],
        [0.1, 0.7, 0.7, 0.1],
        [0.35, 0.35, 0.25, 0.25],
        [0.25, 0.25, 0.35, 0.35],
    ]
    for b in base_inits:
        inits.append(np.array(b, dtype=float))
    
    # Density-dependent initializations
    if density < 0.4:
        inits.append(np.array([0.3, 0.4, 0.4, 0.3]))
        inits.append(np.array([0.2, 0.5, 0.5, 0.2]))
        inits.append(np.array([0.15, 0.45, 0.45, 0.15]))
        inits.append(np.array([0.25, 0.45, 0.45, 0.25]))
    else:
        inits.append(np.array([0.4, 0.3, 0.3, 0.4]))
        inits.append(np.array([0.5, 0.2, 0.2, 0.5]))
        inits.append(np.array([0.45, 0.25, 0.25, 0.45]))
        inits.append(np.array([0.45, 0.35, 0.35, 0.45]))
    
    # Random initializations
    rng = np.random.RandomState(12345)
    for _ in range(12):
        inits.append(rng.uniform(0.05, 0.8, size=4))
    
    # Evaluate all initializations
    init_vals = []
    for init in inits:
        if _remaining_time() < 3.0:
            break
        try:
            val = _objective(init, n, edges, optimum, graph_data)
            init_vals.append((val, init.copy()))
        except:
            continue
    
    if not init_vals:
        return np.array([[0.35, 0.25], [0.25, 0.35]])
    
    init_vals.sort(key=lambda x: x[0])
    
    # Use top initializations for L-BFGS-B optimization
    top_inits = [iv[1] for iv in init_vals[:min(10, len(init_vals))]]
    
    bounds = [(0.01, 1.5), (0.01, 1.5), (0.01, 1.5), (0.01, 1.5)]
    
    for init in top_inits:
        if _remaining_time() < 3.0:
            break
        
        try:
            result = minimize(_objective_grad, init, args=(n, edges, optimum, graph_data),
                             method='L-BFGS-B', jac=True,
                             bounds=bounds,
                             options={'maxiter': 100, 'gtol': 1e-8, 'ftol': 1e-10})
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x.copy()
        except:
            continue
    
    if best_params is None:
        best_params = init_vals[0][1]
    
    # Refine the best point
    if _remaining_time() > 8.0:
        for _ in range(2):
            if _remaining_time() < 3.0:
                break
            try:
                result = minimize(_objective_grad, best_params, args=(n, edges, optimum, graph_data),
                                 method='L-BFGS-B', jac=True,
                                 bounds=bounds,
                                 options={'maxiter': 200, 'gtol': 1e-10, 'ftol': 1e-12})
                if result.fun < best_val:
                    best_val = result.fun
                    best_params = result.x.copy()
            except:
                break
    
    gammas = best_params[0::2]
    betas = best_params[1::2]
    return np.column_stack([gammas, betas])

def angles(n, edges, p):
    """Main entry point: return QAOA angles for the given graph."""
    if _remaining_time() < 2.0:
        return np.array([[0.35, 0.25], [0.25, 0.35]])
    
    try:
        result = _optimize_angles(n, edges, p)
        if result.shape != (p, 2):
            result = np.tile([0.35, 0.25], (p, 1))
        return result
    except:
        return np.array([[0.35, 0.25], [0.25, 0.35]])
# EVOLVE-BLOCK-END
