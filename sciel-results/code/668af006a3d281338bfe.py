"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations


def _maxcut_optimum(n, edges):
    """Compute the exact maximum cut value for a graph."""
    if n == 0:
        return 0
    best = 0
    for mask in range(1 << n):
        cut = 0
        for u, v in edges:
            bit_u = (mask >> u) & 1
            bit_v = (mask >> v) & 1
            if bit_u != bit_v:
                cut += 1
        if cut > best:
            best = cut
    return best


def _simulate_qaoa(n, edges, gammas, betas):
    """
    Simulate the QAOA circuit and return the final state vector.
    Uses exact quantum simulation for small n.
    """
    dim = 1 << n
    
    # Start in |+>^n
    state = np.ones(dim, dtype=complex) / np.sqrt(dim)
    
    # Precompute cut(x) for all basis states
    cuts = np.zeros(dim, dtype=np.float64)
    for x in range(dim):
        c = 0
        for u, v in edges:
            bit_u = (x >> u) & 1
            bit_v = (x >> v) & 1
            if bit_u != bit_v:
                c += 1
        cuts[x] = c
    
    # Apply cost layers
    for gamma in gammas:
        phases = np.exp(-1j * gamma * cuts)
        state *= phases
    
    # Apply mixer layers
    for beta in betas:
        cos_b = np.cos(beta)
        sin_b = np.sin(beta)
        for k in range(n):
            new_state = np.zeros_like(state)
            for x in range(dim):
                x_flip = x ^ (1 << k)
                new_state[x] = cos_b * state[x] - 1j * sin_b * state[x_flip]
            state = new_state
    
    return state, cuts


def _expected_cut(n, edges, gammas, betas):
    """Compute the expected cut value for a given QAOA parameter set."""
    state, cuts = _simulate_qaoa(n, edges, gammas, betas)
    probs = np.abs(state) ** 2
    return np.sum(probs * cuts)


def _optimize_angles(n, edges, p):
    """
    Optimize QAOA parameters to maximize the expected cut.
    Uses scipy optimization with multiple initial points.
    """
    from scipy.optimize import minimize
    
    best_params = None
    best_value = -np.inf
    
    # Try several initial points
    initial_points = [
        [0.35, 0.25, 0.35, 0.25],
        [0.5, 0.5, 0.3, 0.2],
        [0.2, 0.3, 0.4, 0.3],
        [0.1, 0.1, 0.5, 0.5],
        [0.4, 0.4, 0.2, 0.2],
        [0.25, 0.2, 0.3, 0.25],
        [0.45, 0.35, 0.25, 0.3],
        [0.3, 0.4, 0.35, 0.35],
    ]
    
    for init in initial_points:
        x0 = np.array(init)
        
        def objective(params):
            gammas = params[:p]
            betas = params[p:2*p]
            val = _expected_cut(n, edges, gammas, betas)
            return -val
        
        try:
            result = minimize(objective, x0, method='Nelder-Mead', 
                             options={'maxiter': 300, 'xatol': 1e-7, 'fatol': 1e-9})
            if -result.fun > best_value:
                best_value = -result.fun
                best_params = result.x
        except:
            pass
    
    if best_params is None:
        return np.tile([0.35, 0.25], (p, 1))
    
    gammas = best_params[:p]
    betas = best_params[p:2*p]
    
    angles_out = np.column_stack((gammas, betas))
    return angles_out


def angles(n, edges, p):
    """
    Design graph-dependent two-layer QAOA angles maximizing the mean ratio of the 
    independently simulated expected cut to the exact classical optimum.
    
    Args:
        n: number of nodes
        edges: list of edges (u, v)
        p: number of QAOA layers
    
    Returns:
        numpy array of shape (p, 2) with rows (gamma, beta) in radians
    """
    if p != 2:
        return np.tile([0.35, 0.25], (p, 1))
    
    angles_out = _optimize_angles(n, edges, p)
    
    angles_out = np.asarray(angles_out, dtype=np.float64)
    if angles_out.shape != (p, 2):
        angles_out = np.tile([0.35, 0.25], (p, 1))
    
    return angles_out
# EVOLVE-BLOCK-END
