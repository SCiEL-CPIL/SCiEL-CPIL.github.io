"""QAOA angle design: graph-dependent parameter optimization via local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations

# Global deadline guard
_START_TIME = time.time()
TIME_LIMIT = 110.0  # seconds, leave 10s safety margin before 120s limit


def _build_adjacency(n, edges):
    """Build adjacency matrix from edge list."""
    adj = np.zeros((n, n), dtype=np.float64)
    for u, v in edges:
        adj[u, v] = 1.0
        adj[v, u] = 1.0
    return adj


def _expected_cut(n, adj, gammas, betas):
    """
    Compute expected cut size for QAOA circuit with given angles.
    
    Circuit: start in |+>^n, then for each layer (gamma_k, beta_k):
      - apply exp(-i * gamma_k * C) where C = sum_{(u,v) in E} (1 - Z_u Z_v)/2
      - apply exp(-i * beta_k * sum X_j)
    
    For p=2, we compute the expected cut exactly using state evolution.
    The state is a superposition over all 2^n basis states.
    """
    # Initialize state: |+>^n = uniform superposition
    # Amplitudes all 1/sqrt(2^n)
    # Phases all 0
    # We track the complex amplitude for each basis state.
    
    # For efficiency with small n (6,8,10), we can use the full state vector.
    # n up to 10 -> 1024 states, which is manageable.
    
    n_states = 1 << n
    # amplitudes: complex array of size 2^n
    # Start: all amplitudes = 1/sqrt(2^n), phase 0
    amp = np.ones(n_states, dtype=np.complex128) / np.sqrt(n_states)
    
    # For each layer
    for gamma, beta in zip(gammas, betas):
        # Step 1: apply exp(-i * gamma * C)
        # C = sum_{(u,v) in E} (1 - Z_u Z_v) / 2
        # Z_u Z_v has eigenvalue +1 if bits u and v are same, -1 if different
        # (1 - Z_u Z_v)/2 has eigenvalue 0 if same, 1 if different
        # So C |x> = (number of cut edges in x) |x>
        # exp(-i * gamma * C) |x> = exp(-i * gamma * cut(x)) |x>
        
        # Compute cut value for each basis state
        # For each state x (integer), cut(x) = number of edges (u,v) where bit u != bit v
        # We can precompute this for efficiency
        
        cut_vals = np.zeros(n_states, dtype=np.float64)
        for x in range(n_states):
            c = 0
            for u, v in edges:
                bu = (x >> u) & 1
                bv = (x >> v) & 1
                if bu != bv:
                    c += 1
            cut_vals[x] = c
        
        amp *= np.exp(-1j * gamma * cut_vals)
        
        # Step 2: apply exp(-i * beta * sum_j X_j)
        # X_j flips bit j. sum_j X_j is the sum of all X operators.
        # exp(-i * beta * sum_j X_j) = product_j exp(-i * beta * X_j)
        # exp(-i * beta * X_j) = cos(beta) * I - i * sin(beta) * X_j
        # This acts on each bit independently.
        
        # For each bit position j, the operator is:
        # cos(beta) |0><0| + cos(beta) |1><1| - i sin(beta) |0><1| - i sin(beta) |1><0|
        # = [[cos(beta), -i sin(beta)], [-i sin(beta), cos(beta)]]
        
        # We apply this to each qubit. Since the operators on different qubits commute,
        # we can apply them one by one.
        
        c_b = np.cos(beta)
        s_b = np.sin(beta)
        
        for j in range(n):
            # Apply the 2x2 gate to qubit j
            # For each pair of states differing only in bit j:
            # new_amp[x with bit j = 0] = c_b * amp[x with bit j = 0] - 1j * s_b * amp[x with bit j = 1]
            # new_amp[x with bit j = 1] = -1j * s_b * amp[x with bit j = 0] + c_b * amp[x with bit j = 1]
            
            # Create masks for states with bit j = 0 and bit j = 1
            mask_0 = np.zeros(n_states, dtype=bool)
            mask_1 = np.zeros(n_states, dtype=bool)
            for x in range(n_states):
                if (x >> j) & 1 == 0:
                    mask_0[x] = True
                else:
                    mask_1[x] = True
            
            amp_0 = amp[mask_0].copy()
            amp_1 = amp[mask_1].copy()
            
            new_0 = c_b * amp_0 - 1j * s_b * amp_1
            new_1 = -1j * s_b * amp_0 + c_b * amp_1
            
            amp[mask_0] = new_0
            amp[mask_1] = new_1
    
    # Expected cut = sum_x |amp[x]|^2 * cut(x)
    probs = np.abs(amp) ** 2
    # Recompute cut values (or we could have stored them)
    cut_vals = np.zeros(n_states, dtype=np.float64)
    for x in range(n_states):
        c = 0
        for u, v in edges:
            bu = (x >> u) & 1
            bv = (x >> v) & 1
            if bu != bv:
                c += 1
        cut_vals[x] = c
    
    return np.dot(probs, cut_vals)


def _exact_maxcut(n, edges):
    """Compute exact MaxCut by brute force for small n."""
    best = 0
    for x in range(1 << n):
        c = 0
        for u, v in edges:
            bu = (x >> u) & 1
            bv = (x >> v) & 1
            if bu != bv:
                c += 1
        if c > best:
            best = c
    return best


def _evaluate_angles(n, edges, gammas, betas, opt_cut):
    """Evaluate the ratio of expected cut to optimal cut."""
    exp_cut = _expected_cut(n, edges, gammas, betas)
    return exp_cut / opt_cut if opt_cut > 0 else 0.0


def _local_search(n, edges, p, init_angles, opt_cut, max_iter=200, step_size=0.1):
    """Perform local search to improve angles."""
    best_angles = init_angles.copy()
    best_val = _evaluate_angles(n, edges, best_angles[:, 0], best_angles[:, 1], opt_cut)
    
    for iteration in range(max_iter):
        if time.time() - _START_TIME > TIME_LIMIT - 5:
            break
        
        improved = False
        # Try perturbing each parameter
        for i in range(p):
            for j in range(2):  # gamma or beta
                for sign in [1, -1]:
                    candidate = best_angles.copy()
                    candidate[i, j] += sign * step_size
                    
                    # Ensure angles are in reasonable range [0, pi]
                    candidate[i, j] = np.clip(candidate[i, j], 0, np.pi)
                    
                    val = _evaluate_angles(n, edges, candidate[:, 0], candidate[:, 1], opt_cut)
                    if val > best_val + 1e-10:
                        best_angles = candidate
                        best_val = val
                        improved = True
        
        if not improved:
            # Try smaller step
            step_size *= 0.7
            if step_size < 1e-4:
                break
        else:
            step_size = min(step_size * 1.1, 0.1)
    
    return best_angles, best_val


def _grid_search(n, edges, p, opt_cut, grid_points=5):
    """Coarse grid search over parameter space."""
    best_angles = None
    best_val = -1
    
    # Grid over [0, pi] for each parameter
    grid = np.linspace(0, np.pi, grid_points)
    
    # For p=2, we have 4 parameters: gamma1, beta1, gamma2, beta2
    # Grid search over all combinations would be grid_points^4
    # For grid_points=5, that's 625 evaluations, each taking some time.
    
    count = 0
    for g1 in grid:
        for b1 in grid:
            for g2 in grid:
                for b2 in grid:
                    if time.time() - _START_TIME > TIME_LIMIT - 10:
                        return best_angles, best_val
                    angles = np.array([[g1, b1], [g2, b2]])
                    val = _evaluate_angles(n, edges, angles[:, 0], angles[:, 1], opt_cut)
                    if val > best_val:
                        best_val = val
                        best_angles = angles.copy()
                    count += 1
    
    return best_angles, best_val


def angles(n, edges, p):
    """
    Design QAOA angles for the given graph.
    
    Args:
        n: number of nodes
        edges: list of (u, v) tuples
        p: number of layers (2)
    
    Returns:
        numpy array of shape (p, 2) with rows (gamma, beta) in radians
    """
    # Precompute optimal cut
    opt_cut = _exact_maxcut(n, edges)
    
    if opt_cut == 0:
        return np.zeros((p, 2))
    
    # Strategy: 
    # 1. Coarse grid search to find a good basin
    # 2. Local search to refine
    
    # Start with grid search
    best_angles, best_val = _grid_search(n, edges, p, opt_cut, grid_points=6)
    
    if best_angles is None:
        # Fallback to a reasonable default
        return np.tile([0.5, 0.3], (p, 1))
    
    # Refine with local search
    best_angles, best_val = _local_search(n, edges, p, best_angles, opt_cut, max_iter=300, step_size=0.05)
    
    # Check time
    if time.time() - _START_TIME > TIME_LIMIT - 5:
        return best_angles
    
    # Try a few more local searches from different starting points
    for restart in range(3):
        if time.time() - _START_TIME > TIME_LIMIT - 5:
            break
        
        # Perturb the best solution
        perturbed = best_angles + np.random.randn(p, 2) * 0.3
        perturbed = np.clip(perturbed, 0, np.pi)
        
        candidate, val = _local_search(n, edges, p, perturbed, opt_cut, max_iter=100, step_size=0.05)
        
        if val > best_val:
            best_angles = candidate
            best_val = val
    
    return best_angles
# EVOLVE-BLOCK-END
