import numpy as np
from scipy.optimize import minimize
import time

def angles(n, edges, p):
    """
    Compute optimal QAOA angles for a given graph.
    
    Args:
        n: Number of nodes.
        edges: List of edges (u, v).
        p: Number of layers (fixed to 2).
        
    Returns:
        np.array of shape (p, 2) containing (gamma, beta) pairs.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Precompute cost vector
    num_states = 1 << n
    cost_vec = np.zeros(num_states, dtype=np.float64)
    indices = np.arange(num_states)
    
    for u, v in edges:
        bu = (indices >> u) & 1
        bv = (indices >> v) & 1
        mask = bu != bv
        cost_vec[mask] += 1
        
    # Precompute RX rotation matrices for all qubits
    # For each qubit q, the RX matrix acts on the subspace where q is the index
    # We'll precompute the permutation structure for efficient application
    
    # Helper to apply R_x(beta) to all qubits
    def apply_rx_all(state, beta):
        c = np.cos(beta / 2.0)
        s = np.sin(beta / 2.0)
        
        tensor = state.reshape([2] * n)
        
        for q in range(n):
            tensor = np.moveaxis(tensor, q, 0)
            orig_shape = tensor.shape
            tensor = tensor.reshape(2, -1)
            
            v0 = tensor[0, :]
            v1 = tensor[1, :]
            
            new_v0 = c * v0 - 1j * s * v1
            new_v1 = -1j * s * v0 + c * v1
            
            tensor = np.vstack((new_v0, new_v1))
            tensor = tensor.reshape(orig_shape)
            tensor = np.moveaxis(tensor, 0, q)
            
        return tensor.reshape(num_states)
    
    # Helper to apply cost operator
    def apply_cost(state, gamma):
        phase = np.exp(-1j * gamma * cost_vec)
        return state * phase
    
    # Objective function: negative expected cut
    def objective(params):
        g1, b1, g2, b2 = params
        
        state = np.ones(num_states, dtype=np.complex128) / np.sqrt(num_states)
        
        # Layer 1
        state = apply_cost(state, g1)
        state = apply_rx_all(state, b1)
        # Layer 2
        state = apply_cost(state, g2)
        state = apply_rx_all(state, b2)
        
        probs = np.abs(state) ** 2
        exp_cut = np.dot(probs, cost_vec)
        
        return -exp_cut
    
    # Strategy: Grid search + local refinement
    # Parameter space: gamma in [0, pi], beta in [0, pi]
    # For p=2, we have 4 parameters: g1, b1, g2, b2
    
    # Coarse grid search first
    # Use adaptive grid based on time budget
    grid_size = 8  # 8x8x8x8 = 4096 points, manageable for n<=10
    
    best_val = -np.inf
    best_params = np.array([0.35, 0.25, 0.35, 0.25])
    
    # Initial coarse grid
    gamma_grid = np.linspace(0.1, np.pi, grid_size)
    beta_grid = np.linspace(0.1, np.pi, grid_size)
    
    # Evaluate grid points
    for g1 in gamma_grid:
        if time.time() > deadline - 30.0:
            break
        for b1 in beta_grid:
            if time.time() > deadline - 30.0:
                break
            for g2 in gamma_grid:
                if time.time() > deadline - 30.0:
                    break
                for b2 in beta_grid:
                    if time.time() > deadline - 20.0:
                        break
                    params = np.array([g1, b1, g2, b2])
                    val = objective(params)
                    if val < best_val:
                        best_val = val
                        best_params = params.copy()
    
    # Local refinement using Nelder-Mead from best grid point
    if time.time() < deadline - 10.0:
        try:
            res = minimize(objective, best_params, method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except Exception:
            pass
    
    # Try a few additional random restarts if time permits
    if time.time() < deadline - 5.0:
        for _ in range(3):
            if time.time() > deadline - 5.0:
                break
            current_x0 = np.random.uniform(0.1, np.pi, 4)
            try:
                res = minimize(objective, current_x0, method='Nelder-Mead',
                               options={'maxiter': 200, 'xatol': 1e-7, 'fatol': 1e-7})
                if res.fun < best_val:
                    best_val = res.fun
                    best_params = res.x
            except Exception:
                pass
    
    # Return in required shape (p, 2)
    return best_params.reshape(p, 2)
