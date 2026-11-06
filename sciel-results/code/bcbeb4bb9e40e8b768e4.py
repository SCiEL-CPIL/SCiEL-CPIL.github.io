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
    deadline = start_time + 110.0  # Safety margin
    
    # Precompute cost vector
    num_states = 1 << n
    cost_vec = np.zeros(num_states, dtype=np.float64)
    indices = np.arange(num_states)
    
    for u, v in edges:
        bu = (indices >> u) & 1
        bv = (indices >> v) & 1
        mask = bu != bv
        cost_vec[mask] += 1
        
    # Define the objective function
    def objective(params):
        # Check time
        if time.time() > deadline:
            return 1e10
            
        # Unpack parameters
        # params: [g1, b1, g2, b2]
        g1, b1, g2, b2 = params
        
        # Initialize state |+>^n
        state = np.ones(num_states, dtype=np.complex128) / np.sqrt(num_states)
        
        # Helper to apply R_x(beta) to all qubits
        def apply_rx_all(state, beta):
            c = np.cos(beta / 2.0)
            s = np.sin(beta / 2.0)
            
            # Reshape to tensor
            shape = [2] * n
            tensor = state.reshape(shape)
            
            # Apply rotation to each axis
            # R_x = [[c, -i*s], [-i*s, c]]
            for q in range(n):
                # Move axis q to front
                tensor = np.moveaxis(tensor, q, 0)
                # Reshape to (2, -1)
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
            
        # Apply layers
        # Layer 1
        state = apply_cost(state, g1)
        state = apply_rx_all(state, b1)
        # Layer 2
        state = apply_cost(state, g2)
        state = apply_rx_all(state, b2)
        
        # Calculate expected cut
        probs = np.abs(state) ** 2
        exp_cut = np.dot(probs, cost_vec)
        
        return -exp_cut

    # Optimization
    best_val = -np.inf
    best_params = np.array([0.35, 0.25, 0.35, 0.25])
    
    # Initial guess
    x0 = np.array([0.35, 0.25, 0.35, 0.25])
    
    # Try a few random restarts if time permits, otherwise just one good run
    # For n=10, one run of Nelder-Mead with 500 iterations is fast.
    # Let's try 5 restarts with 200 iterations each, or 1 run with 1000 iterations.
    # Given the deadline, let's be conservative.
    
    num_restarts = 3
    for i in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
            
        if i == 0:
            current_x0 = x0
        else:
            # Random initialization in [0, pi]
            current_x0 = np.random.uniform(0, np.pi, 4)
            
        try:
            res = minimize(objective, current_x0, method='Nelder-Mead', 
                           options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-6})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except Exception:
            pass
            
    # Return in required shape (p, 2)
    # best_params is [g1, b1, g2, b2]
    # We want rows (gamma, beta)
    return best_params.reshape(p, 2)
