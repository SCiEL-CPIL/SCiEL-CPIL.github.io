import numpy as np
from scipy.optimize import minimize
import time

def angles(n, edges, p):
    # Deadline: 110 seconds to be safe
    start_time = time.time()
    deadline = start_time + 110.0

    # 1. Precompute cost vector
    num_states = 1 << n
    cost_vec = np.zeros(num_states, dtype=np.float64)
    
    # For each edge (u, v), add 1 to states where bits differ
    indices = np.arange(num_states)
    for u, v in edges:
        bu = (indices >> u) & 1
        bv = (indices >> v) & 1
        mask = bu != bv
        cost_vec[mask] += 1

    # 2. Define the objective function with optimized state evolution
    def objective(params):
        # Check time
        if time.time() > deadline:
            return 1e10

        g1, b1, g2, b2 = params
        
        # Precompute trig values
        c1 = np.cos(b1 / 2.0)
        s1 = np.sin(b1 / 2.0)
        c2 = np.cos(b2 / 2.0)
        s2 = np.sin(b2 / 2.0)
        
        # State initialization: |+>^n
        state = np.ones(num_states, dtype=np.complex128) / np.sqrt(num_states)
        
        # Helper to apply R_x(beta) to all qubits efficiently
        def apply_rx_all(state, c, s):
            # Reshape to tensor
            shape = [2] * n
            tensor = state.reshape(shape)
            
            # Apply to each axis
            for q in range(n):
                # Move axis q to front
                tensor = np.moveaxis(tensor, q, 0)
                # Reshape to (2, -1)
                orig_shape = tensor.shape
                tensor = tensor.reshape(2, -1)
                # Apply matrix [[c, -1j*s], [-1j*s, c]]
                v0 = tensor[0, :]
                v1 = tensor[1, :]
                new_v0 = c * v0 - 1j * s * v1
                new_v1 = -1j * s * v0 + c * v1
                tensor = np.vstack((new_v0, new_v1))
                # Reshape back
                tensor = tensor.reshape(orig_shape)
                # Move axis back
                tensor = np.moveaxis(tensor, 0, q)
                
            return tensor.reshape(num_states)

        # Apply layers
        # Layer 1: cost phase
        state = state * np.exp(-1j * g1 * cost_vec)
        # Layer 1: mixer
        state = apply_rx_all(state, c1, s1)
        # Layer 2: cost phase
        state = state * np.exp(-1j * g2 * cost_vec)
        # Layer 2: mixer
        state = apply_rx_all(state, c2, s2)
        
        # Calculate expected cut
        probs = np.abs(state) ** 2
        exp_cut = np.dot(probs, cost_vec)
        
        return -exp_cut

    # 3. Optimization with multiple restarts
    best_val = -np.inf
    best_params = np.array([0.35, 0.25, 0.35, 0.25])
    
    # Initial guesses - use known good starting points
    initial_points = [
        np.array([0.35, 0.25, 0.35, 0.25]),
        np.array([0.5, 0.3, 0.4, 0.2]),
        np.array([0.2, 0.4, 0.5, 0.3]),
    ]
    
    # Add random points
    rng = np.random.default_rng(42)
    for _ in range(7):
        initial_points.append(rng.uniform(0, np.pi, 4))
    
    for i, x0 in enumerate(initial_points):
        if time.time() > deadline - 5.0:  # Leave 5s buffer
            break
            
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                          options={'maxiter': 300, 'xatol': 1e-7, 'fatol': 1e-7,
                                   'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except Exception:
            pass

    # Return in required shape (p, 2)
    return best_params.reshape(p, 2)
