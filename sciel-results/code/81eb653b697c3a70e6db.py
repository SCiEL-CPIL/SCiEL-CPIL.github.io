import numpy as np
import itertools
import time
from scipy.optimize import differential_evolution

def angles(n, edges, p):
    start_time = time.time()
    time_limit = 110.0  # Safety margin
    
    if not edges:
        return np.zeros((p, 2))
        
    num_states = 1 << n
    c_vals = np.zeros(num_states, dtype=float)
    
    # Precompute C values
    states = np.arange(num_states, dtype=np.uint32)
    
    for i, j in edges:
        mask_i = 1 << i
        mask_j = 1 << j
        
        bits_i = (states & mask_i) > 0
        bits_j = (states & mask_j) > 0
        parity = bits_i ^ bits_j
        contrib = 1.0 - 2.0 * parity.astype(float)
        c_vals += contrib
        
    c_opt = np.max(c_vals)
    
    if c_opt == 0:
        return np.zeros((p, 2))

    # Precompute index permutations for X rotations
    indices = np.arange(num_states)
    idx_arrays = [indices ^ (1 << i) for i in range(n)]
    
    def objective(params):
        if time.time() - start_time > time_limit:
            return 1e9
            
        g1, b1, g2, b2 = params
        
        # Initial state
        psi = np.ones(num_states, dtype=complex)
        psi /= np.sqrt(num_states)
        
        # Layer 1
        psi *= np.exp(-1j * g1 * c_vals)
        
        for i in range(n):
            idx = idx_arrays[i]
            psi = np.cos(b1) * psi - 1j * np.sin(b1) * psi[idx]
            
        # Layer 2
        psi *= np.exp(-1j * g2 * c_vals)
        
        for i in range(n):
            idx = idx_arrays[i]
            psi = np.cos(b2) * psi - 1j * np.sin(b2) * psi[idx]
            
        probs = np.abs(psi) ** 2
        expected_cut = np.sum(c_vals * probs)
        
        return -expected_cut / c_opt

    bounds = [(0, np.pi), (0, np.pi), (0, np.pi), (0, np.pi)]
    
    try:
        result = differential_evolution(
            objective, 
            bounds, 
            maxiter=100, 
            popsize=15, 
            tol=1e-6, 
            seed=42,
            polish=True
        )
        best_params = result.x
    except Exception:
        best_params = np.array([0.5, 0.5, 0.5, 0.5])

    angles_out = np.array([
        [best_params[0], best_params[1]],
        [best_params[2], best_params[3]]
    ])
    
    return angles_out
