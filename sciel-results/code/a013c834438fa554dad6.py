import numpy as np
from scipy.optimize import minimize

N = 14

def construct():
    # Initial guess: uniform distribution
    # Angles are 0, 2pi/N, 4pi/N, ...
    angles_uniform = np.linspace(0, 2 * np.pi, N, endpoint=False)
    
    # We optimize the angular gaps.
    # Uniform gaps: 2*pi/N
    delta_init = np.full(N-1, 2 * np.pi / N)
    
    def objective(x):
        if np.any(x <= 0):
            return 0.0
        deltas = np.empty(N)
        deltas[:N-1] = x
        deltas[N-1] = 2 * np.pi - np.sum(x)
        if deltas[N-1] <= 0:
            return 0.0
        
        # Vectorized calculation of min triangle area factor
        # phi[i] is angle of point i
        phi = np.zeros(N)
        phi[1:] = np.cumsum(deltas[:N-1])
        
        # Calculate arcs for all triples i < j < k
        # a = phi[j] - phi[i]
        # b = phi[k] - phi[j]
        # c = 2*pi - a - b
        
        # Generate indices
        i_idx, j_idx, k_idx = np.indices((N, N, N))
        i_idx = i_idx[i_idx < j_idx]
        j_idx = j_idx[i_idx < j_idx]
        k_idx = k_idx[i_idx < j_idx]
        
        # Filter i < j < k properly
        mask = (i_idx < j_idx) & (j_idx < k_idx)
        i_idx = i_idx[mask]
        j_idx = j_idx[mask]
        k_idx = k_idx[mask]
        
        a = phi[j_idx] - phi[i_idx]
        b = phi[k_idx] - phi[j_idx]
        c = 2 * np.pi - a - b
        
        vals = np.sin(a) + np.sin(b) + np.sin(c)
        min_num = np.min(vals)
        
        den = np.sum(np.sin(deltas))
        
        if den <= 0:
            return 0.0
            
        return -(min_num / den)

    # Use Nelder-Mead with tight tolerances
    # Start from uniform
    result = minimize(
        objective, 
        delta_init, 
        method='Nelder-Mead', 
        options={
            'xatol': 1e-12, 
            'fatol': 1e-12, 
            'maxiter': 5000, 
            'adaptive': True
        }
    )
    
    if result.success and result.fun < objective(delta_init):
        opt_deltas = np.empty(N)
        opt_deltas[:N-1] = result.x
        opt_deltas[N-1] = 2 * np.pi - np.sum(result.x)
    else:
        opt_deltas = np.full(N, 2 * np.pi / N)
        
    angles = np.zeros(N)
    angles[1:] = np.cumsum(opt_deltas[:N-1])
    
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return points
