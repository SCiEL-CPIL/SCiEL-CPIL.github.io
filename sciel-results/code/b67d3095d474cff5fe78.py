import numpy as np
import time

N = 21

def construct():
    start_time = time.time()
    deadline = 115  # Safety margin
    
    # Initial setup: Grid in 1x1 square
    # 5x5 grid = 25 cells, we use 21.
    # Let's use a 5x5 grid, take first 21.
    side = 5
    grid_xs = np.linspace(0.1, 0.9, side)
    grid_ys = np.linspace(0.1, 0.9, side)
    
    # Initialize positions and radii
    # Start with radii that are half the grid spacing to allow immediate overlap resolution
    grid_spacing = 0.8 / (side - 1)
    init_r = grid_spacing / 2.0 * 0.95 # Slightly smaller to start valid
    
    init_params = []
    count = 0
    for i in range(side):
        for j in range(side):
            if count < N:
                x = grid_xs[i]
                y = grid_ys[j]
                r = init_r
                init_params.extend([x, y, r])
                count += 1
            
    init_params = np.array(init_params)
    
    # Simulated Annealing Parameters
    T = 1.0  # Initial temperature
    T_min = 1e-6
    cooling_rate = 0.9995 # Adjust based on iterations
    step_size = 0.05      # Initial step size for perturbation
    
    best_params = init_params.copy()
    best_energy = np.inf
    current_params = init_params.copy()
    current_energy = np.inf
    
    def calc_energy(params):
        x = params[0::3]
        y = params[1::3]
        r = params[2::3]
        
        sum_r = np.sum(r)
        
        # Boundary violations
        # x in [r, 1-r]
        viol_x_low = np.maximum(0, r - x)
        viol_x_high = np.maximum(0, r - (1 - x))
        viol_y_low = np.maximum(0, r - y)
        viol_y_high = np.maximum(0, r - (1 - y))
        
        p_boundary = np.sum(viol_x_low**2) + np.sum(viol_x_high**2) + \
                     np.sum(viol_y_low**2) + np.sum(viol_y_high**2)
        
        # Overlap violations
        # Vectorized
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        min_dist = r[:, None] + r[None, :]
        
        # Only upper triangle to avoid double counting and self-overlap
        # Create a mask for i < j
        i, j = np.triu_indices(N, k=1)
        d_pairs = dist[i, j]
        min_d_pairs = min_dist[i, j]
        
        viol = np.maximum(0, min_d_pairs - d_pairs)
        p_overlap = np.sum(viol**2)
        
        penalty = p_boundary + p_overlap
        # Objective: Maximize sum_r -> Minimize -sum_r
        # Add penalty
        energy = -sum_r + 100.0 * penalty
        
        return energy, sum_r, penalty
    
    current_energy, _, _ = calc_energy(current_params)
    best_energy = current_energy
    
    # Adaptive step size
    # We want to explore for a while, then exploit.
    
    while time.time() < deadline:
        # Perturb
        # Randomly select a subset of circles to move to allow local adjustments
        # Or move all with small noise. Moving all is simpler but might be too chaotic.
        # Let's move all with Gaussian noise.
        noise = np.random.normal(0, step_size, N * 3)
        new_params = current_params + noise
        
        new_energy, _, _ = calc_energy(new_params)
        
        delta_E = new_energy - current_energy
        
        # Metropolis criterion
        if delta_E < 0 or np.random.random() < np.exp(-delta_E / T):
            current_params = new_params
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_params = current_params.copy()
        
        # Cooling
        T *= cooling_rate
        step_size *= 0.9999 # Slowly reduce step size
        
        # If T gets too low, restart with a "kick" or break
        if T < T_min:
            break
            
    # Final refinement with Nelder-Mead on the best found by SA
    # This combines the global exploration of SA with local precision of NM.
    if time.time() < deadline - 10:
        from scipy.optimize import minimize
        
        def objective(params):
            e, _, _ = calc_energy(params)
            return e
            
        # Try a few local optimizations from the best SA solution
        # and maybe a few other good solutions found during SA (if we stored them)
        # For simplicity, just refine the single best.
        
        try:
            res = minimize(objective, best_params, method='Nelder-Mead', 
                           options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8})
            
            res_energy, _, _ = calc_energy(res.x)
            if res_energy < best_energy:
                best_params = res.x
                best_energy = res_energy
        except Exception:
            pass
            
    # Extract result
    x = best_params[0::3]
    y = best_params[1::3]
    r = best_params[2::3]
    
    # Ensure validity by shrinking radii if necessary
    # Check validity
    _, sum_r, penalty = calc_energy(best_params)
    if penalty > 1e-6:
        # Shrink radii
        for _ in range(10):
            r_shrunk = r * 0.99
            test_params = np.array([x, y, r_shrunk] for x, y, r_shrunk in zip(x, y, r_shrunk)).flatten()
            _, _, p = calc_energy(test_params)
            if p < 1e-6:
                r = r_shrunk
                break
            r = r_shrunk
            
    result = np.column_stack((x, y, r))
    return result
