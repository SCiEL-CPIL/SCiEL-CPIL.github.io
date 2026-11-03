import numpy as np
from scipy.optimize import minimize

N = 600

def construct():
    L = 0.5
    M = 15  # Number of harmonics
    
    # Grid for optimization (internal representation)
    n_grid = 1024
    x = np.linspace(-L/2, L/2, n_grid, endpoint=False)
    
    # Precompute basis for f(x)
    # f(x) = sum u_k phi_k(x)
    # phi_0 = 1/sqrt(L)
    # phi_k = sqrt(2/L) cos(2 pi k x / L)
    
    def get_f(u):
        f_vals = np.zeros(n_grid)
        f_vals += u[0] / np.sqrt(L)
        if M > 0:
            k_idx = np.arange(1, M+1)
            cos_terms = np.cos(2 * np.pi * np.outer(k_idx, x) / L)
            f_vals += np.sum(u[1:] * np.sqrt(2.0/L) * cos_terms, axis=0)
        return f_vals

    def objective(u):
        f_vals = get_f(u)
        
        # Penalty for negative f
        neg_f = np.maximum(0, -f_vals)
        penalty = np.sum(neg_f**2)
        
        # Compute linear autoconvolution
        conv_direct = np.convolve(f_vals, f_vals, mode='full')
        max_conv = np.max(conv_direct)
        
        # Energy
        energy = np.sum(f_vals**2) * (L / n_grid)
        
        if energy < 1e-10:
            return 1e10 + penalty
        
        ratio = max_conv / energy
        
        return ratio + 100.0 * penalty

    # Optimization
    u0 = np.zeros(M+1)
    u0[0] = 1.0
    
    best_u = u0.copy()
    best_score = np.inf
    
    # Run optimization
    # We will use a few restarts.
    
    for restart in range(5):
        if restart == 0:
            u = u0.copy()
        else:
            u = np.random.randn(M+1)
            u /= np.linalg.norm(u)
            u[0] = np.abs(u[0]) # Bias towards positive DC
        
        # Use Powell as it doesn't require gradients and handles non-smoothness
        res = minimize(objective, u, method='Powell', 
                       options={'maxiter': 100, 'ftol': 1e-8})
        
        if res.fun < best_score:
            # Check feasibility
            f_check = get_f(res.x)
            if np.min(f_check) >= -1e-6:
                best_score = res.fun
                best_u = res.x.copy()
    
    # Convert to step heights
    interval_width = L / N
    centers = -L/2 + (np.arange(N) + 0.5) * interval_width
    
    # Compute f at centers using best_u
    f_centers = np.zeros(N)
    f_centers += best_u[0] / np.sqrt(L)
    if M > 0:
        k_idx = np.arange(1, M+1)
        cos_terms = np.cos(2 * np.pi * np.outer(k_idx, centers) / L)
        f_centers += np.sum(best_u[1:] * np.sqrt(2.0/L) * cos_terms, axis=0)
    
    heights = np.maximum(0, f_centers)
    
    return list(heights)
