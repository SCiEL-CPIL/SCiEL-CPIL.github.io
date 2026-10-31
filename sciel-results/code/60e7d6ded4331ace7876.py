"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16
D = 2

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    def get_sq_dists(pts):
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        sq_dists = np.sum(diff**2, axis=2)
        iu = np.triu_indices(N, 1)
        return sq_dists[iu]

    def score_raw(pts):
        d_sq = get_sq_dists(pts)
        d_min = np.sqrt(d_sq.min())
        d_max = np.sqrt(d_sq.max())
        if d_min < 1e-12:
            return 1e10
        return (d_max / d_min) ** 2

    def normalize(pts):
        pts = pts - np.mean(pts, axis=0)
        scale = np.sqrt(np.mean(np.sum(pts**2, axis=1)))
        if scale > 1e-9:
            pts = pts / scale
        return pts

    # Strategy: Simulated Annealing (SA) with diverse restarts.
    # SA is a stochastic local search method that accepts worse solutions with a 
    # temperature-dependent probability, allowing it to escape local minima.
    # We use multiple restarts with different initial configurations to explore 
    # the solution space more broadly. The cooling schedule is geometric.
    
    dim = N * 2
    
    def objective(x):
        pts = x.reshape(N, 2)
        return score_raw(pts)

    def sa_search(x0, max_iter, T0, alpha, deadline):
        x = x0.copy()
        current_score = objective(x)
        best_x = x.copy()
        best_score = current_score
        T = T0
        
        for i in range(max_iter):
            if time.time() > deadline:
                break
            
            # Generate a neighbor by perturbing one or two coordinates
            num_perturb = np.random.randint(1, 3)
            indices = np.random.choice(dim, num_perturb, replace=False)
            x_new = x.copy()
            for idx in indices:
                x_new[idx] += np.random.normal(0, T * 0.1)
            
            # Normalize to maintain scale consistency
            x_new = normalize(x_new.reshape(N, 2)).ravel()
            
            new_score = objective(x_new)
            
            # Accept if better, or with probability exp(-delta/T)
            if new_score < current_score:
                x = x_new
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_x = x.copy()
            else:
                delta = new_score - current_score
                if delta > 0:
                    prob = np.exp(-delta / max(T, 1e-10))
                    if np.random.random() < prob:
                        x = x_new
                        current_score = new_score
            
            # Cool down
            T *= alpha
        
        return best_x, best_score

    # Create diverse initial configurations
    initial_configs = []
    
    # 1. Random points
    for _ in range(3):
        pts = np.random.uniform(-2, 2, (N, 2))
        initial_configs.append(normalize(pts).ravel())
    
    # 2. Perturbed 4x4 grid
    x = np.arange(4) - 1.5
    y = np.arange(4) - 1.5
    xx, yy = np.meshgrid(x, y)
    base_grid = np.column_stack([xx.ravel(), yy.ravel()])
    base_grid = normalize(base_grid)
    for _ in range(3):
        noise = np.random.normal(0, 0.1, dim)
        perturbed = base_grid.ravel() + noise
        initial_configs.append(normalize(perturbed.reshape(N, 2)).ravel())
    
    # 3. Circle-based configurations
    for radius in [1.0, 1.5, 2.0]:
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        pts = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
        noise = np.random.normal(0, 0.1, (N, 2))
        pts = pts + noise
        initial_configs.append(normalize(pts).ravel())
    
    # Run SA from each initial configuration
    best_overall_x = None
    best_overall_score = 1e10
    
    T0 = 1.0
    alpha = 0.9995
    
    for config in initial_configs:
        if time.time() > deadline:
            break
        
        x_best, score_best = sa_search(config, max_iter=5000, T0=T0, alpha=alpha, deadline=deadline)
        
        if score_best < best_overall_score:
            best_overall_score = score_best
            best_overall_x = x_best
    
    # Final polish with a short local search (Nelder-Mead)
    from scipy.optimize import minimize
    
    def objective_smooth(x):
        pts = x.reshape(N, 2)
        d_sq = get_sq_dists(pts)
        d_min = d_sq.min()
        d_max = d_sq.max()
        if d_min < 1e-12:
            return 1e10
        return (np.sqrt(d_max) / np.sqrt(d_min)) ** 2
    
    try:
        res = minimize(
            objective_smooth,
            best_overall_x,
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10}
        )
        if res.fun < best_overall_score:
            best_overall_x = res.x
            best_overall_score = res.fun
    except Exception:
        pass
    
    final_pts = best_overall_x.reshape(N, 2)
    return final_pts

# EVOLVE-BLOCK-END
