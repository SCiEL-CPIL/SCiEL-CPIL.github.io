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
    # SA is a stochastic local search method that can escape local minima by
    # accepting worse solutions with a temperature-dependent probability.
    # We use multiple restarts with different initial configurations to increase
    # the chance of finding a better global solution.
    
    def simulated_annealing(init_pts, max_iters=50000, T_start=1.0, T_end=0.001, alpha=0.9999):
        pts = init_pts.copy()
        current_score = score_raw(pts)
        best_pts = pts.copy()
        best_score = current_score
        T = T_start
        
        for i in range(max_iters):
            if time.time() > deadline:
                break
                
            # Generate a perturbed solution
            # Move one random point by a small random displacement
            idx = np.random.randint(0, N)
            displacement = np.random.normal(0, 0.1, 2)
            new_pts = pts.copy()
            new_pts[idx] += displacement
            new_pts = normalize(new_pts)
            
            new_score = score_raw(new_pts)
            
            # Acceptance criterion
            if new_score < current_score or np.random.random() < np.exp(-(new_score - current_score) / T):
                current_score = new_score
                pts = new_pts
                if new_score < best_score:
                    best_score = new_score
                    best_pts = new_pts.copy()
            
            T *= alpha
            
        return best_pts, best_score

    # Create diverse initial configurations
    # 1. Random points
    # 2. Perturbed grid
    # 3. Ring configurations
    # 4. Cluster configurations
    
    initial_configs = []
    
    # Random points
    for _ in range(3):
        pts = np.random.uniform(-2, 2, (N, 2))
        initial_configs.append(normalize(pts))
    
    # Perturbed 4x4 grid
    x = np.arange(4) - 1.5
    y = np.arange(4) - 1.5
    xx, yy = np.meshgrid(x, y)
    base_grid = np.column_stack([xx.ravel(), yy.ravel()])
    for _ in range(3):
        noise = np.random.normal(0, 0.2, (N, 2))
        initial_configs.append(normalize(base_grid + noise))
    
    # Ring configurations (points on concentric circles)
    for _ in range(3):
        # 8 points on outer ring, 8 on inner ring
        angles_outer = np.linspace(0, 2*np.pi, 8, endpoint=False) + np.random.uniform(0, 2*np.pi)
        angles_inner = np.linspace(0, 2*np.pi, 8, endpoint=False) + np.random.uniform(0, 2*np.pi)
        r_outer = 1.0
        r_inner = 0.5
        outer_pts = np.column_stack([r_outer * np.cos(angles_outer), r_outer * np.sin(angles_outer)])
        inner_pts = np.column_stack([r_inner * np.cos(angles_inner), r_inner * np.sin(angles_inner)])
        pts = np.vstack([outer_pts, inner_pts])
        initial_configs.append(normalize(pts))
    
    # Cluster configurations (4 clusters of 4 points)
    for _ in range(3):
        cluster_centers = np.random.uniform(-1, 1, (4, 2))
        pts = []
        for c in cluster_centers:
            local_pts = np.random.uniform(-0.3, 0.3, (4, 2))
            pts.append(c + local_pts)
        pts = np.vstack(pts)
        initial_configs.append(normalize(pts))
    
    # Run SA on each initial configuration
    best_overall = None
    best_overall_score = 1e10
    
    for init_pts in initial_configs:
        if time.time() > deadline:
            break
            
        sa_pts, sa_score = simulated_annealing(init_pts, max_iters=30000)
        
        if sa_score < best_overall_score:
            best_overall_score = sa_score
            best_overall = sa_pts
    
    # Final polish with Nelder-Mead if time permits
    from scipy.optimize import minimize
    
    def objective(x):
        pts = x.reshape(N, 2)
        return score_raw(pts)
    
    if time.time() < deadline - 5:
        try:
            res = minimize(
                objective,
                best_overall.ravel(),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10}
            )
            if res.fun < best_overall_score:
                best_overall = res.x.reshape(N, 2)
                best_overall_score = res.fun
        except Exception:
            pass
    
    final_pts = best_overall
    return final_pts

# EVOLVE-BLOCK-END
