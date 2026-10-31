"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import scipy.optimize as optimize
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin against 150s limit
    
    # Precompute all triples
    triples = list(combinations(range(N), 3))
    idx_i = np.array([t[0] for t in triples])
    idx_j = np.array([t[1] for t in triples])
    idx_k = np.array([t[2] for t in triples])
    n_triples = len(triples)
    
    def _compute_areas(x):
        """Vectorized triangle area computation."""
        x_i = x[idx_i]
        x_j = x[idx_j]
        x_k = x[idx_k]
        v1 = x_j - x_i
        v2 = x_k - x_i
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        return 0.5 * np.abs(cross)
    
    def _hull_area(x):
        """Compute convex hull area."""
        try:
            hull = ConvexHull(x)
            return hull.volume
        except Exception:
            return 0.0
    
    def score(x_flat):
        """
        Returns the minimum triangle area divided by convex hull area.
        Higher is better.
        """
        x = x_flat.reshape(-1, 2)
        hull_area = _hull_area(x)
        if hull_area < 1e-10:
            return 0.0
        areas = _compute_areas(x)
        min_area = np.min(areas)
        return min_area / hull_area
    
    def soft_min_score(x_flat, beta):
        """
        Soft-min version of the score for smoother optimization.
        Uses log-sum-exp trick for numerical stability.
        """
        x = x_flat.reshape(-1, 2)
        hull_area = _hull_area(x)
        if hull_area < 1e-10:
            return 0.0
        areas = _compute_areas(x)
        
        scaled = -areas / beta
        max_scaled = np.max(scaled)
        log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
        soft_min = -beta * log_sum_exp
        
        return soft_min / hull_area
    
    def neg_soft_min(x_flat, beta):
        """Negative soft-min for minimization."""
        return -soft_min_score(x_flat, beta)
    
    def neg_score(x_flat):
        """Negative score for minimization."""
        return -score(x_flat)
    
    def regular_polygon(n=N, cx=0.0, cy=0.0, r=1.0, phase=0.0):
        """Generate points on a regular polygon."""
        angles = np.linspace(0, 2*np.pi, n, endpoint=False) + phase
        return np.column_stack([cx + r*np.cos(angles), cy + r*np.sin(angles)])
    
    def random_ring(n=N, r_min=0.8, r_max=1.0):
        """Generate random points on a ring."""
        angles = np.sort(np.random.uniform(0, 2*np.pi, n))
        radii = np.random.uniform(r_min, r_max, n)
        return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    
    def two_ring(n=N, n_inner=5, r_inner=0.5, r_outer=1.0):
        """Generate points on two concentric rings."""
        n_outer = n - n_inner
        angles_inner = np.linspace(0, 2*np.pi, n_inner, endpoint=False)
        angles_outer = np.linspace(0, 2*np.pi, n_outer, endpoint=False) + np.pi/n_outer
        x_inner = np.column_stack([r_inner*np.cos(angles_inner), r_inner*np.sin(angles_inner)])
        x_outer = np.column_stack([r_outer*np.cos(angles_outer), r_outer*np.sin(angles_outer)])
        return np.vstack([x_inner, x_outer])
    
    def optimize_with_annealing(x_start, beta_start=0.5, beta_end=0.01, 
                                  n_anneal_steps=15, max_iter_per_step=300):
        """
        Perform optimization with annealing of the soft-min parameter.
        """
        x_curr = x_start.copy()
        
        for step in range(n_anneal_steps):
            if time.time() > deadline - 5:
                break
            
            t = step / max(n_anneal_steps - 1, 1)
            beta = beta_start * (beta_end / beta_start) ** t
            
            try:
                res = optimize.minimize(
                    lambda x_flat, b=beta: neg_soft_min(x_flat, b),
                    x_curr,
                    method='Nelder-Mead',
                    options={'maxiter': max_iter_per_step, 'xatol': 1e-7, 'fatol': 1e-7, 'adaptive': True}
                )
                x_curr = res.x
            except Exception:
                break
        
        return x_curr
    
    def _generate_initial_configs():
        """Generate diverse initial configurations."""
        configs = []
        
        # Regular polygon
        configs.append(regular_polygon(N).flatten())
        
        # Regular polygon with phase shift
        configs.append(regular_polygon(N, phase=np.pi/N).flatten())
        
        # Two-ring structures
        configs.append(two_ring(N, n_inner=5, r_inner=0.5, r_outer=1.0).flatten())
        configs.append(two_ring(N, n_inner=6, r_inner=0.6, r_outer=1.0).flatten())
        configs.append(two_ring(N, n_inner=4, r_inner=0.4, r_outer=1.0).flatten())
        
        # Random ring
        configs.append(random_ring(N, r_min=0.85, r_max=1.0).flatten())
        
        # Perturbed regular polygon
        base = regular_polygon(N)
        r = np.linalg.norm(base, axis=1)
        theta = np.arctan2(base[:, 1], base[:, 0])
        for h in [2, 3, 4]:
            r_pert = r * (1 + 0.08 * np.sin(h*theta + np.random.uniform(0, 2*np.pi)))
            configs.append(np.column_stack([r_pert*np.cos(theta), r_pert*np.sin(theta)]).flatten())
        
        # Grid-like configurations
        grid_pts = np.array([
            [-1.0, -1.0], [0.0, -1.0], [1.0, -1.0],
            [-1.0, 0.0], [0.0, 0.0], [1.0, 0.0],
            [-1.0, 1.0], [0.0, 1.0], [1.0, 1.0],
            [-0.5, -1.5], [0.5, -1.5], [-0.5, 1.5], [0.5, 1.5]
        ])
        configs.append(grid_pts.flatten())
        
        # Hexagonal packing style
        hex_pts = []
        for row in range(3):
            for col in range(3):
                x = col * 1.0 + (row % 2) * 0.5
                y = row * 0.866
                hex_pts.append([x - 1.0, y - 0.866])
        # Add 4 more points around the perimeter
        hex_pts.append([-1.5, 0.0])
        hex_pts.append([1.5, 0.0])
        hex_pts.append([0.0, 1.5])
        hex_pts.append([0.0, -1.5])
        configs.append(np.array(hex_pts).flatten())
        
        return configs
    
    def _run_scipy_de(bounds, maxiter=50, popsize=15, tol=0.01):
        """
        Run scipy's differential evolution for global search.
        Uses a smooth objective for better convergence.
        """
        if time.time() > deadline - 15:
            return None, -np.inf
        
        def de_objective(x_flat):
            if time.time() > deadline - 5:
                return 1e10
            return neg_soft_min(x_flat, beta=0.05)
        
        try:
            res = optimize.differential_evolution(
                de_objective,
                bounds=bounds,
                maxiter=maxiter,
                popsize=popsize,
                tol=tol,
                mutation=(0.5, 1.0),
                recombination=0.8,
                seed=42,
                polish=True
            )
            if res.success or res.fun < 1e10:
                return res.x, -res.fun
        except Exception:
            pass
        
        return None, -np.inf
    
    def _polish(x_start, max_iter=3000):
        """Final polishing with Nelder-Mead on the hard objective."""
        if time.time() > deadline - 3:
            return x_start, score(x_start)
        
        try:
            res = optimize.minimize(
                neg_score,
                x_start,
                method='Nelder-Mead',
                options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            s = score(res.x)
            if s > score(x_start):
                return res.x, s
        except Exception:
            pass
        
        return x_start, score(x_start)
    
    best_x = None
    best_score = -np.inf
    
    # Phase 0: Evaluate all initial configurations
    initial_configs = _generate_initial_configs()
    
    for cfg in initial_configs:
        if time.time() > deadline - 30:
            break
        s = score(cfg)
        if s > best_score:
            best_score = s
            best_x = cfg.copy()
    
    # Phase 1: Annealed optimization from each good initial config
    for cfg in initial_configs:
        if time.time() > deadline - 25:
            break
        if score(cfg) < best_score * 0.8:
            continue  # Skip poor starting points
        
        try:
            x_annealed = optimize_with_annealing(
                cfg, beta_start=0.4, beta_end=0.005,
                n_anneal_steps=12, max_iter_per_step=250
            )
            s = score(x_annealed)
            if s > best_score:
                best_score = s
                best_x = x_annealed.copy()
        except Exception:
            pass
    
    # Phase 2: Scipy differential evolution for global search
    if time.time() < deadline - 25:
        # Define bounds: points in [-2, 2] x [-2, 2]
        bounds = [(-2.0, 2.0)] * (2 * N)
        
        x_de, s_de = _run_scipy_de(bounds, maxiter=40, popsize=12, tol=0.02)
        if x_de is not None and s_de > best_score:
            best_score = s_de
            best_x = x_de.copy()
    
    # Phase 3: Polish the best result
    if best_x is not None and time.time() < deadline - 10:
        best_x, best_score = _polish(best_x, max_iter=3000)
    
    # Phase 4: More annealing restarts from best
    num_restarts = 6
    for i in range(num_restarts):
        if time.time() > deadline - 8:
            break
        
        if i == 0:
            x_start = (best_x.reshape(-1, 2) + np.random.randn(N, 2) * 0.03).flatten()
        elif i == 1:
            x_start = (best_x.reshape(-1, 2) + np.random.randn(N, 2) * 0.08).flatten()
        elif i == 2:
            base = best_x.reshape(-1, 2)
            r = np.linalg.norm(base, axis=1)
            theta = np.arctan2(base[:, 1], base[:, 0])
            r_pert = r * (1 + 0.05 * np.sin(3*theta + np.random.uniform(0, 2*np.pi)))
            x_start = np.column_stack([r_pert*np.cos(theta), r_pert*np.sin(theta)]).flatten()
        elif i == 3:
            x_start = two_ring(N, n_inner=5, r_inner=0.45, r_outer=1.0).flatten()
        elif i == 4:
            angles = np.linspace(0, 2*np.pi, N, endpoint=False) + np.random.randn(N) * 0.08
            radii = 1.0 + np.random.randn(N) * 0.04
            x_start = np.column_stack([radii*np.cos(angles), radii*np.sin(angles)]).flatten()
        else:
            x_start = (best_x.reshape(-1, 2) + np.random.randn(N, 2) * 0.05).flatten()
        
        try:
            x_annealed = optimize_with_annealing(
                x_start, beta_start=0.3, beta_end=0.01,
                n_anneal_steps=10, max_iter_per_step=200
            )
            s = score(x_annealed)
            if s > best_score:
                best_score = s
                best_x = x_annealed.copy()
        except Exception:
            pass
        
        if time.time() < deadline - 5:
            best_x, best_score = _polish(best_x, max_iter=1500)
    
    # Final polish
    if best_x is not None and time.time() < deadline - 3:
        best_x, best_score = _polish(best_x, max_iter=5000)
    
    if best_x is None:
        best_x = regular_polygon(N).flatten()
    
    return best_x.reshape(-1, 2)

# EVOLVE-BLOCK-END
