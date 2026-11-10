import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    start_time = time.time()
    TIME_LIMIT = 600.0
    SAFETY_MARGIN = 10.0

    def get_exact_score(pts):
        """Calculate the true metric R = (d_min / d_max)^2. Higher is better."""
        d = pdist(pts)
        if len(d) == 0:
            return 0.0
        d_min = np.min(d)
        d_max = np.max(d)
        if d_min <= 1e-12 or d_max <= 1e-12:
            return 0.0
        return (d_min / d_max) ** 2

    def soft_min_max_ratio(pts, beta):
        """
        Smooth surrogate for d_max / d_min.
        Uses log-sum-exp for numerical stability.
        """
        d = pdist(pts)
        if len(d) == 0:
            return np.inf
        
        min_d = np.min(d)
        max_d = np.max(d)
        
        if max_d - min_d < 1e-9:
             return np.inf

        # Soft min calculation
        term_min = -beta * (d - min_d)
        sum_min = np.sum(np.exp(term_min))
        soft_min = min_d - (1.0 / beta) * np.log(sum_min)
        
        # Soft max calculation
        term_max = beta * (d - max_d)
        sum_max = np.sum(np.exp(term_max))
        soft_max = max_d + (1.0 / beta) * np.log(sum_max)
        
        if soft_min <= 1e-12:
            return np.inf
            
        return soft_max / soft_min

    def hexagonal_lattice_init(n_points=16):
        """Generate a hexagonal lattice subset centered at origin."""
        i_range = np.arange(-3, 4)
        j_range = np.arange(-3, 4)
        lattice_points = []
        for i in i_range:
            for j in j_range:
                x = i + 0.5 * j
                y = (np.sqrt(3) / 2) * j
                lattice_points.append([x, y])
        lattice_points = np.array(lattice_points)
        dists = np.linalg.norm(lattice_points, axis=1)
        idx = np.argsort(dists)
        hex_pts = lattice_points[idx[:n_points]]
        hex_pts = hex_pts - np.mean(hex_pts, axis=0)
        return hex_pts

    def grid_init(n=4):
        """Generate a square grid centered at origin."""
        g = np.array([(i, j) for i in range(n) for j in range(n)], dtype=float)
        g = g - np.mean(g, axis=0)
        return g

    def random_init(scale=1.0, seed=None):
        """Generate random points centered at origin."""
        rng = np.random.RandomState(seed)
        return rng.randn(16, 2) * scale

    # Generate diverse initial candidates
    candidates = [
        grid_init(4),
        hexagonal_lattice_init(16),
        np.column_stack([np.cos(np.linspace(0, 2*np.pi, 16, endpoint=False)),
                         np.sin(np.linspace(0, 2*np.pi, 16, endpoint=False))]),
        np.vstack([
            np.column_stack([np.cos(np.linspace(0, 2*np.pi, 8, endpoint=False)),
                             np.sin(np.linspace(0, 2*np.pi, 8, endpoint=False))]),
            np.column_stack([1.5*np.cos(np.linspace(0, 2*np.pi, 8, endpoint=False) + np.pi/8),
                             1.5*np.sin(np.linspace(0, 2*np.pi, 8, endpoint=False) + np.pi/8)])
        ]),
        random_init(1.0, seed=1),
        random_init(0.5, seed=2),
        random_init(2.0, seed=3),
    ]

    best_score = 0.0
    best_points = None

    # Evaluate initial candidates
    for c in candidates:
        pts = c.reshape(16, 2)
        score = get_exact_score(pts)
        if score > best_score:
            best_score = score
            best_points = pts.copy()

    # Optimization using Smooth Surrogate (Soft Min/Max)
    # We iterate through candidates and apply SCP-like refinement with increasing beta
    
    # Beta schedule: Start low (smooth) to find basin, end high (sharp) to refine
    betas = [10, 50, 100, 200, 500, 1000]
    
    # We will run a limited number of full optimization cycles to fit in time
    # Each cycle takes roughly 5-20 seconds depending on convergence
    max_cycles = 10 

    for cycle in range(max_cycles):
        if time.time() - start_time > TIME_LIMIT - SAFETY_MARGIN:
            break
            
        # Select a candidate to optimize. 
        # Strategy: Rotate through initial candidates, or pick the best current one?
        # Picking best current one allows refinement. Picking others allows exploration.
        # Let's mix: 50% best current, 50% random initial.
        if cycle % 2 == 0 and best_points is not None:
            current_pts = best_points.copy()
        else:
            current_pts = candidates[cycle % len(candidates)].copy()
        
        current_pts = current_pts.reshape(16, 2)
        
        # Normalize scale to help optimizer? 
        # The problem is scale invariant. 
        # Let's fix the scale by setting d_max = 1 or similar? 
        # Or just let the optimizer handle it. 
        # To avoid drifting to 0 or inf, we can penalize large scales or normalize after each step?
        # Actually, (d_max/d_min) is scale invariant. 
        # But numerical issues might arise.
        # Let's normalize the points to have unit variance or fixed d_max before optimization?
        # If we fix d_max, we maximize d_min.
        
        # Let's try fixing the scale: scale points so d_max = 1.
        d = pdist(current_pts)
        d_max = np.max(d)
        if d_max > 0:
            current_pts = current_pts / d_max
            
        def objective(params):
            pts = params.reshape(16, 2)
            return soft_min_max_ratio(pts, beta)
            
        # Run optimization with increasing beta
        for beta in betas:
            if time.time() - start_time > TIME_LIMIT - SAFETY_MARGIN:
                break
                
            try:
                res = minimize(objective, current_pts.flatten(), method='Nelder-Mead',
                               options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
                
                if hasattr(res, 'x') and np.all(np.isfinite(res.x)):
                    opt_pts = res.x.reshape(16, 2)
                    
                    # Check exact score
                    exact_score = get_exact_score(opt_pts)
                    
                    if exact_score > best_score:
                        best_score = exact_score
                        best_points = opt_pts.copy()
                        
                    # Use the optimized points as the start for the next beta (sharper)
                    current_pts = opt_pts
                    
            except Exception:
                pass

    # Final high-precision polish on the best found points
    if best_points is not None and time.time() - start_time < TIME_LIMIT - SAFETY_MARGIN:
        # Try a very sharp beta
        final_beta = 5000
        def final_objective(params):
            pts = params.reshape(16, 2)
            return soft_min_max_ratio(pts, final_beta)
            
        try:
            res = minimize(final_objective, best_points.flatten(), method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            if hasattr(res, 'x') and np.all(np.isfinite(res.x)):
                final_pts = res.x.reshape(16, 2)
                final_score = get_exact_score(final_pts)
                if final_score > best_score:
                    best_score = final_score
                    best_points = final_pts
        except Exception:
            pass

    # Ensure valid output
    if best_points is None:
        best_points = grid_init(4)

    final_pts = best_points.reshape(16, 2)
    if not np.all(np.isfinite(final_pts)):
        final_pts = grid_init(4)

    return final_pts
