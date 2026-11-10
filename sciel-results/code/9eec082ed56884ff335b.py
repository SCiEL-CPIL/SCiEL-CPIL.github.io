import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    n = 16
    d = 2
    start_time = time.time()
    deadline = 600.0
    safety_margin = 10.0

    def compute_ratio(points):
        """Compute the actual score: (d_min/d_max)^2"""
        dists = pdist(points)
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def get_cost_fast(pts):
        """Calculate (d_max / d_min)^2. Lower is better. Optimized for small N."""
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=-1)
        
        mask = ~np.eye(pts.shape[0], dtype=bool)
        
        dist_sq_min = np.where(mask, dist_sq, np.inf)
        d_min_sq = np.min(dist_sq_min)
        d_max_sq = np.max(dist_sq)
        
        if d_min_sq <= 1e-12 or d_max_sq <= 1e-12:
            return np.inf
            
        return d_max_sq / d_min_sq

    def get_hexagonal_lattice_init():
        """Generate a hexagonal lattice initialization with 16 central points."""
        R = 10
        points = []
        a = 1.0
        b = np.sqrt(3) / 2
        for i in range(-R, R + 1):
            for j in range(-R, R + 1):
                x = i * a + j * 0.5 * a
                y = j * b
                points.append([x, y])
        
        points = np.array(points)
        
        centroid = np.mean(points, axis=0)
        dists = np.linalg.norm(points - centroid, axis=1)
        idx = np.argsort(dists)
        selected = points[idx[:n]]
        
        selected = selected - np.mean(selected, axis=0)
        return selected

    def get_two_ring_init():
        """Generate two concentric rings: 6 inner, 10 outer."""
        r_inner = 0.5
        angles_inner = np.linspace(0, 2*np.pi, 6, endpoint=False)
        p_inner = np.column_stack([r_inner * np.cos(angles_inner), r_inner * np.sin(angles_inner)])
        
        r_outer = 1.0
        angles_outer = np.linspace(np.pi/10, 2*np.pi + np.pi/10, 10, endpoint=False)
        p_outer = np.column_stack([r_outer * np.cos(angles_outer), r_outer * np.sin(angles_outer)])
        
        return np.vstack([p_inner, p_outer])

    def get_grid_init():
        """4x4 grid."""
        x = np.linspace(-1, 1, 4)
        y = np.linspace(-1, 1, 4)
        xx, yy = np.meshgrid(x, y)
        return np.column_stack([xx.ravel(), yy.ravel()])

    def simulated_annealing(initial_pts, time_budget):
        """Simulated Annealing with fast cost evaluation."""
        pts = initial_pts.copy()
        current_cost = get_cost_fast(pts)
        best_pts_sa = pts.copy()
        best_cost_sa = current_cost
        
        T = 0.5
        cooling_rate = 0.9999
        min_T = 1e-8
        max_iter = 10000000
        no_improve_count = 0
        max_no_improve = 20000
        
        t_start = time.time()
        
        for i in range(max_iter):
            if time.time() - t_start > time_budget:
                break
            
            idx = np.random.randint(0, len(pts))
            noise = np.random.randn(2) * T
            old_pos = pts[idx].copy()
            pts[idx] += noise
            
            new_cost = get_cost_fast(pts)
            
            if new_cost < current_cost:
                current_cost = new_cost
                no_improve_count = 0
                
                if current_cost < best_cost_sa:
                    best_cost_sa = current_cost
                    best_pts_sa = pts.copy()
            else:
                delta = new_cost - current_cost
                if T > min_T:
                    if delta > T * 50: 
                        accept = False
                    else:
                        prob = np.exp(-delta / T)
                        accept = np.random.rand() < prob
                else:
                    accept = False
                
                if accept:
                    current_cost = new_cost
                else:
                    pts[idx] = old_pos
                    no_improve_count += 1
                    
            T *= cooling_rate
            
            if no_improve_count > max_no_improve:
                pts = best_pts_sa.copy()
                current_cost = best_cost_sa
                T = 0.1
                no_improve_count = 0
                max_no_improve = 10000

        return best_pts_sa, best_cost_sa

    def smooth_objective(params):
        """
        Smooth objective using log-sum-exp for differentiable min/max approximation.
        Minimizes d_max / d_min (equivalent to minimizing (d_max/d_min)^2 up to scaling).
        """
        pts = params.reshape(n, d)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=-1)
        
        mask = ~np.eye(n, dtype=bool)
        dist_sq = np.where(mask, dist_sq, 0)
        
        valid_dsq = dist_sq[dist_sq > 1e-12]
        if len(valid_dsq) == 0:
            return 1e10
        
        # Adaptive beta
        median_dsq = np.median(valid_dsq)
        beta = 0.05 * median_dsq
        beta = max(beta, 1e-4)
        
        # Soft Max: approximates max(dist_sq)
        # log(sum(exp(x/beta))) * beta
        exp_sq = np.exp(dist_sq / beta)
        log_sum_exp_max = beta * np.log(np.sum(exp_sq))
        
        # Soft Min: approximates min(dist_sq)
        # -log(sum(exp(-x/beta))) * beta
        exp_neg_sq = np.exp(-dist_sq / beta)
        log_sum_exp_min = -beta * np.log(np.sum(exp_neg_sq))
        
        d_min_sq_smooth = log_sum_exp_min
        d_max_sq_smooth = log_sum_exp_max
        
        if d_min_sq_smooth <= 1e-12 or d_max_sq_smooth <= 1e-12:
            return 1e10
            
        return d_max_sq_smooth / d_min_sq_smooth

    # 1. Generate Initializations
    initializations = [
        get_hexagonal_lattice_init(),
        get_two_ring_init(),
        get_grid_init()
    ]
    
    # Add some random perturbations to break symmetry
    for _ in range(3):
        init = get_hexagonal_lattice_init() + np.random.randn(n, d) * 0.1
        init -= np.mean(init, axis=0)
        initializations.append(init)

    best_points = None
    best_score = -1.0
    best_cost = np.inf

    for init_points in initializations:
        score = compute_ratio(init_points)
        cost = get_cost_fast(init_points)
        if score > best_score:
            best_score = score
            best_points = init_points.copy()
            best_cost = cost

    # 2. Simulated Annealing Phase
    time_elapsed = time.time() - start_time
    sa_time_budget = deadline - time_elapsed - safety_margin - 30.0
    if sa_time_budget > 10:
        best_pts_sa, best_cost_sa = simulated_annealing(best_points, sa_time_budget)
        if best_cost_sa < best_cost:
            best_points = best_pts_sa
            best_cost = best_cost_sa
            best_score = 1.0 / best_cost_sa

    # 3. Smooth Optimization Phase (Polish)
    time_elapsed = time.time() - start_time
    polish_budget = deadline - time_elapsed - safety_margin
    
    if polish_budget > 10 and best_points is not None:
        try:
            x0 = best_points.flatten()
            res = minimize(
                smooth_objective,
                x0,
                method='Powell',
                options={
                    'maxiter': 500,
                    'xtol': 1e-9,
                    'ftol': 1e-9
                }
            )
            if res.success:
                pts = res.x.reshape(n, d)
                cost = get_cost_fast(pts)
                if cost < best_cost:
                    best_points = pts
                    best_cost = cost
                    best_score = 1.0 / best_cost
        except Exception:
            pass

    # 4. Final Nelder-Mead Polish
    time_elapsed = time.time() - start_time
    final_budget = deadline - time_elapsed - safety_margin
    if final_budget > 5 and best_points is not None:
        try:
            x0 = best_points.flatten()
            res = minimize(
                get_cost_fast,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            if hasattr(res, 'fun') and np.isfinite(res.fun):
                pts = res.x.reshape(n, d)
                cost = get_cost_fast(pts)
                if cost < best_cost:
                    best_points = pts
                    best_cost = cost
                    best_score = 1.0 / best_cost
        except Exception:
            pass

    # Ensure valid output
    if best_points is None or not np.all(np.isfinite(best_points)):
        best_points = initializations[0]

    return best_points

# EVOLVE-BLOCK-END
