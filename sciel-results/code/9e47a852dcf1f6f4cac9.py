import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import random

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    n = 16
    d = 2
    start_time = time.time()
    timeout = 600.0  # Safety margin from 630s limit
    
    best_score = 0.0
    best_points = None

    # Helper to compute score
    def compute_score(pts):
        if pts.shape != (16, 2):
            return 0.0
        dists = pdist(pts)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def normalize_pts(pts):
        """Center and scale points so that d_max = 1."""
        pts = pts - np.mean(pts, axis=0)
        dm = np.max(pdist(pts))
        if dm > 0:
            pts = pts / dm
        return pts

    # Objective: maximize log(d_min) - log(d_max) = log(d_min/d_max)
    # Equivalent to maximizing (d_min/d_max)^2
    # We minimize the negative of this.
    def objective_ratio(x):
        pts = x.reshape(-1, 2)
        dists = pdist(pts)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-15 or d_min < 1e-15:
            return 0.0
        # Return negative log ratio so we minimize
        return -(np.log(d_min) - np.log(d_max))

    def gradient_ratio(x):
        pts = x.reshape(-1, 2)
        dists = pdist(pts)
        if len(dists) == 0:
            return np.zeros_like(x)
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-15 or d_min < 1e-15:
            return np.zeros_like(x)
        
        grad = np.zeros_like(x)
        
        # Find which pair gives d_min and d_max
        idx_min = np.argmin(dists)
        idx_max = np.argmax(dists)
        
        def dist_idx_to_pair(idx):
            # Inverse of pdist indexing
            count = 0
            for i in range(n):
                for j in range(i+1, n):
                    if count == idx:
                        return i, j
                    count += 1
            return 0, 1
        
        i_min, j_min = dist_idx_to_pair(idx_min)
        i_max, j_max = dist_idx_to_pair(idx_max)
        
        vec_min = pts[i_min] - pts[j_min]
        vec_max = pts[i_max] - pts[j_max]
        
        # Gradient of log(d_min) w.r.t. x:
        # d log(d_ij) / d x_i = (x_i - x_j) / d_ij^2
        # d log(d_ij) / d x_j = (x_j - x_i) / d_ij^2 = -(x_i - x_j) / d_ij^2
        
        # Gradient of -log(d_max) w.r.t. x:
        # -d log(d_ij) / d x_i = -(x_i - x_j) / d_ij^2
        # -d log(d_ij) / d x_j = (x_i - x_j) / d_ij^2
        
        # Total gradient = grad(log d_min) + grad(-log d_max)
        if d_min > 1e-15:
            g_min = vec_min / (d_min * d_min)
            grad[2*i_min:2*i_min+2] += g_min
            grad[2*j_min:2*j_min+2] -= g_min
        
        if d_max > 1e-15:
            g_max = vec_max / (d_max * d_max)
            grad[2*i_max:2*i_max+2] -= g_max
            grad[2*j_max:2*j_max+2] += g_max
        
        return grad

    # Objective 2: Minimize -sum(log(dists)) to encourage uniform spread (repulsive force)
    # This is a smooth surrogate that helps escape local minima where d_min is determined by a single "bad" pair.
    def objective_spread(x):
        pts = x.reshape(-1, 2)
        dists = pdist(pts)
        dists = np.maximum(dists, 1e-9)
        return -np.sum(np.log(dists))

    def gradient_spread(x):
        pts = x.reshape(-1, 2)
        grad = np.zeros_like(x)
        for i in range(n):
            for j in range(i + 1, n):
                vec = pts[i] - pts[j]
                dist = np.linalg.norm(vec)
                if dist < 1e-9:
                    vec = np.array([1e-5, 1e-5])
                    dist = np.linalg.norm(vec)
                # Gradient of -sum(log d) is -sum(grad log d)
                # grad log d_ij w.r.t x_i is (x_i - x_j) / d^2
                # So grad obj w.r.t x_i is - (x_i - x_j) / d^2
                grad[2*i:2*i+2] -= vec / (dist * dist)
                grad[2*j:2*j+2] += vec / (dist * dist)
        return grad

    # Simulated Annealing
    def simulated_annealing(init_pts, max_iters=5000, T_start=0.1, T_end=0.001):
        """Run SA to improve the min/max distance ratio."""
        pts = init_pts.copy()
        best_pts = pts.copy()
        best_score = compute_score(pts)
        
        n_iters = max_iters
        T = T_start
        
        for it in range(n_iters):
            if time.time() - start_time > timeout - 30:
                break
                
            # Cooling schedule
            T = T_start * (T_end / T_start) ** (it / max(1, n_iters - 1))
            
            # Random perturbation
            delta = np.random.randn(n, 2) * T
            new_pts = pts + delta
            
            new_score = compute_score(new_pts)
            cur_score = compute_score(pts)
            
            # Accept if better, or with probability exp((new - cur) / T)
            if new_score >= cur_score:
                pts = new_pts
                if new_score > best_score:
                    best_score = new_score
                    best_pts = new_pts.copy()
            else:
                accept_prob = np.exp((new_score - cur_score) / max(T, 1e-10))
                if np.random.random() < accept_prob:
                    pts = new_pts
        
        return best_pts, best_score

    # Gradient-based local optimization
    def gradient_optimize(init_pts, max_iter=200, use_spread=False):
        """Use L-BFGS-B with the ratio or spread objective."""
        x0 = init_pts.flatten()
        try:
            if use_spread:
                res = minimize(objective_spread, x0, method='L-BFGS-B', jac=gradient_spread,
                               options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10})
            else:
                res = minimize(objective_ratio, x0, method='L-BFGS-B', jac=gradient_ratio,
                               options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10})
            opt_pts = res.x.reshape(-1, 2)
            return opt_pts
        except Exception:
            return init_pts

    # Combined approach: SA from good initializations, then gradient polish
    def improve_from_init(init_pts):
        """Improve a starting configuration using SA + gradient polish."""
        if time.time() - start_time > timeout - 30:
            return init_pts, compute_score(init_pts)
        
        # Normalize
        pts = normalize_pts(init_pts)
        score = compute_score(pts)
        
        # Phase 1: Simulated Annealing
        sa_pts, sa_score = simulated_annealing(pts, max_iters=3000, T_start=0.05, T_end=0.0005)
        if sa_score > score:
            pts = sa_pts
            score = sa_score
        
        # Phase 2: Gradient polish (Ratio objective)
        if time.time() - start_time > timeout - 20:
            return pts, score
        
        grad_pts = gradient_optimize(pts, max_iter=100, use_spread=False)
        grad_pts = normalize_pts(grad_pts)
        grad_score = compute_score(grad_pts)
        
        if grad_score > score:
            pts = grad_pts
            score = grad_score
        
        # Phase 2.5: Gradient polish (Spread objective) - helps if stuck in local min of ratio
        if time.time() - start_time > timeout - 15:
            spread_pts = gradient_optimize(pts, max_iter=100, use_spread=True)
            spread_pts = normalize_pts(spread_pts)
            spread_score = compute_score(spread_pts)
            if spread_score > score:
                pts = spread_pts
                score = spread_score
        
        # Phase 3: Another SA round from the improved point
        if time.time() - start_time > timeout - 15:
            sa2_pts, sa2_score = simulated_annealing(pts, max_iters=2000, T_start=0.02, T_end=0.0002)
            if sa2_score > score:
                pts = sa2_pts
                score = sa2_score
        
        # Final gradient polish
        if time.time() - start_time > timeout - 10:
            final_pts = gradient_optimize(pts, max_iter=50, use_spread=False)
            final_pts = normalize_pts(final_pts)
            final_score = compute_score(final_pts)
            
            if final_score > score:
                pts = final_pts
                score = final_score
        
        return pts, score

    # Generate diverse initializations
    initializations = []
    
    np.random.seed(42)
    
    # 1. Hexagonal lattice (structurally close to optimal packing)
    points_hex = []
    for i in range(-6, 7):
        for j in range(-6, 7):
            x = i + 0.5 * j
            y = j * np.sqrt(3) / 2
            points_hex.append([x, y])
    points_hex = np.array(points_hex)
    dists_center = np.linalg.norm(points_hex, axis=1)
    idx = np.argsort(dists_center)[:n]
    initializations.append(points_hex[idx])
    
    # 2. Multiple hexagonal subsets with different centering
    for offset in [(0.3, 0.2), (-0.2, 0.4), (0.1, -0.3), (-0.4, -0.1)]:
        pts_h = points_hex + np.array(offset)
        dc = np.linalg.norm(pts_h, axis=1)
        idx_h = np.argsort(dc)[:n]
        initializations.append(pts_h[idx_h])
    
    # 3. Regular polygon (circle)
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    initializations.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # 4. Perturbed circles
    for _ in range(5):
        angles = np.linspace(0, 2*np.pi, n, endpoint=False) + np.random.randn(n) * 0.1
        radii = 1 + 0.15 * np.random.randn(n)
        initializations.append(np.column_stack([np.cos(angles)*radii, np.sin(angles)*radii]))
    
    # 5. Grid
    grid_x, grid_y = np.meshgrid(np.linspace(-1, 1, 4), np.linspace(-1, 1, 4))
    initializations.append(np.column_stack([grid_x.ravel(), grid_y.ravel()]))
    
    # 6. Random points
    for _ in range(8):
        initializations.append(np.random.randn(n, 2))
    
    # 7. Two-ring structure (inner 6, outer 10)
    inner_angles = np.linspace(0, 2*np.pi, 6, endpoint=False)
    inner_pts = np.column_stack([0.5*np.cos(inner_angles), 0.5*np.sin(inner_angles)])
    outer_angles = np.linspace(0, 2*np.pi, 10, endpoint=False)
    outer_pts = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
    initializations.append(np.vstack([inner_pts, outer_pts]))
    
    # 8. Three-ring structure
    r1_angles = np.linspace(0, 2*np.pi, 4, endpoint=False)
    r1_pts = np.column_stack([0.3*np.cos(r1_angles), 0.3*np.sin(r1_angles)])
    r2_angles = np.linspace(0, 2*np.pi, 6, endpoint=False)
    r2_pts = np.column_stack([0.65*np.cos(r2_angles), 0.65*np.sin(r2_angles)])
    r3_angles = np.linspace(0, 2*np.pi, 6, endpoint=False)
    r3_pts = np.column_stack([np.cos(r3_angles), np.sin(r3_angles)])
    initializations.append(np.vstack([r1_pts, r2_pts, r3_pts]))

    # Process each initialization
    for init in initializations:
        if time.time() - start_time > timeout - 25:
            break
        
        init_norm = normalize_pts(init)
        init_score = compute_score(init_norm)
        
        # Track best initialization
        if init_score > best_score:
            best_score = init_score
            best_points = init_norm.copy()
        
        # Improve
        improved_pts, improved_score = improve_from_init(init_norm)
        
        if improved_score > best_score:
            best_score = improved_score
            best_points = improved_pts.copy()

    # Final: if we have time, do one more SA round from the best point
    if best_points is not None and time.time() - start_time < timeout - 20:
        final_pts, final_score = simulated_annealing(best_points, max_iters=5000, T_start=0.01, T_end=0.0001)
        if final_score > best_score:
            best_score = final_score
            best_points = final_pts.copy()

    # If no points were found or all failed, return a default
    if best_points is None:
        np.random.seed(42)
        best_points = np.random.randn(n, 2)
        best_points = normalize_pts(best_points)
        
    return best_points
