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

    # Precompute pair indices for vectorized operations
    pair_i = []
    pair_j = []
    for i in range(n):
        for j in range(i + 1, n):
            pair_i.append(i)
            pair_j.append(j)
    pair_i = np.array(pair_i)
    pair_j = np.array(pair_j)
    n_pairs = len(pair_i)

    # Soft-min and soft-max using log-sum-exp
    def soft_min_max(pts, beta_min, beta_max):
        """
        Compute soft-min and soft-max of pairwise distances.
        soft_min = -1/beta * log(sum(exp(-beta * d)))
        soft_max = 1/beta * log(sum(exp(beta * d)))
        """
        pts = pts.reshape(-1, 2)
        # Compute all pairwise distances
        vi = pts[pair_i]
        vj = pts[pair_j]
        vecs = vi - vj
        dists = np.sqrt(np.sum(vecs ** 2, axis=1) + 1e-15)
        
        # Soft min: -1/beta * log(sum(exp(-beta * d)))
        # For numerical stability
        min_d = np.min(dists)
        exp_neg_beta_d = np.exp(-beta_min * (dists - min_d))
        soft_min_val = min_d - np.log(np.sum(exp_neg_beta_d)) / beta_min
        
        # Soft max: 1/beta * log(sum(exp(beta * d)))
        max_d = np.max(dists)
        exp_beta_d = np.exp(beta_max * (dists - max_d))
        soft_max_val = max_d + np.log(np.sum(exp_beta_d)) / beta_max
        
        return soft_min_val, soft_max_val, dists, vecs

    def soft_min_max_grad(pts, beta_min, beta_max):
        """
        Compute gradients of soft_min and soft_max with respect to point coordinates.
        Returns (grad_soft_min, grad_soft_max) as arrays of shape (n, 2).
        """
        pts = pts.reshape(-1, 2)
        vi = pts[pair_i]
        vj = pts[pair_j]
        vecs = vi - vj
        dists = np.sqrt(np.sum(vecs ** 2, axis=1) + 1e-15)
        
        # Soft min gradient
        min_d = np.min(dists)
        exp_neg_beta_d = np.exp(-beta_min * (dists - min_d))
        weights_min = exp_neg_beta_d / np.sum(exp_neg_beta_d)
        # d(soft_min)/d(vec_k) = -weights_min[k] * (1/d_k) * vec_k
        # But we want d(soft_min)/d(pts)
        grad_min = np.zeros_like(pts)
        # For each pair k, contribution to grad_min[i] is weights_min[k] * vecs[k] / dists[k]
        # and to grad_min[j] is -weights_min[k] * vecs[k] / dists[k]
        unit_vecs = vecs / (dists[:, np.newaxis] + 1e-15)
        weighted_unit = weights_min[:, np.newaxis] * unit_vecs
        for k in range(n_pairs):
            grad_min[pair_i[k]] += weighted_unit[k]
            grad_min[pair_j[k]] -= weighted_unit[k]
        
        # Soft max gradient
        max_d = np.max(dists)
        exp_beta_d = np.exp(beta_max * (dists - max_d))
        weights_max = exp_beta_d / np.sum(exp_beta_d)
        grad_max = np.zeros_like(pts)
        weighted_unit_max = weights_max[:, np.newaxis] * unit_vecs
        for k in range(n_pairs):
            grad_max[pair_i[k]] -= weighted_unit_max[k]
            grad_max[pair_j[k]] += weighted_unit_max[k]
        
        return grad_min, grad_max

    # Objective: maximize soft_min / soft_max
    # Equivalent to maximize log(soft_min) - log(soft_max)
    # We minimize the negative
    def soft_ratio_objective(x, beta_min, beta_max):
        pts = x.reshape(-1, 2)
        smin, smax, dists, vecs = soft_min_max(pts, beta_min, beta_max)
        if smin < 1e-15 or smax < 1e-15:
            return 0.0
        return -(np.log(smin) - np.log(smax))

    def soft_ratio_grad(x, beta_min, beta_max):
        pts = x.reshape(-1, 2)
        smin, smax, dists, vecs = soft_min_max(pts, beta_min, beta_max)
        if smin < 1e-15 or smax < 1e-15:
            return np.zeros_like(x)
        
        grad_min, grad_max = soft_min_max_grad(pts, beta_min, beta_max)
        # d(log(smin)/d(pts) = grad_min / smin
        # d(-log(smax)/d(pts) = -grad_max / smax
        grad = grad_min / smin - grad_max / smax
        return grad.flatten()

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

    # Gradient-based local optimization with soft ratio
    def gradient_optimize_soft(init_pts, beta_start=50.0, beta_end=500.0, max_iter_per_beta=100):
        """Use L-BFGS-B with the soft ratio objective, annealing beta."""
        pts = init_pts.copy()
        score = compute_score(pts)
        
        # Anneal beta from low to high
        n_beta_steps = 8
        for beta_step in range(n_beta_steps):
            if time.time() - start_time > timeout - 20:
                break
            
            # Interpolate beta
            frac = beta_step / max(1, n_beta_steps - 1)
            beta_min = beta_start * (beta_end / beta_start) ** frac
            beta_max = beta_start * (beta_end / beta_start) ** frac
            
            x0 = pts.flatten()
            try:
                res = minimize(soft_ratio_objective, x0, method='L-BFGS-B', 
                              jac=lambda x: soft_ratio_grad(x, beta_min, beta_max),
                              args=(beta_min, beta_max),
                              options={'maxiter': max_iter_per_beta, 'ftol': 1e-14, 'gtol': 1e-12})
                new_pts = res.x.reshape(-1, 2)
                new_pts = normalize_pts(new_pts)
                new_score = compute_score(new_pts)
                
                if new_score > score:
                    pts = new_pts
                    score = new_score
            except Exception:
                pass
        
        return pts, score

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
        
        # Phase 2: Gradient polish with soft ratio objective
        if time.time() - start_time > timeout - 20:
            soft_pts, soft_score = gradient_optimize_soft(pts, beta_start=20.0, beta_end=300.0, max_iter_per_beta=80)
            if soft_score > score:
                pts = soft_pts
                score = soft_score
        
        # Phase 3: Another SA round from the improved point
        if time.time() - start_time > timeout - 15:
            sa2_pts, sa2_score = simulated_annealing(pts, max_iters=2000, T_start=0.02, T_end=0.0002)
            if sa2_score > score:
                pts = sa2_pts
                score = sa2_score
        
        # Phase 4: Final gradient polish with higher beta
        if time.time() - start_time > timeout - 10:
            final_pts, final_score = gradient_optimize_soft(pts, beta_start=100.0, beta_end=1000.0, max_iter_per_beta=50)
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
