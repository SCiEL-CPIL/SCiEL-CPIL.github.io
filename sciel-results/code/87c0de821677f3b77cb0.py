# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the points.

    """
    start_time = time.time()
    deadline = start_time + 590  # safety margin

    n = 16

    def score(pts):
        """Compute R = (d_min / d_max)^2"""
        d = pdist(pts, 'euclidean')
        d_min = np.min(d)
        d_max = np.max(d)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def neg_ratio_flat(flat_pts):
        """Nonsmooth objective for L-BFGS-B"""
        pts = flat_pts.reshape(n, 2)
        d = pdist(pts, 'euclidean')
        d_min = np.min(d)
        d_max = np.max(d)
        if d_max == 0:
            return 0.0
        return -(d_min / d_max) ** 2

    def soft_neg_ratio_flat(flat_pts, beta=100.0):
        """
        Smooth surrogate using log-sum-exp for soft-min and soft-max.
        """
        pts = flat_pts.reshape(n, 2)
        d = pdist(pts, 'euclidean')
        
        d_max_val = np.max(d)
        d_min_val = np.min(d)
        
        # Soft min approximation
        exp_neg = np.exp(-(d - d_min_val) / beta)
        soft_min = d_min_val - beta * np.log(np.sum(exp_neg))
        
        # Soft max approximation
        exp_pos = np.exp((d - d_max_val) / beta)
        soft_max = d_max_val + beta * np.log(np.sum(exp_pos))
        
        if soft_max <= 0:
            return 0.0
        return -(soft_min / soft_max) ** 2

    def improve_smooth(pts, beta_start=100.0, max_iter=1500):
        """Local optimization using L-BFGS-B on smooth surrogate, annealing beta."""
        x0 = pts.flatten()
        best_pts = pts.copy()
        best_val = score(best_pts)
        
        # Anneal beta from large to small
        betas = [beta_start, beta_start/2, beta_start/4, beta_start/8, beta_start/16]
        
        for beta in betas:
            if time.time() > deadline - 60:
                break
            res = minimize(
                lambda x: soft_neg_ratio_flat(x, beta=beta),
                x0,
                method='L-BFGS-B',
                options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
            )
            new_pts = res.x.reshape(n, 2)
            new_val = score(new_pts)
            if new_val > best_val:
                best_val = new_val
                best_pts = new_pts
                x0 = new_pts.flatten()
        
        return best_pts

    def improve_nonsmooth(pts, max_iter=3000, tol=1e-12):
        """Local optimization using L-BFGS-B on nonsmooth objective."""
        x0 = pts.flatten()
        res = minimize(
            neg_ratio_flat,
            x0,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-10, 'maxls': 50}
        )
        best_pts = res.x.reshape(n, 2)
        return best_pts

    def make_hexagonal_initial():
        """Create a hexagonal lattice-based initial configuration using greedy farthest-point sampling."""
        s = np.sqrt(3) / 2
        candidates = []
        for i in range(-3, 4):
            for j in range(-3, 4):
                x = i + j * 0.5
                y = j * s
                candidates.append([x, y])
        candidates = np.array(candidates)
        candidates -= np.mean(candidates, axis=0)
        
        # Greedy farthest-point sampling to select 16 points
        centroid = np.mean(candidates, axis=0)
        dists_centroid = np.linalg.norm(candidates - centroid, axis=1)
        
        selected = [int(np.argmax(dists_centroid))]
        remaining = list(set(range(len(candidates))) - set(selected))
        
        while len(selected) < n and remaining:
            sel_pts = candidates[selected]
            best_idx = -1
            best_min_dist = -1
            for idx in remaining:
                pt = candidates[idx]
                min_d = np.min(np.linalg.norm(sel_pts - pt, axis=1))
                if min_d > best_min_dist:
                    best_min_dist = min_d
                    best_idx = idx
            if best_idx == -1:
                break
            selected.append(best_idx)
            remaining.remove(best_idx)
        
        initial_pts = candidates[selected]
        initial_pts -= np.mean(initial_pts, axis=0)
        return initial_pts

    def make_ring_initial():
        """Create concentric ring configuration."""
        pts = []
        pts.append([0.0, 0.0])
        for k in range(6):
            angle = k * 2 * np.pi / 6
            pts.append([np.cos(angle), np.sin(angle)])
        for k in range(9):
            angle = k * 2 * np.pi / 9
            r = 2.0
            pts.append([r * np.cos(angle), r * np.sin(angle)])
        pts = np.array(pts[:n])
        pts -= np.mean(pts, axis=0)
        return pts

    def make_fibonacci_initial():
        """Create a Fibonacci-like distribution in 2D (golden angle spiral)."""
        golden_angle = np.pi * (3 - np.sqrt(5))
        pts = np.zeros((n, 2))
        for i in range(n):
            r = np.sqrt((i + 0.5) / n)
            theta = i * golden_angle
            pts[i] = [r * np.cos(theta), r * np.sin(theta)]
        pts -= np.mean(pts, axis=0)
        return pts

    # Generate diverse initial configurations
    rng = np.random.RandomState(42)
    
    # Base configurations
    base_configs = [
        make_hexagonal_initial(),
        make_fibonacci_initial(),
        make_ring_initial()
    ]
    
    # Create perturbed versions of each base configuration
    initial_configs = []
    for base in base_configs:
        initial_configs.append(base)
        # Add 5 perturbed versions with varying noise scales
        for noise_scale in [0.05, 0.1, 0.15, 0.2, 0.3]:
            perturbed = base + rng.randn(n, 2) * noise_scale
            perturbed -= np.mean(perturbed, axis=0)
            initial_configs.append(perturbed)
    
    # Add some fully random configurations
    for _ in range(5):
        angles = rng.uniform(0, 2*np.pi, n)
        radii = rng.uniform(0.2, 1.5, n)
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        pts -= np.mean(pts, axis=0)
        initial_configs.append(pts)
    
    # Optimize each configuration
    best_pts = None
    best_val = -np.inf
    
    for i, init_pts in enumerate(initial_configs):
        if time.time() > deadline - 120:
            break
        
        # Smooth optimization
        opt_pts = improve_smooth(init_pts, beta_start=100.0, max_iter=1000)
        val = score(opt_pts)
        
        if val > best_val:
            best_val = val
            best_pts = opt_pts
        
        # Nonsmooth refinement if time permits
        if time.time() < deadline - 60:
            opt_pts2 = improve_nonsmooth(opt_pts, max_iter=2000, tol=1e-12)
            val2 = score(opt_pts2)
            if val2 > best_val:
                best_val = val2
                best_pts = opt_pts2
    
    # Final high-precision polish
    if best_pts is not None and time.time() < deadline - 30:
        final_pts = improve_nonsmooth(best_pts, max_iter=5000, tol=1e-14)
        final_val = score(final_pts)
        if final_val > best_val:
            best_pts = final_pts
            best_val = final_val
    
    # Ensure we have a valid result
    if best_pts is None:
        best_pts = make_hexagonal_initial()
    
    return best_pts

# EVOLVE-BLOCK-END
