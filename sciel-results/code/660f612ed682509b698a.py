import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

def min_max_dist_dim3_14() -> np.ndarray:
    n = 14
    d = 3
    start_time = time.time()
    deadline = start_time + 580  # safety margin for 630s kill

    def score(pts):
        """Compute (d_min / d_max)^2 for the point configuration."""
        pts = np.asarray(pts, dtype=float)
        if pts.shape != (n, d) or not np.all(np.isfinite(pts)):
            return 0.0
        dists = pdist(pts)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-12:
            return 0.0
        return (d_min / d_max) ** 2

    def smooth_objective(x, lam_min, lam_max):
        """
        Smooth objective: -log(d_min/d_max) using log-sum-exp approximations.
        
        d_min ≈ (1/lam_min) * log(sum(exp(lam_min * d_i)))  -- actually this gives max
        We want: -log(d_min) + log(d_max)
        
        log(d_max) ≈ (1/lam_max) * log(sum(exp(lam_max * d_i)))
        log(d_min) ≈ -(1/lam_min) * log(sum(exp(-lam_min * d_i)))
        
        So objective = -(1/lam_max)*log(sum(exp(lam_max*d_i))) + (1/lam_min)*log(sum(exp(-lam_min*d_i)))
        Minimizing this maximizes d_min/d_max.
        """
        pts = x.reshape((n, d))
        pts = pts - np.mean(pts, axis=0)
        
        dists = pdist(pts)
        if len(dists) == 0:
            return 0.0
        if np.any(dists < 1e-12):
            return 1e6
        
        # log(d_max) approximation
        max_d = np.max(dists)
        log_d_max = (1.0 / lam_max) * (max_d + np.log(np.sum(np.exp(lam_max * (dists - max_d)))))
        
        # log(d_min) approximation  
        min_d = np.min(dists)
        log_d_min = -(1.0 / lam_min) * (min_d + np.log(np.sum(np.exp(-lam_min * (dists - min_d)))))
        
        # We want to minimize: -log(d_min) + log(d_max) = log(d_max) - log(d_min)
        obj = log_d_max - log_d_min
        
        # Add small penalty for large spread to keep points bounded
        spread_penalty = 0.001 * np.var(dists)
        
        return obj + spread_penalty

    def smooth_gradient(x, lam_min, lam_max):
        """Numerical gradient of the smooth objective."""
        pts = x.reshape((n, d))
        pts_centered = pts - np.mean(pts, axis=0)
        
        dists = pdist(pts_centered)
        if len(dists) == 0:
            return np.zeros_like(x)
        if np.any(dists < 1e-12):
            return np.zeros_like(x)
        
        eps = 1e-6
        grad = np.zeros_like(x)
        
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += eps
            x_minus = x.copy()
            x_minus[i] -= eps
            grad[i] = (smooth_objective(x_plus, lam_min, lam_max) - 
                       smooth_objective(x_minus, lam_min, lam_max)) / (2 * eps)
        
        return grad

    def fibonacci_sphere_init(n_pts, seed=42):
        """Generate initial points using Fibonacci sphere."""
        rng = np.random.RandomState(seed)
        indices = np.arange(0, n_pts, dtype=float) + 0.5
        phi = np.arccos(1 - 2 * indices / n_pts)
        theta = np.pi * (1 + 5**0.5) * indices
        
        x = np.cos(theta) * np.sin(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(phi)
        
        pts = np.column_stack([x, y, z])
        
        # Add small perturbation
        pert = rng.randn(n_pts, d) * 0.05
        pts = pts + pert
        
        return pts

    def icosahedron_based_init(n_pts, seed=42):
        """Generate initial points based on icosahedron vertices plus perturbations."""
        rng = np.random.RandomState(seed)
        t = (1 + np.sqrt(5)) / 2
        
        # Icosahedron vertices (12 points)
        ico_verts = np.array([
            [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
            [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
            [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
        ], dtype=float)
        
        # Normalize to unit sphere
        ico_verts = ico_verts / np.linalg.norm(ico_verts, axis=1, keepdims=True)
        
        if n_pts == 12:
            pts = ico_verts
        elif n_pts == 14:
            # Add 2 more points at opposite positions with perturbation
            extra1 = rng.randn(d)
            extra1 = extra1 / np.linalg.norm(extra1)
            extra2 = -extra1 + rng.randn(d) * 0.1
            extra2 = extra2 / np.linalg.norm(extra2)
            pts = np.vstack([ico_verts, extra1, extra2])
        else:
            # For other n, use fibonacci
            return fibonacci_sphere_init(n_pts, seed)
        
        # Add perturbation
        pts = pts + rng.randn(n_pts, d) * 0.03
        
        return pts

    def optimize_from_start(pts, lam_min, lam_max, max_iter=200):
        """Optimize from a starting configuration using BFGS."""
        x0 = pts.flatten()
        
        try:
            result = minimize(
                smooth_objective,
                x0,
                args=(lam_min, lam_max),
                jac=smooth_gradient,
                method='L-BFGS-B',
                options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-6}
            )
            if result.success or result.fun < smooth_objective(x0, lam_min, lam_max):
                opt_pts = result.x.reshape((n, d))
                opt_pts = opt_pts - np.mean(opt_pts, axis=0)
                return opt_pts
        except Exception:
            pass
        
        return pts

    # Generate multiple initial configurations
    rng = np.random.RandomState(12345)
    candidates = []
    
    # Fibonacci sphere with different seeds
    for seed in range(8):
        candidates.append(fibonacci_sphere_init(n, seed=seed))
    
    # Icosahedron-based
    for seed in range(5):
        candidates.append(icosahedron_based_init(n, seed=seed))
    
    # Random configurations on sphere
    for i in range(8):
        pts = rng.randn(n, d)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        pts = pts / norms
        candidates.append(pts)
    
    # Mix of sphere and interior points
    for i in range(4):
        pts = np.zeros((n, d))
        for j in range(n):
            p = rng.randn(d)
            p = p / np.linalg.norm(p)
            # Some points slightly inside
            if j < 10:
                pts[j] = p
            else:
                pts[j] = p * rng.uniform(0.5, 0.9)
        candidates.append(pts)
    
    # Evaluate all candidates
    best_score = -np.inf
    best_points = None
    
    for cand in candidates:
        if time.time() > deadline - 150:
            break
        s = score(cand)
        if s > best_score:
            best_score = s
            best_points = cand.copy()
    
    if best_points is None:
        best_points = candidates[0]
    
    # Multi-stage optimization with increasing lambda values
    # Start with lower lambda (smoother) and increase for sharper min/max detection
    lambda_schedule = [
        (5.0, 5.0),    # Stage 1: smooth
        (10.0, 10.0),  # Stage 2: medium
        (20.0, 20.0),  # Stage 3: sharper
        (40.0, 40.0),  # Stage 4: sharp
        (80.0, 80.0),  # Stage 5: very sharp
    ]
    
    current_best = best_points.copy()
    
    for lam_min, lam_max in lambda_schedule:
        if time.time() > deadline - 60:
            break
        
        # Optimize from current best
        opt_pts = optimize_from_start(current_best, lam_min, lam_max, max_iter=300)
        s = score(opt_pts)
        if s > best_score:
            best_score = s
            best_points = opt_pts.copy()
            current_best = opt_pts.copy()
        
        # Also try from several other starting points
        for i in range(5):
            if time.time() > deadline - 40:
                break
            start_idx = (i * 7) % len(candidates)
            perturbed = candidates[start_idx] + rng.randn(n, d) * 0.02
            opt_pts2 = optimize_from_start(perturbed, lam_min, lam_max, max_iter=200)
            s2 = score(opt_pts2)
            if s2 > best_score:
                best_score = s2
                best_points = opt_pts2.copy()
                current_best = opt_pts2.copy()
    
    # Final refinement with very high lambda
    if time.time() < deadline - 30:
        final_opt = optimize_from_start(best_points, 100.0, 100.0, max_iter=500)
        s_final = score(final_opt)
        if s_final > best_score:
            best_score = s_final
            best_points = final_opt.copy()
    
    # Try a few more perturbations at the end
    for i in range(3):
        if time.time() > deadline - 15:
            break
        perturbed = best_points + rng.randn(n, d) * 0.01
        opt_pts = optimize_from_start(perturbed, 50.0, 50.0, max_iter=300)
        s = score(opt_pts)
        if s > best_score:
            best_score = s
            best_points = opt_pts.copy()
    
    # Final normalization
    best_points = best_points - np.mean(best_points, axis=0)
    norm_max = np.max(np.linalg.norm(best_points, axis=1))
    if norm_max > 0:
        best_points = best_points / norm_max * 100
    
    if not np.all(np.isfinite(best_points)):
        best_points = np.zeros((n, d))
    
    return best_points
