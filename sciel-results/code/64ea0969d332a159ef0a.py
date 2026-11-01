"""Program for covering the sphere with 130 equal caps using enhanced grid-based optimization."""
import numpy as np
import time
from scipy.optimize import minimize

N = 130

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s hard limit

    def fibonacci_spiral(n):
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    def generate_grid(n_lat, n_lon):
        lat = np.linspace(-np.pi/2, np.pi/2, n_lat)
        lon = np.linspace(0, 2*np.pi, n_lon, endpoint=False)
        LA, LO = np.meshgrid(lat, lon, indexing='ij')
        z = np.sin(LA)
        x = np.cos(LA) * np.cos(LO)
        y = np.cos(LA) * np.sin(LO)
        return np.column_stack([x.ravel(), y.ravel(), z.ravel()])

    # Pre-compute grids of different resolutions
    # Coarse grid for initial Lloyd's iterations (fast)
    grid_coarse = generate_grid(60, 120)
    # Medium grid for intermediate optimization
    grid_med = generate_grid(100, 200)
    # Fine grid for final polishing (accurate)
    grid_fine = generate_grid(150, 300)

    def covering_radius(pts, grid):
        dots = np.dot(grid, pts.T)
        dots = np.clip(dots, -1.0, 1.0)
        ang = np.arccos(dots)
        return float(np.max(np.min(ang, axis=1)))

    def soft_max_covering_radius(pts, beta, grid):
        dots = np.dot(grid, pts.T)
        dots = np.clip(dots, -1.0, 1.0)
        ang = np.arccos(dots)
        min_d = np.min(ang, axis=1)
        m = np.max(min_d)
        # Log-sum-exp trick for stability
        return m + np.log(np.sum(np.exp(beta * (min_d - m)))) / beta

    def soft_max_covering_radius_grad(pts, beta, grid):
        """Compute soft covering radius and its gradient w.r.t. pts."""
        dots = np.dot(grid, pts.T)  # (G, N)
        dots_clipped = np.clip(dots, -1.0, 1.0)
        ang = np.arccos(dots_clipped)  # (G, N)
        min_d = np.min(ang, axis=1)  # (G,)
        nearest_idx = np.argmin(ang, axis=1)  # (G,)
        
        m = np.max(min_d)
        # Soft max weight for each grid point
        exps = np.exp(beta * (min_d - m))
        w = exps / np.sum(exps)  # (G,)
        
        val = m + np.log(np.sum(exps)) / beta
        
        # Gradient: d(val)/d(pts[i])
        # val = m + (1/beta) * log(sum_j exp(beta * min_d_j))
        # d(val)/d(pts[i]) = (1/beta) * sum_j w_j * d(min_d_j)/d(pts[i])
        # min_d_j = ang[j, nearest_idx[j]]
        # d(ang)/d(pts[i]) = -d(dots)/d(pts[i]) / sqrt(1 - dots^2) = -grid[j] / sqrt(1 - dots^2)
        
        grad = np.zeros_like(pts)
        # Only grid points whose nearest is i contribute
        for i in range(N):
            mask = (nearest_idx == i)
            if not np.any(mask):
                continue
            j_idx = np.where(mask)[0]
            # For these grid points, d(min_d_j)/d(pts[i]) = -grid[j] / sqrt(1 - dots[j,i]^2)
            d_dots = dots_clipped[j_idx, i]
            denom = np.sqrt(np.maximum(1.0 - d_dots * d_dots, 1e-12))
            # d(ang)/d(pts[i]) = -grid[j] / denom
            # So d(min_d_j)/d(pts[i]) = -grid[j] / denom
            contrib = -grid[j_idx] / denom[:, None]  # (n_j, 3)
            grad[i] = np.sum(w[j_idx][:, None] * contrib, axis=0) / beta
        
        return val, grad

    def lloyd_step(pts, grid, alpha=0.5, temp=2.0):
        """
        Perform one Lloyd's iteration step.
        For each point, find the nearest grid points (Voronoi cell approximation).
        Move the point towards the weighted centroid of its cell, emphasizing far points.
        """
        dots = np.dot(grid, pts.T)
        nearest = np.argmax(dots, axis=1)
        
        new_pts = np.zeros_like(pts)
        for i in range(N):
            mask = (nearest == i)
            if not np.any(mask):
                new_pts[i] = pts[i]
                continue
            
            cell_pts = grid[mask]
            # Calculate angular distances from current point to cell points
            d = np.dot(pts[i], cell_pts.T)
            d = np.clip(d, -1.0, 1.0)
            ang = np.arccos(d)
            
            # Weight by exponential of angular distance to emphasize farthest points in the cell
            # This pushes the point towards the "center" of the largest empty cap in its cell
            w = np.exp(-ang * temp) 
            w_sum = np.sum(w)
            if w_sum < 1e-15:
                new_pts[i] = pts[i]
                continue
            w /= w_sum
            
            new_c = np.sum(cell_pts * w[:, None], axis=0)
            nrm = np.linalg.norm(new_c)
            if nrm < 1e-10:
                new_pts[i] = pts[i]
            else:
                new_pts[i] = new_c / nrm
        
        # Interpolate to avoid oscillations
        current = (1 - alpha) * pts + alpha * new_pts
        norms = np.linalg.norm(current, axis=1, keepdims=True)
        current /= np.maximum(norms, 1e-10)
        return current

    def gap_fill_step(pts, grid, alpha=0.5):
        """
        Identify the worst point on the grid (max distance to nearest config point)
        and move the nearest config point towards it.
        """
        dots = np.dot(grid, pts.T)
        dots = np.clip(dots, -1.0, 1.0)
        ang = np.arccos(dots)
        min_d = np.min(ang, axis=1)
        nearest_idx = np.argmin(ang, axis=1)
        
        # Find the grid point with the maximum distance
        worst_grid_idx = np.argmax(min_d)
        worst_grid_pt = grid[worst_grid_idx]
        nearest_config_idx = nearest_idx[worst_grid_idx]
        
        new_pts = pts.copy()
        # Move the nearest config point towards the worst grid point
        p = pts[nearest_config_idx]
        # Vector addition in tangent space approximation
        v = worst_grid_pt - p
        # Project v onto tangent plane of p
        v_tan = v - np.dot(v, p) * p
        # Move in direction of v_tan
        new_p = p + alpha * v_tan
        new_p /= np.linalg.norm(new_p)
        new_pts[nearest_config_idx] = new_p
        
        return new_pts

    def optimize_smooth(pts, beta, grid, max_iter=50, tol=1e-8):
        if time.time() > deadline - 10.0:
            return pts
        
        x0 = pts.flatten()
        
        def obj(x):
            if time.time() > deadline - 5.0:
                return 1e10
            return soft_max_covering_radius(x.reshape(N, 3), beta, grid)
        
        def obj_grad(x):
            if time.time() > deadline - 5.0:
                return 1e10, np.zeros_like(x)
            val, grad = soft_max_covering_radius_grad(x.reshape(N, 3), beta, grid)
            return val, grad.flatten()
        
        try:
            res = minimize(obj_grad, x0, method='L-BFGS-B',
                           options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-6, 'maxls': 50})
            if res.fun < obj(x0):
                p = res.x.reshape(N, 3)
                norms = np.linalg.norm(p, axis=1, keepdims=True)
                p /= np.maximum(norms, 1e-10)
                return p
        except Exception:
            pass
        return pts

    best_pts = None
    best_rad = np.inf

    # Strategy: Multiple restarts with different initializations and optimization paths
    n_restarts = 15
    
    for r in range(n_restarts):
        if time.time() > deadline - 30.0:
            break
            
        # Initialization
        if r == 0:
            pts = fibonacci_spiral(N)
        elif r == 1:
            rng = np.random.RandomState(42)
            pts = rng.randn(N, 3)
            pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        elif r == 2:
            rng = np.random.RandomState(123)
            pts = rng.randn(N, 3)
            pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        elif r == 3:
            rng = np.random.RandomState(789)
            pts = rng.randn(N, 3)
            pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        elif r >= 4 and best_pts is not None:
            # Perturb the best found so far
            rng = np.random.RandomState(1000 + r)
            scale = 0.05 * (1.0 - r / (n_restarts + 1.0))
            pts = best_pts + scale * rng.randn(N, 3)
            pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        else:
            pts = fibonacci_spiral(N)

        # Stage 1: Coarse Lloyd's iterations
        for i in range(20):
            if time.time() > deadline - 25.0:
                break
            pts = lloyd_step(pts, grid_coarse, alpha=0.5, temp=2.0)
            
        # Stage 2: Gap filling and Medium grid Lloyd's
        for i in range(10):
            if time.time() > deadline - 20.0:
                break
            pts = lloyd_step(pts, grid_med, alpha=0.4, temp=3.0)
            if i % 5 == 0:
                pts = gap_fill_step(pts, grid_med, alpha=0.2)

        # Stage 3: Smooth optimization on Medium grid
        pts = optimize_smooth(pts, beta=50, grid=grid_med, max_iter=30)
        
        # Stage 4: Fine grid Lloyd's and Gap filling
        for i in range(10):
            if time.time() > deadline - 15.0:
                break
            pts = lloyd_step(pts, grid_fine, alpha=0.3, temp=4.0)
            if i % 3 == 0:
                pts = gap_fill_step(pts, grid_fine, alpha=0.1)

        # Stage 5: Final Smooth optimization on Fine grid
        pts = optimize_smooth(pts, beta=100, grid=grid_fine, max_iter=25)
        
        # Stage 6: Additional Lloyd's + gap fill polish
        for i in range(5):
            if time.time() > deadline - 12.0:
                break
            pts = lloyd_step(pts, grid_fine, alpha=0.2, temp=5.0)
            pts = gap_fill_step(pts, grid_fine, alpha=0.08)
        
        # Stage 7: Very high beta smooth optimization for final precision
        pts = optimize_smooth(pts, beta=150, grid=grid_fine, max_iter=20)
        
        rad = covering_radius(pts, grid_fine)
        
        if rad < best_rad:
            best_rad = rad
            best_pts = pts.copy()
            # print(f"Restart {r}: {np.degrees(best_rad):.6f} deg")

    # Final polish if time permits
    if best_pts is not None and time.time() < deadline - 5.0:
        # One more round of gap filling and fine optimization with higher beta
        for i in range(5):
            if time.time() > deadline - 3.0:
                break
            best_pts = gap_fill_step(best_pts, grid_fine, alpha=0.05)
            best_pts = optimize_smooth(best_pts, beta=200, grid=grid_fine, max_iter=15)
            # Additional Lloyd's polish
            best_pts = lloyd_step(best_pts, grid_fine, alpha=0.2, temp=5.0)
            
        rad = covering_radius(best_pts, grid_fine)
        if rad < best_rad:
            best_rad = rad

    if best_pts is None:
        best_pts = fibonacci_spiral(N)

    return best_pts
