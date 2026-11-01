"""Program for covering the sphere with 130 equal caps."""
import numpy as np
import time
import scipy.optimize
from scipy.spatial import ConvexHull

N = 130

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Leave 15s margin for safety

    def fibonacci_spiral(n):
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    def normalize(pts):
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return pts / norms

    def compute_covering_radius(pts):
        """Compute exact covering radius using convex hull circumradii."""
        try:
            hull = ConvexHull(pts)
            faces = hull.simplices
            a = pts[faces[:, 0]]
            b = pts[faces[:, 1]]
            c = pts[faces[:, 2]]
            
            ab = np.cross(a, b)
            bc = np.cross(b, c)
            ca = np.cross(c, a)
            c_dir = ab + bc + ca
            norms = np.linalg.norm(c_dir, axis=1)
            
            radii = np.zeros(len(faces))
            valid = norms > 1e-10
            
            if np.any(valid):
                c_dir_valid = c_dir[valid] / norms[valid, np.newaxis]
                cos_r = np.clip(np.sum(c_dir_valid * a[valid], axis=1), -1.0, 1.0)
                r = np.arccos(cos_r)
                # If the center is on the opposite side, the cap is > 90 deg? 
                # For convex hull of points on sphere, the circumcenter of the face 
                # pointing outwards defines the empty cap.
                # The radius is the angle between the face normal and the vertices.
                # We take the max of these angles.
                radii[valid] = r
            
            if np.any(~valid):
                # Degenerate face, use max edge length
                a_deg = a[~valid]
                b_deg = b[~valid]
                c_deg = c[~valid]
                d_ab = np.arccos(np.clip(np.sum(a_deg * b_deg, axis=1), -1, 1))
                d_bc = np.arccos(np.clip(np.sum(b_deg * c_deg, axis=1), -1, 1))
                d_ca = np.arccos(np.clip(np.sum(c_deg * a_deg, axis=1), -1, 1))
                radii[~valid] = np.maximum(np.maximum(d_ab, d_bc), d_ca)
            
            return float(np.max(radii))
        except Exception:
            # Fallback: grid-based estimate
            lat = np.linspace(-np.pi/2, np.pi/2, 50)
            lon = np.linspace(0, 2*np.pi, 100, endpoint=False)
            LA, LO = np.meshgrid(lat, lon, indexing='ij')
            z = np.sin(LA)
            x = np.cos(LA) * np.cos(LO)
            y = np.cos(LA) * np.sin(LO)
            grid_pts = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
            dot = np.dot(grid_pts, pts.T)
            dot = np.clip(dot, -1.0, 1.0)
            ang_dists = np.arccos(dot)
            min_dists = np.min(ang_dists, axis=1)
            return float(np.max(min_dists))

    def get_smooth_objective_and_grad(pts, grid_pts, beta, gamma):
        """
        Compute the smooth surrogate objective and its gradient.
        Objective: (1/beta) * log(sum_g exp(beta * S_g))
        where S_g = -1/gamma * log(sum_i exp(-gamma * D_{g,i}))
        D_{g,i} = arccos(g . p_i)
        """
        # D_{g,i} shape (G, N)
        dot = np.dot(grid_pts, pts.T)
        dot = np.clip(dot, -1.0, 1.0)
        D = np.arccos(dot)
        
        # Soft-min over i for each g: S_g
        # S_g = -1/gamma * log(sum_i exp(-gamma * D_{g,i}))
        neg_gamma_D = -gamma * D
        max_neg = np.max(neg_gamma_D, axis=1, keepdims=True)
        log_sum_exp_i = max_neg + np.log(np.sum(np.exp(neg_gamma_D - max_neg), axis=1, keepdims=True))
        S = -log_sum_exp_i / gamma  # shape (G, 1)
        S = S.flatten() # shape (G,)
        
        # Soft-max over g: J
        # J = 1/beta * log(sum_g exp(beta * S_g))
        beta_S = beta * S
        max_beta_S = np.max(beta_S)
        log_sum_exp_g = max_beta_S + np.log(np.sum(np.exp(beta_S - max_beta_S)))
        J = log_sum_exp_g / beta
        
        # Gradient
        # Weights for g: w_g = exp(beta * S_g - max_beta_S) / sum(exp(...))
        exp_beta_S = np.exp(beta_S - max_beta_S)
        w_g = exp_beta_S / np.sum(exp_beta_S)
        
        # Weights for i given g: v_{g,i} = exp(-gamma * D_{g,i}) / sum_j exp(-gamma * D_{g,j})
        # We already have the terms for log_sum_exp_i
        exp_neg_gamma_D = np.exp(neg_gamma_D - max_neg)
        sum_exp_i = np.sum(exp_neg_gamma_D, axis=1, keepdims=True)
        v = exp_neg_gamma_D / sum_exp_i  # shape (G, N)
        
        # Gradient of D_{g,i} w.r.t p_i
        # D = arccos(dot). d(arccos(x))/dx = -1/sqrt(1-x^2)
        # dot = g . p_i. d(dot)/d(p_i) = g
        # grad_{p_i} D_{g,i} = -g / sqrt(1 - dot^2)
        
        # We need sum_g w_g * (1/beta) * sum_i v_{g,i} * grad_{p_i} D_{g,i}
        # Note: The chain rule for J -> S -> D
        # dJ/dp_i = (1/beta) * sum_g w_g * dS_g/dp_i
        # dS_g/dp_i = -1/gamma * d(log(sum_j exp(-gamma D_{g,j}))/dp_i)
        # = -1/gamma * (1/sum_j ...) * sum_j exp(-gamma D_{g,j}) * (-gamma) * dD_{g,j}/dp_i
        # = sum_j v_{g,j} * dD_{g,j}/dp_i
        # Since dD_{g,j}/dp_i is 0 for j != i, this simplifies to v_{g,i} * dD_{g,i}/dp_i
        
        # So grad_{p_i} J = (1/beta) * sum_g w_g * v_{g,i} * (-g / sqrt(1 - dot_{g,i}^2))
        
        # Vectorized:
        # term = w_g[:, None] * v * (-1 / sqrt(1 - dot^2))
        # grad = sum_g term * grid_pts[g, :]
        
        denom = np.sqrt(np.maximum(1.0 - dot * dot, 1e-12))
        # v_{g,i} * w_g * (-1/denom_{g,i})
        coeff = (w_g[:, None] * v) / denom
        
        # grad_i = sum_g coeff_{g,i} * grid_pts[g, :]
        # This is a matrix multiplication: coeff.T @ grid_pts
        grad = coeff.T @ grid_pts
        grad /= beta
        
        return J, grad

    best_points = None
    best_rad = np.inf

    # Grid for optimization
    # Use a moderately dense grid for the smooth optimization
    lat = np.linspace(-np.pi/2, np.pi/2, 30)
    lon = np.linspace(0, 2*np.pi, 60, endpoint=False)
    LA, LO = np.meshgrid(lat, lon, indexing='ij')
    z_grid = np.sin(LA)
    x_grid = np.cos(LA) * np.cos(LO)
    y_grid = np.cos(LA) * np.sin(LO)
    grid_pts = np.column_stack([x_grid.ravel(), y_grid.ravel(), z_grid.ravel()])
    
    # Finer grid for final checks if needed, but hull is exact.
    
    def optimize_start(pts, beta, gamma, max_iter=100):
        """Run L-BFGS-B optimization from a starting point."""
        if time.time() > deadline - 10.0:
            return pts, compute_covering_radius(pts)
        
        def fun(x):
            pts = x.reshape(N, 3)
            pts = normalize(pts)
            val, grad = get_smooth_objective_and_grad(pts, grid_pts, beta, gamma)
            return val, grad.flatten()
        
        try:
            res = scipy.optimize.minimize(
                fun, pts.flatten(), method='L-BFGS-B', jac=True,
                options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            pts_opt = normalize(res.x.reshape(N, 3))
            rad = compute_covering_radius(pts_opt)
            return pts_opt, rad
        except Exception:
            return pts, compute_covering_radius(pts)

    # Phase 1: Multiple starts with different parameters
    starts = []
    
    # 1. Fibonacci
    starts.append(fibonacci_spiral(N))
    
    # 2. Random
    rng = np.random.RandomState(42)
    p_rand = rng.randn(N, 3)
    starts.append(normalize(p_rand))
    
    # 3. Perturbed Fibonacci
    p_pert = fibonacci_spiral(N) + 0.2 * rng.randn(N, 3)
    starts.append(normalize(p_pert))
    
    # 4. Another random
    p_rand2 = rng.randn(N, 3)
    starts.append(normalize(p_rand2))

    # Parameters for annealing: start smooth (low beta, low gamma) then sharpen
    # beta controls the softness of the max over grid points
    # gamma controls the softness of the min over sites
    schedules = [
        (5.0, 5.0, 100),   # Smooth
        (10.0, 10.0, 100), # Medium
        (20.0, 20.0, 100), # Sharp
        (40.0, 40.0, 100), # Very Sharp
    ]

    for start_idx, start_pts in enumerate(starts):
        if time.time() > deadline - 30.0:
            break
            
        current_pts = start_pts.copy()
        
        for beta, gamma, max_iter in schedules:
            if time.time() > deadline - 10.0:
                break
            
            current_pts, rad = optimize_start(current_pts, beta, gamma, max_iter)
            
            if rad < best_rad:
                best_rad = rad
                best_points = current_pts.copy()

    # Phase 2: Refinement on the best found
    if best_points is not None:
        # Try a few more random perturbations of the best point
        for _ in range(5):
            if time.time() > deadline - 10.0:
                break
            rng = np.random.RandomState(int(time.time() * 1000) % 10000)
            perturbed = best_points + 0.05 * rng.randn(N, 3)
            perturbed = normalize(perturbed)
            
            # Use a sharp schedule for quick refinement
            for beta, gamma, max_iter in [(30.0, 30.0, 50)]:
                perturbed, rad = optimize_start(perturbed, beta, gamma, max_iter)
                if rad < best_rad:
                    best_rad = rad
                    best_points = perturbed.copy()

    if best_points is None:
        best_points = fibonacci_spiral(N)
        
    return best_points
