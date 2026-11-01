"""Program for covering the sphere with 130 equal caps using smooth grid-based optimization."""
import numpy as np
import time
from scipy.optimize import minimize

N = 130

def _make_fib_grid(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

# Use a dense grid for high accuracy of the surrogate objective
GRID = _make_fib_grid(50000)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def fibonacci_spiral(n):
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    def compute_covering_radius_exact(pts):
        """Compute covering radius exactly as max circumradius over convex hull faces."""
        try:
            from scipy.spatial import ConvexHull
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
                r[r > np.pi / 2] = np.pi - r[r > np.pi / 2]
                radii[valid] = r
            
            if np.any(~valid):
                a_deg = a[~valid]
                b_deg = b[~valid]
                c_deg = c[~valid]
                d_ab = np.arccos(np.clip(np.sum(a_deg * b_deg, axis=1), -1, 1))
                d_bc = np.arccos(np.clip(np.sum(b_deg * c_deg, axis=1), -1, 1))
                d_ca = np.arccos(np.clip(np.sum(c_deg * a_deg, axis=1), -1, 1))
                radii[~valid] = np.maximum(np.maximum(d_ab, d_bc), d_ca)
            
            return float(np.max(radii))
        except Exception:
            # Fallback: dense grid-based estimate
            dot = np.dot(GRID, pts.T)
            dot = np.clip(dot, -1.0, 1.0)
            ang_dists = np.arccos(dot)
            min_dists = np.min(ang_dists, axis=1)
            return float(np.max(min_dists))

    def smooth_obj_grad(x_flat, alpha=30.0, beta=30.0):
        """
        Compute the smooth surrogate objective and its gradient.
        Objective: soft_max_j ( soft_min_k d(x_j, p_k) )
        """
        pts = x_flat.reshape(N, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1e-10
        pts = pts / norms
        
        # 1. Compute angular distances D (M x N)
        dots = np.dot(GRID, pts.T)
        dots = np.clip(dots, -1.0, 1.0)
        D = np.arccos(dots)
        
        # 2. Soft-min over points k for each grid point j
        # S_j = -1/alpha * log( sum_k exp(-alpha * D_jk) )
        exp_neg = np.exp(-alpha * D)
        sum_exp_neg = np.sum(exp_neg, axis=1, keepdims=True)
        W = exp_neg / sum_exp_neg  # Weights for soft-min (M x N)
        S = -1.0 / alpha * np.log(np.squeeze(sum_exp_neg))  # Soft-min values (M)
        
        # 3. Soft-max over grid points j
        # Val = 1/beta * log( sum_j exp(beta * S_j) )
        exp_pos = np.exp(beta * S)
        sum_exp_pos = np.sum(exp_pos)
        V = exp_pos / sum_exp_pos  # Weights for soft-max (M)
        val = 1.0 / beta * np.log(sum_exp_pos)
        
        # 4. Gradient computation
        # dVal/dp_k = sum_j V_j * (dS_j/dp_k)
        # dS_j/dp_k = sum_l W_jl * (dD_jl/dp_k)
        # dD_jl/dp_k is 0 if l != k.
        # So dS_j/dp_k = W_jk * (dD_jk/dp_k)
        # dD_jk/dp_k = - (x_j - p_k) / ||x_j - p_k|| (Euclidean direction)
        
        # A_jk = V_j * W_jk
        A = V[:, np.newaxis] * W  # M x N
        
        # Compute Euclidean differences and distances
        # diff[j, k, :] = x_j - p_k
        diff = GRID[:, np.newaxis, :] - pts[np.newaxis, :, :]  # M x N x 3
        E = np.linalg.norm(diff, axis=2)  # M x N
        E = np.maximum(E, 1e-6)  # Avoid division by zero
        
        # Unit vectors U[j, k, :] = diff[j, k, :] / E[j, k]
        U = diff / E[:, :, np.newaxis]
        
        # g_k = - sum_j A_jk * U[j, k, :]
        # g is N x 3
        g = -np.sum(A[:, :, np.newaxis] * U, axis=0)
        
        # Project gradient to tangent space
        # g_k -= (g_k dot p_k) p_k
        dots_g_p = np.sum(g * pts, axis=1, keepdims=True)
        g = g - dots_g_p * pts
        
        return float(val), g.flatten()

    def objective(x):
        val, _ = smooth_obj_grad(x)
        return val

    def gradient(x):
        _, g = smooth_obj_grad(x)
        return g

    best_points = None
    best_rad = np.inf

    # Initial guess: Fibonacci spiral
    points = fibonacci_spiral(N).astype(np.float64)
    x0 = points.flatten()

    # Optimization
    try:
        if time.time() < deadline - 20.0:
            result = minimize(
                objective,
                x0,
                jac=gradient,
                method='L-BFGS-B',
                options={'maxiter': 300, 'maxfun': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            x_opt = result.x.reshape(N, 3)
            x_opt /= np.linalg.norm(x_opt, axis=1, keepdims=True)
            
            rad = compute_covering_radius_exact(x_opt)
            if rad < best_rad:
                best_rad = rad
                best_points = x_opt.copy()
    except Exception as e:
        print(f"Optimization error: {e}")
        best_points = points.copy()
        best_rad = compute_covering_radius_exact(best_points)

    # Refinement: If time permits, do a second pass with higher temperatures or different start
    if best_points is not None and time.time() < deadline - 10.0:
        # Perturb slightly and re-optimize
        rng = np.random.RandomState(42)
        points2 = best_points + 0.01 * rng.randn(N, 3)
        points2 /= np.linalg.norm(points2, axis=1, keepdims=True)
        x0_2 = points2.flatten()
        
        try:
            result2 = minimize(
                objective,
                x0_2,
                jac=gradient,
                method='L-BFGS-B',
                options={'maxiter': 100, 'maxfun': 300, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            x_opt2 = result2.x.reshape(N, 3)
            x_opt2 /= np.linalg.norm(x_opt2, axis=1, keepdims=True)
            
            rad2 = compute_covering_radius_exact(x_opt2)
            if rad2 < best_rad:
                best_rad = rad2
                best_points = x_opt2.copy()
        except Exception:
            pass

    if best_points is None:
        best_points = points

    return best_points

# EVOLVE-BLOCK-END
