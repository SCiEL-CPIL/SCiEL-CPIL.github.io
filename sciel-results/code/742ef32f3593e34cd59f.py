import numpy as np
import math
import time
from scipy.optimize import minimize

N = 150

# EVOLVE-BLOCK-START
import numpy as np
import math
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    def get_fibonacci_points(n, jitter=0.0, seed=42):
        rng = np.random.RandomState(seed)
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        if jitter > 0:
            pts += jitter * rng.randn(n, 3)
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        return pts

    def get_random_points(n, seed=42):
        rng = np.random.RandomState(seed)
        pts = rng.randn(n, 3)
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        return pts

    def log_energy(pts):
        # Compute pairwise distances squared
        # diff shape (N, N, 3)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        # Set diagonal to 1 to avoid log(0)
        np.fill_diagonal(dist_sq, 1.0)
        # Energy is sum_{i<j} log(1/d_ij) = -0.5 * sum_{i!=j} log(d_ij)
        # log(1/d) = -log(d). d = sqrt(dist_sq). log(d) = 0.5 log(dist_sq).
        # So term is -0.5 * 0.5 * log(dist_sq) = -0.25 log(dist_sq).
        # Sum over i!=j counts each pair twice.
        # So total energy = -0.25 * sum_{i!=j} log(dist_sq[i,j])
        e = -0.25 * np.sum(np.log(dist_sq))
        return e

    def gradient(pts):
        """
        Compute gradient of log_energy w.r.t Cartesian coordinates,
        projected onto the tangent plane of the sphere.
        """
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dist_sq, 1.0)
        
        # Gradient of E w.r.t pts[i]
        # E = sum_{j != i} -0.5 log(d_ij) ... wait, let's re-derive carefully.
        # E = sum_{i<j} -log(d_ij)
        # dE/dx_i = sum_{j>i} -1/d_ij * d(d_ij)/dx_i + sum_{j<i} -1/d_ij * d(d_ij)/dx_i
        # d(d_ij)/dx_i = (x_i - x_j) / d_ij
        # So term is -1/d_ij * (x_i - x_j)/d_ij = -(x_i - x_j)/d_ij^2 = (x_j - x_i)/d_ij^2
        # So grad_i = sum_{j != i} (x_j - x_i) / ||x_j - x_i||^2
        
        # Vectorized:
        # diff[i,j] = x_i - x_j
        # We want sum_j (x_j - x_i) / dist_sq[i,j]
        # Which is - sum_j diff[i,j] / dist_sq[i,j]
        
        inv_dist_sq = 1.0 / dist_sq
        mask = ~np.eye(N, dtype=bool)
        inv_dist_sq = inv_dist_sq * mask
        
        # grad[i] = - sum_j diff[i,j] * inv_dist_sq[i,j]
        grad = -np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
        
        # Project gradient onto tangent plane: g_tan = g - (g . x) x
        dot = np.sum(grad * pts, axis=1, keepdims=True)
        proj_grad = grad - dot * pts
        
        return proj_grad

    def normalize(pts):
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        return pts / norms

    def optimize_l_bfgs_spherical(pts_init, max_time=20.0):
        """
        Optimize using L-BFGS-B on spherical coordinates (theta, phi).
        """
        best_pts = pts_init.copy()
        best_e = log_energy(pts_init)
        
        # Convert to spherical coordinates
        # z = cos(theta) => theta = arccos(z)
        # y/x = tan(phi) => phi = arctan2(y, x)
        theta = np.arccos(np.clip(pts_init[:, 2], -1.0, 1.0))
        phi = np.arctan2(pts_init[:, 1], pts_init[:, 0])
        
        # Initial parameters
        x0 = np.concatenate([theta, phi])
        
        def spherical_to_cartesian(theta, phi):
            sin_theta = np.sin(theta)
            return np.column_stack([
                sin_theta * np.cos(phi),
                sin_theta * np.sin(phi),
                np.cos(theta)
            ])
        
        def objective_and_grad(params):
            theta = params[:N]
            phi = params[N:2*N]
            
            pts = spherical_to_cartesian(theta, phi)
            e = log_energy(pts)
            
            # Compute Cartesian gradient
            grad_pts = gradient(pts)
            
            # Chain rule:
            # dpts/dtheta = [cos(theta)cos(phi), cos(theta)sin(phi), -sin(theta)]
            # dpts/dphi = [-sin(phi), cos(phi), 0]
            
            sin_theta = np.sin(theta)
            cos_theta = np.cos(theta)
            sin_phi = np.sin(phi)
            cos_phi = np.cos(phi)
            
            dpts_dtheta = np.column_stack([
                cos_theta * cos_phi,
                cos_theta * sin_phi,
                -sin_theta
            ])
            dpts_dphi = np.column_stack([
                -sin_phi,
                cos_phi,
                np.zeros(N)
            ])
            
            grad_theta = np.sum(grad_pts * dpts_dtheta, axis=1)
            grad_phi = np.sum(grad_pts * dpts_dphi, axis=1)
            
            grad = np.concatenate([grad_theta, grad_phi])
            return e, grad
        
        # Use L-BFGS-B
        try:
            # Bound theta to [0, pi] and phi to [-pi, pi] or [0, 2pi]
            # L-BFGS-B handles bounds well.
            bounds = [(0, np.pi)] * N + [(-np.pi, np.pi)] * N
            
            result = minimize(
                objective_and_grad,
                x0,
                method='L-BFGS-B',
                jac=True,
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50, 'maxfun': 10000}
            )
            
            if result.fun < best_e:
                new_theta = result.x[:N]
                new_phi = result.x[N:2*N]
                new_pts = spherical_to_cartesian(new_theta, new_phi)
                new_pts = normalize(new_pts)
                e_new = log_energy(new_pts)
                if e_new < best_e:
                    best_e = e_new
                    best_pts = new_pts
        except Exception:
            pass
        
        return best_pts, best_e

    def optimize_gradient_descent(pts, max_time=10.0):
        """
        Final polishing step with simple gradient descent and adaptive step size.
        """
        best_pts = pts.copy()
        best_e = log_energy(pts)
        lr = 1e-4
        momentum = 0.9
        v = np.zeros_like(pts)
        
        while time.time() < start_time + max_time:
            grad = gradient(pts)
            v = momentum * v - lr * grad
            pts_new = normalize(pts + v)
            
            e_new = log_energy(pts_new)
            
            if e_new < best_e:
                best_e = e_new
                best_pts = pts_new.copy()
                # Increase learning rate slightly if we are making progress
                lr *= 1.05
            else:
                # Decrease learning rate if we are not
                lr *= 0.9
                v *= 0.5
            
            if lr < 1e-8:
                break
            
            pts = pts_new
        
        return best_pts, best_e

    best_pts = None
    best_e = float('inf')
    
    # Strategy: Run multiple restarts with L-BFGS, then polish the best one.
    
    # Restart 1: Fibonacci
    if time.time() < deadline - 20.0:
        pts = get_fibonacci_points(N, seed=42)
        pts, e = optimize_l_bfgs_spherical(pts, max_time=30.0)
        if e < best_e:
            best_e = e
            best_pts = pts.copy()

    # Restart 2: Random
    if time.time() < deadline - 20.0:
        pts = get_random_points(N, seed=123)
        pts, e = optimize_l_bfgs_spherical(pts, max_time=30.0)
        if e < best_e:
            best_e = e
            best_pts = pts.copy()

    # Restart 3: Jittered Fibonacci
    if time.time() < deadline - 20.0:
        pts = get_fibonacci_points(N, jitter=0.01, seed=777)
        pts, e = optimize_l_bfgs_spherical(pts, max_time=30.0)
        if e < best_e:
            best_e = e
            best_pts = pts.copy()
            
    # Restart 4: Another Random
    if time.time() < deadline - 10.0:
        pts = get_random_points(N, seed=999)
        pts, e = optimize_l_bfgs_spherical(pts, max_time=20.0)
        if e < best_e:
            best_e = e
            best_pts = pts.copy()

    # Final Polish
    if best_pts is not None and time.time() < deadline - 2.0:
        best_pts, best_e = optimize_gradient_descent(best_pts, max_time=10.0)
        
    # Final Normalization and Check
    if best_pts is None:
        return get_fibonacci_points(N)
        
    best_pts = normalize(best_pts)
    
    # Check for invalid states
    if not np.all(np.isfinite(best_pts)):
        return get_fibonacci_points(N)

    # Verify minimum distance
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    min_dist = np.sqrt(np.min(dist_sq))
    
    if min_dist < 1e-9:
        # Perturb points that are too close
        rng = np.random.RandomState(42)
        for i in range(N):
            for j in range(i+1, N):
                if dist_sq[i,j] < 1e-18:
                    direction = rng.randn(3)
                    direction -= np.dot(direction, best_pts[i]) * best_pts[i]
                    best_pts[i] += 1e-8 * direction
                    best_pts[i] /= np.linalg.norm(best_pts[i])
                    # Re-normalize all to be safe
                    best_pts = normalize(best_pts)
                    break
            if min_dist < 1e-9:
                break

    return best_pts

# EVOLVE-BLOCK-END
