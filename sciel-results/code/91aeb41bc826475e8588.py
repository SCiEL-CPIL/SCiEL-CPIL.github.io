"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import math
import time


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    def get_fibonacci_points(n):
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        return pts

    def log_energy(pts):
        # Compute pairwise squared distances
        # diff[i, j, k] = pts[i, k] - pts[j, k]
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        # Avoid log(0) on diagonal
        np.fill_diagonal(dist_sq, 1.0)
        # Energy = sum_{i<j} log(1/d) = -0.5 * sum_{i!=j} log(d) = -0.25 * sum_{i!=j} log(d^2)
        # Note: sum over i!=j counts each pair twice.
        # So -0.25 * sum_{i!=j} log(d_ij^2) is correct for sum_{i<j} log(1/d_ij)
        e = -0.25 * np.sum(np.log(dist_sq))
        return e

    def gradient(pts):
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dist_sq, 1.0)
        
        # Gradient of log(1/d) wrt x_i is (x_i - x_j) / d^2 ?
        # Let's derive: E = sum_{j != i} -log(||x_i - x_j||)
        # dE/dx_i = sum_{j != i} - (1/||x_i - x_j||) * (x_i - x_j)/||x_i - x_j||
        # = sum_{j != i} -(x_i - x_j) / ||x_i - x_j||^2
        # Wait, the energy is sum log(1/d) = -sum log(d).
        # d(-log d)/dx = -(1/d) * (x - y)/d = -(x-y)/d^2.
        # So grad_i = sum_{j != i} -(pts[i] - pts[j]) / dist_sq[i, j]
        # = sum_{j != i} (pts[j] - pts[i]) / dist_sq[i, j]
        
        # diff[i, j] = pts[i] - pts[j]
        # So we want -diff[i, j] / dist_sq[i, j]
        
        inv_dist_sq = 1.0 / dist_sq
        mask = ~np.eye(N, dtype=bool)
        inv_dist_sq = inv_dist_sq * mask
        
        # grad_i = sum_j (pts[j] - pts[i]) / dist_sq[i, j]
        # = sum_j -diff[i, j] * inv_dist_sq[i, j]
        
        grad = -np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
        
        # Project to tangent space: grad - (grad . x_i) x_i
        dot = np.sum(grad * pts, axis=1, keepdims=True)
        proj_grad = grad - dot * pts
        
        return proj_grad

    def normalize(pts):
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        return pts / norms

    def optimize(pts):
        best_pts = pts.copy()
        best_e = log_energy(pts)
        
        # Adaptive parameters
        lr = 0.01
        momentum = 0.9
        v = np.zeros_like(pts)
        
        # Backtracking line search parameters
        c1 = 1e-4
        rho = 0.5
        
        while time.time() < deadline - 2.0:
            grad = gradient(pts)
            
            # Update velocity
            v = momentum * v - lr * grad
            direction = v
            
            # Check if direction is zero
            dir_norm = np.linalg.norm(direction)
            if dir_norm < 1e-10:
                # If gradient is small, we might be at a minimum or stuck
                # Try reducing momentum or just break if energy is very low
                if np.linalg.norm(grad) < 1e-8:
                    break
                else:
                    # Reset velocity to gradient direction if stuck
                    v = -lr * grad
                    direction = v
                    dir_norm = np.linalg.norm(direction)
                    if dir_norm < 1e-10:
                        break

            # Normalize direction to unit length for line search? 
            # Or just use it as is. Let's use raw direction.
            
            e_curr = log_energy(pts)
            
            # Line search
            alpha = 1.0
            max_backtracks = 10
            backtracks = 0
            new_pts = None
            
            while backtracks < max_backtracks:
                # Move along direction
                candidate = pts + alpha * direction
                candidate = normalize(candidate)
                
                e_cand = log_energy(candidate)
                
                # Armijo condition: e_cand <= e_curr + c1 * alpha * (grad . direction)
                # Note: grad is projected, so grad . direction is the directional derivative
                slope = np.sum(grad * direction)
                
                if e_cand <= e_curr + c1 * alpha * slope:
                    new_pts = candidate
                    break
                else:
                    alpha *= rho
                    backtracks += 1
            
            if new_pts is None:
                # If line search fails, accept the step anyway but reduce learning rate
                new_pts = normalize(pts + alpha * direction)
                e_new = log_energy(new_pts)
                if e_new < best_e:
                    best_e = e_new
                    best_pts = new_pts.copy()
                # Reduce learning rate if we are stuck
                lr *= 0.5
                if lr < 1e-6:
                    break
            else:
                e_new = log_energy(new_pts)
                if e_new < best_e:
                    best_e = e_new
                    best_pts = new_pts.copy()
                
                # Adapt learning rate
                # If we didn't backtrack much, increase lr slightly
                # If we backtracked a lot, decrease lr
                if backtracks == 0:
                    lr *= 1.05
                elif backtracks > 2:
                    lr *= 0.9
                
                # Cap lr
                lr = np.clip(lr, 1e-6, 0.1)
                
                pts = new_pts
                # Reset momentum if we took a large step? Or keep it.
                # Keeping momentum is standard.
        
        return best_pts, best_e

    # Phase 1: Initial optimization
    pts = get_fibonacci_points(N)
    pts, e1 = optimize(pts)

    # Phase 2: Refinement with lower momentum and smaller steps
    # This helps to settle into the local minimum more precisely
    # We can run another round with different parameters
    if time.time() < deadline - 10.0:
        # Reset velocity
        v = np.zeros_like(pts)
        lr = 1e-4
        momentum = 0.5
        best_pts = pts.copy()
        best_e = e1
        
        while time.time() < deadline - 2.0:
            grad = gradient(pts)
            
            v = momentum * v - lr * grad
            direction = v
            dir_norm = np.linalg.norm(direction)
            if dir_norm < 1e-10:
                break
            
            e_curr = log_energy(pts)
            alpha = 1.0
            max_backtracks = 10
            backtracks = 0
            new_pts = None
            
            while backtracks < max_backtracks:
                candidate = pts + alpha * direction
                candidate = normalize(candidate)
                e_cand = log_energy(candidate)
                slope = np.sum(grad * direction)
                
                if e_cand <= e_curr + 1e-4 * alpha * slope:
                    new_pts = candidate
                    break
                else:
                    alpha *= 0.5
                    backtracks += 1
            
            if new_pts is None:
                new_pts = normalize(pts + alpha * direction)
                e_new = log_energy(new_pts)
                if e_new < best_e:
                    best_e = e_new
                    best_pts = new_pts.copy()
                lr *= 0.5
                if lr < 1e-8:
                    break
            else:
                e_new = log_energy(new_pts)
                if e_new < best_e:
                    best_e = e_new
                    best_pts = new_pts.copy()
                
                if backtracks == 0:
                    lr *= 1.1
                else:
                    lr *= 0.8
                lr = np.clip(lr, 1e-8, 1e-2)
                pts = new_pts

        best_pts = normalize(best_pts)
        return best_pts

    return normalize(pts)

# EVOLVE-BLOCK-END
