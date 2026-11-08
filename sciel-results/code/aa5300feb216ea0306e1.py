"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import random


def _compute_inradii(centers, width, height):
    """
    Compute the maximum radius for each circle center within the rectangle,
    given the constraint that circles must not overlap.
    
    r_i = min(dist_to_boundary_i, min_{j != i} d_ij / 2)
    """
    n = len(centers)
    radii = np.zeros(n)
    
    for i in range(n):
        cx, cy = centers[i]
        # Distance to rectangle boundaries
        dist_to_boundary = min(cx, width - cx, cy, height - cy)
        
        # Minimum distance to other centers
        min_dist = dist_to_boundary
        for j in range(n):
            if i == j:
                continue
            dx = cx - centers[j][0]
            dy = cy - centers[j][1]
            d = np.sqrt(dx * dx + dy * dy)
            half_d = d / 2.0
            if half_d < min_dist:
                min_dist = half_d
        
        radii[i] = min_dist
    
    return radii


def _objective_and_gradient(centers, width, height):
    """
    Compute the negative sum of inradii (for minimization) and its gradient.
    We want to maximize sum(r_i) where r_i = min(dist_to_boundary_i, min_j d_ij/2).
    """
    n = len(centers)
    total_r = 0.0
    grad = np.zeros_like(centers)
    
    for i in range(n):
        cx, cy = centers[i]
        
        # Distance to boundaries
        dist_left = cx
        dist_right = width - cx
        dist_bottom = cy
        dist_top = height - cy
        
        # Find the minimum constraint
        min_val = dist_left
        # Gradient contribution from boundary constraint
        gx = -1.0  # d(dist_left)/dx = 1, but we're maximizing, so gradient of -r is -1 * dr/dx
        gy = 0.0
        
        # Check right boundary
        if dist_right < min_val:
            min_val = dist_right
            gx = 1.0  # d(dist_right)/dx = -1, gradient of -r is -(-1) = 1
            gy = 0.0
        
        # Check bottom boundary
        if dist_bottom < min_val:
            min_val = dist_bottom
            gx = 0.0
            gy = -1.0  # d(dist_bottom)/dy = 1, gradient of -r is -1
        
        # Check top boundary
        if dist_top < min_val:
            min_val = dist_top
            gx = 0.0
            gy = 1.0  # d(dist_top)/dy = -1, gradient of -r is -(-1) = 1
        
        # Check other circles
        for j in range(n):
            if i == j:
                continue
            dx = cx - centers[j][0]
            dy = cy - centers[j][1]
            d = np.sqrt(dx * dx + dy * dy)
            if d < 1e-10:
                d = 1e-10
            half_d = d / 2.0
            
            if half_d < min_val:
                min_val = half_d
                # d(half_d)/dx = dx / (2 * d) for circle i's position
                # Actually: d_ij = sqrt((cx-cj)^2 + (cy-cj)^2)
                # d(half_d)/dx = (cx - cj) / (2 * d)
                gx = dx / (2.0 * d)
                gy = dy / (2.0 * d)
        
        total_r += min_val
        # Gradient of -total_r with respect to centers[i]
        grad[i][0] = -gx
        grad[i][1] = -gy
    
    return -total_r, grad


def _optimize_centers(centers, width, height):
    """
    Optimize center positions to maximize sum of inradii.
    """
    n = len(centers)
    centers = centers.astype(float).copy()
    
    # Use Nelder-Mead or BFGS with bounds
    # Flatten for scipy
    def obj_flat(x):
        c = x.reshape(n, 2)
        val, _ = _objective_and_gradient(c, width, height)
        return val
    
    def grad_flat(x):
        c = x.reshape(n, 2)
        _, g = _objective_and_gradient(c, width, height)
        return g.flatten()
    
    # Bounds: circles must be inside rectangle
    bounds = []
    for i in range(n):
        bounds.append((0.0, width))
        bounds.append((0.0, height))
    
    # Try multiple optimization methods
    best_val = float('inf')
    best_x = centers.flatten().copy()
    
    # Method 1: L-BFGS-B
    try:
        result = minimize(
            obj_flat, 
            centers.flatten(), 
            jac=grad_flat,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        if result.fun < best_val:
            best_val = result.fun
            best_x = result.x
    except Exception:
        pass
    
    # Method 2: Nelder-Mead (no gradient, more robust)
    try:
        result = minimize(
            obj_flat,
            centers.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-8, 'initial_simplex': 0.01}
        )
        if result.fun < best_val:
            best_val = result.fun
            best_x = result.x
    except Exception:
        pass
    
    return best_x.reshape(n, 2)


def construct():
    """
    Pack 21 circles in a rectangle of perimeter 4 (so w + h = 2).
    Maximize the sum of radii.
    """
    random.seed(42)
    np.random.seed(42)
    
    best_sum = 0
    best_centers = None
    best_width = 1.0
    best_height = 1.0
    
    # Try several aspect ratios
    aspect_ratios = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.8, 2.0, 2.5, 3.0]
    
    for ar in aspect_ratios:
        # w + h = 2, w/h = ar => h(1 + ar) = 2 => h = 2/(1+ar), w = 2*ar/(1+ar)
        h = 2.0 / (1.0 + ar)
        w = 2.0 * ar / (1.0 + ar)
        
        # Try multiple initializations
        for init_iter in range(8):
            # Different grid patterns
            if init_iter == 0:
                # 5x5 grid, take first 21
                cols, rows = 5, 5
            elif init_iter == 1:
                # 7x3 grid
                cols, rows = 7, 3
            elif init_iter == 2:
                # 3x7 grid
                cols, rows = 3, 7
            elif init_iter == 3:
                # 6x4 grid
                cols, rows = 6, 4
            elif init_iter == 4:
                # 4x6 grid
                cols, rows = 4, 6
            elif init_iter == 5:
                # Random
                cols, rows = 0, 0
            elif init_iter == 6:
                # 5x5 with offset
                cols, rows = 5, 5
            else:
                # Random with different seed
                cols, rows = 0, 0
            
            centers = []
            if cols > 0 and rows > 0:
                count = 0
                for i in range(cols):
                    for j in range(rows):
                        if count >= 21:
                            break
                        cx = (i + 0.5) * w / cols
                        cy = (j + 0.5) * h / rows
                        # Add small perturbation
                        if init_iter >= 5:
                            cx += random.uniform(-w * 0.05, w * 0.05)
                            cy += random.uniform(-h * 0.05, h * 0.05)
                        else:
                            cx += random.uniform(-w * 0.02, w * 0.02)
                            cy += random.uniform(-h * 0.02, h * 0.02)
                        cx = max(0.01, min(w - 0.01, cx))
                        cy = max(0.01, min(h - 0.01, cy))
                        centers.append([cx, cy])
                        count += 1
                    if count >= 21:
                        break
            else:
                # Random initialization
                for _ in range(21):
                    cx = random.uniform(0.05, w - 0.05)
                    cy = random.uniform(0.05, h - 0.05)
                    centers.append([cx, cy])
            
            if len(centers) < 21:
                continue
            
            centers = np.array(centers[:21])
            
            # Optimize
            optimized = _optimize_centers(centers, w, h)
            
            # Clamp to ensure valid positions
            optimized[:, 0] = np.clip(optimized[:, 0], 0.0, w)
            optimized[:, 1] = np.clip(optimized[:, 1], 0.0, h)
            
            # Compute sum of inradii
            radii = _compute_inradii(optimized, w, h)
            sum_radii = np.sum(radii)
            
            if sum_radii > best_sum:
                best_sum = sum_radii
                best_centers = optimized.copy()
                best_width = w
                best_height = h
    
    # Final verification and cleanup
    radii = _compute_inradii(best_centers, best_width, best_height)
    
    # Ensure no overlaps by adjusting radii down if needed
    for i in range(N):
        for j in range(i + 1, N):
            dx = best_centers[i][0] - best_centers[j][0]
            dy = best_centers[i][1] - best_centers[j][1]
            d = np.sqrt(dx * dx + dy * dy)
            max_r = d / 2.0
            if radii[i] > max_r:
                radii[i] = max_r
            if radii[j] > max_r:
                radii[j] = max_r
    
    # Ensure circles are inside rectangle
    for i in range(N):
        cx, cy = best_centers[i]
        max_r = min(cx, best_width - cx, cy, best_height - cy)
        if radii[i] > max_r:
            radii[i] = max_r
        radii[i] = max(0.0, radii[i])
    
    result = np.zeros((N, 3))
    for i in range(N):
        result[i][0] = best_centers[i][0]
        result[i][1] = best_centers[i][1]
        result[i][2] = radii[i]
    
    return result
# EVOLVE-BLOCK-END
