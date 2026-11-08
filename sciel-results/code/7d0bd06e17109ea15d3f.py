"""Rewritten program for circle packing in perimeter-4 rectangle."""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Voronoi
import time

N = 21
PERIMETER = 4.0
TIME_LIMIT = 145.0  # Safety margin before 150s limit


def _is_valid(circles, tol=1e-9):
    """Check if the configuration is valid."""
    if len(circles) == 0:
        return False
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    
    if np.any(r <= 0):
        return False
    
    x_min = np.min(x - r)
    x_max = np.max(x + r)
    y_min = np.min(y - r)
    y_max = np.max(y + r)
    
    W = x_max - x_min
    H = y_max - y_min
    
    if 2 * (W + H) > PERIMETER + tol:
        return False
    
    n = len(circles)
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx * dx + dy * dy
            sum_r = r[i] + r[j]
            if dist_sq < (sum_r - tol) ** 2:
                return False
    return True


def _compute_value(circles):
    """Compute the sum of radii for a valid configuration, else 0."""
    if _is_valid(circles):
        return np.sum(circles[:, 2])
    return 0.0


def _voronoi_inradii(points, W, H):
    """
    Compute the inradius of each point's Voronoi cell within [0,W] x [0,H].
    Returns array of inradii.
    """
    n = len(points)
    if n == 0:
        return np.array([])
    
    # Add boundary points to ensure Voronoi diagram is bounded
    # We'll use a simple approach: for each point, find max distance to any
    # other point or boundary such that the circle stays within bounds
    
    inradii = np.zeros(n)
    
    for i in range(n):
        xi, yi = points[i]
        
        # Distance to boundaries
        d_bound = min(xi, W - xi, yi, H - yi)
        
        # Distance to other points
        min_dist = d_bound
        for j in range(n):
            if j == i:
                continue
            dx = xi - points[j, 0]
            dy = yi - points[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = min(min_dist, dist / 2.0)
        
        inradii[i] = min_dist
    
    return inradii


def _voronoi_inradii_fast(points, W, H):
    """
    Fast computation of inradii using vectorized distance computations.
    For each point i, inradius = min(min distance to boundaries, 
                                      0.5 * min distance to other points)
    """
    n = len(points)
    if n == 0:
        return np.array([])
    
    # Distance to boundaries for each point
    d_bound = np.minimum(np.minimum(points[:, 0], W - points[:, 0]),
                         np.minimum(points[:, 1], H - points[:, 1]))
    
    # Pairwise distances
    # dist[i,j] = distance between point i and point j
    dx = points[:, 0][:, np.newaxis] - points[:, 0][np.newaxis, :]
    dy = points[:, 1][:, np.newaxis] - points[:, 1][np.newaxis, :]
    dist = np.sqrt(dx * dx + dy * dy)
    
    # For each point i, the minimum distance to any other point is min(dist[i,:])
    # But we need to exclude self (dist[i,i] = 0)
    # Set diagonal to inf
    np.fill_diagonal(dist, np.inf)
    
    min_pair_dist = np.min(dist, axis=1)
    
    # Inradius is min of boundary distance and half the min pair distance
    inradii = np.minimum(d_bound, 0.5 * min_pair_dist)
    
    return inradii


def _seed_grid(W, H):
    """Create a grid-based seed configuration within a box of size W x H."""
    best_val = -1
    best_circles = None
    
    for cols in range(1, N + 1):
        rows = (N + cols - 1) // cols
        if rows * cols < N:
            continue
            
        s_x = W / cols
        s_y = H / rows
        
        circles = []
        for i in range(cols):
            for j in range(rows):
                if len(circles) >= N:
                    break
                cx = (i + 0.5) * s_x
                cy = (j + 0.5) * s_y
                circles.append([cx, cy, 0.0])  # radius will be derived
        
        circles = np.array(circles[:N])
        
        # Compute inradii for this grid
        inradii = _voronoi_inradii_fast(circles, W, H)
        circles[:, 2] = inradii
        
        if _is_valid(circles):
            val = np.sum(circles[:, 2])
            if val > best_val:
                best_val = val
                best_circles = circles.copy()
    
    return best_circles


def _optimize_voronoi(W, H, initial_points, time_budget):
    """
    Optimize center positions only, with radii derived from Voronoi inradii.
    Objective: maximize sum of inradii.
    """
    t_start = time.time()
    best_points = initial_points.copy()
    best_circles = np.column_stack([best_points, _voronoi_inradii_fast(best_points, W, H)])
    best_val = _compute_value(best_circles)
    
    def objective_and_grad(params):
        if time.time() - t_start > time_budget:
            return 1e10, np.zeros_like(params)
        
        x = params[0::2]
        y = params[1::2]
        points = np.column_stack([x, y])
        
        # Compute inradii
        n = len(points)
        
        # Distance to boundaries
        d_bound = np.minimum(np.minimum(x, W - x), np.minimum(y, H - y))
        
        # Pairwise distances
        dx = x[:, np.newaxis] - x[np.newaxis, :]
        dy = y[:, np.newaxis] - y[np.newaxis, :]
        dist = np.sqrt(dx * dx + dy * dy)
        np.fill_diagonal(dist, np.inf)
        
        min_pair_dist = np.min(dist, axis=1)
        
        # Inradii
        inradii = np.minimum(d_bound, 0.5 * min_pair_dist)
        
        obj = -np.sum(inradii)
        
        # Gradient
        grad = np.zeros(2 * n)
        
        for i in range(n):
            # Gradient of d_bound[i]
            g_x_bound = 0.0
            g_y_bound = 0.0
            if x[i] < W - x[i] and x[i] < y[i] and x[i] < H - y[i]:
                g_x_bound = 1.0  # d(xi)/dx_i = 1
            elif W - x[i] < x[i] and W - x[i] < y[i] and W - x[i] < H - y[i]:
                g_x_bound = -1.0  # d(W-xi)/dx_i = -1
            elif y[i] < x[i] and y[i] < W - x[i] and y[i] < H - y[i]:
                g_y_bound = 1.0
            elif H - y[i] < x[i] and H - y[i] < W - x[i] and H - y[i] < y[i]:
                g_y_bound = -1.0
            
            # Gradient of 0.5 * min_pair_dist[i]
            # Find which j gives the min
            min_j = np.argmin(dist[i, :])
            
            if dist[i, min_j] < d_bound[i] + 1e-12:
                # The pair distance is the binding constraint
                dx_ij = x[i] - x[min_j]
                dy_ij = y[i] - y[min_j]
                d_ij = dist[i, min_j]
                
                if d_ij > 1e-9:
                    # d(0.5 * d_ij)/dx_i = 0.5 * (x_i - x_j) / d_ij
                    g_x_pair = 0.5 * dx_ij / d_ij
                    g_y_pair = 0.5 * dy_ij / d_ij
                else:
                    g_x_pair = 0.5
                    g_y_pair = 0.0
            else:
                g_x_pair = 0.0
                g_y_pair = 0.0
            
            # inradius[i] = min(d_bound[i], 0.5 * min_pair_dist[i])
            # Gradient depends on which is smaller
            if d_bound[i] < 0.5 * min_pair_dist[i] + 1e-12:
                # Boundary is binding
                grad[2*i] += -g_x_bound  # d(-sum inradii)/dx_i
                grad[2*i+1] += -g_y_bound
            else:
                # Pair distance is binding
                grad[2*i] += -g_x_pair
                grad[2*i+1] += -g_y_pair
                
                # Also contribute to the other point's gradient
                if d_ij > 1e-9:
                    grad[2*min_j] += -g_x_pair  # d(0.5*d_ij)/dx_j = -0.5*(x_i-x_j)/d_ij = -g_x_pair
                    grad[2*min_j+1] += -g_y_pair
        
        return obj, grad

    def callback(xk):
        nonlocal best_points, best_circles, best_val
        current_points = xk.reshape(-1, 2)
        current_inradii = _voronoi_inradii_fast(current_points, W, H)
        current_circles = np.column_stack([current_points, current_inradii])
        val = _compute_value(current_circles)
        if val > best_val:
            best_val = val
            best_points = current_points.copy()
            best_circles = current_circles.copy()
        return False  # Continue

    try:
        bounds = [(0, W)] * N + [(0, H)] * N
        res = minimize(objective_and_grad, initial_points.flatten(), method='L-BFGS-B',
                       jac=True, bounds=bounds, options={'maxiter': 300, 'ftol': 1e-8},
                       callback=callback)
    except Exception:
        pass
    
    return best_circles


def _optimize(W, H, initial_circles, time_budget):
    """Optimize circle positions and radii in a fixed W x H box."""
    t_start = time.time()
    best_circles = initial_circles.copy()
    best_val = _compute_value(best_circles)
    
    def objective_and_grad(params):
        if time.time() - t_start > time_budget:
            return 1e10, np.zeros_like(params)
        
        n = len(params) // 3
        x = params[0::3]
        y = params[1::3]
        r = params[2::3]
        
        obj = -np.sum(r)
        grad = np.zeros_like(params)
        grad[2::3] = -1.0  # d(-sum r)/dr = -1
        
        # Boundary penalties
        lam_bound = 1000.0
        
        for i in range(n):
            if x[i] < r[i]:
                pen = (r[i] - x[i]) ** 2
                obj += lam_bound * pen
                grad[3*i] += lam_bound * 2 * (x[i] - r[i])
                grad[3*i+2] += lam_bound * 2 * (r[i] - x[i])
                
            if W - x[i] < r[i]:
                pen = (r[i] - (W - x[i])) ** 2
                obj += lam_bound * pen
                grad[3*i] += lam_bound * 2 * (W - x[i] - r[i])
                grad[3*i+2] += lam_bound * 2 * (r[i] - (W - x[i]))
                
            if y[i] < r[i]:
                pen = (r[i] - y[i]) ** 2
                obj += lam_bound * pen
                grad[3*i+1] += lam_bound * 2 * (y[i] - r[i])
                grad[3*i+2] += lam_bound * 2 * (r[i] - y[i])
                
            if H - y[i] < r[i]:
                pen = (r[i] - (H - y[i])) ** 2
                obj += lam_bound * pen
                grad[3*i+1] += lam_bound * 2 * (H - y[i] - r[i])
                grad[3*i+2] += lam_bound * 2 * (r[i] - (H - y[i]))
                
        # Overlap penalties
        lam_ov = 1000.0
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                sum_r = r[i] + r[j]
                
                if dist < sum_r:
                    overlap = sum_r - dist
                    pen = overlap ** 2
                    obj += lam_ov * pen
                    
                    if dist > 1e-6:
                        g_x = 2 * overlap * (-(x[i] - x[j]) / dist)
                        g_y = 2 * overlap * (-(y[i] - y[j]) / dist)
                        g_r = 2 * overlap * 1
                        
                        grad[3*i] += lam_ov * g_x
                        grad[3*i+1] += lam_ov * g_y
                        grad[3*i+2] += lam_ov * g_r
                        
                        grad[3*j] += lam_ov * -g_x
                        grad[3*j+1] += lam_ov * -g_y
                        grad[3*j+2] += lam_ov * g_r
                    else:
                        grad[3*i] += lam_ov
                        grad[3*j] -= lam_ov
        
        return obj, grad

    def callback(xk):
        nonlocal best_circles, best_val
        current_circles = xk.reshape(-1, 3)
        current_circles[:, 2] = np.maximum(current_circles[:, 2], 1e-6)
        val = _compute_value(current_circles)
        if val > best_val:
            best_val = val
            best_circles = current_circles.copy()
        return False

    try:
        bounds = [
            (0, W) for _ in range(N)
        ] + [
            (0, H) for _ in range(N)
        ] + [
            (1e-6, min(W, H)/2) for _ in range(N)
        ]
        
        res = minimize(objective_and_grad, initial_circles.flatten(), method='L-BFGS-B', 
                       jac=True, bounds=bounds, options={'maxiter': 500, 'ftol': 1e-8}, callback=callback)
    except Exception:
        pass
    
    return best_circles


def construct():
    """Construct 21 disjoint circles in a perimeter-4 rectangle maximizing sum of radii."""
    t_start = time.time()
    best_overall = None
    best_overall_val = 0.0
    
    # Try different aspect ratios
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.6]
    
    for ar in aspect_ratios:
        if time.time() - t_start > TIME_LIMIT - 10.0:
            break
            
        W = 2.0 * ar / (1.0 + ar)
        H = 2.0 - W
        
        # Seed
        circles = _seed_grid(W, H)
        if circles is None:
            cols = 5
            rows = 5
            s = min(W/cols, H/rows)
            circles = []
            for i in range(cols):
                for j in range(rows):
                    if len(circles) >= N: break
                    circles.append([(i+0.5)*s, (j+0.5)*s, 0.4*s])
            circles = np.array(circles[:N])
        
        # Try Voronoi-based optimization
        time_remaining = TIME_LIMIT - (time.time() - t_start) - 5.0
        if time_remaining > 2.0:
            initial_points = circles[:, :2].copy()
            optimized_v = _optimize_voronoi(W, H, initial_points, time_remaining)
            val_v = _compute_value(optimized_v)
            if val_v > best_overall_val:
                best_overall_val = val_v
                best_overall = optimized_v
        
        # Also try the original free-radius optimization
        if time.time() - t_start > TIME_LIMIT - 20.0:
            break
            
        time_remaining = TIME_LIMIT - (time.time() - t_start) - 5.0
        if time_remaining > 2.0:
            optimized = _optimize(W, H, circles, time_remaining)
            val = _compute_value(optimized)
            if val > best_overall_val:
                best_overall_val = val
                best_overall = optimized

        # Random perturbation for diversity
        if time.time() - t_start > TIME_LIMIT - 15.0:
            break
            
        circles_pert = circles.copy()
        circles_pert[:, :2] += np.random.normal(0, 0.05, circles_pert[:, :2].shape)
        circles_pert[:, :2] = np.clip(circles_pert[:, :2], 0, np.array([W, H]))
        
        time_remaining = TIME_LIMIT - (time.time() - t_start) - 5.0
        if time_remaining > 2.0:
            initial_points_pert = circles_pert[:, :2].copy()
            optimized_pert = _optimize_voronoi(W, H, initial_points_pert, time_remaining)
            val_pert = _compute_value(optimized_pert)
            if val_pert > best_overall_val:
                best_overall_val = val_pert
                best_overall = optimized_pert

    if best_overall is None:
        W, H = 1.0, 1.0
        circles = _seed_grid(W, H)
        if circles is None or not _is_valid(circles):
             s = 2.0 / (7 + 3)
             circles = []
             for i in range(7):
                 for j in range(3):
                     if len(circles) >= N: break
                     circles.append([(i+0.5)*s, (j+0.5)*s, 0.4*s])
             circles = np.array(circles[:N])
        best_overall = circles

    return best_overall
