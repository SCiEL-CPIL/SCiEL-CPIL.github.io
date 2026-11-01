"""Circle packing in perimeter-4 rectangle: Voronoi-based soft-body simulation."""
import numpy as np
import time
import random
from scipy.spatial import Voronoi
from scipy.optimize import minimize

N = 21


def _compute_inradii(points, W, H):
    """
    Compute the inradius of the Voronoi cell for each point, clipped by the rectangle boundary.
    Returns an array of radii.
    """
    n = len(points)
    radii = np.zeros(n)
    
    # Pad points with boundary points to handle open cells
    # The rectangle corners and midpoints of sides help close the Voronoi diagram
    # We add points slightly outside the rectangle to ensure all cells are bounded
    # Or simpler: add the 4 corners and 4 midpoints of the rectangle to the Voronoi set,
    # but we only care about the inradii of the original N points.
    # Actually, a robust way is to add points at infinity (far away) or just use the
    # rectangle boundaries as constraints.
    
    # A common trick: Add the 4 corners of the rectangle to the point set for Voronoi,
    # then the cells of the original points will be clipped by the rectangle.
    # However, Voronoi of points inside a box is equivalent to Voronoi of points + box boundaries.
    # We can simulate this by adding points far outside in 4 directions, or just adding
    # the corners. Adding corners is not sufficient for mid-edge cells.
    
    # Better approach: Add points at (0,0), (W,0), (0,H), (W,H) and also points along the edges?
    # Actually, the inradius of a cell in a bounded domain is the distance to the nearest
    # boundary (either another point or the wall).
    # So we can compute the Voronoi diagram of the N points, and then for each cell,
    # find the minimum distance from the site to the cell edges, AND to the rectangle walls.
    
    try:
        vor = Voronoi(points)
    except Exception:
        # Degenerate case, return small radii
        return np.full(n, 0.01)
    
    for i in range(n):
        r = np.inf
        # Distance to walls
        x, y = points[i]
        r = min(r, x, W - x, y, H - y)
        
        # Distance to Voronoi edges
        region_idx = vor.point_region[i]
        region = vor.regions[region_idx]
        
        if -1 not in region and len(region) > 0:
            for edge_idx in vor.ridge_points[i]:
                # Get the vertices of the ridge
                v1, v2 = vor.ridge_points[i]
                # The ridge is the line segment between vor.vertices[v1] and vor.vertices[v2]
                # Wait, ridge_points gives indices into vertices array.
                # But a ridge might be open (one vertex is -1).
                pass
                
        # Simpler: For each edge in the region, compute distance from point to segment.
        # But we need to know which edges belong to the region.
        # Voronoi regions are defined by vertex indices.
        # The edges are the segments connecting these vertices.
        # However, if the region is unbounded (contains -1), the edges go to infinity.
        # In a bounded rectangle, we can just clamp the Voronoi vertices to the rectangle?
        # No, that's not correct.
        
        # Alternative robust method:
        # The inradius is the minimum distance to any other point's Voronoi boundary or the wall.
        # Since we already have the Voronoi diagram, we can iterate over the ridges incident to point i.
        for ridge_idx in range(len(vor.ridge_points)):
            if i in vor.ridge_points[ridge_idx]:
                v1, v2 = vor.ridge_points[ridge_idx]
                if v1 == -1 or v2 == -1:
                    # Unbounded ridge, distance is determined by walls (already handled)
                    # But we need to ensure the wall distance is the correct one for this direction.
                    # Since we took min with all walls, it's safe.
                    continue
                p1 = vor.vertices[v1]
                p2 = vor.vertices[v2]
                
                # Distance from points[i] to segment p1-p2
                px, py = points[i]
                x1, y1 = p1
                x2, y2 = p2
                
                dx = x2 - x1
                dy = y2 - y1
                length_sq = dx*dx + dy*dy
                if length_sq < 1e-12:
                    dist = np.sqrt((px-x1)**2 + (py-y1)**2)
                else:
                    t = ((px-x1)*dx + (py-y1)*dy) / length_sq
                    t = np.clip(t, 0, 1)
                    proj_x = x1 + t*dx
                    proj_y = y1 + t*dy
                    dist = np.sqrt((px-proj_x)**2 + (py-proj_y)**2)
                
                r = min(r, dist)
        
        # Ensure radius is positive and small
        if r < 1e-6:
            r = 1e-6
        radii[i] = r
        
    return radii


def _objective(z, W, H):
    """
    Objective function: minimize negative sum of inradii.
    z contains x[0:N], y[0:N].
    """
    n = N
    x = z[0:n]
    y = z[n:2*n]
    
    # Ensure points are inside rectangle with small margin
    eps = 1e-5
    x = np.clip(x, eps, W - eps)
    y = np.clip(y, eps, H - eps)
    
    points = np.column_stack((x, y))
    radii = _compute_inradii(points, W, H)
    
    # Penalize if any radius is extremely small (degenerate)
    # But the inradius computation handles this.
    
    # We want to maximize sum(radii), so minimize -sum(radii)
    return -np.sum(radii)


def _slsqp_voronoi(x_init, y_init, W, H, time_budget=30.0):
    """
    Optimize positions using SLSQP with Voronoi inradii objective.
    """
    n = N
    z0 = np.concatenate([x_init, y_init])
    
    bounds = [(0, W)] * n + [(0, H)] * n
    
    def obj(zv):
        return _objective(zv, W, H)
    
    # We don't have explicit constraints in the objective, but we can add them
    # to keep points well inside.
    # Actually, clipping in the objective is non-smooth.
    # Let's use bounds to keep them inside.
    
    maxiter = int(min(200, max(50, time_budget * 20)))
    
    try:
        res = minimize(obj, z0, method='SLSQP', bounds=bounds,
                       options={'maxiter': maxiter, 'ftol': 1e-10, 'maxls': 50})
        
        z_opt = res.x
        x_opt = z_opt[0:n]
        y_opt = z_opt[n:2*n]
        
        # Clip to be safe
        eps = 1e-5
        x_opt = np.clip(x_opt, eps, W - eps)
        y_opt = np.clip(y_opt, eps, H - eps)
        
        points = np.column_stack((x_opt, y_opt))
        radii = _compute_inradii(points, W, H)
        
        return x_opt, y_opt, radii, W, float(np.sum(radii))
    except Exception:
        return None


def _pack_voronoi(W, H, seed, time_budget):
    """
    Full pipeline for Voronoi-based packing.
    """
    rng = np.random.default_rng(seed)
    
    # Initial placement: random or grid
    if seed % 2 == 0:
        # Grid
        cols = 5
        rows = (N + cols - 1) // cols
        x = np.zeros(N)
        y = np.zeros(N)
        for i in range(N):
            row, col = divmod(i, cols)
            x[i] = (col + 0.5) / cols * W
            y[i] = (row + 0.5) / rows * H
    else:
        # Random
        x = rng.uniform(0.1*W, 0.9*W, N)
        y = rng.uniform(0.1*H, 0.9*H, N)
    
    # Add small perturbation
    x += rng.normal(0, 0.01, N)
    y += rng.normal(0, 0.01, N)
    x = np.clip(x, 0.05, W-0.05)
    y = np.clip(y, 0.05, H-0.05)
    
    result = _slsqp_voronoi(x, y, W, H, time_budget)
    if result is not None:
        xo, yo, ro, wo, val = result
        return xo, yo, ro, wo, val
    
    # Fallback: return initial with computed radii
    points = np.column_stack((x, y))
    radii = _compute_inradii(points, W, H)
    return x, y, radii, W, float(np.sum(radii))


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_val = 0.0
    best_x = None
    best_y = None
    best_r = None
    best_W = None
    
    # Aspect ratios
    aspect_configs = [
        (1.0, 1.0),
        (0.8, 1.2),
        (1.2, 0.8),
        (0.7, 1.3),
        (1.3, 0.7),
        (0.9, 1.1),
        (1.1, 0.9),
    ]
    
    all_configs = []
    for w, h in aspect_configs:
        for s in range(4):
            all_configs.append((w, h, s))
            
    random.seed(42)
    random.shuffle(all_configs)
    
    trial = 0
    for W, H, seed in all_configs:
        if time.time() > deadline - 15:
            break
            
        trial_budget = min(15.0, deadline - 10 - (time.time() - start_time))
        if trial_budget < 5:
            break
            
        try:
            result = _pack_voronoi(W, H, seed + trial * 11, trial_budget)
            if result is not None:
                xo, yo, ro, wo, val = result
                if val > best_val:
                    best_val = val
                    best_x = xo.copy()
                    best_y = yo.copy()
                    best_r = ro.copy()
                    best_W = wo
        except Exception:
            pass
        trial += 1
        
    # Final polish
    if best_x is not None and time.time() < deadline - 10:
        # Try to optimize aspect ratio slightly? 
        # Or just re-run SLSQP on the best config with more time
        remaining = deadline - 5 - (time.time() - start_time)
        if remaining > 10:
            result = _slsqp_voronoi(best_x, best_y, best_W, 2.0 - best_W, remaining)
            if result is not None:
                xo, yo, ro, wo, val = result
                if val > best_val:
                    best_x, best_y, best_r, best_W = xo, yo, ro, wo
                    best_val = val
                    
    # Fallback
    if best_x is None:
        W, H = 1.0, 1.0
        side = 5
        best_x = np.array([(c + 0.5) / side * W for c in range(5) for _ in range(5)])[:N]
        best_y = np.array([(r_ + 0.5) / side * H for r_ in range(5) for c in range(5)])[:N]
        points = np.column_stack((best_x, best_y))
        best_r = _compute_inradii(points, W, H)
        best_W = W
        best_val = float(np.sum(best_r))
        
    # Ensure validity: radii are by definition inradii of Voronoi cells, so they are disjoint.
    # But we must ensure they are inside the rectangle.
    # The inradius computation clips by walls, so they are inside.
    # However, numerical errors might cause slight violations.
    # Shrink slightly to be safe.
    for _ in range(10):
        # Check if any circle violates boundary
        if np.any(best_x - best_r < -1e-6) or np.any(best_x + best_r > best_W + 1e-6) or \
           np.any(best_y - best_r < -1e-6) or np.any(best_y + best_r > (2.0 - best_W) + 1e-6):
            best_r *= 0.99
        else:
            break
            
    return np.column_stack((best_x, best_y, best_r))

