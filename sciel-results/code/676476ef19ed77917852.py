"""
Circle packing in perimeter-4 rectangle.
Strategy: Voronoi-based objective with soft-min surrogate and hybrid optimization.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Voronoi
import time
import sys

N = 21
PERIMETER = 4.0
TIME_LIMIT = 145.0 # Safety margin before 150s limit

def _soft_min(x, t=10.0):
    """
    Approximate min(x) using log-sum-exp.
    min(x) ~ -log-sum-exp(-x/t) * t
    """
    # Shift for numerical stability
    x_min = np.min(x)
    if x_min < -100: # Avoid overflow in exp
        return x_min
    x_shifted = x - x_min
    # log-sum-exp(-x_shifted/t)
    # Note: we want min(x), so we use -x in the exp
    # min(x) = - ( -min(-x) ) = - ( - logsumexp(-x/t) * t ) ? 
    # Actually: min(x) = - logsumexp(-x/t) * t is NOT quite right for arbitrary t.
    # Standard soft min: m_t(x) = -t * log( sum( exp(-x_i/t) ) )
    # As t -> 0, m_t(x) -> min(x_i)
    
    # To avoid overflow, we factor out exp(-x_min/t)
    # sum exp(-x_i/t) = exp(-x_min/t) * sum exp(-(x_i-x_min)/t)
    # log(sum) = -x_min/t + log(sum exp(-(x_i-x_min)/t))
    # m_t = -t * ( -x_min/t + log(sum exp(-(x_i-x_min)/t)) )
    # m_t = x_min - t * log(sum exp(-(x_i-x_min)/t))
    
    # Wait, if t is small, exp(-(x_i-x_min)/t) for x_i > x_min is ~ 0.
    # So sum ~ 1. log(1) = 0. m_t ~ x_min. Correct.
    
    # However, standard log-sum-exp is usually for max.
    # Let's use the standard stable log-sum-exp formula for sum(exp(a_i))
    # Let a_i = -x_i / t
    # We want -t * log( sum(exp(a_i)) )
    
    a = -x / t
    a_max = np.max(a)
    if a_max > 500: # overflow risk in exp(a - a_max) is low if we subtract max
        # If a_max is huge, x_min is very negative. 
        # Just return min(x) directly if t is effectively 0 or values are extreme
        pass
    
    # Stable log-sum-exp
    # log(sum(exp(a))) = a_max + log(sum(exp(a - a_max)))
    sum_exp = np.sum(np.exp(a - a_max))
    if sum_exp == 0:
        return np.min(x)
    log_sum = a_max + np.log(sum_exp)
    
    return -t * log_sum

def _voronoi_inradii(x, y, W, H, t=1.0):
    """
    Compute the inradius of each Voronoi cell clipped by the rectangle [0, W]x[0, H].
    Returns an array of radii.
    """
    points = np.column_stack((x, y))
    n = len(points)
    radii = np.zeros(n)
    
    try:
        vor = Voronoi(points)
    except Exception:
        # Fallback if Voronoi fails (e.g. all points identical)
        # Estimate distance to boundary
        for i in range(n):
            radii[i] = min(x[i], W-x[i], y[i], H-y[i])
        return radii
    
    for i in range(n):
        region_idx = vor.point_region[i]
        region = vor.regions[region_idx]
        
        if -1 in region:
            # Unbounded region, but we are clipping by box, so it's bounded by box
            # The inradius is limited by the box boundaries
            # We need to find the max distance to the boundary of the intersection of the Voronoi cell and the box.
            # This is effectively the distance to the closest point on the boundary of the box or the Voronoi edges.
            
            # Simple approximation: distance to box boundary
            dist_box = min(x[i], W-x[i], y[i], H-y[i])
            
            # If the Voronoi cell is unbounded, it extends to infinity, but the box cuts it off.
            # The "inradius" in the context of packing in a box is the distance to the closest constraint.
            # Constraints are:
            # 1. Box boundaries: x=0, x=W, y=0, y=H
            # 2. Voronoi edges (lines equidistant to i and j)
            
            # For a robust estimate, we can use the distance to the box boundary if the cell is unbounded,
            # but we must also check if a neighbor is closer than the box boundary.
            # Actually, the Voronoi diagram already encodes the neighbors.
            # If the region is unbounded, it means there are no Voronoi edges in some direction.
            # But the box boundary acts as a "wall".
            
            # A safer approach for unbounded regions in a box:
            # The inradius is min( dist_to_box_boundary, dist_to_nearest_neighbor )
            # Wait, dist_to_nearest_neighbor is NOT the inradius if the neighbor is far.
            # The inradius is the radius of the largest circle centered at i that fits inside the cell.
            # If the cell is unbounded, it doesn't fit in a box unless we consider the box boundary.
            
            # Let's calculate the distance to the box boundary.
            # And distance to the closest other point? No, that's not the inradius.
            # The inradius of a Voronoi cell (clipped by box) is the max r such that circle(i, r) is inside Cell_i ∩ Box.
            
            # For simplicity and speed in the inner loop, we approximate:
            # r_i = min( dist_to_box_boundary, 0.5 * min_dist_to_neighbor )
            # This is a lower bound. It might be conservative.
            
            min_dist_neigh = np.inf
            for j in range(n):
                if i == j: continue
                d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                if d < min_dist_neigh:
                    min_dist_neigh = d
            
            r = min(dist_box, 0.5 * min_dist_neigh)
            radii[i] = r
        else:
            # Bounded region
            # The inradius is the max distance from center to the edges of the polygon defined by the vertices in `region`.
            # Vertices are vor.vertices[region]
            verts = vor.vertices[region]
            if len(verts) == 0:
                radii[i] = 0
                continue
            
            # The inradius is the minimum distance from the center to the edges of the Voronoi cell.
            # The edges are segments between consecutive vertices in the region.
            # However, the Voronoi vertices are the circumcenters of the triangles.
            # The distance from the center to the edge (which is the perpendicular bisector of the segment to the neighbor)
            # is simply 0.5 * distance to the neighbor.
            
            # So for a bounded cell, the inradius is min( 0.5 * dist(i, j) for all neighbors j )
            # BUT, we must also clip by the box.
            # If the Voronoi cell extends outside the box, the box boundary is a constraint.
            # The Voronoi diagram computed on points only doesn't know about the box.
            # So we must intersect the Voronoi cell with the box.
            
            # Approximation:
            # r_i = min( dist_to_box_boundary, 0.5 * min_dist_to_neighbor )
            
            min_dist_neigh = np.inf
            for j in range(n):
                if i == j: continue
                d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                if d < min_dist_neigh:
                    min_dist_neigh = d
            
            dist_box = min(x[i], W-x[i], y[i], H-y[i])
            
            radii[i] = min(dist_box, 0.5 * min_dist_neigh)
            
    return radii

def _objective_and_grad(params, W, H, t_soft):
    """
    Objective: -sum(soft_min(radii))
    We want to maximize sum(radii), so we minimize -sum(radii).
    Since radii are derived from positions, we use the soft-min of the inradii to make it smooth.
    """
    n = len(params) // 2
    x = params[0::2]
    y = params[1::2]
    
    # Compute inradii
    # Note: _voronoi_inradii is not differentiable.
    # For the gradient, we use finite differences or a simplified analytical gradient.
    # Given the complexity of Voronoi gradients, we will use L-BFGS-B with finite differences
    # for the objective, but provide a gradient if we can approximate it.
    # Actually, L-BFGS-B with `jac=None` uses finite differences which is slow but robust.
    # To speed up, we can use a "soft" repulsion force instead of exact Voronoi for the gradient?
    # No, let's stick to the objective value being exact (or soft-min of exact inradii)
    # and let the optimizer estimate the gradient.
    
    radii = _voronoi_inradii(x, y, W, H, t_soft)
    
    # Apply soft-min to the radii to smooth out the "min" operation inherent in the inradius calculation?
    # The inradius calculation already takes a min. 
    # The non-smoothness comes from the Voronoi topology changes.
    # Soft-min on the radii doesn't help with topology changes.
    # However, if we use a "soft" repulsion potential for the objective, it is smooth.
    
    # Alternative Objective:
    # Minimize -sum(r_i) where r_i is the inradius.
    # This is non-smooth.
    
    # Let's try a different objective:
    # Minimize sum( max(0, (r_i + r_j - dist_ij)^2) ) + sum( max(0, (r_i - dist_to_boundary)^2) )
    # This is the penalty method from before, but with dynamic radii?
    # No, the task is to maximize sum of radii.
    
    # Let's use the "Voronoi Inradius" value directly.
    # Value = sum(radii)
    # Objective = -Value
    
    # To help the optimizer, we can add a small penalty for overlapping if the Voronoi logic is flawed?
    # The Voronoi inradius logic `min(dist_box, 0.5*dist_neigh)` ensures no overlap and no boundary violation.
    # So the configuration is ALWAYS valid.
    # We just need to maximize the sum.
    
    val = np.sum(radii)
    
    # Gradient estimation is handled by the optimizer (finite diff) or we can provide a dummy gradient?
    # Providing a dummy gradient is bad.
    # Let's use `method='L-BFGS-B'` which can handle finite differences.
    # But finite differences on 42 variables (21*2) with a Voronoi calculation (O(N^2) or O(N log N)) per call is slow.
    # 42 * 2 (forward diff) = 84 evaluations per step.
    # If each eval is 1ms, 84ms/step. 100 steps = 8.4s. Feasible.
    
    return -val

def _optimize(W, H, initial_pos, time_budget):
    """
    Optimize positions to maximize sum of Voronoi inradii.
    """
    t_start = time.time()
    best_pos = initial_pos.copy()
    best_val = -_objective_and_grad(best_pos, W, H, 1.0) # -(-sum) = sum
    
    def objective(params):
        return _objective_and_grad(params, W, H, 1.0)
        
    def callback(xk):
        nonlocal best_pos, best_val
        current_val = -objective(xk)
        if current_val > best_val:
            best_val = current_val
            best_pos = xk.copy()
        if time.time() - t_start > time_budget:
            return True # Stop
        return False

    try:
        # Bounds: positions must be within the box
        bounds = [(0, W)] * N + [(0, H)] * N
        
        # Use L-BFGS-B with finite differences
        # maxiter limited by time
        res = minimize(objective, initial_pos, method='L-BFGS-B', 
                       bounds=bounds, 
                       options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6},
                       callback=callback)
        
        # Update best if the final result is better
        final_val = -objective(res.x)
        if final_val > best_val:
            best_val = final_val
            best_pos = res.x.copy()
            
    except Exception as e:
        pass
        
    return best_pos

def _construct_circles(pos, W, H):
    """
    Construct the final circle array from optimized positions.
    """
    x = pos[0::2]
    y = pos[1::2]
    radii = _voronoi_inradii(x, y, W, H, 1.0)
    
    # Ensure validity
    # The inradius logic guarantees validity, but let's double check for numerical errors
    circles = np.zeros((N, 3))
    circles[:, 0] = x
    circles[:, 1] = y
    circles[:, 2] = radii
    
    # Clip radii to be positive
    circles[:, 2] = np.maximum(circles[:, 2], 1e-9)
    
    # Final validity check and repair
    # If any circle violates, reduce its radius
    for i in range(N):
        for j in range(N):
            if i == j: continue
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx*dx + dy*dy)
            req_r = (dist - 1e-9) / 2.0
            if circles[i, 2] > req_r:
                circles[i, 2] = req_r
            if circles[j, 2] > req_r:
                circles[j, 2] = req_r
        
        # Boundary check
        max_r = min(x[i], W-x[i], y[i], H-y[i])
        if circles[i, 2] > max_r:
            circles[i, 2] = max_r
            
    return circles

def _seed_grid(W, H):
    """
    Generate a grid seed for N circles in W x H.
    """
    # Estimate grid size
    # Try to fit N circles in a grid
    # 21 is close to 5x4=20 or 6x4=24 or 7x3=21
    best_val = -1
    best_pos = None
    
    for cols in range(1, N+1):
        rows = (N + cols - 1) // cols
        if rows * cols < N:
            continue
            
        # Check aspect ratio compatibility
        # Spacing
        sx = W / cols
        sy = H / rows
        
        pos = []
        for i in range(cols):
            for j in range(rows):
                if len(pos) >= N:
                    break
                cx = (i + 0.5) * sx
                cy = (j + 0.5) * sy
                pos.append([cx, cy])
        
        pos = np.array(pos[:N]).flatten()
        
        # Evaluate
        val = -_objective_and_grad(pos, W, H, 1.0)
        if val > best_val:
            best_val = val
            best_pos = pos.copy()
            
    if best_pos is None:
        # Fallback: random
        best_pos = np.random.rand(N*2) * np.array([W, H] * N)
        
    return best_pos

def construct():
    t_start = time.time()
    best_circles = None
    best_value = 0.0
    
    # Aspect ratios to try
    # W + H = 2
    # Square: 1x1
    # Rect: 1.5x0.5, 1.2x0.8, etc.
    aspect_ratios = [1.0, 1.2, 0.8, 1.5, 0.6, 1.8, 0.5]
    
    # Shuffle to ensure diversity in case time runs out
    np.random.shuffle(aspect_ratios)
    
    for ar in aspect_ratios:
        elapsed = time.time() - t_start
        if elapsed > TIME_LIMIT - 10.0:
            break
            
        W = 2.0 * ar / (1.0 + ar)
        H = 2.0 - W
        
        # Seed 1: Grid
        pos1 = _seed_grid(W, H)
        
        # Optimize
        time_remaining = TIME_LIMIT - (time.time() - t_start) - 5.0
        if time_remaining > 5.0:
            opt_pos = _optimize(W, H, pos1, time_remaining)
            circles = _construct_circles(opt_pos, W, H)
            
            # Validate and compute value
            if _is_valid(circles):
                val = np.sum(circles[:, 2])
                if val > best_value:
                    best_value = val
                    best_circles = circles.copy()
                    
        # Seed 2: Random perturbation of grid
        if time.time() - t_start > TIME_LIMIT - 15.0:
            break
            
        pos2 = pos1.copy()
        pos2 += np.random.normal(0, 0.1, pos2.shape)
        pos2 = np.clip(pos2, 0, np.array([W, H] * N))
        
        time_remaining = TIME_LIMIT - (time.time() - t_start) - 5.0
        if time_remaining > 5.0:
            opt_pos = _optimize(W, H, pos2, time_remaining)
            circles = _construct_circles(opt_pos, W, H)
            
            if _is_valid(circles):
                val = np.sum(circles[:, 2])
                if val > best_value:
                    best_value = val
                    best_circles = circles.copy()
                    
        # Seed 3: Fully random
        if time.time() - t_start > TIME_LIMIT - 20.0:
            break
            
        pos3 = np.random.rand(N*2) * np.array([W, H] * N)
        
        time_remaining = TIME_LIMIT - (time.time() - t_start) - 5.0
        if time_remaining > 5.0:
            opt_pos = _optimize(W, H, pos3, time_remaining)
            circles = _construct_circles(opt_pos, W, H)
            
            if _is_valid(circles):
                val = np.sum(circles[:, 2])
                if val > best_value:
                    best_value = val
                    best_circles = circles.copy()

    if best_circles is None:
        # Fallback
        W, H = 1.0, 1.0
        pos = _seed_grid(W, H)
        best_circles = _construct_circles(pos, W, H)
        if not _is_valid(best_circles):
            # Last resort: tiny circles
            best_circles = np.zeros((N, 3))
            for i in range(N):
                best_circles[i, 0] = 0.1 + (i % 5) * 0.2
                best_circles[i, 1] = 0.1 + (i // 5) * 0.2
                best_circles[i, 2] = 0.05

    return best_circles

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

# Expose the function as required
# The prompt asks to keep function names. 
# The main entry point is construct().
