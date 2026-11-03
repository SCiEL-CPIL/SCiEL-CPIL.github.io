"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 21

# EVOLVE-BLOCK-START

def _build_hex_grid():
    """
    Build a hexagonal close-packed arrangement of 21 circles in a unit-square-ish region.
    We'll use a hex grid: rows with 5,4,5,4,3 circles (total 21) or similar.
    Actually, let's think: for N=21, a hex packing with 5 columns and 5 rows offset.
    Let's use rows of 5,4,5,4,3 = 21.
    """
    # Hex grid: each row offset by half spacing
    # Row 0: 5 circles
    # Row 1: 4 circles (offset)
    # Row 2: 5 circles
    # Row 3: 4 circles (offset)
    # Row 4: 3 circles
    # Total = 21
    
    # In hex packing, horizontal spacing = 2r, vertical spacing = sqrt(3)*r
    # Let's set r=1 for now, then scale
    r = 1.0
    dx = 2 * r
    dy = np.sqrt(3) * r
    
    points = []
    # Row 0: 5 circles at y=0
    for i in range(5):
        points.append((i * dx, 0))
    # Row 1: 4 circles at y=dy, offset by dx/2
    for i in range(4):
        points.append((i * dx + dx/2, dy))
    # Row 2: 5 circles at y=2*dy
    for i in range(5):
        points.append((i * dx, 2*dy))
    # Row 3: 4 circles at y=3*dy, offset by dx/2
    for i in range(4):
        points.append((i * dx + dx/2, 3*dy))
    # Row 4: 3 circles at y=4*dy
    for i in range(3):
        points.append((i * dx, 4*dy))
    
    pts = np.array(points)
    return pts


def _compute_bounding_box(pts):
    """Compute bounding box dimensions."""
    xmin, xmax = pts[:, 0].min(), pts[:, 0].max()
    ymin, ymax = pts[:, 1].min(), pts[:, 1].max()
    return xmax - xmin, ymax - ymin


def _scale_to_perimeter(pts, target_perim=4.0):
    """Scale points so that the bounding rectangle has the given perimeter."""
    w, h = _compute_bounding_box(pts)
    perim = 2 * (w + h)
    scale = target_perim / perim
    return pts * scale


def _evaluate_packing(pts):
    """
    Evaluate a packing: scale to perimeter 4, then compute sum of radii.
    Radii are half the minimum distance to any other circle or to the boundary.
    """
    w, h = _compute_bounding_box(pts)
    perim = 2 * (w + h)
    scale = 4.0 / perim
    scaled = pts * scale
    
    n = len(scaled)
    radii = np.zeros(n)
    
    for i in range(n):
        # Distance to boundary
        dist_boundary = min(scaled[i, 0], scaled[i, 1], 
                           (scaled[:, 0].max() - scaled[i, 0]),
                           (scaled[:, 1].max() - scaled[i, 1]))
        
        # Distance to other circles
        dists = np.sqrt(np.sum((scaled - scaled[i])**2, axis=1))
        dists[i] = np.inf
        min_dist = np.min(dists)
        
        radii[i] = min(dist_boundary, min_dist / 2)
    
    return np.sum(radii)


def _objective_and_grad(x):
    """
    Objective: maximize sum of radii. We minimize negative sum of radii.
    x is a flat array of 2*N coordinates (positions of circle centers).
    We work in a normalized coordinate system where the bounding box perimeter is fixed.
    Actually, let's parameterize differently: we fix the rectangle to have perimeter 4,
    and optimize the positions within it.
    
    Let's use: the rectangle has width w and height h with 2(w+h) = 4, so w + h = 2.
    We parameterize by aspect ratio or just fix w and h and optimize positions.
    
    Better approach: optimize positions in a unit square, then the actual rectangle
    dimensions are determined by the bounding box. But that's implicit.
    
    Alternative: parameterize the rectangle by one parameter (e.g., w, with h = 2 - w),
    and optimize circle positions within that rectangle.
    
    Let's do: x = [w, x_1, y_1, x_2, y_2, ..., x_N, y_N]
    where w is the width (h = 2 - w), and positions are in [0, w] x [0, h].
    """
    w = x[0]
    h = 2.0 - w
    pts = x[1:].reshape(N, 2)
    
    # Clamp w to reasonable range
    w = np.clip(w, 0.1, 1.9)
    h = 2.0 - w
    
    # Scale positions to be within [0, w] x [0, h]
    # Actually, let's just use the positions directly and compute radii
    # The bounding box of the points might not fill the whole rectangle.
    # We need to ensure the rectangle contains all circles.
    
    # Let's define: the rectangle is [0, w] x [0, h].
    # Circle i has center (pts[i, 0], pts[i, 1]) and radius r_i.
    # r_i = min(pts[i,0], pts[i,1], w - pts[i,0], h - pts[i,1], 
    #           min_{j != i} dist(i,j)/2)
    
    n = N
    radii = np.zeros(n)
    
    for i in range(n):
        dist_boundary = min(pts[i, 0], pts[i, 1], w - pts[i, 0], h - pts[i, 1])
        if i < n - 1:
            dists = np.sqrt(np.sum((pts[i+1:] - pts[i])**2, axis=1))
            if i > 0:
                dists_prev = np.sqrt(np.sum((pts[:i] - pts[i])**2, axis=1))
                min_dist = min(np.min(dists), np.min(dists_prev))
            else:
                min_dist = np.min(dists)
        else:
            dists_prev = np.sqrt(np.sum((pts[:i] - pts[i])**2, axis=1))
            min_dist = np.min(dists_prev)
        radii[i] = min(dist_boundary, min_dist / 2)
    
    # Ensure circles are within bounds
    # Soft penalty if any circle goes out of bounds
    penalty = 0.0
    for i in range(n):
        if pts[i, 0] < 0 or pts[i, 0] > w or pts[i, 1] < 0 or pts[i, 1] > h:
            penalty += 100.0
    
    return -np.sum(radii) + penalty


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s
    
    # Initial guess: hexagonal packing
    pts_init = _build_hex_grid()
    w0, h0 = _compute_bounding_box(pts_init)
    # Normalize to w + h = 2
    scale0 = 2.0 / (w0 + h0)
    pts_init = pts_init * scale0
    w_init = w0 * scale0
    h_init = h0 * scale0
    
    x0 = np.concatenate([[w_init], pts_init.flatten()])
    
    best_x = x0.copy()
    best_val = -_objective_and_grad(x0)
    
    # Try multiple restarts with different initial configurations
    n_restarts = 5
    
    for restart in range(n_restarts):
        if time.time() > deadline - 30.0:
            break
        
        if restart == 0:
            x_init = x0.copy()
        else:
            # Perturb the initial solution
            x_init = x0.copy()
            # Add small random perturbation to positions
            noise = np.random.normal(0, 0.05, size=(N, 2))
            x_init[1:] += noise.flatten()
            # Slightly perturb w
            x_init[0] += np.random.normal(0, 0.05)
        
        # Use L-BFGS-B
        try:
            result = minimize(
                _objective_and_grad,
                x_init,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.success and -result.fun < best_val:
                best_val = -result.fun
                best_x = result.x.copy()
        except Exception:
            pass
        
        if time.time() > deadline - 20.0:
            break
    
    # Polish with more iterations
    if time.time() < deadline - 10.0:
        try:
            result = minimize(
                _objective_and_grad,
                best_x,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if result.success and -result.fun < best_val:
                best_val = -result.fun
                best_x = result.x.copy()
        except Exception:
            pass
    
    # Extract final solution
    w = best_x[0]
    h = 2.0 - w
    pts = best_x[1:].reshape(N, 2)
    
    # Compute radii
    radii = np.zeros(N)
    for i in range(N):
        dist_boundary = min(pts[i, 0], pts[i, 1], w - pts[i, 0], h - pts[i, 1])
        if i < N - 1:
            dists = np.sqrt(np.sum((pts[i+1:] - pts[i])**2, axis=1))
            if i > 0:
                dists_prev = np.sqrt(np.sum((pts[:i] - pts[i])**2, axis=1))
                min_dist = min(np.min(dists), np.min(dists_prev))
            else:
                min_dist = np.min(dists)
        else:
            dists_prev = np.sqrt(np.sum((pts[:i] - pts[i])**2, axis=1))
            min_dist = np.min(dists_prev)
        radii[i] = min(dist_boundary, min_dist / 2)
    
    # Clamp to ensure validity
    for i in range(N):
        pts[i, 0] = np.clip(pts[i, 0], radii[i], w - radii[i])
        pts[i, 1] = np.clip(pts[i, 1], radii[i], h - radii[i])
    
    # Re-compute radii after clamping
    radii = np.zeros(N)
    for i in range(N):
        dist_boundary = min(pts[i, 0], pts[i, 1], w - pts[i, 0], h - pts[i, 1])
        if i < N - 1:
            dists = np.sqrt(np.sum((pts[i+1:] - pts[i])**2, axis=1))
            if i > 0:
                dists_prev = np.sqrt(np.sum((pts[:i] - pts[i])**2, axis=1))
                min_dist = min(np.min(dists), np.min(dists_prev))
            else:
                min_dist = np.min(dists)
        else:
            dists_prev = np.sqrt(np.sum((pts[:i] - pts[i])**2, axis=1))
            min_dist = np.min(dists_prev)
        radii[i] = min(dist_boundary, min_dist / 2)
    
    result = np.column_stack([pts, radii])
    return result

# EVOLVE-BLOCK-END
