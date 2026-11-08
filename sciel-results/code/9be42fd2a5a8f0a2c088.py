"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import Voronoi, Delaunay
import random
import math

def _compute_voronoi_inradii(centers, W, H):
    """Compute the inradius of each Voronoi cell for points in [0,W]x[0,H].
    
    The inradius is the distance from the point to the boundary of its Voronoi cell,
    clipped by the rectangle boundaries.
    """
    n = len(centers)
    if n < 3:
        # Special case: just compute min distance to rectangle edges
        inradii = np.zeros(n)
        for i in range(n):
            x, y = centers[i]
            inradii[i] = min(x, W - x, y, H - y)
        return inradii
    
    try:
        vor = Voronoi(centers)
    except Exception:
        # Fallback: min distance to rectangle edges
        inradii = np.zeros(n)
        for i in range(n):
            x, y = centers[i]
            inradii[i] = min(x, W - x, y, H - y)
        return inradii
    
    inradii = np.zeros(n)
    for i in range(n):
        x, y = centers[i]
        # Distance to rectangle boundaries
        d_rect = min(x, W - x, y, H - y)
        
        # Get the Voronoi region for point i
        region_idx = vor.point_region[i]
        region = vor.regions[region_idx]
        
        if -1 in region:
            # Unbounded region (shouldn't happen for interior points, but handle it)
            inradii[i] = d_rect
            continue
        
        d_voronoi = d_rect
        for edge_idx in region:
            p1 = vor.vertices[edge_idx[0]]
            p2 = vor.vertices[edge_idx[1]]
            # Distance from (x,y) to line segment p1-p2
            # Actually, for Voronoi, we need distance to the infinite line, but clipped
            # The Voronoi edges are infinite lines, so compute distance to the line
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            length = math.sqrt(dx*dx + dy*dy)
            if length < 1e-15:
                continue
            # Distance from point to line through p1, p2
            # Line equation: (y - p1[1])*(p2[0]-p1[0]) - (x - p1[0])*(p2[1]-p1[1]) = 0
            dist = abs((y - p1[1])*dx - (x - p1[0])*dy) / length
            d_voronoi = min(d_voronoi, dist)
        
        inradii[i] = d_voronoi
    
    return inradii


def _objective(centers, W, H):
    """Compute the sum of Voronoi inradii for given centers in rectangle W x H."""
    inradii = _compute_voronoi_inradii(centers, W, H)
    return np.sum(inradii), inradii


def _repulsion_forces(centers, W, H):
    """Compute repulsion forces to push centers apart, weighted by their inradii.
    
    We want to maximize sum of inradii. A gradient-like approach:
    - If a circle is close to its neighbor, increasing distance helps both.
    - Use pairwise repulsion inversely proportional to distance.
    """
    n = len(centers)
    forces = np.zeros((n, 2))
    inradii = _compute_voronoi_inradii(centers, W, H)
    
    for i in range(n):
        for j in range(i+1, n):
            dx = centers[i][0] - centers[j][0]
            dy = centers[i][1] - centers[j][1]
            dist = math.sqrt(dx*dx + dy*dy)
            if dist < 1e-10:
                continue
            # Force proportional to (r_i + r_j - dist) if overlapping, or just repulsion
            # Use a soft repulsion: F ~ (r_i + r_j)^2 / dist^3
            r_sum = inradii[i] + inradii[j]
            if dist < r_sum:
                # Overlapping: strong repulsion
                strength = 10.0 * (r_sum - dist) / dist
            else:
                # Non-overlapping: gentle repulsion to spread out
                strength = 0.1 * (r_sum * r_sum) / (dist * dist)
            
            fx = strength * dx / dist
            fy = strength * dy / dist
            forces[i][0] += fx
            forces[i][1] += fy
            forces[j][0] -= fx
            forces[j][1] -= fy
        
        # Attraction toward center of rectangle (to keep circles inside)
        cx = centers[i][0]
        cy = centers[i][1]
        # Pull toward center if too close to edge
        edge_margin = inradii[i]
        if cx < edge_margin:
            forces[i][0] += 0.5 * (edge_margin - cx)
        elif cx > W - edge_margin:
            forces[i][0] -= 0.5 * (cx - (W - edge_margin))
        if cy < edge_margin:
            forces[i][1] += 0.5 * (edge_margin - cy)
        elif cy > H - edge_margin:
            forces[i][1] -= 0.5 * (cy - (H - edge_margin))
    
    return forces


def _optimize_centers(centers, W, H, max_iters=5000, lr_init=0.01, deadline=None):
    """Optimize center positions to maximize sum of Voronoi inradii."""
    n = len(centers)
    centers = centers.copy()
    best_sum, best_centers = _objective(centers, W, H)
    best_centers = centers.copy()
    
    lr = lr_init
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        forces = _repulsion_forces(centers, W, H)
        
        # Update centers
        centers = centers + lr * forces
        
        # Clip to rectangle (with small margin to avoid numerical issues)
        centers[:, 0] = np.clip(centers[:, 0], 0.0, W)
        centers[:, 1] = np.clip(centers[:, 1], 0.0, H)
        
        # Evaluate
        cur_sum, _ = _objective(centers, W, H)
        if cur_sum > best_sum:
            best_sum = cur_sum
            best_centers = centers.copy()
        
        # Adaptive learning rate
        if it % 100 == 0 and it > 0:
            # Decay learning rate
            lr *= 0.95
        
        # Random restarts within the optimization
        if it % 2000 == 1000:
            # Small random perturbation
            centers += np.random.randn(n, 2) * 0.01
            centers[:, 0] = np.clip(centers[:, 0], 0.0, W)
            centers[:, 1] = np.clip(centers[:, 1], 0.0, H)
    
    return best_centers, best_sum


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_value = -1.0
    best_result = None
    
    # Try multiple aspect ratios
    # Perimeter = 2*(W+H) = 4, so W+H = 2
    # Let W = w, H = 2 - w
    # Try a range of aspect ratios
    aspect_ratios = [
        0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.8, 2.0, 2.5, 3.0
    ]
    
    for W in aspect_ratios:
        if time.time() > deadline:
            break
        H = 2.0 - W
        if H <= 0:
            continue
        
        # Initialize centers in a grid-like pattern
        side = int(math.ceil(math.sqrt(N * W / H))) if H > 0 else int(math.ceil(math.sqrt(N)))
        side = max(side, 1)
        rows = int(math.ceil(N / side))
        
        centers = np.zeros((N, 2))
        for idx in range(N):
            i = idx % side
            j = idx // side
            centers[idx, 0] = (i + 0.5) * W / side
            centers[idx, 1] = (j + 0.5) * H / rows
        
        # Ensure centers are within bounds
        centers[:, 0] = np.clip(centers[:, 0], 0.0, W)
        centers[:, 1] = np.clip(centers[:, 1], 0.0, H)
        
        # Optimize
        opt_centers, opt_sum = _optimize_centers(centers, W, H, max_iters=3000, lr_init=0.02, deadline=deadline)
        
        if opt_sum > best_value:
            best_value = opt_sum
            best_result = (opt_centers, W, H)
    
    # If we haven't found anything, fall back to a simple grid
    if best_result is None:
        W = 1.0
        H = 1.0
        side = int(math.ceil(math.sqrt(N)))
        centers = np.zeros((N, 2))
        for idx in range(N):
            i = idx % side
            j = idx // side
            centers[idx, 0] = (i + 0.5) / side
            centers[idx, 1] = (j + 0.5) / side
        best_result = (centers, W, H)
    
    centers, W, H = best_result
    
    # Compute final radii using Voronoi inradii
    inradii = _compute_voronoi_inradii(centers, W, H)
    
    # Build result: (x, y, radius) for each circle
    result = np.zeros((N, 3))
    for i in range(N):
        result[i, 0] = centers[i, 0]
        result[i, 1] = centers[i, 1]
        result[i, 2] = inradii[i]
    
    return result
# EVOLVE-BLOCK-END
