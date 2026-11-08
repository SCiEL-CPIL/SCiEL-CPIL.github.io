"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Voronoi, ConvexHull
import random


def _voronoi_inradii(centers, width, height):
    """
    Compute the inradius (max radius) for each circle center within the rectangle,
    using Voronoi tessellation.
    """
    n = len(centers)
    if n == 0:
        return np.array([])
    
    # Add boundary points to Voronoi to handle the rectangle boundary
    # We need to compute the distance from each center to the boundary of its Voronoi cell
    # which is the min distance to:
    # 1. Other centers (half the distance to nearest neighbor)
    # 2. The rectangle boundaries (x=0, x=width, y=0, y=height)
    
    radii = np.zeros(n)
    
    for i in range(n):
        cx, cy = centers[i]
        # Distance to rectangle boundaries
        dist_to_boundary = min(cx, width - cx, cy, height - cy)
        
        # Distance to other centers: half the minimum distance to any other center
        min_dist_to_other = float('inf')
        for j in range(n):
            if i == j:
                continue
            dx = cx - centers[j][0]
            dy = cy - centers[j][1]
            d = np.sqrt(dx * dx + dy * dy)
            if d < min_dist_to_other:
                min_dist_to_other = d
        
        # The inradius is the minimum of distance to boundary and half distance to nearest neighbor
        # But actually, the Voronoi cell inradius is the max radius such that the circle
        # is contained in the Voronoi cell. The Voronoi cell edges are perpendicular bisectors
        # between center i and center j. So the distance from center i to the edge between i and j
        # is d/2. But the cell is also bounded by the rectangle.
        # The inradius is the min over all Voronoi edges of the distance from the center to that edge.
        # For edges between i and j: distance is d_ij / 2
        # For edges with the rectangle boundary: distance is the distance to that boundary
        
        r = dist_to_boundary
        for j in range(n):
            if i == j:
                continue
            dx = cx - centers[j][0]
            dy = cy - centers[j][1]
            d = np.sqrt(dx * dx + dy * dy)
            half_d = d / 2.0
            if half_d < r:
                r = half_d
        
        radii[i] = r
    
    return radii


def _repulsion_forces(centers, width, height):
    """
    Compute repulsion-based forces on centers.
    We want to maximize the sum of inradii. The inradius of circle i is limited by:
    - Distance to rectangle boundary
    - Distance to other circles (half the center-to-center distance)
    
    To maximize sum of inradii, we want to push circles apart and away from boundaries
    when they are constrained by boundaries, but actually we want to balance things.
    
    A simple approach: use a force model where each pair of circles repels, and
    circles are also attracted to the center of the rectangle to avoid being too close
    to boundaries.
    
    Actually, let's think more carefully. The inradius r_i = min(dist_to_boundary_i, min_j d_ij/2).
    To maximize sum of r_i, we should:
    - Move circles apart to increase d_ij
    - Move circles away from boundaries to increase dist_to_boundary
    
    But there's a trade-off: moving a circle away from one boundary may bring it closer to another.
    
    Let's use a gradient-based approach with a penalty for overlaps and a reward for being
    well-centered.
    """
    n = len(centers)
    forces = np.zeros_like(centers)
    
    for i in range(n):
        cx, cy = centers[i]
        fx = 0.0
        fy = 0.0
        
        # Repulsion from other circles
        for j in range(n):
            if i == j:
                continue
            dx = cx - centers[j][0]
            dy = cy - centers[j][1]
            d = np.sqrt(dx * dx + dy * dy)
            if d < 1e-10:
                # Add small random perturbation
                fx += random.uniform(-1, 1)
                fy += random.uniform(-1, 1)
                continue
            # Force magnitude inversely proportional to distance squared
            force_mag = 1.0 / (d * d)
            fx += force_mag * dx / d
            fy += force_mag * dy / d
        
        # Attraction to center of rectangle (to avoid boundary)
        cx_center = width / 2.0
        cy_center = height / 2.0
        attract_strength = 0.1
        fx += attract_strength * (cx_center - cx)
        fy += attract_strength * (cy_center - cy)
        
        forces[i][0] = fx
        forces[i][1] = fy
    
    return forces


def construct():
    """
    Pack 21 circles in a rectangle of perimeter 4 (so w + h = 2).
    Maximize the sum of radii.
    """
    random.seed(42)
    np.random.seed(42)
    
    # Perimeter is 4, so 2*(w + h) = 4, thus w + h = 2
    # We'll try different aspect ratios
    
    best_sum = 0
    best_centers = None
    best_width = 1.0
    best_height = 1.0
    
    # Try several aspect ratios
    aspect_ratios = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 2.0]
    
    for ar in aspect_ratios:
        # w + h = 2, w/h = ar => h(1 + ar) = 2 => h = 2/(1+ar), w = 2*ar/(1+ar)
        h = 2.0 / (1.0 + ar)
        w = 2.0 * ar / (1.0 + ar)
        
        # Initialize centers with a grid-like pattern plus some randomness
        # For 21 circles, try a 5x5 grid and remove 4, or 7x3 grid
        # Let's try multiple initializations
        
        for init_iter in range(5):
            # Grid initialization
            cols = 5
            rows = 5
            centers = []
            count = 0
            for i in range(cols):
                for j in range(rows):
                    if count >= 21:
                        break
                    cx = (i + 0.5) * w / cols + random.uniform(-0.02, 0.02)
                    cy = (j + 0.5) * h / rows + random.uniform(-0.02, 0.02)
                    # Clamp to stay inside
                    cx = max(0.01, min(w - 0.01, cx))
                    cy = max(0.01, min(h - 0.01, cy))
                    centers.append([cx, cy])
                    count += 1
                if count >= 21:
                    break
            
            if len(centers) < 21:
                continue
            
            centers = np.array(centers[:21])
            
            # Optimize using force-based relaxation
            learning_rate = 0.01
            for step in range(2000):
                forces = _repulsion_forces(centers, w, h)
                
                # Update centers
                centers += learning_rate * forces
                
                # Clamp to stay inside the rectangle with a small margin
                centers[:, 0] = np.clip(centers[:, 0], 0.001, w - 0.001)
                centers[:, 1] = np.clip(centers[:, 1], 0.001, h - 0.001)
                
                # Decrease learning rate
                learning_rate *= 0.999
            
            # Compute the sum of inradii
            radii = _voronoi_inradii(centers, w, h)
            sum_radii = np.sum(radii)
            
            if sum_radii > best_sum:
                best_sum = sum_radii
                best_centers = centers.copy()
                best_width = w
                best_height = h
    
    # Return the best packing
    # Format: (n, 3) array with [x, y, r]
    radii = _voronoi_inradii(best_centers, best_width, best_height)
    
    # Ensure circles don't overlap by slightly reducing radii if needed
    # Check for overlaps and adjust
    for i in range(N):
        for j in range(i + 1, N):
            dx = best_centers[i][0] - best_centers[j][0]
            dy = best_centers[i][1] - best_centers[j][1]
            d = np.sqrt(dx * dx + dy * dy)
            min_r = d / 2.0
            if radii[i] > min_r:
                radii[i] = min_r
            if radii[j] > min_r:
                radii[j] = min_r
    
    # Also ensure circles don't go outside the rectangle
    for i in range(N):
        cx, cy = best_centers[i]
        max_r = min(cx, best_width - cx, cy, best_height - cy)
        if radii[i] > max_r:
            radii[i] = max_r
    
    result = np.zeros((N, 3))
    for i in range(N):
        result[i][0] = best_centers[i][0]
        result[i][1] = best_centers[i][1]
        result[i][2] = radii[i]
    
    return result
# EVOLVE-BLOCK-END
