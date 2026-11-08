"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, voronoi

def _compute_inradii(centers, w, h):
    """
    Compute the inradius of the Voronoi cell for each center,
    clipped to the boundary of the rectangle [0, w] x [0, h].
    """
    n = len(centers)
    radii = np.zeros(n)
    
    # Compute Voronoi diagram
    # Note: scipy.spatial.Voronoi works on 2D points
    try:
        vor = voronoi(centers)
    except Exception:
        # Fallback if Voronoi fails (e.g. duplicate points)
        return radii

    for i in range(n):
        # Get the region index for point i
        region_idx = vor.point_region[i]
        region = vor.regions[region_idx]
        
        # If region is empty, the point is on the convex hull and region is -1
        # But wait, scipy Voronoi regions for hull points are empty lists in some versions, 
        # or they might be handled differently. Actually, for points on the hull,
        # the region is open to infinity. We need to intersect with the rectangle.
        
        # The vertices of the Voronoi cell are vor.vertices[region]
        # However, for hull points, the cell extends to infinity.
        # We need to find the largest circle centered at centers[i] that fits inside
        # the Voronoi cell AND the rectangle [0,w]x[0,h].
        
        # Distance to rectangle boundaries
        d_rect = min(centers[i][0], centers[i][1], w - centers[i][0], h - centers[i][1])
        
        # Distance to Voronoi edges
        # A Voronoi edge is defined by two vertices.
        # If the region is bounded (not on hull), we can check distance to edges.
        # If on hull, the "edge" is at infinity, so only rectangle boundaries matter?
        # No, the Voronoi cell for a hull point is unbounded in some directions, 
        # but bounded by the perpendicular bisectors of edges to other points.
        # Actually, the Voronoi cell is always the intersection of half-planes.
        
        # More robust approach:
        # The inradius is the minimum distance from the center to:
        # 1. The boundary of the rectangle
        # 2. The perpendicular bisectors of the segments connecting to other centers?
        # No, the Voronoi cell boundary is composed of segments from the Voronoi diagram.
        
        # Let's use the vertices of the Voronoi region.
        # If the region is empty (hull point), we need to handle the unbounded edges.
        # Actually, a simpler geometric check:
        # The radius r_i is the minimum over all other points j of:
        #   0.5 * ||p_i - p_j||
        # And the boundary of the rectangle.
        # Wait, that's only true if the nearest neighbor defines the constraint.
        # The Voronoi cell is defined by the nearest neighbors.
        # The inradius of the Voronoi cell is the distance to the nearest edge.
        # The edges are perpendicular bisectors.
        # The distance from p_i to the bisector of p_i and p_j is 0.5 * ||p_i - p_j||.
        # So the inradius is min( min_j 0.5*||p_i-p_j||, distance_to_boundary ).
        # Is this always true?
        # The Voronoi cell V_i = { x | ||x-p_i|| <= ||x-p_j|| for all j }.
        # The boundary of V_i consists of parts of the bisectors.
        # The distance from p_i to the boundary of V_i is indeed the minimum distance to any bisector.
        # The distance from p_i to the bisector of p_i and p_j is exactly 0.5 * ||p_i - p_j||.
        # So, r_i = min( min_{j != i} 0.5 * ||p_i - p_j||, d_rect_i ).
        
        min_dist = np.inf
        for j in range(n):
            if i == j:
                continue
            dist = np.linalg.norm(centers[i] - centers[j])
            min_dist = min(min_dist, 0.5 * dist)
        
        radii[i] = min(min_dist, d_rect)
        
    return radii

def _objective_and_grad(centers, w, h):
    """
    Objective: -sum of inradii
    Gradient: computed via finite differences or analytic if possible.
    Since inradius is a min function, it's non-smooth.
    We will use a smooth approximation or just rely on L-BFGS with finite differences?
    L-BFGS with finite differences is slow.
    Let's try to compute a subgradient or use a smooth surrogate.
    
    Actually, for a quick implementation, we can just use the objective function
    and let scipy's minimize handle it, but it's better to provide a gradient.
    
    Let's stick to a simple approach first: use the objective function only.
    """
    radii = _compute_inradii(centers, w, h)
    return -np.sum(radii)

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    best_solution = None
    best_value = -np.inf
    
    # Try multiple aspect ratios
    aspect_ratios = [1.0, 1.2, 1.5, 2.0, 0.8, 0.6, 0.5, 1.3, 1.8, 1.1, 0.9]
    
    for aspect in aspect_ratios:
        if time.time() > deadline - 5:
            break
            
        # w + h = 2, h/w = 1/aspect => w*(1 + 1/aspect) = 2
        w = 2.0 / (1 + 1/aspect)
        h = 2.0 - w
        
        if w <= 0 or h <= 0:
            continue
        
        # Initialize centers in a grid
        n = N
        side = int(np.ceil(np.sqrt(n)))
        centers0 = np.zeros((n, 2))
        for i in range(side):
            for j in range(side):
                idx = i * side + j
                if idx >= n:
                    break
                centers0[idx, 0] = (i + 0.5) * w / side
                centers0[idx, 1] = (j + 0.5) * h / side
        
        # Flatten for optimization
        x0 = centers0.flatten()
        
        # Optimize
        try:
            def obj(x):
                c = x.reshape(-1, 2)
                return _objective_and_grad(c, w, h)
            
            result = minimize(
                obj,
                x0,
                method='L-BFGS-B',
                bounds=[(0, w)] * n + [(0, h)] * n,
                options={'maxiter': 1000, 'ftol': 1e-8}
            )
            
            if result.fun < best_value:
                best_value = -result.fun
                best_solution = result.x.reshape(-1, 2)
        except Exception:
            pass
        
        if time.time() > deadline - 5:
            break
    
    # If no valid solution found, fall back to simple grid
    if best_solution is None:
        side = int(np.ceil(np.sqrt(N)))
        w = 1.0
        h = 1.0
        centers = np.zeros((N, 2))
        for i in range(side):
            for j in range(side):
                idx = i * side + j
                if idx >= N:
                    break
                centers[idx, 0] = (i + 0.5) * w / side
                centers[idx, 1] = (j + 0.5) * h / side
        best_solution = centers
        best_value = 0.0

    # Compute final radii
    # We need to determine w and h for the best solution
    # We didn't store them, so let's re-derive or store them.
    # Let's modify the loop to store w, h with the best solution.
    
    # Re-run to find best w,h
    best_solution_final = None
    best_value_final = -np.inf
    best_w = 1.0
    best_h = 1.0
    
    for aspect in aspect_ratios:
        if time.time() > deadline - 2:
            break
            
        w = 2.0 / (1 + 1/aspect)
        h = 2.0 - w
        
        if w <= 0 or h <= 0:
            continue
        
        n = N
        side = int(np.ceil(np.sqrt(n)))
        centers0 = np.zeros((n, 2))
        for i in range(side):
            for j in range(side):
                idx = i * side + j
                if idx >= n:
                    break
                centers0[idx, 0] = (i + 0.5) * w / side
                centers0[idx, 1] = (j + 0.5) * h / side
        
        x0 = centers0.flatten()
        
        try:
            def obj(x):
                c = x.reshape(-1, 2)
                return _objective_and_grad(c, w, h)
            
            result = minimize(
                obj,
                x0,
                method='L-BFGS-B',
                bounds=[(0, w)] * n + [(0, h)] * n,
                options={'maxiter': 1000, 'ftol': 1e-8}
            )
            
            if result.fun < best_value_final:
                best_value_final = -result.fun
                best_solution_final = result.x.reshape(-1, 2)
                best_w = w
                best_h = h
        except Exception:
            pass

    if best_solution_final is None:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        w = 1.0
        h = 1.0
        centers = np.zeros((N, 2))
        for i in range(side):
            for j in range(side):
                idx = i * side + j
                if idx >= N:
                    break
                centers[idx, 0] = (i + 0.5) * w / side
                centers[idx, 1] = (j + 0.5) * h / side
        best_solution_final = centers
        best_w = 1.0
        best_h = 1.0

    radii = _compute_inradii(best_solution_final, best_w, best_h)
    
    # Construct output array (cx, cy, r)
    result = np.zeros((N, 3))
    result[:, 0] = best_solution_final[:, 0]
    result[:, 1] = best_solution_final[:, 1]
    result[:, 2] = radii
    
    return result
# EVOLVE-BLOCK-END
