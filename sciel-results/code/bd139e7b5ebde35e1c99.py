"""Improved program for the 200-circle packing task using Delaunay-based optimization."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Delaunay, ConvexHull
from scipy.spatial.distance import cdist
import time


def _compute_radius(centres):
    """Compute the maximum common radius for given centres."""
    if len(centres) < 2:
        min_dist_to_border = np.min(0.5 - np.abs(centres))
        return float(min_dist_to_border)
    
    dists = cdist(centres, centres)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    min_dist_to_border = np.min(0.5 - np.abs(centres))
    
    return float(min(min_pair_dist / 2.0, min_dist_to_border))


def _get_delaunay_edges(tri):
    """Extract unique edges from a Delaunay triangulation."""
    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                edge = (min(simplex[i], simplex[j]), max(simplex[i], simplex[j]))
                edges.add(edge)
    return list(edges)


def _find_flippable_edges(tri, centres):
    """
    Find edges that can be flipped in the Delaunay triangulation.
    An edge is flippable if it's shared by exactly two triangles and the
    opposite edge would lie inside the convex hull of the four points.
    """
    edges = _get_delaunay_edges(tri)
    flippable = []
    
    for e in edges:
        i, j = e
        # Find triangles containing this edge
        tris_with_edge = []
        for simplex in tri.simplices:
            if i in simplex and j in simplex:
                tris_with_edge.append(simplex)
        
        if len(tris_with_edge) != 2:
            continue
        
        # Get the two third vertices
        third_a = [v for v in tris_with_edge[0] if v != i and v != j][0]
        third_b = [v for v in tris_with_edge[1] if v != i and v != j][0]
        
        # Check if the new edge (third_a, third_b) would be valid
        # by checking if it doesn't cross other edges
        new_edge = (min(third_a, third_b), max(third_a, third_b))
        
        # Simple check: verify the four points form a convex quadrilateral
        # where the diagonal (i,j) can be flipped to (third_a, third_b)
        pts = np.array([centres[i], centres[third_a], centres[j], centres[third_b]])
        
        # Check if the quadrilateral is convex and the new diagonal is inside
        # Use cross product to check orientation
        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        
        # Check if the four points form a convex quadrilateral
        # in the order i, third_a, j, third_b
        c1 = cross(pts[0], pts[1], pts[2])
        c2 = cross(pts[1], pts[2], pts[3])
        c3 = cross(pts[2], pts[3], pts[0])
        c4 = cross(pts[3], pts[0], pts[1])
        
        if c1 * c2 > 0 and c2 * c3 > 0 and c3 * c4 > 0 and c4 * c1 > 0:
            # Convex quadrilateral - the edge is flippable
            flippable.append((i, j, third_a, third_b))
    
    return flippable


def _apply_flip(centres, i, j, a, b):
    """
    Attempt to flip an edge by moving points to increase the minimum edge length.
    Returns new centres if the flip improves the minimum distance, else None.
    """
    new_centres = centres.copy()
    
    # The idea: after flipping, we want the new configuration to have
    # larger minimum edge lengths. We'll nudge points slightly.
    
    # Compute current minimum edge length
    dists = cdist(centres, centres)
    np.fill_diagonal(dists, np.inf)
    current_min = np.min(dists)
    
    # Try small perturbations to see if we can improve
    # Move points i and j slightly apart, and a and b slightly apart
    for step in [0.001, 0.002, 0.005, 0.01]:
        # Move i away from j
        dir_ij = centres[i] - centres[j]
        norm_ij = np.linalg.norm(dir_ij)
        if norm_ij > 1e-10:
            dir_ij = dir_ij / norm_ij
        
        # Move a away from b
        dir_ab = centres[a] - centres[b]
        norm_ab = np.linalg.norm(dir_ab)
        if norm_ab > 1e-10:
            dir_ab = dir_ab / norm_ab
        
        new_centres[i] += dir_ij * step
        new_centres[j] -= dir_ij * step
        new_centres[a] += dir_ab * step
        new_centres[b] -= dir_ab * step
        
        # Keep within bounds
        new_centres = np.clip(new_centres, -0.5, 0.5)
        
        # Check if this improves the minimum distance
        new_dists = cdist(new_centres, new_centres)
        np.fill_diagonal(new_dists, np.inf)
        new_min = np.min(new_dists)
        
        if new_min > current_min * 1.001:
            return new_centres
    
    return None


def _delaunay_optimize(centres, max_iterations=100, time_limit=100):
    """
    Optimize centres using Delaunay triangulation and edge flips.
    """
    best_centres = centres.copy()
    best_radius = _compute_radius(centres)
    start_time = time.time()
    
    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        
        # Compute Delaunay triangulation
        try:
            tri = Delaunay(centres)
        except Exception:
            break
        
        # Find flippable edges
        flippable = _find_flippable_edges(tri, centres)
        
        if not flippable:
            # No flippable edges - try random perturbations
            for _ in range(10):
                idx = np.random.randint(len(centres))
                perturbation = np.random.randn(2) * 0.005
                new_centres = centres.copy()
                new_centres[idx] += perturbation
                new_centres = np.clip(new_centres, -0.5, 0.5)
                
                new_radius = _compute_radius(new_centres)
                if new_radius > best_radius + 1e-10:
                    best_radius = new_radius
                    best_centres = new_centres.copy()
                    centres = new_centres.copy()
                    break
            else:
                break
            continue
        
        # Try each flippable edge
        improved = False
        for i, j, a, b in flippable:
            new_centres = _apply_flip(centres, i, j, a, b)
            if new_centres is not None:
                new_radius = _compute_radius(new_centres)
                if new_radius > best_radius + 1e-10:
                    best_radius = new_radius
                    best_centres = new_centres.copy()
                    centres = new_centres.copy()
                    improved = True
                    break
        
        if not improved:
            # Try a different strategy: nudge the most constrained points
            dists = cdist(centres, centres)
            np.fill_diagonal(dists, np.inf)
            min_i, min_j = np.unravel_index(np.argmin(dists), dists.shape)
            
            direction = centres[min_i] - centres[min_j]
            norm = np.linalg.norm(direction)
            if norm > 1e-10:
                direction = direction / norm
            
            for step in [0.002, 0.005, 0.01]:
                new_centres = centres.copy()
                new_centres[min_i] += direction * step
                new_centres[min_j] -= direction * step
                new_centres = np.clip(new_centres, -0.5, 0.5)
                
                new_radius = _compute_radius(new_centres)
                if new_radius > best_radius + 1e-10:
                    best_radius = new_radius
                    best_centres = new_centres.copy()
                    centres = new_centres.copy()
                    improved = True
                    break
            
            if not improved:
                break
    
    return best_centres, best_radius


def construct():
    """
    Construct centres of 200 equal circles inside the unit square [-1/2, 1/2]^2
    to maximize the common radius.
    """
    # Start with a hexagonal-like grid arrangement
    # Use a slightly distorted grid to break symmetry
    m = 15  # 15x15 = 225, we'll use 200
    ticks = (np.arange(m) + 0.5) / m - 0.5
    
    # Create a grid with slight hexagonal distortion
    centres_list = []
    for row, y in enumerate(ticks):
        for col, x in enumerate(ticks):
            # Add slight offset for hexagonal pattern
            offset = 0.3 / m if row % 2 == 1 else 0
            cx = x + offset
            cy = y
            if -0.5 <= cx <= 0.5 and -0.5 <= cy <= 0.5:
                centres_list.append([cx, cy])
    
    centres = np.array(centres_list[:N])
    
    best_centres = centres.copy()
    best_radius = _compute_radius(centres)
    
    # Run Delaunay-based optimization
    optimised, radius = _delaunay_optimize(centres, max_iterations=50, time_limit=100)
    
    if radius > best_radius:
        best_radius = radius
        best_centres = optimised.copy()
    
    return best_centres
# EVOLVE-BLOCK-END
