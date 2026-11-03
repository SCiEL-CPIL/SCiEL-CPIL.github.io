"""Improved program for the 200-circle packing task using Delaunay edge-flip optimization."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist
import random
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
    """Extract unique edges from a Delaunay triangulation as a set of frozensets."""
    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                edge = frozenset([simplex[i], simplex[j]])
                edges.add(edge)
    return edges


def _find_adjacent_triangles(tri, edge):
    """Find triangles that contain the given edge."""
    a, b = tuple(edge)
    result = []
    for simplex in tri.simplices:
        if a in simplex and b in simplex:
            result.append(simplex)
    return result


def _can_flip(tri, centres, edge):
    """
    Check if an edge can be flipped. An edge can be flipped if:
    1. It is shared by exactly two triangles
    2. The four vertices form a convex quadrilateral
    3. The new diagonal would be inside the convex hull
    """
    a, b = tuple(edge)
    triangles = _find_adjacent_triangles(tri, edge)
    
    if len(triangles) != 2:
        return False
    
    # Get the other two vertices
    tri1 = set(triangles[0])
    tri2 = set(triangles[1])
    other1 = tri1 - {a, b}
    other2 = tri2 - {a, b}
    
    if len(other1) != 1 or len(other2) != 1:
        return False
    
    c = list(other1)[0]
    d = list(other2)[0]
    
    # Check if the quadrilateral a-b-c-d is convex and if the flip would be valid
    # We need to check that c and d are on opposite sides of the line ab
    # and that the new edge cd would be inside the convex hull
    
    # Compute cross products to check convexity
    ab = centres[b] - centres[a]
    ac = centres[c] - centres[a]
    ad = centres[d] - centres[a]
    
    cross1 = ab[0] * ac[1] - ab[1] * ac[0]
    cross2 = ab[0] * ad[1] - ab[1] * ad[0]
    
    # c and d should be on opposite sides of line ab
    if cross1 * cross2 > 0:
        return False
    
    # Check that the new edge cd doesn't create an invalid configuration
    # by verifying that the midpoint of cd is inside the quadrilateral
    mid_cd = (centres[c] + centres[d]) / 2
    
    # Check if mid_cd is inside triangle abc or abd
    def point_in_triangle(p, t1, t2, t3):
        """Check if point p is inside triangle t1-t2-t3."""
        def sign(p1, p2, p3):
            return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
        
        d1 = sign(p, t1, t2)
        d2 = sign(p, t2, t3)
        d3 = sign(p, t3, t1)
        
        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
        
        return not (has_neg and has_pos)
    
    inside_abc = point_in_triangle(mid_cd, centres[a], centres[b], centres[c])
    inside_abd = point_in_triangle(mid_cd, centres[a], centres[b], centres[d])
    
    return inside_abc or inside_abd


def _flip_edge(centres, edge):
    """
    Perform an edge flip by adjusting the positions of the four vertices
    to maximize the minimum edge length in the local neighborhood.
    Returns new centres if the flip improves the minimum distance.
    """
    a, b = tuple(edge)
    
    # Find the other two vertices
    tri = Delaunay(centres)
    triangles = _find_adjacent_triangles(tri, edge)
    
    if len(triangles) != 2:
        return None
    
    tri1 = set(triangles[0])
    tri2 = set(triangles[1])
    other1 = tri1 - {a, b}
    other2 = tri2 - {a, b}
    
    if len(other1) != 1 or len(other2) != 1:
        return None
    
    c = list(other1)[0]
    d = list(other2)[0]
    
    # The four vertices involved in the flip
    verts = [a, b, c, d]
    
    # Current positions
    orig_positions = centres[verts].copy()
    
    # Compute current minimum distance among these four vertices
    dists_4 = cdist(orig_positions, orig_positions)
    np.fill_diagonal(dists_4, np.inf)
    current_min = np.min(dists_4)
    
    # Try to adjust positions to increase the minimum distance
    # We'll use a simple optimization: try moving each vertex slightly
    # in a direction that increases its distance to the closest neighbor
    
    best_positions = orig_positions.copy()
    best_min = current_min
    
    # Try small perturbations
    for _ in range(20):
        # Random direction for each vertex
        directions = np.random.randn(4, 2)
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        
        # Small step
        step = 0.002
        new_positions = orig_positions + directions * step
        
        # Keep within bounds
        new_positions = np.clip(new_positions, -0.5, 0.5)
        
        # Check minimum distance
        dists_new = cdist(new_positions, new_positions)
        np.fill_diagonal(dists_new, np.inf)
        new_min = np.min(dists_new)
        
        if new_min > best_min + 1e-10:
            best_min = new_min
            best_positions = new_positions.copy()
    
    # If we found an improvement, update the centres
    if best_min > current_min + 1e-10:
        new_centres = centres.copy()
        new_centres[verts] = best_positions
        return new_centres
    
    return None


def _delaunay_optimize(centres, max_iterations=1000, time_limit=60):
    """
    Optimize using Delaunay triangulation and edge flips.
    """
    start_time = time.time()
    best_centres = centres.copy()
    best_radius = _compute_radius(centres)
    
    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        
        # Compute Delaunay triangulation
        try:
            tri = Delaunay(centres)
        except Exception:
            break
        
        # Get all edges
        edges = _get_delaunay_edges(tri)
        
        # Shuffle edges for random order
        edge_list = list(edges)
        random.shuffle(edge_list)
        
        improved = False
        
        for edge in edge_list:
            if time.time() - start_time > time_limit:
                break
            
            # Check if this edge can be flipped
            if not _can_flip(tri, centres, edge):
                continue
            
            # Try the flip
            new_centres = _flip_edge(centres, edge)
            
            if new_centres is not None:
                new_radius = _compute_radius(new_centres)
                
                if new_radius > best_radius + 1e-10:
                    best_radius = new_radius
                    best_centres = new_centres.copy()
                    improved = True
                    # Recompute Delaunay after improvement
                    break
        
        if not improved:
            # Try a random perturbation to escape local minima
            if iteration > max_iterations // 2:
                idx = np.random.randint(len(centres))
                perturbation = np.random.randn(2) * 0.003
                new_centres = centres.copy()
                new_centres[idx] += perturbation
                new_centres = np.clip(new_centres, -0.5, 0.5)
                
                new_radius = _compute_radius(new_centres)
                if new_radius > best_radius + 1e-10:
                    best_radius = new_radius
                    best_centres = new_centres.copy()
                    improved = True
            
            if not improved and iteration > max_iterations // 3:
                break
    
    return best_centres, best_radius


def construct():
    """
    Construct centres of 200 equal circles inside the unit square [-1/2, 1/2]^2
    to maximize the common radius.
    """
    # Start with a hexagonal-like grid arrangement
    # Use a staggered grid for better initial packing
    m = int(np.ceil(np.sqrt(N)))
    
    # Create a staggered grid
    centres_list = []
    for row in range(m + 1):
        y = (row + 0.5) / (m + 1) - 0.5
        offset = 0.5 / (m + 1) if row % 2 == 0 else 0.0
        for col in range(m + 1):
            x = (col + 0.5) / (m + 1) - 0.5 + offset * (0.5 / (m + 1))
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                centres_list.append([x, y])
    
    centres = np.array(centres_list[:N])
    
    # If we don't have enough, pad with random points
    if len(centres) < N:
        random_points = np.random.rand(N - len(centres), 2) - 0.5
        centres = np.vstack([centres, random_points])
    
    best_centres = centres.copy()
    best_radius = _compute_radius(centres)
    
    # Run Delaunay-based optimization
    optimised, radius = _delaunay_optimize(centres, max_iterations=500, time_limit=50)
    
    if radius > best_radius:
        best_radius = radius
        best_centres = optimised.copy()
    
    # Try a few random restarts
    for restart in range(2):
        if time.time() > 100:
            break
        
        # Perturb the best solution
        perturbed = best_centres + np.random.randn(N, 2) * 0.004
        perturbed = np.clip(perturbed, -0.5, 0.5)
        
        optimised, radius = _delaunay_optimize(perturbed, max_iterations=300, time_limit=30)
        
        if radius > best_radius:
            best_radius = radius
            best_centres = optimised.copy()
    
    return best_centres
# EVOLVE-BLOCK-END
