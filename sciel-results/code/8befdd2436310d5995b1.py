"""Improved program for the 200-circle packing task using Delaunay-based optimization."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Delaunay
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


def _get_adjacency(tri):
    """Get adjacency map: vertex -> list of adjacent vertices."""
    adj = {i: set() for i in range(len(tri.points))}
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                u, v = simplex[i], simplex[j]
                adj[u].add(v)
                adj[v].add(u)
    return adj


def _is_valid_flip(tri, centres, i, j, k, l):
    """
    Check if flipping edge (i,j) to (k,l) in quadrilateral i-k-j-l is valid.
    The flip is valid if the new edge (k,l) lies inside the convex hull
    and doesn't create self-intersections.
    """
    # Check if k and l are on opposite sides of edge (i,j)
    # and i and j are on opposite sides of edge (k,l)
    
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    # Check if segments (i,j) and (k,l) intersect properly
    d1 = cross(centres[k], centres[i], centres[j])
    d2 = cross(centres[l], centres[i], centres[j])
    d3 = cross(centres[i], centres[k], centres[l])
    d4 = cross(centres[j], centres[k], centres[l])
    
    # For a valid flip, the diagonals should cross
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False


def _flip_edge_and_optimize(centres, i, j, k, l, best_radius):
    """
    Try to flip the diagonal in quadrilateral formed by points i, k, j, l.
    Move points to maximize the minimum edge length.
    """
    new_centres = centres.copy()
    
    # The idea: after flipping, we want the new configuration to have
    # larger minimum edge length. We'll adjust positions slightly.
    
    # For now, just return the same centres - the flip changes topology,
    # and we'll rely on local optimization to adjust positions
    return new_centres


def _optimize_with_delaunay(centres, max_iterations=1000, time_limit=100):
    """
    Optimize circle positions using Delaunay triangulation structure.
    Focus on edges in the triangulation that are shortest.
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
        
        # Get edges and their lengths
        edges = _get_delaunay_edges(tri)
        if not edges:
            break
        
        # Find the shortest edge in the triangulation
        min_edge_len = np.inf
        min_edge = None
        for u, v in edges:
            d = np.linalg.norm(centres[u] - centres[v])
            if d < min_edge_len:
                min_edge_len = d
                min_edge = (u, v)
        
        if min_edge is None:
            break
        
        u, v = min_edge
        
        # Check if this edge is also the global minimum pair
        dists = cdist(centres, centres)
        np.fill_diagonal(dists, np.inf)
        global_min_dist = np.min(dists)
        
        # Also check border constraint
        min_border_dist = np.min(0.5 - np.abs(centres))
        
        if global_min_dist < 2 * min_border_dist:
            # Pair constraint is active - move u and v apart
            direction = centres[u] - centres[v]
            norm = np.linalg.norm(direction)
            if norm > 1e-10:
                direction = direction / norm
            else:
                direction = np.random.randn(2)
                direction = direction / np.linalg.norm(direction)
            
            # Try different step sizes
            improved = False
            for step_mult in [0.02, 0.01, 0.005, 0.002]:
                step = step_mult
                new_centres = centres.copy()
                new_centres[u] += direction * step
                new_centres[v] -= direction * step
                new_centres = np.clip(new_centres, -0.5, 0.5)
                
                new_radius = _compute_radius(new_centres)
                if new_radius > best_radius + 1e-12:
                    best_radius = new_radius
                    best_centres = new_centres.copy()
                    improved = True
                    break
            
            if not improved:
                # Try moving other nearby circles to make room
                adj = _get_adjacency(tri)
                for neighbor in adj.get(u, []):
                    if neighbor == v:
                        continue
                    dir_u_n = centres[u] - centres[neighbor]
                    norm = np.linalg.norm(dir_u_n)
                    if norm > 1e-10:
                        dir_u_n = dir_u_n / norm
                    else:
                        continue
                    
                    for step_mult in [0.01, 0.005]:
                        step = step_mult
                        new_centres = centres.copy()
                        new_centres[neighbor] -= dir_u_n * step
                        new_centres = np.clip(new_centres, -0.5, 0.5)
                        
                        new_radius = _compute_radius(new_centres)
                        if new_radius > best_radius + 1e-12:
                            best_radius = new_radius
                            best_centres = new_centres.copy()
                            improved = True
                            break
                    if improved:
                        break
                if not improved:
                    for neighbor in adj.get(v, []):
                        if neighbor == u:
                            continue
                        dir_v_n = centres[v] - centres[neighbor]
                        norm = np.linalg.norm(dir_v_n)
                        if norm > 1e-10:
                            dir_v_n = dir_v_n / norm
                        else:
                            continue
                        
                        for step_mult in [0.01, 0.005]:
                            step = step_mult
                            new_centres = centres.copy()
                            new_centres[neighbor] -= dir_v_n * step
                            new_centres = np.clip(new_centres, -0.5, 0.5)
                            
                            new_radius = _compute_radius(new_centres)
                            if new_radius > best_radius + 1e-12:
                                best_radius = new_radius
                                best_centres = new_centres.copy()
                                improved = True
                                break
                        if improved:
                            break
                
                if not improved:
                    # Small random perturbation
                    if np.random.rand() < 0.3:
                        idx = np.random.randint(len(centres))
                        new_centres = centres.copy()
                        new_centres[idx] += np.random.randn(2) * 0.003
                        new_centres = np.clip(new_centres, -0.5, 0.5)
                        new_radius = _compute_radius(new_centres)
                        if new_radius > best_radius + 1e-12:
                            best_radius = new_radius
                            best_centres = new_centres.copy()
                            improved = True
            
            if not improved:
                # Try to find a flip opportunity
                # Look for quadrilaterals containing the short edge
                for simplex in tri.simplices:
                    if u in simplex and v in simplex:
                        # Find the third vertex in this triangle
                        third = [x for x in simplex if x != u and x != v][0]
                        
                        # Find another triangle sharing edge (u,v)
                        for simplex2 in tri.simplices:
                            if simplex2.tolist() == simplex.tolist():
                                continue
                            if u in simplex2 and v in simplex2:
                                third2 = [x for x in simplex2 if x != u and x != v][0]
                                
                                # Check if flip is valid
                                if _is_valid_flip(tri, centres, u, v, third, third2):
                                    # Try the flip: move third and third2 slightly
                                    # to encourage the new configuration
                                    for step_mult in [0.005, 0.002]:
                                        new_centres = centres.copy()
                                        # Move third towards third2 slightly
                                        dir = centres[third2] - centres[third]
                                        norm = np.linalg.norm(dir)
                                        if norm > 1e-10:
                                            dir = dir / norm
                                            new_centres[third] += dir * step_mult
                                            new_centres[third2] -= dir * step_mult
                                            new_centres = np.clip(new_centres, -0.5, 0.5)
                                            
                                            new_radius = _compute_radius(new_centres)
                                            if new_radius > best_radius + 1e-12:
                                                best_radius = new_radius
                                                best_centres = new_centres.copy()
                                                improved = True
                                                break
                                    if improved:
                                        break
            if not improved:
                # No improvement, try a small global perturbation
                if iteration > 100:
                    idx = np.random.randint(len(centres))
                    new_centres = centres.copy()
                    new_centres[idx] += np.random.randn(2) * 0.002
                    new_centres = np.clip(new_centres, -0.5, 0.5)
                    new_radius = _compute_radius(new_centres)
                    if new_radius > best_radius + 1e-12:
                        best_radius = new_radius
                        best_centres = new_centres.copy()
                        improved = True
        
        else:
            # Border constraint is active
            # Find the circle closest to the border
            dist_to_border = 0.5 - np.abs(centres)
            min_border_idx, min_border_k = np.unravel_index(np.argmin(dist_to_border), dist_to_border.shape)
            
            # Move it away from the border
            pos = centres[min_border_idx]
            direction = np.zeros(2)
            if pos[0] > 0:
                direction[0] = -1.0
            elif pos[0] < 0:
                direction[0] = 1.0
            if pos[1] > 0:
                direction[1] = -1.0
            elif pos[1] < 0:
                direction[1] = 1.0
            
            norm = np.linalg.norm(direction)
            if norm > 1e-10:
                direction = direction / norm
            else:
                direction = np.random.randn(2)
                direction = direction / np.linalg.norm(direction)
            
            improved = False
            for step_mult in [0.01, 0.005, 0.002]:
                step = step_mult
                new_centres = centres.copy()
                new_centres[min_border_idx] += direction * step
                new_centres = np.clip(new_centres, -0.5, 0.5)
                
                new_radius = _compute_radius(new_centres)
                if new_radius > best_radius + 1e-12:
                    best_radius = new_radius
                    best_centres = new_centres.copy()
                    improved = True
                    break
            
            if not improved:
                # Perturb
                idx = np.random.randint(len(centres))
                new_centres = centres.copy()
                new_centres[idx] += np.random.randn(2) * 0.002
                new_centres = np.clip(new_centres, -0.5, 0.5)
                new_radius = _compute_radius(new_centres)
                if new_radius > best_radius + 1e-12:
                    best_radius = new_radius
                    best_centres = new_centres.copy()
        
        # Update centres for next iteration
        centres = best_centres
    
    return best_centres, best_radius


def construct():
    """
    Construct centres of 200 equal circles inside the unit square [-1/2, 1/2]^2
    to maximize the common radius.
    """
    # Start with a grid arrangement
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    centres = grid[:N].copy()
    
    best_centres = centres.copy()
    best_radius = _compute_radius(centres)
    
    # Run Delaunay-based optimization
    optimised, radius = _optimize_with_delaunay(centres, max_iterations=2000, time_limit=100)
    
    if radius > best_radius:
        best_radius = radius
        best_centres = optimised.copy()
    
    return best_centres
# EVOLVE-BLOCK-END
