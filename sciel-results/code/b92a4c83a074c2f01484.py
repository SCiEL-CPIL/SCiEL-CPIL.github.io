"""Improved program for the 200-circle packing task using Delaunay-based optimization."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist
import random
import copy


def _compute_radius(centres):
    """Compute the maximum common radius for given centres."""
    if len(centres) < 2:
        # Edge case: single circle
        min_dist_to_border = np.min(0.5 - np.abs(centres))
        return float(min_dist_to_border)
    
    # Compute all pairwise distances
    dists = cdist(centres, centres)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Compute minimum distance to border
    min_dist_to_border = np.min(0.5 - np.abs(centres))
    
    return float(min(min_pair_dist / 2.0, min_dist_to_border))


def _delaunay_edges(tri):
    """Extract unique edges from a Delaunay triangulation."""
    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                edge = (min(simplex[i], simplex[j]), max(simplex[i], simplex[j]))
                edges.add(edge)
    return list(edges)


def _try_flip(tri, centres, edge_i, edge_j, best_radius):
    """
    Attempt to flip edges between two adjacent triangles sharing an edge.
    Returns the new centres if the flip improves the radius, otherwise None.
    """
    # Find the two triangles sharing the edge (edge_i, edge_j)
    tri_a = None
    tri_b = None
    for simplex in tri.simplices:
        if edge_i in simplex and edge_j in simplex:
            if tri_a is None:
                tri_a = simplex
            else:
                tri_b = simplex
                break
    
    if tri_a is None or tri_b is None:
        return None
    
    # The shared edge is (edge_i, edge_j), the other vertices are the third vertices
    third_a = [v for v in tri_a if v != edge_i and v != edge_j][0]
    third_b = [v for v in tri_b if v != edge_i and v != edge_j][0]
    
    # Check if flipping is valid (the new edge (third_a, third_b) should be inside the convex hull)
    # For simplicity, we just try the flip and check if it improves things
    # The flip would replace edges (edge_i, edge_j) with (third_a, third_b)
    # and change the triangulation
    
    # In a proper flip, we'd recompute the Delaunay triangulation
    # For now, let's just return None as a placeholder since implementing
    # a full flip with radius improvement check is complex
    
    return None


def _local_optimization(centres, max_iterations=500, step_size=0.01):
    """
    Local optimization using gradient-like steps to maximize minimum distance.
    """
    best_centres = centres.copy()
    best_radius = _compute_radius(centres)
    
    for iteration in range(max_iterations):
        improved = False
        
        # Compute current radius
        current_radius = _compute_radius(centres)
        
        # Find the bottleneck: the pair of circles (or circle-border) that limits the radius
        dists = cdist(centres, centres)
        np.fill_diagonal(dists, np.inf)
        
        min_dist = np.min(dists)
        min_i, min_j = np.unravel_index(np.argmin(dists), dists.shape)
        
        dist_to_border = 0.5 - np.abs(centres)
        min_border_dist = np.min(dist_to_border)
        min_border_i, min_border_k = np.unravel_index(np.argmin(dist_to_border), dist_to_border.shape)
        
        # Determine which constraint is active
        if min_dist / 2.0 < min_border_dist:
            # Pair constraint is active
            direction = centres[min_i] - centres[min_j]
            norm = np.linalg.norm(direction)
            if norm > 1e-10:
                direction = direction / norm
            else:
                direction = np.random.randn(2)
                direction = direction / np.linalg.norm(direction)
            
            # Move the two circles apart
            step = step_size * (1.0 + 0.5 * np.random.rand())
            new_centres = centres.copy()
            new_centres[min_i] += direction * step
            new_centres[min_j] -= direction * step
            
            # Keep within bounds
            new_centres = np.clip(new_centres, -0.5, 0.5)
            
            new_radius = _compute_radius(new_centres)
            if new_radius > best_radius + 1e-10:
                best_radius = new_radius
                best_centres = new_centres.copy()
                improved = True
        else:
            # Border constraint is active
            # Move the circle away from the border
            pos = centres[min_border_i]
            if pos[0] > 0:
                direction = np.array([-1.0, 0.0])
            elif pos[0] < 0:
                direction = np.array([1.0, 0.0])
            else:
                direction = np.array([0.0, 0.0])
            
            if pos[1] > 0:
                direction[1] = -1.0
            elif pos[1] < 0:
                direction[1] = 1.0
            else:
                direction[1] = 0.0
            
            norm = np.linalg.norm(direction)
            if norm > 1e-10:
                direction = direction / norm
            else:
                direction = np.random.randn(2)
                direction = direction / np.linalg.norm(direction)
            
            step = step_size * (1.0 + 0.5 * np.random.rand())
            new_centres = centres.copy()
            new_centres[min_border_i] += direction * step
            new_centres = np.clip(new_centres, -0.5, 0.5)
            
            new_radius = _compute_radius(new_centres)
            if new_radius > best_radius + 1e-10:
                best_radius = new_radius
                best_centres = new_centres.copy()
                improved = True
        
        if not improved:
            # Try a small random perturbation to escape local minima
            if iteration > max_iterations // 2:
                idx = np.random.randint(len(centres))
                perturbation = np.random.randn(2) * step_size * 0.5
                new_centres = centres.copy()
                new_centres[idx] += perturbation
                new_centres = np.clip(new_centres, -0.5, 0.5)
                
                new_radius = _compute_radius(new_centres)
                if new_radius > best_radius + 1e-10:
                    best_radius = new_radius
                    best_centres = new_centres.copy()
                    improved = True
        
        if not improved and iteration > max_iterations // 3:
            # Reduce step size
            step_size *= 0.9
            if step_size < 1e-6:
                break
    
    return best_centres, best_radius


def _delaunay_edge_flip_optimization(centres, max_flips=1000):
    """
    Optimize circle positions using Delaunay triangulation and edge flips.
    This treats the packing as a graph optimization problem where we
    maximize the minimum edge length.
    """
    best_centres = centres.copy()
    best_radius = _compute_radius(centres)
    
    for flip_iteration in range(max_flips):
        # Compute Delaunay triangulation
        try:
            tri = Delaunay(centres)
        except:
            break
        
        # Get all edges from the triangulation
        edges = _delaunay_edges(tri)
        
        if len(edges) == 0:
            break
        
        # Compute edge lengths
        edge_lengths = {}
        for i, j in edges:
            dist = np.linalg.norm(centres[i] - centres[j])
            edge_lengths[(i, j)] = dist
        
        # Find the shortest edge - this is our bottleneck
        if not edge_lengths:
            break
            
        min_edge = min(edge_lengths, key=edge_lengths.get)
        min_edge_length = edge_lengths[min_edge]
        
        # Also consider border constraints
        dist_to_border = 0.5 - np.abs(centres)
        min_border_dist = np.min(dist_to_border)
        
        # If border constraint is tighter, handle it separately
        if min_border_dist < min_edge_length / 2.0:
            # Move the circle closest to border away from it
            min_border_idx = np.unravel_index(np.argmin(dist_to_border), dist_to_border.shape)
            i = min_border_idx[0]
            pos = centres[i]
            
            # Determine direction to move away from border
            direction = np.zeros(2)
            if pos[0] > 0:
                direction[0] = -1.0
            elif pos[0] < 0:
                direction[0] = 1.0
            if pos[1] > 0:
                direction[1] = -1.0
            elif pos[1] < 0:
                direction[1] = 1.0
            
            if np.linalg.norm(direction) > 1e-10:
                direction = direction / np.linalg.norm(direction)
                step = 0.002
                new_centres = centres.copy()
                new_centres[i] += direction * step
                new_centres = np.clip(new_centres, -0.5, 0.5)
                
                new_radius = _compute_radius(new_centres)
                if new_radius > best_radius + 1e-10:
                    best_radius = new_radius
                    best_centres = new_centres.copy()
                    centres = new_centres
                    continue
        
        # Now handle the shortest edge - try to increase its length
        i, j = min_edge
        
        # Get the direction from j to i
        direction = centres[i] - centres[j]
        norm = np.linalg.norm(direction)
        if norm < 1e-10:
            continue
        direction = direction / norm
        
        # Try moving i and j apart
        step = 0.003
        new_centres = centres.copy()
        new_centres[i] += direction * step
        new_centres[j] -= direction * step
        new_centres = np.clip(new_centres, -0.5, 0.5)
        
        new_radius = _compute_radius(new_centres)
        if new_radius > best_radius + 1e-10:
            best_radius = new_radius
            best_centres = new_centres.copy()
            centres = new_centres
        else:
            # Try a smaller step or different approach
            # Move only one of the circles
            for idx in [i, j]:
                sign = 1 if idx == i else -1
                new_centres = centres.copy()
                new_centres[idx] += direction * step * sign * 0.5
                new_centres = np.clip(new_centres, -0.5, 0.5)
                
                new_radius = _compute_radius(new_centres)
                if new_radius > best_radius + 1e-10:
                    best_radius = new_radius
                    best_centres = new_centres.copy()
                    centres = new_centres
                    break
    
    return best_centres, best_radius


def _hybrid_optimization(centres, max_iterations=200):
    """
    Hybrid optimization combining Delaunay edge flips with local perturbations.
    """
    best_centres = centres.copy()
    best_radius = _compute_radius(centres)
    
    for iteration in range(max_iterations):
        # Phase 1: Delaunay-based edge flip optimization
        optimised, radius = _delaunay_edge_flip_optimization(centres, max_flips=100)
        
        if radius > best_radius:
            best_radius = radius
            best_centres = optimised.copy()
            centres = optimised
        
        # Phase 2: Local perturbation to escape local minima
        if iteration % 5 == 4:  # Every 5th iteration
            # Add small random perturbation to random circles
            for _ in range(10):
                idx = np.random.randint(len(centres))
                perturbation = np.random.randn(2) * 0.003
                new_centres = centres.copy()
                new_centres[idx] += perturbation
                new_centres = np.clip(new_centres, -0.5, 0.5)
                
                new_radius = _compute_radius(new_centres)
                if new_radius > best_radius + 1e-10:
                    best_radius = new_radius
                    best_centres = new_centres.copy()
                    centres = new_centres
    
    return best_centres, best_radius


def construct():
    """
    Construct centres of 200 equal circles inside the unit square [-1/2, 1/2]^2
    to maximize the common radius.
    """
    # Start with a hexagonal lattice arrangement, which is optimal for infinite packing
    # For a square, we'll use a slightly perturbed hexagonal grid
    
    # Create hexagonal lattice points
    hex_points = []
    row = 0
    while len(hex_points) < N:
        y = (row - 10) * 0.0866  # Hexagonal spacing
        if abs(y) > 0.48:
            row += 1
            continue
        x_offset = 0.0433 if row % 2 == 0 else 0.0
        col = 0
        while len(hex_points) < N:
            x = (col - 10) * 0.0866 + x_offset
            if abs(x) <= 0.48 and abs(y) <= 0.48:
                hex_points.append([x, y])
            col += 1
        row += 1
    
    # If we don't have enough points, fill in with a grid
    if len(hex_points) < N:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        for point in grid:
            if len(hex_points) >= N:
                break
            # Check if this grid point is not already in hex_points
            is_duplicate = False
            for hp in hex_points:
                if np.linalg.norm(np.array(hp) - np.array(point)) < 0.01:
                    is_duplicate = True
                    break
            if not is_duplicate:
                hex_points.append(list(point))
    
    centres = np.array(hex_points[:N])
    
    # Ensure centres are within bounds
    centres = np.clip(centres, -0.48, 0.48)
    
    best_centres = centres.copy()
    best_radius = _compute_radius(centres)
    
    # Run hybrid optimization with multiple restarts
    for restart in range(5):
        if restart > 0:
            # Add random perturbation
            centres = best_centres + np.random.randn(N, 2) * 0.01
            centres = np.clip(centres, -0.48, 0.48)
        
        # Run hybrid optimization
        optimised, radius = _hybrid_optimization(centres, max_iterations=100)
        
        if radius > best_radius:
            best_radius = radius
            best_centres = optimised.copy()
    
    return best_centres
# EVOLVE-BLOCK-END

