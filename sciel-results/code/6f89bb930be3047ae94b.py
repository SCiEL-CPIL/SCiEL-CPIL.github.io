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
        min_dist_to_border = np.min(0.5 - np.abs(centres))
        return float(min_dist_to_border)
    
    dists = cdist(centres, centres)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
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


def _local_optimization(centres, max_iterations=500, step_size=0.01):
    """
    Local optimization using gradient-like steps to maximize minimum distance.
    """
    best_centres = centres.copy()
    best_radius = _compute_radius(centres)
    
    for iteration in range(max_iterations):
        improved = False
        
        dists = cdist(centres, centres)
        np.fill_diagonal(dists, np.inf)
        
        min_dist = np.min(dists)
        min_i, min_j = np.unravel_index(np.argmin(dists), dists.shape)
        
        dist_to_border = 0.5 - np.abs(centres)
        min_border_dist = np.min(dist_to_border)
        min_border_i, min_border_k = np.unravel_index(np.argmin(dist_to_border), dist_to_border.shape)
        
        if min_dist / 2.0 < min_border_dist:
            direction = centres[min_i] - centres[min_j]
            norm = np.linalg.norm(direction)
            if norm > 1e-10:
                direction = direction / norm
            else:
                direction = np.random.randn(2)
                direction = direction / np.linalg.norm(direction)
            
            step = step_size * (1.0 + 0.5 * np.random.rand())
            new_centres = centres.copy()
            new_centres[min_i] += direction * step
            new_centres[min_j] -= direction * step
            
            new_centres = np.clip(new_centres, -0.5, 0.5)
            
            new_radius = _compute_radius(new_centres)
            if new_radius > best_radius + 1e-10:
                best_radius = new_radius
                best_centres = new_centres.copy()
                improved = True
        else:
            pos = centres[min_border_i]
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
            step_size *= 0.9
            if step_size < 1e-6:
                break
    
    return best_centres, best_radius


def _delaunay_flip_optimization(centres, max_rounds=20, max_flips_per_round=50):
    """
    Delaunay-based combinatorial optimization: repeatedly compute the Delaunay 
    triangulation, identify short edges, and attempt to flip them or move vertices
    to increase the minimum edge length.
    """
    best_centres = centres.copy()
    best_radius = _compute_radius(centres)
    
    for round_idx in range(max_rounds):
        # Compute Delaunay triangulation
        try:
            tri = Delaunay(centres)
        except Exception:
            break
        
        edges = _delaunay_edges(tri)
        
        # Sort edges by length (shortest first)
        edge_lengths = []
        for i, j in edges:
            d = np.linalg.norm(centres[i] - centres[j])
            edge_lengths.append((d, i, j))
        
        edge_lengths.sort()
        
        flips_attempted = 0
        improved = False
        
        for d, i, j in edge_lengths:
            if flips_attempted >= max_flips_per_round:
                break
            
            # Try to move vertices i and j apart along the edge direction
            direction = centres[i] - centres[j]
            norm = np.linalg.norm(direction)
            if norm < 1e-10:
                continue
            direction = direction / norm
            
            # Small step to push them apart
            step = 0.002
            new_centres = best_centres.copy()
            new_centres[i] += direction * step
            new_centres[j] -= direction * step
            new_centres = np.clip(new_centres, -0.5, 0.5)
            
            new_radius = _compute_radius(new_centres)
            if new_radius > best_radius + 1e-12:
                best_radius = new_radius
                best_centres = new_centres.copy()
                improved = True
                flips_attempted += 1
                # Recompute triangulation after a significant improvement
                if flips_attempted % 5 == 0:
                    break
        
        if not improved:
            # Try random perturbations of poorly placed points
            for _ in range(10):
                idx = np.random.randint(len(centres))
                perturbation = np.random.randn(2) * 0.003
                new_centres = best_centres.copy()
                new_centres[idx] += perturbation
                new_centres = np.clip(new_centres, -0.5, 0.5)
                
                new_radius = _compute_radius(new_centres)
                if new_radius > best_radius + 1e-12:
                    best_radius = new_radius
                    best_centres = new_centres.copy()
                    improved = True
                    break
        
        if not improved and round_idx > max_rounds // 2:
            break
    
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
    
    # Phase 1: Delaunay-based combinatorial optimization
    delaunay_centres, delaunay_radius = _delaunay_flip_optimization(
        centres, max_rounds=15, max_flips_per_round=30
    )
    
    if delaunay_radius > best_radius:
        best_radius = delaunay_radius
        best_centres = delaunay_centres.copy()
    
    # Phase 2: Fine-tune with local optimization
    for restart in range(3):
        if restart == 0:
            current_centres = best_centres.copy()
        else:
            current_centres = best_centres + np.random.randn(N, 2) * 0.003
            current_centres = np.clip(current_centres, -0.5, 0.5)
        
        optimised, radius = _local_optimization(current_centres, max_iterations=200, step_size=0.005)
        
        if radius > best_radius:
            best_radius = radius
            best_centres = optimised.copy()
    
    # Phase 3: Another round of Delaunay optimization on the result
    delaunay_centres2, delaunay_radius2 = _delaunay_flip_optimization(
        best_centres, max_rounds=10, max_flips_per_round=20
    )
    
    if delaunay_radius2 > best_radius:
        best_radius = delaunay_radius2
        best_centres = delaunay_centres2.copy()
    
    return best_centres
# EVOLVE-BLOCK-END
