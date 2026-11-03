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
    
    # Compute all pairwise distances
    dists = cdist(centres, centres)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Compute minimum distance to border
    min_dist_to_border = np.min(0.5 - np.abs(centres))
    
    return float(min(min_pair_dist / 2.0, min_dist_to_border))

def _delaunay_optimize(centres, max_time=10):
    """
    Optimize circle positions using Delaunay triangulation to identify
    closest pairs and push them apart.
    """
    start_time = time.time()
    best_centres = centres.copy()
    best_radius = _compute_radius(centres)
    step_size = 0.01
    no_improve_count = 0
    
    while time.time() - start_time < max_time:
        # Compute Delaunay triangulation
        try:
            tri = Delaunay(centres)
        except Exception:
            # If triangulation fails (e.g., duplicate points), break
            break
            
        # Get unique edges
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    edge = (min(simplex[i], simplex[j]), max(simplex[i], simplex[j]))
                    edges.add(edge)
        
        if not edges:
            break
            
        # Find the shortest edge in the Delaunay graph
        # The global minimum distance pair is always in the Delaunay triangulation
        min_dist = np.inf
        min_edge = None
        for u, v in edges:
            d = np.linalg.norm(centres[u] - centres[v])
            if d < min_dist:
                min_dist = d
                min_edge = (u, v)
        
        # Find min distance to border
        dist_to_border = 0.5 - np.abs(centres)
        min_border_dist = np.min(dist_to_border)
        min_border_idx, _ = np.unravel_index(np.argmin(dist_to_border), dist_to_border.shape)
        
        # Determine which constraint is active
        current_radius = min(min_dist / 2.0, min_border_dist)
        
        new_centres = centres.copy()
        improved = False
        
        if min_dist / 2.0 < min_border_dist:
            # Pair constraint is active
            u, v = min_edge
            direction = centres[v] - centres[u]
            norm = np.linalg.norm(direction)
            if norm > 1e-10:
                direction = direction / norm
            else:
                direction = np.random.randn(2)
                direction = direction / np.linalg.norm(direction)
            
            # Move u and v apart
            new_centres[u] -= direction * step_size
            new_centres[v] += direction * step_size
        else:
            # Border constraint is active
            idx = min_border_idx
            pos = centres[idx]
            # Determine direction to move away from the closest boundary
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
            
            new_centres[idx] += direction * step_size
        
        # Keep within bounds
        new_centres = np.clip(new_centres, -0.5, 0.5)
        
        new_radius = _compute_radius(new_centres)
        
        if new_radius > best_radius + 1e-12:
            best_radius = new_radius
            best_centres = new_centres.copy()
            centres = new_centres.copy()
            no_improve_count = 0
            # Slightly increase step size if we are making progress?
            # Or keep it constant. Let's keep it constant for stability.
        else:
            # No improvement, try to escape local minimum
            no_improve_count += 1
            if no_improve_count > 10:
                # Reduce step size
                step_size *= 0.9
                no_improve_count = 0
                if step_size < 1e-6:
                    break
            else:
                # Try a random perturbation of a random point
                idx = np.random.randint(N)
                perturbation = np.random.randn(2) * step_size
                new_centres = centres.copy()
                new_centres[idx] += perturbation
                new_centres = np.clip(new_centres, -0.5, 0.5)
                new_radius = _compute_radius(new_centres)
                if new_radius > best_radius + 1e-12:
                    best_radius = new_radius
                    best_centres = new_centres.copy()
                    centres = new_centres.copy()
                    no_improve_count = 0
                else:
                    # Revert perturbation, keep current centres
                    pass

    return best_centres, best_radius

def construct():
    """
    Construct centres of 200 equal circles inside the unit square [-1/2, 1/2]^2
    to maximize the common radius.
    """
    best_centres = None
    best_radius = -1.0
    
    # Multiple restarts with different initializations
    for restart in range(5):
        if restart == 0:
            # Grid initialization
            m = int(np.ceil(np.sqrt(N)))
            ticks = (np.arange(m) + 0.5) / m - 0.5
            grid = np.array([(x, y) for y in ticks for x in ticks])
            centres = grid[:N].copy()
        else:
            # Random initialization or perturbed grid
            if best_centres is not None:
                centres = best_centres + np.random.randn(N, 2) * 0.02
            else:
                centres = np.random.uniform(-0.45, 0.45, (N, 2))
            
            centres = np.clip(centres, -0.5, 0.5)
        
        # Run Delaunay optimization
        optimised, radius = _delaunay_optimize(centres, max_time=15)
        
        if radius > best_radius:
            best_radius = radius
            best_centres = optimised.copy()
    
    return best_centres
# EVOLVE-BLOCK-END
