"""Improved program for the 200-circle packing task using Delaunay-based optimization."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist
import random
import copy
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


def _local_optimization_delaunay(centres, max_iterations=1000, initial_step=0.01):
    """
    Local optimization using Delaunay triangulation to identify neighbors
    and apply repulsive forces to increase the minimum distance.
    """
    best_centres = centres.copy()
    best_radius = _compute_radius(centres)
    step = initial_step
    
    for iteration in range(max_iterations):
        if time.time() > 110: # Safety break
            break
            
        # 1. Compute Delaunay Triangulation
        try:
            tri = Delaunay(centres)
        except Exception:
            # If triangulation fails (e.g., collinear points), perturb slightly
            centres += np.random.randn(N, 2) * 1e-5
            continue
            
        # 2. Identify close neighbors from Delaunay edges
        # We want to repel points that are closer than the current best radius * 2
        # Actually, we want to increase the *minimum* distance.
        # So we focus on edges that are shorter than a threshold.
        # Let's use the current minimum distance as a reference, or a slightly larger threshold.
        
        dists = cdist(centres, centres)
        np.fill_diagonal(dists, np.inf)
        current_min_dist = np.min(dists)
        
        # Threshold: any pair closer than 1.1 * current_min_dist is considered "stressed"
        # But we only look at Delaunay neighbors to keep it sparse.
        threshold = current_min_dist * 1.2 
        
        # Build adjacency list of "stressed" neighbors
        # We iterate over Delaunay simplices
        stressed_pairs = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    idx_i = simplex[i]
                    idx_j = simplex[j]
                    if dists[idx_i, idx_j] < threshold:
                        stressed_pairs.add((idx_i, idx_j))
        
        # 3. Calculate forces
        forces = np.zeros_like(centres)
        
        for i, j in stressed_pairs:
            diff = centres[i] - centres[j]
            dist = np.linalg.norm(diff)
            if dist > 1e-10:
                # Repulsive force: stronger if closer
                force_mag = (threshold - dist) 
                force_vec = diff / dist * force_mag
                forces[i] += force_vec
                forces[j] -= force_vec
        
        # 4. Border forces
        # Push points away from borders if they are too close
        # Target distance from border: let's say we want to maintain at least current_min_dist/2 from border?
        # Or just push if < some small epsilon?
        # Let's push if distance to border < current_min_dist / 2
        border_threshold = current_min_dist / 2.0
        for i in range(N):
            # x boundaries
            if centres[i][0] < -0.5 + border_threshold:
                forces[i][0] += (border_threshold - (centres[i][0] + 0.5))
            elif centres[i][0] > 0.5 - border_threshold:
                forces[i][0] -= (border_threshold - (0.5 - centres[i][0]))
            
            # y boundaries
            if centres[i][1] < -0.5 + border_threshold:
                forces[i][1] += (border_threshold - (centres[i][1] + 0.5))
            elif centres[i][1] > 0.5 - border_threshold:
                forces[i][1] -= (border_threshold - (0.5 - centres[i][1]))
        
        # 5. Update positions
        new_centres = centres + forces * step
        new_centres = np.clip(new_centres, -0.5, 0.5)
        
        # 6. Check improvement
        new_radius = _compute_radius(new_centres)
        
        if new_radius > best_radius + 1e-12:
            best_radius = new_radius
            best_centres = new_centres.copy()
            centres = new_centres.copy()
            # Decay step size slowly
            step *= 0.99
        else:
            # No improvement, reduce step size more aggressively
            step *= 0.9
            # If step is very small, break
            if step < 1e-7:
                break
            
            # Occasionally add noise to escape local minima
            if iteration % 50 == 0:
                noise_idx = np.random.randint(N)
                new_centres = centres.copy()
                new_centres[noise_idx] += np.random.randn(2) * step * 2
                new_centres = np.clip(new_centres, -0.5, 0.5)
                noise_radius = _compute_radius(new_centres)
                if noise_radius > best_radius:
                    best_radius = noise_radius
                    best_centres = new_centres.copy()
                    centres = new_centres.copy()
    
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
    
    # Run multiple iterations of local optimization with different random restarts
    # We use the Delaunay-based optimizer which is more robust
    for restart in range(5):
        # Add small random perturbation
        if restart > 0:
            centres = best_centres + np.random.randn(N, 2) * 0.01
            centres = np.clip(centres, -0.5, 0.5)
        
        # Local optimization
        optimised, radius = _local_optimization_delaunay(centres, max_iterations=500, initial_step=0.01)
        
        if radius > best_radius:
            best_radius = radius
            best_centres = optimised.copy()
    
    return best_centres
# EVOLVE-BLOCK-END
