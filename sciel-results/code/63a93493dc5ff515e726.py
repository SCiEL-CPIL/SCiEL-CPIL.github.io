"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import Voronoi
from scipy.optimize import minimize
import random

def _compute_radius(c):
    """Compute the common radius for centres c (N,3) in cube [-0.5,0.5]^3."""
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    # Wall distances
    wall_dists = 0.5 - np.abs(c)
    min_wall = np.min(wall_dists)
    return min(min_pair / 2.0, min_wall)

def _initial_configuration():
    """Create a good initial configuration using a BCC-like arrangement."""
    # Start with a 4x4x4 grid (64 points) and pick 50
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    points = []
    for z in ticks:
        for y in ticks:
            for x in ticks:
                points.append([x, y, z])
    
    # Add some BCC-like points to improve packing
    # Shift some points slightly to create BCC structure
    random.seed(42)
    random.shuffle(points)
    return np.array(points[:N])

def _soft_min(values, alpha=50.0):
    """Compute a smooth approximation of the minimum using log-sum-exp."""
    # We want to maximize the min, so we minimize the -min.
    # -min(x) = logsumexp(-alpha * x) / alpha
    v = -alpha * values
    v_max = np.max(v)
    if v_max > 700: return -np.min(values) # Overflow safety
    return np.log(np.sum(np.exp(v - v_max))) / alpha + v_max / alpha

def _smooth_objective(c, alpha=50.0):
    """
    Objective to minimize: -soft_min(constraints).
    Constraints are: pairwise distances / 2 and wall distances.
    """
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    pair_constr = dists / 2.0
    
    # Wall distances
    wall_constr = 0.5 - np.abs(c)
    
    # Combine all constraints
    # We take the soft min of all pair distances and all wall distances
    # To avoid huge arrays, we can just compute the min of the two soft mins? 
    # No, the min of the union is the soft min of the union.
    # But the array of pair distances is N^2. 
    # Let's just compute the exact min for the objective if alpha is high enough?
    # No, the point is to be smooth.
    
    # Efficient soft min of the union of sets A and B is roughly min(soft_min(A), soft_min(B))
    # But strictly it's the soft min of the combined vector.
    # For N=50, 1250 pairs + 150 walls = 1400 values. This is small.
    
    all_constr = np.concatenate([pair_constr.flatten(), wall_constr.flatten()])
    return -_soft_min(all_constr, alpha)

def _lloyd_iteration(c, max_iter=50, tol=1e-8):
    """
    Lloyd's algorithm variant: Move each center to the centroid of its Voronoi cell
    clipped by the cube boundary.
    """
    c = c.copy()
    for _ in range(max_iter):
        # Add boundary points to help Voronoi computation clip correctly
        # We need points outside the domain to ensure cells are bounded
        # Standard trick: add points at +/- large distance
        boundary_points = []
        # 8 corners extended
        for x in [-1.0, 1.0]:
            for y in [-1.0, 1.0]:
                for z in [-1.0, 1.0]:
                    boundary_points.append([x, y, z])
        # Add points on faces extended
        for x in [-1.0, 1.0]:
            for y in [-0.5, 0.5]:
                for z in [-0.5, 0.5]:
                    boundary_points.append([x, y, z])
        for x in [-0.5, 0.5]:
            for y in [-1.0, 1.0]:
                for z in [-0.5, 0.5]:
                    boundary_points.append([x, y, z])
        for x in [-0.5, 0.5]:
            for y in [-0.5, 0.5]:
                for z in [-1.0, 1.0]:
                    boundary_points.append([x, y, z])
        
        points = np.vstack([c, np.array(boundary_points)])
        
        try:
            vor = Voronoi(points)
        except Exception:
            break
            
        new_c = c.copy()
        max_move = 0.0
        
        for i in range(N):
            # Get region for point i
            region_index = vor.point_region[i]
            region = vor.regions[region_index]
            
            if -1 in region or len(region) == 0:
                continue
                
            vertices = vor.vertices[region]
            
            # Clip vertices to the cube [-0.5, 0.5]^3
            # A vertex is valid if it's inside the cube
            # However, Voronoi vertices might be far away.
            # We need the centroid of the polygon (convex hull of vertices) clipped to the cube.
            # For simplicity and robustness, we approximate the centroid by averaging
            # the vertices that are inside the cube.
            # This is a heuristic approximation of the true clipped centroid.
            
            inside_mask = (vertices[:, 0] >= -0.5) & (vertices[:, 0] <= 0.5) & \
                          (vertices[:, 1] >= -0.5) & (vertices[:, 1] <= 0.5) & \
                          (vertices[:, 2] >= -0.5) & (vertices[:, 2] <= 0.5)
            
            if np.sum(inside_mask) == 0:
                continue
                
            valid_vertices = vertices[inside_mask]
            
            # Simple average of valid vertices
            centroid = np.mean(valid_vertices, axis=0)
            
            # Clip centroid to be strictly inside to avoid boundary issues
            centroid = np.clip(centroid, -0.5 + 1e-9, 0.5 - 1e-9)
            
            move = np.linalg.norm(centroid - c[i])
            max_move = max(max_move, move)
            new_c[i] = centroid
            
        if max_move < tol:
            break
        c = new_c
        
    return c

def _polish(c, time_budget=20.0):
    """Polish the configuration using Nelder-Mead on the smooth objective."""
    start_time = time.time()
    best_c = c.copy()
    best_r = _compute_radius(best_c)
    
    # Use a few restarts with small perturbations
    for _ in range(5):
        if time.time() - start_time > time_budget:
            break
            
        c_try = c.copy()
        # Small random perturbation
        c_try += np.random.normal(0, 1e-4, c_try.shape)
        c_try = np.clip(c_try, -0.5 + 1e-9, 0.5 - 1e-9)
        
        # Nelder-Mead optimization
        try:
            res = minimize(_smooth_objective, c_try, method='Nelder-Mead', 
                           options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 500, 'maxfev': 500})
            c_opt = res.x
            c_opt = np.clip(c_opt, -0.5 + 1e-9, 0.5 - 1e-9)
            
            r_opt = _compute_radius(c_opt)
            if r_opt > best_r:
                best_c = c_opt
                best_r = r_opt
        except Exception:
            continue
            
    return best_c, best_r

def construct():
    """Construct the optimal packing of 50 spheres in a unit cube."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Initial configuration
    c = _initial_configuration()
    
    # Lloyd's relaxation
    if time.time() < deadline - 30.0:
        c = _lloyd_iteration(c, max_iter=100)
    
    # Polish with Nelder-Mead
    if time.time() < deadline - 10.0:
        c, best_r = _polish(c, time_budget=min(20.0, deadline - time.time() - 5.0))
        
    # Final check
    c = np.clip(c, -0.5 + 1e-12, 0.5 - 1e-12)
    
    return c
# EVOLVE-BLOCK-END
