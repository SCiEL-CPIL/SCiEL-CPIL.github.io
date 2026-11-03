"""Improved program using Delaunay-based sparse optimization with topology refinement."""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay
import time

N = 50

def _calculate_true_radius(centers):
    """Calculate the true maximum radius for a given set of centers."""
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    wall_dists = 0.5 - np.abs(centers)
    min_wall = np.min(wall_dists)
    return min(min_pair, min_wall)

def _get_delaunay_edges(centers):
    """Get unique edges from Delaunay triangulation."""
    try:
        tri = Delaunay(centers)
        simplices = tri.simplices
        edges = set()
        for simplex in simplices:
            for i in range(3):
                for j in range(i+1, 3):
                    a, b = simplex[i], simplex[j]
                    if a > b:
                        a, b = b, a
                    edges.add((a, b))
        return list(edges)
    except Exception:
        # Fallback to all pairs if Delaunay fails
        return [(i, j) for i in range(N) for j in range(i+1, N)]

def _optimize_delaunay(centers, bounds, max_iters=50):
    """
    Optimizes centers using Delaunay-aware objective.
    Re-evaluates Delaunay edges periodically to handle topology changes.
    """
    best_centers = centers.copy()
    best_score = _calculate_true_radius(best_centers)
    
    def objective(params, edges):
        c = params.reshape(N, 2)
        c = np.clip(c, -0.5, 0.5)
        
        # Distances for active edges
        edge_dists = []
        for i, j in edges:
            d = np.linalg.norm(c[i] - c[j])
            edge_dists.append(d / 2.0)
            
        # Wall distances
        wall_dists = 0.5 - np.abs(c)
        edge_dists.extend(wall_dists.flatten())
        
        if not edge_dists:
            return 0.0
            
        edge_dists = np.array(edge_dists)
        m = np.min(edge_dists)
        
        # Soft min (log-sum-exp) for differentiability
        k = 500.0
        shifted = edge_dists - m
        # Prevent underflow/overflow
        shifted = np.clip(shifted, 0, 50)
        exp_terms = np.exp(-k * shifted)
        log_sum = np.log(np.sum(exp_terms))
        soft_min = m - (1.0 / k) * log_sum
        return -soft_min

    current_centers = best_centers.copy()
    edges = _get_delaunay_edges(current_centers)
    
    for iter in range(max_iters):
        try:
            result = minimize(
                objective, 
                current_centers.reshape(-1), 
                args=(edges,),
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-12, 'disp': False}
            )
            new_centers = result.x.reshape(N, 2)
            new_centers = np.clip(new_centers, -0.5, 0.5)
            
            new_score = _calculate_true_radius(new_centers)
            
            if new_score > best_score:
                best_score = new_score
                best_centers = new_centers
                current_centers = new_centers
            else:
                # If no improvement, try a smaller step or break
                if new_score >= best_score - 1e-9:
                    current_centers = new_centers
                else:
                    break
            
            # Update Delaunay edges every few iterations to catch topology changes
            if iter % 5 == 4:
                new_edges = _get_delaunay_edges(current_centers)
                if len(new_edges) != len(edges):
                    edges = new_edges
        except Exception:
            break
            
    return best_centers, best_score

def construct():
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    # High-quality initializations
    inits = []
    
    # 1. Hexagonal Grid (Best for dense packing)
    # We want ~50 points in a unit square.
    # Hex grid with spacing s. Area per point ~ s^2 * sqrt(3)/2.
    # 50 * s^2 * 0.866 ~ 1 => s ~ sqrt(1/43) ~ 0.15
    s = 0.145
    points = []
    # Generate a large grid and clip
    x_range = np.arange(-0.5, 0.5 + s, s)
    y_range = np.arange(-0.5, 0.5 + s, s)
    for i, x in enumerate(x_range):
        offset = (s * np.sqrt(3) / 2) if i % 2 == 0 else 0
        # Shift y to center
        for y in y_range:
            points.append((x, y + offset - s*np.sqrt(3)/4))
    points = np.array(points)
    mask = (np.abs(points[:, 0]) <= 0.49) & (np.abs(points[:, 1]) <= 0.49)
    points = points[mask]
    if len(points) >= N:
        np.random.seed(42)
        indices = np.random.choice(len(points), N, replace=False)
        inits.append(points[indices])
        
    # 2. Square Grid
    grid_x = np.linspace(-0.45, 0.45, 7)
    grid_y = np.linspace(-0.45, 0.45, 7)
    grid_points = np.array([[x, y] for x in grid_x for y in grid_y])
    if len(grid_points) >= N:
        np.random.seed(123)
        indices = np.random.choice(len(grid_points), N, replace=False)
        inits.append(grid_points[indices])
        
    # 3. Jittered Hex (to break symmetry)
    if len(points) >= N:
        np.random.seed(43)
        indices = np.random.choice(len(points), N, replace=False)
        jittered = points[indices] + np.random.normal(0, 0.01, (N, 2))
        jittered = np.clip(jittered, -0.49, 0.49)
        inits.append(jittered)
        
    # 4. Random
    np.random.seed(999)
    inits.append(np.random.uniform(-0.45, 0.45, (N, 2)))
    
    # 5. Another Jittered Hex
    if len(points) >= N:
        np.random.seed(44)
        indices = np.random.choice(len(points), N, replace=False)
        jittered = points[indices] + np.random.normal(0, 0.03, (N, 2))
        jittered = np.clip(jittered, -0.49, 0.49)
        inits.append(jittered)

    best_centers = None
    best_score = -1.0
    
    start_time = time.time()
    
    for i, init in enumerate(inits):
        if time.time() - start_time > 100.0: 
            break
            
        centers, score = _optimize_delaunay(init, bounds, max_iters=30)
        if score > best_score:
            best_score = score
            best_centers = centers
            
    if best_centers is None:
        # Fallback
        best_centers = np.random.uniform(-0.45, 0.45, (N, 2))
        
    best_centers = np.clip(best_centers, -0.4999, 0.4999)
    
    return best_centers
