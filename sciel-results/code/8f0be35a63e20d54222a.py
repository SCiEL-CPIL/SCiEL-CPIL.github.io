import numpy as np
from scipy.optimize import minimize
import time
from scipy.spatial import Voronoi

N = 32

def sum_radii(c):
    p = c.reshape(N, 2)
    dist_walls = np.minimum(p, 1 - p)
    min_wall_dist = np.min(dist_walls, axis=1)
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_neighbor_dist = np.min(dists, axis=1) / 2.0
    radii = np.minimum(min_wall_dist, min_neighbor_dist)
    return np.sum(radii)

def voronoi_inradii(points):
    """
    Given N points in unit square, compute the inradius of each point's Voronoi cell.
    The inradius is the maximum radius of a circle centered at the point that fits 
    within its Voronoi cell (bounded by the unit square).
    """
    n = len(points)
    radii = np.zeros(n)
    
    # Create augmented points: original points + boundary points far away
    # We'll use a different approach: for each point, find the minimum distance
    # to any Voronoi edge (which is a perpendicular bisector between it and a neighbor)
    # or to the boundary of the unit square.
    
    for i in range(n):
        # Distance to walls
        wall_dist = np.min([points[i][0], 1 - points[i][0], 
                            points[i][1], 1 - points[i][1]])
        
        # Distance to other points' Voronoi boundaries
        # The Voronoi boundary between i and j is the perpendicular bisector
        # of the segment connecting i and j.
        # The distance from point i to this bisector is half the distance to j.
        min_bisector_dist = np.inf
        for j in range(n):
            if i == j:
                continue
            d = np.linalg.norm(points[i] - points[j])
            # Distance from i to bisector is d/2
            bisector_dist = d / 2.0
            if bisector_dist < min_bisector_dist:
                min_bisector_dist = bisector_dist
        
        # The inradius is the minimum of wall distance and bisector distance
        radii[i] = min(wall_dist, min_bisector_dist)
    
    return radii

def voronoi_objective(points_flat):
    """Negative sum of inradii for optimization"""
    points = points_flat.reshape(N, 2)
    # Clip to valid range
    points = np.clip(points, 0.001, 0.999)
    radii = voronoi_inradii(points)
    return -np.sum(radii)

def voronoi_gradient(points_flat):
    """
    Compute gradient of -sum(inradii) with respect to point positions.
    
    For each point i, the inradius r_i = min(wall_dist_i, min_j d(i,j)/2)
    
    If wall_dist_i < min_j d(i,j)/2:
        r_i = wall_dist_i
        dr_i/dx_i depends on which wall is closest
    elif min_j d(i,j)/2 < wall_dist_i:
        r_i = d(i,j*)/2 where j* is nearest neighbor
        dr_i/dx_i = (x_i - x_j*) / (2 * d(i,j*))
        dr_j/dx_i = -(x_i - x_j*) / (2 * d(i,j*))  (affects r_j through d(j,i))
    else:
        Average of both
    """
    points = points_flat.reshape(N, 2).copy()
    points = np.clip(points, 0.001, 0.999)
    grad = np.zeros_like(points)
    
    for i in range(N):
        pi = points[i]
        
        # Compute wall distances
        wall_dists = [pi[0], 1 - pi[0], pi[1], 1 - pi[1]]
        min_wall = np.min(wall_dists)
        
        # Find nearest neighbor
        dists = np.linalg.norm(points - pi, axis=1)
        dists[i] = np.inf
        j = np.argmin(dists)
        d_nearest = dists[j]
        min_bisector = d_nearest / 2.0
        
        eps = 1e-10
        
        if min_wall < min_bisector - eps:
            # Wall-constrained
            # Determine which wall(s) are active
            active_grads = []
            for w, g in zip(wall_dists, [[1,0], [-1,0], [0,1], [0,-1]]):
                if abs(w - min_wall) < eps:
                    active_grads.append(g)
            if active_grads:
                avg_grad = np.mean(active_grads, axis=0)
                grad[i] += avg_grad
        elif min_bisector < min_wall - eps:
            # Neighbor-constrained
            if d_nearest > eps:
                vec = pi - points[j]
                # dr_i/dpi = vec / (2 * d)
                # dr_j/dpi = -vec / (2 * d)  (because d(j,i) = d(i,j))
                grad[i] += 0.5 * vec / d_nearest
                grad[j] += -0.5 * vec / d_nearest
        else:
            # Both active
            # Wall gradient
            active_grads = []
            for w, g in zip(wall_dists, [[1,0], [-1,0], [0,1], [0,-1]]):
                if abs(w - min_wall) < eps:
                    active_grads.append(g)
            wall_grad = np.mean(active_grads, axis=0) if active_grads else np.zeros(2)
            
            # Neighbor gradient
            if d_nearest > eps:
                vec = pi - points[j]
                neigh_grad_i = 0.5 * vec / d_nearest
                neigh_grad_j = -0.5 * vec / d_nearest
            else:
                neigh_grad_i = np.zeros(2)
                neigh_grad_j = np.zeros(2)
            
            grad[i] += 0.5 * (wall_grad + neigh_grad_i)
            grad[j] += 0.5 * neigh_grad_j
    
    return grad.flatten()

def phyllotaxis_points(n, radius=0.5, center=(0.5, 0.5)):
    """Generate points using phyllotaxis arrangement (sunflower pattern)"""
    points = []
    for i in range(n):
        r = radius * np.sqrt(i / n)
        theta = i * 137.508 * np.pi / 180.0 # Golden angle in radians
        x = center[0] + r * np.cos(theta)
        y = center[1] + r * np.sin(theta)
        # Clip to unit square
        x = np.clip(x, 0.01, 0.99)
        y = np.clip(y, 0.01, 0.99)
        points.append([x, y])
    return np.array(points)

def create_grid_init(rows, cols, n):
    xs = (np.arange(cols) + 0.5) / cols
    ys = (np.arange(rows) + 0.5) / rows
    points = []
    for i in range(rows):
        for j in range(cols):
            if len(points) < n:
                points.append([xs[j], ys[i]])
    return np.array(points[:n]).flatten()

def halton_sequence(n, dims=2):
    def halton(index, base):
        result = 0.0
        f = 1.0
        i = index
        while i > 0:
            f /= base
            result += f * (i % base)
            i //= base
        return result
    points = []
    for i in range(1, n + 1):
        points.append([halton(i, 2), halton(i, 3)])
    return np.array(points)

def voronoi_refine(c, max_sweeps=10, time_limit=10.0):
    """
    Refine using Voronoi-based coordinate descent: for each point, 
    optimize its position to maximize its Voronoi inradius.
    """
    p = c.reshape(N, 2).copy()
    start_time = time.time()
    
    for sweep in range(max_sweeps):
        if time.time() - start_time > time_limit:
            break
        improved = False
        for i in range(N):
            p_i = p[i].copy()
            
            def neg_inradius_i(pos):
                pos = np.array(pos)
                pos = np.clip(pos, 0.001, 0.999)
                # Wall distance
                wall_dist = np.min([pos[0], 1-pos[0], pos[1], 1-pos[1]])
                # Bisector distance to nearest other point
                dists = np.linalg.norm(p - pos, axis=1)
                dists[i] = np.inf
                min_bisector = np.min(dists) / 2.0
                return -min(wall_dist, min_bisector)
            
            old_val = neg_inradius_i(p_i)
            try:
                res = minimize(neg_inradius_i, p_i, method='Nelder-Mead',
                               options={'maxiter': 50, 'xatol': 1e-7, 'fatol': 1e-9})
                new_val = res.fun
                if new_val < old_val - 1e-10:
                    p[i] = np.clip(res.x, 0.001, 0.999)
                    improved = True
            except:
                pass
        
        if not improved:
            break
            
    return p.flatten()

def construct():
    start_time = time.time()
    timeout = 115
    
    best_c = None
    best_val = -np.inf
    
    inits = []
    
    # 1. Grids
    for r, c in [(6,6), (5,7), (7,5), (4,8), (8,4), (6,5), (5,6)]:
        inits.append(create_grid_init(r, c, N))
        
    # 2. Halton
    inits.append(halton_sequence(N).flatten())
    
    # 3. Phyllotaxis
    inits.append(phyllotaxis_points(N).flatten())
    
    # 4. Jittered Grids
    np.random.seed(42)
    base = create_grid_init(6, 6, N).reshape(N, 2)
    for jitter_scale in [0.02, 0.04, 0.06]:
        for _ in range(5):
            jitter = np.random.uniform(-jitter_scale, jitter_scale, size=(N, 2))
            p = np.clip(base + jitter, 0.01, 0.99)
            inits.append(p.flatten())
            
    # 5. Random
    for _ in range(10):
        p = np.random.uniform(0.05, 0.95, size=(N, 2))
        inits.append(p.flatten())
        
    bounds = [(0.001, 0.999), (0.001, 0.999)] * N
    
    for idx, c_init in enumerate(inits):
        if time.time() - start_time > timeout - 20:
            break
            
        # 1. L-BFGS-B with Voronoi objective
        try:
            res = minimize(voronoi_objective, c_init, jac=voronoi_gradient,
                           method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12})
            val = -res.fun
            if val > best_val:
                best_val = val
                best_c = res.x.copy()
        except:
            pass
        
        # 2. Voronoi-based coordinate descent refinement
        if time.time() - start_time < timeout - 10:
            try:
                start_refine = res.x if 'res' in locals() and res.x is not None else c_init
                refined = voronoi_refine(start_refine, max_sweeps=5, time_limit=5.0)
                val = -voronoi_objective(refined)
                if val > best_val:
                    best_val = val
                    best_c = refined.copy()
            except:
                pass
        
        if time.time() - start_time > timeout - 15:
            break

    if best_c is not None:
        # Final Polish with L-BFGS-B
        if time.time() - start_time < timeout - 5:
            try:
                res = minimize(voronoi_objective, best_c, jac=voronoi_gradient,
                               method='L-BFGS-B', bounds=bounds,
                               options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12})
                val = -res.fun
                if val > best_val:
                    best_val = val
                    best_c = res.x.copy()
            except:
                pass
        
        # Final Voronoi refinement
        if time.time() - start_time < timeout - 2:
            try:
                refined = voronoi_refine(best_c, max_sweeps=5, time_limit=3.0)
                val = -voronoi_objective(refined)
                if val > best_val:
                    best_val = val
                    best_c = refined.copy()
            except:
                pass

    if best_c is None:
        best_c = create_grid_init(6, 6, N)
        
    # Compute final radii using the standard method (minimum of wall and neighbor distances)
    p = best_c.reshape(N, 2)
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_neighbor_dist = np.min(dists, axis=1) / 2.0
    dist_walls = np.minimum(p, 1 - p)
    min_wall_dist = np.min(dist_walls, axis=1)
    radii = np.minimum(min_wall_dist, min_neighbor_dist)
    
    result = np.column_stack((p, radii))
    return result
