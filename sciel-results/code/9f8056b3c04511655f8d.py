import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Voronoi, voronoi_plot_2d
import time

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
    Compute inradii of Voronoi cells for points in unit square.
    Returns array of inradii and incenters.
    """
    points = np.array(points)
    # Add boundary points to make Voronoi cells bounded
    # Corners and edge midpoints
    boundary_pts = np.array([
        [0, 0], [1, 0], [0, 1], [1, 1],
        [0.5, 0], [0.5, 1], [0, 0.5], [1, 0.5]
    ])
    
    # Extend points with boundary
    all_pts = np.vstack([points, boundary_pts])
    
    try:
        vor = Voronoi(all_pts)
    except:
        # Fallback: use simple distance-based radii
        dist_walls = np.minimum(points, 1 - points)
        min_wall_dist = np.min(dist_walls, axis=1)
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_neighbor_dist = np.min(dists, axis=1) / 2.0
        radii = np.minimum(min_wall_dist, min_neighbor_dist)
        return radii, points.copy()
    
    inradii = np.zeros(N)
    incenters = points.copy()
    
    for i in range(N):
        # Get vertices of cell i
        cell_vertices = vor.regions[vor.point_region[i]]
        if len(cell_vertices) == 0:
            # Unbounded or invalid cell
            inradii[i] = 0.0
            continue
        
        # Get the actual vertex coordinates
        verts = vor.vertices[cell_vertices]
        
        # Filter to only vertices inside [0,1]²
        inside = (verts[:, 0] >= -1e-10) & (verts[:, 0] <= 1 + 1e-10) & \
                 (verts[:, 1] >= -1e-10) & (verts[:, 1] <= 1 + 1e-10)
        verts = verts[inside]
        
        if len(verts) < 3:
            # Degenerate cell
            inradii[i] = 0.0
            continue
        
        # Find incenter: point that maximizes minimum distance to edges
        # For a polygon, this is the center of the largest inscribed circle
        # We can use the fact that incenter is equidistant from all edges
        
        # Compute edges
        edges = []
        for j in range(len(verts)):
            p1 = verts[j]
            p2 = verts[(j + 1) % len(verts)]
            edges.append((p1, p2))
        
        # For each edge, compute the line equation and distance
        # Distance from point (x,y) to line through p1,p2:
        # |(p2-p1) × (p-p1)| / |p2-p1|
        
        def min_dist_to_edges(x, y):
            min_d = np.inf
            for p1, p2 in edges:
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                # Line: (y - p1[1])/(x - p1[0]) = dy/dx if dx != 0
                # Standard form: dy*x - dx*y + (dx*p1[1] - dy*p1[0]) = 0
                a = dy
                b = -dx
                c = dx * p1[1] - dy * p1[0]
                norm = np.sqrt(a**2 + b**2)
                if norm < 1e-15:
                    continue
                d = abs(a * x + b * y + c) / norm
                # Check if projection is on the segment
                t = ((x - p1[0]) * dx + (y - p1[1]) * dy) / (dx**2 + dy**2)
                if 0 <= t <= 1:
                    min_d = min(min_d, d)
                else:
                    # Distance to endpoints
                    d1 = np.sqrt((x - p1[0])**2 + (y - p1[1])**2)
                    d2 = np.sqrt((x - p2[0])**2 + (y - p2[1])**2)
                    min_d = min(min_d, d1, d2)
            return min_d
        
        # Use optimization to find incenter
        # Start from centroid
        centroid = np.mean(verts, axis=0)
        
        # Bounded optimization
        bounds = [(0.0, 1.0), (0.0, 1.0)]
        
        def neg_min_dist(pos):
            return -min_dist_to_edges(pos[0], pos[1])
        
        try:
            res = minimize(neg_min_dist, centroid, method='Nelder-Mead',
                           options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-10})
            incenter = res.x
            inradius = -res.fun
        except:
            incenter = centroid
            inradius = min_dist_to_edges(centroid[0], centroid[1])
        
        # Ensure incenter is inside the cell
        inradii[i] = max(0, inradius)
        incenters[i] = incenter
    
    return inradii, incenters

def voronoi_relax(points, n_iter=50, time_limit=30.0):
    """
    Relax points toward their Voronoi cell incenters.
    """
    points = points.reshape(N, 2).copy()
    start_time = time.time()
    
    for it in range(n_iter):
        if time.time() - start_time > time_limit:
            break
        
        inradii, incenters = voronoi_inradii(points)
        
        # Move points toward incenters with damping
        alpha = 0.5 / (1 + it * 0.1)
        new_points = points + alpha * (incenters - points)
        
        # Clip to unit square
        new_points = np.clip(new_points, 0.001, 0.999)
        
        # Check convergence
        if np.max(np.abs(new_points - points)) < 1e-10:
            points = new_points
            break
        
        points = new_points
    
    return points.flatten()

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

def phyllotaxis_points(n, radius=0.5, center=(0.5, 0.5)):
    points = []
    for i in range(n):
        r = radius * np.sqrt(i / n)
        theta = i * 137.508 * np.pi / 180.0
        x = center[0] + r * np.cos(theta)
        y = center[1] + r * np.sin(theta)
        x = np.clip(x, 0.01, 0.99)
        y = np.clip(y, 0.01, 0.99)
        points.append([x, y])
    return np.array(points)

def construct():
    start_time = time.time()
    timeout = 115
    
    best_c = None
    best_val = -np.inf
    
    inits = []
    
    # Grids
    for r, c in [(6,6), (5,7), (7,5), (4,8), (8,4), (6,5), (5,6)]:
        inits.append(create_grid_init(r, c, N))
    
    # Halton
    inits.append(halton_sequence(N).flatten())
    
    # Phyllotaxis
    inits.append(phyllotaxis_points(N).flatten())
    
    # Jittered Grids
    np.random.seed(42)
    base = create_grid_init(6, 6, N).reshape(N, 2)
    for jitter_scale in [0.02, 0.04, 0.06]:
        for _ in range(5):
            jitter = np.random.uniform(-jitter_scale, jitter_scale, size=(N, 2))
            p = np.clip(base + jitter, 0.01, 0.99)
            inits.append(p.flatten())
    
    # Random
    for _ in range(10):
        p = np.random.uniform(0.05, 0.95, size=(N, 2))
        inits.append(p.flatten())
    
    bounds = [(0.001, 0.999), (0.001, 0.999)] * N
    
    for idx, c_init in enumerate(inits):
        if time.time() - start_time > timeout - 25:
            break
        
        # 1. Voronoi relaxation
        try:
            relaxed = voronoi_relax(c_init, n_iter=30, time_limit=15.0)
            val = sum_radii(relaxed)
            if val > best_val:
                best_val = val
                best_c = relaxed.copy()
        except:
            pass
        
        # 2. Local optimization on relaxed result
        if time.time() - start_time < timeout - 10:
            try:
                start_point = best_c if best_c is not None and idx > 0 else c_init
                res = minimize(lambda c: -sum_radii(c), start_point,
                               method='Nelder-Mead',
                               options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12})
                val = -res.fun
                if val > best_val:
                    best_val = val
                    best_c = res.x.copy()
            except:
                pass
        
        if time.time() - start_time > timeout - 15:
            break

    if best_c is None:
        best_c = create_grid_init(6, 6, N)
    
    # Final refinement
    if time.time() - start_time < timeout - 5:
        try:
            refined = voronoi_relax(best_c, n_iter=20, time_limit=10.0)
            val = sum_radii(refined)
            if val > best_val:
                best_val = val
                best_c = refined.copy()
        except:
            pass
    
    if time.time() - start_time < timeout - 2:
        try:
            res = minimize(lambda c: -sum_radii(c), best_c,
                           method='Nelder-Mead',
                           options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-12})
            val = -res.fun
            if val > best_val:
                best_val = val
                best_c = res.x.copy()
        except:
            pass
    
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
