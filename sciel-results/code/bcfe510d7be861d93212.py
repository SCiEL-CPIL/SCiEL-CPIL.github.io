# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from scipy.optimize import minimize
import time
import itertools

def _precompute_indices(n):
    """Precompute triangle indices for efficiency."""
    return np.array(list(itertools.combinations(range(n), 3)))

def _compute_dets(points, idx):
    """Compute 2 * signed area (determinant) for all triangles."""
    p1 = points[idx[:, 0]]
    p2 = points[idx[:, 1]]
    p3 = points[idx[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    det = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return det

def _compute_hull_area(points):
    """Compute convex hull area safely."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def compute_score(points):
    """
    Compute the Heilbronn score: A_min / A_hull.
    Returns 0 if invalid.
    """
    n = len(points)
    if n < 3:
        return 0.0
    
    if not np.all(np.isfinite(points)):
        return 0.0
        
    A_hull = _compute_hull_area(points)
    if A_hull <= 1e-12:
        return 0.0
        
    idx = _precompute_indices(n)
    dets = _compute_dets(points, idx)
    A_min = 0.5 * np.min(np.abs(dets))
    
    return A_min / A_hull

def _delaunay_repair(points, max_iters=1000, step_size=0.1):
    """
    Structural repair strategy:
    1. Compute Delaunay triangulation.
    2. Identify triangles with small area.
    3. Move vertices of these triangles towards the centroid of their adjacent 
       Delaunay triangles (or hull center if on boundary) to "inflate" them.
    4. Repeat.
    """
    n = len(points)
    current = points.copy()
    
    # Precompute adjacency for Delaunay triangles
    # Delaunay returns indices of points for each triangle
    # We need to know which triangles are adjacent to which.
    
    for it in range(max_iters):
        try:
            tri = Delaunay(current)
        except Exception:
            break
            
        # Get triangle areas
        # tri.simplices has shape (M, 3)
        simplices = tri.simplices
        M = len(simplices)
        
        # Calculate areas of Delaunay triangles
        p1 = current[simplices[:, 0]]
        p2 = current[simplices[:, 1]]
        p3 = current[simplices[:, 2]]
        
        v1 = p2 - p1
        v2 = p3 - p1
        dets = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(dets)
        
        # Find the minimum area triangle
        min_idx = np.argmin(areas)
        min_area = areas[min_idx]
        
        # If the minimum area is very large relative to others, we might be converged
        # Or if it's very small, we need repair.
        
        # Identify the vertices of the smallest triangle
        tri_verts = simplices[min_idx]
        
        # Find adjacent triangles to this one
        # Two triangles are adjacent if they share an edge
        # We can check all pairs, but M is small (typically < 20 for n=13)
        
        # Let's build a simple adjacency map
        # Edge defined by sorted pair of vertex indices
        # For each triangle, get its 3 edges
        edges = []
        for i in range(M):
            v = simplices[i]
            e1 = tuple(sorted((v[0], v[1])))
            e2 = tuple(sorted((v[1], v[2])))
            e3 = tuple(sorted((v[0], v[2])))
            edges.append([e1, e2, e3])
        
        # Find neighbors of min_idx
        neighbors = []
        min_edges = edges[min_idx]
        for j in range(M):
            if j == min_idx: continue
            if any(e in edges[j] for e in min_edges):
                neighbors.append(j)
        
        # Calculate the target movement for each vertex of the smallest triangle
        # Strategy: Move vertex towards the centroid of the adjacent triangles' centroids
        # This pushes the "pinched" vertex outwards into the space of its neighbors
        
        # Centroid of the smallest triangle
        c_min = np.mean(current[tri_verts], axis=0)
        
        # For each vertex in the smallest triangle, find the "outward" direction
        # or the direction towards the centroid of the neighbors that share that vertex?
        # Actually, a simpler heuristic:
        # If a vertex is shared with a neighbor, moving it towards the neighbor's centroid
        # might increase the area of the small triangle if the neighbor is "larger" or "fuller".
        
        # Let's try: For each vertex v in tri_verts:
        #   Find all neighbor triangles that contain v.
        #   Compute the average centroid of those neighbors.
        #   Move v towards that average centroid.
        
        new_points = current.copy()
        
        for v_idx in tri_verts:
            # Find neighbors sharing v_idx
            neighbor_centroids = []
            for n_idx in neighbors:
                if v_idx in simplices[n_idx]:
                    n_c = np.mean(current[simplices[n_idx]], axis=0)
                    neighbor_centroids.append(n_c)
            
            if neighbor_centroids:
                target = np.mean(neighbor_centroids, axis=0)
            else:
                # On the hull, move towards the center of the hull
                # Approximate hull center
                target = np.mean(current, axis=0)
            
            # Move a small fraction of the way
            # Use a step size that is adaptive?
            # Fixed small step for stability
            new_points[v_idx] += step_size * (target - current[v_idx])
            
        # Check if improvement
        # Recalculate min area? That's expensive.
        # Just check if the specific triangle area increased?
        # Let's just update and rely on the loop to find the new min.
        
        # Convergence check: if the move is tiny
        if np.linalg.norm(new_points - current) < 1e-6:
            break
            
        current = new_points
        
    return current

def heilbronn_convex13() -> np.ndarray:
    n = 13
    start_time = time.time()
    deadline = start_time + 590  # Safety margin
    
    best_points = None
    best_score = 0.0
    
    idx = _precompute_indices(n)
    
    # Initializations
    inits = []
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    inits.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # 2. Concentric polygons (1 center, 5 inner, 7 outer)
    center = np.array([[0, 0]])
    angles_inner = np.linspace(0, 2*np.pi, 5, endpoint=False)
    angles_outer = np.linspace(0, 2*np.pi, 7, endpoint=False) + np.pi/7
    inner_pts = np.column_stack([0.5 * np.cos(angles_inner), 0.5 * np.sin(angles_inner)])
    outer_pts = np.column_stack([1.0 * np.cos(angles_outer), 1.0 * np.sin(angles_outer)])
    inits.append(np.vstack([center, inner_pts, outer_pts]))
    
    # 3. Random
    rng = np.random.default_rng(seed=42)
    inits.append(rng.random((n, 2)) * 2 - 1)
    
    # 4. Grid
    grid_x = np.linspace(-1, 1, 4)
    grid_y = np.linspace(-1, 1, 4)
    gx, gy = np.meshgrid(grid_x, grid_y)
    pts = np.column_stack([gx.ravel(), gy.ravel()])[:13]
    pts += rng.random((13, 2)) * 0.05
    inits.append(pts)
    
    # 5. Hexagonal lattice snippet
    hex_pts = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            x = i + 0.5 * (j % 2)
            y = j * np.sqrt(3) / 2
            hex_pts.append([x, y])
    hex_pts = np.array(hex_pts)
    hex_pts = (hex_pts - hex_pts.mean(axis=0)) / (np.abs(hex_pts).max() + 1e-9)
    hex_pts = hex_pts[:13]
    inits.append(hex_pts)
    
    # 6. Fibonacci spiral
    angles_fib = np.arange(n) * 2 * np.pi * 0.6180339887
    radii_fib = np.sqrt(np.arange(1, n + 1) / n)
    inits.append(np.column_stack([radii_fib * np.cos(angles_fib), radii_fib * np.sin(angles_fib)]))

    for i_init, init_pts in enumerate(inits):
        if time.time() > deadline:
            break
            
        # Phase 1: Delaunay-based structural repair
        # This is the NEW strategy. It directly manipulates geometry based on local topology.
        repaired = _delaunay_repair(init_pts, max_iters=2000, step_size=0.05)
        
        # Check score
        score = compute_score(repaired)
        if score > best_score:
            best_score = score
            best_points = repaired.copy()
            
        # Phase 2: Local Nelder-Mead polish on the repaired points
        # Nelder-Mead is robust to the non-smoothness of the min-area function.
        def nm_objective(x):
            pts = x.reshape((n, 2))
            s = compute_score(pts)
            if s == 0.0:
                return 1.0
            return -s
        
        try:
            res_nm = minimize(nm_objective, repaired.flatten(), method='Nelder-Mead',
                              options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-5, 'adaptive': True})
            final_pts_nm = res_nm.x.reshape((n, 2))
            final_score_nm = compute_score(final_pts_nm)
            
            if final_score_nm > best_score:
                best_score = final_score_nm
                best_points = final_pts_nm.copy()
                
        except Exception:
            pass

    # Fallback
    if best_points is None or best_score < 0.01:
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        best_score = compute_score(best_points)

    return best_points

# EVOLVE-BLOCK-END
