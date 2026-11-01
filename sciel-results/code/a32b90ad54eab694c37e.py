import numpy as np
from scipy.spatial import ConvexHull, Delaunay
import time
import warnings

N = 130

def _covering_radius(points):
    """
    Compute the exact covering radius from the convex hull.
    For each facet (triangle) of the convex hull, compute the circumradius
    of the spherical triangle. The covering radius is the maximum.
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi
    
    max_radius = 0.0
    n = len(points)
    
    for simplex in hull.simplices:
        tri = points[simplex]
        
        # Side lengths (angular distances)
        a = np.arccos(np.clip(np.dot(tri[1], tri[2]), -1, 1))
        b = np.arccos(np.clip(np.dot(tri[0], tri[2]), -1, 1))
        c = np.arccos(np.clip(np.dot(tri[0], tri[1]), -1, 1))
        
        # Spherical circumradius using the formula:
        # For a spherical triangle with sides a, b, c, the circumradius R satisfies:
        # cos(R) = (cos(a) + cos(b) + cos(c) + cos(a+b+c)) / (1 + cos(a) + cos(b) + cos(c) + cos(a+b+c)) ... 
        # Actually, let's use the normal-based approach.
        
        v1 = tri[1] - tri[0]
        v2 = tri[2] - tri[0]
        normal = np.cross(v1, v2)
        norm_len = np.linalg.norm(normal)
        
        if norm_len < 1e-12:
            continue
        
        normal /= norm_len
        
        # The circumcenter is along the normal direction (outward for convex hull)
        # Compute angular distance from circumcenter to each vertex
        d0 = np.arccos(np.clip(np.dot(normal, tri[0]), -1, 1))
        d1 = np.arccos(np.clip(np.dot(normal, tri[1]), -1, 1))
        d2 = np.arccos(np.clip(np.dot(normal, tri[2]), -1, 1))
        
        # The circumradius is the angular distance from circumcenter to vertices
        # For a valid circumcenter, all three should be equal
        r = max(d0, d1, d2)
        
        # Also check the other direction (inward)
        d0n = np.arccos(np.clip(np.dot(-normal, tri[0]), -1, 1))
        d1n = np.arccos(np.clip(np.dot(-normal, tri[1]), -1, 1))
        d2n = np.arccos(np.clip(np.dot(-normal, tri[2]), -1, 1))
        rn = max(d0n, d1n, d2n)
        
        # The covering radius for this facet is the smaller of the two (the cap)
        # Actually, for the convex hull, the outward normal gives the cap center
        # The covering radius contribution is the distance from cap center to farthest vertex
        # But we want the maximum over all caps
        facet_radius = min(r, rn) if min(r, rn) < np.pi/2 else max(r, rn)
        
        # Actually, the correct approach: the circumcenter of the spherical triangle
        # is the point equidistant from all three vertices. For a convex hull facet,
        # the relevant circumcenter is the one on the same side as the outward normal.
        # The covering radius is the angular distance from this circumcenter to each vertex.
        
        # Let me reconsider: for the covering radius, we need the largest empty cap.
        # Each convex hull facet defines a cap. The radius of that cap is the angular
        # distance from the cap's center (along the outward normal) to the facet vertices.
        
        # The cap center is the normalized outward normal
        # The radius is the angular distance from cap center to any vertex
        # (all three should be equal for a proper circumcenter)
        
        # Use the outward normal direction
        cap_center = normal
        dists = np.array([d0, d1, d2])
        
        # If the cap is very large (> 90 degrees), the center might be on the other side
        if np.max(dists) > np.pi / 2:
            cap_center = -normal
            dists = np.array([d0n, d1n, d2n])
        
        cap_radius = np.max(dists)
        
        if cap_radius > max_radius:
            max_radius = cap_radius
    
    return max_radius

def _lloyd_iteration(points):
    """
    One Lloyd iteration: for each point, compute the centroid of its Voronoi cell
    on the sphere and move the point there.
    
    Uses Delaunay triangulation to identify Voronoi cells.
    """
    try:
        # Create Delaunay triangulation
        tri = Delaunay(points)
    except Exception:
        return points
    
    n = len(points)
    new_points = np.zeros_like(points)
    
    # For each point, find all triangles that contain it
    # The Voronoi cell of point i consists of all points on the sphere
    # closer to i than to any other point.
    
    # A practical approach: for each point, find its neighboring points
    # (connected via Delaunay edges) and compute the centroid of the
    # region bounded by the perpendicular bisectors.
    
    # Simpler approach using the convex hull:
    # The Voronoi cell of point i on the sphere can be approximated by
    # the set of points whose nearest neighbor is i.
    
    # Efficient method: use the fact that in a Delaunay triangulation,
    # the Voronoi cell of vertex i is the polygon formed by the circumcenters
    # of all triangles containing i.
    
    # Let's use a grid-based approach for the centroid computation
    # but make it efficient.
    
    # Actually, let's use the following efficient method:
    # For each point, the new position is the normalized sum of all points
    # in its Voronoi cell. We can approximate this by:
    # 1. Finding all Delaunay neighbors of point i
    # 2. Computing the centroid of the spherical polygon formed by the
    #    circumcenters of triangles containing i
    
    # Let's build a neighbor list from the Delaunay triangulation
    neighbor_list = [[] for _ in range(n)]
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(3):
                if i != j:
                    neighbor_list[simplex[i]].append(simplex[j])
    
    # For each point, compute the centroid of its Voronoi cell
    # using the circumcenters of adjacent triangles
    
    # Build triangle-to-vertex mapping
    tri_vertices = tri.simplices  # (n_tris, 3)
    n_tris = len(tri_vertices)
    
    # For each triangle, compute its circumcenter
    circumcenters = np.zeros((n_tris, 3))
    for t in range(n_tris):
        tri_pts = points[tri_vertices[t]]
        v1 = tri_pts[1] - tri_pts[0]
        v2 = tri_pts[2] - tri_pts[0]
        normal = np.cross(v1, v2)
        norm_len = np.linalg.norm(normal)
        if norm_len > 1e-12:
            normal /= norm_len
            # Project to sphere
            circumcenters[t] = normal
        else:
            circumcenters[t] = points[tri_vertices[t][0]]
    
    # For each vertex, find all triangles containing it
    vertex_tris = [[] for _ in range(n)]
    for t in range(n_tris):
        for v in tri_vertices[t]:
            vertex_tris[v].append(t)
    
    # Compute new positions
    for i in range(n):
        if not vertex_tris[i]:
            new_points[i] = points[i]
            continue
        
        # The Voronoi cell of point i is bounded by arcs between circumcenters
        # of adjacent triangles. The centroid of this cell can be approximated
        # by the average of the circumcenters of adjacent triangles, weighted
        # by the angle at vertex i in each triangle.
        
        # Simple approximation: average of circumcenters of adjacent triangles
        cc_indices = vertex_tris[i]
        cc_sum = np.sum(circumcenters[cc_indices], axis=0)
        
        # Also include the point itself with some weight
        # The centroid should be pulled toward the point
        cc_sum += points[i] * 0.5
        
        norm = np.linalg.norm(cc_sum)
        if norm > 1e-12:
            new_points[i] = cc_sum / norm
        else:
            new_points[i] = points[i]
    
    return new_points

def _lloyd_optimize(points, max_iter=200, tol=1e-10):
    """
    Run Lloyd's algorithm for spherical k-means covering.
    """
    current = points.copy()
    
    for it in range(max_iter):
        new_points = _lloyd_iteration(current)
        
        # Check convergence
        diff = np.linalg.norm(new_points - current)
        current = new_points
        
        if diff < tol:
            break
    
    return current

def _repulsion_optimize(points, time_limit=5.0):
    """
    Optimize using repulsive forces to spread points evenly.
    Each point repels others with a force that increases with proximity.
    """
    start_time = time.time()
    current = points.copy()
    n = len(current)
    
    # Learning rate schedule
    lr = 0.01
    
    for it in range(500):
        if time.time() - start_time > time_limit:
            break
        
        # Compute forces
        forces = np.zeros_like(current)
        
        # Pairwise repulsion
        # For efficiency, compute in batches
        for i in range(n):
            # Distance to all other points
            diffs = current - current[i]  # (n, 3)
            dists = np.linalg.norm(diffs, axis=1)  # (n,)
            
            # Avoid self-interaction
            dists[i] = np.inf
            
            # Repulsive force: F = k / d^2 * direction
            # But we want to maximize the minimum distance, so use a soft repulsion
            # F_i = sum_j (current[i] - current[j]) / |current[i] - current[j]|^2
            # Only for points that are too close
            
            # Use a threshold: only repel if distance < threshold
            threshold = 0.5  # ~28.6 degrees
            mask = dists < threshold
            
            if np.any(mask):
                # Force from each too-close point
                for j in np.where(mask)[0]:
                    d = dists[j]
                    if d > 1e-10:
                        # Repulsive force proportional to 1/d^2
                        force = (current[i] - current[j]) / (d ** 2)
                        forces[i] += force
        
        # Project force onto tangent plane
        for i in range(n):
            # Tangent space: remove radial component
            radial = np.dot(forces[i], current[i])
            forces[i] -= radial * current[i]
        
        # Update
        current = current + lr * forces
        current /= np.linalg.norm(current, axis=1, keepdims=True)
        
        # Decay learning rate
        lr *= 0.995
    
    return current

def _gradient_descent_covering(points, time_limit=10.0):
    """
    Gradient descent on the covering radius using a dense grid.
    """
    start_time = time.time()
    current = points.copy()
    n = len(current)
    
    # Create a dense grid of test points on the sphere
    n_grid = 20000
    i_grid = np.arange(n_grid) + 0.5
    z_grid = 1.0 - 2.0 * i_grid / n_grid
    phi_grid = np.pi * (1.0 + np.sqrt(5.0)) * i_grid
    r_grid = np.sqrt(np.clip(1.0 - z_grid * z_grid, 0, 1))
    grid_points = np.column_stack([r_grid * np.cos(phi_grid), 
                                    r_grid * np.sin(phi_grid), 
                                    z_grid])
    
    lr = 0.005
    
    for it in range(200):
        if time.time() - start_time > time_limit:
            break
        
        # Compute covering radius and gradient
        # For each grid point, find nearest configuration point
        dot = grid_points @ current.T  # (n_grid, n)
        dot = np.clip(dot, -1.0, 1.0)
        dists = np.arccos(dot)  # (n_grid, n)
        
        # Nearest point for each grid point
        min_dists = np.min(dists, axis=1)  # (n_grid,)
        nearest = np.argmin(dists, axis=1)  # (n_grid,)
        
        # Covering radius is the max of min_dists
        # Gradient: for each configuration point i, the gradient is the sum
        # of contributions from grid points for which i is the nearest
        
        forces = np.zeros((n, 3))
        for k in range(n_grid):
            i_nearest = nearest[k]
            d = min_dists[k]
            
            if d > 0.1:  # Only contribute if not too close
                # The grid point wants to be closer to i_nearest
                # Force on i_nearest: pull it toward the grid point
                # Direction: from i_nearest to grid_point, projected to tangent
                diff = grid_points[k] - current[i_nearest]
                # Project to tangent plane at current[i_nearest]
                radial = np.dot(diff, current[i_nearest])
                tangent = diff - radial * current[i_nearest]
                forces[i_nearest] += tangent / (d + 1e-6)
        
        # Update
        current = current + lr * forces
        current /= np.linalg.norm(current, axis=1, keepdims=True)
        
        # Decay learning rate
        lr *= 0.99
    
    return current

def _fibonacci_spiral(n):
    """Generate points on a Fibonacci spiral on the sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0, 1))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_score = np.inf
    best_points = None
    
    # Generate multiple starting configurations
    rng = np.random.default_rng(12345)
    
    starts = []
    
    # 1. Fibonacci spiral
    starts.append(_fibonacci_spiral(N))
    
    # 2. Several random starts
    for seed in range(5):
        rng_local = np.random.default_rng(seed * 1000 + 42)
        rp = rng_local.normal(size=(N, 3))
        rp /= np.linalg.norm(rp, axis=1, keepdims=True)
        starts.append(rp)
    
    # 3. Icosahedron-based
    t = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [-1,  t,  0], [ 1,  t,  0], [-1, -t,  0], [ 1, -t,  0],
        [ 0, -1,  t], [ 0,  1,  t], [ 0, -1, -t], [ 0,  1, -t],
        [ t,  0, -1], [ t,  0,  1], [-t,  0, -1], [-t,  0,  1]
    ])
    ico_verts /= np.linalg.norm(ico_verts, axis=1, keepdims=True)
    ico_expanded = []
    rng_ico = np.random.default_rng(777)
    for v in ico_verts:
        for _ in range(11):
            jittered = v + rng_ico.normal(scale=0.15, size=3)
            jittered /= np.linalg.norm(jittered)
            ico_expanded.append(jittered)
    starts.append(np.array(ico_expanded[:N]))
    
    # Evaluate each start with Lloyd's algorithm
    for idx, start_pts in enumerate(starts):
        elapsed = time.time() - start_time
        remaining = deadline - time.time()
        
        if remaining < 5.0:
            break
        
        # Evaluate initial covering radius
        score = _covering_radius(start_pts)
        if score < best_score:
            best_score = score
            best_points = start_pts.copy()
        
        # Run Lloyd's algorithm
        lloyd_time = min(20.0, max(5.0, remaining * 0.4))
        lloyd_points = _lloyd_optimize(start_pts, max_iter=100)
        
        score = _covering_radius(lloyd_points)
        if score < best_score:
            best_score = score
            best_points = lloyd_points.copy()
        
        # Further refine with gradient descent
        remaining = deadline - time.time()
        if remaining > 5.0:
            gd_time = min(10.0, remaining * 0.3)
            gd_points = _gradient_descent_covering(lloyd_points, time_limit=gd_time)
            
            score = _covering_radius(gd_points)
            if score < best_score:
                best_score = score
                best_points = gd_points.copy()
        
        # Final Lloyd polish
        remaining = deadline - time.time()
        if remaining > 3.0:
            polish_time = min(5.0, remaining * 0.5)
            # Quick Lloyd refinement
            polish_points = _lloyd_optimize(best_points, max_iter=50)
            score = _covering_radius(polish_points)
            if score < best_score:
                best_score = score
                best_points = polish_points.copy()
    
    # Fallback
    if best_points is None:
        best_points = _fibonacci_spiral(N)
    
    # Ensure all points are normalized
    best_points /= np.linalg.norm(best_points, axis=1, keepdims=True)
    
    return best_points
