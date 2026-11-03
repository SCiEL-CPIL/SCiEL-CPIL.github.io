import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 130

def normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms

def fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def get_voronoi_vertices(points):
    # points: (N, 3)
    try:
        hull = ConvexHull(points)
    except Exception:
        return None
    
    # Hull simplices are triangles
    # For each triangle, compute circumcenter on unit sphere
    vertices = []
    for tri in hull.simplices:
        a, b, c = points[tri]
        # Circumcenter of a, b, c on unit sphere
        # Solve |x-a|^2 = |x-b|^2 = |x-c|^2 = 1
        # x . (a-b) = (|a|^2 - |b|^2)/2 = 0 (since on sphere)
        # So x is orthogonal to a-b and b-c.
        # x is parallel to (a-b) x (b-c)
        v1 = a - b
        v2 = b - c
        cross = np.cross(v1, v2)
        norm = np.linalg.norm(cross)
        if norm < 1e-10:
            continue
        center = cross / norm
        # Ensure it's on the sphere (it is, by construction)
        # Check if it's the "correct" center (outside the triangle? No, Voronoi vertex is circumcenter)
        # There are two circumcenters (center and -center). We want the one that is "far" from the origin? No, both are on sphere.
        # The Voronoi vertex is the point on the sphere equidistant from a,b,c.
        # It must be such that the angle is < 180?
        # Actually, for a convex hull, the circumcenter of a facet is the Voronoi vertex.
        # We need to pick the correct sign.
        # The center should be on the same side of the plane abc as the origin? No, opposite?
        # Let's check distance to a.
        # If center . a > 0, it's in the hemisphere of a.
        # The Voronoi vertex for the facet is the point on the sphere that is the circumcenter.
        # It is unique up to sign. The correct one is the one that is "outside" the convex hull?
        # Actually, for points on the unit sphere, the circumcenter of the triangle formed by a,b,c is either c or -c.
        # The Voronoi vertex is the one that is "far" from the interior of the hull?
        # Let's just pick the one that maximizes distance to the centroid of the triangle?
        # Or simply: The Voronoi cell is on the sphere. The vertex is the intersection of the sphere with the line through the origin and the 3D circumcenter?
        # Yes. The 3D circumcenter of the triangle on the sphere lies on the line passing through the origin and the point on the sphere equidistant from a,b,c.
        # So `center` and `-center` are candidates.
        # The correct Voronoi vertex is the one that is "outside" the convex hull?
        # Actually, the Voronoi vertex is the point on the sphere that is the circumcenter.
        # Let's test: if a,b,c are close, the circumcenter is close to them.
        # The vector `cross` points in the direction of the normal.
        # We need to ensure the center is on the "correct" side.
        # The centroid of the triangle is `centroid = (a+b+c)/3`.
        # The center should be in the direction of `centroid`?
        # If `center . centroid > 0`, then `center` is the Voronoi vertex.
        if np.dot(center, (a+b+c)/3) < 0:
            center = -center
        vertices.append(center)
    return np.array(vertices)

def covering_radius(points):
    # Compute max angular distance from any Voronoi vertex to its nearest point
    vv = get_voronoi_vertices(points)
    if vv is None or len(vv) == 0:
        return np.pi # invalid
    # For each vv, find min angle to any point
    # cos(theta) = vv . p
    # theta = arccos(vv . p)
    # We want max theta
    # dists = 1 - vv @ points.T  (small dist -> large cos)
    # min_dist = min(1 - vv @ points.T)
    # max_angle = arccos(1 - min_dist)
    dots = vv @ points.T
    # Clip to [-1, 1]
    dots = np.clip(dots, -1.0, 1.0)
    min_dots = np.min(dots, axis=1) # Max distance corresponds to min dot product
    angles = np.arccos(min_dots)
    return np.max(angles)

def smooth_objective(flat_points, beta=50.0):
    points = flat_points.reshape(-1, 3)
    points = normalize(points)
    vv = get_voronoi_vertices(points)
    if vv is None or len(vv) == 0:
        return 1e10
    dots = vv @ points.T
    dots = np.clip(dots, -1.0, 1.0)
    min_dots = np.min(dots, axis=1)
    # Distance is arccos(min_dots).
    # We want to minimize max distance.
    # Use log-sum-exp of (beta * distance)
    dists = np.arccos(min_dots)
    # To avoid overflow, shift
    m = np.max(dists)
    lse = np.log(np.sum(np.exp(beta * (dists - m)))) + beta * m
    return lse / beta

def lloyd_step(points):
    vv = get_voronoi_vertices(points)
    if vv is None:
        return points
    
    # Assign each voronoi vertex to the nearest point?
    # No, a voronoi vertex is shared by 3 points.
    # We need to build the Voronoi cells.
    # For each point i, the cell is formed by the voronoi vertices of the triangles incident to i.
    
    # Re-compute hull to get simplices
    try:
        hull = ConvexHull(points)
    except:
        return points
        
    # Map point index to list of voronoi vertices
    cell_vertices = [[] for _ in range(len(points))]
    
    for idx, tri in enumerate(hull.simplices):
        # tri contains indices of points
        # vv[idx] is the voronoi vertex for this triangle
        for p_idx in tri:
            cell_vertices[p_idx].append(vv[idx])
            
    new_points = points.copy()
    for i in range(len(points)):
        if cell_vertices[i]:
            verts = np.array(cell_vertices[i])
            centroid = np.mean(verts, axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 1e-10:
                new_points[i] = centroid / norm
            else:
                # Fallback: keep old point or random
                pass
    return new_points

def construct():
    start_time = time.time()
    deadline = 140.0 # 10s margin
    
    best_points = None
    best_radius = np.pi
    
    # Initial guess
    base_points = fibonacci_sphere(N)
    
    # Try a few restarts with different noise
    for restart in range(20):
        if time.time() - start_time > deadline:
            break
            
        # Add noise
        noise = np.random.normal(0, 0.1, size=(N, 3))
        points = normalize(base_points + noise)
        
        # Lloyd iterations
        for _ in range(20):
            if time.time() - start_time > deadline:
                break
            points = lloyd_step(points)
            
        # Polish with scipy
        flat_points = points.flatten()
        
        def obj(x):
            return smooth_objective(x, beta=50.0)
            
        try:
            res = minimize(obj, flat_points, method='L-BFGS-B', options={'maxiter': 200, 'maxfun': 2000})
            if res.success or res.fun < obj(flat_points):
                polished_points = normalize(res.x.reshape(-1, 3))
                radius = covering_radius(polished_points)
                if radius < best_radius:
                    best_radius = radius
                    best_points = polished_points
        except Exception:
            pass
        
        # Evaluate current points if polish failed or didn't improve
        current_radius = covering_radius(points)
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points
            
    return best_points
