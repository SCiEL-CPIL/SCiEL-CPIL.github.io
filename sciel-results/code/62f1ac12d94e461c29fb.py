"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

N = 150
TIME_LIMIT = 145.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import math

def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms

def _log_energy(points):
    """Compute the logarithmic energy sum_{i<j} log(1/|x_i - x_j|)."""
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dists_sq, 1.0)
    dists = np.sqrt(dists_sq)
    log_dists = np.log(dists)
    np.fill_diagonal(log_dists, 0.0)
    energy = -0.5 * np.sum(log_dists)
    return energy

def _spherical_centroid(vertices, center):
    """
    Compute the spherical centroid (mean direction) of a set of vertices
    defining a spherical polygon (face of the convex hull).
    """
    if len(vertices) == 0:
        return center
    if len(vertices) == 1:
        return vertices[0]
    
    # Convert vertices to unit vectors
    v_norms = np.linalg.norm(vertices, axis=1, keepdims=True)
    v_norms[v_norms < 1e-15] = 1.0
    verts = vertices / v_norms
    
    # Compute the centroid in 3D space, then normalize back to sphere
    # This is the standard approximation for spherical centroid (mean of vectors)
    centroid = np.mean(verts, axis=0)
    norm = np.linalg.norm(centroid)
    if norm < 1e-15:
        return center
    return centroid / norm

def _lloyd_sphere(points, iterations=100, deadline=None):
    """
    Perform Lloyd's algorithm on the sphere using ConvexHull for Voronoi cells.
    """
    pts = points.copy()
    n = len(pts)
    
    for it in range(iterations):
        if deadline and time.time() > deadline - 5:
            break
            
        try:
            # Compute convex hull
            hull = ConvexHull(pts)
            
            # Map faces to point indices
            # hull.simplices contains indices of points forming each face
            # We need to assign each point to the face(s) it "owns" in the dual sense.
            # Actually, the Voronoi cell of point i is the polar dual of the face(s) of the hull containing i.
            # For a strictly convex position, each point is a vertex of the hull.
            # The Voronoi cell of point i corresponds to the set of hull faces that have i as a vertex.
            # The centroid of the Voronoi cell is the normalized sum of the face normals of those faces.
            
            # 1. Compute face normals
            normals = hull.equations[:, :3]
            # The equations are of the form a*x + b*y + c*z = 1 (since points are on unit sphere, 
            # the support function is 1). The normal vector is (a,b,c).
            
            # 2. For each point i, find all faces that contain i
            # hull.simplices is a list of (3 indices) for each face.
            point_to_faces = [[] for _ in range(n)]
            for face_idx, simplex in enumerate(hull.simplices):
                for p_idx in simplex:
                    point_to_faces[p_idx].append(face_idx)
            
            new_pts = pts.copy()
            for i in range(n):
                faces = point_to_faces[i]
                if not faces:
                    continue
                
                # Sum the normals of the adjacent faces
                # The Voronoi cell is the intersection of half-spaces. 
                # The "center" of the cell is approximately the normalized sum of the outward normals of the adjacent faces.
                # This is a standard heuristic for spherical Lloyd's (often called "Voronoi relaxation").
                cell_normal = np.sum(normals[faces], axis=0)
                
                norm = np.linalg.norm(cell_normal)
                if norm < 1e-15:
                    continue
                    
                new_pts[i] = cell_normal / norm
            
            # Check if we improved
            old_e = _log_energy(pts)
            new_e = _log_energy(new_pts)
            
            if new_e < old_e:
                pts = new_pts
            else:
                # If it got worse, we might be stuck or oscillating. 
                # In strict Lloyd's, energy should decrease monotonically.
                # If it doesn't, break or try a smaller step.
                # For now, break to avoid infinite loops on non-convergent cases.
                break
                
        except Exception:
            # ConvexHull can fail if points are coplanar or duplicate
            break
            
    return pts

def _gradient_polish(points, deadline, maxiter=200):
    """Polish the points using L-BFGS-B with analytic gradient."""
    n = len(points)
    
    def objective_and_grad(x):
        pts = _normalize(x.reshape(-1, 3))
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff * diff, axis=1)
        np.fill_diagonal(dists_sq, 1.0)
        dists = np.sqrt(dists_sq)
        log_dists = np.log(dists)
        np.fill_diagonal(log_dists, 0.0)
        energy = -0.5 * np.sum(log_dists)
        
        # Gradient of sum_{i<j} log(1/|xi-xj|)
        # d/dxi sum_{j!=i} -log|xi-xj| = sum_{j!=i} -(xi-xj)/|xi-xj|^2
        # Note: The factor 0.5 in energy is for double counting, but gradient is sum over j!=i.
        # Let's verify: E = -1/2 sum_{i,j, i!=j} log|xi-xj|.
        # dE/dxi = -1/2 [ d/dxi sum_{j!=i} log|xi-xj| + d/dxi sum_{k!=i} log|xi-xk| (where i is the second index) ]
        # = -1/2 [ sum_{j!=i} (xi-xj)/|xi-xj|^2 + sum_{k!=i} (xi-xk)/|xi-xk|^2 ]
        # = - sum_{j!=i} (xi-xj)/|xi-xj|^2
        
        grad_vec = np.zeros_like(pts)
        for i in range(n):
            # Exclude i
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            diff_i = pts[i] - pts[mask]
            dists_sq_i = np.sum(diff_i * diff_i, axis=1)
            # grad_i = - sum (xi - xj) / |xi - xj|^2
            # = - sum diff_i / dists_sq_i
            forces = diff_i / dists_sq_i[:, np.newaxis]
            grad_vec[i] = -np.sum(forces, axis=0)
            
        # Project to tangent space
        grad_tangent = np.zeros_like(grad_vec)
        for i in range(n):
            proj = np.dot(grad_vec[i], pts[i])
            grad_tangent[i] = grad_vec[i] - proj * pts[i]
            
        return energy, grad_tangent.flatten()
    
    try:
        result = minimize(
            objective_and_grad,
            points.flatten(),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': maxiter, 'maxls': 50, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        return _normalize(result.x.reshape(-1, 3))
    except Exception:
        return points

def _fibonacci_sphere(n, seed=0.0):
    """Generate n points on the sphere using a Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * (i + seed)
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n, rng=None):
    """Generate n random points on the unit sphere."""
    if rng is None:
        rng = np.random.default_rng()
    pts = rng.standard_normal((n, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT

    best_energy = float('inf')
    best_points = _fibonacci_sphere(N)
    
    # Evaluate initial Fibonacci
    try:
        e = _log_energy(best_points)
        if e < best_energy:
            best_energy = e
    except Exception:
        pass

    rng = np.random.default_rng(42)
    
    # Strategy: Multi-start with Lloyd's (CVT) then Gradient Polish
    # Lloyd's is excellent for global structure, Gradient is excellent for local precision.
    
    num_restarts = 30 # We can afford more restarts if Lloyd's is fast enough
    
    for k in range(num_restarts):
        if time.time() > deadline - 20:
            break
        
        # Generate initial points
        if k == 0:
            pts = _fibonacci_sphere(N)
        else:
            # Mix of random and perturbed fibonacci
            if k % 2 == 0:
                pts = _random_sphere(N, rng)
            else:
                pts = _fibonacci_sphere(N, seed=k*0.1)
                # Add small noise to break symmetry
                noise = rng.standard_normal((N, 3)) * 0.1
                pts = _normalize(pts + noise)
        
        try:
            # Phase 1: Lloyd's Algorithm (Voronoi Relaxation)
            # This step is crucial for escaping local minima of simple gradient descent
            if time.time() < deadline - 15:
                lloyd_pts = _lloyd_sphere(pts, iterations=50, deadline=deadline)
                e = _log_energy(lloyd_pts)
                
                if e < best_energy:
                    best_energy = e
                    best_points = lloyd_pts
                    
                    # Phase 2: Gradient Polish
                    if time.time() < deadline - 5:
                        polished = _gradient_polish(lloyd_pts, deadline, maxiter=100)
                        e2 = _log_energy(polished)
                        if e2 < best_energy:
                            best_energy = e2
                            best_points = polished
                            
        except Exception:
            continue

    # Final polish on the best found so far
    try:
        if time.time() < deadline - 2:
            final_pts = _gradient_polish(best_points, deadline, maxiter=200)
            e = _log_energy(final_pts)
            if e < best_energy:
                best_energy = e
                best_points = final_pts
    except Exception:
        pass

    return _normalize(best_points)

# EVOLVE-BLOCK-END
