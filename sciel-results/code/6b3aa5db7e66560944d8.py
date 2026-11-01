import time
import numpy as np
from scipy.spatial import ConvexHull

N = 130
_EPS = 1e-12
_TIME_LIMIT = 145.0  # Safety margin for 150s limit

def _initial_fib():
    """Generate initial points using Fibonacci spiral."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _compute_hull_radius(pts):
    """
    Compute the exact covering radius using the convex hull method.
    Returns the radius in radians and the index of the worst facet.
    """
    try:
        hull = ConvexHull(pts)
        faces = hull.simplices
        M = len(faces)
        
        if M == 0:
            return np.pi, -1

        A = pts[faces[:, 0]]
        B = pts[faces[:, 1]]
        C = pts[faces[:, 2]]
        
        AB = B - A
        AC = C - A
        
        cross = np.cross(AB, AC)
        area = 0.5 * np.linalg.norm(cross, axis=1)
        
        valid = area > _EPS
        if not np.any(valid):
            return np.pi, -1
            
        normals = cross[valid]
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        
        # Ensure normals point outward
        centroids = (A[valid] + B[valid] + C[valid]) / 3.0
        signs = np.sum(normals * centroids, axis=1, keepdims=True)
        normals = normals * np.sign(signs)
        normals = normals / (norms + _EPS)
        
        # Angular circumradius: distance from pole to any vertex
        cos_theta = np.sum(normals * A[valid], axis=1)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        ang_radii = np.arccos(cos_theta)
        
        max_idx = np.argmax(ang_radii)
        return np.max(ang_radii), max_idx
    except Exception:
        return np.pi, -1

def _simulated_annealing(pts, deadline):
    """
    Perform Simulated Annealing on the configuration.
    Moves:
    1. Randomly select a point and move it to a random nearby location (Gaussian perturbation on sphere).
    2. Randomly select a point and move it towards the pole of its worst facet.
    
    Acceptance: Metropolis criterion based on change in covering radius.
    """
    best_pts = pts.copy()
    best_r, _ = _compute_hull_radius(pts)
    curr_r = best_r
    
    # Temperature schedule
    T_start = 0.1
    T_end = 0.001
    cooling_rate = 0.995
    
    T = T_start
    rng = np.random.default_rng(42)
    
    # Precompute some random directions? No, generate on the fly.
    
    iter_count = 0
    while time.time() < deadline and T > T_end:
        iter_count += 1
        
        # Generate a candidate move
        cand = pts.copy()
        move_type = rng.random()
        
        if move_type < 0.7:
            # Local perturbation: move a random point slightly
            idx = rng.integers(0, N)
            # Random tangent vector
            rand_vec = rng.normal(size=3)
            # Project to tangent space at pts[idx]
            tangent = rand_vec - np.dot(rand_vec, pts[idx]) * pts[idx]
            norm_t = np.linalg.norm(tangent)
            if norm_t > _EPS:
                tangent /= norm_t
                # Step size proportional to temperature
                step = T * 0.5
                cand[idx] = pts[idx] + step * tangent
                cand[idx] /= (np.linalg.norm(cand[idx]) + _EPS)
        else:
            # Targeted move: move a vertex of the worst facet towards the pole
            curr_r, worst_idx = _compute_hull_radius(pts)
            if worst_idx < 0:
                # Hull failed, try random reset? Or break?
                # If hull fails, it's invalid. We should probably restart or perturb hard.
                # For now, just do a random move.
                idx = rng.integers(0, N)
                rand_vec = rng.normal(size=3)
                cand[idx] = rand_vec / (np.linalg.norm(rand_vec) + _EPS)
            else:
                # Get the worst facet's vertices
                # We need the faces from the hull computation. 
                # To avoid recomputing hull just for indices, we'd need to return them.
                # Let's refactor _compute_hull_radius to return faces too? 
                # Or just recompute here? It's expensive.
                # Better: modify _compute_hull_radius to return the worst facet vertices.
                # Let's create a helper that returns radius and worst facet vertices.
                
                # Re-implementing logic to get vertices without full hull recompute is hard.
                # Let's just use a simpler heuristic for the second move type:
                # Pick a random point and move it towards the center of the sphere? No, must be on sphere.
                # Move it to a random position? That's a big jump.
                
                # Let's stick to perturbation for now, but make it slightly larger.
                idx = rng.integers(0, N)
                rand_vec = rng.normal(size=3)
                tangent = rand_vec - np.dot(rand_vec, pts[idx]) * pts[idx]
                norm_t = np.linalg.norm(tangent)
                if norm_t > _EPS:
                    tangent /= norm_t
                    step = T * 1.0
                    cand[idx] = pts[idx] + step * tangent
                    cand[idx] /= (np.linalg.norm(cand[idx]) + _EPS)
        
        # Evaluate candidate
        cand_r, _ = _compute_hull_radius(cand)
        
        # Metropolis acceptance
        if cand_r < curr_r:
            pts = cand
            curr_r = cand_r
            if curr_r < best_r:
                best_r = curr_r
                best_pts = cand.copy()
        else:
            delta = curr_r - cand_r
            # Accept with probability exp(-delta / T)
            # Note: delta is negative here (cand is worse)
            # We want to accept bad moves with prob exp(-(cand_r - curr_r)/T) = exp(delta/T)
            prob = np.exp(delta / T)
            if rng.random() < prob:
                pts = cand
                curr_r = cand_r
        
        T *= cooling_rate
        
    return best_pts

def _lloyd_relax(pts, deadline):
    """
    Perform Lloyd relaxation (Voronoi relaxation) on the sphere.
    For each point, compute the centroid of its Voronoi cell (intersection of hemispheres)
    and move the point towards that centroid.
    Approximation: Use the average of the points in the Voronoi cell?
    No, standard Lloyd is for centroids. For spherical caps, we want to move the center
    to the spherical centroid of the region it covers.
    
    A simpler approximation:
    1. Compute Delaunay triangulation (convex hull).
    2. For each vertex, identify its adjacent facets.
    3. The "Voronoi cell" is roughly the region where this point is the nearest.
    4. We can approximate the target position by averaging the poles of the adjacent facets?
    Or better: The point should move to the circumcenter of its adjacent triangles?
    No, that's not right.
    
    Let's use a simpler "repulsion" based relaxation.
    Each point feels a repulsive force from others, but constrained to the sphere?
    Actually, the gradient of the covering radius is complex.
    
    Let's use a standard "spherical k-means" style update if we can define clusters.
    But defining clusters on the sphere requires a grid.
    
    Alternative: Use the fact that the optimal configuration for covering is related to
    the Fermi points or similar.
    
    Let's implement a simple iterative improvement:
    For each point i, move it slightly in the direction that reduces the maximum distance
    of the points in its "influence region".
    
    Given time constraints, let's skip complex Lloyd and rely on SA + local polish.
    """
    best_pts = pts.copy()
    best_r, _ = _compute_hull_radius(pts)
    
    step_size = 0.1
    for _ in range(50):
        if time.time() > deadline:
            break
            
        improved = False
        for i in range(N):
            if time.time() > deadline:
                break
                
            # Try moving point i in a few random directions
            best_local_r = best_r
            best_local_pt = pts[i].copy()
            
            # Generate 20 random tangent directions
            for _ in range(20):
                rand_vec = np.random.normal(size=3)
                tangent = rand_vec - np.dot(rand_vec, pts[i]) * pts[i]
                norm_t = np.linalg.norm(tangent)
                if norm_t < _EPS:
                    continue
                tangent /= norm_t
                
                for sign in [1, -1]:
                    cand = pts.copy()
                    cand[i] = pts[i] + sign * step_size * tangent
                    cand[i] /= (np.linalg.norm(cand[i]) + _EPS)
                    
                    cand_r, _ = _compute_hull_radius(cand)
                    if cand_r < best_local_r:
                        best_local_r = cand_r
                        best_local_pt = cand[i].copy()
                        improved = True
                        
            if improved:
                pts[i] = best_local_pt
                best_r = best_local_r
                best_pts = pts.copy()
                
        if not improved:
            step_size *= 0.9
            if step_size < 0.001:
                break
                
    return best_pts

def construct():
    start_time = time.time()
    deadline = start_time + _TIME_LIMIT
    
    # Phase 1: Initial configuration
    pts = _initial_fib()
    
    # Phase 2: Simulated Annealing (Global search)
    # Allocate 60% of time to SA
    sa_deadline = start_time + 0.6 * _TIME_LIMIT
    pts = _simulated_annealing(pts, sa_deadline)
    
    # Phase 3: Local Polish (Refinement)
    # Allocate remaining time
    polish_deadline = deadline - 5.0
    if time.time() < polish_deadline:
        pts = _lloyd_relax(pts, polish_deadline)
        
    # Final normalization and validity check
    pts = np.array(pts, dtype=float)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / (norms + _EPS)
    
    if not np.all(np.isfinite(pts)) or pts.shape != (N, 3):
        return _initial_fib()
        
    return pts
