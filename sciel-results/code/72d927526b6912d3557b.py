"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 130

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 120.0  # seconds, with safety margin
    
    # 1. Initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    
    # 2. Compute covering radius via convex hull facets
    def _covering_radius(pts):
        from scipy.spatial import ConvexHull
        try:
            hull = ConvexHull(pts)
            max_r = 0.0
            for simplex in hull.simplices:
                face_pts = pts[simplex]
                # Centroid of the face
                centroid = np.mean(face_pts, axis=0)
                centroid = centroid / np.linalg.norm(centroid)
                # Circumradius of the face: max angular distance from centroid to face vertices
                # Actually, for a spherical cap covering the face, the center is the spherical circumcenter
                # For a triangle on the sphere, the circumradius is the max angular distance from the circumcenter to vertices
                # The circumcenter of a spherical triangle is the intersection of perpendicular bisectors
                # For small faces, the centroid is a good approximation
                # More accurately: the covering radius for this facet is the angular radius of the smallest spherical cap containing the facet
                # This is the circumradius of the spherical triangle
                # For a planar triangle projected to sphere, circumradius ≈ max distance from circumcenter to vertices
                # Let's use the centroid as an approximation for the cap center
                dists = np.arccos(np.clip(np.dot(face_pts, centroid), -1, 1))
                max_r = max(max_r, np.max(dists))
            return np.degrees(max_r)
        except Exception:
            return 180.0
    
    current_cr = _covering_radius(pts)
    
    # 3. Local refinement: move points to reduce covering radius
    def _move_point(pts, idx, step_size=0.01):
        """Try moving point idx in random directions, keep if covering radius decreases."""
        p = pts[idx]
        # Generate random tangent direction
        rand_vec = np.random.randn(3)
        tangent = rand_vec - np.dot(rand_vec, p) * p
        tangent = tangent / (np.linalg.norm(tangent) + 1e-15)
        
        new_p = p + step_size * tangent
        new_p = new_p / np.linalg.norm(new_p)
        
        new_pts = pts.copy()
        new_pts[idx] = new_p
        return new_pts
    
    # Iterative local search
    step_size = 0.05
    for iteration in range(5000):
        if time.time() > deadline - 10.0:
            break
        
        # Pick a random point to move
        idx = np.random.randint(N)
        
        # Try several random directions
        best_new_pts = None
        best_cr = current_cr
        
        for _ in range(5):
            new_pts = _move_point(pts, idx, step_size * np.random.uniform(0.5, 1.5))
            cr = _covering_radius(new_pts)
            if cr < best_cr - 1e-10:
                best_cr = cr
                best_new_pts = new_pts
        
        if best_new_pts is not None:
            pts = best_new_pts
            current_cr = best_cr
            # Decrease step size occasionally
            if iteration % 500 == 0 and step_size > 0.001:
                step_size *= 0.9
        else:
            # No improvement, try smaller step
            if iteration % 100 == 0 and step_size > 0.001:
                step_size *= 0.95
    
    # 4. Final polish: more careful local optimization
    step_size = 0.01
    for iteration in range(3000):
        if time.time() > deadline - 5.0:
            break
        
        idx = np.random.randint(N)
        
        best_new_pts = None
        best_cr = current_cr
        
        for _ in range(10):
            new_pts = _move_point(pts, idx, step_size * np.random.uniform(0.3, 1.0))
            cr = _covering_radius(new_pts)
            if cr < best_cr - 1e-12:
                best_cr = cr
                best_new_pts = new_pts
        
        if best_new_pts is not None:
            pts = best_new_pts
            current_cr = best_cr
            if iteration % 300 == 0 and step_size > 0.0001:
                step_size *= 0.9
        else:
            if iteration % 200 == 0 and step_size > 0.0001:
                step_size *= 0.95
    
    # 5. Ensure points are on unit sphere and non-zero
    norms = np.linalg.norm(pts, axis=1)
    pts = pts / norms[:, np.newaxis]
    
    return pts
# EVOLVE-BLOCK-END
