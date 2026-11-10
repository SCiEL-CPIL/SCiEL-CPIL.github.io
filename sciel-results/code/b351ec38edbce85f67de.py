import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution
import time

N = 50

def _covering_radius_exact(points):
    """Calculate exact covering radius using convex hull."""
    try:
        hull = ConvexHull(points)
        faces = hull.simplices
        max_radius = 0.0
        for face in faces:
            v1, v2, v3 = points[face]
            # Calculate circumcenter of spherical triangle
            n = np.cross(v2 - v1, v3 - v1)
            norm_n = np.linalg.norm(n)
            if norm_n < 1e-10:
                continue
            n = n / norm_n
            
            u = v2 - v1
            w = v3 - v1
            A = np.array([[np.dot(u, u), np.dot(w, u)], [np.dot(u, w), np.dot(w, w)]])
            B = np.array([np.dot(u, u), np.dot(w, w)])
            
            try:
                coeffs = np.linalg.solve(A, B)
            except np.linalg.LinAlgError:
                continue
            a, b = coeffs
            c_p = v1 + a * u + b * w
            norm_cp = np.linalg.norm(c_p)
            if norm_cp < 1e-10:
                continue
            c_s = c_p / norm_cp
            cos_R = np.dot(c_s, v1)
            cos_R = np.clip(cos_R, -1.0, 1.0)
            R = np.arccos(cos_R)
            if R > max_radius:
                max_radius = R
        return max_radius
    except Exception:
        return np.pi

def _covering_radius_approx(points, test_points):
    """Calculate approximate covering radius using sampled points."""
    # points: (N, 3)
    # test_points: (M, 3)
    # Compute distances: (M, N)
    # d^2 = |p|^2 + |t|^2 - 2 p.t
    # Since |p|=|t|=1, d^2 = 2 - 2 p.t
    # We want min d, so max p.t
    dots = test_points @ points.T
    max_dot_per_point = np.max(dots, axis=1)
    max_dot = np.max(max_dot_per_point)
    max_dot = np.clip(max_dot, -1.0, 1.0)
    return np.arccos(max_dot)

def construct():
    start_time = time.time()
    max_time = 110.0
    
    # Generate fixed test points for approximate objective
    # Use Fibonacci sphere for uniform distribution
    M = 5000
    i = np.arange(M) + 0.5
    z = 1.0 - 2.0 * i / M
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    test_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Initial guess: Random or Fibonacci
    # Let's start with Fibonacci spiral as a good initial guess for DE
    i_init = np.arange(N) + 0.5
    z_init = 1.0 - 2.0 * i_init / N
    phi_init = np.pi * (1.0 + np.sqrt(5.0)) * i_init
    r_init = np.sqrt(1.0 - z_init * z_init)
    init_points = np.column_stack([r_init * np.cos(phi_init), r_init * np.sin(phi_init), z_init])
    
    # Define objective for DE
    def objective(x):
        points = x.reshape(N, 3)
        # Normalize
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        points = points / norms
        
        # Check for coincident points (optional, DE handles it)
        # If points are too close, the hull might be invalid or radius large
        # Approx radius is a good proxy
        rad = _covering_radius_approx(points, test_points)
        return rad
    
    # Bounds for DE: [-1, 1] for each coordinate
    bounds = [(-1.0, 1.0)] * (N * 3)
    
    # Run Differential Evolution
    # maxiter and popsize tuned for speed
    # seed for reproducibility
    result = differential_evolution(
        objective, 
        bounds, 
        maxiter=30, 
        popsize=10, 
        tol=0.01,
        seed=42,
        polish=False # We will polish manually
    )
    
    best_points = result.x.reshape(N, 3)
    # Normalize best_points
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    best_points = best_points / norms
    
    # Polish with local optimization using exact radius
    # Use the logic from the original seed but faster
    step_size = 0.05
    decay = 0.98
    current_points = best_points.copy()
    
    while time.time() - start_time < max_time:
        try:
            hull = ConvexHull(current_points)
            faces = hull.simplices
            
            max_radius = 0
            max_face_idx = -1
            max_circumcenter = None
            
            for idx, face in enumerate(faces):
                v1, v2, v3 = current_points[face]
                n = np.cross(v2 - v1, v3 - v1)
                norm_n = np.linalg.norm(n)
                if norm_n < 1e-10:
                    continue
                n = n / norm_n
                
                u = v2 - v1
                w = v3 - v1
                A = np.array([[np.dot(u, u), np.dot(w, u)], [np.dot(u, w), np.dot(w, w)]])
                B = np.array([np.dot(u, u), np.dot(w, w)])
                
                try:
                    coeffs = np.linalg.solve(A, B)
                except np.linalg.LinAlgError:
                    continue
                a, b = coeffs
                c_p = v1 + a * u + b * w
                norm_cp = np.linalg.norm(c_p)
                if norm_cp < 1e-10:
                    continue
                c_s = c_p / norm_cp
                cos_R = np.dot(c_s, v1)
                cos_R = np.clip(cos_R, -1.0, 1.0)
                R = np.arccos(cos_R)
                
                if R > max_radius:
                    max_radius = R
                    max_face_idx = idx
                    max_circumcenter = c_s
            
            if max_face_idx == -1:
                break
                
            face = faces[max_face_idx]
            for i in face:
                v = current_points[i]
                current_points[i] = v + step_size * (max_circumcenter - v)
            
            norms = np.linalg.norm(current_points, axis=1, keepdims=True)
            norms[norms < 1e-10] = 1.0
            current_points = current_points / norms
            
            step_size *= decay
            
        except Exception:
            break
            
    return current_points
