"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull


def _spherical_circumradius(p1, p2, p3):
    """Compute the circumradius (angular distance) of a spherical triangle."""
    # Side lengths (angular distances)
    a = np.arccos(np.clip(np.dot(p1, p2), -1.0, 1.0))
    b = np.arccos(np.clip(np.dot(p2, p3), -1.0, 1.0))
    c = np.arccos(np.clip(np.dot(p3, p1), -1.0, 1.0))
    
    # Spherical law of cosines for angles
    # We need the circumradius R.
    # Formula: cos(R) = (cos(A) cos(B) cos(C) + cos(A) + cos(B) + cos(C) - 1) / (2 * (cos(A) cos(B) + cos(B) cos(C) + cos(C) cos(A) - 1)) ? No, that's for area.
    
    # Simpler: The circumcenter is the intersection of the perpendicular bisectors.
    # Or use the formula for the circumradius of a spherical triangle:
    # sin(R) = sin(a) sin(b) sin(c) / (4 * sin(s) * sin(s-a) * sin(s-b) * sin(s-c)) ? No.
    
    # Let's find the circumcenter point C on the sphere.
    # C is equidistant from p1, p2, p3.
    # C = normalize( p1 * (||p2||^2 ||p3||^2 - (p2.p3)^2) + ... ) ?
    # In 3D, the circumcenter of a triangle with vertices on the unit sphere is the normalized vector:
    # v = (||p2||^2 ||p3||^2 - (p2.p3)^2) p1 + (||p1||^2 ||p3||^2 - (p1.p3)^2) p2 + (||p1||^2 ||p2||^2 - (p1.p2)^2) p3
    # Since ||pi|| = 1:
    # v = (1 - (p2.p3)^2) p1 + (1 - (p1.p3)^2) p2 + (1 - (p1.p2)^2) p3
    
    p2p3 = np.dot(p2, p3)
    p1p3 = np.dot(p1, p3)
    p1p2 = np.dot(p1, p2)
    
    v = (1.0 - p2p3**2) * p1 + (1.0 - p1p3**2) * p2 + (1.0 - p1p2**2) * p3
    
    norm_v = np.linalg.norm(v)
    if norm_v < 1e-10:
        return 0.0 # Degenerate
        
    c_center = v / norm_v
    
    # The circumradius is the angular distance from c_center to any vertex, e.g., p1
    R = np.arccos(np.clip(np.dot(c_center, p1), -1.0, 1.0))
    return R


def _covering_radius_and_worst_face(points):
    """Compute the covering radius and the indices of the worst face."""
    try:
        hull = ConvexHull(points)
    except Exception:
        # Fallback if hull fails (e.g. points are coplanar or degenerate)
        # Use a grid-based approximation
        n_test = 2000
        i = np.arange(n_test) + 0.5
        z = 1.0 - 2.0 * i / n_test
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(0, 1.0 - z * z))
        test_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        cos_d = np.clip(test_points @ points.T, -1.0, 1.0)
        min_cos = np.max(cos_d, axis=1)
        min_angles = np.arccos(np.clip(min_cos, -1.0, 1.0))
        return np.max(min_angles), None

    max_radius = 0.0
    worst_face = None
    
    # Iterate over all triangular facets
    for simplex in hull.simplices:
        i, j, k = simplex
        R = _spherical_circumradius(points[i], points[j], points[k])
        if R > max_radius:
            max_radius = R
            worst_face = simplex
            
    return max_radius, worst_face


def construct():
    # Initialize with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Optimization loop
    # We will use a simple iterative improvement:
    # 1. Find the worst face (largest circumradius).
    # 2. Move the vertices of that face towards the circumcenter of that face.
    # 3. This reduces the covering radius of that face.
    # 4. To avoid getting stuck, we also apply small random perturbations.
    
    max_iter = 500
    step_size = 0.1
    noise_scale = 0.05
    
    current_radius, _ = _covering_radius_and_worst_face(points)
    best_radius = current_radius
    best_points = points.copy()
    
    for it in range(max_iter):
        # Decay step size and noise
        decay = 1.0 - it / max_iter
        curr_step = step_size * decay
        curr_noise = noise_scale * decay
        
        radius, face = _covering_radius_and_worst_face(points)
        
        if radius < best_radius:
            best_radius = radius
            best_points = points.copy()
            
        if face is None:
            # Fallback if hull failed, use random moves
            noise = np.random.normal(0, curr_noise, size=(N, 3))
            points = points + noise
            points /= np.linalg.norm(points, axis=1, keepdims=True)
            continue
            
        i, j, k = face
        p_i = points[i]
        p_j = points[j]
        p_k = points[k]
        
        # Compute circumcenter of the worst face
        p2p3 = np.dot(p_j, p_k)
        p1p3 = np.dot(p_i, p_k)
        p1p2 = np.dot(p_i, p_j)
        
        v = (1.0 - p2p3**2) * p_i + (1.0 - p1p3**2) * p_j + (1.0 - p1p2**2) * p_k
        norm_v = np.linalg.norm(v)
        
        if norm_v > 1e-10:
            c_center = v / norm_v
        else:
            c_center = (p_i + p_j + p_k) / 3.0
            c_center /= np.linalg.norm(c_center)
            
        # Move vertices towards circumcenter
        # We move them a fraction of the way
        points[i] = p_i + curr_step * (c_center - p_i)
        points[j] = p_j + curr_step * (c_center - p_j)
        points[k] = p_k + curr_step * (c_center - p_k)
        
        # Add small noise to all points to escape local minima
        noise = np.random.normal(0, curr_noise, size=(N, 3))
        points = points + noise
        
        # Normalize
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        
    # Final polish: run a few more iterations with very small step
    for _ in range(50):
        radius, face = _covering_radius_and_worst_face(points)
        if radius < best_radius:
            best_radius = radius
            best_points = points.copy()
        if face is None:
            break
        i, j, k = face
        p_i = points[i]
        p_j = points[j]
        p_k = points[k]
        p2p3 = np.dot(p_j, p_k)
        p1p3 = np.dot(p_i, p_k)
        p1p2 = np.dot(p_i, p_j)
        v = (1.0 - p2p3**2) * p_i + (1.0 - p1p3**2) * p_j + (1.0 - p1p2**2) * p_k
        norm_v = np.linalg.norm(v)
        if norm_v > 1e-10:
            c_center = v / norm_v
        else:
            c_center = (p_i + p_j + p_k) / 3.0
            c_center /= np.linalg.norm(c_center)
            
        points[i] = p_i + 0.01 * (c_center - p_i)
        points[j] = p_j + 0.01 * (c_center - p_j)
        points[k] = p_k + 0.01 * (c_center - p_k)
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        
    # Return the best configuration found
    return best_points
# EVOLVE-BLOCK-END
