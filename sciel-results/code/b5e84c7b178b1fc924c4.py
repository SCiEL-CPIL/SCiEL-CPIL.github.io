import numpy as np
from scipy.spatial.distance import pdist

def min_max_dist_dim3_14() -> np.ndarray:
    n = 14
    
    # Initialize 14 unit quaternions
    # Start with Icosahedron vertices mapped to pure imaginary quaternions (w=0)
    # This places them on the equator of S3.
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    ico_norms = np.linalg.norm(ico, axis=1, keepdims=True)
    ico = ico / ico_norms
    
    # Add 2 more points, e.g., (1,0,0) and (-1,0,0)
    p13 = np.array([1.0, 0.0, 0.0])
    p14 = np.array([-1.0, 0.0, 0.0])
    
    vec_parts = np.vstack([ico, p13, p14])
    
    # Create quaternions: w=0, vec=vec_parts
    # q = [w, x, y, z]
    quats = np.zeros((n, 4))
    quats[:, 1:] = vec_parts
    # Normalize quats (they are already unit if vec_parts are unit, w=0)
    quats = quats / np.linalg.norm(quats, axis=1, keepdims=True)
    
    # Optimization
    def get_points(quats):
        # Map quaternion to point in R3
        # Use vector part, normalized
        v = quats[:, 1:]
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        # Handle case where vector part is zero (w=+-1)
        # If norm is 0, project to (1,0,0) or use w to decide?
        # For now, add small epsilon
        norms = np.where(norms < 1e-9, 1e-9, norms)
        return v / norms

    def objective(quats):
        pts = get_points(quats)
        d = pdist(pts)
        d_min = np.min(d)
        d_max = np.max(d)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    # Gradient estimation using finite differences
    # We will optimize the log of the objective to improve conditioning?
    # Or just the objective.
    
    eta = 0.05 # Initial step size
    best_score = objective(quats)
    quats_best = quats.copy()
    
    for step in range(2000):
        # Compute gradient
        grad = np.zeros_like(quats)
        eps = 1e-4
        
        for i in range(n):
            for j in range(4):
                # Central difference
                q_plus = quats.copy()
                q_plus[i, j] += eps
                q_plus[i] = q_plus[i] / np.linalg.norm(q_plus[i])
                
                q_minus = quats.copy()
                q_minus[i, j] -= eps
                q_minus[i] = q_minus[i] / np.linalg.norm(q_minus[i])
                
                f_plus = objective(q_plus)
                f_minus = objective(q_minus)
                
                grad[i, j] = (f_plus - f_minus) / (2 * eps)
        
        # Project gradient to tangent space of S3 for each quaternion
        # Tangent space at q is {v | v . q = 0}
        # g_tan = g - (g . q) q
        for i in range(n):
            dot = np.dot(grad[i], quats[i])
            grad[i] = grad[i] - dot * quats[i]
        
        # Update quaternions
        quats_new = quats + eta * grad
        quats_new = quats_new / np.linalg.norm(quats_new, axis=1, keepdims=True)
        
        new_score = objective(quats_new)
        
        if new_score > best_score:
            best_score = new_score
            quats_best = quats_new.copy()
            quats = quats_new
            # Maybe increase eta slightly?
            # eta *= 1.01
        else:
            # Decrease eta
            eta *= 0.9
            # If eta is too small, break
            if eta < 1e-6:
                break
        
    return get_points(quats_best)
