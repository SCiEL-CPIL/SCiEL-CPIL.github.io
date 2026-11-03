import numpy as np
import scipy.spatial
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # 1. Initialization: Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    P = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    def covering_radius(points):
        # points: (N, 3) unit vectors
        try:
            hull = scipy.spatial.ConvexHull(points)
            faces = hull.simplices
            max_radius = 0.0
            for face in faces:
                A, B, C = points[face[0]], points[face[1]], points[face[2]]
                # Solve for circumcenter
                # K c = [1,1,1]
                # K is Gram matrix
                K = np.array([
                    [np.dot(A,A), np.dot(A,B), np.dot(A,C)],
                    [np.dot(B,A), np.dot(B,B), np.dot(B,C)],
                    [np.dot(C,A), np.dot(C,B), np.dot(C,C)]
                ])
                try:
                    c = np.linalg.solve(K, np.ones(3))
                except np.linalg.LinAlgError:
                    continue
                U = c[0]*A + c[1]*B + c[2]*C
                norm_U = np.linalg.norm(U)
                if norm_U < 1e-10:
                    continue
                U = U / norm_U
                # Angular distance
                cos_theta = np.dot(U, A)
                cos_theta = np.clip(cos_theta, -1.0, 1.0)
                theta = np.arccos(cos_theta)
                if theta > max_radius:
                    max_radius = theta
            return max_radius
        except Exception:
            return np.pi # Invalid hull, return large value
    
    best_P = P.copy()
    best_R = covering_radius(best_P)
    
    step_size = 0.1
    min_step = 1e-5
    
    # Local Search
    while time.time() < deadline - 5.0:
        improved = False
        # Randomize order of points to avoid bias
        indices = np.random.permutation(N)
        
        for i in indices:
            if time.time() > deadline - 5.0:
                break
            
            # Generate random tangent vector
            # Pick random vector, project to tangent plane
            v = np.random.randn(3)
            v -= np.dot(v, best_P[i]) * best_P[i]
            norm_v = np.linalg.norm(v)
            if norm_v < 1e-10:
                continue
            v = v / norm_v
            
            # Try move
            P_new = best_P.copy()
            P_new[i] += step_size * v
            P_new[i] /= np.linalg.norm(P_new[i])
            
            R_new = covering_radius(P_new)
            
            if R_new < best_R:
                best_P = P_new
                best_R = R_new
                improved = True
                # If improved, try to continue in same direction or randomize?
                # Just continue loop, next iteration will pick new random dir
                # We can also try to do a line search here, but keep it simple
        
        if not improved:
            step_size *= 0.9
            if step_size < min_step:
                break
            # Occasionally increase step size to escape local minima?
            # For now, just decrease.
        
        # Every few iterations, try a larger random jump?
        # Keep it simple for now.
        
    return best_P

# EVOLVE-BLOCK-END
