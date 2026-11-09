import numpy as np

N = 12

def construct():
    sqrt3 = np.sqrt(3)
    
    # Hexagonal lattice basis
    v1 = np.array([sqrt3, 0.0])
    v2 = np.array([sqrt3/2, 1.5])
    
    # Generate lattice points
    pts = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            pts.append(i * v1 + j * v2)
    pts = np.array(pts)
    
    # Sort by distance from origin
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)
    
    best_S = 1e10
    best_pts = None
    best_center = np.array([0.0, 0.0])
    best_theta = 0.0
    
    # Try subsets of 12 from the top 30 closest points
    top_idx = idx[:30]
    
    for start in range(0, 30 - 11):
        subset = pts[top_idx[start:start+12]]
        
        # Try several rotations
        for theta_deg in [0, 15, 30, 45, 60]:
            theta = np.radians(theta_deg)
            normals = np.array([[np.cos(theta + k*np.pi/3), np.sin(theta + k*np.pi/3)] for k in range(6)])
            
            # Optimal center: minimize max_{i,k} |dot(p_i - c, n_k)|
            # This is the Chebyshev center in hexagonal norm
            # Approximate by trying centroid and a few nearby points
            candidates_c = [np.mean(subset, axis=0)]
            for p in subset:
                candidates_c.append(p)
            
            best_S_for_subset = 1e10
            best_c_for_subset = None
            
            for c in candidates_c:
                max_val = 0
                for p in subset:
                    for n in normals:
                        val = abs(np.dot(p - c, n))
                        if val > max_val:
                            max_val = val
                S = (max_val + 1.0) * 2.0 / sqrt3
                if S < best_S_for_subset:
                    best_S_for_subset = S
                    best_c_for_subset = c
            
            if best_S_for_subset < best_S:
                best_S = best_S_for_subset
                best_pts = subset.copy()
                best_center = best_c_for_subset.copy()
                best_theta = theta_deg
    
    # Construct inner hexagons
    inner = []
    for p in best_pts:
        inner.append([float(p[0]), float(p[1]), 0.0])
    
    outer_center = [float(best_center[0]), float(best_center[1])]
    outer_side = float(best_S)
    outer_angle = float(best_theta)
    
    return inner, outer_center, outer_side, outer_angle
