import numpy as np
import math

N = 12

def construct():
    # Generate 3x4 grid of hexagons
    # Lattice vectors
    u = np.array([math.sqrt(3), 0.0])
    v = np.array([math.sqrt(3)/2, 1.5])
    
    centers = []
    # i from -1 to 1, j from -1.5 to 1.5 step 1
    for i in [-1, 0, 1]:
        for j in [-1.5, -0.5, 0.5, 1.5]:
            c = i * u + j * v
            centers.append(c)
            
    centers = np.array(centers)
    
    # We want to find the best global rotation theta
    # to minimize the required outer hexagon side S
    
    def calc_S(theta):
        # Rotate centers
        c_theta = np.array([
            [math.cos(theta), -math.sin(theta)],
            [math.sin(theta), math.cos(theta)]
        ]) @ centers.T
        c_theta = c_theta.T
        
        # Inner hexagons are rotated by theta as well (originally 0)
        # Container normals are fixed at 30, 90, ...
        # We need to check all 6 normals
        
        max_extent = 0.0
        # Normals angles in radians
        normals = [math.radians(30 + 60*k) for k in range(6)]
        
        for alpha in normals:
            # Projection of centers
            proj_centers = c_theta[:, 0] * math.cos(alpha) + c_theta[:, 1] * math.sin(alpha)
            max_center_proj = np.max(proj_centers)
            
            # Support function of inner hexagon
            # Inner hexagon angle is theta.
            # Relative angle beta = alpha - theta
            beta = alpha - theta
            # Normalize beta to [0, 2*pi)
            beta = beta % (2 * math.pi)
            # Find distance to nearest multiple of pi/3
            # The support function is cos(dist)
            # dist = min(|beta - k*pi/3|)
            # Since cos is even and periodic, we can map beta to [0, pi/6]
            # Actually, the vertices are at 0, 60, 120...
            # The max projection is cos(delta) where delta is angle diff to nearest vertex.
            # Nearest vertex angle is round(beta / (pi/3)) * (pi/3)
            k_nearest = round(beta / (math.pi / 3))
            angle_diff = abs(beta - k_nearest * (math.pi / 3))
            # Due to periodicity and symmetry, angle_diff is in [0, pi/6]
            h_inner = math.cos(angle_diff)
            
            extent = max_center_proj + h_inner
            if extent > max_extent:
                max_extent = extent
                
        # S = 2 * max_extent / sqrt(3)
        return 2.0 * max_extent / math.sqrt(3)

    # Optimize theta
    best_S = float('inf')
    best_theta = 0.0
    # Search from 0 to 60 degrees
    for t in np.linspace(0, math.pi/3, 600):
        S = calc_S(t)
        if S < best_S:
            best_S = S
            best_theta = t
            
    # Construct return values
    inner = []
    rot_mat = np.array([
        [math.cos(best_theta), -math.sin(best_theta)],
        [math.sin(best_theta), math.cos(best_theta)]
    ])
    
    for c in centers:
        c_rot = rot_mat @ c
        inner.append([c_rot[0], c_rot[1], math.degrees(best_theta)])
        
    return inner, (0.0, 0.0), best_S, 0.0

# EVOLVE-BLOCK-END
