import numpy as np
import time
from scipy.optimize import minimize_scalar

def construct():
    start_time = time.time()
    deadline = 145.0

    # Centers of the 11 hexagons in a "flower" pattern
    # 1 center + 6 at distance sqrt(3) + 4 at distance 3
    eps = 1e-10
    centers = np.array([[0.0, 0.0]])
    
    r1 = np.sqrt(3) + eps
    for k in range(6):
        angle = np.deg2rad(k * 60)
        centers = np.vstack([centers, [r1 * np.cos(angle), r1 * np.sin(angle)]])
    
    r2 = 3.0 + 2 * eps
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        centers = np.vstack([centers, [r2 * np.cos(angle), r2 * np.sin(angle)]])
    
    n_hex = len(centers)
    
    # Base vertices of unit hexagon (pointy-right orientation, angle 0)
    base_vertices = np.array([
        [np.cos(np.deg2rad(60 * k)), np.sin(np.deg2rad(60 * k))] for k in range(6)
    ])

    def compute_min_side_for_config(phis):
        """
        Computes the minimum outer hexagon side length for a given set of inner rotations.
        Optimizes the outer rotation theta.
        """
        # Precompute vertices for all hexagons with given rotations
        all_vertices = []
        for i in range(n_hex):
            phi = phis[i]
            c, s = np.cos(phi), np.sin(phi)
            rot = np.array([[c, -s], [s, c]])
            verts = base_vertices @ rot.T + centers[i]
            all_vertices.append(verts)
        
        all_vertices = np.vstack(all_vertices) # Shape (66, 2)

        def side_length(theta):
            # Outer hexagon face normals: at angles 30 + k*60 + theta
            normals = np.array([
                [np.cos(np.deg2rad(30 + k * 60) + theta),
                 np.sin(np.deg2rad(30 + k * 60) + theta)]
                for k in range(6)
            ])
            
            # Project all vertices onto all normals
            # dots shape: (66, 6)
            dots = all_vertices @ normals.T
            max_proj = np.max(dots)
            
            # Side length S = 2 * apothem = 2 * max_proj / sqrt(3)
            return 2.0 * max_proj / np.sqrt(3)

        # Optimize theta in [0, pi/3]
        res = minimize_scalar(side_length, bounds=(0, np.pi/3), method='bounded', options={'xatol': 1e-10})
        return res.fun, res.x

    best_side = float('inf')
    best_theta = 0.0
    best_phis = None

    # Try configuration 1: All inner hexagons aligned with outer (phi=0)
    phis_0 = np.zeros(n_hex)
    side_0, theta_0 = compute_min_side_for_config(phis_0)
    if side_0 < best_side:
        best_side = side_0
        best_theta = theta_0
        best_phis = phis_0

    # Try configuration 2: All inner hexagons rotated by 30 deg (phi=pi/6)
    if time.time() < deadline - 10:
        phis_30 = np.full(n_hex, np.pi/6)
        side_30, theta_30 = compute_min_side_for_config(phis_30)
        if side_30 < best_side:
            best_side = side_30
            best_theta = theta_30
            best_phis = phis_30

    # Construct output
    inner = []
    for i in range(n_hex):
        inner.append([centers[i][0], centers[i][1], np.rad2deg(best_phis[i])])
    
    # Add tiny margin for floating point safety
    side = best_side + 1e-9
    outer_angle = np.rad2deg(best_theta)
    
    return inner, (0.0, 0.0), side, outer_angle

if __name__ == "__main__":
    construct()
