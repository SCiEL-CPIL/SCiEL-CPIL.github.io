import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s margin
    
    # 1. Generate initial cluster
    # Lattice spacing
    a = np.sqrt(3)
    centers = []
    # Center
    centers.append([0.0, 0.0])
    # Ring 1
    for i in range(6):
        angle = np.pi/3 * i
        centers.append([a * np.cos(angle), a * np.sin(angle)])
    # Ring 2 - select 4
    # Indices in full ring 2 (12 points)
    # We want to pick 4 that are symmetric.
    # Let's pick the ones at angles 0, pi/2, pi, 3pi/2 relative to lattice?
    # In lattice, Ring 2 points are at distance 2a from center? No, distance is 2 * (a/2 * 2)?
    # Distance from center to Ring 1 is a.
    # Distance from center to Ring 2 is 2a?
    # Let's just generate Ring 2 points and pick 4.
    ring2 = []
    for i in range(12):
        angle = np.pi/6 * i + np.pi/6 # Offset?
        # Actually, let's just use the lattice coordinates.
        pass
    
    # Simpler: Hardcode the 11 points based on the lattice.
    # (0,0)
    # (a, 0), (-a, 0), (0, 1.5), (0, -1.5), (a/2, 1.5), (-a/2, 1.5) -> Wait, (a/2, 1.5) is not in Ring 1.
    # Ring 1 neighbors of (0,0) in hex lattice:
    # (a, 0), (a/2, 1.5), (-a/2, 1.5), (-a, 0), (-a/2, -1.5), (a/2, -1.5)
    # Let's verify distances.
    # dist((0,0), (a,0)) = a = sqrt(3). Correct.
    # dist((0,0), (a/2, 1.5)) = sqrt(a^2/4 + 2.25) = sqrt(3/4 + 9/4) = sqrt(12/4) = sqrt(3). Correct.
    
    centers = [
        [0.0, 0.0],
        [a, 0.0], [a/2, 1.5], [-a/2, 1.5], [-a, 0.0], [-a/2, -1.5], [a/2, -1.5]
    ]
    # Add 4 from Ring 2
    # Ring 2 points are further out.
    # e.g., (2a, 0), (3a/2, 1.5), (a/2, 3.0), (-a/2, 3.0), ...
    # Let's pick (2a, 0), (-2a, 0), (a/2, 3.0), (-a/2, 3.0)
    centers.append([2*a, 0.0])
    centers.append([-2*a, 0.0])
    centers.append([a/2, 3.0])
    centers.append([-a/2, 3.0])
    
    # This shape is not perfectly symmetric (top is wider than bottom).
    # Let's try a more symmetric selection.
    # Pick (2a, 0), (-2a, 0), (a/2, -3.0), (-a/2, -3.0)? No.
    # Let's just use the 7 central + 4 arbitrary from the next shell and let the optimizer find the best container.
    # Actually, the shape matters. A "diamond" shape might be better for a rotated container.
    
    # Let's generate the vertices for these centers.
    # Unit hexagon vertices (angle 0):
    # (1, 0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1, 0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
    hex_verts = np.array([
        [1.0, 0.0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1.0, 0.0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])
    
    all_points = []
    for c in centers:
        pts = hex_verts + np.array(c)
        all_points.append(pts)
    all_points = np.vstack(all_points)
    
    def objective(params):
        cx, cy, theta = params
        # Normals
        angles = theta + np.pi/3 * np.arange(6)
        normals = np.column_stack((np.cos(angles), np.sin(angles)))
        
        # (p - c) . n
        # all_points shape (66, 2)
        # normals shape (6, 2)
        # We need max over all points and all normals.
        # dist = (all_points - [cx, cy]) @ normals.T  -> shape (66, 6)
        dists = (all_points - np.array([cx, cy])) @ normals.T
        max_dist = np.max(dists)
        if max_dist < 0:
             return 0.0 # Should not happen
        side = max_dist * 2.0 / np.sqrt(3)
        return side
    
    best_side = float('inf')
    best_params = None
    
    # Try a few initial angles
    for init_theta in [0.0, np.pi/6, np.pi/3]:
        x0 = [0.0, 0.0, init_theta]
        try:
            res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6})
            if res.success and res.fun < best_side:
                best_side = res.fun
                best_params = res.x
        except Exception:
            pass
            
        if time.time() > deadline:
            break
            
    if best_params is None:
        # Fallback
        best_side = 40.0
        best_params = [0.0, 0.0, 0.0]
        
    cx, cy, theta = best_params
    # Construct inner list
    inner = []
    for c in centers:
        inner.append([c[0], c[1], 0.0]) # Inner hexagons are axis-aligned (0 deg)
        
    return inner, (cx, cy), best_side, np.degrees(theta)

