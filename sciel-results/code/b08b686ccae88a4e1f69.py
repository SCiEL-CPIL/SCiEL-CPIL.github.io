import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize_scalar
import time

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    # Helper: Hexagon vertices
    def hexagon_vertices(cx, cy, side, angle_rad):
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    # Helper: Generate a set of inner hexagon centers
    def generate_centers(shape_type):
        d = np.sqrt(3)
        # Neighbor vectors for flat-top hexagons
        # v0: 30 deg, v1: 90, v2: 150, v3: 210, v4: 270, v5: 330
        v0 = np.array([1.5, d/2])
        v1 = np.array([0.0, d])
        v2 = np.array([-1.5, d/2])
        v3 = np.array([-1.5, -d/2])
        v4 = np.array([0.0, -d])
        v5 = np.array([1.5, -d/2])
        
        centers = []
        
        if shape_type == "2x6":
            # Row 1
            for i in range(6):
                centers.append(np.array([0.0, 0.0]) + i * v5)
            # Row 2
            for i in range(6):
                centers.append(np.array([0.0, 0.0]) + v4 + i * v5)
                
        elif shape_type == "3x4":
            # Row 1
            for i in range(4):
                centers.append(np.array([0.0, 0.0]) + i * v5)
            # Row 2
            for i in range(4):
                centers.append(np.array([0.0, 0.0]) + v4 + i * v5)
            # Row 3
            for i in range(4):
                centers.append(np.array([0.0, 0.0]) + 2*v4 + i * v5)
                
        elif shape_type == "1x12":
            for i in range(12):
                centers.append(np.array([0.0, 0.0]) + i * v5)
                
        elif shape_type == "compact":
            # Center
            centers.append(np.array([0.0, 0.0]))
            # Ring 1
            for v in [v0, v1, v2, v3, v4, v5]:
                centers.append(v)
            # Add 5 more to make 12
            # Extend the "flat" sides?
            # v1 is (0, sqrt(3)). Add above it?
            centers.append(v1 + v1) # Top
            centers.append(v1 + v0) # Top Right
            centers.append(v1 + v2) # Top Left
            centers.append(v4 + v4) # Bottom
            centers.append(v4 + v3) # Bottom Left
            
        return np.array(centers)

    # Helper: Find minimal outer hexagon side length and rotation for a set of points
    def min_outer_hexagon(points):
        # points: Nx2 array
        # Translate to centroid
        centroid = np.mean(points, axis=0)
        points_centered = points - centroid
        
        # Get convex hull
        try:
            hull = ConvexHull(points_centered)
            hull_points = points_centered[hull.vertices]
        except Exception:
            # Fallback if hull fails (shouldn't happen for 12 hexes)
            hull_points = points_centered
            
        # Objective function: S(theta)
        # S = (2/sqrt(3)) * max(p . n_k)
        # n_k are normals of the hexagon sides
        def objective(theta):
            # Normals are at theta + k*60
            angles = theta + np.radians(np.arange(0, 360, 60))
            normals = np.column_stack([np.cos(angles), np.sin(angles)])
            
            # Project hull points onto normals
            # proj[i, j] = p_i . n_j
            proj = hull_points @ normals.T
            max_proj = np.max(proj)
            side = (2.0 / np.sqrt(3)) * max_proj
            return side
            
        # Minimize over theta in [0, pi/3]
        res = minimize_scalar(objective, bounds=(0, np.pi/3), method='bounded')
        if res.success:
            theta_opt = res.x
            side_opt = res.fun
        else:
            theta_opt = 0
            side_opt = objective(0)
            
        return centroid, side_opt, theta_opt

    best_side = float('inf')
    best_inner = None
    best_center = None
    best_angle = None

    shapes = ["2x6", "3x4", "1x12", "compact"]
    
    for shape in shapes:
        if time.time() > deadline:
            break
            
        centers = generate_centers(shape)
        
        # Get all vertices of the inner hexagons to define the exact boundary
        # Inner hexagons are flat-top (angle 0), side 1
        all_points = []
        for c in centers:
            verts = hexagon_vertices(c[0], c[1], 1.0, 0.0)
            all_points.append(verts)
        all_points = np.vstack(all_points)
        
        centroid, side, theta = min_outer_hexagon(all_points)
        
        if side < best_side:
            best_side = side
            best_center = centroid
            best_angle = np.degrees(theta)
            
            # Construct inner list
            # Inner hexagons are at 'centers', angle 0
            # But we need to return them in the global frame.
            # The 'centers' were generated in a local frame.
            # The 'centroid' is the center of the outer hexagon in the local frame.
            # The return format expects:
            # inner: list of [x, y, angle]
            # center: [x, y] of outer
            # side: outer side
            # angle: outer rotation
            
            # We can keep the local frame as the global frame.
            # Inner positions are 'centers'.
            # Outer center is 'centroid'.
            # Outer angle is 'theta'.
            
            inner_list = []
            for c in centers:
                inner_list.append([float(c[0]), float(c[1]), 0.0])
                
            best_inner = inner_list
            best_center = [float(centroid[0]), float(centroid[1])]
            best_angle = float(best_angle)

    # If no shape was processed (should not happen), return a fallback
    if best_inner is None:
        # Fallback: Line packing
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    return best_inner, best_center, float(best_side), float(best_angle)

# EVOLVE-BLOCK-END
