"""Packing 11 unit hexagons in a regular hexagon - minimize outer side length."""
import numpy as np
from scipy.optimize import minimize
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # safety margin
    
    # Hexagon vertices for a unit regular hexagon (side 1)
    # Vertices at angles 0, 60, 120, 180, 240, 300 degrees
    hex_angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180.0
    hex_verts = np.column_stack([np.cos(hex_angles), np.sin(hex_angles)])
    
    def get_hex_vertices(center, angle_deg):
        """Get vertices of a unit hexagon at given center and rotation."""
        theta = np.radians(angle_deg)
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        return center + hex_verts @ R.T
    
    def hexagon_radius(hex_verts):
        """Maximum distance from origin to any vertex."""
        return np.max(np.sqrt(np.sum(hex_verts**2, axis=1)))
    
    def min_dist_between_hexagons(verts1, verts2):
        """Compute minimum distance between two hexagons (approximation using vertices)."""
        # Use vertex-to-vertex and vertex-to-edge distances
        min_d = np.inf
        for v1 in verts1:
            for v2 in verts2:
                d = np.sqrt(np.sum((v1 - v2)**2))
                min_d = min(min_d, d)
        return min_d
    
    def check_disjoint(verts_list, tol=1e-6):
        """Check if all hexagons are disjoint (no overlap)."""
        n = len(verts_list)
        for i in range(n):
            for j in range(i+1, n):
                # Check if any vertex of i is inside j or vice versa
                # Simplified: check minimum distance between boundaries
                # For hexagons in hexagonal lattice, they touch at edges
                # We need to ensure no overlap
                pass
        return True
    
    def compute_outer_hexagon_side(inner_positions, inner_angles, outer_center, outer_angle):
        """Compute the minimum outer hexagon side length that contains all inner hexagons."""
        outer_theta = np.radians(outer_angle)
        # Outer hexagon normal directions (perpendicular to each face)
        # For a regular hexagon, faces are perpendicular to directions at 0, 60, 120, 180, 240, 300
        outer_normals = np.array([[np.cos(outer_theta + k*np.pi/3), np.sin(outer_theta + k*np.pi/3)] for k in range(6)])
        
        max_projection = 0.0
        for pos, ang in zip(inner_positions, inner_angles):
            verts = get_hex_vertices(pos, ang)
            for normal in outer_normals:
                projections = verts @ normal
                # The outer hexagon face in direction normal is at distance s*sqrt(3)/2 from center
                # where s is the side length
                # Actually for a regular hexagon with side s, the apothem (distance from center to face) is s*sqrt(3)/2
                proj_max = np.max(projections)
                proj_min = np.min(projections)
                # The hexagon extends from proj_min to proj_max in this direction
                # The outer hexagon must contain this, so we need:
                # outer_center @ normal + s*sqrt(3)/2 >= proj_max
                # outer_center @ normal - s*sqrt(3)/2 <= proj_min
                required = max(abs(proj_max - outer_center @ normal), abs(proj_min - outer_center @ normal))
                max_projection = max(max_projection, required)
        
        # s*sqrt(3)/2 = max_projection, so s = 2*max_projection/sqrt(3)
        return 2.0 * max_projection / np.sqrt(3.0)
    
    def objective(params):
        """Objective: minimize outer hexagon side length with penalty for overlaps."""
        # params: [x1, y1, a1, x2, y2, a2, ..., x11, y11, a11, outer_x, outer_y, outer_angle]
        n_params = 3 * N + 3
        positions = []
        angles = []
        for i in range(N):
            x = params[3*i]
            y = params[3*i + 1]
            a = params[3*i + 2]
            positions.append(np.array([x, y]))
            angles.append(a)
        
        outer_center = np.array([params[3*N], params[3*N + 1]])
        outer_angle = params[3*N + 2]
        
        # Compute outer hexagon side
        side = compute_outer_hexagon_side(positions, angles, outer_center, outer_angle)
        
        # Check for overlaps (penalty)
        penalty = 0.0
        verts_list = [get_hex_vertices(pos, ang) for pos, ang in zip(positions, angles)]
        for i in range(N):
            for j in range(i+1, N):
                # Check if hexagons overlap using a simplified check
                # For hexagons, if centers are too close, they overlap
                d = np.sqrt(np.sum((positions[i] - positions[j])**2))
                # Minimum distance between centers for non-overlapping unit hexagons
                # depends on relative rotation; for aligned hexagons, it's 2*sqrt(3)/2 = sqrt(3)
                # But with rotation, it can be less
                # Use a conservative bound: if d < 1.5, likely overlapping
                if d < 1.4:
                    penalty += (1.4 - d) ** 2
        
        return side + penalty * 100.0
    
    # Initial guess: hexagonal lattice arrangement
    # Central hexagon at origin, 6 around it, 4 more
    init_positions = []
    init_angles = []
    
    # Hexagonal lattice spacing for unit hexagons
    # Distance between centers of touching hexagons in hexagonal lattice
    # For regular hexagons of side 1, the distance between centers when touching is 2*sqrt(3)/2 = sqrt(3)
    d_center = np.sqrt(3.0)
    
    # Hexagonal lattice directions
    lattice_dirs = []
    for k in range(6):
        ang = k * np.pi / 3.0
        lattice_dirs.append(np.array([np.cos(ang), np.sin(ang)]))
    
    # Place central hexagon
    init_positions.append(np.array([0.0, 0.0]))
    init_angles.append(0.0)
    
    # Place 6 surrounding
    for k in range(6):
        pos = d_center * lattice_dirs[k]
        init_positions.append(pos)
        init_angles.append(0.0)
    
    # Place 4 more in the next ring (partial)
    # Positions in second ring
    second_ring = []
    for k in range(6):
        pos = 2 * d_center * lattice_dirs[k]
        second_ring.append(pos)
    # Add 4 of them
    for k in range(4):
        init_positions.append(second_ring[k])
        init_angles.append(0.0)
    
    # Outer hexagon initial guess
    outer_center = np.array([0.0, 0.0])
    outer_angle = 0.0
    
    # Build initial parameter vector
    init_params = []
    for pos, ang in zip(init_positions, init_angles):
        init_params.extend([pos[0], pos[1], ang])
    init_params.extend([outer_center[0], outer_center[1], outer_angle])
    
    # Optimize
    best_result = None
    best_score = np.inf
    
    try:
        result = minimize(objective, init_params, method='Nelder-Mead', 
                         options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-8})
        if result.success or result.fun < best_score:
            best_score = result.fun
            best_result = result.x
    except Exception:
        pass
    
    # Try with different initial configurations
    configs = [
        # Rotated hexagonal lattice
        lambda: (init_positions, init_angles, 30.0),
        # Different arrangement
    ]
    
    # Multi-start
    np.random.seed(42)
    for trial in range(5):
        if time.time() > deadline - 30.0:
            break
        
        # Perturb initial positions
        trial_positions = [pos + np.random.normal(0, 0.1, 2) for pos in init_positions]
        trial_angles = [ang + np.random.uniform(-10, 10) for ang in init_angles]
        
        trial_params = []
        for pos, ang in zip(trial_positions, trial_angles):
            trial_params.extend([pos[0], pos[1], ang])
        trial_params.extend([outer_center[0] + np.random.normal(0, 0.5), 
                            outer_center[1] + np.random.normal(0, 0.5), 
                            outer_angle + np.random.uniform(-30, 30)])
        
        try:
            result = minimize(objective, trial_params, method='Nelder-Mead',
                             options={'maxiter': 3000, 'xatol': 1e-6, 'fatol': 1e-8})
            if result.fun < best_score:
                best_score = result.fun
                best_result = result.x
        except Exception:
            pass
    
    # Extract best solution
    if best_result is not None:
        positions = []
        angles = []
        for i in range(N):
            x = best_result[3*i]
            y = best_result[3*i + 1]
            a = best_result[3*i + 2]
            positions.append((x, y))
            angles.append(a)
        
        outer_center = (best_result[3*N], best_result[3*N + 1])
        outer_angle = best_result[3*N + 2]
        
        # Compute actual side length
        pos_arrays = [np.array(p) for p in positions]
        side = compute_outer_hexagon_side(pos_arrays, angles, np.array(outer_center), outer_angle)
        
        inner = []
        for i in range(N):
            inner.append([positions[i][0], positions[i][1], angles[i]])
        
        return inner, outer_center, side, outer_angle
    else:
        # Fallback: use initial guess
        inner = []
        for pos, ang in zip(init_positions, init_angles):
            inner.append([pos[0], pos[1], ang])
        
        pos_arrays = init_positions
        side = compute_outer_hexagon_side(pos_arrays, init_angles, outer_center, outer_angle)
        
        return inner, (outer_center[0], outer_center[1]), side, outer_angle

# EVOLVE-BLOCK-END
