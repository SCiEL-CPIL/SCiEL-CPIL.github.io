"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import linprog
import time
import math
from itertools import product

N = 11

# EVOLVE-BLOCK-START

def _hex_vertices(angle_deg):
    """Return 6 vertices of a unit regular hexagon rotated by angle_deg degrees, centered at origin."""
    a = math.radians(angle_deg)
    return np.array([[math.cos(a + i * math.pi / 3), math.sin(a + i * math.pi / 3)] for i in range(6)])

def _support_function(vertices, direction):
    """Max dot product of vertices with direction vector."""
    return np.max(vertices @ direction)

def _min_outer_hex_side(inner_hexes, center, angle_outer_deg):
    """
    Given inner hexagons (list of (pos, angle_deg)), outer center and outer angle,
    find minimum outer side length s such that all inner hexagons fit inside.
    
    For a regular hexagon with center c, side s, rotation phi, the support in direction u
    is s * sqrt(3) / 2 * max over 6 face normals... Actually for a regular hexagon of side s,
    the support function in direction u (unit vector) is s * sqrt(3)/2 * max_i (u . n_i)
    where n_i are the outward normals of the 6 faces.
    
    But we need: for each inner hexagon vertex v, v must be inside the outer hexagon.
    A point p is inside a regular hexagon (center c, side s, rotation phi) iff
    for all 6 face directions, (p - c) . n_i <= s * sqrt(3)/2
    
    where n_i are unit normals to the 6 faces.
    
    So we need: s >= max over all inner vertices v, over all 6 face normals n_i:
        (v - c) . n_i / (sqrt(3)/2)
    
    But also need to handle the case where (v-c).n_i could be negative (vertex on wrong side),
    in which case no constraint from that face.
    
    So s_min = max(0, max_{v, i} ((v - c) . n_i) / (sqrt(3)/2))
    where n_i are the 6 outward unit normals of the outer hexagon.
    """
    phi = math.radians(angle_outer_deg)
    # The 6 outward face normals of a regular hexagon with rotation phi
    # A regular hexagon with "pointy" orientation (vertex at angle 0) has faces
    # with normals at angles 0, 60, 120, 180, 240, 300 degrees.
    # Actually, let's think carefully. A regular hexagon with side s, center at origin,
    # with one vertex at (s, 0) (pointy-top in x-direction), has vertices at:
    # (s*cos(k*60), s*sin(k*60)) for k=0..5
    # The faces are between consecutive vertices. The outward normal of the face
    # between vertex k and vertex k+1 is at angle (k+0.5)*60 degrees.
    # So face normals are at 30, 90, 150, 210, 270, 330 degrees (for phi=0).
    # With rotation phi, they are at 30+phi, 90+phi, etc.
    
    normals = []
    for k in range(6):
        theta = math.radians(30 + k * 60 + angle_outer_deg)
        normals.append(np.array([math.cos(theta), math.sin(theta)]))
    normals = np.array(normals)  # shape (6, 2)
    
    c = np.array(center)
    max_constraint = 0.0
    for pos, ang in inner_hexes:
        verts = _hex_vertices(ang) + pos
        # For each vertex, check against each face normal
        diffs = verts - c  # shape (6, 2)
        projections = diffs @ normals.T  # shape (6, 6)
        max_proj = np.max(projections)
        if max_proj > max_constraint:
            max_constraint = max_proj
    
    s_min = max_constraint / (math.sqrt(3) / 2)
    return max(s_min, 0.0)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # We'll try to pack 11 unit hexagons inside a regular hexagon.
    # Strategy: use a hexagonal grid arrangement, then optimize positions and rotations.
    
    # Initial arrangement: hexagonal cluster of 19 hexagons (radius 3), take 11 of them
    # Actually, let's try a simpler approach: linear row, then optimize.
    
    # Let's use scipy to optimize. We'll parameterize each hexagon by (x, y, angle).
    # Objective: minimize outer hexagon side length.
    
    # For the outer hexagon, we can fix its center and rotation to reduce parameters.
    # By symmetry, we can fix outer center at (0,0) and try a few outer rotations.
    
    best_result = None
    best_score = float('inf')
    
    # Try several outer rotations
    outer_rotations = [0.0, 30.0, 15.0, 45.0]
    
    for outer_rot in outer_rotations:
        if time.time() > deadline:
            break
        
        # Start with a hexagonal grid arrangement
        # Unit hexagons in a hex grid: adjacent hexagons center distance = 2 * (sqrt(3)/2) = sqrt(3)
        # For "pointy" hexagons (flat-top), the distance between centers of adjacent hexagons
        # in the 3 directions is: sqrt(3) in x-direction, and sqrt(3)/2 in x, 3/2 in y (or similar)
        
        # Let's use axial coordinates. For a regular hexagon of side 1, the center-to-center
        # distance to a neighbor is 2 * apothem = 2 * (sqrt(3)/2) = sqrt(3).
        
        # Generate 11 positions in a hexagonal pattern
        # Hex grid neighbors in axial coords (q, r): (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)
        # Convert axial to Cartesian: x = sqrt(3) * (q + r/2), y = 3/2 * r
        
        hex_positions_axial = [(0, 0), (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), 
                               (2, -1), (2, -2), (1, -2), (-1, -1), (-2, 0)]
        
        hex_positions = []
        for q, r in hex_positions_axial:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            hex_positions.append((x, y))
        
        # Initial angles: all 0
        initial_angles = [0.0] * N
        
        # Parameters: x_i, y_i, angle_i for i=0..N-1, plus outer center x, y
        # We fix outer rotation to outer_rot
        
        n_params = 3 * N + 2  # x, y, angle for each hexagon + outer center x, y
        
        def objective_and_grad(params):
            # Unpack
            outer_cx = params[0]
            outer_cy = params[1]
            hex_params = params[2:]
            
            inner_hexes = []
            for i in range(N):
                hx = hex_params[3*i]
                hy = hex_params[3*i + 1]
                ha = hex_params[3*i + 2] * 180.0 / math.pi  # convert to degrees for _hex_vertices
                inner_hexes.append((np.array([hx, hy]), ha))
            
            s = _min_outer_hex_side(inner_hexes, (outer_cx, outer_cy), outer_rot)
            return s
        
        # Use Nelder-Mead optimization
        from scipy.optimize import minimize
        
        # Multiple starting points
        best_local = float('inf')
        best_params = None
        
        n_starts = 3
        for si in range(n_starts):
            if time.time() > deadline:
                break
            
            # Perturb initial positions
            scale = 0.5 if si == 0 else (0.5 + 0.3 * si)
            rng = np.random.RandomState(42 + si)
            params_init = np.zeros(n_params)
            params_init[0] = 0.0
            params_init[1] = 0.0
            for i in range(N):
                params_init[2 + 3*i] = hex_positions[i][0] + rng.randn() * scale
                params_init[2 + 3*i + 1] = hex_positions[i][1] + rng.randn() * scale
                params_init[2 + 3*i + 2] = (rng.rand() - 0.5) * math.pi / 6  # small random angle
        
            res = minimize(objective_and_grad, params_init, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
            
            if res.fun < best_local:
                best_local = res.fun
                best_params = res.x.copy()
        
        if best_params is not None and best_local < best_score:
            best_score = best_local
            # Extract result
            outer_cx = best_params[0]
            outer_cy = best_params[1]
            inner = []
            for i in range(N):
                hx = best_params[2 + 3*i]
                hy = best_params[2 + 3*i + 1]
                ha = best_params[2 + 3*i + 2] * 180.0 / math.pi
                inner.append([hx, hy, ha])
            best_result = (inner, (outer_cx, outer_cy), best_score, outer_rot)
    
    # If no optimization result, fall back to the simple row arrangement
    if best_result is None:
        inner = [[math.sqrt(3) * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        # Compute the actual outer side needed
        outer_c = (0.0, 0.0)
        outer_rot = 0.0
        s = _min_outer_hex_side([(np.array([x, y]), a) for x, y, a in inner], outer_c, outer_rot)
        best_result = (inner, outer_c, s, outer_rot)
    
    return best_result

# EVOLVE-BLOCK-END
