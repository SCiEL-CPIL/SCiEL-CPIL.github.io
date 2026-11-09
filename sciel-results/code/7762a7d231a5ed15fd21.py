# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def _hex_vertices(cx, cy, s, a_deg):
    a = np.radians(a_deg)
    angles = a + np.radians(60.0) * np.arange(6)
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def _min_gap_hex(hex1, hex2):
    """Compute minimum separation between two unit hexagons using 12 edge normals."""
    v1 = _hex_vertices(hex1[0], hex1[1], 1.0, hex1[2])
    v2 = _hex_vertices(hex2[0], hex2[1], 1.0, hex2[2])
    
    normals = []
    for v in [v1, v2]:
        for i in range(6):
            p1 = v[i]
            p2 = v[(i+1) % 6]
            edge = p2 - p1
            n = np.array([-edge[1], edge[0]])
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-12:
                continue
            n = n / n_norm
            normals.append(n)
    
    min_gap = np.inf
    for n in normals:
        proj1 = v1 @ n
        proj2 = v2 @ n
        gap_forward = proj2.min() - proj1.max()
        gap_backward = proj1.min() - proj2.max()
        sep = max(gap_forward, gap_backward)
        if sep < min_gap:
            min_gap = sep
    
    return min_gap

def _required_side_for_point(x, y):
    """Compute the minimum side length of a regular hexagon centered at origin
    with angle 0 that contains point (x,y)."""
    # Hexagon with angle 0 has vertices at angles 0, 60, 120, 180, 240, 300
    # Edges are between consecutive vertices
    # The hexagon can be described by 6 half-planes
    # For a hexagon with side s and angle 0:
    # Vertex i: (s*cos(60*i), s*sin(60*i))
    # Edge from vertex i to i+1 has normal direction
    
    # The 6 supporting lines of the hexagon (with outward normals):
    # The distance from origin to each edge is s * sqrt(3) / 2 (apothem)
    # But we need to check if point is inside
    
    # Actually, let's compute directly: for each edge, check if point is on the inside
    # A point (x,y) is inside hexagon of side s (angle 0) iff:
    # For each of the 6 edges, the point is on the correct side
    
    # Edge normals for hexagon with angle 0:
    # The hexagon has 6 edges. Let's compute the constraint for each.
    # Vertex 0: (s, 0), Vertex 1: (s/2, s*sqrt(3)/2)
    # Edge 0-1: from (s,0) to (s/2, s*sqrt(3)/2)
    # Direction: (-s/2, s*sqrt(3)/2), Normal: (-s*sqrt(3)/2, -s/2) or (sqrt(3)/2, 1/2) normalized
    
    # The 6 outward normals for a hexagon with angle 0 are at angles 30, 90, 150, 210, 270, 330 degrees
    # Each edge is at distance s*sqrt(3)/2 from center along the normal direction
    
    # For a point (x,y) to be inside:
    # x * cos(theta) + y * sin(theta) <= s * sqrt(3) / 2 for each normal angle theta
    
    thetas = np.radians(30 + 60 * np.arange(6))
    apothem = s * np.sqrt(3) / 2 if s > 0 else 1.0
    # We want the minimum s such that all constraints are satisfied
    # x*cos(t) + y*sin(t) <= s * sqrt(3)/2 for all t
    # So s >= 2 * max_i(x*cos(t_i) + y*sin(t_i)) / sqrt(3)
    
    projections = x * np.cos(thetas) + y * np.sin(thetas)
    max_proj = np.max(projections)
    if max_proj <= 0:
        return 0.0
    return 2.0 * max_proj / np.sqrt(3)

def _required_side_for_hexagon(cx, cy, a_deg):
    """Compute minimum side length of outer hexagon (centered at origin, angle 0)
    to contain the inner hexagon."""
    verts = _hex_vertices(cx, cy, 1.0, a_deg)
    max_req = 0.0
    for v in verts:
        req = _required_side_for_point(v[0], v[1])
        if req > max_req:
            max_req = req
    return max_req

def _objective(params):
    """Objective: minimize outer hexagon side length with penalties for overlaps and containment."""
    inner_params = params[:33].reshape(11, 3)
    side = params[33]
    
    # Compute required side to contain all inner hexagons
    required_side = 0.0
    for i in range(11):
        cx, cy, a_deg = inner_params[i]
        req = _required_side_for_hexagon(cx, cy, a_deg)
        if req > required_side:
            required_side = req
    
    # Penalty for overlapping inner hexagons
    overlap_penalty = 0.0
    for i in range(11):
        for j in range(i+1, 11):
            gap = _min_gap_hex(inner_params[i], inner_params[j])
            if gap < 1e-6:
                overlap_penalty += (1e-6 - gap) ** 2
    
    # Penalty if side < required_side
    containment_penalty = 0.0
    if side < required_side:
        containment_penalty = (required_side - side) ** 2 * 100
    
    return side + overlap_penalty + containment_penalty

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees)
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree)
        outer_hex_side_length: float
    """
    sqrt3 = np.sqrt(3)
    
    # Better initial guess: Start with a valid 7-hex flower configuration
    # Center at (0,0), 6 surrounding at distance sqrt(3)
    # Then add 4 more in the "corner" positions of a larger hexagonal grid
    
    inner_init = np.array([
        [0.0, 0.0, 0.0],           # Center
        [sqrt3, 0.0, 0.0],          # Right
        [sqrt3/2, 1.5, 0.0],        # Top-right
        [-sqrt3/2, 1.5, 0.0],       # Top-left
        [-sqrt3, 0.0, 0.0],         # Left
        [-sqrt3/2, -1.5, 0.0],      # Bottom-left
        [sqrt3/2, -1.5, 0.0],       # Bottom-right
        # Add 4 more in the "extended" positions
        [sqrt3*1.5, sqrt3/2, 0.0],  # Far top-right
        [0.0, sqrt3*1.5, 0.0],      # Far top
        [-sqrt3*1.5, sqrt3/2, 0.0], # Far top-left
        [sqrt3*1.5, -sqrt3/2, 0.0]  # Far bottom-right
    ])
    
    # Check initial configuration for overlaps and compute required side
    required_side = 0.0
    for i in range(11):
        req = _required_side_for_hexagon(inner_init[i, 0], inner_init[i, 1], inner_init[i, 2])
        if req > required_side:
            required_side = req
    
    # Verify no overlaps in initial config
    has_overlap = False
    for i in range(11):
        for j in range(i+1, 11):
            gap = _min_gap_hex(inner_init[i], inner_init[j])
            if gap < 1e-6:
                has_overlap = True
    
    # If initial config has overlaps, use a more spread out configuration
    if has_overlap:
        inner_init = np.array([
            [0.0, 0.0, 0.0],
            [sqrt3*1.2, 0.0, 0.0],
            [sqrt3*0.6, 1.8, 0.0],
            [-sqrt3*0.6, 1.8, 0.0],
            [-sqrt3*1.2, 0.0, 0.0],
            [-sqrt3*0.6, -1.8, 0.0],
            [sqrt3*0.6, -1.8, 0.0],
            [sqrt3*1.8, sqrt3*0.6, 0.0],
            [0.0, sqrt3*1.8, 0.0],
            [-sqrt3*1.8, sqrt3*0.6, 0.0],
            [sqrt3*1.8, -sqrt3*0.6, 0.0]
        ])
        required_side = 0.0
        for i in range(11):
            req = _required_side_for_hexagon(inner_init[i, 0], inner_init[i, 1], inner_init[i, 2])
            if req > required_side:
                required_side = req
    
    outer_side_init = required_side * 1.1
    
    params_init = np.concatenate([inner_init.flatten(), [outer_side_init]])
    
    # Optimize with multiple restarts
    best_params = params_init.copy()
    best_val = _objective(params_init)
    
    np.random.seed(42)
    for restart in range(5):
        if restart == 0:
            params = params_init.copy()
        else:
            perturbation = np.random.randn(33) * (0.1 + 0.05 * restart)
            perturbation[2::3] *= 0.05  # smaller angle perturbation
            params = params_init.copy()
            params[:33] += perturbation
            params[33] = outer_side_init
        
        try:
            result = minimize(_objective, params, method='Powell',
                             options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-10})
            if result.fun < best_val:
                best_val = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
    
    inner_hex_data = best_params[:33].reshape(11, 3)
    outer_side = best_params[33]
    
    # Ensure outer side is large enough to contain all inner hexagons
    required_side = 0.0
    for i in range(11):
        req = _required_side_for_hexagon(inner_hex_data[i, 0], inner_hex_data[i, 1], inner_hex_data[i, 2])
        if req > required_side:
            required_side = req
    
    outer_side = max(outer_side, required_side + 1e-8)
    
    # Final validation: check for overlaps and adjust if needed
    for i in range(11):
        for j in range(i+1, 11):
            gap = _min_gap_hex(inner_hex_data[i], inner_hex_data[j])
            if gap < 1e-6:
                # Separate overlapping hexagons
                dx = inner_hex_data[j, 0] - inner_hex_data[i, 0]
                dy = inner_hex_data[j, 1] - inner_hex_data[i, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist > 1e-10:
                    needed = sqrt3 - gap + 1e-5
                    if needed > 0:
                        scale = needed / dist
                        inner_hex_data[i, 0] -= (dx * scale) / 2
                        inner_hex_data[i, 1] -= (dy * scale) / 2
                        inner_hex_data[j, 0] += (dx * scale) / 2
                        inner_hex_data[j, 1] += (dy * scale) / 2
    
    # Recompute required side after separation
    required_side = 0.0
    for i in range(11):
        req = _required_side_for_hexagon(inner_hex_data[i, 0], inner_hex_data[i, 1], inner_hex_data[i, 2])
        if req > required_side:
            required_side = req
    
    outer_side = max(outer_side, required_side + 1e-8)
    
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, float(outer_side)

# EVOLVE-BLOCK-END
