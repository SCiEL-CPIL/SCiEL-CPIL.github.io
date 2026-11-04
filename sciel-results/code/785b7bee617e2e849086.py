import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START

def _hex_vertices(cx, cy, angle_deg):
    """Return 6 vertices of a unit regular hexagon centered at (cx,cy) with given rotation (degrees)."""
    a = np.radians(angle_deg)
    cos_a, sin_a = np.cos(a), np.sin(a)
    # Unit hexagon vertices at angles 0, 60, 120, 180, 240, 300
    base_angles = np.arange(6) * np.pi / 3.0
    rx = np.cos(base_angles)
    ry = np.sin(base_angles)
    # Rotate
    vx = cx + rx * cos_a - ry * sin_a
    vy = cy + rx * sin_a + ry * cos_a
    return np.column_stack([vx, vy])

def _max_projection(inner_positions, inner_angles, center, container_angle_deg):
    """
    Compute the minimum side length s of a regular hexagon (centered at center, rotated by container_angle_deg)
    that contains all inner hexagons.
    A regular hexagon of side s, centered at origin, rotated by theta, has 6 supporting lines with normals
    at angles theta + k*60 for k=0..5. The half-width in direction theta+k*60 is s*cos(0)=s (for flat side) 
    or s*sqrt(3)/2 (for pointy). Actually for a regular hexagon of side s:
    - The distance from center to each flat side (apothem) is s*sqrt(3)/2
    - The distance from center to each vertex is s
    The supporting lines are at distance apothem = s*sqrt(3)/2 from center, with normals at angles 
    theta + 30 + k*60 (for flat-to-flat) or theta + k*60 (for pointy-to-pointy).
    
    Let me use: container hexagon with vertices at angles container_angle_deg + k*60, k=0..5.
    Then the flat sides are perpendicular to directions container_angle_deg + 30 + k*60.
    The apothem (inradius) = s * sqrt(3)/2.
    So for each of 6 directions d_k = container_angle_deg + 30 + k*60 (degrees), the constraint is:
      max over all inner vertices v of dot(v - center, n_k) <= s * sqrt(3)/2
    where n_k = (cos(d_k), sin(d_k)).
    Thus s >= max_k [ max_v dot(v-center, n_k) ] / (sqrt(3)/2)
    Also s >= max_k [ max_v dot(v-center, n_k) ] / (sqrt(3)/2) must be non-negative.
    """
    container_angle_rad = np.radians(container_angle_deg)
    # 6 normal directions (perpendicular to flat sides)
    normal_angles = container_angle_rad + np.pi/6 + np.arange(6) * np.pi/3
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])  # 6x2
    
    cx, cy = center
    apothem_factor = np.sqrt(3) / 2.0
    
    max_proj = 0.0
    for i in range(N):
        verts = _hex_vertices(inner_positions[i, 0], inner_positions[i, 1], inner_angles[i])
        # Subtract center
        verts_c = verts - np.array([cx, cy])
        # Project onto each normal
        projections = verts_c @ normals.T  # N_verts x 6
        max_p = np.max(projections, axis=0)  # 6
        max_proj = max(max_proj, np.max(max_p))
    
    if max_proj < 0:
        max_proj = 0.0
    
    s = max_proj / apothem_factor
    return s

def _pairwise_min_dist(inner_positions, inner_angles):
    """
    Compute minimum distance between any two hexagons' boundaries.
    For two unit hexagons, if centers are distance d apart, they don't overlap if d >= 2.
    But with rotation, the minimum distance between boundaries depends on orientation.
    For simplicity, use center distance >= 2 as non-overlap condition (sufficient for same orientation).
    For different orientations, we need to check more carefully.
    Approximation: use center distance >= 2.0 as the constraint.
    """
    min_dist = np.inf
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(inner_positions[i] - inner_positions[j])
            if d < min_dist:
                min_dist = d
    return min_dist

def _overlap_penalty(inner_positions, inner_angles, min_dist_required=2.0, penalty_weight=100.0):
    """Penalty for overlapping hexagons (center distance < min_dist_required)."""
    pen = 0.0
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(inner_positions[i] - inner_positions[j])
            if d < min_dist_required:
                pen += (min_dist_required - d) ** 2
    return penalty_weight * pen

def construct():
    sqrt3 = np.sqrt(3)
    
    # Initial layout: hexagonal lattice pattern
    # 12 hexagons: try a compact arrangement
    # Row 0 (y = -sqrt3): 4 hexagons at x = -3, -1, 1, 3
    # Row 1 (y = 0): 4 hexagons at x = -2, 0, 2, 4 -> shift to -3, -1, 1, 3? 
    # Let's use offset rows:
    # Row 0 (y = -sqrt3): x = -2, 0, 2 (3 hexagons)
    # Row 1 (y = 0): x = -3, -1, 1, 3 (4 hexagons)  
    # Row 2 (y = sqrt3): x = -2, 0, 2 (3 hexagons)
    # Row 3 (y = 2*sqrt3): x = -1, 1 (2 hexagons)
    # Total: 3+4+3+2 = 12
    
    # Actually let me use a simpler initial guess: 3 rows of 4
    positions = np.zeros((N, 2))
    inner_angles = np.zeros(N)
    
    # 3 rows, 4 columns
    idx = 0
    for row in range(3):
        y = (row - 1) * sqrt3
        for col in range(4):
            x = (col - 1.5) * 2.0
            positions[idx, 0] = x
            positions[idx, 1] = y
            idx += 1
    
    # Objective function for Nelder-Mead
    def objective(params):
        pos = params[:2*N].reshape(N, 2)
        cont_angle = params[2*N]
        
        s = _max_projection(pos, inner_angles, (0.0, 0.0), cont_angle)
        pen = _overlap_penalty(pos, inner_angles)
        
        return s + pen
    
    # Initial parameters
    init_params = np.concatenate([positions.flatten(), [0.0]])
    
    # Run Nelder-Mead optimization
    result = minimize(objective, init_params, method='Nelder-Mead',
                      options={'maxiter': 300, 'xatol': 1e-5, 'fatol': 1e-5, 'adaptive': True})
    
    opt_params = result.x
    opt_pos = opt_params[:2*N].reshape(N, 2)
    opt_cont_angle = opt_params[2*N]
    
    # Final side length
    final_s = _max_projection(opt_pos, inner_angles, (0.0, 0.0), opt_cont_angle)
    
    # Build inner list
    inner = []
    for i in range(N):
        inner.append([float(opt_pos[i, 0]), float(opt_pos[i, 1]), float(inner_angles[i])])
    
    return inner, (0.0, 0.0), float(final_s), float(opt_cont_angle)

# EVOLVE-BLOCK-END
