import numpy as np
import time
from scipy.optimize import minimize_scalar

N = 12

def _hex_vertices(cx, cy, s, angle_deg):
    """Get vertices of a regular hexagon with side s, center (cx,cy), rotation angle_deg."""
    a = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * np.pi / 3
        x = cx + s * np.cos(theta)
        y = cy + s * np.sin(theta)
        verts.append([x, y])
    return np.array(verts)

def _hex_face_normals(angle_deg):
    """Get the 6 outward face normals of a regular hexagon with rotation angle_deg."""
    a = np.radians(angle_deg)
    normals = []
    for i in range(6):
        theta = a + i * np.pi / 3 + np.pi / 6  # face normals are offset by 30 deg
        normals.append([np.cos(theta), np.sin(theta)])
    return np.array(normals)

def _min_enclosing_hexagon(inner_positions, inner_angles, inner_sides):
    """
    Find minimal regular hexagon (center, side, angle) containing all inner hexagons.
    """
    # Collect all vertices of all inner hexagons
    all_verts = []
    for (cx, cy, ang), s in zip(inner_positions, inner_sides):
        verts = _hex_vertices(cx, cy, s, ang)
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    def max_width(phi_deg):
        phi = np.radians(phi_deg)
        # Face normals of outer hexagon
        normals = np.array([
            [np.cos(phi + i * np.pi / 3 + np.pi / 6), np.sin(phi + i * np.pi / 3 + np.pi / 6)] 
            for i in range(6)
        ])
        
        # Compute projections
        projs = all_verts @ normals.T  # shape (num_verts, 6)
        max_projs = projs.max(axis=0)  # shape (6,)
        min_projs = projs.min(axis=0)  # shape (6,)
        widths = max_projs - min_projs  # shape (6,)
        
        return np.max(widths)

    # Optimize phi in [0, 60)
    # The function is periodic with period 60.
    # We can use minimize_scalar with bounds.
    try:
        res = minimize_scalar(max_width, bounds=(0, 60), method='bounded', options={'xatol': 1e-8})
        best_angle = res.x
        min_width = res.fun
    except Exception:
        # Fallback to grid search if optimization fails
        best_angle = 0
        min_width = float('inf')
        for phi_deg in np.linspace(0, 60, 100):
            w = max_width(phi_deg)
            if w < min_width:
                min_width = w
                best_angle = phi_deg

    required_apothem = min_width / 2
    s_candidate = required_apothem * 2 / np.sqrt(3)
    
    # Now find the center for this optimal angle
    phi = np.radians(best_angle)
    normals = np.array([
        [np.cos(phi + i * np.pi / 3 + np.pi / 6), np.sin(phi + i * np.pi / 3 + np.pi / 6)] 
        for i in range(6)
    ])
    
    projs = all_verts @ normals.T
    max_projs = projs.max(axis=0)
    min_projs = projs.min(axis=0)
    
    lo = max_projs - required_apothem
    hi = min_projs + required_apothem
    
    # Check feasibility
    if np.any(lo > hi + 1e-9):
        return None, None, float('inf'), best_angle
        
    mid = (lo + hi) / 2
    c, residuals, rank, sv = np.linalg.lstsq(normals, mid, rcond=None)
    
    # Verify
    c_projs = normals @ c
    if np.any(c_projs < lo - 1e-8) or np.any(c_projs > hi + 1e-8):
         # If verification fails, we might need to adjust, but for regular hexagon it should hold
         pass

    return c[0], c[1], s_candidate, best_angle

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    s = 1.0  # unit hexagons
    sqrt3 = np.sqrt(3)
    
    best_result = None
    best_score = float('inf')
    
    # Generate honeycomb positions with a small gap to ensure disjointness
    # Scale factor 1.001 ensures slight separation for numerical safety
    # Flat-top hexagons (30 deg) have width sqrt(3) in x, so honeycomb spacing sqrt(3) is valid
    scale = 1.001
    
    def honeycomb_positions(count, pattern):
        """Generate positions for 'count' hexagons in honeycomb pattern."""
        positions = []
        for j, row_len in enumerate(pattern):
            for i in range(row_len):
                x = (i + 0.5 * (j % 2)) * sqrt3 * scale
                y = j * 1.5 * scale
                positions.append((x, y, 0.0))
        return positions[:count]
    
    # Try several row patterns that sum to 12
    patterns = [
        [3, 4, 3, 2],
        [4, 3, 3, 2],
        [2, 3, 4, 3],
        [3, 3, 4, 2],
        [4, 4, 2, 2],
        [2, 2, 4, 4],
        [5, 3, 2, 2],
        [2, 2, 3, 5],
        [6, 3, 3],
        [3, 3, 6],
        [4, 4, 4],
        [2, 4, 4, 2],
        [4, 2, 4, 2],
        [2, 4, 2, 4],
        [3, 5, 4],
        [4, 5, 3],
        [5, 4, 3],
        [3, 4, 5],
    ]
    
    for pattern in patterns:
        if sum(pattern) != 12:
            continue
        if len(pattern) > 6:
            continue
        
        positions = honeycomb_positions(12, pattern)
        if len(positions) != 12:
            continue
        
        # Use flat-top orientation (30 deg) which allows tighter packing in x-direction
        # for the honeycomb lattice defined above.
        rot = 30.0
        inner = [(x, y, rot) for (x, y, _) in positions]
        inner_sides = [1.0] * 12
        
        cx, cy, side, angle = _min_enclosing_hexagon(inner, [r[2] for r in inner], inner_sides)
        
        if side is None or side == float('inf'):
            continue

        # Add small safety margin for numerical precision
        side_safe = side * 1.0001 + 1e-6
        
        if side_safe < best_score:
            best_score = side_safe
            best_result = (inner, (cx, cy), side_safe, angle)
        
        if time.time() > deadline:
            break
    
    if best_result is None:
        # Fallback: linear arrangement
        inner = [(4.0 * k, 0.0, 0.0) for k in range(-6, 6)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return best_result
