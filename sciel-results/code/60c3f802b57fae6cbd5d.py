import numpy as np
from scipy.optimize import minimize
import time

N = 12
START_TIME = time.time()
TIME_LIMIT = 140.0  # safety margin before 150s kill

def hex_vertices(cx, cy, angle_deg, side=1.0):
    """Return 6 vertices of a unit regular hexagon centered at (cx,cy) with rotation angle_deg."""
    angle = np.radians(angle_deg)
    # Vertices at angles 0, 60, 120, 180, 240, 300 degrees + rotation
    base_angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180 + angle
    xs = cx + side * np.cos(base_angles)
    ys = cy + side * np.sin(base_angles)
    return np.column_stack([xs, ys])

def point_in_hexagon(p, center, side, rotation_deg):
    """Check if point p is inside a regular hexagon with given center, side, rotation.
    Returns minimum margin (positive = inside, negative = outside)."""
    rot = np.radians(rotation_deg)
    # Edge normals for a hexagon with vertices at angles 0,60,...: edges perpendicular to 30,90,150,210,270,330
    edge_normal_angles = np.array([30, 90, 150, 210, 270, 330]) * np.pi / 180 + rot
    normals = np.column_stack([np.cos(edge_normal_angles), np.sin(edge_normal_angles)])
    inradius = side * np.sqrt(3) / 2
    # For each edge normal, the constraint is: (p - center) . n <= inradius
    diffs = p - center
    margins = inradius - np.dot(diffs, normals.T)
    return np.min(margins)

def compute_min_containing_hexagon(inner_hexes, outer_center):
    """Compute minimum side length S and rotation for outer hexagon containing all inner hexagons.
    We search over rotation of outer hexagon."""
    # Get all vertices of all inner hexagons
    all_pts = []
    for (cx, cy, ang) in inner_hexes:
        verts = hex_vertices(cx, cy, ang, 1.0)
        all_pts.append(verts)
    all_pts = np.vstack(all_pts)
    
    # For a given outer rotation, compute required S
    # S must satisfy: for all points p, for all 6 edge normals n_k: (p - c) . n_k <= S * sqrt(3)/2
    # So S >= max over all p, all k of (p-c).n_k / (sqrt(3)/2)
    # But we also need S >= 0
    
    best_S = np.inf
    best_rot = 0.0
    
    # Try a range of rotations
    for rot_deg in np.linspace(0, 60, 13):  # 13 angles (0 to 60 inclusive, step 5)
        rot = np.radians(rot_deg)
        edge_normal_angles = np.array([30, 90, 150, 210, 270, 330]) * np.pi / 180 + rot
        normals = np.column_stack([np.cos(edge_normal_angles), np.sin(edge_normal_angles)])
        
        diffs = all_pts - np.array(outer_center)
        # (N_pts x 6) matrix of dot products
        dots = diffs @ normals.T  # shape (N_pts, 6)
        max_dot = np.max(dots)
        S_required = max_dot / (np.sqrt(3) / 2)
        S_required = max(S_required, 0.0)
        
        if S_required < best_S:
            best_S = S_required
            best_rot = rot_deg
    
    return best_S, best_rot

def objective(params):
    """Objective: minimize outer side length S, with penalty for non-containment.
    params: [x1,y1,a1, x2,y2,a2, ..., x12,y12,a12] (36 params)
    Outer center fixed at (0,0)."""
    elapsed = time.time() - START_TIME
    if elapsed > TIME_LIMIT:
        return 1e10
    
    inner = []
    for i in range(N):
        x = params[3*i]
        y = params[3*i+1]
        ang = params[3*i+2]
        inner.append((x, y, ang))
    
    outer_center = (0.0, 0.0)
    S, rot = compute_min_containing_hexagon(inner, outer_center)
    return S

def construct():
    global START_TIME
    START_TIME = time.time()
    
    best_result = None
    best_score = np.inf
    
    # Initial guess: place hexagons in a honeycomb-like pattern
    # A regular hexagonal packing of 12 hexagons
    # Let's try a 3-4-3-2 arrangement or similar
    
    # Simple initial: grid-like arrangement
    def make_initial(seed_offset=0):
        np.random.seed(42 + seed_offset)
        # Start with a rough hexagonal arrangement
        positions = []
        # 12 hexagons: try a pattern
        # Row 0: 4 hexagons
        # Row 1: 3 hexagons
        # Row 2: 3 hexagons
        # Row 3: 2 hexagons
        # This is a rough hexagonal cluster
        
        # Actually, let's use a more systematic approach
        # Place in a hexagonal grid pattern
        dx = 2 * np.sqrt(3)  # horizontal spacing for hexagons (side 1)
        dy = 1.5             # vertical spacing
        
        # For a hexagonal packing of unit hexagons:
        # Horizontal distance between centers in same row: 2*sqrt(3)
        # Vertical distance between rows: 1.5
        # Offset alternate rows by sqrt(3)
        
        # Let's place 12 in a compact hexagonal shape
        # Center at origin
        coords = []
        # Ring 0: center (1 hex)
        coords.append((0, 0))
        # Ring 1: 6 hexagons around center
        for k in range(6):
            angle = np.radians(60 * k)
            r = 2 * np.sqrt(3)
            coords.append((r * np.cos(angle), r * np.sin(angle)))
        # Ring 2: 5 more to make 12 total
        for k in range(6):
            angle = np.radians(60 * k + 30)
            r = 4 * np.sqrt(3)
            coords.append((r * np.cos(angle), r * np.sin(angle)))
            if len(coords) == 12:
                break
        
        params = np.zeros(36)
        for i in range(12):
            params[3*i] = coords[i][0] + np.random.randn() * 0.5
            params[3*i+1] = coords[i][1] + np.random.randn() * 0.5
            params[3*i+2] = np.random.choice([0, 30, 60]) + np.random.randn() * 5
        return params
    
    # Multi-start optimization
    n_starts = 8
    for start in range(n_starts):
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
        
        params = make_initial(start)
        
        try:
            result = minimize(
                objective, 
                params, 
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 0.01, 'fatol': 0.001, 'adaptive': True}
            )
            
            if result.fun < best_score:
                best_score = result.fun
                best_result = result.x
        except Exception:
            pass
    
    # If we found a good result, use it; otherwise use initial
    if best_result is not None:
        params = best_result
    else:
        params = make_initial(0)
    
    # Build inner hexagons
    inner = []
    for i in range(N):
        x = params[3*i]
        y = params[3*i+1]
        ang = params[3*i+2]
        inner.append([float(x), float(y), float(ang)])
    
    outer_center = (0.0, 0.0)
    S, rot = compute_min_containing_hexagon(inner, outer_center)
    
    return inner, outer_center, float(S), float(rot)
