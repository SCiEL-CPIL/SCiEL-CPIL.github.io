"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize, linprog

N = 12

# EVOLVE-BLOCK-START

def hex_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon centered at (cx, cy) with given side length and rotation."""
    angle_rad = np.radians(angle_deg)
    # Vertices at angles 0, 60, ... relative to rotation
    # Standard flat-top hexagon has vertices at 0, 60, 120, 180, 240, 300?
    # No, flat top means top and bottom are flat.
    # Vertices at 30, 90, 150, 210, 270, 330?
    # Let's use the definition:
    # Angle 0: Vertex at (s, 0)? No, that's pointy right.
    # The task doesn't specify orientation, just "regular hexagon".
    # The evaluator checks containment.
    # I will use vertices at angles 0, 60, ... + angle_rad.
    angles = angle_rad + np.arange(6) * np.pi / 3
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def construct():
    """Construct 12 unit hexagons inside a minimal outer hexagon."""
    
    # 1. Define inner hexagon positions
    # Using a 3x4 lattice block (parallelogram)
    # Axial coordinates (q, r)
    # x = 1.5 * q
    # y = sqrt(3) * (r + 0.5 * q)
    sqrt3 = np.sqrt(3)
    
    positions = []
    for q in range(4):
        for r in range(3):
            x = 1.5 * q
            y = sqrt3 * (r + 0.5 * q)
            positions.append((x, y, 0.0)) # Angle 0 for all
    
    # 2. Generate all vertices of the 12 inner hexagons
    all_points = []
    for x, y, a in positions:
        verts = hex_vertices(x, y, 1.0, a)
        all_points.append(verts)
    all_points = np.vstack(all_points)
    
    # 3. Find minimum enclosing regular hexagon
    # Variables: cx, cy, S
    # We scan the rotation angle theta of the outer hexagon
    best_S = float('inf')
    best_cx = 0
    best_cy = 0
    best_theta = 0
    
    # Scan theta from 0 to 60 degrees
    # Step 0.1 degrees is ~600 iterations. LP is fast.
    thetas = np.linspace(0, 60, 601)
    
    for theta in thetas:
        theta_rad = np.radians(theta)
        # Normals for the 6 sides of the outer hexagon
        # Side normals are perpendicular to sides.
        # If vertices are at theta + k*pi/3, sides are at theta + pi/6 + k*pi/3
        # Normals point outward.
        # Normal angles: theta + pi/6 + k*pi/3
        normal_angles = theta_rad + np.pi/6 + np.arange(6) * np.pi/3
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        
        # Setup LP
        # Minimize S
        # c = [0, 0, 1]
        c = np.array([0.0, 0.0, 1.0])
        
        # Constraints:
        # For each point p and each normal n_k:
        # n_k . p - n_k . c <= S * apothem
        # n_k . p - n_k . [cx, cy] <= S * (sqrt(3)/2)
        # -n_kx * cx - n_ky * cy + (sqrt(3)/2) * S >= n_k . p
        # Multiply by -1 for linprog (A_ub @ x <= b_ub):
        # n_kx * cx + n_ky * cy - (sqrt(3)/2) * S <= - (n_k . p)
        
        A_ub = []
        b_ub = []
        
        apothem_factor = sqrt3 / 2.0
        
        for p in all_points:
            for n in normals:
                row = np.array([n[0], n[1], -apothem_factor])
                val = -np.dot(n, p)
                A_ub.append(row)
                b_ub.append(val)
                
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        
        # Bounds for cx, cy, S
        # S >= 0
        # cx, cy can be anything, but let's bound them to avoid numerical issues
        # The points are roughly in x:[0, 4.5], y:[0, 4]
        # Center should be near [2, 2]
        bounds = [(-10, 10), (-10, 10), (0, None)]
        
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.status == 0:
                S = res.x[2]
                if S < best_S:
                    best_S = S
                    best_cx = res.x[0]
                    best_cy = res.x[1]
                    best_theta = theta
        except Exception:
            pass
            
    # Return the optimal configuration
    inner = [[p[0], p[1], p[2]] for p in positions]
    center = (best_cx, best_cy)
    side = best_S
    angle = best_theta
    
    return inner, center, side, angle

# EVOLVE-BLOCK-END

