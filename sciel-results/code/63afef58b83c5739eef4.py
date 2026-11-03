"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize, linprog
from scipy.spatial import ConvexHull

N = 12
TIME_LIMIT = 120.0  # seconds

# Critical directions: 30 directions at 6-degree intervals (for hexagon symmetry)
# Actually, for regular hexagons, the critical directions for non-overlap are at multiples of 30 degrees
# For a regular hexagon, the support function has critical directions at the normal directions of its faces
# The normal directions of a regular hexagon (flat-top) are at 0, 60, 120, 180, 240, 300 degrees
# But when two hexagons can rotate relative to each other, we need to consider more directions
# For safety, use 30 directions at 6-degree intervals

N_DIRS = 30
DIR_ANGLES = np.linspace(0, 2 * np.pi, N_DIRS, endpoint=False)
DIR_VECTORS = np.column_stack([np.cos(DIR_ANGLES), np.sin(DIR_ANGLES)])

# Precompute hexagon vertices for unit hexagon (side length 1)
# A regular hexagon with side length 1 has vertices at distance 1 from center
# For a "pointy-top" hexagon, vertices are at angles 0, 60, 120, 180, 240, 300
# For a "flat-top" hexagon, vertices are at angles 30, 90, 150, 210, 270, 330
# Let's use the standard: vertices at angles 0, 60, 120, 180, 240, 300 (pointy-top)
UNIT_HEX_VERTS = np.column_stack([
    np.cos(np.linspace(0, 2 * np.pi, 6, endpoint=False)),
    np.sin(np.linspace(0, 2 * np.pi, 6, endpoint=False))
])

def hex_vertices(cx, cy, side=1.0, rot_deg=0.0):
    """Get vertices of a regular hexagon."""
    rot = np.deg2rad(rot_deg)
    rot_mat = np.array([[np.cos(rot), -np.sin(rot)],
                        [np.sin(rot), np.cos(rot)]])
    return side * (UNIT_HEX_VERTS @ rot_mat.T) + np.array([cx, cy])

def support_function(hex_verts, dir_vec):
    """Support function: max dot product of vertices with direction vector."""
    return np.max(hex_verts @ dir_vec)

def polygon_area_signed(verts):
    """Compute signed area of polygon (positive for CCW)."""
    n = len(verts)
    area = 0.0
    for i in range(n):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return 0.5 * area

def point_in_hexagon(px, py, cx, cy, side, rot_deg):
    """Check if point (px, py) is inside a regular hexagon."""
    rot = np.deg2rad(rot_deg)
    # Transform point to hexagon's local coordinate system
    dx, dy = px - cx, py - cy
    local = np.array([[np.cos(rot), np.sin(rot)],
                      [-np.sin(rot), np.cos(rot)]]) @ np.array([dx, dy])
    lx, ly = local
    # Regular hexagon with side length s: |x| <= s*sqrt(3)/2, and constraints from edges
    # For a pointy-top hexagon (vertices at 0, 60, 120, 180, 240, 300):
    # The hexagon can be described as: |x| <= s*sqrt(3)/2
    # |x + sqrt(3)*y| <= s*sqrt(3)/2
    # |x - sqrt(3)*y| <= s*sqrt(3)/2
    # Wait, let me think again. For a regular hexagon with side length s, the distance from center to vertex is s.
    # The vertices are at (s*cos(k*pi/3), s*sin(k*pi/3)) for k=0..5
    # The edges are perpendicular to directions at angles pi/6, pi/2, 5pi/6, 7pi/6, 3pi/2, 11pi/6
    # The support in direction (cos(alpha), sin(alpha)) is s*cos(alpha - k*pi/3) for the appropriate k
    # Actually, for a regular hexagon with circumradius s, the inradius is s*sqrt(3)/2
    # The constraints are: for each of the 6 edge normals, the projection is <= inradius
    
    # For vertices at angles 0, 60, 120, 180, 240, 300 degrees:
    # Edge normals are at 30, 90, 150, 210, 270, 330 degrees
    # The constraint is: |lx*cos(30) + ly*sin(30)| <= s*sqrt(3)/2, etc.
    # More simply: a point (lx, ly) is inside if:
    # |lx| <= s*sqrt(3)/2
    # |lx + sqrt(3)*ly| <= s*sqrt(3)/2  
    # |lx - sqrt(3)*ly| <= s*sqrt(3)/2
    
    s3 = s * np.sqrt(3)
    return (abs(lx) <= s3 / 2 and 
            abs(lx + np.sqrt(3) * ly) <= s3 / 2 and 
            abs(lx - np.sqrt(3) * ly) <= s3 / 2)

def outer_hex_contains_point(px, py, ocx, ocy, oside, orot_deg):
    """Check if point is inside the outer hexagon."""
    return point_in_hexagon(px, py, ocx, ocy, oside, orot_deg)

def check_packing_valid(inner, ocenter, oside, orot_deg, tol=1e-6):
    """Check if the packing is valid: no overlaps, all inside outer hexagon."""
    ocx, ocy = ocenter
    n_inner = len(inner)
    
    # Check each inner hexagon is inside the outer hexagon
    for i in range(n_inner):
        cx, cy, rot = inner[i]
        verts = hex_vertices(cx, cy, 1.0, rot)
        for v in verts:
            if not outer_hex_contains_point(v[0], v[1], ocx, ocy, oside, orot_deg):
                return False
    
    # Check no two inner hexagons overlap
    # For convex polygons, they overlap if and only if there exists a direction where their projections overlap
    # For regular hexagons, it's sufficient to check the 6 face-normal directions of each (or a common set)
    for i in range(n_inner):
        for j in range(i + 1, n_inner):
            cx1, cy1, rot1 = inner[i]
            cx2, cy2, rot2 = inner[j]
            verts1 = hex_vertices(cx1, cy1, 1.0, rot1)
            verts2 = hex_vertices(cx2, cy2, 1.0, rot2)
            
            # Use separating axis theorem with the 6 normal directions of each hexagon
            # The normal directions for hex1 are at rot1 + 30 + k*60 degrees
            # The normal directions for hex2 are at rot2 + 30 + k*60 degrees
            normals1 = rot1 + np.pi / 6 + np.arange(6) * np.pi / 3
            normals2 = rot2 + np.pi / 6 + np.arange(6) * np.pi / 3
            all_normals = np.concatenate([normals1, normals2])
            
            for n_angle in all_normals:
                n_vec = np.array([np.cos(n_angle), np.sin(n_angle)])
                proj1 = verts1 @ n_vec
                proj2 = verts2 @ n_vec
                if np.max(proj1) < np.min(proj2) + tol or np.max(proj2) < np.min(proj1) + tol:
                    break
            else:
                # No separating axis found, they overlap
                return False
    
    return True

def compute_min_outer_side_for_inner(inner, ocenter, orot_deg, max_iter=5000):
    """Given inner hexagons, find minimum outer side length."""
    ocx, ocy = ocenter
    # Binary search on outer side length
    lo = 1.0
    hi = 20.0
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        if check_packing_valid(inner, (ocx, ocy), mid, orot_deg):
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-6:
            break
    return hi

# EVOLVE-BLOCK-START
def construct():
    """Construct a packing of 12 unit hexagons inside a regular hexagon, minimizing outer side length."""
    t0 = time.time()
    best_score = float('inf')
    best_result = None
    
    # Strategy: Use a structured lattice approach with optimization
    # Start with a 2x3 or 3x2 grid-like arrangement and optimize
    
    # Parameterization: 
    # - Lattice basis vectors (v1, v2) for the hexagonal lattice
    # - Rotation of the lattice
    # - Global rotation of inner hexagons
    # - Outer hexagon center and rotation
    
    # For a hexagonal lattice, the basis vectors are at 60 degrees
    # Let's parameterize by:
    # - d: distance between adjacent hexagon centers (must be >= 2 for no overlap if aligned)
    # - rot_lattice: rotation of the lattice
    # - rot_inner: rotation of inner hexagons relative to lattice
    # - orot: rotation of outer hexagon
    # - ox, oy: outer hexagon center offset
    
    # Actually, for regular hexagons of side 1, the minimum distance between centers 
    # for non-overlap depends on relative rotation. For same rotation, min distance is 2.
    # For optimal packing, we can use a hexagonal lattice.
    
    # Let's try a 2x6 or 3x4 grid arrangement
    
    # Approach: Use differential evolution to optimize positions directly
    # 12 hexagons * 3 params (x, y, angle) = 36 parameters
    # Plus outer hexagon: 2 (center) + 1 (side) + 1 (rotation) = 4
    # Total: 40 parameters - too many for differential evolution in 120s
    
    # Better approach: Use symmetry. Place hexagons in a symmetric pattern.
    # For 12 hexagons, a good arrangement is a hexagonal cluster.
    # A hexagonal number: 1, 3, 6, 10, 15... 
    # 12 is close to 10 (center + 6 + 3) or 15 (center + 6 + 8)
    # Let's use a pattern: center + 6 around + 5 more, or a 3x4 rectangular grid
    
    # Simplest: use a 3x4 grid of hexagons
    # For hexagonal lattice with spacing d (center-to-center for adjacent hexagons with same rotation):
    # d = 2 * cos(0) = 2 for same rotation, but for 60-degree relative rotation, d can be smaller
    
    # Let's just do a direct optimization with a reduced parameter space
    # Assume all inner hexagons have the same rotation (symmetry assumption)
    # Positions are on a hexagonal lattice
    
    # Hexagonal lattice basis:
    # v1 = d * (cos(theta), sin(theta))
    # v2 = d * (cos(theta + pi/3), sin(theta + pi/3))
    # where d is the lattice spacing and theta is the lattice rotation
    
    # For 12 hexagons, use a 2x6 or 3x4 arrangement
    # Let's use a 3x4 rectangular grid on a hexagonal lattice
    
    def objective(params):
        if time.time() - t0 > TIME_LIMIT - 5:
            return 1e6
        
        # Extract parameters
        # params: [d, theta, rot_inner, orot, ox, oy]
        # d: lattice spacing (center-to-center)
        # theta: lattice rotation
        # rot_inner: rotation of inner hexagons
        # orot: rotation of outer hexagon
        # ox, oy: offset of outer hexagon center
        
        d, theta, rot_inner, orot, ox, oy = params
        
        # Generate 12 positions on a 3x4 grid using hexagonal lattice
        # v1 = d * (cos(theta), sin(theta))
        # v2 = d * (cos(theta + pi/3), sin(theta + pi/3))
        v1 = d * np.array([np.cos(theta), np.sin(theta)])
        v2 = d * np.array([np.cos(theta + np.pi / 3), np.sin(theta + np.pi / 3)])
        
        positions = []
        for i in range(3):
            for j in range(4):
                pos = i * v1 + j * v2
                positions.append(pos)
        
        # Center the positions
        center_pos = np.mean(positions, axis=0)
        positions = [p - center_pos for p in positions]
        
        # Create inner hexagons
        inner = []
        for p in positions:
            inner.append([float(p[0] + ox), float(p[1] + oy), float(rot_inner)])
        
        # Find minimum outer side length
        ocenter = (ox, oy)
        oside = compute_min_outer_side_for_inner(inner, ocenter, orot, max_iter=100)
        
        # Verify validity
        if not check_packing_valid(inner, ocenter, oside, orot):
            return oside + 10.0
        
        return oside
    
    # Bounds for differential evolution
    # d: lattice spacing, typically between 1.5 and 3
    # theta: 0 to pi/3 (hexagonal symmetry)
    # rot_inner: 0 to pi/6 (hexagonal symmetry)
    # orot: 0 to pi/6
    # ox, oy: -5 to 5
    
    bounds = [
        (1.5, 3.0),   # d
        (0, np.pi / 3),  # theta
        (0, np.pi / 6),  # rot_inner
        (0, np.pi / 6),  # orot
        (-5, 5),       # ox
        (-5, 5)        # oy
    ]
    
    # Run differential evolution
    result = differential_evolution(
        objective, bounds, 
        maxiter=200, popsize=15,
        tol=1e-4, seed=42,
        polish=False,  # We'll polish separately
        workers=1
    )
    
    if result.success or result.fun < best_score:
        best_score = result.fun
        best_params = result.x
        d, theta, rot_inner, orot, ox, oy = best_params
        
        v1 = d * np.array([np.cos(theta), np.sin(theta)])
        v2 = d * np.array([np.cos(theta + np.pi / 3), np.sin(theta + np.pi / 3)])
        
        positions = []
        for i in range(3):
            for j in range(4):
                pos = i * v1 + j * v2
                positions.append(pos)
        
        center_pos = np.mean(positions, axis=0)
        positions = [p - center_pos for p in positions]
        
        inner = []
        for p in positions:
            inner.append([float(p[0] + ox), float(p[1] + oy), float(rot_inner)])
        
        ocenter = (float(ox), float(oy))
        oside = compute_min_outer_side_for_inner(inner, ocenter, orot, max_iter=200)
        
        if check_packing_valid(inner, ocenter, oside, orot):
            best_result = (inner, ocenter, oside, orot)
    
    # Try a few other grid arrangements
    # 2x6 grid
    def objective_2x6(params):
        if time.time() - t0 > TIME_LIMIT - 5:
            return 1e6
        
        d, theta, rot_inner, orot, ox, oy = params
        
        v1 = d * np.array([np.cos(theta), np.sin(theta)])
        v2 = d * np.array([np.cos(theta + np.pi / 3), np.sin(theta + np.pi / 3)])
        
        positions = []
        for i in range(2):
            for j in range(6):
                pos = i * v1 + j * v2
                positions.append(pos)
        
        center_pos = np.mean(positions, axis=0)
        positions = [p - center_pos for p in positions]
        
        inner = []
        for p in positions:
            inner.append([float(p[0] + ox), float(p[1] + oy), float(rot_inner)])
        
        ocenter = (ox, oy)
        oside = compute_min_outer_side_for_inner(inner, ocenter, orot, max_iter=100)
        
        if not check_packing_valid(inner, ocenter, oside, orot):
            return oside + 10.0
        
        return oside
    
    result2 = differential_evolution(
        objective_2x6, bounds,
        maxiter=150, popsize=12,
        tol=1e-4, seed=43,
        polish=False,
        workers=1
    )
    
    if result2.success or result2.fun < best_score:
        best_score = result2.fun
        d, theta, rot_inner, orot, ox, oy = result2.x
        
        v1 = d * np.array([np.cos(theta), np.sin(theta)])
        v2 = d * np.array([np.cos(theta + np.pi / 3), np.sin(theta + np.pi / 3)])
        
        positions = []
        for i in range(2):
            for j in range(6):
                pos = i * v1 + j * v2
                positions.append(pos)
        
        center_pos = np.mean(positions, axis=0)
        positions = [p - center_pos for p in positions]
        
        inner = []
        for p in positions:
            inner.append([float(p[0] + ox), float(p[1] + oy), float(rot_inner)])
        
        ocenter = (float(ox), float(oy))
        oside = compute_min_outer_side_for_inner(inner, ocenter, orot, max_iter=200)
        
        if check_packing_valid(inner, ocenter, oside, orot):
            best_result = (inner, ocenter, oside, orot)
    
    # If no valid result, return a safe default
    if best_result is None:
        # Simple linear arrangement
        inner = [[2.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 20.0, 0.0
    
    return best_result
# EVOLVE-BLOCK-END

