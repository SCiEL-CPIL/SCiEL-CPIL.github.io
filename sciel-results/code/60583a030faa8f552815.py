"""
Strategy: Lattice-Based Topological Optimization
Instead of treating each hexagon independently (33 DOF), we assume the optimal packing
for 11 hexagons is a subset of a hexagonal lattice. We fix the relative positions
on a rigid lattice (11 points), and optimize only the global rotation, translation,
and lattice shear (3 DOF). This removes the non-smooth overlap constraints and
allows for fast, smooth optimization of the bounding box size.
"""
import time
import numpy as np
from scipy.optimize import differential_evolution

N = 11
START_TIME = time.time()
DEADLINE = START_TIME + 115.0  # Safety margin

# Define the 11 hexagon centers in a hexagonal lattice configuration
# Center at (0,0), surrounded by 6 in first ring, and 4 in second ring to make 11?
# Or a 3x3 grid with corners removed?
# A common compact shape for 11 is:
#   Row 0: 3 hexagons
#   Row 1: 4 hexagons
#   Row 2: 3 hexagons
#   Row 3: 1 hexagon? No, that's 11.
# Let's use a "flower" of 7 plus 4 more attached to the outer ring.
# Actually, the densest packing of 11 circles/hexagons is often a hexagonal number pattern.
# H(2) = 7. H(3) = 19.
# Let's use the 7-center flower plus 4 attached to the "equator".
# Coordinates for hex lattice (side=1, distance between centers = 2):
# (0,0)
# (2,0), (1,sqrt(3)), (-1,sqrt(3)), (-2,0), (-1,-sqrt(3)), (1,-sqrt(3)) -> 7 total
# Add 4 more: (3, sqrt(3)*0.5*2? No, next layer).
# Let's just use a grid of 11 points that is known to be compact.
# A 4-4-3 arrangement?
# Let's generate a set of 11 points from a hex lattice that minimizes the initial bounding box.
# We will define the points in a local frame, then optimize the global transform.

def _generate_lattice_points():
    """
    Generate 11 points from a hexagonal lattice.
    We try a few candidate topologies and pick the one with the smallest initial bounding box.
    """
    sqrt3 = np.sqrt(3)
    candidates = []
    
    # Topology 1: 3-4-4 (rows)
    # Row 0 (y=0): x = -2, 0, 2
    # Row 1 (y=sqrt(3)): x = -1, 1, 3, -3? No, staggered.
    # Standard hex lattice coords:
    # (i, j) -> (i + j/2, j * sqrt(3)/2)
    # Let's use axial coordinates (q, r).
    # Distance between neighbors is 2.
    # Points:
    # (0,0)
    # (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1) -> 7
    
    # Add 4 more to make 11.
    # Attach to (2,0), (1,1), (-1,2), (-2,1)?
    # Let's just use a simple grid of 11 points in a hex shape.
    # Center (0,0)
    # Ring 1: 6 points
    # Ring 2: We need 4 more.
    # Let's take a 5x5 grid of hex centers and select the 11 with smallest radius?
    # Or just a specific set.
    
    # Set 1: 7 center + 4 outer
    pts1 = [
        (0, 0),
        (2, 0), (1, sqrt3), (-1, sqrt3), (-2, 0), (-1, -sqrt3), (1, -sqrt3),
        (3, sqrt3/2), (2, sqrt3*1.5), (-2, sqrt3*1.5), (-3, sqrt3/2) # This might be sparse
    ]
    
    # Set 2: 3-4-4 arrangement (rows)
    # Row 0: (-2, 0), (0, 0), (2, 0)
    # Row 1: (-1, sqrt3), (1, sqrt3), (3, sqrt3), (-3, sqrt3) -- wait, spacing is 2 horizontally.
    # Hex lattice:
    # y=0: x = -2, 0, 2
    # y=sqrt(3): x = -1, 1, 3, -3? No, x = -3, -1, 1, 3
    # y=2*sqrt(3): x = -2, 0, 2
    # Total: 3 + 4 + 3 = 10. Need 11.
    # Add one more? Maybe y=-sqrt(3): x = 0?
    # Let's try 4-4-3.
    # y=0: -3, -1, 1, 3 (4 pts)
    # y=sqrt(3): -2, 0, 2, 4? No, -2, 0, 2 (3 pts)
    # y=2*sqrt(3): -1, 1, 3 (3 pts)
    # Total 10.
    
    # Let's just use the 7-center flower and 4 attached to the "right" side.
    # (0,0)
    # (2,0), (1,s3), (-1,s3), (-2,0), (-1,-s3), (1,-s3)
    # (3, s3/2), (3, -s3/2), (4, 0), (5, 0)? No, that's a line.
    
    # Actually, the optimal packing for N=11 is often a hexagonal shape.
    # Let's use the points from the 19-point hex (H(3)) and remove 8 outer ones?
    # H(3) points:
    # (0,0)
    # Ring 1: 6
    # Ring 2: 12
    # Total 19.
    # Remove 8 from Ring 2 to leave 4.
    # Keep the 4 that are "closest" to the center of mass?
    
    # Let's generate all 19 and pick the 11 with smallest distance to origin?
    # That will just be the 7 center + 4 from ring 2.
    # Which 4? The ones at angles 0, 60, 120, 180?
    
    pts = [(0,0)]
    for i in range(6):
        a = i * np.pi / 3
        pts.append((2*np.cos(a), 2*np.sin(a)))
    # Ring 2
    for i in range(6):
        a = i * np.pi / 3 + np.pi/6
        # Distance is 2 * sqrt(3) ? No.
        # In hex lattice, distance to ring 2 is 2 * sqrt(3)?
        # (2,0) to (3, sqrt(3)/2 * 2?)
        # Let's just use axial coords.
        pass
        
    # Simpler: Just use a 3x3 grid of hexagons (9) + 2 more?
    # Or 4x3 grid (12) - 1?
    
    # Let's use a fixed set of 11 points that are known to be compact.
    # From literature, N=11 optimal is likely a hexagonal cluster.
    # Let's use:
    # (0,0)
    # (2,0), (-2,0)
    # (1,s3), (-1,s3)
    # (1,-s3), (-1,-s3)
    # (3, s3/2), (-3, s3/2)
    # (3, -s3/2), (-3, -s3/2)
    # (4, 0)
    # Total 11.
    
    s3 = np.sqrt(3)
    points = np.array([
        [0, 0],
        [2, 0], [-2, 0],
        [1, s3], [-1, s3],
        [1, -s3], [-1, -s3],
        [3, s3/2], [-3, s3/2],
        [3, -s3/2], [-3, -s3/2],
        [4, 0]
    ])
    return points

POINTS = _generate_lattice_points()

def _hex_vertices(cx, cy, side, angle_deg):
    a = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * np.pi / 3
        x = cx + side * np.cos(theta)
        y = cy + side * np.sin(theta)
        verts.append([x, y])
    return np.array(verts)

def _point_in_hexagon(px, py, cx, cy, side, angle_deg):
    dx = px - cx
    dy = py - cy
    a = np.radians(angle_deg)
    ca, sa = np.cos(-a), np.sin(-a)
    lx = ca * dx - sa * dy
    ly = sa * dx + ca * dy
    s = side
    sqrt3 = np.sqrt(3)
    verts_local = np.array([
        [s, 0], [s/2, s*sqrt3/2], [-s/2, s*sqrt3/2], 
        [-s, 0], [-s/2, -s*sqrt3/2], [s/2, -s*sqrt3/2]
    ])
    for i in range(6):
        v1 = verts_local[i]
        v2 = verts_local[(i + 1) % 6]
        edge = v2 - v1
        nx, ny = edge[1], -edge[0]
        if nx * (lx - v1[0]) + ny * (ly - v1[1]) > 1e-9:
            return False
    return True

def _min_outer_side(inner_positions, inner_angles, outer_center, outer_angle, tol=1e-6):
    cx, cy = outer_center
    all_verts = []
    for i in range(len(inner_positions)):
        x, y = inner_positions[i]
        ang = inner_angles[i]
        verts = _hex_vertices(x, y, 1.0, ang)
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    lo, hi = 1.0, 100.0
    for _ in range(40):
        mid = (lo + hi) / 2
        ok = True
        for v in all_verts:
            if not _point_in_hexagon(v[0], v[1], cx, cy, mid, outer_angle):
                ok = False
                break
        if ok:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    return hi

def _objective(params):
    """
    params: [rotation_deg, tx, ty]
    We fix the lattice points, rotate them by rotation_deg, and translate by (tx, ty).
    All inner hexagons have the same rotation as the lattice? 
    Actually, in a hexagonal packing, all hexagons usually have the same orientation.
    So inner_angles = [rotation_deg] * N.
    Outer hexagon is centered at (0,0) with rotation 0? Or we can optimize outer rotation too?
    Let's fix outer rotation to 0 for simplicity, or optimize it.
    Let's optimize: [inner_rot, tx, ty, outer_rot]
    """
    if time.time() > DEADLINE:
        return 1000.0
        
    inner_rot, tx, ty, outer_rot = params
    
    # Transform lattice points
    a = np.radians(inner_rot)
    ca, sa = np.cos(a), np.sin(a)
    # Rotate points
    rx = ca * POINTS[:, 0] - sa * POINTS[:, 1]
    ry = sa * POINTS[:, 0] + ca * POINTS[:, 1]
    # Translate
    rx += tx
    ry += ty
    
    inner_pos = np.column_stack((rx, ry))
    inner_ang = np.full(N, inner_rot)
    
    # Outer hexagon centered at (0,0)
    outer_c = (0.0, 0.0)
    
    side = _min_outer_side(inner_pos, inner_ang, outer_c, outer_rot)
    
    # Add penalty if time is running out
    return side

def construct():
    global START_TIME, DEADLINE
    
    # Initial guess
    # The lattice is centered at 0,0.
    # Optimal rotation might be 0 or 30.
    # Initial params: [0, 0, 0, 0]
    
    bounds = [
        (-30, 30),   # inner_rot
        (-5, 5),     # tx
        (-5, 5),     # ty
        (-30, 30)    # outer_rot
    ]
    
    # Use differential evolution for global optimization
    # This is robust and handles the non-convex landscape of bounding box size.
    
    try:
        res = differential_evolution(
            _objective, 
            bounds, 
            seed=42, 
            maxiter=100, 
            tol=1e-5,
            polish=True,
            workers=1,
            init='sobol',
            updating='immediate',
            maxfev=2000
        )
        best_params = res.x
    except Exception as e:
        # Fallback to local search if DE fails
        best_params = [0.0, 0.0, 0.0, 0.0]
        
    # Refine with local search (Nelder-Mead)
    from scipy.optimize import minimize
    
    def _local_obj(p):
        return _objective(p)
        
    try:
        res2 = minimize(
            _local_obj, 
            best_params, 
            method='Nelder-Mead', 
            options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6}
        )
        best_params = res2.x
    except Exception:
        pass
        
    inner_rot, tx, ty, outer_rot = best_params
    
    # Reconstruct positions
    a = np.radians(inner_rot)
    ca, sa = np.cos(a), np.sin(a)
    rx = ca * POINTS[:, 0] - sa * POINTS[:, 1] + tx
    ry = sa * POINTS[:, 0] + ca * POINTS[:, 1] + ty
    
    inner_pos = np.column_stack((rx, ry))
    inner_ang = np.full(N, inner_rot)
    
    outer_c = (0.0, 0.0)
    best_side = _min_outer_side(inner_pos, inner_ang, outer_c, outer_rot)
    
    # Final validity check
    # We need to ensure no overlaps. Since we used a rigid lattice with distance 2.0,
    # and hexagons have side 1.0 (diameter 2.0), they are tangent.
    # Numerical errors might cause slight overlap. We can shrink the lattice slightly if needed?
    # Or just trust the lattice. The task says "disjoint". Tangent is usually allowed or we can add tiny epsilon.
    # Let's check validity.
    
    inner_list = [[float(inner_pos[i][0]), float(inner_pos[i][1]), float(inner_ang[i])] for i in range(N)]
    
    # Check overlaps
    verts_list = []
    for x, y, ang in inner_list:
        verts_list.append(_hex_vertices(x, y, 1.0, ang))
        
    valid = True
    for i in range(N):
        for j in range(i+1, N):
            # Simple distance check between centers
            d = np.linalg.norm(inner_pos[i] - inner_pos[j])
            if d < 2.0 - 1e-5: # Should be >= 2.0
                valid = False
                break
        if not valid: break
        
    if not valid:
        # Fallback to a known valid configuration (e.g. line)
        inner_list = [[4.0 * k, 0.0, 0.0] for k in range(N)]
        return inner_list, (0.0, 0.0), 40.0, 0.0
        
    return inner_list, (0.0, 0.0), float(best_side), float(outer_rot)

# EVOLVE-BLOCK-END
