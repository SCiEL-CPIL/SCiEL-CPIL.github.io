"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 12

# EVOLVE-BLOCK-START
def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon."""
    angle = np.radians(angle_deg)
    verts = []
    for k in range(6):
        theta = angle + k * np.pi / 3
        verts.append((cx + side * np.cos(theta), cy + side * np.sin(theta)))
    return np.array(verts)

def _point_in_hexagon(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside regular hexagon."""
    # Transform to hexagon's local frame
    dx = px - cx
    dy = py - cy
    angle = np.radians(angle_deg)
    cos_a = np.cos(-angle)
    sin_a = np.sin(-angle)
    lx = dx * cos_a - dy * sin_a
    ly = dx * sin_a + dy * cos_a
    # Regular hexagon with flat top: |x| <= side, |y| <= side*sqrt(3)/2, and |y| <= -sqrt(3)*|x|/2 + sqrt(3)*side/2
    # Actually for a regular hexagon with vertices at angles 0,60,120,...:
    # The hexagon is defined by 6 half-planes. For side length s, the inradius is s*sqrt(3)/2
    # In local coords, the hexagon with "pointy top" (vertex at top) has:
    # |x| <= s*sqrt(3)/2 (inradius)
    # |y| <= s (circumradius)
    # And the 6 constraints: for each edge...
    # Simpler: a regular hexagon centered at origin with circumradius R=side has vertices at R*exp(i*k*pi/3)
    # The half-plane constraints are: for each k, x*cos(theta_k) + y*sin(theta_k) <= R*cos(pi/6) = R*sqrt(3)/2
    # where theta_k = k*pi/3 + pi/6 (normal to each edge)
    R = side
    inradius = R * np.sqrt(3) / 2
    for k in range(6):
        theta = k * np.pi / 3 + np.pi / 6
        if lx * np.cos(theta) + ly * np.sin(theta) > inradius + 1e-10:
            return False
    return True

def _hexagon_area(cx, cy, side, angle_deg):
    return (3 * np.sqrt(3) / 2) * side ** 2

def _min_enclosing_hexagon(centers, angles, side_inner=1.0):
    """Find minimal regular hexagon containing all given hexagons."""
    # The outer hexagon must contain all vertices of all inner hexagons
    # We need to find center (cx, cy), side length L, and rotation phi such that
    # all vertices of all inner hexagons are inside the outer hexagon.
    # This is non-convex in general, but we can use optimization.
    
    all_verts = []
    for i in range(len(centers)):
        cx, cy = centers[i]
        ang = angles[i]
        verts = _hex_vertices(cx, cy, side_inner, ang)
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    def objective(params):
        cx, cy, L, phi = params
        if L <= 0:
            return 1e10
        # Check all vertices are inside outer hexagon
        inradius = L * np.sqrt(3) / 2
        max_violation = 0.0
        for k in range(6):
            theta = np.radians(phi) + k * np.pi / 3 + np.pi / 6
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)
            # For each vertex, compute the projection
            dx = all_verts[:, 0] - cx
            dy = all_verts[:, 1] - cy
            projections = dx * cos_t + dy * sin_t
            # Must be <= inradius
            violation = np.max(projections) - inradius
            if violation > max_violation:
                max_violation = violation
        if max_violation > 0:
            return L + 100 * max_violation
        return L
    
    # Initial guess
    cx0 = np.mean(all_verts[:, 0])
    cy0 = np.mean(all_verts[:, 1])
    x_range = np.max(all_verts[:, 0]) - np.min(all_verts[:, 0])
    y_range = np.max(all_verts[:, 1]) - np.min(all_verts[:, 1])
    L0 = max(x_range, y_range) / 1.5 + 2.0
    phi0 = 0.0
    
    best_result = None
    best_L = 1e10
    
    # Try multiple initial rotations
    for phi_init in [0, 30, 60, 90, 120, 150]:
        x0 = [cx0, cy0, L0, float(phi_init)]
        res = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
        L_val = res.x[2]
        viol = 0.0
        cx, cy, L, phi = res.x
        inradius = L * np.sqrt(3) / 2
        for k in range(6):
            theta = np.radians(phi) + k * np.pi / 3 + np.pi / 6
            dx = all_verts[:, 0] - cx
            dy = all_verts[:, 1] - cy
            projections = dx * np.cos(theta) + dy * np.sin(theta)
            v = np.max(projections) - inradius
            if v > viol:
                viol = v
        if viol < 1e-6 and L_val < best_L:
            best_L = L_val
            best_result = (cx, cy, L, phi)
    
    if best_result is None:
        # Fallback: use bounding box diagonal
        cx, cy = cx0, cy0
        L = max(x_range, y_range) * 0.6 + 2.0
        best_result = (cx, cy, L, 0.0)
    
    return best_result

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin
    
    # Strategy: place 12 unit hexagons in a hexagonal lattice pattern
    # Then optimize positions and find minimal enclosing hexagon
    
    # Pattern: hexagonal arrangement
    # Row 0: 4 hexagons
    # Row 1: 3 hexagons (offset)
    # Row 2: 3 hexagons (offset)
    # Row 3: 2 hexagons
    # That's 12. But let's try other patterns too.
    
    # Actually, let's use a grid: 3 rows of 4, or 4 rows of 3
    # For hexagonal lattice, adjacent rows are offset by half a spacing
    
    # Spacing between centers of touching hexagons = sqrt(3) * side = sqrt(3)
    # Wait: for regular hexagons of side 1, the distance between centers when sharing an edge is 2 * inradius = sqrt(3)
    
    d = np.sqrt(3)  # distance between centers of edge-touching hexagons
    
    best_score = 1e10
    best_solution = None
    
    # Try several arrangement patterns
    patterns = []
    
    # Pattern 1: 4x3 grid (4 columns, 3 rows) with hex offset
    for row_offset in [0, 0.5]:
        centers = []
        for row in range(3):
            for col in range(4):
                x = col * d + (row_offset if row % 2 == 1 else 0)
                y = row * d * 0.5  # Wait, for hex lattice, vertical spacing is d*sqrt(3)/2? No...
                # Actually for hexagonal lattice with horizontal spacing d, vertical spacing is d*sqrt(3)/2
                # Let me reconsider. If hexagons are flat-top, neighbors are at 60 degree intervals.
                # Center to center distance for edge-sharing = sqrt(3)
                # Neighbors at angles 0, 60, 120, 180, 240, 300 degrees
                # So horizontal neighbors at distance sqrt(3), vertical at distance sqrt(3)*sin(60) = sqrt(3)*sqrt(3)/2 = 1.5
                # And offset by sqrt(3)/2 horizontally
                pass
        # Let me use proper hex lattice coordinates
        # For flat-top hexagons, the 6 neighbors of a hexagon at (0,0) are at:
        # (sqrt(3), 0), (sqrt(3)/2, 1.5), (-sqrt(3)/2, 1.5), (-sqrt(3), 0), (-sqrt(3)/2, -1.5), (sqrt(3)/2, -1.5)
        # So the lattice vectors are a1 = (sqrt(3), 0) and a2 = (sqrt(3)/2, 1.5)
        
        centers = []
        for row in range(3):
            for col in range(4):
                x = col * d + row * d / 2
                y = row * 1.5
                centers.append((x, y))
        if len(centers) == 12:
            patterns.append(centers)
    
    # Pattern 2: 3x4 grid
    centers = []
    for row in range(4):
        for col in range(3):
            x = col * d + row * d / 2
            y = row * 1.5
            centers.append((x, y))
    if len(centers) == 12:
        patterns.append(centers)
    
    # Pattern 3: hexagonal cluster (1 center + 6 ring + 5 more)
    # Center at origin, 6 at distance d, then 5 more at distance 2d
    centers = [(0, 0)]
    for k in range(6):
        theta = k * np.pi / 3
        centers.append((d * np.cos(theta), d * np.sin(theta)))
    # Add 5 more at the "corners" of the hexagonal ring
    for k in range(6):
        theta = k * np.pi / 3 + np.pi / 6
        centers.append((2 * d * np.cos(theta), 2 * d * np.sin(theta)))
        if len(centers) == 12:
            break
    if len(centers) == 12:
        patterns.append(centers)
    
    # Pattern 4: 2 rows of 6
    centers = []
    for row in range(2):
        for col in range(6):
            x = col * d
            y = row * 1.5
            centers.append((x, y))
    if len(centers) == 12:
        patterns.append(centers)
    
    # Pattern 5: 6x2
    centers = []
    for row in range(6):
        for col in range(2):
            x = col * d + row * d / 2
            y = row * 1.5
            centers.append((x, y))
    if len(centers) == 12:
        patterns.append(centers)
    
    # For each pattern, optimize the enclosing hexagon
    for pattern in patterns:
        if time.time() > deadline - 10:
            break
        
        centers_arr = np.array(pattern)
        # Center the pattern
        centers_arr = centers_arr - np.mean(centers_arr, axis=0)
        
        angles = [0.0] * 12
        
        result = _min_enclosing_hexagon(centers_arr[:, 0].tolist(), centers_arr[:, 1].tolist(), 1.0, angles)
        cx, cy, L, phi = result
        
        # Verify and check time
        if L < best_score:
            best_score = L
            best_solution = (pattern, cx, cy, L, phi)
    
    # Now do local optimization on the best pattern's positions
    if best_solution is not None and time.time() < deadline - 30:
        pattern, cx, cy, L, phi = best_solution
        centers_arr = np.array(pattern)
        centers_arr = centers_arr - np.mean(centers_arr, axis=0)
        
        # Optimize all positions and the outer hexagon simultaneously
        def full_objective(params):
            # params: 12*(x,y) + cx, cy, L, phi = 24 + 4 = 28
            if time.time() > deadline - 5:
                return 1e10
            pos = params[:24].reshape(12, 2)
            cx, cy, L, phi = params[24], params[25], params[26], params[27]
            if L <= 0.5:
                return 1e10
            
            # Check non-overlap: distance between any two centers >= sqrt(3) (for axis-aligned)
            # But with rotation, it's more complex. For simplicity, assume all aligned.
            # Minimum distance between centers of non-overlapping unit hexagons (same orientation) is sqrt(3)
            min_dist = np.sqrt(3) - 1e-9
            for i in range(12):
                for j in range(i+1, 12):
                    dist = np.sqrt((pos[i,0]-pos[j,0])**2 + (pos[i,1]-pos[j,1])**2)
                    if dist < min_dist:
                        return 1e10 + (min_dist - dist) * 1000
            
            # Check all vertices inside outer hexagon
            inradius = L * np.sqrt(3) / 2
            max_violation = 0.0
            for i in range(12):
                verts = _hex_vertices(pos[i,0], pos[i,1], 1.0, 0.0)
                for v in verts:
                    for k in range(6):
                        theta = np.radians(phi) + k * np.pi / 3 + np.pi / 6
                        proj = (v[0]-cx)*np.cos(theta) + (v[1]-cy)*np.sin(theta)
                        if proj > inradius:
                            max_violation = max(max_violation, proj - inradius)
            
            if max_violation > 0:
                return L + 100 * max_violation
            return L
        
        x0 = np.zeros(28)
        x0[:24] = centers_arr.flatten()
        x0[24] = cx
        x0[25] = cy
        x0[26] = L
        x0[27] = phi
        
        try:
            res = minimize(full_objective, x0, method='Nelder-Mead',
                          options={'maxiter': 300, 'xatol': 1e-5, 'fatol': 1e-7, 'adaptive': True})
            L_val = res.x[26]
            if L_val < best_score:
                best_score = L_val
                pos = res.x[:24].reshape(12, 2)
                cx, cy, L, phi = res.x[24], res.x[25], res.x[26], res.x[27]
                best_solution = (pos.tolist(), cx, cy, L, phi)
        except:
            pass
    
    # Construct final answer
    if best_solution is None:
        # Fallback to seed
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    pattern, cx, cy, L, phi = best_solution
    if isinstance(pattern, list) and len(pattern) > 0 and isinstance(pattern[0], (list, tuple)):
        inner = [[p[0], p[1], 0.0] for p in pattern]
    else:
        inner = [[p[0], p[1], 0.0] for p in pattern]
    
    return inner, (cx, cy), L, np.degrees(phi)
# EVOLVE-BLOCK-END
