"""Packing 12 unit hexagons in a regular hexagon - minimize outer side length."""
import numpy as np
from scipy.optimize import minimize, linprog
import time

N = 12

def hex_vertices(cx, cy, s, angle_deg=0.0):
    """Vertices of regular hexagon centered at (cx,cy), side s, rotation angle_deg."""
    a = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * np.pi / 3
        verts.append([cx + s * np.cos(theta), cy + s * np.sin(theta)])
    return np.array(verts)

def project_polygon(vertices, axis):
    """Project polygon vertices onto axis."""
    projections = np.dot(vertices, axis)
    return min(projections), max(projections)

def sat_overlap(v1, v2):
    """
    Check if two convex polygons (given as arrays of vertices) overlap using SAT.
    Returns True if they overlap.
    """
    # Get edges for v1 and v2 to define axes
    axes = []
    for v in [v1, v2]:
        for i in range(len(v)):
            p1 = v[i]
            p2 = v[(i + 1) % len(v)]
            edge = p2 - p1
            # Normal to edge
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-10:
                axes.append(normal / norm)
    
    for axis in axes:
        min1, max1 = project_polygon(v1, axis)
        min2, max2 = project_polygon(v2, axis)
        # Check for separation
        if max1 < min2 or max2 < min1:
            return False
    return True

def check_overlap(i, j, positions, angles, s=1.0):
    """Check if hexagons i and j overlap."""
    vi = hex_vertices(positions[i,0], positions[i,1], s, angles[i])
    vj = hex_vertices(positions[j,0], positions[j,1], s, angles[j])
    return sat_overlap(vi, vj)

def bounding_hexagon_side(positions, angles, outer_angle=0.0):
    """
    Find minimum side length of outer hexagon containing all inner hexagons.
    Uses Linear Programming to find the smallest hexagon (fixed orientation) 
    enclosing all vertices of the inner hexagons.
    """
    s = 1.0
    all_points = []
    for i in range(N):
        vi = hex_vertices(positions[i,0], positions[i,1], s, angles[i])
        all_points.append(vi)
    all_points = np.vstack(all_points)
    
    # Outer hexagon constraints in local frame (centered at origin, rotated by outer_angle):
    # The hexagon is defined by 3 pairs of parallel lines.
    # Let the outer center be (cx, cy) in global coords.
    # Let the local coordinates of a point p be (lx, ly) after rotating by -outer_angle.
    # The hexagon with side S has apothem A = S * sqrt(3) / 2.
    # The constraints for a point (lx, ly) to be inside a hexagon centered at (u, v) in local frame are:
    # |lx - u| <= A
    # |(lx - u)*cos(30) + (ly - v)*sin(30)| <= A
    # |(lx - u)*cos(30) - (ly - v)*sin(30)| <= A
    # Wait, the normals for a "flat-top" hexagon (vertices at 0, 60...) are at 90, 30, -30?
    # Let's check hex_vertices: vertices at 0, 60, 120...
    # Edge between 0 and 60 deg. Midpoint angle 30. Normal is 30+90=120?
    # Actually, for a regular hexagon with vertices at angles $k \cdot 60^\circ$,
    # the supporting lines are at angles $30^\circ + k \cdot 60^\circ$.
    # The distances from center to these lines is the apothem $A = S \frac{\sqrt{3}}{2}$.
    
    a = np.radians(outer_angle)
    ca, sa = np.cos(-a), np.sin(-a)
    
    px, py = all_points[:,0], all_points[:,1]
    lx = px * ca - py * sa
    ly = px * sa + py * ca
    
    # We want to find center (u, v) in local frame and scale S (or apothem A)
    # such that all points are inside.
    # Minimize S (which is proportional to A).
    # Variables: u, v, A (apothem)
    # Objective: minimize A
    # Constraints for each point (lx_i, ly_i):
    # 1. |(lx_i - u) * cos(30) + (ly_i - v) * sin(30)| <= A
    # 2. |(lx_i - u) * cos(150) + (ly_i - v) * sin(150)| <= A  (which is -cos30, sin30)
    # 3. |(lx_i - u) * cos(270) + (ly_i - v) * sin(270)| <= A  (which is 0, -1)
    
    # Let's define the 3 normal vectors for the hexagon edges.
    # Vertices at 0, 60, 120, 180, 240, 300.
    # Edge 1 (0-60): Normal at 30 deg. n1 = (cos30, sin30)
    # Edge 2 (60-120): Normal at 90 deg. n2 = (0, 1)
    # Edge 3 (120-180): Normal at 150 deg. n3 = (-cos30, sin30)
    # Edge 4 (180-240): Normal at 210 deg. -n1
    # Edge 5 (240-300): Normal at 270 deg. -n2
    # Edge 6 (300-0): Normal at 330 deg. -n3
    
    c30, s30 = np.cos(np.pi/6), np.sin(np.pi/6)
    normals = np.array([
        [c30, s30],
        [0, 1],
        [-c30, s30]
    ])
    
    n_pts = len(px)
    A_ub = []
    b_ub = []
    
    # Variables: u, v, A
    # For each point i and each normal n_k:
    # | dot(n_k, [lx_i - u, ly_i - v]) | <= A
    # dot(n_k, [lx_i, ly_i]) - dot(n_k, [u, v]) <= A
    # dot(n_k, [u, v]) - dot(n_k, [lx_i, ly_i]) <= A
    
    for i in range(n_pts):
        for k in range(3):
            nk = normals[k]
            val = lx[i]*nk[0] + ly[i]*nk[1]
            
            # Constraint 1: dot(n_k, [u,v]) - val <= A  =>  dot(n_k, [u,v]) - A <= val
            A_ub.append([nk[0], nk[1], -1.0])
            b_ub.append(val)
            
            # Constraint 2: val - dot(n_k, [u,v]) <= A  =>  -dot(n_k, [u,v]) - A <= -val
            A_ub.append([-nk[0], -nk[1], -1.0])
            b_ub.append(-val)
            
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    # Objective: minimize A
    c = np.array([0.0, 0.0, 1.0])
    
    # Bounds: A >= 0
    bounds = [(None, None), (None, None), (0, None)]
    
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if res.success:
        S = res.x[2] * 2 / np.sqrt(3) # A = S * sqrt(3)/2 => S = 2*A/sqrt(3)
        # The center in local frame is (res.x[0], res.x[1])
        # Convert back to global frame
        cx_local, cy_local = res.x[0], res.x[1]
        # Rotate back by outer_angle
        ca_pos, sa_pos = np.cos(a), np.sin(a)
        cx_global = cx_local * ca_pos - cy_local * sa_pos
        cy_global = cx_local * sa_pos + cy_local * ca_pos
        return S, cx_global, cy_global
    else:
        return float('inf'), 0, 0

def objective(x, fixed_angles=None):
    """Objective: minimize outer hexagon side length with overlap penalty."""
    positions = x[:2*N].reshape(N, 2)
    if fixed_angles is not None:
        angles = fixed_angles
    else:
        angles = x[2*N:]
    
    # Check overlaps and add penalty
    overlap_penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            if check_overlap(i, j, positions, angles):
                overlap_penalty += 100.0
                
    # Find bounding hexagon
    best_S = float('inf')
    # Try several outer orientations to find the tightest fit
    # For regular hexagon packing, aligning with lattice is often good, but we can search.
    # To save time, we check 0, 30, 60 degrees.
    for oa in [0.0, 30.0, 60.0]:
        S, _, _ = bounding_hexagon_side(positions, angles, oa)
        if S < best_S:
            best_S = S
            
    return best_S + overlap_penalty

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    best_S = float('inf')
    best_positions = None
    best_angles = None
    best_cx = 0.0
    best_cy = 0.0
    best_outer_angle = 0.0

    # Initial Configuration: Hexagonal Lattice
    # 12 hexagons.
    # A compact arrangement is a hexagon of side length 3 (which holds 1+3+5+3+1=13 hexagons? No, 1+3+5+3+1 is 13).
    # Side 2 hexagon holds 1+3+1 = 5? No.
    # Number of hexagons in hexagon of side n: 3n(n-1)+1.
    # n=1: 1
    # n=2: 7
    # n=3: 19
    # So 12 hexagons can fit in a side 3 hexagon (19 spots) with some empty.
    # Let's construct a specific valid layout.
    
    # Layout:
    #   . X X X .
    #   X X X X X
    #   X X X X X
    #   . X X X .
    # This is 3+5+5+3 = 16? No, 3+5+5+3 is 16.
    # Let's try:
    #   . X X X .
    #   . X X X .
    #   X X X X X
    #   X X X X X
    #   . X X X .
    # 3+3+5+5+3 = 19.
    
    # Let's just use a grid and trim.
    # Spacing: sqrt(3)
    # Vertical spacing: 1.5
    
    spacing = np.sqrt(3)
    v_spacing = 1.5
    
    # Generate a 4x4 grid centered at 0
    pos_list = []
    for r in range(4):
        for c in range(4):
            x = (c - 1.5) * spacing
            y = (r - 1.5) * v_spacing
            # Offset odd rows
            if r % 2 == 1:
                x += spacing / 2
            pos_list.append([x, y])
            
    # We have 16 positions. We need 12.
    # Remove 4 corners to make it more hexagonal/compact?
    # The 4 corners are (0,0), (0,3), (3,0), (3,3) in grid indices.
    # Let's keep the center 12.
    # Center of 4x4 is between (1,1), (1,2), (2,1), (2,2).
    # Let's just take the first 12 generated? No, that's a rectangle.
    # Let's pick specific indices for a hexagonal shape.
    # Indices in 4x4 (0..3, 0..3):
    # 0 1 2 3
    # 4 5 6 7
    # 8 9 10 11
    # 12 13 14 15
    
    # Hexagon of side 2 has 7 hexes.
    # Hexagon of side 3 has 19 hexes.
    # We want 12.
    # Take the 7 from center of side 3, plus 5 more?
    
    # Simpler: Use the 4x4 grid but remove the 4 extreme corners: (0,0), (0,3), (3,0), (3,3).
    # This leaves 12 hexagons in a "rounded square" shape.
    # This is a valid non-overlapping configuration.
    
    keep_indices = [1, 2, 5, 6, 9, 10, 4, 7, 8, 11, 13, 14]
    # Let's verify these form a connected compact shape.
    # Row 0: 1, 2 (2 hexes)
    # Row 1: 5, 6 (2 hexes) -- wait, 4,5,6,7 are row 1. 5,6 are middle.
    # Row 2: 9, 10 (2 hexes)
    # Row 3: 13, 14 (2 hexes)
    # Plus 4, 7, 8, 11?
    # My list: 1,2 (row 0 mid), 5,6 (row 1 mid), 9,10 (row 2 mid), 13,14 (row 3 mid) -> 8 hexes.
    # 4,7 (row 1 sides), 8,11 (row 2 sides) -> 4 hexes.
    # Total 12.
    # This forms a 2-wide column in the middle, and 1-wide on sides?
    # Let's just use the full 4x4 grid and let the optimizer remove/shift? No, N is fixed.
    # I will use the 12 points from the 4x4 grid excluding corners.
    
    final_pos = np.array([pos_list[i] for i in keep_indices])
    
    angles = np.zeros(N)
    
    # Evaluate initial
    S_init, cx_init, cy_init = bounding_hexagon_side(final_pos, angles, 0.0)
    best_S = S_init
    best_positions = final_pos
    best_cx = cx_init
    best_cy = cy_init
    best_outer_angle = 0.0

    # Optimization
    x0 = final_pos.flatten()
    
    def obj_func(x):
        positions = x.reshape(N, 2)
        # Quick overlap check for penalty
        penalty = 0
        for i in range(N):
            for j in range(i+1, N):
                if check_overlap(i, j, positions, angles):
                    penalty += 100
        S_val, _, _ = bounding_hexagon_side(positions, angles, 0.0)
        return S_val + penalty

    try:
        res = minimize(obj_func, x0, method='Nelder-Mead', 
                      options={'maxiter': 2000, 'xatol': 1e-3, 'fatol': 1e-3})
        
        if res.fun < best_S:
            best_S = res.fun - 100 * 0 # Subtract penalty if we assume it's 0? 
            # Better: re-evaluate without penalty to get true S
            pos_opt = res.x.reshape(N, 2)
            # Verify no overlap
            has_overlap = False
            for i in range(N):
                for j in range(i+1, N):
                    if check_overlap(i, j, pos_opt, angles):
                        has_overlap = True
            if not has_overlap:
                S_opt, cx_opt, cy_opt = bounding_hexagon_side(pos_opt, angles, 0.0)
                if S_opt < best_S:
                    best_S = S_opt
                    best_positions = pos_opt
                    best_cx = cx_opt
                    best_cy = cy_opt
    except Exception as e:
        pass

    # Final Result
    inner = []
    for i in range(N):
        inner.append([best_positions[i,0], best_positions[i,1], angles[i] * 180.0 / np.pi])
    
    return inner, (best_cx, best_cy), best_S, best_outer_angle
# EVOLVE-BLOCK-END
