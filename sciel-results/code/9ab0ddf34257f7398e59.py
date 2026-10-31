"""Optimized hexagon packing: minimize outer hexagon side length for N unit hexagons."""
import numpy as np
import time
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START

def hexagon_vertices(cx, cy, s, angle_deg):
    """Return 6 vertices of a regular hexagon with side s, center (cx,cy), rotation angle_deg."""
    a = np.radians(angle_deg)
    # Regular hexagon vertices at angles 0, 60, 120, 180, 240, 300 degrees (flat-top orientation)
    # For a regular hexagon with side length s, vertices are at distance s from center
    angles = a + np.pi/3 * np.arange(6)
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def polygon_area(vertices):
    """Shoelace formula for polygon area."""
    n = len(vertices)
    x = vertices[:, 0]
    y = vertices[:, 1]
    return 0.5 * np.abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))

def polygons_overlap(v1, v2, tol=1e-9):
    """Check if two convex polygons overlap using SAT (Separating Axis Theorem)."""
    # Get all edges from both polygons
    axes = []
    for v in (v1, v2):
        n = len(v)
        for i in range(n):
            edge = v[(i+1) % n] - v[i]
            # Normal to edge
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-12:
                axes.append(normal / norm)
    
    for axis in axes:
        proj1 = np.dot(v1, axis)
        proj2 = np.dot(v2, axis)
        min1, max1 = proj1.min(), proj1.max()
        min2, max2 = proj2.min(), proj2.max()
        # Check for gap
        if max1 < min2 - tol or max2 < min1 - tol:
            return False, axis
    return True, None

def point_in_hexagon(px, py, cx, cy, s, angle_deg):
    """Check if point (px,py) is inside hexagon centered at (cx,cy) with side s and rotation."""
    # Transform to local coordinates
    dx, dy = px - cx, py - cy
    a = np.radians(angle_deg)
    cos_a, sin_a = np.cos(a), np.sin(a)
    # Rotate point into hexagon's local frame
    lx = dx * cos_a + dy * sin_a
    ly = -dx * sin_a + dy * cos_a
    # Regular hexagon with flat top: |lx| <= s, |ly| <= s*sqrt(3)/2, and |lx| + |ly|*2/sqrt(3) <= 2s
    # Actually for a regular hexagon with side s, the condition is:
    # max(|lx|, |ly| * 2/sqrt(3) - |lx|) <= s  ... let me use a simpler check
    # A regular hexagon with vertices at distance s from center:
    # The hexagon contains points where max over 6 half-plane constraints is satisfied
    # For flat-top hexagon (vertices at 0, 60, ...):
    # |lx| <= s
    # |ly| <= s * sqrt(3)/2
    # |lx| + sqrt(3)*|ly| <= 2*s  (this is for pointy-top)
    # Let me just use the vertex-based approach
    verts = hexagon_vertices(0, 0, s, 0)  # local frame, no rotation
    # Check if point is inside using cross products (convex polygon)
    n = len(verts)
    for i in range(n):
        v1 = verts[i]
        v2 = verts[(i+1) % n]
        edge = v2 - v1
        normal = np.array([-edge[1], edge[0]])
        # Check same side
        d1 = np.dot(normal, np.array([lx, ly]) - v1)
        d2 = np.dot(normal, verts[(i+2) % n] - v1)
        if d1 * d2 < 0:
            return False
    return True

def compute_outer_side_length(inner_hexagons, outer_center, outer_angle):
    """Compute minimum outer hexagon side length that contains all inner hexagons.
    inner_hexagons: list of (cx, cy, angle_deg)
    outer_center: (x, y)
    outer_angle: rotation of outer hexagon in degrees
    Returns the minimum side length, or inf if impossible."""
    ox, oy = outer_center
    oa = np.radians(outer_angle)
    
    # The outer hexagon has 6 sides. For each inner hexagon vertex,
    # we need it to be inside the outer hexagon.
    # A regular hexagon with center (ox,oy), side length S, rotation oa:
    # The 6 supporting lines are at distance S*sqrt(3)/2 from center (apothem = S*sqrt(3)/2)
    # Actually, for a regular hexagon with side length S, the apothem (distance from center to side) is S*sqrt(3)/2
    # The vertices are at distance S from center.
    # The 6 normals to the sides are at angles oa + 30 + 60*k degrees (for flat-top)
    # Let me think more carefully.
    
    # A regular hexagon with side length S:
    # - Circumradius (center to vertex) = S
    # - Inradius (center to side, apothem) = S * sqrt(3)/2
    # - The 6 sides have outward normals at angles: oa + 0, oa + 60, oa + 120, oa + 180, oa + 240, oa + 300
    #   Wait, depends on orientation. Let's say vertices are at angles oa + 0, oa+60, ...
    #   Then side normals are at oa + 30, oa + 90, ...
    
    # Let me define: outer hexagon vertices at angles oa + k*60 for k=0..5, at distance S from center.
    # The 6 side normals (outward) are at angles oa + 30 + k*60 for k=0..5.
    # A point P is inside if for all 6 sides: dot(P - center, normal_k) <= apothem = S*sqrt(3)/2
    
    normals = []
    for k in range(6):
        theta = oa + np.radians(30 + 60 * k)
        normals.append(np.array([np.cos(theta), np.sin(theta)]))
    
    # For each inner hexagon vertex, compute the required S
    # dot(P - O, n_k) <= S * sqrt(3)/2  =>  S >= dot(P - O, n_k) / (sqrt(3)/2) = 2*dot(P-O,n_k)/sqrt(3)
    # We need S >= max over all vertices and all normals of (2 * dot(P-O, n_k) / sqrt(3))
    # But also S must be >= 0
    
    min_S = 0.0
    for cx, cy, ang in inner_hexagons:
        verts = hexagon_vertices(cx, cy, 1.0, ang)
        for v in verts:
            for n in normals:
                d = np.dot(v - np.array([ox, oy]), n)
                if d > 0:
                    S_needed = 2.0 * d / np.sqrt(3.0)
                    if S_needed > min_S:
                        min_S = S_needed
    return min_S

def check_validity(inner_hexagons, outer_center, outer_side, outer_angle):
    """Check if the configuration is valid: no overlaps, all inside outer hexagon."""
    ox, oy = outer_center
    oa = np.radians(outer_angle)
    
    # Check no overlaps between inner hexagons
    for i in range(len(inner_hexagons)):
        for j in range(i+1, len(inner_hexagons)):
            cx1, cy1, a1 = inner_hexagons[i]
            cx2, cy2, a2 = inner_hexagons[j]
            v1 = hexagon_vertices(cx1, cy1, 1.0, a1)
            v2 = hexagon_vertices(cx2, cy2, 1.0, a2)
            overlap, _ = polygons_overlap(v1, v2, tol=1e-8)
            if overlap:
                return False
    
    # Check all inner hexagons are inside outer hexagon
    # Outer hexagon: center (ox,oy), side outer_side, rotation outer_angle
    # Vertices of outer hexagon
    outer_verts = hexagon_vertices(ox, oy, outer_side, outer_angle)
    
    for cx, cy, ang in inner_hexagons:
        verts = hexagon_vertices(cx, cy, 1.0, ang)
        for v in verts:
            px, py = v
            # Check if point is inside outer hexagon using SAT or point-in-polygon
            # Use ray casting or cross product method
            n = len(outer_verts)
            inside = True
            for i in range(n):
                v1 = outer_verts[i]
                v2 = outer_verts[(i+1) % n]
                edge = v2 - v1
                normal = np.array([-edge[1], edge[0]])
                d1 = np.dot(normal, np.array([px, py]) - v1)
                d2 = np.dot(normal, outer_verts[(i+2) % n] - v1)
                if d1 * d2 < 0:
                    inside = False
                    break
            if not inside:
                return False
    
    return True

def objective(params):
    """Objective function: outer side length + penalty for overlaps and containment."""
    # params layout: for each inner hexagon: (cx, cy, angle_rad), then outer: (ox, oy, outer_angle_rad)
    # Total: 3*N + 3 parameters
    n = N
    inner = []
    for i in range(n):
        cx = params[3*i]
        cy = params[3*i+1]
        ang = np.degrees(params[3*i+2]) % 360
        inner.append((cx, cy, ang))
    ox = params[3*n]
    oy = params[3*n+1]
    outer_angle = np.degrees(params[3*n+2]) % 360
    
    # Compute required outer side length
    S = compute_outer_side_length(inner, (ox, oy), outer_angle)
    
    # Penalty for overlaps
    penalty = 0.0
    for i in range(n):
        for j in range(i+1, n):
            cx1, cy1, a1 = inner[i]
            cx2, cy2, a2 = inner[j]
            v1 = hexagon_vertices(cx1, cy1, 1.0, a1)
            v2 = hexagon_vertices(cx2, cy2, 1.0, a2)
            # Approximate overlap penalty using minimum distance
            # Use centroid distance as a quick check
            d = np.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)
            if d < 2.0:  # Potential overlap
                overlap, _ = polygons_overlap(v1, v2, tol=1e-8)
                if overlap:
                    penalty += (2.0 - d) ** 2 * 100
    
    # Penalty for containment (if S is computed, it should contain them by construction)
    # But we add a small penalty if any vertex is slightly outside due to numerical issues
    return S + penalty

def construct():
    """Main entry point: find optimal packing of N unit hexagons in minimum outer hexagon."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Initial guess: hexagonal lattice arrangement
    # For N hexagons, use a hexagonal packing pattern
    # Positions in a hex grid with spacing ~2 (center-to-center for non-overlapping unit hexagons)
    # Actually, for regular hexagons of side 1, the minimum center-to-center distance is 2*apothem = 2*sqrt(3)/2 = sqrt(3) ≈ 1.732
    # Wait, two adjacent hexagons sharing an edge: center-to-center = 2 * apothem = 2 * (sqrt(3)/2) = sqrt(3)
    
    # Create hexagonal lattice positions
    # Hex grid coordinates: for row r, col c
    # Position: x = c * sqrt(3) + (r % 2) * sqrt(3)/2, y = r * 1.5
    # Actually for flat-top hexagons:
    # Horizontal spacing: sqrt(3), vertical spacing: 1.5
    # Offset every other row by sqrt(3)/2
    
    positions = []
    # Use a hexagonal number arrangement
    # For N=11, use center + ring of 6 + partial ring of 4
    # Or just use a simple hex grid
    hex_positions = [(0, 0)]  # center
    # Ring 1: 6 positions
    for k in range(6):
        angle = np.radians(60 * k)
        x = np.sqrt(3) * np.cos(angle)
        y = np.sqrt(3) * np.sin(angle)
        hex_positions.append((x, y))
    # Ring 2: need 4 more for total 11
    for k in range(6):
        angle = np.radians(60 * k)
        x = 2 * np.sqrt(3) * np.cos(angle)
        y = 2 * np.sqrt(3) * np.sin(angle)
        hex_positions.append((x, y))
        if len(hex_positions) >= N:
            break
    
    hex_positions = hex_positions[:N]
    
    # Initial parameters
    x0 = np.zeros(3 * N + 3)
    for i, (px, py) in enumerate(hex_positions):
        x0[3*i] = px
        x0[3*i+1] = py
        x0[3*i+2] = 0.0  # angle in radians
    x0[3*N] = 0.0  # outer center x
    x0[3*N+1] = 0.0  # outer center y
    x0[3*N+2] = 0.0  # outer angle
    
    # Try multiple initial configurations
    best_result = None
    best_score = float('inf')
    
    # Strategy 1: Optimize from hexagonal lattice
    results = []
    
    # Run optimization with different initial conditions
    initial_conditions = []
    
    # Condition 1: Hexagonal lattice, no rotation
    ic1 = x0.copy()
    initial_conditions.append(ic1)
    
    # Condition 2: Hexagonal lattice with slight random perturbation
    np.random.seed(42)
    ic2 = x0 + np.random.randn(3*N+3) * 0.1
    initial_conditions.append(ic2)
    
    # Condition 3: Linear arrangement
    ic3 = np.zeros(3*N+3)
    for i in range(N):
        ic3[3*i] = i * np.sqrt(3)
        ic3[3*i+1] = 0
        ic3[3*i+2] = 0
    ic3[3*N] = (N-1) * np.sqrt(3) / 2
    ic3[3*N+1] = 0
    ic3[3*N+2] = 0
    initial_conditions.append(ic3)
    
    # Condition 4: Two rows
    ic4 = np.zeros(3*N+3)
    for i in range(N):
        row = i // 6
        col = i % 6
        ic4[3*i] = col * np.sqrt(3) + (row % 2) * np.sqrt(3)/2
        ic4[3*i+1] = row * 1.5
        ic4[3*i+2] = 0
    ic4[3*N] = 2.5 * np.sqrt(3)
    ic4[3*N+1] = 0.75
    ic4[3*N+2] = 0
    initial_conditions.append(ic4)
    
    for ic in initial_conditions:
        if time.time() > deadline - 30:
            break
        
        try:
            res = minimize(objective, ic, method='L-BFGS-B', 
                          options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if res.success or res.fun < best_score:
                params = res.x
                inner = []
                for i in range(N):
                    cx = params[3*i]
                    cy = params[3*i+1]
                    ang = np.degrees(params[3*i+2]) % 360
                    inner.append([cx, cy, ang])
                ox = params[3*N]
                oy = params[3*N+1]
                outer_angle = np.degrees(params[3*N+2]) % 360
                
                # Compute the actual outer side length
                S = compute_outer_side_length(inner, (ox, oy), outer_angle)
                
                # Verify validity
                if check_validity(inner, (ox, oy), S, outer_angle):
                    if S < best_score:
                        best_score = S
                        best_result = (inner, (ox, oy), S, outer_angle)
                        results.append((S, inner, (ox, oy), outer_angle))
        except Exception as e:
            pass
    
    # If no valid result found, use a fallback
    if best_result is None:
        # Fallback: use the initial hexagonal arrangement with a large enough outer hexagon
        inner = []
        for i, (px, py) in enumerate(hex_positions):
            inner.append([px, py, 0.0])
        ox, oy = 0.0, 0.0
        outer_angle = 0.0
        S = compute_outer_side_length(inner, (ox, oy), outer_angle)
        # Add some margin
        S += 0.1
        best_result = (inner, (ox, oy), S, outer_angle)
    
    # Time check
    if time.time() > deadline:
        return best_result
    
    # Try to refine with additional optimization if time permits
    if time.time() < deadline - 60:
        try:
            # Refine the best result
            params_best = np.zeros(3*N+3)
            for i, (cx, cy, ang) in enumerate(best_result[0]):
                params_best[3*i] = cx
                params_best[3*i+1] = cy
                params_best[3*i+2] = np.radians(ang)
            params_best[3*N] = best_result[1][0]
            params_best[3*N+1] = best_result[1][1]
            params_best[3*N+2] = np.radians(best_result[3])
            
            res = minimize(objective, params_best, method='L-BFGS-B',
                          options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10})
            
            if res.fun < best_score:
                params = res.x
                inner = []
                for i in range(N):
                    cx = params[3*i]
                    cy = params[3*i+1]
                    ang = np.degrees(params[3*i+2]) % 360
                    inner.append([cx, cy, ang])
                ox = params[3*N]
                oy = params[3*N+1]
                outer_angle = np.degrees(params[3*N+2]) % 360
                S = compute_outer_side_length(inner, (ox, oy), outer_angle)
                
                if check_validity(inner, (ox, oy), S, outer_angle):
                    best_result = (inner, (ox, oy), S, outer_angle)
                    best_score = S
        except Exception:
            pass
    
    return best_result

# EVOLVE-BLOCK-END
