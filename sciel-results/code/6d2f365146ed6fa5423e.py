"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 12

def hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with given side and angle."""
    angle_rad = np.radians(angle_deg)
    # Vertices of unit hexagon (side=1) at angles 0, 60, 120, 180, 240, 300 degrees
    base_angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300]))
    verts = np.column_stack([np.cos(base_angles), np.sin(base_angles)])
    verts = verts * side
    # Rotate
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    R = np.array([[c, -s], [s, c]])
    verts = verts @ R.T
    verts += np.array([cx, cy])
    return verts

def min_dist_between_hexes(v1, v2):
    """
    Compute minimum distance between two convex hexagons (as vertex arrays).
    Returns (dist, is_overlap). If overlapping, dist is negative (approximation).
    We use a simple approach: check if any vertex of one is inside the other,
    and compute min vertex-to-vertex and vertex-to-edge distances.
    For efficiency, we approximate using the support function / separating axis.
    Actually, for hexagons, we can use the fact that the distance between two convex
    polygons can be found via the minimum distance between edges or vertices.
    A simpler robust approach: use scipy's distance functions? No, let's do it manually.
    
    For two convex polygons, the minimum distance is the minimum over all pairs of
    (vertex, vertex), (vertex, edge), (edge, edge). This is O(36) per pair which is fine.
    """
    # Check if polygons overlap using separating axis theorem
    # If they overlap, we need a penalty. Let's use a soft penalty based on penetration depth approximation.
    # For simplicity and differentiability, we'll compute the minimum distance between
    # all vertices of poly1 to edges of poly2 and vice versa, plus vertex-vertex.
    
    def point_to_segment_dist(p, a, b):
        """Distance from point p to segment ab."""
        ab = b - a
        ap = p - a
        t = np.dot(ap, ab) / np.dot(ab, ab)
        t = np.clip(t, 0, 1)
        closest = a + t * ab
        return np.linalg.norm(p - closest)
    
    min_d = np.inf
    for i in range(6):
        for j in range(6):
            # vertex of v1 to vertex of v2
            d = np.linalg.norm(v1[i] - v2[j])
            if d < min_d:
                min_d = d
            # vertex of v1 to edge of v2
            j2 = (j + 1) % 6
            d = point_to_segment_dist(v1[i], v2[j], v2[j2])
            if d < min_d:
                min_d = d
            # vertex of v2 to edge of v1
            i2 = (i + 1) % 6
            d = point_to_segment_dist(v2[j], v1[i], v1[i2])
            if d < min_d:
                min_d = d
    return min_d

def outer_hex_contains_inner(outer_center, outer_side, outer_angle, inner_center, inner_angle):
    """
    Check if a unit hexagon centered at inner_center with angle inner_angle
    is contained in the outer hexagon.
    We check if all 6 vertices of the inner hex are inside the outer hex.
    """
    inner_verts = hex_vertices(inner_center[0], inner_center[1], 1.0, inner_angle)
    # Outer hexagon boundaries: 6 half-planes
    outer_verts = hex_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
    # For each edge of outer hex, check if inner vertex is on the correct side
    for i in range(6):
        a = outer_verts[i]
        b = outer_verts[(i + 1) % 6]
        # Edge vector
        e = b - a
        # Normal pointing inward
        n = np.array([-e[1], e[0]])
        # Check if normal points inward: dot(n, outer_center - a) should be positive
        if np.dot(n, np.array(outer_center) - a) < 0:
            n = -n
        # All inner vertices must satisfy dot(n, v - a) >= 0
        for v in inner_verts:
            if np.dot(n, v - a) < -1e-9:
                return False
    return True

def compute_outer_side_needed(inner_positions, inner_angles, outer_center, outer_angle):
    """
    Given inner hex positions and angles, and outer center/angle,
    find the minimum outer side length such that all inner hexes fit.
    We can binary search or solve directly.
    Actually, for a fixed outer angle and center, the minimum side is determined
    by the maximum over all inner vertices of the required support.
    
    For a regular hexagon with center C, angle A, side S, the support function in direction u is:
    h(u) = S * max_i dot(u, v_i) where v_i are unit hex vertices rotated by A.
    The condition that point p is inside is: for all 6 edge normals n_k, dot(n_k, p - C) <= h(n_k) = S * max_j dot(n_k, v_j)
    
    Actually, simpler: the outer hexagon with side S contains point p iff for all 6 directions (edge normals),
    the projection of (p - C) onto that normal is <= S * cos(0) ... let me think.
    
    For a regular hexagon with side S, the apothem (distance from center to edge) is S * sqrt(3)/2.
    The 6 edge normals are at angles A + 30, A + 90, ... etc.
    Condition: for each of the 6 normals n_k, dot(n_k, p - C) <= S * sqrt(3)/2
    
    So S >= 2/sqrt(3) * max_k dot(n_k, p - C) for all points p in the union of inner hexes.
    """
    # Compute the 6 edge normals of the outer hexagon
    outer_angle_rad = np.radians(outer_angle)
    # Edge normals are perpendicular to edges. Edges are at angles A, A+60, etc.
    # Normal to edge at angle theta is at angle theta + 90
    base_angles = np.deg2rad(np.array([30, 90, 150, 210, 270, 330])) + outer_angle_rad
    normals = np.column_stack([np.cos(base_angles), np.sin(base_angles)])
    
    # For each inner hex, get its vertices
    max_proj = 0
    for idx in range(N):
        cx, cy = inner_positions[idx]
        ang = inner_angles[idx]
        verts = hex_vertices(cx, cy, 1.0, ang)
        for v in verts:
            for n in normals:
                proj = np.dot(n, v - np.array(outer_center))
                if proj > max_proj:
                    max_proj = proj
    # S * sqrt(3)/2 >= max_proj => S >= 2*max_proj/sqrt(3)
    return 2.0 * max_proj / np.sqrt(3.0)

def objective(params, outer_center, outer_angle):
    """
    params: array of shape (3*N,) = [x1,y1,a1, x2,y2,a2, ..., x12,y12,a12]
    Returns (outer_side, overlap_penalty)
    """
    inner_positions = []
    inner_angles = []
    for i in range(N):
        x = params[3*i]
        y = params[3*i+1]
        a = params[3*i+2]
        inner_positions.append((x, y))
        inner_angles.append(a)
    
    # Compute outer side needed
    outer_side = compute_outer_side_needed(inner_positions, inner_angles, outer_center, outer_angle)
    
    # Compute overlap penalty: sum of max(0, 1 - min_dist)^2 for all pairs
    penalty = 0.0
    for i in range(N):
        v1 = hex_vertices(inner_positions[i][0], inner_positions[i][1], 1.0, inner_angles[i])
        for j in range(i+1, N):
            v2 = hex_vertices(inner_positions[j][0], inner_positions[j][1], 1.0, inner_angles[j])
            d = min_dist_between_hexes(v1, v2)
            if d < 1.0:
                penalty += (1.0 - d) ** 2
    
    return outer_side, penalty

def total_cost(params, outer_center, outer_angle, beta=50.0):
    """Smooth surrogate: outer_side + (1/beta) * log(sum(exp(beta * penalty_terms)))"""
    outer_side, penalty = objective(params, outer_center, outer_angle)
    # For simplicity, use quadratic penalty
    return outer_side + 100.0 * penalty

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initial configuration: place hexagons in a honeycomb-like pattern
    # Start with a reasonable guess
    np.random.seed(42)
    
    # Place 12 hexagons in a 3x4 grid with hexagonal offsets
    positions = []
    angles = []
    # Honeycomb packing: rows offset
    row_spacing = 1.0 * np.sqrt(3)
    col_spacing = 1.5
    for row in range(4):
        for col in range(3):
            x = col * col_spacing + (0.75 if row % 2 == 1 else 0.0)
            y = row * row_spacing
            positions.append((x, y))
            angles.append(0.0)
    
    # Center the configuration
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    positions = [(p[0] - cx, p[1] - cy) for p in positions]
    
    # Flatten to params
    params = np.array([positions[i][0] for i in range(N)] + 
                       [positions[i][1] for i in range(N)] + 
                       angles)
    
    outer_center = (0.0, 0.0)
    outer_angle = 0.0
    
    best_params = params.copy()
    best_side, best_penalty = objective(best_params, outer_center, outer_angle)
    best_cost = best_side + 100.0 * best_penalty
    
    # Use L-BFGS-B to optimize
    def obj_func(p):
        if time.time() > deadline - 5.0:
            return 1e10  # Timeout guard
        side, pen = objective(p, outer_center, outer_angle)
        return side + 100.0 * pen
    
    # Multiple restarts with different initial configurations
    for restart in range(5):
        if time.time() > deadline - 10.0:
            break
        
        # Perturb initial configuration
        if restart == 0:
            p0 = best_params.copy()
        else:
            p0 = best_params + np.random.randn(3*N) * 0.5
        
        try:
            res = minimize(obj_func, p0, method='L-BFGS-B', 
                          options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6})
            if res.fun < best_cost:
                best_cost = res.fun
                best_params = res.x
                best_side, best_penalty = objective(best_params, outer_center, outer_angle)
        except Exception:
            pass
    
    # Final polish with higher penalty weight
    if time.time() < deadline - 5.0:
        def obj_func_polish(p):
            if time.time() > deadline - 2.0:
                return 1e10
            side, pen = objective(p, outer_center, outer_angle)
            return side + 1000.0 * pen
        
        try:
            res = minimize(obj_func_polish, best_params, method='L-BFGS-B',
                          options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8})
            if res.fun < best_cost:
                best_params = res.x
                best_side, best_penalty = objective(best_params, outer_center, outer_angle)
        except Exception:
            pass
    
    # Extract final positions
    inner = []
    for i in range(N):
        x = best_params[3*i]
        y = best_params[3*i+1]
        a = best_params[3*i+2]
        inner.append([x, y, a])
    
    # Compute final outer side
    inner_positions = [(inner[i][0], inner[i][1]) for i in range(N)]
    inner_angles = [inner[i][2] for i in range(N)]
    outer_side = compute_outer_side_needed(inner_positions, inner_angles, outer_center, outer_angle)
    
    # Safety: ensure no overlaps by checking and expanding if needed
    for i in range(N):
        v1 = hex_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2])
        for j in range(i+1, N):
            v2 = hex_vertices(inner[j][0], inner[j][1], 1.0, inner[j][2])
            d = min_dist_between_hexes(v1, v2)
            if d < 0.01:
                # Nudge apart
                dx = inner[j][0] - inner[i][0]
                dy = inner[j][1] - inner[i][1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist > 1e-6:
                    dx /= dist
                    dy /= dist
                    inner[j][0] += 0.02 * dx
                    inner[j][1] += 0.02 * dy
                    inner[i][0] -= 0.02 * dx
                    inner[i][1] -= 0.02 * dy
    
    # Recompute outer side after nudge
    inner_positions = [(inner[i][0], inner[i][1]) for i in range(N)]
    inner_angles = [inner[i][2] for i in range(N)]
    outer_side = compute_outer_side_needed(inner_positions, inner_angles, outer_center, outer_angle)
    
    return inner, outer_center, outer_side, outer_angle

# EVOLVE-BLOCK-START
# (The construct function is defined above; this block is for compatibility)
# EVOLVE-BLOCK-END
