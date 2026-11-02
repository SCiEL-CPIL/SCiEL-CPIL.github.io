"""Packing 12 unit hexagons in a regular hexagon.
Combined approach: lattice-based initialization + penalty-based local optimization
with support-function containment, plus outer-orientation sweep.
"""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 12
SQRT3 = math.sqrt(3.0)

def hexagon_vertices(cx, cy, side, angle_rad):
    """Return 6 vertices of a regular hexagon (pointy-top orientation at angle 0)."""
    angles = angle_rad + np.radians(np.arange(0, 360, 60))
    return np.column_stack([
        cx + side * np.cos(angles),
        cy + side * np.sin(angles)
    ])

def hex_support_function(center, side, angle_rad, directions):
    """Support function of a unit hexagon at given center/side/angle in given directions.
    directions: (M, 2) array. Returns (M,) array of max dot products."""
    # Vertices
    angles = angle_rad + np.radians(np.arange(0, 360, 60))
    verts = np.column_stack([
        center[0] + side * np.cos(angles),
        center[1] + side * np.sin(angles)
    ])
    # max over vertices for each direction
    return verts @ directions.T  # (M, 6) -> take max along axis 1
    # Actually: verts.T @ directions gives (6, M), then max over axis 0
    # Let's do: (directions @ verts.T).max(axis=1)
    # directions: (M,2), verts: (6,2)
    # directions @ verts.T: (M, 6)
    # max over axis=1: (M,)

def compute_support(center, side, angle_rad, directions):
    """Compute support function values for a hexagon in given directions."""
    angles = angle_rad + np.radians(np.arange(0, 360, 60))
    verts = np.column_stack([
        center[0] + side * np.cos(angles),
        center[1] + side * np.sin(angles)
    ])
    # directions: (M, 2), verts: (6, 2)
    dots = directions @ verts.T  # (M, 6)
    return dots.max(axis=1)  # (M,)

def outer_side_from_support(inner_centers, inner_angles, inner_sides, outer_cx, outer_cy, outer_angle_rad):
    """Compute minimum outer hexagon side length to contain all inner hexagons.
    Uses support functions in 6 outer-edge normal directions."""
    # Outer hexagon edge normals (for a regular hexagon with center (cx,cy) and angle theta)
    # The 6 outward normals of the outer hexagon edges
    theta = outer_angle_rad
    normals = np.column_stack([
        np.cos(theta + np.radians(np.arange(0, 360, 60))),
        np.sin(theta + np.radians(np.arange(0, 360, 60)))
    ])
    
    max_required = 0.0
    for k in range(6):
        n = normals[k]  # outward normal of edge k
        # For containment: for each inner hex, support in direction n must be <= 
        # outer support in direction n
        # Outer hex support in direction n: cx*n[0] + cy*n[1] + side * (support of unit hex in n)
        # For a regular hexagon with side S and angle theta, the support in direction n (which is one of its edge normals) is:
        # S * sqrt(3) / 2 (the apothem) when n is perpendicular to an edge
        # Actually, the support function of a regular hexagon with side S in the direction of an edge normal is S * sqrt(3)/2
        # But we need to be careful about the orientation.
        
        # Let's compute: outer hex vertices
        outer_verts = hexagon_vertices(outer_cx, outer_cy, 1.0, theta)
        # Support of outer hex (unit side) in direction n:
        outer_support_unit = max((outer_verts @ n))
        # So outer support in direction n = outer_cx*n[0] + outer_cy*n[1] + S * outer_support_unit
        # Wait, that's not quite right because the hexagon is centered at (cx, cy)
        
        # Let me redo: outer hexagon with center (cx, cy), side S, angle theta
        # Its support in direction n is: cx*n[0] + cy*n[1] + S * h_unit(n, theta)
        # where h_unit is the support of a unit hexagon centered at origin with angle theta
        
        # For containment: for each inner hex i:
        #   support_inner_i(n) <= cx*n[0] + cy*n[1] + S * h_unit(n, theta)
        # So S >= (support_inner_i(n) - cx*n[0] - cy*n[1]) / h_unit(n, theta)
        
        # Compute h_unit(n, theta): support of unit hex at origin with angle theta in direction n
        unit_verts = hexagon_vertices(0.0, 0.0, 1.0, theta)
        h_unit = (unit_verts @ n).max()
        
        if abs(h_unit) < 1e-10:
            continue
        
        for i in range(N):
            inner_support = compute_support(inner_centers[i], inner_sides[i], inner_angles[i], n.reshape(1, 2))[0]
            required_S = (inner_support - (outer_cx * n[0] + outer_cy * n[1])) / h_unit
            if required_S > max_required:
                max_required = required_S
    
    return max(0.0, max_required)

def polygon_distance(p1, p2):
    """Compute minimum distance between two convex polygons using SAT.
    Returns negative value if overlapping."""
    min_dist = np.inf
    for poly in [p1, p2]:
        for i in range(6):
            v1 = poly[i]
            v2 = poly[(i + 1) % 6]
            edge = v2 - v1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-10:
                continue
            normal = normal / norm
            proj1 = p1 @ normal
            proj2 = p2 @ normal
            dist = max(np.min(proj2) - np.max(proj1), np.min(proj1) - np.max(proj2))
            if dist < min_dist:
                min_dist = dist
            if min_dist <= 0:
                return min_dist
    return min_dist

def sat_disjoint(poly1, poly2, tol=1e-7):
    """Check if two convex polygons are disjoint using SAT."""
    for poly in [poly1, poly2]:
        for i in range(6):
            v1 = poly[i]
            v2 = poly[(i + 1) % 6]
            edge = v2 - v1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-10:
                continue
            normal = normal / norm
            proj1 = poly1 @ normal
            proj2 = poly2 @ normal
            if np.max(proj1) < np.min(proj2) - tol or np.max(proj2) < np.min(proj1) - tol:
                return True
    return False

def validate(inner_hexes):
    """Check all pairs are disjoint."""
    for i in range(N):
        for j in range(i + 1, N):
            if not sat_disjoint(inner_hexes[i], inner_hexes[j]):
                return False
    return True

def generate_hex_lattice_config(rows, cols, spacing_x, spacing_y, stagger=False):
    """Generate a hexagonal lattice configuration."""
    positions = []
    angles = []
    for r in range(rows):
        for c in range(cols):
            x = c * spacing_x
            y = r * spacing_y
            if stagger and r % 2 == 1:
                x += spacing_x / 2
            positions.append([x, y])
            angles.append(0.0)
    positions = np.array(positions)
    angles = np.array(angles)
    positions -= positions.mean(axis=0)
    return positions, angles

def generate_ring_config(inner_r, outer_r):
    """Generate a ring configuration: 1 center + 6 inner + 5 outer."""
    positions = [[0.0, 0.0]]
    angles = [0.0]
    for i in range(6):
        angle = 2 * np.pi * i / 6
        positions.append([inner_r * np.cos(angle), inner_r * np.sin(angle)])
        angles.append(angle)
    for i in range(5):
        angle = 2 * np.pi * (i + 0.5) / 6
        positions.append([outer_r * np.cos(angle), outer_r * np.sin(angle)])
        angles.append(angle)
    positions = np.array(positions)
    angles = np.array(angles)
    positions -= positions.mean(axis=0)
    return positions, angles

def generate_circular_config(radius, angle_offset=0):
    """Generate a circular arrangement."""
    positions = []
    angles = []
    for i in range(12):
        angle = 2 * np.pi * i / 12 + angle_offset
        positions.append([radius * np.cos(angle), radius * np.sin(angle)])
        angles.append(angle)
    positions = np.array(positions)
    angles = np.array(angles)
    positions -= positions.mean(axis=0)
    return positions, angles

def penalty_objective(params, outer_cx=0.0, outer_cy=0.0, outer_angle_rad=0.0):
    """Penalty-based objective for optimization.
    params: [x0, y0, a0, x1, y1, a1, ..., x11, y11, a11]
    Returns scalar penalty + outer side length."""
    pos = params[:2*N].reshape(N, 2)
    ang = params[2*N:3*N]
    
    inner_hexes = []
    for i in range(N):
        verts = hexagon_vertices(pos[i, 0], pos[i, 1], 1.0, ang[i])
        inner_hexes.append(verts)
    
    # Separation penalty
    penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            dist = polygon_distance(inner_hexes[i], inner_hexes[j])
            if dist < 0:
                penalty += (-dist) ** 2 * 500.0
            elif dist < 0.05:
                penalty += (0.05 - dist) ** 2 * 50.0
    
    # Containment penalty using support functions
    # Outer hexagon edge normals
    theta = outer_angle_rad
    normals = np.column_stack([
        np.cos(theta + np.radians(np.arange(0, 360, 60))),
        np.sin(theta + np.radians(np.arange(0, 360, 60)))
    ])
    
    # Compute outer side length
    side = outer_side_from_support(pos, ang, np.ones(N), outer_cx, outer_cy, outer_angle_rad)
    
    # Add containment check: all inner vertices should be inside outer hex
    outer_verts = hexagon_vertices(outer_cx, outer_cy, side, outer_angle_rad)
    for i in range(N):
        for v in inner_hexes[i]:
            # Check if v is inside outer hex
            inside = True
            for k in range(6):
                v1 = outer_verts[k]
                v2 = outer_verts[(k + 1) % 6]
                edge = v2 - v1
                normal = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(normal)
                if norm < 1e-10:
                    continue
                normal = normal / norm
                # For CCW polygon, interior is to the left of each edge
                cross = (v2[0] - v1[0]) * (v[1] - v1[1]) - (v2[1] - v1[1]) * (v[0] - v1[0])
                if cross < -1e-8:
                    inside = False
                    break
            if not inside:
                # Distance to boundary
                penalty += 100.0
    
    return side + penalty

def refine_with_optimization(positions, angles, outer_cx=0.0, outer_cy=0.0, outer_angle_rad=0.0, max_iters=200):
    """Refine a configuration using penalty-based local optimization."""
    x0 = np.concatenate([positions.flatten(), angles])
    
    try:
        result = minimize(penalty_objective, x0, method='L-BFGS-B',
                         args=(outer_cx, outer_cy, outer_angle_rad),
                         options={'maxiter': max_iters, 'ftol': 1e-14, 'gtol': 1e-12})
        pos = result.x[:2*N].reshape(N, 2)
        ang = result.x[2*N:3*N]
        return pos, ang
    except Exception:
        return positions, angles

def evaluate_config(positions, angles, outer_cx=0.0, outer_cy=0.0, outer_angle_rad=0.0):
    """Evaluate a configuration: return (side_length, is_valid)."""
    inner_hexes = []
    for i in range(N):
        verts = hexagon_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i])
        inner_hexes.append(verts)
    
    # Check validity (disjoint)
    valid = True
    for i in range(N):
        for j in range(i + 1, N):
            if not sat_disjoint(inner_hexes[i], inner_hexes[j]):
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        return np.inf, False
    
    side = outer_side_from_support(positions, angles, np.ones(N), outer_cx, outer_cy, outer_angle_rad)
    return side, True

def construct():
    start_time = time.time()
    deadline = start_time + 135.0
    
    np.random.seed(42)
    configs = []
    
    # Hexagonal lattice configurations
    for sx in [SQRT3 * 0.95, SQRT3 * 1.0, SQRT3 * 1.05, SQRT3 * 1.1]:
        for sy in [1.4, 1.45, 1.5, 1.55, 1.6]:
            for stagger in [False, True]:
                pos, ang = generate_hex_lattice_config(3, 4, sx, sy, stagger)
                configs.append((pos, ang))
    
    # Ring configurations
    for ir in [1.6, 1.7, 1.75, 1.8, 1.9]:
        for or_ in [2.8, 3.0, 3.1, 3.2, 3.4]:
            pos, ang = generate_ring_config(ir, or_)
            configs.append((pos, ang))
    
    # Circular configurations
    for r in [2.2, 2.3, 2.4, 2.5, 2.6, 2.8]:
        for offset in [0, np.pi/12, np.pi/6, np.pi/4]:
            pos, ang = generate_circular_config(r, offset)
            configs.append((pos, ang))
    
    # Random configurations
    for _ in range(30):
        pos = np.random.uniform(-3, 3, (N, 2))
        ang = np.random.uniform(0, 2*np.pi, N)
        pos -= pos.mean(axis=0)
        configs.append((pos, ang))
    
    # Evaluate all configurations
    best_side = np.inf
    best_positions = None
    best_angles = None
    best_outer = (0.0, 0.0, 0.0)
    
    for pos, ang in configs:
        if time.time() > deadline - 40:
            break
        side, valid = evaluate_config(pos, ang)
        if valid and side < best_side:
            best_side = side
            best_positions = pos.copy()
            best_angles = ang.copy()
    
    # Refine the best configuration with multiple outer angles
    if best_positions is not None:
        if time.time() < deadline - 30:
            for outer_a_deg in np.linspace(-30, 30, 13):
                if time.time() > deadline - 30:
                    break
                outer_a = np.radians(outer_a_deg)
                refined_pos, refined_ang = refine_with_optimization(
                    best_positions, best_angles, 0.0, 0.0, outer_a, max_iters=100)
                side, valid = evaluate_config(refined_pos, refined_ang, 0.0, 0.0, outer_a)
                if valid and side < best_side:
                    best_side = side
                    best_positions = refined_pos
                    best_angles = refined_ang
                    best_outer = (0.0, 0.0, outer_a)
        
        # Try more outer angles for the best configuration
        if time.time() < deadline - 20:
            for outer_a_deg in np.linspace(-30, 30, 25):
                if time.time() > deadline - 20:
                    break
                outer_a = np.radians(outer_a_deg)
                side, valid = evaluate_config(best_positions, best_angles, 0.0, 0.0, outer_a)
                if valid and side < best_side:
                    best_side = side
                    best_outer = (0.0, 0.0, outer_a)
    
    # If no valid configuration found, use a widely-spaced fallback
    if best_positions is None:
        positions = []
        angles = []
        for i in range(12):
            angle = 2 * np.pi * i / 12
            r = 4.0
            positions.append([r * np.cos(angle), r * np.sin(angle)])
            angles.append(angle)
        positions = np.array(positions)
        angles = np.array(angles)
        positions -= positions.mean(axis=0)
        best_positions = positions
        best_angles = angles
        inner_hexes = [hexagon_vertices(best_positions[i,0], best_positions[i,1], 1.0, best_angles[i]) for i in range(N)]
        best_side = outer_side_from_support(best_positions, best_angles, np.ones(N), 0.0, 0.0, 0.0)
    
    # Final validation
    inner_hexes = []
    for i in range(N):
        verts = hexagon_vertices(best_positions[i, 0], best_positions[i, 1], 1.0, best_angles[i])
        inner_hexes.append(verts)
    
    valid = validate(inner_hexes)
    if not valid:
        # Fallback to even more spaced out configuration
        positions = []
        angles = []
        for i in range(12):
            angle = 2 * np.pi * i / 12
            r = 5.0
            positions.append([r * np.cos(angle), r * np.sin(angle)])
            angles.append(angle)
        positions = np.array(positions)
        angles = np.array(angles)
        positions -= positions.mean(axis=0)
        inner_hexes = []
        for i in range(N):
            verts = hexagon_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i])
            inner_hexes.append(verts)
        best_positions = positions
        best_angles = angles
        best_side = outer_side_from_support(best_positions, best_angles, np.ones(N), 0.0, 0.0, 0.0)
    
    outer_cx, outer_cy, outer_a = best_outer
    exact_side = outer_side_from_support(best_positions, best_angles, np.ones(N), outer_cx, outer_cy, outer_a)
    
    inner = [[best_positions[i, 0], best_positions[i, 1], np.degrees(best_angles[i])] for i in range(N)]
    return inner, (outer_cx, outer_cy), exact_side, np.degrees(outer_a)
