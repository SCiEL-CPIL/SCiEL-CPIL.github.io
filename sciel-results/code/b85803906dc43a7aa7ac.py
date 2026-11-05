"""Packing 12 unit regular hexagons inside a minimal regular hexagon."""
import numpy as np
from scipy.optimize import minimize

N = 12

# Precompute unit regular hexagon vertices (pointy-top orientation, side=1)
_base_angles = np.array([0, 60, 120, 180, 240, 360 - 60]) * np.pi / 180.0
_base_vertices = np.column_stack([np.cos(_base_angles), np.sin(_base_angles)])

# Edge normals for the base hexagon (perpendicular to each edge, pointing outward)
# For a hexagon with vertices at angles 0, 60, 120, 180, 240, 300,
# the edge between vertex i and vertex i+1 has a normal at angle 30 + 60*i
_edge_normal_angles = np.array([30, 90, 150, 210, 270, 330]) * np.pi / 180.0
_edge_normals = np.column_stack([np.cos(_edge_normal_angles), np.sin(_edge_normal_angles)])

def _hexagon_vertices(cx, cy, angle_deg):
    """Return 6 vertices of a unit regular hexagon centered at (cx,cy) rotated by angle_deg."""
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    return _base_vertices @ R.T + np.array([cx, cy])

def _smooth_overlap(v1, v2, sharpness=25.0):
    """
    Smooth approximation of the overlap depth between two hexagons using SAT.
    Returns 0 if separated, >0 if overlapping (smoothly).
    """
    def get_axes(vertices):
        axes = []
        for i in range(len(vertices)):
            p1 = vertices[i]
            p2 = vertices[(i + 1) % len(vertices)]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-10:
                normal /= norm
            axes.append(normal)
        return np.array(axes)

    axes1 = get_axes(v1)
    axes2 = get_axes(v2)
    all_axes = np.vstack([axes1, axes2])

    overlaps = np.zeros(len(all_axes))
    for i, axis in enumerate(all_axes):
        proj1 = v1 @ axis
        proj2 = v2 @ axis
        min1, max1 = proj1.min(), proj1.max()
        min2, max2 = proj2.min(), proj2.max()
        overlap = min(max1, max2) - max(min1, min2)
        overlaps[i] = overlap

    # Violation = smooth_max(0, -min(overlaps))
    # Use: 1/sharpness * log(1 + sum(exp(sharpness * neg_overlaps)))
    neg_overlaps = -overlaps
    scaled = sharpness * neg_overlaps
    scaled = np.clip(scaled, -100, 100)
    violation = (1.0 / sharpness) * np.log(1.0 + np.sum(np.exp(scaled)))
    return violation

def _point_in_hexagon(p, center, side, angle_deg):
    """
    Check if point p is inside the outer hexagon centered at 'center' with
    side length 'side' and rotation 'angle_deg'. Returns margin (positive if inside).
    """
    p_centered = p - np.array(center)
    dist = np.linalg.norm(p_centered)
    if dist < 1e-10:
        return side * 0.866  # Inradius
    
    direction = p_centered / dist
    theta = np.radians(-angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    u_rot = R @ direction
    dots = _base_vertices @ u_rot
    max_dot = np.max(dots)
    support = side * max_dot
    return support - dist

def _objective(params, sharpness=25.0):
    """
    Objective: minimize side + penalties for containment violations and overlaps.
    """
    inner_params = params[:3 * N].reshape(N, 3)
    ocx, ocy, side, oangle = params[3 * N:3 * N + 4]
    
    inner_verts = []
    for i in range(N):
        cx, cy, angle = inner_params[i]
        verts = _hexagon_vertices(cx, cy, angle)
        inner_verts.append(verts)
    
    # Containment penalty
    containment_penalty = 0.0
    for i in range(N):
        for j in range(6):
            vertex = inner_verts[i][j]
            margin = _point_in_hexagon(vertex, np.array([ocx, ocy]), side, oangle)
            if margin < 0:
                containment_penalty += margin * margin  # Quadratic penalty
    
    # Overlap penalty
    overlap_penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            ov = _smooth_overlap(inner_verts[i], inner_verts[j], sharpness)
            overlap_penalty += ov
    
    obj = side + 100.0 * containment_penalty + 50.0 * overlap_penalty
    return obj

def construct():
    """
    Find optimal packing of 12 unit hexagons in minimal outer hexagon.
    """
    np.random.seed(42)
    
    # Initial guess: hexagonal grid arrangement
    initial_inner = []
    spacing = 1.85
    positions = []
    for row in range(3):
        for col in range(4):
            x = col * spacing * np.sqrt(3) / 2
            y = row * spacing * 1.5
            if row % 2 == 1:
                x += spacing * np.sqrt(3) / 4
            positions.append([x, y, 0.0])
    
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    cx_mean = (max(xs) + min(xs)) / 2
    cy_mean = (max(ys) + min(ys)) / 2
    for p in positions:
        p[0] -= cx_mean
        p[1] -= cy_mean
    
    initial_inner = positions
    
    max_r = max(np.sqrt(p[0]**2 + p[1]**2) for p in initial_inner)
    initial_side = max_r + 1.5
    
    params0 = np.zeros(3 * N + 4)
    for i in range(N):
        params0[3*i] = initial_inner[i][0]
        params0[3*i+1] = initial_inner[i][1]
        params0[3*i+2] = initial_inner[i][2]
    params0[3*N] = 0.0
    params0[3*N+1] = 0.0
    params0[3*N+2] = initial_side
    params0[3*N+3] = 0.0
    
    bounds = []
    for i in range(N):
        bounds.append((-8.0, 8.0))
        bounds.append((-8.0, 8.0))
        bounds.append((0.0, 360.0))
    bounds.append((-5.0, 5.0))
    bounds.append((-5.0, 5.0))
    bounds.append((2.0, 15.0))
    bounds.append((0.0, 60.0))
    
    # Multi-start optimization
    best_obj = float('inf')
    best_params = params0.copy()
    
    rng = np.random.RandomState(123)
    starts = [params0.copy()]
    for _ in range(4):
        p = params0.copy()
        for i in range(N):
            p[3*i] += rng.uniform(-0.8, 0.8)
            p[3*i+1] += rng.uniform(-0.8, 0.8)
            p[3*i+2] = rng.uniform(0, 60)
        p[3*N+2] = params0[3*N+2] + rng.uniform(-0.5, 0.5)
        p = np.clip(p, [b[0] for b in bounds], [b[1] for b in bounds])
        starts.append(p)
    
    for start in starts:
        result = minimize(
            _objective,
            start,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 800, 'ftol': 1e-12, 'gtol': 1e-9, 'disp': False}
        )
        if result.fun < best_obj:
            best_obj = result.fun
            best_params = result.x.copy()
    
    # Extract solution
    inner = []
    for i in range(N):
        x = best_params[3*i]
        y = best_params[3*i+1]
        angle = best_params[3*i+2] % 360.0
        inner.append([x, y, angle])
    
    center = (best_params[3*N], best_params[3*N+1])
    side = best_params[3*N+2]
    angle = best_params[3*N+3]
    
    # Validate
    inner_verts = []
    for i in range(N):
        verts = _hexagon_vertices(inner[i][0], inner[i][1], inner[i][2])
        inner_verts.append(verts)
    
    valid = True
    for i in range(N):
        for j in range(6):
            vertex = inner_verts[i][j]
            margin = _point_in_hexagon(vertex, np.array(center), side, angle)
            if margin < -1e-5:
                valid = False
                break
        if not valid:
            break
    
    if valid:
        for i in range(N):
            for j in range(i+1, N):
                # Check for overlap using SAT
                def get_axes(vertices):
                    axes = []
                    for k in range(len(vertices)):
                        p1 = vertices[k]
                        p2 = vertices[(k + 1) % len(vertices)]
                        edge = p2 - p1
                        normal = np.array([-edge[1], edge[0]])
                        norm = np.linalg.norm(normal)
                        if norm > 1e-10:
                            normal /= norm
                        axes.append(normal)
                    return np.array(axes)
                
                axes1 = get_axes(inner_verts[i])
                axes2 = get_axes(inner_verts[j])
                all_axes = np.vstack([axes1, axes2])
                for axis in all_axes:
                    proj1 = inner_verts[i] @ axis
                    proj2 = inner_verts[j] @ axis
                    if proj1.max() > proj2.min() + 1e-6 and proj2.max() > proj1.min() + 1e-6:
                        pass  # Overlapping on this axis
                    else:
                        break
                else:
                    # All axes show overlap
                    valid = False
                    break
            if not valid:
                break
    
    if not valid:
        # Fallback: safe configuration
        inner = []
        for k in range(N):
            x = k * 2.1
            inner.append([x, 0.0, 0.0])
        cx = (N-1) * 2.1 / 2
        for i in range(N):
            inner[i][0] -= cx
        max_r = max(np.sqrt(p[0]**2 + p[1]**2) for p in inner)
        side = max_r + 1.5
        center = (0.0, 0.0)
        angle = 0.0
    
    return inner, center, side, angle

# EVOLVE-BLOCK-START
# (The construct function above is the main entry point)
# EVOLVE-BLOCK-END
