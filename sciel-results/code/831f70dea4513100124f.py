"""Optimized hexagon packing: 11 unit hexagons in a regular hexagon.
Strategy: Smooth differentiable penalties with L-BFGS-B optimization.
Uses smooth max-based overlap and boundary violation metrics that provide
meaningful gradients through near-contact states, unlike discrete SAT.
"""
import numpy as np
import time
import random
import math
from scipy.optimize import minimize

N = 11
DEADLINE = 150.0
START_TIME = time.time()

def _remaining_time():
    return DEADLINE - (time.time() - START_TIME)

def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon with side length 'side'."""
    angle_rad = np.radians(angle_deg)
    verts = []
    for k in range(6):
        a = angle_rad + k * np.pi / 3
        verts.append([cx + side * np.cos(a), cy + side * np.sin(a)])
    return np.array(verts)

def _hex_face_normals(angle_deg):
    """Return unit outward normals of hexagon faces at given angle."""
    angle_rad = np.radians(angle_deg)
    normals = []
    for k in range(6):
        a = angle_rad + k * np.pi / 3
        normals.append([np.cos(a), np.sin(a)])
    return np.array(normals)

def _smooth_max(values, beta=50.0):
    """Smooth approximation of max using log-sum-exp."""
    values = np.asarray(values)
    if len(values) == 0:
        return 0.0
    v_max = np.max(values)
    if v_max > 50.0:
        return v_max
    exp_vals = np.exp(beta * (values - v_max))
    return v_max + np.log(np.sum(exp_vals)) / beta

def _hex_pair_overlap_distance(p1, p2):
    """
    Compute a smooth measure of overlap between two unit hexagons.
    Returns 0 if disjoint, positive if overlapping.
    Uses support function differences for each face direction.
    """
    x1, y1, a1 = p1
    x2, y2, a2 = p2
    
    # Get face normals for both hexagons
    n1 = _hex_face_normals(a1)
    n2 = _hex_face_normals(a2)
    
    # All candidate separating directions
    all_normals = np.vstack([n1, n2])
    
    # For each direction, compute the gap between the two hexagons
    # Gap = (max proj of hex1) - (min proj of hex2)
    # If gap < 0 for all directions, they overlap
    # We want a smooth measure of how much they overlap
    
    overlaps = []
    for nx, ny in all_normals:
        # Project hex1 vertices onto this normal
        v1 = _hex_vertices(x1, y1, 1.0, a1)
        proj1 = v1 @ np.array([nx, ny])
        max1 = np.max(proj1)
        
        # Project hex2 vertices onto this normal
        v2 = _hex_vertices(x2, y2, 1.0, a2)
        proj2 = v2 @ np.array([nx, ny])
        min2 = np.min(proj2)
        
        # Gap in this direction (positive means separated)
        gap = max1 - min2
        # Negative gap means overlap in this direction
        overlaps.append(-gap)
    
    overlaps = np.array(overlaps)
    # If all gaps are positive (disjoint), smooth max of negatives is negative
    # We want 0 when disjoint, positive when overlapping
    return max(0.0, _smooth_max(overlaps, beta=30.0))

def _hex_boundary_violation(p, outer_center, outer_side, outer_angle):
    """
    Compute smooth boundary violation: how far hexagon extends outside outer.
    Returns 0 if contained, positive if violating.
    """
    x, y, a = p
    cx, cy = outer_center
    
    v = _hex_vertices(x, y, 1.0, a)
    violations = []
    
    # Outer hexagon face normals
    outer_normals = _hex_face_normals(outer_angle)
    inradius_outer = outer_side * np.sqrt(3) / 2.0
    
    for nx, ny in outer_normals:
        # Project inner hex vertices onto outer face normal
        dx = v[:, 0] - cx
        dy = v[:, 1] - cy
        proj = dx * nx + dy * ny
        max_proj = np.max(proj)
        # Violation if max_proj > inradius
        violations.append(max_proj - inradius_outer)
    
    violations = np.array(violations)
    return max(0.0, _smooth_max(violations, beta=50.0))

def _objective(x, outer_angle):
    """
    Objective: outer_side + smooth penalties for overlap and boundary.
    x: flattened [x1,y1,a1, x2,y2,a2, ...]
    """
    positions = []
    for i in range(N):
        positions.append([x[i*3], x[i*3+1], x[i*3+2]])
    
    # Compute required outer side length (smooth max of vertex distances)
    cx, cy = 0.0, 0.0
    outer_angle_rad = np.radians(outer_angle)
    ca = np.cos(-outer_angle_rad)
    sa = np.sin(-outer_angle_rad)
    
    # Outer face normals in local coords
    outer_normals_local = []
    for k in range(6):
        a = np.radians(30 + k * 60)
        outer_normals_local.append([np.cos(a), np.sin(a)])
    outer_normals_local = np.array(outer_normals_local)
    
    required_sides = []
    for p in positions:
        v = _hex_vertices(p[0], p[1], 1.0, p[2])
        dx = v[:, 0] - cx
        dy = v[:, 1] - cy
        # Rotate to outer frame
        lx = dx * ca - dy * sa
        ly = dx * sa + dy * ca
        
        for nx, ny in outer_normals_local:
            proj = lx * nx + ly * ny
            # Required side = 2 * max_proj / sqrt(3)
            required_sides.append(2.0 * proj / np.sqrt(3))
    
    required_sides = np.array(required_sides)
    side = max(0.0, _smooth_max(required_sides, beta=20.0))
    
    # Overlap penalties
    overlap_penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            overlap_penalty += _hex_pair_overlap_distance(positions[i], positions[j])
    overlap_penalty *= 10.0  # Weight
    
    # Boundary violations (should be ~0 since we're minimizing side that contains them)
    # But add small penalty to encourage tight containment
    boundary_penalty = 0.0
    for p in positions:
        boundary_penalty += _hex_boundary_violation(p, (0.0, 0.0), side, outer_angle)
    boundary_penalty *= 5.0
    
    return side + overlap_penalty + boundary_penalty

def _objective_grad(x, outer_angle):
    """
    Numerical gradient for L-BFGS-B (analytical is complex with smooth max).
    Uses central differences.
    """
    grad = np.zeros_like(x)
    h = 1e-6
    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += h
        x_minus[i] -= h
        grad[i] = (_objective(x_plus, outer_angle) - _objective(x_minus, outer_angle)) / (2 * h)
    return grad

def _check_validity(inner_positions, outer_center, outer_side, outer_angle, tol=1e-6):
    """Check if configuration is valid: all inner hexagons inside outer, no overlaps."""
    cx, cy = outer_center
    
    for (ix, iy, iangle) in inner_positions:
        iverts = _hex_vertices(ix, iy, 1.0, iangle)
        for v in iverts:
            dx = v[0] - cx
            dy = v[1] - cy
            angle_rad = np.radians(outer_angle)
            ca = np.cos(-angle_rad)
            sa = np.sin(-angle_rad)
            lx = dx * ca - dy * sa
            ly = dx * sa + dy * ca
            
            inradius = outer_side * np.sqrt(3) / 2
            for k in range(6):
                a = np.radians(30 + k * 60)
                nx = np.cos(a)
                ny = np.sin(a)
                if lx * nx + ly * ny > inradius + tol:
                    return False
    
    all_verts = [_hex_vertices(ix, iy, 1.0, iangle) for (ix, iy, iangle) in inner_positions]
    for i in range(N):
        for j in range(i+1, N):
            if not _hexagons_disjoint(all_verts[i], all_verts[j], tol=tol):
                return False
    
    return True

def _hexagons_disjoint(verts1, verts2, tol=1e-7):
    """Check if two convex hexagons are disjoint using SAT."""
    def get_normals(verts):
        normals = []
        n = len(verts)
        for i in range(n):
            j = (i + 1) % n
            edge = verts[j] - verts[i]
            nx = -edge[1]
            ny = edge[0]
            norm = np.sqrt(nx*nx + ny*ny)
            if norm > 1e-12:
                normals.append([nx/norm, ny/norm])
        return normals
    
    normals1 = get_normals(verts1)
    normals2 = get_normals(verts2)
    all_normals = normals1 + normals2
    
    for nx, ny in all_normals:
        proj1 = np.array([v[0]*nx + v[1]*ny for v in verts1])
        proj2 = np.array([v[0]*nx + v[1]*ny for v in verts2])
        
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        if max1 < min2 - tol or max2 < min1 - tol:
            return True
    
    return False

def _initial_lattice():
    """Create initial hexagonal lattice arrangement."""
    positions = []
    positions.append([0.0, 0.0, 0.0])
    dist = 2.0 * np.sqrt(3)
    for k in range(6):
        a = np.radians(60 * k)
        positions.append([dist * np.cos(a), dist * np.sin(a), 0.0])
    dist2 = dist * np.sqrt(3)
    angles2 = [30, 90, 150, 210, 270, 330]
    for a_deg in angles2:
        a = np.radians(a_deg)
        positions.append([dist2 * np.cos(a), dist2 * np.sin(a), 0.0])
    return positions

def _initial_random(seed=42):
    """Create a random initial configuration."""
    rng = random.Random(seed)
    positions = []
    positions.append([0.0, 0.0, 0.0])
    for i in range(1, N):
        r = rng.uniform(1.5, 5.0)
        a = rng.uniform(0, 2*np.pi)
        angle = rng.uniform(0, 360)
        positions.append([r * np.cos(a), r * np.sin(a), angle])
    return positions

def _initial_clustered():
    """Initial config with tighter clustering."""
    positions = []
    positions.append([0.0, 0.0, 0.0])
    # First ring closer
    dist = 3.0
    for k in range(6):
        a = np.radians(60 * k)
        positions.append([dist * np.cos(a), dist * np.sin(a), 0.0])
    # Second ring
    dist2 = 5.0
    for k in range(4):
        a = np.radians(30 + 90 * k)
        positions.append([dist2 * np.cos(a), dist2 * np.sin(a), 0.0])
    return positions

def construct():
    """Main entry point."""
    random.seed(42)
    np.random.seed(42)
    
    best_result = None
    best_score = float('inf')
    
    outer_angles = [0.0, 30.0]
    
    for oa in outer_angles:
        if _remaining_time() < 20.0:
            break
        
        initial_configs = [
            ("lattice", _initial_lattice()),
            ("random1", _initial_random(42)),
            ("random2", _initial_random(123)),
            ("clustered", _initial_clustered()),
            ("random3", _initial_random(999)),
        ]
        
        for config_name, init_pos in initial_configs:
            if _remaining_time() < 8.0:
                break
            
            x0 = np.array([p[0] for p in init_pos] + [p[1] for p in init_pos] + [p[2] for p in init_pos])
            # Actually flatten properly
            x0 = []
            for p in init_pos:
                x0.extend(p)
            x0 = np.array(x0, dtype=np.float64)
            
            try:
                # Use L-BFGS-B with numerical gradient
                res = minimize(
                    _objective,
                    x0,
                    args=(oa,),
                    method='L-BFGS-B',
                    jac=_objective_grad,
                    options={
                        'maxiter': 300,
                        'ftol': 1e-8,
                        'gtol': 1e-6,
                        'maxls': 50
                    }
                )
                
                x_opt = res.x
                positions = []
                for i in range(N):
                    positions.append([x_opt[i*3], x_opt[i*3+1], x_opt[i*3+2]])
                
                # Compute actual side length
                side = _min_enclosing_side_exact(positions, (0.0, 0.0), oa)
                
                if _check_validity(positions, (0.0, 0.0), side, oa, tol=1e-5):
                    if side < best_score:
                        best_score = side
                        best_result = (positions, (0.0, 0.0), side, oa)
                        
            except Exception as e:
                continue

    if best_result is not None:
        return best_result
    
    # Fallback
    positions = []
    positions.append([0.0, 0.0, 0.0])
    dist = 3.5
    for k in range(6):
        a = np.radians(60 * k)
        positions.append([dist * np.cos(a), dist * np.sin(a), 0.0])
    dist2 = 6.0
    for k in range(4):
        a = np.radians(30 + 60 * k)
        positions.append([dist2 * np.cos(a), dist2 * np.sin(a), 0.0])
    
    side = _min_enclosing_side_exact(positions, (0.0, 0.0), 0.0)
    return positions, (0.0, 0.0), side, 0.0

def _min_enclosing_side_exact(inner_positions, outer_center, outer_angle):
    """Exact minimum side length to contain all inner hexagons."""
    cx, cy = outer_center
    best_side = 0.0
    outer_angle_rad = np.radians(outer_angle)
    ca = np.cos(-outer_angle_rad)
    sa = np.sin(-outer_angle_rad)
    
    for (ix, iy, iangle) in inner_positions:
        iverts = _hex_vertices(ix, iy, 1.0, iangle)
        for v in iverts:
            dx = v[0] - cx
            dy = v[1] - cy
            lx = dx * ca - dy * sa
            ly = dx * sa + dy * ca
            
            r = np.sqrt(lx*lx + ly*ly)
            if r < 1e-12:
                continue
            
            theta = np.arctan2(ly, lx)
            best_cos = 0.0
            for k in range(6):
                a = np.radians(30 + k * 60)
                d = abs(((theta - a) % (2*np.pi) + np.pi) % (2*np.pi))
                d = min(d, 2*np.pi - d)
                c = np.cos(d)
                if c > best_cos:
                    best_cos = c
            
            if best_cos < 1e-12:
                best_side = max(best_side, 1000.0)
            else:
                required_side = 2.0 * r * best_cos / np.sqrt(3)
                best_side = max(best_side, required_side)
    
    return best_side
