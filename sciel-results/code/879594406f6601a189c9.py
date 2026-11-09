# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _hex_vertices(cx, cy, s, a_deg):
    """Compute 6 vertices of a regular hexagon."""
    a = np.radians(a_deg)
    angles = a + np.radians(60 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def _check_separation(v1, v2, eps=1e-6):
    """Check if two convex hexagons are separated (non-intersecting).
    Returns True if separated, False if intersecting."""
    # Get all 12 edge normals (6 from each hexagon)
    for verts in [v1, v2]:
        for i in range(6):
            p1 = verts[i]
            p2 = verts[(i + 1) % 6]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-12:
                continue
            normal = normal / norm
            # Project both hexagons onto this normal
            proj1 = np.dot(v1, normal)
            proj2 = np.dot(v2, normal)
            min1, max1 = proj1.min(), proj1.max()
            min2, max2 = proj2.min(), proj2.max()
            # Check if intervals are apart by more than eps
            if max1 + eps < min2 or max2 + eps < min1:
                return True
    return False

def _check_inside(v_inner, v_outer, eps=1e-6):
    """Check if all vertices of inner hex are inside outer hex."""
    for p in v_inner:
        for i in range(6):
            p1 = v_outer[i]
            p2 = v_outer[(i + 1) % 6]
            cross = np.cross(p2 - p1, p - p1)
            if cross < -eps:
                return False
    return True

def _validate(inner, outer, side):
    """Full validation check."""
    if not np.all(np.isfinite(inner)) or not np.all(np.isfinite(outer)):
        return False
    if side < np.sqrt(11) - 1e-6:
        return False
    
    ox, oy, oa = outer
    # Compute outer vertices
    v_outer = _hex_vertices(ox, oy, side, oa)
    
    # Shift coordinates by outer center
    inner_shifted = inner.copy()
    inner_shifted[:, 0] -= ox
    inner_shifted[:, 1] -= oy
    
    # Check edge lengths of shifted inner hexagons
    for i in range(11):
        x, y, a = inner_shifted[i]
        verts = _hex_vertices(x, y, 1.0, a)
        for j in range(6):
            d = np.linalg.norm(verts[(j+1)%6] - verts[j])
            if abs(d - 1.0) > 1e-8:
                return False
    
    # Compute all inner hex vertices
    all_verts = []
    for i in range(11):
        x, y, a = inner_shifted[i]
        verts = _hex_vertices(x, y, 1.0, a)
        all_verts.append(verts)
    
    # Check separation between all pairs
    for i in range(11):
        for j in range(i+1, 11):
            if not _check_separation(all_verts[i], all_verts[j]):
                return False
    
    # Check all inner vertices are inside outer
    for i in range(11):
        if not _check_inside(all_verts[i], v_outer):
            return False
    
    return True

def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s limit
    
    # Initial honeycomb arrangement for 11 hexagons
    # 1 center, 6 surrounding, 4 more in a ring
    sqrt3 = np.sqrt(3)
    # Standard hex grid spacing: horizontal 2.5, vertical 2.5*sqrt(3)/2
    dx = 2.5
    dy = 2.5 * sqrt3 / 2
    
    # 11 hexagons in a compact cluster
    # Row 0 (top): 2 hexagons
    # Row 1: 3 hexagons  
    # Row 2 (middle): 3 hexagons
    # Row 3: 3 hexagons
    # Actually, let's use the standard 11-hexagon flower pattern
    
    # Better: use the known optimal 11-hexagon packing
    # Center at origin, with hexagons at:
    positions = [
        (0, 0),           # center
        (2.5, 0),         # right
        (-2.5, 0),        # left
        (1.25, dy),       # upper-right
        (-1.25, dy),      # upper-left
        (1.25, -dy),      # lower-right
        (-1.25, -dy),     # lower-left
        (3.75, dy),       # far upper-right
        (-3.75, dy),      # far upper-left
        (3.75, -dy),      # far lower-right
        (-3.75, -dy),     # far lower-left
    ]
    
    inner_init = np.array([[x, y, 0.0] for x, y in positions])
    
    # Try different outer hexagon orientations and sizes
    best_score = -1
    best_result = None
    
    # First, find a valid configuration with a reasonable outer hex
    # The bounding box of the 11 hexagons spans roughly:
    # x: -3.75-1 to 3.75+1 = -4.75 to 4.75, so width ~9.5
    # y: -dy-1 to dy+1 = -3.165 to 3.165, so height ~6.33
    # For a regular hexagon with side s, the width is 2s and height is s*sqrt(3)
    # So we need s >= max(9.5/2, 6.33/sqrt(3)) = max(4.75, 3.65) = 4.75
    # But with rotation, we can do better
    
    # Try various outer hexagon side lengths and angles
    for trial in range(50):
        if time.time() > deadline:
            break
        
        # Vary the outer hexagon angle and side length
        outer_angle = trial * 15.0  # degrees
        # Estimate needed side length
        side_est = 5.0 + trial * 0.1
        
        outer = np.array([0.0, 0.0, outer_angle])
        
        if _validate(inner_init, outer, side_est):
            score = 1.0 / side_est
            if score > best_score:
                best_score = score
                best_result = (inner_init.copy(), outer.copy(), side_est)
    
    # Now optimize: minimize outer side length
    if best_result is not None:
        inner0, outer0, side0 = best_result
    else:
        # Fallback: use a large enough outer hex
        inner0 = inner_init
        outer0 = np.array([0.0, 0.0, 0.0])
        side0 = 8.0
        best_score = 1.0 / side0
        best_result = (inner0.copy(), outer0.copy(), side0)
    
    # Optimize the outer hexagon parameters (x, y, angle) and side length
    def objective(params):
        if time.time() > deadline:
            return 1e10
        ox, oy, oa, side = params
        if side < np.sqrt(11):
            return 1e10
        outer = np.array([ox, oy, oa])
        if not _validate(inner0, outer, side):
            return 1e10
        return side  # minimize side length
    
    # Try to optimize from the current best
    x0 = np.array([outer0[0], outer0[1], outer0[2], side0])
    
    try:
        result = minimize(objective, x0, method='Nelder-Mead',
                         options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-6})
        if time.time() < deadline and result.fun < side0:
            ox, oy, oa, side = result.x
            if side > np.sqrt(11) and _validate(inner0, np.array([ox, oy, oa]), side):
                score = 1.0 / side
                if score > best_score:
                    best_score = score
                    best_result = (inner0.copy(), np.array([ox, oy, oa]), side)
    except Exception:
        pass
    
    # Also try optimizing inner positions slightly
    if time.time() < deadline - 30:
        def objective_full(params):
            if time.time() > deadline:
                return 1e10
            # params: 33 inner coords (11*3) + 4 outer (ox,oy,oa,side)
            inner_params = params[:33].reshape(11, 3)
            ox, oy, oa, side = params[33:37]
            if side < np.sqrt(11):
                return 1e10
            outer = np.array([ox, oy, oa])
            if not _validate(inner_params, outer, side):
                return 1e10
            return side
        
        x0_full = np.concatenate([inner0.ravel(), [outer0[0], outer0[1], outer0[2], side0]])
        
        try:
            result = minimize(objective_full, x0_full, method='Nelder-Mead',
                             options={'maxiter': 300, 'xatol': 1e-3, 'fatol': 1e-5})
            if time.time() < deadline and result.fun < side0:
                inner_params = result.x[:33].reshape(11, 3)
                ox, oy, oa, side = result.x[33:37]
                if side > np.sqrt(11) and _validate(inner_params, np.array([ox, oy, oa]), side):
                    score = 1.0 / side
                    if score > best_score:
                        best_score = score
                        best_result = (inner_params.copy(), np.array([ox, oy, oa]), side)
        except Exception:
            pass
    
    if best_result is None:
        # Absolute fallback
        best_result = (inner_init, np.array([0.0, 0.0, 0.0]), 8.0)
    
    return best_result

# EVOLVE-BLOCK-END
