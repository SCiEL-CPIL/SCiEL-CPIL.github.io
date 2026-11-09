"""Pack 11 unit hexagons in smallest regular hexagon using gradient-based optimization."""
import numpy as np
from scipy.optimize import minimize
import time

N = 11
DEADLINE = time.time() + 130  # Safety margin before 150s kill

def _hex_vertices(cx, cy, angle_deg):
    """Vertices of unit regular hexagon (side=1) centered at (cx,cy) rotated by angle_deg."""
    a = np.radians(angle_deg)
    cos_a, sin_a = np.cos(a), np.sin(a)
    # Base vertices of unit hexagon (flat-top, side=1)
    base = np.array([
        [1.0, 0.0], [0.5, np.sqrt(3)/2], [-0.5, np.sqrt(3)/2],
        [-1.0, 0.0], [-0.5, -np.sqrt(3)/2], [0.5, -np.sqrt(3)/2]
    ])
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    return base @ R.T + np.array([cx, cy])

def _dist_point_to_segment(p, a, b):
    """Distance from point p to segment ab."""
    ab = b - a
    t = np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-12)
    t = np.clip(t, 0.0, 1.0)
    proj = a + t * ab
    return np.linalg.norm(p - proj)

def _point_in_convex_polygon(p, verts):
    """Check if point p is inside convex polygon verts (CCW)."""
    n = len(verts)
    for i in range(n):
        a = verts[i]
        b = verts[(i+1) % n]
        cross = (b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0])
        if cross < -1e-9:
            return False
    return True

def _hexagon_overlap_penalty(v1, v2, eps=0.5):
    """
    Smooth penalty for two hexagons overlapping.
    Uses: for each edge of hex1, check if any vertex of hex2 is inside hex1,
    and for each vertex of hex1, check if it's inside hex2.
    Returns a smooth positive penalty when overlapping.
    """
    penalty = 0.0
    # Check vertices of v2 inside v1
    for p in v2:
        if _point_in_convex_polygon(p, v1):
            # Distance to nearest edge of v1
            dmin = min(_dist_point_to_segment(p, v1[i], v1[(i+1)%6]) for i in range(6))
            penalty += max(0, eps - dmin) ** 2
    # Check vertices of v1 inside v2
    for p in v1:
        if _point_in_convex_polygon(p, v2):
            dmin = min(_dist_point_to_segment(p, v2[i], v2[(i+1)%6]) for i in range(6))
            penalty += max(0, eps - dmin) ** 2
    return penalty

def _outer_hex_vertices(cx, cy, side, angle_deg):
    """Vertices of outer regular hexagon."""
    a = np.radians(angle_deg)
    cos_a, sin_a = np.cos(a), np.sin(a)
    base = np.array([
        [side, 0.0], [side/2, side*np.sqrt(3)/2], [-side/2, side*np.sqrt(3)/2],
        [-side, 0.0], [-side/2, -side*np.sqrt(3)/2], [side/2, -side*np.sqrt(3)/2]
    ])
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    return base @ R.T + np.array([cx, cy])

def _outer_containment_penalty(inner_verts, outer_verts, eps=0.5):
    """Penalty if any inner hexagon vertex is outside outer hexagon."""
    penalty = 0.0
    for p in inner_verts:
        if not _point_in_convex_polygon(p, outer_verts):
            dmin = min(_dist_point_to_segment(p, outer_verts[i], outer_verts[(i+1)%6]) for i in range(6))
            penalty += max(0, eps - dmin) ** 2
    return penalty

def _total_penalty(params, side, outer_center, outer_angle):
    """
    params: flat array of [x1,y1,ang1, x2,y2,ang2, ..., xN,yN,angN]
    """
    n = N
    vertices_list = []
    for i in range(n):
        x, y, ang = params[3*i], params[3*i+1], params[3*i+2]
        vertices_list.append(_hex_vertices(x, y, ang))
    
    penalty = 0.0
    # Pairwise overlap
    for i in range(n):
        for j in range(i+1, n):
            penalty += _hexagon_overlap_penalty(vertices_list[i], vertices_list[j])
    
    # Containment
    outer_verts = _outer_hex_vertices(outer_center[0], outer_center[1], side, outer_angle)
    for i in range(n):
        penalty += _outer_containment_penalty(vertices_list[i], outer_verts)
    
    return penalty

def _binary_search_side(params, outer_center, outer_angle, lo=1.0, hi=15.0, iters=10):
    """Binary search for minimum side that satisfies constraints."""
    for _ in range(iters):
        mid = (lo + hi) / 2
        if _total_penalty(params, mid, outer_center, outer_angle) < 1e-6:
            hi = mid
        else:
            lo = mid
    return hi

def construct():
    """Construct packing of 11 unit hexagons in smallest regular hexagon."""
    best_side = 40.0
    best_params = None
    best_outer = (0.0, 0.0, 0.0)
    
    # Try different initial configurations
    initializations = []
    
    # Init 1: Hexagonal cluster (center + 6 around + 4 more)
    s = np.sqrt(3)  # distance between centers of adjacent hexagons
    init1 = np.array([0.0, 0.0, 0.0])
    for k in range(6):
        ang = k * np.pi / 3
        init1 = np.concatenate([init1, [s*np.cos(ang), s*np.sin(ang), 0.0]])
    # Add 4 more in next ring
    for k in range(0, 6, 2):
        ang = k * np.pi / 3 + np.pi/3
        init1 = np.concatenate([init1, [2*s*np.cos(ang), 2*s*np.sin(ang), 0.0]])
    initializations.append(init1[:3*N])
    
    # Init 2: Linear arrangement (like seed but tighter)
    init2 = np.array([0.0, 0.0, 0.0])
    for k in range(1, N):
        init2 = np.concatenate([init2, [k*2.0, 0.0, 0.0]])
    initializations.append(init2[:3*N])
    
    # Init 3: Two rows
    init3 = np.array([0.0, 0.0, 0.0])
    for k in range(1, 6):
        init3 = np.concatenate([init3, [k*2.0, 0.0, 0.0]])
    for k in range(5):
        init3 = np.concatenate([init3, [k*2.0 + 1.0, s, 0.0]])
    initializations.append(init3[:3*N])
    
    # Init 4: Random
    np.random.seed(42)
    init4 = np.random.uniform(-5, 5, 3*N)
    init4[2::3] = np.random.uniform(0, 60, N)  # angles
    initializations.append(init4)
    
    # Init 5: 3-4-4 arrangement
    init5 = np.array([0.0, 0.0, 0.0])
    for k in range(1, 3):
        init5 = np.concatenate([init5, [k*2.0, 0.0, 0.0]])
    for k in range(4):
        init5 = np.concatenate([init5, [k*2.0 - 1.0, s, 0.0]])
    for k in range(4):
        init5 = np.concatenate([init5, [k*2.0 - 1.0, -s, 0.0]])
    initializations.append(init5[:3*N])
    
    outer_center = (0.0, 0.0)
    outer_angle = 0.0
    
    for init_idx, init in enumerate(initializations):
        if time.time() > DEADLINE:
            break
        
        # Optimize with penalty method
        def objective(params):
            if time.time() > DEADLINE:
                return 1e10
            return _total_penalty(params, best_side, outer_center, outer_angle)
        
        try:
            # Multiple optimization passes with decreasing side
            current_side = 12.0
            for pass_num in range(5):
                if time.time() > DEADLINE:
                    break
                
                # Optimize positions for current side
                result = minimize(
                    objective,
                    init,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-8}
                )
                init = result.x
                
                # Check if feasible, try smaller side
                pen = _total_penalty(init, current_side, outer_center, outer_angle)
                if pen < 1e-6:
                    # Try to reduce side
                    current_side *= 0.92
                    if current_side < 1.5:
                        current_side = 1.5
                else:
                    # Increase side slightly
                    current_side *= 1.05
                    if current_side > 15:
                        current_side = 15
                
                if time.time() > DEADLINE:
                    break
            
            # Binary search for optimal side
            if time.time() > DEADLINE:
                break
            
            final_side = _binary_search_side(init, outer_center, outer_angle, lo=1.5, hi=15.0, iters=8)
            
            if final_side < best_side:
                best_side = final_side
                best_params = init.copy()
                best_outer = (outer_center[0], outer_center[1], outer_angle)
                
        except Exception:
            continue
    
    # Build final result
    if best_params is None:
        # Fallback to seed
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    inner = []
    for i in range(N):
        x = float(best_params[3*i])
        y = float(best_params[3*i+1])
        ang = float(best_params[3*i+2]) % 360.0
        inner.append([x, y, ang])
    
    return inner, (best_outer[0], best_outer[1]), float(best_side), float(best_outer[2])
