"""Direct construction of 12 unit hexagons inside a regular hexagon, minimizing outer side."""
N = 12

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import math

def _hex_verts(cx, cy, side, angle_deg):
    """Vertices of a regular hexagon centered at (cx, cy) with given side and rotation."""
    a = np.radians(angle_deg)
    return np.array([
        [cx + side * np.cos(a + np.radians(60 * i)),
         cy + side * np.sin(a + np.radians(60 * i))]
        for i in range(6)
    ])

def _sat_overlap_depth(h1, h2):
    """Minimum penetration depth between two convex polygons using SAT.
    Returns 0 if disjoint."""
    min_depth = float('inf')
    axes = []
    for poly in [h1, h2]:
        n = len(poly)
        for i in range(n):
            p1 = poly[i]
            p2 = poly[(i + 1) % n]
            edge = p2 - p1
            norm = np.linalg.norm(edge)
            if norm < 1e-10:
                continue
            normal = np.array([-edge[1], edge[0]]) / norm
            axes.append(normal)
    
    for axis in axes:
        proj1 = np.dot(h1, axis)
        proj2 = np.dot(h2, axis)
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        gap = max(min1, min2) - min(max1, max2)  # positive if separated
        if gap >= 0:
            return 0.0  # Disjoint on this axis
        
        overlap = -gap
        if overlap < min_depth:
            min_depth = overlap
    
    if min_depth == float('inf'):
        return 0.0
    
    return min_depth

def _containment_violation(inner_verts, outer_verts):
    """Check if inner hexagon is fully inside outer hexagon.
    Returns max violation depth (0 if contained)."""
    max_violation = 0.0
    for pt in inner_verts:
        for i in range(len(outer_verts)):
            v1 = outer_verts[i]
            v2 = outer_verts[(i + 1) % len(outer_verts)]
            edge = v2 - v1
            edge_len = np.linalg.norm(edge)
            if edge_len < 1e-12:
                continue
            rel = pt - v1
            cross = edge[0] * rel[1] - edge[1] * rel[0]
            if cross < 0:
                violation = -cross / edge_len
                if violation > max_violation:
                    max_violation = violation
    return max_violation

def _total_violation(params, outer_side, outer_center, outer_angle):
    """Total violation: containment + pairwise overlap. Returns max of all violations."""
    inner_verts_list = []
    for i in range(N):
        cx, cy, ang = params[3*i], params[3*i+1], params[3*i+2]
        inner_verts_list.append(_hex_verts(cx, cy, 1.0, ang))
    
    outer_verts = _hex_verts(outer_center[0], outer_center[1], outer_side, outer_angle)
    
    # Containment
    max_viol = 0.0
    for iv in inner_verts_list:
        viol = _containment_violation(iv, outer_verts)
        if viol > max_viol:
            max_viol = viol
    
    # Disjointness
    for i in range(N):
        for j in range(i + 1, N):
            depth = _sat_overlap_depth(inner_verts_list[i], inner_verts_list[j])
            if depth > max_viol:
                max_viol = depth
    
    return max_viol

def _is_valid(params, outer_side, outer_center, outer_angle, tol=1e-6):
    """Check if configuration is valid (no violations)."""
    return _total_violation(params, outer_side, outer_center, outer_angle) < tol

def _make_hex_grid(rows, cols):
    """Create hexagonal grid positions, centered at origin."""
    positions = []
    for r in range(rows):
        for c in range(cols):
            x = c * 2.0 + (1.0 if r % 2 == 1 else 0.0)
            y = r * math.sqrt(3)
            positions.append([x, y])
    arr = np.array(positions)
    centroid = arr.mean(axis=0)
    return (arr - centroid).tolist()

def _make_ring():
    """1 center + 6 ring + 5 outer = 12."""
    positions = [[0.0, 0.0]]
    # First ring at distance 2
    for i in range(6):
        ang = math.radians(60 * i)
        positions.append([2.0 * math.cos(ang), 2.0 * math.sin(ang)])
    # Second ring: 5 of 6 positions at distance 2*sqrt(3)
    for i in range(5):
        ang = math.radians(60 * i + 30)
        d = 2.0 * math.sqrt(3)
        positions.append([d * math.cos(ang), d * math.sin(ang)])
    return positions

def _make_linear():
    """12 in a row."""
    return [[float(i - 5.5), 0.0] for i in range(12)]

def _optimize_for_side(init_params, outer_side, outer_center, outer_angle, max_iter=300):
    """Try to find valid configuration for given outer side."""
    def objective(p):
        return _total_violation(p, outer_side, outer_center, outer_angle)
    
    try:
        result = minimize(
            objective,
            init_params,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'maxfun': max_iter * 20, 'ftol': 1e-14, 'gtol': 1e-12}
        )
        return result.x, result.fun
    except Exception:
        return init_params, float('inf')

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s limit
    
    # Generate initial configurations
    configs = []
    
    # Hex grids of various shapes
    for rows, cols in [(3, 4), (4, 3), (2, 6), (6, 2), (3, 5)]:
        pos = _make_hex_grid(rows, cols)
        if len(pos) == N:
            configs.append(pos)
    
    # Ring configuration
    ring_pos = _make_ring()
    if len(ring_pos) == N:
        configs.append(ring_pos)
    
    # Linear
    configs.append(_make_linear())
    
    # Try multiple outer angles
    outer_angles = [0.0, 30.0]
    outer_center = (0.0, 0.0)
    
    best_side = float('inf')
    best_params = None
    
    for outer_angle in outer_angles:
        if time.time() > deadline - 30.0:
            break
        
        # Try sides from large to small
        for outer_side in [5.0, 4.8, 4.6, 4.5, 4.4, 4.3, 4.2, 4.1, 4.0, 3.9, 3.8, 3.7, 3.6, 3.5, 3.4, 3.3, 3.2, 3.1]:
            if time.time() > deadline - 25.0:
                break
            
            found = False
            for config in configs:
                if time.time() > deadline - 20.0:
                    break
                
                init_params = []
                for (x, y) in config:
                    init_params.extend([float(x), float(y), 0.0])
                init_params = np.array(init_params)
                
                # Quick check
                viol = _total_violation(init_params, outer_side, outer_center, outer_angle)
                if viol < 1e-6:
                    found = True
                    best_params = init_params.copy()
                    break
                
                # Try to optimize
                if time.time() > deadline - 15.0:
                    continue
                
                opt_params, opt_viol = _optimize_for_side(init_params, outer_side, outer_center, outer_angle, max_iter=200)
                
                if opt_viol < 1e-6:
                    found = True
                    best_params = opt_params
                    break
            
            if found:
                if outer_side < best_side:
                    best_side = outer_side
                    best_params = best_params.copy()
            else:
                # If this side didn't work, smaller ones won't either (for this config family)
                break
    
    # Binary search for minimal valid side
    if best_params is not None and time.time() < deadline - 10.0:
        lo = best_side - 0.5
        hi = best_side
        for _ in range(15):
            if time.time() > deadline - 5.0:
                break
            mid = (lo + hi) / 2.0
            opt_params, opt_viol = _optimize_for_side(best_params, mid, outer_center, outer_angle, max_iter=300)
            if opt_viol < 1e-6:
                hi = mid
                best_params = opt_params
            else:
                lo = mid
        
        best_side = hi
        best_params = best_params.copy()
    
    # Fallback
    if best_params is None:
        pos = _make_hex_grid(3, 4)
        inner = []
        for (x, y) in pos:
            inner.append([float(x), float(y), 0.0])
        return inner, (0.0, 0.0), 5.5, 0.0
    
    # Build final result
    inner = []
    for i in range(N):
        x = float(best_params[3*i])
        y = float(best_params[3*i+1])
        ang = float(best_params[3*i+2])
        inner.append([x, y, ang])
    
    # Final validation
    if not _is_valid(best_params, best_side, outer_center, outer_angle, tol=1e-5):
        # Invalid, use safe fallback
        pos = _make_hex_grid(3, 4)
        inner = []
        for (x, y) in pos:
            inner.append([float(x), float(y), 0.0])
        return inner, (0.0, 0.0), 5.5, 0.0
    
    return inner, (float(outer_center[0]), float(outer_center[1])), float(best_side), float(outer_angle)

# EVOLVE-BLOCK-END
