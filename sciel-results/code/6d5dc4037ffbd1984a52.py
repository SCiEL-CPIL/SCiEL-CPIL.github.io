"""Place 12 unit regular hexagons inside a regular hexagon, minimizing outer side length."""
N = 12

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import math

def _hex_verts(cx, cy, side, angle_deg):
    """Vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    a = np.radians(angle_deg)
    verts = np.empty((6, 2))
    for i in range(6):
        theta = a + np.radians(60 * i)
        verts[i, 0] = cx + side * np.cos(theta)
        verts[i, 1] = cy + side * np.sin(theta)
    return verts

def _sat_overlap_depth(h1, h2):
    """
    Calculate the minimum penetration depth between two convex polygons h1 and h2 using SAT.
    Returns 0 if disjoint, > 0 if overlapping.
    """
    min_depth = float('inf')
    
    # Get all edge normals (axes) from both polygons
    axes = []
    for poly in [h1, h2]:
        for i in range(6):
            p1 = poly[i]
            p2 = poly[(i + 1) % 6]
            edge = p2 - p1
            # Normal to the edge
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-10:
                axes.append(normal / norm)
    
    # Check each axis
    for axis in axes:
        # Project vertices of h1 onto axis
        proj1 = np.dot(h1, axis)
        min1 = np.min(proj1)
        max1 = np.max(proj1)
        
        # Project vertices of h2 onto axis
        proj2 = np.dot(h2, axis)
        min2 = np.min(proj2)
        max2 = np.max(proj2)
        
        # Check for separation
        if max1 < min2 or max2 < min1:
            return 0.0  # Disjoint
        
        # Calculate overlap depth on this axis
        overlap = min(max1, max2) - max(min1, min2)
        if overlap < min_depth:
            min_depth = overlap
            
    return min_depth if min_depth < float('inf') else 0.0

def _containment_violation(inner_verts, outer_verts):
    """
    Check if inner hexagon is contained in outer hexagon.
    Returns total penetration depth if any vertex is outside, 0 if contained.
    Uses the half-plane test for a convex polygon.
    """
    penalty = 0.0
    # For a regular hexagon with vertices in CCW order, the interior is to the left of each edge
    for i in range(6):
        v1 = outer_verts[i]
        v2 = outer_verts[(i + 1) % 6]
        edge = v2 - v1
        edge_len = np.linalg.norm(edge)
        if edge_len < 1e-12:
            continue
        # Inward normal (for CCW polygon, left of edge is inside)
        normal = np.array([-edge[1], edge[0]]) / edge_len
        
        # For each vertex of inner, check distance to this edge plane
        for iv in inner_verts:
            rel = iv - v1
            dist = np.dot(rel, normal)
            if dist < 0:
                # Vertex is outside this half-plane
                # The penetration depth is -dist
                penalty += -dist
    
    return penalty

def _compute_penalty(params, outer_side, outer_center, outer_angle):
    """Compute total penalty: containment + disjointness."""
    inner_verts_list = []
    for i in range(N):
        cx = params[3*i]
        cy = params[3*i+1]
        ang = params[3*i+2]
        inner_verts_list.append(_hex_verts(cx, cy, 1.0, ang))
    
    outer_verts = _hex_verts(outer_center[0], outer_center[1], outer_side, outer_angle)
    
    # Containment penalty
    cont_pen = 0.0
    for iv in inner_verts_list:
        cont_pen += _containment_violation(iv, outer_verts)
    
    # Disjointness penalty
    disj_pen = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            depth = _sat_overlap_depth(inner_verts_list[i], inner_verts_list[j])
            if depth > 0:
                disj_pen += depth
    
    return cont_pen + disj_pen

def _make_hex_grid(nrows, ncols):
    """Create hexagonal grid positions for unit hexagons."""
    positions = []
    for r in range(nrows):
        for c in range(ncols):
            x = c * 2.0 + (1.0 if r % 2 == 1 else 0.0)
            y = r * np.sqrt(3)
            positions.append((x, y))
    return positions

def _ring_init():
    """Create ring + center initial configuration."""
    positions = []
    # Center
    positions.append((0.0, 0.0))
    # Inner ring of 6
    for i in range(6):
        angle = np.radians(60 * i)
        positions.append((2.0 * np.cos(angle), 2.0 * np.sin(angle)))
    # Outer partial ring of 5
    for i in range(5):
        angle = np.radians(60 * i + 30)
        positions.append((3.0 * np.cos(angle), 3.0 * np.sin(angle)))
    
    arr = np.array(positions)
    centroid = arr.mean(axis=0)
    return (arr - centroid).tolist()

def _optimize(params_init, outer_side, outer_center, outer_angle, deadline, maxiter=300):
    """Run L-BFGS-B optimization."""
    def objective(params):
        return _compute_penalty(params, outer_side, outer_center, outer_angle)
    
    try:
        result = minimize(
            objective,
            params_init,
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'maxfun': 10000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
        )
        return result.x, result.fun
    except Exception:
        return params_init, objective(params_init)

def _polish(params, outer_side, outer_center, outer_angle, deadline):
    """High-precision polish pass with tighter tolerances."""
    remaining = deadline - time.time()
    if remaining < 2.0:
        return params
    
    def objective(params):
        return _compute_penalty(params, outer_side, outer_center, outer_angle)
    
    try:
        result = minimize(
            objective,
            params,
            method='L-BFGS-B',
            options={'maxiter': 500, 'maxfun': 20000, 'ftol': 1e-16, 'gtol': 1e-12, 'maxls': 100}
        )
        return result.x
    except Exception:
        return params

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit
    
    best_score = float('inf')
    best_result = None
    
    # Generate initial configurations
    configs = []
    
    # 3x4 hex grid
    pos34 = _make_hex_grid(3, 4)
    arr = np.array(pos34)
    centroid = arr.mean(axis=0)
    configs.append((arr - centroid).tolist())
    
    # 4x3 hex grid
    pos43 = _make_hex_grid(4, 3)
    arr = np.array(pos43)
    centroid = arr.mean(axis=0)
    configs.append((arr - centroid).tolist())
    
    # Ring + center
    configs.append(_ring_init())
    
    # Try different outer side lengths and angles
    outer_angles = [0.0, 30.0]
    outer_sides = [4.5, 4.2, 4.0, 3.8, 3.6]
    
    for outer_angle in outer_angles:
        for outer_side in outer_sides:
            if time.time() > deadline - 15.0:
                break
            
            outer_center = (0.0, 0.0)
            
            for config in configs:
                if time.time() > deadline - 15.0:
                    break
                
                # Initialize params: x, y, angle for each hex
                init_params = []
                for (x, y) in config:
                    init_params.extend([x, y, 0.0])
                init_params = np.array(init_params)
                
                # Run optimization
                opt_params, opt_val = _optimize(
                    init_params, outer_side, outer_center, outer_angle, deadline
                )
                
                # Check if solution is valid (penalty ≈ 0)
                if opt_val < 1e-6:
                    if outer_side < best_score:
                        best_score = outer_side
                        best_result = (opt_params, outer_center, outer_side, outer_angle)
                    
                    # Try smaller outer sides
                    for smaller_side in [outer_side - 0.15, outer_side - 0.1, outer_side - 0.05]:
                        if smaller_side < 3.0:
                            continue
                        if time.time() > deadline - 15.0:
                            break
                        
                        opt_params2, opt_val2 = _optimize(
                            opt_params, smaller_side, outer_center, outer_angle, deadline
                        )
                        if opt_val2 < 1e-6:
                            if smaller_side < best_score:
                                best_score = smaller_side
                                best_result = (opt_params2, outer_center, smaller_side, outer_angle)
    
    # Polish the best result
    if best_result is not None:
        opt_params, outer_center, outer_side, outer_angle = best_result
        polished_params = _polish(opt_params, outer_side, outer_center, outer_angle, deadline)
        final_val = _compute_penalty(polished_params, outer_side, outer_center, outer_angle)
        if final_val < 1e-5:
            best_result = (polished_params, outer_center, outer_side, outer_angle)
    
    # Fallback: safe default
    if best_result is None:
        inner = []
        pos34 = _make_hex_grid(3, 4)
        arr = np.array(pos34)
        centroid = arr.mean(axis=0)
        for (x, y) in (arr - centroid):
            inner.append([float(x), float(y), 0.0])
        return inner, (0.0, 0.0), 5.0, 0.0
    
    opt_params, outer_center, outer_side, outer_angle = best_result
    
    inner = []
    for i in range(N):
        x = float(opt_params[3*i])
        y = float(opt_params[3*i+1])
        ang = float(opt_params[3*i+2])
        inner.append([x, y, ang])
    
    # Final validation: ensure no overlaps
    inner_verts_list = []
    for i in range(N):
        cx = opt_params[3*i]
        cy = opt_params[3*i+1]
        ang = opt_params[3*i+2]
        inner_verts_list.append(_hex_verts(cx, cy, 1.0, ang))
    
    outer_verts = _hex_verts(outer_center[0], outer_center[1], outer_side, outer_angle)
    
    cont_pen = 0.0
    for iv in inner_verts_list:
        cont_pen += _containment_violation(iv, outer_verts)
    
    disj_pen = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            depth = _sat_overlap_depth(inner_verts_list[i], inner_verts_list[j])
            if depth > 0:
                disj_pen += depth
    
    if cont_pen > 1e-4 or disj_pen > 1e-4:
        # Invalid, return fallback
        inner = []
        pos34 = _make_hex_grid(3, 4)
        arr = np.array(pos34)
        centroid = arr.mean(axis=0)
        for (x, y) in (arr - centroid):
            inner.append([float(x), float(y), 0.0])
        return inner, (0.0, 0.0), 5.0, 0.0

    return inner, (float(outer_center[0]), float(outer_center[1])), float(outer_side), float(outer_angle)

# EVOLVE-BLOCK-END
