"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 12

# EVOLVE-BLOCK-START
def _hex_vertices(cx, cy, side, angle_deg=0.0):
    """Return vertices of a regular hexagon centered at (cx, cy) with given side length and rotation."""
    angle_rad = math.radians(angle_deg)
    # Regular hexagon vertices at angles 0, 60, 120, 180, 240, 300 degrees offset by angle
    angles = [angle_rad + math.pi * k / 3.0 for k in range(6)]
    xs = cx + side * np.cos(angles)
    ys = cy + side * np.sin(angles)
    return np.column_stack([xs, ys])

def _point_in_hexagon(px, py, cx, cy, side, angle_deg=0.0):
    """Check if point (px, py) is inside hexagon centered at (cx, cy) with side length and rotation."""
    # Transform point to hexagon's local coordinate system
    dx = px - cx
    dy = py - cy
    angle_rad = math.radians(angle_deg)
    # Rotate point by -angle
    cos_a = math.cos(-angle_rad)
    sin_a = math.sin(-angle_rad)
    lx = dx * cos_a - dy * sin_a
    ly = dx * sin_a + dy * cos_a
    
    # For a regular hexagon with side length s, the distance from center to each side is s * sqrt(3) / 2
    # The hexagon can be defined as |lx| <= s*sqrt(3)/2 and |ly| <= s*sqrt(3)/2 and |lx + ly| <= s*sqrt(3) ... 
    # Actually, let's use the standard form. A regular hexagon with flat top/bottom (angle=0) has vertices at:
    # (s, 0), (s/2, s*sqrt(3)/2), (-s/2, s*sqrt(3)/2), (-s, 0), (-s/2, -s*sqrt(3)/2), (s/2, -s*sqrt(3)/2)
    # The six sides are lines. Let's check using the support function or half-plane tests.
    
    # For angle=0 (pointy top? or flat top?): Let's assume the vertices are at angles 0, 60, ...
    # So at angle=0, vertices are at (s,0), (s/2, s*sqrt(3)/2), etc.
    # The hexagon is the intersection of 6 half-planes.
    # Side 1: between (s,0) and (s/2, s*sqrt(3)/2). Normal pointing inward.
    # This is getting complex. Let's use a simpler check: distance from center in the direction of each normal.
    
    # The 6 outward normals for a hexagon with vertex at angle 0 are at angles 30, 90, 150, 210, 270, 330.
    # Wait, the normal to the edge connecting vertex at 0 and vertex at 60 is at angle 30.
    # The distance from center to each edge is s * sqrt(3) / 2.
    
    normals = [angle_rad + math.pi * k / 3.0 + math.pi / 6.0 for k in range(6)]
    inradius = side * math.sqrt(3) / 2.0
    
    for n in normals:
        dist = lx * math.cos(n) + ly * math.sin(n)
        if dist > inradius:
            return False
    return True

def _hex_hex_distance(v1, v2, s1, a1, v2c, v2s, v2a):
    """This is not the right approach. Let's compute the minimum distance between two hexagons."""
    pass

def _min_distance_hex_hex(c1, s1, a1, c2, s2, a2):
    """Compute minimum distance between two hexagons. If they overlap, return negative value (penetration depth)."""
    # We can use the separating axis theorem. For convex polygons, if there's a separating axis, they don't overlap.
    # The axes to check are the normals to all edges of both polygons.
    verts1 = _hex_vertices(c1[0], c1[1], s1, a1)
    verts2 = _hex_vertices(c2[0], c2[1], s2, a2)
    
    # Get edge normals for both
    def get_normals(verts):
        normals = []
        for i in range(len(verts)):
            j = (i + 1) % len(verts)
            edge = verts[j] - verts[i]
            # Normal perpendicular to edge
            n = np.array([-edge[1], edge[0]])
            n_norm = np.linalg.norm(n)
            if n_norm > 1e-10:
                n = n / n_norm
                normals.append(n)
        return normals
    
    normals1 = get_normals(verts1)
    normals2 = get_normals(verts2)
    all_normals = normals1 + normals2
    
    min_dist = float('inf')
    
    for n in all_normals:
        # Project both polygons onto this axis
        proj1 = verts1 @ n
        proj2 = verts2 @ n
        
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        # Gap between the two projections
        if max1 < min2:
            gap = min2 - max1
        elif max2 < min1:
            gap = min1 - max2
        else:
            gap = 0.0  # Overlapping on this axis
        
        min_dist = min(min_dist, gap)
    
    return min_dist

def _outer_hex_contains_inner(inner_verts, outer_c, outer_s, outer_a):
    """Check if all inner hexagon vertices are inside the outer hexagon."""
    for v in inner_verts:
        if not _point_in_hexagon(v[0], v[1], outer_c[0], outer_c[1], outer_s, outer_a):
            return False
    return True

def _penetration_depth(inner_c, inner_s, inner_a, outer_c, outer_s, outer_a):
    """Compute how much the inner hexagon penetrates the outer hexagon. 0 if fully inside, negative if penetrating."""
    inner_verts = _hex_vertices(inner_c[0], inner_c[1], inner_s, inner_a)
    min_dist = float('inf')
    
    # Check distance from each inner vertex to the outer hexagon boundary
    # Actually, let's use the separating axis theorem between the inner hex and the outer hex.
    # If they don't overlap, the distance is positive. If they do, it's negative.
    # But we want to know if the inner is inside the outer.
    
    # For each vertex of the inner hex, compute the signed distance to the outer hex.
    # If any vertex is outside, we have a problem.
    
    # Let's compute the minimum distance from the inner hex to the outside of the outer hex.
    # This is equivalent to: for each edge of the outer hex, compute the distance from the inner hex to that edge.
    
    outer_verts = _hex_vertices(outer_c[0], outer_c[1], outer_s, outer_a)
    
    # Get the 6 edge normals of the outer hex (pointing outward)
    def get_outward_normals(verts):
        normals = []
        for i in range(len(verts)):
            j = (i + 1) % len(verts)
            edge = verts[j] - verts[i]
            # For a convex polygon with vertices in counterclockwise order, the outward normal is (edge[1], -edge[0]) normalized
            # Let's check the orientation. If vertices are CCW, the interior is to the left of each edge.
            # So the outward normal is to the right: (edge[1], -edge[0])
            n = np.array([edge[1], -edge[0]])
            n_norm = np.linalg.norm(n)
            if n_norm > 1e-10:
                n = n / n_norm
                normals.append(n)
        return normals
    
    normals = get_outward_normals(outer_verts)
    
    # For each outward normal, compute the maximum projection of the inner hex onto that normal.
    # The distance from the inner hex to the edge is: (projection of outer vertex) - (max projection of inner hex)
    # If this is negative, the inner hex penetrates that edge.
    
    min_margin = float('inf')
    for i, n in enumerate(normals):
        # The edge is at a certain distance from the center
        # Project the outer vertex onto the normal
        edge_point = outer_verts[i]
        edge_proj = edge_point @ n
        
        # Project all inner vertices onto the normal
        inner_projs = inner_verts @ n
        max_inner_proj = np.max(inner_projs)
        
        # The margin is how much the inner hex is inside the edge
        margin = edge_proj - max_inner_proj
        min_margin = min(min_margin, margin)
    
    return min_margin

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5 second safety margin
    
    # We need to place 12 unit hexagons inside a larger hexagon, minimizing the outer side length.
    # Each inner hexagon has: x, y, angle (3 parameters each) = 36 parameters
    # Outer hexagon: x, y, side, angle (4 parameters)
    # Total: 40 parameters
    
    # Objective: minimize the outer side length, subject to:
    # 1. All inner hexagons are fully inside the outer hexagon
    # 2. No two inner hexagons overlap
    
    # We'll use a penalty-based approach with differential evolution.
    
    # Initial guess for the outer side length
    # 12 unit hexagons: if arranged in a 3x4 grid or similar, the outer hex needs to be quite large.
    # A single unit hex has circumradius 1. If we put them in a row, 6 in a row would need length ~6*2=12?
    # Actually, let's start with a reasonable initial side length.
    
    # For a hexagonal packing, the density is high. Let's estimate:
    # Area of 12 unit hexagons = 12 * (3*sqrt(3)/2) * 1^2 = 12 * 2.598 = 31.18
    # Area of outer hexagon with side s = (3*sqrt(3)/2) * s^2
    # So s^2 >= 31.18 / 2.598 = 12, so s >= sqrt(12) = 3.46
    # But due to packing inefficiency, we'll need more. Let's start with s = 5.0
    
    def objective(params):
        # params: [x1, y1, a1, x2, y2, a2, ..., x12, y12, a12, ox, oy, os, oa]
        inner_params = params[:36].reshape(12, 3)
        outer_c = (params[36], params[37])
        outer_s = params[38]
        outer_a = params[39]
        
        # Ensure outer side length is positive
        outer_s = max(outer_s, 0.5)
        
        # Compute penalty
        penalty = 0.0
        
        # Check each inner hexagon is inside the outer hexagon
        for i in range(12):
            ic = (inner_params[i, 0], inner_params[i, 1])
            ia = inner_params[i, 2]
            margin = _penetration_depth(ic, 1.0, ia, outer_c, outer_s, outer_a)
            if margin < 0:
                penalty += (-margin) ** 2 * 100.0
        
        # Check no two inner hexagons overlap
        for i in range(12):
            for j in range(i + 1, 12):
                ic1 = (inner_params[i, 0], inner_params[i, 1])
                ia1 = inner_params[i, 2]
                ic2 = (inner_params[j, 0], inner_params[j, 1])
                ia2 = inner_params[j, 2]
                dist = _min_distance_hex_hex(ic1, 1.0, ia1, ic2, 1.0, ia2)
                if dist < 0:
                    penalty += (-dist) ** 2 * 100.0
        
        return outer_s + penalty
    
    # Define bounds for each parameter
    # Inner hexagons: x, y in [-10, 10], angle in [0, 360)
    # Outer: x, y in [-10, 10], side in [3, 10], angle in [0, 360)
    bounds = []
    for _ in range(12):
        bounds.append((-10.0, 10.0))  # x
        bounds.append((-10.0, 10.0))  # y
        bounds.append((0.0, 360.0))   # angle
    bounds.append((-10.0, 10.0))  # outer x
    bounds.append((-10.0, 10.0))  # outer y
    bounds.append((3.0, 10.0))    # outer side
    bounds.append((0.0, 360.0))   # outer angle
    
    # Run differential evolution
    best_result = None
    best_score = float('inf')
    
    try:
        result = differential_evolution(
            objective,
            bounds=bounds,
            maxiter=50,
            popsize=20,
            tol=1e-6,
            seed=42,
            polish=True,
            update_interval=10,
            callback=lambda xk, convergence: None
        )
        
        if result.success and result.fun < best_score:
            best_score = result.fun
            best_result = result.x
            
    except Exception as e:
        pass
    
    # If we have time, try a few more runs with different seeds
    if time.time() < deadline - 30.0:
        for seed in [1, 2, 3]:
            if time.time() > deadline - 10.0:
                break
            try:
                result = differential_evolution(
                    objective,
                    bounds=bounds,
                    maxiter=30,
                    popsize=15,
                    tol=1e-6,
                    seed=seed,
                    polish=True
                )
                
                if result.success and result.fun < best_score:
                    best_score = result.fun
                    best_result = result.x
                    
            except Exception as e:
                pass
    
    # If we didn't find a good result, use a simple initial guess
    if best_result is None:
        # Simple initial guess: arrange hexagons in a hexagonal pattern
        inner = []
        # Center hexagon
        inner.append([0.0, 0.0, 0.0])
        # 6 surrounding
        for k in range(6):
            angle = k * 60.0
            x = 2.0 * math.cos(math.radians(angle))
            y = 2.0 * math.sin(math.radians(angle))
            inner.append([x, y, 0.0])
        # 5 more in the next ring
        for k in range(6):
            angle = k * 60.0 + 30.0
            x = 4.0 * math.cos(math.radians(angle))
            y = 4.0 * math.sin(math.radians(angle))
            if len(inner) < 12:
                inner.append([x, y, 0.0])
        
        return inner, (0.0, 0.0), 6.0, 0.0
    
    # Extract the best result
    inner_params = best_result[:36].reshape(12, 3)
    outer_c = (best_result[36], best_result[37])
    outer_s = max(best_result[38], 0.5)
    outer_a = best_result[39]
    
    # Polish with local optimization
    if time.time() < deadline - 15.0:
        try:
            def local_obj(params):
                inner_params = params[:36].reshape(12, 3)
                outer_c = (params[36], params[37])
                outer_s = max(params[38], 0.5)
                outer_a = params[39]
                
                penalty = 0.0
                for i in range(12):
                    ic = (inner_params[i, 0], inner_params[i, 1])
                    ia = inner_params[i, 2]
                    margin = _penetration_depth(ic, 1.0, ia, outer_c, outer_s, outer_a)
                    if margin < 0:
                        penalty += (-margin) ** 2 * 100.0
                
                for i in range(12):
                    for j in range(i + 1, 12):
                        ic1 = (inner_params[i, 0], inner_params[i, 1])
                        ia1 = inner_params[i, 2]
                        ic2 = (inner_params[j, 0], inner_params[j, 1])
                        ia2 = inner_params[j, 2]
                        dist = _min_distance_hex_hex(ic1, 1.0, ia1, ic2, 1.0, ia2)
                        if dist < 0:
                            penalty += (-dist) ** 2 * 100.0
                
                return outer_s + penalty
            
            res = minimize(local_obj, best_result, method='L-BFGS-B', 
                          bounds=bounds, options={'maxiter': 100, 'ftol': 1e-10})
            
            if res.fun < best_score:
                best_score = res.fun
                best_result = res.x
                inner_params = best_result[:36].reshape(12, 3)
                outer_c = (best_result[36], best_result[37])
                outer_s = max(best_result[38], 0.5)
                outer_a = best_result[39]
                
        except Exception as e:
            pass
    
    inner = []
    for i in range(12):
        inner.append([inner_params[i, 0], inner_params[i, 1], inner_params[i, 2]])
    
    return inner, outer_c, outer_s, outer_a

# EVOLVE-BLOCK-END
