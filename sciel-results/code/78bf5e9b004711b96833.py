import numpy as np
import time

def _hex_vertices(cx, cy, angle_deg, s=1.0):
    """Return 6 vertices of a regular hexagon with center (cx,cy), rotation angle_deg, side length s."""
    a = np.deg2rad(angle_deg)
    offsets = []
    for k in range(6):
        theta = a + np.deg2rad(k * 60)
        offsets.append((s * np.cos(theta), s * np.sin(theta)))
    return np.array([(cx + ox, cy + oy) for ox, oy in offsets])

def _separating_axis_hex_hex(v1, v2):
    """Return True if two hexagons (arrays of 6 (x,y) vertices) are separated (non-overlapping)."""
    # Collect all edge normals from both hexagons
    axes = []
    for v in (v1, v2):
        for i in range(6):
            e = v[(i + 1) % 6] - v[i]
            # Normal perpendicular to edge
            n = np.array([-e[1], e[0]])
            norm = np.linalg.norm(n)
            if norm > 1e-12:
                axes.append(n / norm)
    
    for ax in axes:
        proj1 = np.dot(v1, ax)
        proj2 = np.dot(v2, ax)
        min1, max1 = proj1.min(), proj1.max()
        min2, max2 = proj2.min(), proj2.max()
        # Check for separation
        if max1 < min2 - 1e-9 or max2 < min1 - 1e-9:
            return True
    return False

def _hex_hex_overlap(v1, v2):
    """Return True if two hexagons overlap (interiors intersect)."""
    return not _separating_axis_hex_hex(v1, v2)

def _hex_in_hex(inner_v, outer_v, outer_cx, outer_cy, outer_side, outer_angle):
    """Return True if inner hexagon (vertices inner_v) is contained within outer hexagon."""
    # Check each vertex of inner hex against outer hex edges
    # Outer hex vertices
    a = np.deg2rad(outer_angle)
    outer_vertices = []
    for k in range(6):
        theta = a + np.deg2rad(k * 60)
        outer_vertices.append((outer_cx + outer_side * np.cos(theta), 
                               outer_cy + outer_side * np.sin(theta)))
    outer_vertices = np.array(outer_vertices)
    
    # For each edge of outer hex, check that inner vertex is on the interior side
    for i in range(6):
        p1 = outer_vertices[i]
        p2 = outer_vertices[(i + 1) % 6]
        # Edge vector
        e = p2 - p1
        # Inward normal (for a convex polygon with vertices in CCW order, inward is left of edge)
        n = np.array([-e[1], e[0]])
        # Check orientation
        # The center of the outer hex should be on the interior side
        center_to_p1 = np.array([outer_cx, outer_cy]) - p1
        # If dot(center_to_p1, n) > 0, then n points inward
        if np.dot(center_to_p1, n) < 0:
            n = -n
        # Check all inner vertices
        for v in inner_v:
            if np.dot(np.array(v) - p1, n) < -1e-9:
                return False
    return True

def _compute_outer_side(centers, angles, eps=1e-8):
    """Given inner hexagon centers and angles, find minimum outer side length for a hexagon at origin with variable angle.
    We'll try several outer angles and return the best side length."""
    best_side = np.inf
    best_outer_angle = 0.0
    
    # Try outer angles from 0 to 60 degrees in steps of 1 degree
    for outer_angle_deg in range(0, 61, 1):
        # For each inner hex, compute the required outer side to contain it
        # The outer hex has center at origin
        # For a point (x,y) to be inside a regular hexagon with side S, center origin, rotation angle:
        # We can use the support function: max over vertices of dot(v, direction) <= S * cos(30)
        # Actually, let's just check: a regular hexagon with side S, center origin, rotation a
        # contains a convex shape if all vertices of the shape are inside.
        
        # Compute vertices of each inner hex
        required_s = 0.0
        for cx, cy, ang in zip(centers, angles):
            v = _hex_vertices(cx, cy, ang)
            # For each vertex, compute the minimum S such that it's inside the outer hex
            # The outer hex (side S, center origin, rotation outer_angle) contains point p if:
            # For all 6 edges, the point is on the interior side.
            # Equivalent: the maximum over 6 directions of the support function.
            # For a regular hexagon, the width in direction theta is 2*S*cos(theta - a - k*60) for appropriate k.
            # Simpler: the point p is inside iff for all k, dot(p, n_k) <= S * cos(30) where n_k are outward normals.
            # Outward normals for hex with rotation a: at angles a+30+k*60 for k=0..5
            # Actually, the apothem is S*cos(30) = S*sqrt(3)/2
            # The constraint is: for all k, dot(p, u_k) <= S * sqrt(3)/2
            # where u_k = (cos(a+30+k*60), sin(a+30+k*60))
            
            a_rad = np.deg2rad(outer_angle_deg)
            for k in range(6):
                theta = a_rad + np.deg2rad(30 + k * 60)
                u = np.array([np.cos(theta), np.sin(theta)])
                for px, py in v:
                    val = px * u[0] + py * u[1]
                    # We need val <= S * sqrt(3)/2, so S >= 2*val/sqrt(3)
                    if val > 0:
                        s_req = 2.0 * val / np.sqrt(3)
                        if s_req > required_s:
                            required_s = s_req
        
        if required_s < best_side:
            best_side = required_s
            best_outer_angle = outer_angle_deg
    
    return best_side, best_outer_angle

def construct():
    start_time = time.time()
    DEADLINE = 145.0
    
    # Start with the flower packing (11 hexagons)
    eps = 1e-7
    centers = []
    angles = []
    
    # Ring 0
    centers.append((0.0, 0.0))
    angles.append(30.0)
    
    # Ring 1: 6 hexagons
    for k in range(6):
        ang = np.deg2rad(k * 60)
        r = np.sqrt(3) + eps
        centers.append((r * np.cos(ang), r * np.sin(ang)))
        angles.append(30.0)
    
    # Ring 2: 4 hexagons
    for k in range(4):
        ang = np.deg2rad(30 + k * 60)
        r = 3.0 + 2 * eps
        centers.append((r * np.cos(ang), r * np.sin(ang)))
        angles.append(30.0)
    
    # Compute initial outer side
    side, outer_angle = _compute_outer_side(centers, angles)
    
    # Nelder-Mead optimization
    # Parameters: 2*n centers + n angles
    n = len(centers)
    x0 = []
    for cx, cy in centers:
        x0.extend([cx, cy])
    for a in angles:
        x0.append(a)
    x0 = np.array(x0, dtype=float)
    
    def objective(x):
        # Check deadline
        if time.time() - start_time > DEADLINE:
            return 1e6
        
        # Parse parameters
        cs = []
        as_ = []
        for i in range(n):
            cx = x[2*i]
            cy = x[2*i+1]
            cs.append((cx, cy))
        for i in range(n):
            a = x[n*2 + i]
            as_.append(a)
        
        # Compute outer side
        s, oa = _compute_outer_side(cs, as_)
        
        # Check overlaps
        verts = [_hex_vertices(cx, cy, a) for cx, cy, a in zip(cs, as_)]
        for i in range(n):
            for j in range(i+1, n):
                if _hex_hex_overlap(verts[i], verts[j]):
                    return 1e6
        
        # Check containment
        for v in verts:
            if not _hex_in_hex(v, None, 0.0, 0.0, s, oa):
                return 1e6
        
        return s
    
    from scipy.optimize import minimize
    
    # Use Nelder-Mead
    result = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
    
    if time.time() - start_time > DEADLINE:
        # Return the initial solution
        inner = [[cx, cy, a] for cx, cy, a in zip(centers, angles)]
        return inner, (0.0, 0.0), side, outer_angle
    
    # Parse result
    if result.success or result.fun < side:
        x_opt = result.x
        cs_opt = []
        as_opt = []
        for i in range(n):
            cs_opt.append((x_opt[2*i], x_opt[2*i+1]))
        for i in range(n):
            as_opt.append(x_opt[n*2 + i])
        
        # Verify validity
        verts = [_hex_vertices(cx, cy, a) for cx, cy, a in zip(cs_opt, as_opt)]
        valid = True
        for i in range(n):
            for j in range(i+1, n):
                if _hex_hex_overlap(verts[i], verts[j]):
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            s_opt, oa_opt = _compute_outer_side(cs_opt, as_opt)
            for v in verts:
                if not _hex_in_hex(v, None, 0.0, 0.0, s_opt, oa_opt):
                    valid = False
                    break
        
        if valid and s_opt < side:
            side = s_opt
            outer_angle = oa_opt
            centers = cs_opt
            angles = as_opt
    
    inner = [[cx, cy, a] for cx, cy, a in zip(centers, angles)]
    return inner, (0.0, 0.0), side, outer_angle

