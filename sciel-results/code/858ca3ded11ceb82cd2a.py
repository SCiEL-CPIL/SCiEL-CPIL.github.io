import numpy as np
from scipy.optimize import minimize

N = 11

def get_hex_vertices(cx, cy, S, angle):
    """
    Returns vertices of a regular hexagon with side length S, center (cx, cy), rotation angle.
    Angle 0 corresponds to pointy-top (vertex at 90 deg, i.e., pi/2).
    Wait, standard pointy top has vertices at 30, 90, 150...
    Let's stick to: angles = pi/6 + k*pi/3.
    """
    base_angles = np.pi / 6 + np.pi / 3 * np.arange(6)
    angles = base_angles + angle
    x = cx + S * np.cos(angles)
    y = cy + S * np.sin(angles)
    return np.column_stack((x, y))

def point_to_segment_distance_sq(p, a, b):
    """
    Squared distance from point p to segment ab.
    """
    ab = b - a
    ap = p - a
    len_sq = np.dot(ab, ab)
    if len_sq < 1e-12:
        return np.dot(ap, ap)
    t = np.dot(ap, ab) / len_sq
    t = np.clip(t, 0.0, 1.0)
    proj = a + t * ab
    diff = p - proj
    return np.dot(diff, diff)

def signed_distance_point_to_polygon(p, verts):
    """
    Computes signed distance from point p to convex polygon defined by verts (CCW).
    Negative if inside, positive if outside.
    """
    n = len(verts)
    inside = True
    min_dist_sq = float('inf')
    min_dist = float('inf')
    
    for i in range(n):
        a = verts[i]
        b = verts[(i + 1) % n]
        
        # Check inside
        # Cross product (b-a) x (p-a)
        cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
        if cross < 0: # Assuming CCW, inside means left of edge
            inside = False
            break
        
        # Distance to edge
        d_sq = point_to_segment_distance_sq(p, a, b)
        if d_sq < min_dist_sq:
            min_dist_sq = d_sq
            min_dist = np.sqrt(d_sq)
            
    if inside:
        return -min_dist
    else:
        return min_dist

def min_clearance_between_hexagons(verts1, verts2):
    """
    Returns the minimum clearance between two convex hexagons.
    Negative if they overlap.
    """
    min_cl = float('inf')
    # Distance from vertices of 1 to polygon 2
    for i in range(6):
        p = verts1[i]
        d = signed_distance_point_to_polygon(p, verts2)
        if d < min_cl:
            min_cl = d
            
    # Distance from vertices of 2 to polygon 1
    for i in range(6):
        p = verts2[i]
        d = signed_distance_point_to_polygon(p, verts1)
        if d < min_cl:
            min_cl = d
            
    return min_cl

def min_clearance_to_outer(verts_inner, verts_outer):
    """
    Returns minimum clearance from inner hex to outer boundary.
    Negative if outside.
    """
    min_cl = float('inf')
    for i in range(6):
        p = verts_inner[i]
        d = signed_distance_point_to_polygon(p, verts_outer)
        if d < min_cl:
            min_cl = d
    return min_cl

def construct():
    # Variables: x1, y1, a1, ..., xN, yN, aN, S
    # Outer angle fixed to 0 for symmetry
    
    def objective(params):
        S = params[-1]
        if S < 1.0:
            S = 1.0
            
        centers_x = params[0:2*N:3]
        centers_y = params[1:2*N+1:3]
        angles = params[2:2*N+2:3]
        
        outer_verts = get_hex_vertices(0.0, 0.0, S, 0.0)
        
        inner_verts = []
        for i in range(N):
            v = get_hex_vertices(centers_x[i], centers_y[i], 1.0, angles[i])
            inner_verts.append(v)
            
        penalty = 0.0
        
        # Non-overlap between inner hexagons
        for i in range(N):
            for j in range(i + 1, N):
                cl = min_clearance_between_hexagons(inner_verts[i], inner_verts[j])
                if cl < 0:
                    penalty += cl * cl
        
        # Containment in outer hexagon
        for i in range(N):
            cl = min_clearance_to_outer(inner_verts[i], outer_verts)
            if cl < 0:
                penalty += cl * cl
        
        return S + 50.0 * penalty

    # Initial guesses
    best_S = float('inf')
    best_params = None
    
    # Multi-start
    num_starts = 10
    
    for start_idx in range(num_starts):
        if start_idx == 0:
            # Hexagonal lattice init
            d = np.sqrt(3) + 0.01
            x0 = np.zeros(3 * N + 1)
            
            # Center
            x0[0] = 0.0
            x0[1] = 0.0
            x0[2] = 0.0
            
            # 6 neighbors
            idx = 1
            for k in range(6):
                ang = k * np.pi / 3
                x0[3*idx] = d * np.cos(ang)
                x0[3*idx+1] = d * np.sin(ang)
                x0[3*idx+2] = 0.0
                idx += 1
                
            # 4 more at next ring, optimized positions
            # Place them at 30, 150, 270, 390 deg?
            # Or just randomize slightly
            for k in range(4):
                ang = (k + 0.5) * np.pi / 2
                r = 2.0 * d
                x0[3*idx] = r * np.cos(ang)
                x0[3*idx+1] = r * np.sin(ang)
                x0[3*idx+2] = 0.0
                idx += 1
                
            x0[-1] = 4.0
        else:
            # Random starts with tight clustering
            np.random.seed(start_idx + 100)
            x0 = np.zeros(3 * N + 1)
            # Cluster in a small area
            for i in range(N):
                r = np.sqrt(np.random.random()) * 2.0
                theta = np.random.random() * 2 * np.pi
                x0[3*i] = r * np.cos(theta)
                x0[3*i+1] = r * np.sin(theta)
                x0[3*i+2] = np.random.random() * np.pi
            x0[-1] = 4.0

        # Optimize
        try:
            # Use L-BFGS-B
            res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8})
            
            # Check validity
            S_val = res.x[-1]
            # Recalculate penalty to be sure
            # We can extract penalty from res.fun
            # res.fun = S + 50 * penalty
            # penalty = (res.fun - S) / 50
            penalty_val = (res.fun - S_val) / 50.0
            
            if penalty_val < 1e-4:
                if S_val < best_S:
                    best_S = S_val
                    best_params = res.x
        except Exception:
            pass

    if best_params is None:
        # Fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    inner = []
    for i in range(N):
        x = best_params[3*i]
        y = best_params[3*i+1]
        ang = best_params[3*i+2]
        inner.append([x, y, np.degrees(ang)])
    
    outer_center = (0.0, 0.0)
    outer_side = best_params[-1]
    outer_angle = 0.0
    
    return inner, outer_center, outer_side, outer_angle
