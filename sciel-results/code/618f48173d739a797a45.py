import numpy as np
from scipy.optimize import minimize

N = 11

def get_hex_vertices(cx, cy, angle, side=1.0):
    """
    Get vertices of a regular hexagon.
    angle=0 corresponds to pointy-top (vertices at 30, 90, 150, ... degrees).
    """
    base_angles = np.pi / 6 + np.pi / 3 * np.arange(6)
    angles = base_angles + angle
    x = cx + side * np.cos(angles)
    y = cy + side * np.sin(angles)
    return np.column_stack((x, y))

def smooth_min_distance_hexagons(verts1, verts2, eps=0.01):
    """
    Compute a smooth approximation of the minimum distance between two convex polygons (hexagons).
    Returns negative value if overlapping (approximately), positive if separated.
    Uses soft-min of distances from vertices of one to edges of the other.
    """
    dists = []
    # Vertices of 1 to edges of 2
    for i in range(6):
        p = verts1[i]
        for j in range(6):
            a = verts2[j]
            b = verts2[(j + 1) % 6]
            ab = b - a
            ap = p - a
            len_sq = np.dot(ab, ab)
            if len_sq < 1e-10:
                d = np.linalg.norm(ap)
            else:
                t = np.dot(ap, ab) / len_sq
                t = np.clip(t, 0, 1)
                proj = a + t * ab
                d = np.linalg.norm(p - proj)
            dists.append(d)
    
    # Vertices of 2 to edges of 1
    for i in range(6):
        p = verts2[i]
        for j in range(6):
            a = verts1[j]
            b = verts1[(j + 1) % 6]
            ab = b - a
            ap = p - a
            len_sq = np.dot(ab, ab)
            if len_sq < 1e-10:
                d = np.linalg.norm(ap)
            else:
                t = np.dot(ap, ab) / len_sq
                t = np.clip(t, 0, 1)
                proj = a + t * ab
                d = np.linalg.norm(p - proj)
            dists.append(d)
    
    dists = np.array(dists)
    # Soft min: -log-sum-exp(-dists) approximates min(dists)
    # To handle numerical stability, shift by min
    min_d = np.min(dists)
    soft_min = -eps * np.log(np.sum(np.exp(-(dists - min_d) / eps))) + min_d
    return soft_min

def construct():
    """
    Minimize the outer hexagon side length S such that N unit hexagons fit inside.
    Uses penalty method with smooth distance functions.
    """
    
    def objective(params):
        S = params[-1]
        if S < 1.0:
            S = 1.0
        
        # Extract parameters
        centers_x = params[0:2*N:3]
        centers_y = params[1:2*N+1:3]
        angles = params[2:2*N+2:3]
        
        outer_angle = 0.0
        
        # Calculate vertices
        inner_verts = []
        for i in range(N):
            v = get_hex_vertices(centers_x[i], centers_y[i], angles[i], side=1.0)
            inner_verts.append(v)
            
        outer_verts = get_hex_vertices(0, 0, outer_angle, side=S)
        
        penalty = 0.0
        
        # Check non-overlap between inner hexagons
        for i in range(N):
            for j in range(i + 1, N):
                d = smooth_min_distance_hexagons(inner_verts[i], inner_verts[j])
                if d < 0:
                    penalty += d * d
        
        # Check containment: inner hexagons must be inside outer
        # Distance from inner vertices to outer boundary edges should be >= 0
        # We check if all inner vertices are inside the outer hexagon
        # A point is inside if its distance to all edges (signed) is >= 0
        # For a convex polygon, we can check signed distance to each edge
        for i in range(N):
            v = inner_verts[i]
            for edge_i in range(6):
                a = outer_verts[edge_i]
                b = outer_verts[(edge_i + 1) % 6]
                # Edge vector
                ab = b - a
                # Normal vector pointing inward
                # For a counter-clockwise polygon, inward normal is (ab_y, -ab_x) normalized? 
                # Let's check orientation.
                # Standard hexagon vertices at 30, 90, 150... are CCW.
                # ab = (cos(90)-cos(30), sin(90)-sin(30)) = (-0.866, 0.5) approx
                # Inward normal should point towards center (0,0).
                # Normal n = (ab_y, -ab_x) = (0.5, 0.866). Dot with (0,0) - midpoint?
                # Midpoint of edge is roughly (0.5, 0.866). Dot with n is positive.
                # So n = (ab_y, -ab_x) points outward?
                # Let's verify. Center is (0,0). Vector from midpoint to center is -midpoint.
                # If n . (-midpoint) > 0, n points inward.
                # mid = (a+b)/2.
                # n = (ab_y, -ab_x).
                # Let's just compute signed distance: (p - a) . n / |n|
                # If the polygon is CCW, the interior is to the left of the edge.
                # Left normal is (-ab_y, ab_x).
                # Let's test: Edge from 30 deg to 90 deg.
                # a = (cos30, sin30) = (0.866, 0.5)
                # b = (cos90, sin90) = (0, 1)
                # ab = (-0.866, 0.5)
                # Left normal = (-0.5, -0.866).
                # Point (0,0) is the center.
                # (0 - 0.866) . (-0.5, -0.866) = 0.433 + 0.433 = 0.866 > 0.
                # So left normal points inward.
                
                n = np.array([-ab[1], ab[0]])
                n_norm = np.linalg.norm(n)
                if n_norm < 1e-10:
                    continue
                n = n / n_norm
                
                # Signed distance of all vertices of inner hex i to this edge
                # p - a
                vecs = v - a
                dists = np.dot(vecs, n)
                
                # If any distance is negative, the vertex is outside
                min_dist = np.min(dists)
                if min_dist < 0:
                    penalty += min_dist * min_dist
        
        return S + 50.0 * penalty

    # Initial guesses
    best_S = float('inf')
    best_params = None
    
    num_starts = 10
    
    for start_idx in range(num_starts):
        if start_idx == 0:
            # Hexagonal lattice
            d = np.sqrt(3) + 0.05
            init_x = [0.0]
            init_y = [0.0]
            init_a = [0.0]
            
            for k in range(6):
                ang = k * np.pi / 3
                init_x.append(d * np.cos(ang))
                init_y.append(d * np.sin(ang))
                init_a.append(0.0)
            
            # Add 4 more
            for k in range(4):
                ang = (k + 0.5) * np.pi / 2
                init_x.append(2.1 * d * np.cos(ang))
                init_y.append(2.1 * d * np.sin(ang))
                init_a.append(0.0)
            
            x0 = np.zeros(3 * N + 1)
            for i in range(N):
                x0[3*i] = init_x[i]
                x0[3*i+1] = init_y[i]
                x0[3*i+2] = init_a[i]
            x0[-1] = 4.0
        else:
            np.random.seed(start_idx * 12345 + 67890)
            x0 = np.zeros(3 * N + 1)
            for i in range(N):
                r = np.sqrt(np.random.random()) * 3.5
                theta = np.random.random() * 2 * np.pi
                x0[3*i] = r * np.cos(theta)
                x0[3*i+1] = r * np.sin(theta)
                x0[3*i+2] = np.random.random() * np.pi
            x0[-1] = 4.0

        try:
            res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8})
            if res.fun < best_S:
                S_val = res.x[-1]
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
