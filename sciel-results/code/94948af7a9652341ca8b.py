import numpy as np
from scipy.optimize import minimize

N = 11

def get_hex_vertices(cx, cy, r, angle):
    # r is circumradius (side length for regular hex)
    base_angles = np.pi / 6 + np.pi / 3 * np.arange(6)
    angles = base_angles + angle
    x = cx + r * np.cos(angles)
    y = cy + r * np.sin(angles)
    return np.column_stack((x, y))

def point_to_segment_dist_vec(points, a, b):
    # points: (M, 2)
    # a, b: (2,)
    # Returns min distance from any point to segment ab
    ab = b - a
    ab_len_sq = np.dot(ab, ab)
    if ab_len_sq < 1e-12:
        return np.min(np.linalg.norm(points - a, axis=1))
    
    # Project points onto line ab
    # t = dot(p-a, b-a) / dot(b-a, b-a)
    # t shape (M,)
    t = np.einsum('ij,j->i', points - a, ab) / ab_len_sq
    t = np.clip(t, 0, 1)
    # proj = a + t * ab
    # proj shape (M, 2)
    proj = a + np.outer(t, ab)
    dists = np.linalg.norm(points - proj, axis=1)
    return np.min(dists)

def min_dist_between_hexes(v1, v2):
    # v1, v2 are (6, 2)
    min_d = float('inf')
    # Vertices of v1 to edges of v2
    for i in range(6):
        a = v2[i]
        b = v2[(i + 1) % 6]
        d = point_to_segment_dist_vec(v1, a, b)
        if d < min_d:
            min_d = d
    # Vertices of v2 to edges of v1
    for i in range(6):
        a = v1[i]
        b = v1[(i + 1) % 6]
        d = point_to_segment_dist_vec(v2, a, b)
        if d < min_d:
            min_d = d
    return min_d

def min_dist_to_outer(v_inner, v_outer):
    # v_inner (6,2), v_outer (6,2)
    # We need distance from inner hex to outer boundary.
    # This is min over vertices of inner to edges of outer?
    # No, if inner is inside outer, the distance is min distance from any point in inner to any point in outer boundary.
    # Since both are convex, min distance is achieved at vertex-edge or edge-edge.
    # However, for containment, we just need to ensure all vertices of inner are inside outer?
    # No, a hexagon can have all vertices inside but an edge sticking out? No, convex polygon is intersection of half-planes.
    # If all vertices of a convex polygon are inside another convex polygon, the whole polygon is inside.
    # So, checking vertices of inner against edges of outer is sufficient for containment?
    # Wait. If a vertex of inner is inside, it's fine. But what if an edge of inner crosses an edge of outer?
    # If all vertices of inner are strictly inside outer, the edges are also inside.
    # So, to check if inner is inside outer, we just need to check if all vertices of inner are inside outer.
    # The distance from inner to outer boundary is the minimum distance from any vertex of inner to the boundary of outer?
    # Actually, the closest point on the inner hex to the outer boundary could be on an edge of the inner hex, not a vertex.
    # Example: Inner hex is large, outer hex is slightly larger and aligned. The midpoints of edges are closest.
    # So we must check:
    # 1. Vertices of inner to edges of outer.
    # 2. Vertices of outer to edges of inner? No, inner is inside.
    # 3. Edges of inner to vertices of outer?
    # Generally, min distance between two convex polygons is min over:
    # - Vertices of A to Edges of B
    # - Vertices of B to Edges of A
    # - Edge-Edge intersections (if they cross, dist=0).
    # So we can reuse min_dist_between_hexes logic.
    return min_dist_between_hexes(v_inner, v_outer)

def construct():
    # Variables: x1, y1, a1, ..., xN, yN, aN, S, outer_angle
    # Total 3*N + 2 variables
    
    def objective(params):
        S = params[-2]
        outer_angle = params[-1]
        
        if S < 1.0:
            S = 1.0
            
        centers_x = params[0:2*N:3]
        centers_y = params[1:2*N+1:3]
        angles = params[2:2*N+2:3]
        
        inner_verts = []
        for i in range(N):
            v = get_hex_vertices(centers_x[i], centers_y[i], 1.0, angles[i])
            inner_verts.append(v)
            
        outer_verts = get_hex_vertices(0.0, 0.0, S, outer_angle)
        
        penalty = 0.0
        
        # Non-overlap
        for i in range(N):
            for j in range(i + 1, N):
                d = min_dist_between_hexes(inner_verts[i], inner_verts[j])
                if d < 0:
                    penalty += d * d
                else:
                    # Small reward for gap to encourage spreading? 
                    # No, just penalty for violation is enough if S is minimized.
                    pass
                    
        # Containment
        for i in range(N):
            d = min_dist_to_outer(inner_verts[i], outer_verts)
            if d < 0:
                penalty += d * d
                # Also check if vertices are inside? 
                # min_dist_to_outer handles it. If d < 0, they overlap.
                
        return S + 500.0 * penalty

    best_S = float('inf')
    best_params = None
    
    num_starts = 30
    
    for start_idx in range(num_starts):
        np.random.seed(start_idx * 101 + 42)
        
        if start_idx == 0:
            # Hexagonal lattice start
            # 1 center, 6 neighbors, 4 more
            d = np.sqrt(3) + 0.1
            init_x = [0.0]
            init_y = [0.0]
            init_a = [0.0]
            for k in range(6):
                ang = k * np.pi / 3
                init_x.append(d * np.cos(ang))
                init_y.append(d * np.sin(ang))
                init_a.append(0.0)
            # 4 more
            for k in range(4):
                ang = (k + 0.5) * np.pi / 2
                r = 2.2 * d
                init_x.append(r * np.cos(ang))
                init_y.append(r * np.sin(ang))
                init_a.append(0.0)
            
            x0 = np.zeros(3 * N + 2)
            for i in range(N):
                x0[3*i] = init_x[i]
                x0[3*i+1] = init_y[i]
                x0[3*i+2] = init_a[i]
            x0[-2] = 6.0
            x0[-1] = 0.0
        else:
            # Random start
            x0 = np.zeros(3 * N + 2)
            for i in range(N):
                r = np.sqrt(np.random.random()) * 4.0
                theta = np.random.random() * 2 * np.pi
                x0[3*i] = r * np.cos(theta)
                x0[3*i+1] = r * np.sin(theta)
                x0[3*i+2] = np.random.random() * 2 * np.pi
            x0[-2] = 6.0
            x0[-1] = np.random.random() * 2 * np.pi

        try:
            res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-10})
            
            # Check validity
            S_val = res.x[-2]
            # Recalculate penalty to be sure
            # We can't easily get penalty from res.fun without re-running objective logic, 
            # but res.fun = S + 500 * penalty.
            # If penalty is 0, res.fun = S.
            # If penalty > 0, res.fun > S.
            # We want S such that penalty is effectively 0.
            
            # Let's verify by checking constraints manually
            valid = True
            centers_x = res.x[0:2*N:3]
            centers_y = res.x[1:2*N+1:3]
            angles = res.x[2:2*N+2:3]
            outer_angle = res.x[-1]
            
            inner_verts = []
            for i in range(N):
                v = get_hex_vertices(centers_x[i], centers_y[i], 1.0, angles[i])
                inner_verts.append(v)
            outer_verts = get_hex_vertices(0.0, 0.0, S_val, outer_angle)
            
            min_gap = float('inf')
            for i in range(N):
                for j in range(i + 1, N):
                    d = min_dist_between_hexes(inner_verts[i], inner_verts[j])
                    if d < 1e-6:
                        valid = False
                        break
                    min_gap = min(min_gap, d)
                if not valid: break
                d_out = min_dist_to_outer(inner_verts[i], outer_verts)
                if d_out < 1e-6:
                    valid = False
                    break
                min_gap = min(min_gap, d_out)
            
            if valid and S_val < best_S:
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
    outer_side = best_params[-2]
    outer_angle = best_params[-1]
    
    return inner, outer_center, outer_side, np.degrees(outer_angle)
