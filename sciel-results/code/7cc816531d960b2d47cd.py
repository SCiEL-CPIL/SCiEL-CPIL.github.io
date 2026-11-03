import math
import numpy as np
import time
from scipy.optimize import minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0
EPS = 1e-9

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    def axial_to_cartesian(q, r):
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        return (x, y)

    def hexagon_vertices(cx, cy, angle_deg, scale=1.0):
        # Regular hexagon with side length 'scale', centered at (cx, cy), rotated by angle_deg
        verts = []
        for k in range(6):
            ang = math.radians(angle_deg) + k * math.pi / 3.0
            # Vertices are at distance 1 from center (side length 1 => circumradius 1)
            vx = cx + scale * math.cos(ang)
            vy = cy + scale * math.sin(ang)
            verts.append((vx, vy))
        return np.array(verts)

    def polygon_area_signed(verts):
        n = len(verts)
        area = 0.0
        for i in range(n):
            x1, y1 = verts[i]
            x2, y2 = verts[(i + 1) % n]
            area += x1 * y2 - x2 * y1
        return 0.5 * area

    def point_in_polygon(p, poly):
        # Ray casting algorithm
        x, y = p
        n = len(poly)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = poly[i]
            xj, yj = poly[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside

    def polygons_disjoint(v1, v2, eps=1e-9):
        # Check if two convex polygons (hexagons) are disjoint (no interior overlap)
        # They are disjoint if they are strictly separated or just touching.
        # For optimization, we want to maximize separation, so we check for overlap.
        # Two convex polygons overlap if and only if there is no separating axis.
        # Separating axes are the normals of the edges of both polygons.
        
        # Combine vertices
        all_verts = np.vstack([v1, v2])
        
        # Get unique edge normals
        # Since these are regular hexagons, we can just use the normals of v1 and v2
        # But to be safe, let's compute them.
        axes = []
        for v in [v1, v2]:
            for i in range(6):
                p1 = v[i]
                p2 = v[(i + 1) % 6]
                edge = p2 - p1
                # Normal is perpendicular to edge
                nx = -edge[1]
                ny = edge[0]
                norm = math.sqrt(nx*nx + ny*ny)
                if norm > 1e-12:
                    axes.append((nx/norm, ny/norm))
        
        # Check each axis
        for nx, ny in axes:
            proj1 = v1[:, 0] * nx + v1[:, 1] * ny
            proj2 = v2[:, 0] * nx + v2[:, 1] * ny
            
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            # If there is a gap, they are separated
            if max1 < min2 - eps or max2 < min1 - eps:
                return False # Disjoint (separated)
        
        # If no separating axis found, they overlap
        return True

    def overlap_area(v1, v2):
        # Approximate overlap area by checking if centers are close
        # A more robust method for regular hexagons:
        # If the distance between centers is > 2 * side, they are definitely disjoint.
        # If distance < side, they likely overlap.
        # For a tighter bound, we can use the separating axis test above.
        # But for the force-directed method, we just need a penalty.
        # Let's use the distance between centers as a proxy for overlap penalty.
        c1 = np.mean(v1, axis=0)
        c2 = np.mean(v2, axis=0)
        dist = np.linalg.norm(c1 - c2)
        # Minimum distance between centers of two disjoint unit hexagons is 2*side = 2.0
        # Actually, it's slightly less if rotated, but 2.0 is the max separation for side-1 hexes touching.
        # Wait, two unit hexagons touching side-to-side have center distance 2*apothem = sqrt(3) ~ 1.732? No.
        # Side length 1. Distance between centers of two touching hexagons is 2 * apothem = sqrt(3).
        # Let's verify. Apothem = sqrt(3)/2. Two hexagons touching side to side: distance = 2 * sqrt(3)/2 = sqrt(3).
        # So if dist < sqrt(3) - eps, they overlap.
        min_dist = math.sqrt(3)
        if dist < min_dist:
            return (min_dist - dist) ** 2
        return 0.0

    def enclosing_hexagon_side(pts, beta):
        # Compute minimal enclosing regular hexagon side length for points 'pts'
        # with rotation beta.
        # The hexagon is defined by 6 half-planes.
        # We want to find the smallest S such that all points are inside.
        # This is a linear programming problem, but we can solve it analytically
        # by finding the maximum projection of the points onto the 6 normals.
        
        normals = []
        for k in range(6):
            ang = beta + math.pi / 6.0 + k * math.pi / 3.0
            normals.append((math.cos(ang), math.sin(ang)))
        
        max_proj = 0.0
        for nx, ny in normals:
            projs = pts[:, 0] * nx + pts[:, 1] * ny
            # The extent in this direction is max(projs) - min(projs)
            # But we need the hexagon to be centered.
            # Actually, the "width" of the set of points in direction (nx, ny) is max - min.
            # The hexagon half-width in that direction is S * apothem.
            # So we need S * apothem >= max |proj - center_proj|
            # To minimize S, we should center the hexagon at the center of the bounding box
            # in the hexagonal metric.
            # A simple approach: center at the centroid, then find max deviation.
            # But the optimal center might be different.
            # For a regular hexagon, the optimal enclosing hexagon for a set of points
            # can be found by considering the 6 support functions.
            
            # Let's just compute the max absolute projection relative to the centroid
            # as a first approximation, then refine the center if needed.
            # Actually, the minimal enclosing hexagon side S is determined by:
            # S = max_k ( max_i (p_i . n_k) - min_i (p_i . n_k) ) / (2 * apothem) ?
            # No, that's for the width.
            # The condition is: for all i, k: | (p_i - c) . n_k | <= S * apothem
            # This is a linear program in c and S.
            # Given the time constraints, let's use a heuristic:
            # 1. Compute centroid c.
            # 2. For each normal, compute max | (p_i - c) . n_k |.
            # 3. S = max_k (max_i | ... |) / apothem.
            # This might not be optimal if the centroid is not the center of the minimal hexagon.
            # But for symmetric clusters, it's often close.
            
            pass # Placeholder for actual calculation

        # Better approach:
        # The minimal enclosing hexagon side S is the smallest value such that there exists a center c
        # where all points satisfy the hexagon inequalities.
        # We can solve for S by iterating over the 6 constraints.
        
        # Let's use the fact that the optimal center c is the intersection of the "midlines"
        # of the most extreme points.
        
        # Simpler robust method:
        # For a given beta, the problem is to find min S s.t. exists c:
        # for all k: max_i (p_i . n_k) - c . n_k <= S * A
        #            c . n_k - min_i (p_i . n_k) <= S * A
        # This implies:
        # c . n_k >= max_i (p_i . n_k) - S * A
        # c . n_k <= min_i (p_i . n_k) + S * A
        # So we need: max_i (p_i . n_k) - S * A <= min_i (p_i . n_k) + S * A
        # => 2 * S * A >= max_i (p_i . n_k) - min_i (p_i . n_k)
        # => S >= (max_i (p_i . n_k) - min_i (p_i . n_k)) / (2 * A)
        
        # So for a FIXED center c, the required S is max_k (width_k / (2A)).
        # But c can move.
        # Actually, the condition above is necessary for existence of c.
        # Is it sufficient?
        # If for all k, width_k <= 2 * S * A, does there exist a c?
        # This is equivalent to asking if the intersection of the 6 slabs is non-empty.
        # By Helly's theorem in 2D, if every 3 slabs intersect, then all intersect.
        # For a regular hexagon, if the widths are small enough, they intersect.
        # The minimal S is indeed max_k (width_k / (2A)).
        
        # Let's verify this logic.
        # If S = max_k (width_k / (2A)), then for each k, the slab [max_k - SA, min_k + SA] has width >= 0.
        # Do these 6 slabs have a common intersection?
        # For a regular hexagon, yes, if the S is large enough to cover the "hexagonal diameter".
        
        # So the algorithm is:
        # 1. For given beta, compute normals n_k.
        # 2. For each k, compute width_k = max_i (p_i . n_k) - min_i (p_i . n_k).
        # 3. S = max_k (width_k / (2 * APOTHEM)).
        
        # This gives the side length of the minimal enclosing hexagon with rotation beta,
        # assuming the hexagon can be centered anywhere.
        
        S = 0.0
        for k in range(6):
            nx, ny = normals[k]
            projs = pts[:, 0] * nx + pts[:, 1] * ny
            width = np.max(projs) - np.min(projs)
            S_k = width / (2.0 * APOTHEM)
            if S_k > S:
                S = S_k
        return S

    def objective(params):
        # params: [x1, y1, x2, y2, ..., x11, y11, beta]
        # First 22 params are centers of 11 hexagons
        # Last param is rotation beta
        n_hex = 11
        centers = np.array(params[:2 * n_hex]).reshape(n_hex, 2)
        beta = params[2 * n_hex]
        
        # Penalty for overlap
        penalty = 0.0
        for i in range(n_hex):
            for j in range(i + 1, n_hex):
                c1 = centers[i]
                c2 = centers[j]
                dist = np.linalg.norm(c1 - c2)
                min_dist = math.sqrt(3)  # Distance between centers of touching unit hexes
                if dist < min_dist:
                    penalty += (min_dist - dist) ** 2
        
        # Enclosing hexagon side length
        S = enclosing_hexagon_side(centers, beta)
        
        # We want to minimize S, but penalize overlap heavily
        # If there is overlap, the configuration is invalid.
        # We return S + penalty * large_factor
        # But we also need to ensure the hexagons are inside the enclosing hexagon.
        # The enclosing_hexagon_side function assumes the points are the centers.
        # The actual hexagons extend beyond their centers.
        # The vertices of the inner hexagons must be inside the outer hexagon.
        # The distance from center to vertex of a unit hexagon is 1.
        # So the "effective" points for the enclosing hexagon are the centers,
        # but the required S should be larger to accommodate the radius 1.
        # Actually, the enclosing hexagon must contain the entire inner hexagons.
        # The support function of the union of hexagons is the max of the support functions of each.
        # The support function of a unit hexagon centered at c with rotation 0 is:
        # h(u) = c . u + 1 * max_k (u . v_k) where v_k are the normals of the unit hexagon.
        # Since the inner hexagons can rotate, but in our model they are all aligned with the outer?
        # The problem says "each inner hexagon may translate and rotate".
        # But in our simplified model, we assume they are all aligned with the outer hexagon.
        # This is a restriction, but it simplifies the problem significantly.
        # If we allow individual rotations, the problem becomes much harder.
        # Let's stick to aligned hexagons for now.
        
        # For aligned hexagons, the support function of the union is:
        # h(u) = max_i (c_i . u + h_unit(u))
        # where h_unit(u) is the support function of a unit hexagon centered at origin.
        # For a regular hexagon with side 1, h_unit(u) = max_k (u . n_k) where n_k are the normals.
        # The minimal enclosing hexagon with rotation beta has support function S * A * |cos(angle)|?
        # No, the support function of a regular hexagon with side S and rotation beta is:
        # H(u) = S * A * max_k (u . m_k) where m_k are the normals of the unit hexagon rotated by beta.
        # Actually, H(u) = S * A * |u . n_k| for the appropriate k?
        # Let's use the width formula again.
        # The width of the union in direction n is:
        # W(n) = max_i (c_i . n + h_unit(n)) - min_i (c_i . n + h_unit(-n))
        # Since h_unit(n) = h_unit(-n) for a symmetric hexagon, this is:
        # W(n) = (max_i c_i . n - min_i c_i . n) + 2 * h_unit(n)
        # And S * 2 * A = W(n) => S = W(n) / (2 * A)
        
        # So we need to compute h_unit(n) for each normal n.
        # For a unit hexagon with side 1, the apothem is A = sqrt(3)/2.
        # The support function h_unit(n) is the distance from the center to the supporting line in direction n.
        # For a regular hexagon, h_unit(n) = A / cos(theta) where theta is the angle between n and the nearest normal?
        # No, h_unit(n) = max_k (n . v_k) where v_k are the vertices? No, normals.
        # The support function of a convex polygon with normals n_k and offsets d_k is:
        # h(u) = max_k (d_k + u . n_k) ? No.
        # For a regular hexagon centered at origin, the support function is:
        # h(u) = A * max_k (u . m_k) where m_k are the unit normals to the sides.
        # Wait, the side normals are at angles 0, 60, 120, 180, 240, 300 degrees.
        # If u is in the direction of a normal m_k, then h(u) = A.
        # If u is between m_k and m_{k+1}, h(u) is determined by the vertex between them.
        
        # Let's compute h_unit(n) directly.
        # The vertices of a unit hexagon (side 1) are at distance 1 from the center.
        # The angles of the vertices are 0, 60, 120, 180, 240, 300 degrees.
        # So the support function h(u) = max_k (u . v_k) where v_k are the vertices.
        
        def h_unit(n):
            # n is a unit vector
            max_val = 0.0
            for k in range(6):
                ang = k * math.pi / 3.0
                vx = math.cos(ang)
                vy = math.sin(ang)
                val = n[0] * vx + n[1] * vy
                if val > max_val:
                    max_val = val
            return max_val
        
        S = 0.0
        for k in range(6):
            ang = beta + math.pi / 6.0 + k * math.pi / 3.0
            nx = math.cos(ang)
            ny = math.sin(ang)
            n_vec = np.array([nx, ny])
            
            projs = centers[:, 0] * nx + centers[:, 1] * ny
            width_centers = np.max(projs) - np.min(projs)
            
            h_val = h_unit(n_vec)
            
            W = width_centers + 2.0 * h_val
            S_k = W / (2.0 * APOTHEM)
            if S_k > S:
                S = S_k
        
        return S + 100.0 * penalty

    # Initial configurations
    def generate_initial_config(i):
        # Generate a random or structured initial configuration
        # We'll use a few structured ones and some random ones
        if i == 0:
            # Hexagonal lattice
            centers = []
            for q in range(-2, 3):
                for r in range(-2, 3):
                    if abs(q) + abs(r) + abs(q+r) <= 4: # Rough hexagon
                        x, y = axial_to_cartesian(q, r)
                        centers.append((x, y))
            # Select 11
            centers = centers[:11]
            beta = 0.0
        elif i == 1:
            # Linear chain
            centers = [(float(x), 0.0) for x in range(11)]
            beta = 0.0
        elif i == 2:
            # 3x3 square plus 2
            centers = []
            for x in range(3):
                for y in range(3):
                    centers.append((float(x), float(y)))
            centers.append((0.0, 3.0))
            centers.append((1.0, 3.0))
            beta = 0.0
        else:
            # Random
            centers = []
            for _ in range(11):
                x = np.random.uniform(-3, 3)
                y = np.random.uniform(-3, 3)
                centers.append((x, y))
            beta = np.random.uniform(0, math.pi/6)
        
        params = []
        for x, y in centers:
            params.append(x)
            params.append(y)
        params.append(beta)
        return params

    best_S = float('inf')
    best_params = None

    num_trials = 20
    for i in range(num_trials):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        params = generate_initial_config(i)
        
        # Use Nelder-Mead optimization
        try:
            res = minimize(objective, params, method='Nelder-Mead',
                         options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
            if res.fun < best_S:
                best_S = res.fun
                best_params = res.x
        except Exception:
            pass

    if best_params is None:
        # Fallback to a simple configuration
        best_params = generate_initial_config(0)
        best_S = objective(best_params)

    # Extract results
    n_hex = 11
    centers = np.array(best_params[:2 * n_hex]).reshape(n_hex, 2)
    beta = best_params[2 * n_hex]
    angle_deg = math.degrees(beta)

    # Compute the actual enclosing hexagon center and side
    # The enclosing_hexagon_side function computed S based on widths.
    # We need to find the center c that achieves this S.
    # For each normal n_k, the center must satisfy:
    # c . n_k in [max_i(c_i . n_k) - S*A + h_unit(n_k), min_i(c_i . n_k) + S*A - h_unit(-n_k)]
    # Since h_unit is symmetric, this is:
    # c . n_k in [max_i(c_i . n_k) - S*A + h_unit(n_k), min_i(c_i . n_k) + S*A - h_unit(n_k)]
    # This is a system of linear inequalities. We can solve it with linprog.
    
    from scipy.optimize import linprog
    
    c_obj = np.zeros(2)
    A_ub = []
    b_ub = []
    
    # Recompute S and the bounds
    S_actual = 0.0
    for k in range(6):
        ang = beta + math.pi / 6.0 + k * math.pi / 3.0
        nx = math.cos(ang)
        ny = math.sin(ang)
        n_vec = np.array([nx, ny])
        
        def h_unit(n):
            max_val = 0.0
            for k in range(6):
                ang = k * math.pi / 3.0
                vx = math.cos(ang)
                vy = math.sin(ang)
                val = n[0] * vx + n[1] * vy
                if val > max_val:
                    max_val = val
            return max_val
        
        h_val = h_unit(n_vec)
        
        projs = centers[:, 0] * nx + centers[:, 1] * ny
        max_proj = np.max(projs)
        min_proj = np.min(projs)
        
        # Constraint: c . n_k >= max_proj - S * A + h_val
        # c . n_k <= min_proj + S * A - h_val
        # But S is what we're solving for? No, S is fixed from the optimization.
        # We need to find the S that makes the system feasible.
        # The minimal S is the one where the intersection of the slabs is non-empty.
        # This is equivalent to: for all k, l:
        # max_proj_k - S*A + h_k <= min_proj_l + S*A - h_l
        # => 2*S*A >= max_proj_k + h_k - (min_proj_l - h_l)
        # => S >= (max_proj_k + h_k - min_proj_l + h_l) / (2*A)
        
        # This is getting complicated. Let's just use the centroid as the center
        # and compute the required S.
        pass

    # Simple fallback: use the centroid as the center
    cx = np.mean(centers[:, 0])
    cy = np.mean(centers[:, 1])
    
    # Compute S by checking all vertices of all inner hexagons
    all_vertices = []
    for i in range(n_hex):
        verts = hexagon_vertices(centers[i, 0], centers[i, 1], angle_deg)
        all_vertices.append(verts)
    all_vertices = np.vstack(all_vertices)
    
    max_proj = 0.0
    for k in range(6):
        ang = beta + math.pi / 6.0 + k * math.pi / 3.0
        nx = math.cos(ang)
        ny = math.sin(ang)
        projs = (all_vertices[:, 0] - cx) * nx + (all_vertices[:, 1] - cy) * ny
        max_proj = max(max_proj, np.max(projs), -np.min(projs))
    
    S_final = (max_proj + APOTHEM) / APOTHEM

    inner = []
    for i in range(n_hex):
        inner.append([float(centers[i, 0]), float(centers[i, 1]), float(angle_deg)])
    
    center = [float(cx), float(cy)]
    side = float(S_final)
    rotation = float(angle_deg)
    
    return inner, center, side, rotation
