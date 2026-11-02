"""
Task: Place N=11 disjoint unit regular hexagons inside a regular hexagon, minimizing outer side length.
"""
import numpy as np
from scipy.optimize import minimize
import time

N = 11

def _hex_vertices(cx, cy, angle_deg, size=1.0):
    """Return 6 vertices of a regular hexagon with side length `size`, center (cx,cy), rotated by angle_deg."""
    a = np.radians(angle_deg)
    # Vertices at angles 0, 60, ..., 300 degrees from center, distance = size
    theta = np.deg2rad(np.arange(0, 360, 60) + angle_deg)
    return np.column_stack([cx + size * np.cos(theta), cy + size * np.sin(theta)])

def _point_hex_distance(px, py, hx, hy, angle_deg):
    """
    Minimum distance from point (px,py) to the hexagon centered at (hx,hy) with rotation angle_deg, side 1.
    We compute the signed distance: positive if outside, negative if inside.
    For disjointness, we need this distance >= 0 for points on boundary of other hexes? No.
    Actually, we need the two hexagons to be disjoint. A robust way is to check all 6 vertices of each hex
    against the other hex. But that's expensive for optimization.
    
    Alternative: use the fact that hexagons are convex. The distance between two convex polygons
    is the minimum distance between any edge of one to any edge of the other.
    
    For simplicity and speed in optimization, let's use a different approach:
    We can model the non-overlap constraint using the support function or by checking
    if any vertex of hex1 is inside hex2, and vice versa, AND if edges intersect.
    
    However, a simpler and effective method for this specific problem is to use the
    "separating axis" concept, but that's complex.
    
    Let's use a simpler heuristic: the distance between centers must be at least 2.0
    (since two unit hexagons side-by-side have centers 2*sqrt(3)/2 * something? No.)
    Wait, two regular hexagons of side 1 touching face-to-face have center distance = 2 * (apothem) = 2 * (sqrt(3)/2) = sqrt(3) ≈ 1.732.
    But they can touch at vertices or edges. The minimum distance between centers for non-overlapping
    hexagons depends on relative orientation.
    
    A safe and easy constraint to use in optimization is:
    For each pair, the minimum distance between their boundaries must be >= 0.
    We can approximate this by checking the 6 vertices of hex i against hex j, and vice versa.
    If any vertex of hex i is inside hex j, or any vertex of hex j is inside hex i, they overlap.
    Also, edges might cross without vertices being inside (for convex shapes, if no vertex is inside the other,
    and they are not nested, they don't intersect? For convex polygons, if no vertex of A is inside B and
    no vertex of B is inside A, they are disjoint. This is true for convex sets.)
    
    So, constraint: no vertex of any hexagon is inside any other hexagon.
    
    Point-in-convex-polygon test:
    For a convex polygon defined by vertices V in CCW order, a point P is inside if
    P is on the same side (left) of every edge (V[i], V[i+1]).
    
    We will compute a "penetration depth" metric: for each pair, compute the maximum of the
    signed distances of vertices of one to the edges of the other? No, simpler:
    For each pair (i,j), compute d_ij = min over all vertices v of hex_i of (point-in-hex_j signed distance)
    and d_ji = min over all vertices v of hex_j of (point-in-hex_i signed distance).
    Actually, if a vertex of i is inside j, that's a violation.
    
    Let's define a function _is_inside_hex(px, py, hx, hy, angle_deg) that returns True/False or a signed distance.
    Signed distance: we can compute the minimum of the signed distances to the 6 edges.
    For a convex polygon, the signed distance to the polygon is the min of the signed distances to its edges
    (where negative means inside).
    
    So, for a pair (i,j), we compute:
    min_dist = min( min_vertex_i (signed_dist_to_hex_j), min_vertex_j (signed_dist_to_hex_i) )
    We need min_dist >= 0.
    
    This is a smooth-ish function (min of linear functions), but non-smooth.
    For optimization, we can use a penalty method.
    """
    # This function is called many times, so it needs to be fast.
    # Precompute vertices? No, they depend on hx, hy, angle.
    
    # For now, let's just return 1.0 as a placeholder, and implement properly in the constraint function.
    pass

def _signed_dist_to_hex(px, py, hx, hy, angle_deg):
    """
    Returns the signed distance from point (px,py) to the hexagon.
    Positive if outside, negative if inside.
    """
    verts = _hex_vertices(hx, hy, angle_deg, 1.0)
    min_dist = np.inf
    for i in range(6):
        x1, y1 = verts[i]
        x2, y2 = verts[(i+1) % 6]
        # Edge vector
        ex = x2 - x1
        ey = y2 - y1
        # Normal vector (inward)
        # For CCW polygon, the inward normal is (ey, -ex) normalized?
        # Let's check: if edge is (1,0) to (0,1), vector is (-1,1). Inward normal should point to center.
        # Center is (0,0) relative to this edge? No, this is generic.
        # Let's use the cross product to determine the side.
        # Vector from x1 to p: (px-x1, py-y1)
        # Cross product of edge (ex,ey) and vector to p: ex*(py-y1) - ey*(px-x1)
        # For CCW polygon, points inside are to the left of the edge, so cross product > 0?
        # Let's verify with a square: (0,0), (1,0), (1,1), (0,1).
        # Edge (0,0)->(1,0): e=(1,0). Point (0.5, 0.5). v=(0.5, 0.5). cross = 1*0.5 - 0*0.5 = 0.5 > 0. Inside.
        # Edge (1,0)->(1,1): e=(0,1). Point (0.5, 0.5). v=(-0.5, 0.5). cross = 0*0.5 - 1*(-0.5) = 0.5 > 0. Inside.
        # So, cross > 0 means inside relative to that edge.
        cross = ex * (py - y1) - ey * (px - x1)
        edge_len = np.sqrt(ex*ex + ey*ey)
        # Distance to the line
        dist_line = cross / edge_len
        # If cross <= 0, the point is outside (or on boundary) relative to this edge.
        # The signed distance to the polygon is the minimum of these distances.
        # But if cross < 0, the distance is dist_line (which is negative).
        # If cross > 0, it's a positive distance, but we need to check if the projection is within the edge segment.
        # Actually, for a convex polygon, the minimum distance to the polygon is the minimum of:
        # 1. The distances to the edges (if the projection is within the segment).
        # 2. The distances to the vertices (if the projection is outside).
        # However, a simpler approximation: just use the distance to the lines, but only consider the "active" edges.
        # For our purpose (penalty), we can use: if cross < 0, dist = cross/edge_len (negative).
        # If cross >= 0, we need to check if the closest point on the segment is the projection.
        # Let's compute the closest point on the segment.
        t = ((px - x1) * ex + (py - y1) * ey) / (ex*ex + ey*ey)
        if t < 0:
            closest_x, closest_y = x1, y1
        elif t > 1:
            closest_x, closest_y = x2, y2
        else:
            closest_x = x1 + t * ex
            closest_y = y1 + t * ey
        d = np.sqrt((px - closest_x)**2 + (py - closest_y)**2)
        if cross < 0:
            signed_d = -d
        else:
            signed_d = d
        if signed_d < min_dist:
            min_dist = signed_d
    return min_dist

def _non_overlap_penalty(params, n=N):
    """
    params: flat array of length 3*n: [x1,y1,ang1, x2,y2,ang2, ...]
    Returns a penalty value: sum over pairs of max(0, -min_dist_ij)^2
    """
    penalty = 0.0
    for i in range(n):
        x1, y1, a1 = params[3*i], params[3*i+1], params[3*i+2]
        verts1 = _hex_vertices(x1, y1, a1, 1.0)
        for j in range(i+1, n):
            x2, y2, a2 = params[3*j], params[3*j+1], params[3*j+2]
            verts2 = _hex_vertices(x2, y2, a2, 1.0)
            min_dist = np.inf
            # Check vertices of hex1 against hex2
            for vx, vy in verts1:
                d = _signed_dist_to_hex(vx, vy, x2, y2, a2)
                if d < min_dist:
                    min_dist = d
            # Check vertices of hex2 against hex1
            for vx, vy in verts2:
                d = _signed_dist_to_hex(vx, vy, x1, y1, a1)
                if d < min_dist:
                    min_dist = d
            if min_dist < 0:
                penalty += (-min_dist)**2
    return penalty

def _outer_hexagon_side_length(params, n=N):
    """
    Given the positions of the inner hexagons, find the minimum side length of a regular hexagon
    that contains all of them. The outer hexagon can be translated and rotated.
    
    For a fixed rotation alpha of the outer hexagon, the minimal side length L is determined by
    the maximum extent of the inner hexagons in the 6 directions of the outer hexagon.
    
    A regular hexagon with side L, center (Cx, Cy), rotation alpha, contains a point (x,y) if and only if
    for all k=0..5: dot((x-Cx, y-Cy), n_k) <= L * apothem_outer?
    Actually, the support function of a hexagon in direction n is L * apothem?
    No. The distance from center to a side is L * sqrt(3)/2.
    The distance from center to a vertex is L.
    
    Let's use the following:
    A regular hexagon with side L, rotation alpha, center C, is the set of points P such that
    for all k=0..5: dot(P-C, u_k) <= L * sqrt(3)/2, where u_k are unit normals to the sides.
    The normals are at angles alpha + k*60.
    
    So, for a given alpha and C, the condition for point P is:
    dot(P-C, u_k) <= L * sqrt(3)/2 for all k.
    => L >= max_k ( dot(P-C, u_k) ) / (sqrt(3)/2)
    
    To minimize L, we can choose C.
    For a fixed alpha, the problem becomes:
    min_C max_{P in union of hexes, k=0..5} ( dot(P-C, u_k) ) / (sqrt(3)/2)
    
    Let h_k(C) = max_P dot(P-C, u_k) = max_P (dot(P, u_k) - dot(C, u_k)) = (max_P dot(P, u_k)) - dot(C, u_k)
    Let M_k = max_P dot(P, u_k). This is the support function of the union in direction u_k.
    We want to minimize L = max_k ( M_k - dot(C, u_k) ) / (sqrt(3)/2)
    subject to C being the center.
    
    This is a convex optimization problem in C for fixed alpha.
    We can solve it by linear programming or analytically.
    Alternatively, we can note that the optimal C is the center of the bounding box in the dual space?
    
    Actually, a simpler approach: for a fixed alpha, the minimal L is the minimal radius of a hexagon
    that covers the set. This is equivalent to finding the minimal L such that the intersection of the 6 half-planes
    (defined by the hexagon) contains the set.
    
    Let's iterate over alpha from 0 to 60 degrees in small steps.
    For each alpha, we compute M_k for k=0..5 (support function of the union of inner hexes in direction u_k).
    Then we need to find C that minimizes max_k ( M_k - dot(C, u_k) ).
    Let g(C) = max_k ( M_k - dot(C, u_k) ). We want to minimize g(C).
    This is a convex piecewise-linear function. The minimum occurs where the gradient is zero or on a kink.
    
    A known result: the optimal C is the center of the minimal enclosing hexagon.
    We can solve this by observing that the conditions M_k - dot(C, u_k) <= L for all k
    imply that C is in the intersection of the half-planes dot(C, u_k) >= M_k - L.
    We want the smallest L such that this intersection is non-empty.
    
    This can be solved by binary search on L, or by linear programming.
    Since we are doing this for each alpha, and we have many alphas, let's use a simpler heuristic:
    Assume the optimal C is the centroid of the inner hexagons' centers. Then compute L.
    This might not be optimal, but it's a good approximation.
    
    Even simpler: for each alpha, compute the bounding box of the vertices in the rotated coordinate system.
    The side length L can be estimated from the extents.
    
    Let's use the following:
    For a fixed alpha, transform all vertices of all inner hexagons to the frame where the outer hexagon is aligned.
    In this frame, the outer hexagon (centered at origin for now) has inequalities:
    |x| <= L * sqrt(3)/2
    |y| <= L
    |x/2 + y*sqrt(3)/2| <= L * sqrt(3)/2  (This is for the other directions? No.)
    
    Actually, the 6 inequalities for a hexagon with side L, flat top (alpha=0), center (0,0):
    -L*sqrt(3)/2 <= x <= L*sqrt(3)/2
    -L <= y <= L
    -L*sqrt(3)/2 <= -x/2 + y*sqrt(3)/2 <= L*sqrt(3)/2  (Wait, need to derive properly)
    
    Let's stick to the support function method.
    For a given alpha, compute u_k = (cos(alpha + k*60), sin(alpha + k*60)) for k=0..5.
    Compute M_k = max over all vertices v of all inner hexagons of dot(v, u_k).
    Now, we want to find C = (Cx, Cy) and L such that:
    M_k - (Cx * u_kx + Cy * u_ky) <= L * sqrt(3)/2 for all k.
    => L >= (M_k - C dot u_k) / (sqrt(3)/2)
    We want to minimize L.
    
    Let's try C = 0 first. Then L_0 = max_k M_k / (sqrt(3)/2).
    But we can do better by shifting C.
    Note that M_k is the support of the set in direction u_k.
    The minimal enclosing hexagon of side L centered at C satisfies:
    The set is contained in the hexagon.
    
    A known formula for the minimal enclosing regular hexagon:
    It's the solution to a linear program.
    Variables: Cx, Cy, L.
    Minimize L.
    Subject to: Cx * u_kx + Cy * u_ky + L * sqrt(3)/2 >= M_k for k=0..5.
              Cx * u_kx + Cy * u_ky - L * sqrt(3)/2 >= m_k for k=0..5. (where m_k is the support in -u_k)
    
    Wait, the hexagon is defined by |dot(P-C, u_k)| <= L * sqrt(3)/2 for k=0..2?
    No, a hexagon has 6 sides, so 6 inequalities.
    The normals are u_0, u_1, ..., u_5.
    The inequalities are dot(P-C, u_k) <= L * sqrt(3)/2 for k=0..5.
    This is sufficient because the hexagon is the intersection of these 6 half-planes.
    
    So the LP is:
    Minimize L
    Subject to:
      Cx * u_kx + Cy * u_ky + L * sqrt(3)/2 >= M_k  for k=0..5
      L >= 0
    
    This is a linear program in 3 variables (Cx, Cy, L) with 5 constraints (L>=0 and the 6? No, 6 constraints).
    We can solve this LP for each alpha.
    
    Since we don't want to add an LP solver, let's use scipy.optimize.linprog.
    """
    best_L = np.inf
    # We only need to check alpha in [0, 60) due to symmetry.
    for alpha_deg in np.linspace(0, 60, 13): # 0, 5, 10, ..., 60
        alpha = np.radians(alpha_deg)
        u = np.array([np.array([np.cos(alpha + k * np.pi/3), np.sin(alpha + k * np.pi/3)]) for k in range(6)])
        
        # Compute M_k
        M = np.zeros(6)
        for i in range(n):
            x, y, a = params[3*i], params[3*i+1], params[3*i+2]
            verts = _hex_vertices(x, y, a, 1.0)
            for k in range(6):
                dots = verts @ u[k]
                M[k] = max(M[k], np.max(dots))
        
        # LP: minimize L
        # Variables: x = [Cx, Cy, L]
        # Constraints: A_ub @ x <= b_ub
        # A_eq @ x = b_eq
        # 
        # Cx * u_kx + Cy * u_ky + L * sqrt(3)/2 >= M_k
        # => -Cx * u_kx - Cy * u_ky - L * sqrt(3)/2 <= -M_k
        # L >= 0 => -L <= 0
        
        A_ub = np.zeros((7, 3))
        b_ub = np.zeros(7)
        for k in range(6):
            A_ub[k, 0] = -u[k, 0]
            A_ub[k, 1] = -u[k, 1]
            A_ub[k, 2] = -np.sqrt(3)/2
            b_ub[k] = -M[k]
        A_ub[6, 2] = -1.0
        b_ub[6] = 0.0
        
        c = np.array([0.0, 0.0, 1.0])
        
        try:
            res = minimize(lambda x: x[2], x0=[0,0,10], method='SLSQP', bounds=[(None,None), (None,None), (0, None)], constraints=[{'type': 'ineq', 'fun': lambda x: A_ub @ (-x) + b_ub}])
            # This is not right. Let's use linprog.
            from scipy.optimize import linprog
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(None, None), (None, None), (0, None)], method='highs')
            if res.success:
                L = res.x[2]
                if L < best_L:
                    best_L = L
                    best_alpha = alpha_deg
        except:
            pass
    return best_L

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Initial guess: honeycomb cluster
    # Center at (0,0)
    # 6 neighbors at distance sqrt(3) in directions 0, 60, 120, 180, 240, 300
    # 4 more further out.
    
    s3 = np.sqrt(3)
    init_params = []
    # Center
    init_params.append([0.0, 0.0, 0.0])
    # 6 neighbors
    for k in range(6):
        a = np.radians(k * 60)
        x = s3 * np.cos(a)
        y = s3 * np.sin(a)
        init_params.append([x, y, 0.0])
    # 4 more, e.g., in directions 0, 90, 180, 270 but further?
    # Let's place them at distance 2*s3 in directions 0, 120, 240? No, that's 3.
    # Let's place them at (2*s3, 0), (-s3, 3), (-s3, -3), (s3, 3)?
    # Actually, let's just use a random perturbation and let the optimizer work.
    # For now, add 4 more at arbitrary positions, say (3, 0), (0, 3), (-3, 0), (0, -3)
    init_params.append([3.0, 0.0, 0.0])
    init_params.append([0.0, 3.0, 0.0])
    init_params.append([-3.0, 0.0, 0.0])
    init_params.append([0.0, -3.0, 0.0])
    
    x0 = np.array([p for pt in init_params for p in pt])
    
    # Define the objective: outer side length + penalty for non-overlap
    def objective(x):
        if time.time() > deadline:
            return np.inf
        penalty = _non_overlap_penalty(x)
        L = _outer_hexagon_side_length(x)
        return L + 10.0 * penalty
    
    # Optimize
    best_x = x0
    best_val = objective(x0)
    
    # Use a local optimization method
    # We can't easily define constraints for SLSQP without gradients.
    # Let's just use the penalty method with Nelder-Mead or Powell.
    # Powell is derivative-free and robust.
    
    try:
        res = minimize(objective, x0, method='Powell', options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-6})
        if res.success and res.fun < best_val:
            best_x = res.x
            best_val = res.fun
    except Exception as e:
        pass
    
    # Check time
    if time.time() > deadline - 10:
        # Return current best
        pass
    
    # Extract positions
    inner = []
    for i in range(N):
        x, y, a = best_x[3*i], best_x[3*i+1], best_x[3*i+2]
        inner.append([x, y, np.degrees(a) % 360])
    
    # Find the optimal outer hexagon center and rotation
    # We need to return the outer hexagon center, side length, and rotation.
    # The side length is best_val (approximately, ignoring penalty).
    # The rotation is the one that gave the best L.
    # The center C is the solution of the LP for that alpha.
    
    # Re-compute the best alpha and C
    best_L = np.inf
    best_alpha = 0
    best_C = (0.0, 0.0)
    
    for alpha_deg in np.linspace(0, 60, 13):
        alpha = np.radians(alpha_deg)
        u = np.array([np.array([np.cos(alpha + k * np.pi/3), np.sin(alpha + k * np.pi/3)]) for k in range(6)])
        
        M = np.zeros(6)
        for i in range(N):
            x, y, a = best_x[3*i], best_x[3*i+1], best_x[3*i+2]
            verts = _hex_vertices(x, y, a, 1.0)
            for k in range(6):
                dots = verts @ u[k]
                M[k] = max(M[k], np.max(dots))
        
        A_ub = np.zeros((7, 3))
        b_ub = np.zeros(7)
        for k in range(6):
            A_ub[k, 0] = -u[k, 0]
            A_ub[k, 1] = -u[k, 1]
            A_ub[k, 2] = -np.sqrt(3)/2
            b_ub[k] = -M[k]
        A_ub[6, 2] = -1.0
        b_ub[6] = 0.0
        
        c = np.array([0.0, 0.0, 1.0])
        
        try:
            from scipy.optimize import linprog
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(None, None), (None, None), (0, None)], method='highs')
            if res.success:
                L = res.x[2]
                if L < best_L:
                    best_L = L
                    best_alpha = alpha_deg
                    best_C = (res.x[0], res.x[1])
        except:
            pass
    
    # If no LP solution, use fallback
    if best_L == np.inf:
        best_L = 5.0
        best_C = (0.0, 0.0)
        best_alpha = 0.0
    
    return inner, best_C, best_L, best_alpha

if __name__ == "__main__":
    inner, center, side, angle = construct()
    print(f"Side: {side}, Angle: {angle}, Center: {center}")
    for h in inner:
        print(h)
