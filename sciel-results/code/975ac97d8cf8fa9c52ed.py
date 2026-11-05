"""Program for packing 11 unit regular hexagons inside a minimal regular hexagon."""
import numpy as np
from scipy.optimize import minimize_scalar

N = 11

# EVOLVE-BLOCK-START
def _hex_vertices(cx, cy, side, rot_deg):
    """Return vertices of a regular hexagon with center (cx,cy), side length, rotation in degrees."""
    # A regular hexagon with 'flat' top (pointy left/right) has vertices at angles 0, 60, 120, 180, 240, 300
    # Standard: vertices at angles 0, 60, ... relative to center
    # For a hexagon with side length s, the circumradius is s
    # We use the convention where angle 0 points to the right
    angles = np.deg2rad([0, 60, 120, 180, 240, 300] + [rot_deg])
    r = side
    return np.column_stack([cx + r * np.cos(angles), cy + r * np.sin(angles)])

def _hex_vertices_unit(cx, cy, rot_deg):
    """Return vertices of a unit regular hexagon (side=1) at center (cx,cy) with rotation rot_deg."""
    return _hex_vertices(cx, cy, 1.0, rot_deg)

def _convex_hull(points):
    """Compute convex hull of a set of points. Returns list of hull vertices in CCW order."""
    points = np.array(points)
    if len(points) < 3:
        return points
    
    # Andrew's monotone chain
    points = np.unique(points, axis=0)
    points = points[np.lexsort((points[:, 1], points[:, 0]))]
    
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    
    upper = []
    for p in points[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    
    hull = np.vstack([lower[:-1], upper[:-1]])
    return hull

def _min_enclosing_hexagon_side(hull_points, outer_rot_deg):
    """
    Given hull points and the rotation of the outer hexagon, compute the minimal side length
    of a regular hexagon (with that rotation) that encloses all hull points.
    
    The outer hexagon is centered optimally. We compute the minimal side length such that
    there exists a center where all points are inside.
    
    For a regular hexagon with rotation theta, the support function in direction phi is:
    h(phi) = s * max_k cos(phi - theta - k*pi/3) for the 6 sides... 
    
    Actually, a regular hexagon with side s and rotation theta can be described as the intersection
    of 6 half-planes. The distance from center to each side is s * sqrt(3)/2 (the apothem).
    The normal directions of the 6 sides are at angles theta + k*pi/3 for k=0..5.
    
    For a point p to be inside the hexagon centered at c:
    dot(p - c, n_k) <= apothem for all k, where n_k are unit normals.
    
    Equivalently: for all k, dot(p, n_k) - dot(c, n_k) <= apothem
    => dot(c, n_k) >= dot(p, n_k) - apothem for all k
    
    Also, by symmetry of the hexagon, we also need:
    dot(p - c, -n_k) <= apothem => dot(c, -n_k) >= dot(p, -n_k) - apothem
    => -dot(c, n_k) >= -dot(p, n_k) - apothem
    => dot(c, n_k) <= dot(p, n_k) + apothem
    
    So for each normal direction n_k:
    max_p dot(p, n_k) - apothem <= dot(c, n_k) <= min_p dot(p, n_k) + apothem
    
    This must hold for all 6 directions. The question is whether there exists a c satisfying
    all these constraints.
    
    Let a_k = max_p dot(p, n_k), b_k = min_p dot(p, n_k).
    We need: a_k - s*sqrt(3)/2 <= dot(c, n_k) <= b_k + s*sqrt(3)/2 for all k.
    
    Since the n_k span the plane (they are 6 directions 60 deg apart), this is a feasibility
    problem in 2D. The minimal s is when the feasible region for c just becomes non-empty.
    
    Alternative approach: The width of the point set in direction n is w(n) = max_p dot(p,n) - min_p dot(p,n).
    For the hexagon to enclose the points, we need the "width" in each of the 6 normal directions
    to be at most 2*apothem = s*sqrt(3).
    
    Wait, that's not quite right because the center can shift. Let me think again.
    
    Actually, the condition is: there exists c such that for all k:
    dot(c, n_k) >= a_k - apothem
    dot(c, n_k) <= b_k + apothem
    
    This is a system of linear inequalities in c. The minimal apothem (and hence s) is the
    smallest value for which this system is feasible.
    
    Since we have 6 pairs of inequalities (12 total) in 2 variables, we can find the minimal
    apothem via linear programming or by checking critical values.
    
    A simpler approach: the minimal enclosing hexagon with fixed orientation can be found by
    noting that the optimal center is determined by the most constraining pairs.
    
    Let's use a different formulation. The support function of the point set in direction n_k
    is h_k = max_p dot(p, n_k), and the anti-support is l_k = min_p dot(p, n_k).
    
    For the hexagon centered at c with apothem a:
    We need h_k - dot(c, n_k) <= a for all k
    And dot(c, n_k) - l_k <= a for all k
    
    So: dot(c, n_k) >= h_k - a and dot(c, n_k) <= l_k + a for all k.
    
    This means: max_k (h_k - a) <= dot(c, n_k) <= min_k (l_k + a) ... no, that's not right either.
    
    For each k independently: h_k - a <= dot(c, n_k) <= l_k + a.
    
    Note that h_k >= l_k always. So we need l_k + a >= h_k - a, i.e., 2a >= h_k - l_k = w_k.
    So a >= w_k / 2 for all k. But this is necessary but not sufficient because c must satisfy
    all constraints simultaneously.
    
    Let me just use scipy's linprog or a simple 2D feasibility check.
    """
    n_k = np.deg2rad([0, 60, 120, 180, 240, 300] + [outer_rot_deg])
    normals = np.column_stack([np.cos(n_k), np.sin(n_k)])
    
    hull_points = np.array(hull_points)
    # h_k = max dot(p, n_k), l_k = min dot(p, n_k)
    dots = hull_points @ normals.T  # shape (n_points, 6)
    h = dots.max(axis=0)
    l = dots.min(axis=0)
    
    # We need to find minimal a (apothem) such that there exists c with:
    # h[k] - a <= dot(c, normals[k]) <= l[k] + a for all k
    
    # This is: for each k, dot(c, normals[k]) - h[k] <= -a
    # and: -dot(c, normals[k]) + l[k] <= -a
    
    # Let's binary search on a, or use the fact that this is a 2D LP.
    # Actually, let's just do a 1D search on a since the problem is convex in a.
    
    def feasible(a):
        # Check if there exists c such that:
        # dot(c, n_k) >= h_k - a for all k
        # dot(c, n_k) <= l_k + a for all k
        
        # This is a system of 12 linear inequalities in 2 variables.
        # We can check feasibility by trying to find a point.
        # Since it's 2D, we can use scipy linprog or just check if the intersection is non-empty.
        
        # Simple approach: the feasible region is an intersection of half-planes.
        # We can find it by checking if the minimum of a dummy objective over the feasible region is finite.
        
        # Use a simple approach: try to find c by solving the LP:
        # minimize 0
        # subject to: normals[k] @ c >= h[k] - a for all k
        #              -normals[k] @ c >= -(l[k] + a) for all k
        
        from scipy.optimize import linprog
        A_ub = []
        b_ub = []
        for k in range(6):
            A_ub.append(-normals[k])
            b_ub.append(-(h[k] - a))
            A_ub.append(normals[k])
            b_ub.append(l[k] + a)
        
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        
        res = linprog(c=np.zeros(2), A_ub=A_ub, b_ub=b_ub, bounds=[(None, None), (None, None)])
        return res.status == 0
    
    # Binary search on a
    lo = 0.0
    hi = 100.0
    for _ in range(50):
        mid = (lo + hi) / 2
        if feasible(mid):
            hi = mid
        else:
            lo = mid
    
    apothem = hi
    side = apothem * 2 / np.sqrt(3)
    return side

def construct():
    # Place 11 unit hexagons in a linear chain
    # Each hexagon has side 1, so the distance between centers of adjacent hexagons
    # in a chain depends on orientation.
    
    # For simplicity, let's place them in a row with 0-degree rotation.
    # A unit hexagon with rotation 0 has vertices at angles 0,60,120,180,240,300.
    # The width (in x-direction) is 2 (from -1 to 1 in x, since circumradius = 1).
    # To pack them side by side without overlap, we need to space them appropriately.
    
    # Actually, let's just place them in a line with spacing that avoids overlap.
    # For two unit hexagons with the same orientation, the minimum center distance
    # to avoid overlap is 2 (diameter).
    
    # Let's place centers at x = 2*k, y = 0 for k = -5, -4, ..., 4 (11 hexagons)
    inner = []
    for k in range(-5, 6):
        inner.append([2.0 * k, 0.0, 0.0])
    
    # Compute the convex hull of all vertices of all hexagons
    all_points = []
    for x, y, rot in inner:
        verts = _hex_vertices_unit(x, y, rot)
        all_points.extend(verts.tolist())
    
    hull = _convex_hull(all_points)
    
    # Now minimize the outer hexagon side length over rotation angle
    def objective(rot_deg):
        return _min_enclosing_hexagon_side(hull, rot_deg)
    
    # Minimize over rotation angle
    res = minimize_scalar(objective, bounds=(0, 60), method='bounded', options={'xatol': 1e-8})
    best_rot = res.x
    best_side = res.fun
    
    # The outer hexagon center can be optimized, but for the score we just need the side length.
    # The center doesn't affect the score, so we can set it to (0,0).
    return inner, (0.0, 0.0), best_side, best_rot

# EVOLVE-BLOCK-END
