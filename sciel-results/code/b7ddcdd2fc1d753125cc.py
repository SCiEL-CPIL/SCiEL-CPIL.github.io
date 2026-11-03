"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return pts / norms

def _spherical_circumradius(a, b, c):
    """
    Compute the angular radius of the smallest spherical cap on the unit sphere
    that contains a, b, c.
    This is the angular distance from the spherical circumcenter to any of the vertices.
    """
    # Check for degeneracy
    if np.linalg.norm(a) < 1e-12 or np.linalg.norm(b) < 1e-12 or np.linalg.norm(c) < 1e-12:
        return np.pi / 2.0
    
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    c = c / np.linalg.norm(c)
    
    # Side lengths (angular distances)
    c_len = np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))
    a_len = np.arccos(np.clip(np.dot(b, c), -1.0, 1.0))
    b_len = np.arccos(np.clip(np.dot(c, a), -1.0, 1.0))
    
    # Spherical circumradius formula:
    # sin(R) = sin(a_len) * sin(b_len) * sin(c_len) / (4 * Area)
    # where Area is the spherical excess.
    # Alternatively, use the formula for the circumcenter.
    # The circumcenter C satisfies |C-a| = |C-b| = |C-c| (Euclidean distance).
    # This implies C is perpendicular to (a-b) and (b-c).
    # C is proportional to (a x b) x (b x c) ? No, that's the normal to the plane.
    # Let's use the linear system:
    # (a-a) . C = 0 => a . C = a . a = 1? No.
    # |C-a|^2 = |C-b|^2 => |C|^2 - 2 C.a + |a|^2 = |C|^2 - 2 C.b + |b|^2
    # Since |a|=|b|=1, we have C.a = C.b.
    # Similarly C.b = C.c.
    # So C is in the direction of the vector that has equal dot product with a, b, c.
    # This vector is the normal to the plane passing through a, b, c.
    # Wait, the normal to the plane is n = (b-a) x (c-a).
    # If we pick C = k * n, then C.a = k * n.a.
    # Is n.a = n.b = n.c?
    # n . (b-a) = 0 => n.b = n.a.
    # n . (c-a) = 0 => n.c = n.a.
    # Yes! So the circumcenter is in the direction of the normal to the plane defined by a,b,c.
    # However, there are two normals (n and -n). We want the one on the same side as the origin?
    # No, we want the one that minimizes the distance.
    # Actually, the circumcenter is the intersection of the perpendicular bisectors.
    # The vector n = (b-a) x (c-a) is perpendicular to the plane.
    # The distance from origin to the plane is d = |a . n| / |n|.
    # The circumradius R (angular) satisfies sin(R) = d / 1? No.
    # Let's use the property that the circumcenter C is k * n.
    # We need |C - a| to be minimized? No, it's fixed for a given triangle.
    # The angular distance theta satisfies cos(theta) = C . a / |C| (since |a|=1).
    # We need to determine the sign of k.
    # The circumcenter must be on the same side of the triangle as the origin?
    # For a covering problem, we are looking for the point in the facet that is farthest from the vertices.
    # This is the circumcenter if the triangle is acute (on the sphere).
    # If the triangle is obtuse, the circumcenter is outside, and the farthest point is the midpoint of the longest side.
    
    n = np.cross(b - a, c - a)
    norm_n = np.linalg.norm(n)
    if norm_n < 1e-10:
        # Collinear
        # Farthest point is midpoint of the longest arc
        # Sides are a_len, b_len, c_len
        max_side = max(a_len, b_len, c_len)
        return max_side / 2.0

    n = n / norm_n
    
    # There are two candidates: n and -n.
    # The circumcenter C must satisfy C . a = C . b = C . c.
    # Both n and -n satisfy this.
    # Which one is the circumcenter?
    # The circumcenter is the center of the circle passing through a, b, c.
    # The circle lies on the plane.
    # The distance from the origin to the plane is h = |a . n|.
    # The radius of the circle in 3D is r = sqrt(1 - h^2).
    # The angular radius R is the angle subtended by this circle at the origin.
    # sin(R) = r / 1 = sqrt(1 - h^2).
    # So R = arcsin(sqrt(1 - h^2)) = arccos(h).
    # Wait, is this always the covering radius for the facet?
    # The facet is the spherical triangle.
    # The covering radius is the max distance from any point in the triangle to the nearest vertex.
    # If the circumcenter lies inside the triangle, the max distance is R.
    # If the circumcenter lies outside, the max distance is the max distance to the vertices from the boundary.
    # For a spherical triangle, if it contains the antipode of the circumcenter, it's large.
    # Generally, for small triangles (which we expect in a good covering), the circumcenter is inside.
    
    h = np.abs(np.dot(a, n))
    # h is distance from origin to plane.
    # h <= 1.
    h = np.clip(h, 0.0, 1.0)
    R = np.arccos(h)
    
    # Check if circumcenter is inside the triangle.
    # This is true if the triangle is acute.
    # A spherical triangle is acute if all angles are < pi/2.
    # Or simpler: if the circumcenter is on the same side of each edge as the opposite vertex.
    # Let's check if the circumcenter C = n (or -n) is inside.
    # The vector from a to b is b-a. The normal to the great circle a-b is m1 = a x b.
    # The side containing c is determined by sign(m1 . c).
    # The side containing C is determined by sign(m1 . C).
    # They must match.
    
    # Let's determine which normal (n or -n) is the circumcenter.
    # Actually, the circumcenter is the point C such that C is in the plane and equidistant.
    # We established C is parallel to n.
    # Let's test both n and -n.
    # The condition for C to be inside is that C is on the same side of each edge as the opposite vertex.
    
    def _is_inside(a, b, c, C):
        # C must be on the same side of edge ab as c
        # Normal to great circle ab is n_ab = a x b
        n_ab = np.cross(a, b)
        side_c = np.sign(np.dot(c, n_ab))
        side_C = np.sign(np.dot(C, n_ab))
        if side_c != side_C:
            return False
        
        n_bc = np.cross(b, c)
        side_a = np.sign(np.dot(a, n_bc))
        side_C2 = np.sign(np.dot(C, n_bc))
        if side_a != side_C2:
            return False
            
        n_ca = np.cross(c, a)
        side_b = np.sign(np.dot(b, n_ca))
        side_C3 = np.sign(np.dot(C, n_ca))
        if side_b != side_C3:
            return False
            
        return True

    C1 = n
    C2 = -n
    
    if _is_inside(a, b, c, C1):
        return R
    elif _is_inside(a, b, c, C2):
        return R
    else:
        # Circumcenter is outside.
        # The farthest point in the triangle from the vertices is one of the vertices?
        # No, it's on the boundary.
        # Specifically, it's the midpoint of the longest side if the triangle is obtuse.
        # Let's verify.
        # If the circumcenter is outside, the max distance to a vertex is achieved at the midpoint of the longest side?
        # Actually, if the circumcenter is outside, the function d(x) = min(|x-a|, |x-b|, |x-c|)
        # increases as we move away from the vertices.
        # The maximum is at the "incenter" of the Voronoi cell?
        # For a facet, the Voronoi vertex is the circumcenter.
        # If the circumcenter is outside the triangle, the Voronoi vertex is not in the facet.
        # Then the max distance is on the boundary of the facet.
        # The boundary consists of 3 edges.
        # On an edge, the max distance to the endpoints is at the midpoint.
        # So we check the midpoints of the 3 edges.
        
        mid_ab = (a + b) / 2
        mid_ab = _normalize(mid_ab)
        d_ab = np.arccos(np.clip(np.dot(mid_ab, a), -1, 1))
        
        mid_bc = (b + c) / 2
        mid_bc = _normalize(mid_bc)
        d_bc = np.arccos(np.clip(np.dot(mid_bc, b), -1, 1))
        
        mid_ca = (c + a) / 2
        mid_ca = _normalize(mid_ca)
        d_ca = np.arccos(np.clip(np.dot(mid_ca, c), -1, 1))
        
        return max(d_ab, d_bc, d_ca)

def _covering_radius(pts):
    """Compute the exact covering radius using convex hull facets."""
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
        max_cr = 0.0
        for simplex_idx in hull.simplices:
            if len(simplex_idx) == 3:
                a, b, c = pts[simplex_idx]
                cr = _spherical_circumradius(a, b, c)
                max_cr = max(max_cr, cr)
            elif len(simplex_idx) == 2:
                a, b = pts[simplex_idx]
                mid = (a + b) / 2
                mid = _normalize(mid)
                cr = np.arccos(np.clip(np.dot(mid, a), -1, 1))
                max_cr = max(max_cr, cr)
            else:
                # Degenerate
                max_cr = np.pi
        return np.degrees(max_cr)
    except Exception:
        # Fallback: sampling
        rng = np.random.RandomState(42)
        samples = rng.randn(10000, 3)
        samples = _normalize(samples)
        # Min distance to nearest point
        # (M, N) distances
        dots = samples @ pts.T
        dots = np.clip(dots, -1, 1)
        min_dots = np.max(dots, axis=1) # Max dot product = Min angular distance
        max_ang = np.max(np.arccos(min_dots))
        return np.degrees(max_ang)

def _smooth_covering_radius(pts, beta=100.0):
    """
    A smooth approximation of the covering radius for gradient-based optimization.
    Uses the log-sum-exp of the distances to the nearest point?
    No, we want to minimize the max distance.
    Let f_i(x) be the distance from x to point i.
    We want min_x max_i f_i(x).
    This is a minimax problem.
    We can approximate max_i f_i(x) with (1/beta) * log(sum(exp(beta * f_i(x)))).
    But f_i(x) is the angular distance.
    This is expensive to compute for all points on the sphere.
    
    Alternative: Use the dual problem.
    The covering radius is the smallest r such that the union of caps of radius r covers the sphere.
    
    Let's stick to a simpler smooth proxy for optimization:
    Minimize the sum of squared distances to the "Voronoi vertices"?
    
    Actually, let's use the gradient of the exact covering radius?
    It's non-differentiable.
    
    Let's use the method from the inspirations:
    Optimize the points to minimize the sum of the circumradii of the facets?
    Or maximize the minimum distance to the nearest facet circumcenter?
    
    Let's try a different approach:
    The covering radius is determined by the worst facet.
    We can iteratively identify the worst facet and optimize the points to reduce its circumradius.
    
    However, for a smooth objective, we can use:
    Objective = sum over all facets of exp(beta * circumradius(facet))
    This focuses on the large facets.
    """
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
        radii = []
        for simplex_idx in hull.simplices:
            if len(simplex_idx) == 3:
                a, b, c = pts[simplex_idx]
                cr = _spherical_circumradius(a, b, c)
                radii.append(cr)
            elif len(simplex_idx) == 2:
                a, b = pts[simplex_idx]
                mid = (a + b) / 2
                mid = _normalize(mid)
                cr = np.arccos(np.clip(np.dot(mid, a), -1, 1))
                radii.append(cr)
        
        if not radii:
            return 0.0
            
        radii = np.array(radii)
        # Smooth max
        max_r = np.max(radii)
        if max_r == 0:
            return 0.0
        shifted = radii * beta
        shifted -= np.max(shifted)
        lse = np.log(np.sum(np.exp(shifted))) + np.max(shifted)
        return lse / beta
    except Exception:
        # Fallback if hull fails
        return 100.0

def _fibonacci_points(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return _normalize(np.column_stack([r * np.cos(phi), r * np.sin(phi), z]))

def _random_points(n, seed):
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    return _normalize(pts)

def _local_optimize(pts, time_limit, beta_start=50.0, beta_end=500.0):
    """
    Optimize points using L-BFGS-B or Nelder-Mead on the smooth covering radius.
    We use a schedule for beta to sharpen the objective.
    """
    pts = _normalize(pts).copy()
    best_pts = pts.copy()
    best_score = _covering_radius(best_pts)
    start_time = time.time()
    
    # We will do a few stages of optimization with increasing beta.
    # Nelder-Mead is robust but slow. L-BFGS-B is faster but needs gradients.
    # We don't have easy gradients for the hull-based objective.
    # So we use Nelder-Mead.
    
    # To make Nelder-Mead faster, we can reduce the dimensionality? No.
    # We can use a smaller initial simplex.
    
    # Let's try 3 stages.
    betas = [50.0, 100.0, 500.0]
    
    for beta in betas:
        if time.time() - start_time > time_limit - 5:
            break
            
        def objective(x):
            p = _normalize(x.reshape(-1, 3))
            return _smooth_covering_radius(p, beta=beta)
        
        x0 = best_pts.flatten()
        
        try:
            # Nelder-Mead options
            # maxiter is tricky. Let's set a reasonable limit.
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
            
            if res.fun < 100.0: # Check for validity
                new_pts = _normalize(res.x.reshape(-1, 3))
                new_score = _covering_radius(new_pts)
                if new_score < best_score:
                    best_score = new_score
                    best_pts = new_pts
        except Exception:
            pass
            
    return best_pts, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 130.0 # Safety margin
    
    # 1. Generate diverse initializations
    initializations = []
    initializations.append(_fibonacci_points(N))
    for seed in [42, 123, 456, 789, 101]:
        initializations.append(_random_points(N, seed))
    
    best_pts = None
    best_cr = np.inf
    
    # 2. Evaluate initializations
    for pts in initializations:
        if time.time() > deadline - 30:
            break
        cr = _covering_radius(pts)
        if cr < best_cr:
            best_cr = cr
            best_pts = pts.copy()
            
    # 3. Optimize the best initialization
    if best_pts is not None:
        remaining_time = deadline - time.time() - 10
        if remaining_time > 10:
            opt_pts, opt_cr = _local_optimize(best_pts, remaining_time)
            if opt_cr < best_cr:
                best_cr = opt_cr
                best_pts = opt_pts
                
    # 4. If time permits, try optimizing the second best initialization
    if time.time() < deadline - 20:
        # Find second best
        scores = []
        for pts in initializations:
            cr = _covering_radius(pts)
            scores.append((cr, pts))
        scores.sort(key=lambda x: x[0])
        
        # Try the second one if it's close
        if len(scores) > 1 and scores[1][0] < best_cr + 2.0:
            remaining_time = deadline - time.time() - 10
            if remaining_time > 10:
                opt_pts, opt_cr = _local_optimize(scores[1][1], remaining_time)
                if opt_cr < best_cr:
                    best_cr = opt_cr
                    best_pts = opt_pts

    if best_pts is None:
        best_pts = _fibonacci_points(N)
        
    return _normalize(best_pts)

# EVOLVE-BLOCK-END
