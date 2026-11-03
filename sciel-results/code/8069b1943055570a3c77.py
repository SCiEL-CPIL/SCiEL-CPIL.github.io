"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull

# Deadline guard: evaluator kills at 150s, return safely before that
DEADLINE = 135.0  # seconds, leave margin

def _covering_radius(pts):
    """Compute the covering radius in radians for points on the unit sphere.
    
    Uses the convex hull approach: the covering radius is the maximum circumradius
    of the spherical facets (triangles) of the convex hull.
    For a triangle with vertices a, b, c on the unit sphere, the circumradius 
    (angular distance from centroid of the great-circle triangle to the farthest vertex)
    can be computed via the spherical excess or via the formula using the dihedral angles.
    
    A simpler and robust approach: for each facet (triangle) of the convex hull,
    compute the circumcenter of the spherical triangle and the max angular distance
    to the three vertices. The covering radius is the max over all facets.
    
    Alternative: use the fact that for a spherical triangle with side lengths a,b,c
    (angular distances), the circumradius R satisfies:
    sin(R) = (sin(a)*sin(b)*sin(c)) / (4 * sin(s-a) * sin(s-b) * sin(s-c))
    where s = (a+b+c)/2. This is the spherical circumradius formula.
    
    Actually, a more direct approach: the covering radius is the maximum over all 
    points y on the sphere of min_i angle(y, p_i). This equals the maximum circumradius
    of the Voronoi cells. For the convex hull of points on the sphere, each facet 
    corresponds to a Voronoi cell boundary. The covering radius is the max over facets
    of the distance from the facet's circumcenter (on the sphere) to the farthest vertex.
    
    For a triangle with vertices at angular distances d1, d2, d3 from the circumcenter,
    the circumradius is max(d1, d2, d3). But we need to find the circumcenter first.
    
    Let me use a numerical approach: for each triangular facet, find the point on the 
    sphere that maximizes the minimum distance to the three vertices. This is the 
    spherical circumcenter. We can compute it using the formula for the circumcenter
    of a spherical triangle.
    
    For a spherical triangle with vertices A, B, C (unit vectors), the circumcenter
    is proportional to A + B + C if the triangle is acute, but this isn't always correct
    on the sphere. A better approach: the circumcenter O satisfies |O - A| = |O - B| = |O - C|
    in 3D, and |O| = 1. This gives O as the intersection of the unit sphere with the 
    plane perpendicular to the edge midpoints... 
    
    Actually, the 3D circumcenter of the triangle (in the plane of the triangle) projected
    to the sphere gives the spherical circumcenter. Let me compute it as follows:
    
    For three points A, B, C on the unit sphere, the circumcenter of the spherical triangle
    is the unit vector in the direction of the 3D circumcenter of triangle ABC.
    The 3D circumcenter of a triangle is the intersection of the perpendicular bisectors.
    
    Let me use a simpler numerical method: for each facet, sample the circumcenter by 
    solving for the point equidistant from the three vertices.
    
    Actually, the most robust method: the covering radius for a set of points on the sphere
    can be computed by finding, for each facet of the convex hull, the circumradius of that
    facet. For a planar triangle with vertices a, b, c, the circumradius is:
    R = |a-b|*|b-c|*|c-a| / (4 * area)
    But this is the 3D circumradius, not the spherical one.
    
    The spherical circumradius R_s satisfies: the angular distance from the circumcenter
    to each vertex is R_s. If we let the 3D circumcenter be O (not necessarily unit length),
    then the spherical circumradius is arccos(O · A / |O|) where A is a vertex.
    
    Let me just compute it numerically for each facet.
    """
    pts = np.asarray(pts)
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    
    try:
        hull = ConvexHull(pts)
    except Exception:
        # Fallback: compute covering radius by sampling
        return _covering_radius_sample(pts)
    
    max_r = 0.0
    for simplex in hull.simplices:
        a, b, c = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
        # Compute the 3D circumcenter of triangle abc
        # Using the formula: circumcenter = a + ((|a-c|^2*(b-a) - |a-b|^2*(c-a)) / (2 * cross(b-a, c-a)))
        ab = b - a
        ac = c - a
        bc = c - b
        cross_ab_ac = np.cross(ab, ac)
        denom = 2.0 * np.dot(cross_ab_ac, cross_ab_ac)
        
        if abs(denom) < 1e-15:
            # Degenerate triangle, use midpoint
            center = (a + b + c) / 3.0
            center = center / np.linalg.norm(center)
            r = np.arccos(np.clip(np.min([np.dot(center, a), np.dot(center, b), np.dot(center, c)]), -1, 1))
            max_r = max(max_r, r)
            continue
        
        # Circumcenter in the plane of the triangle
        t = (np.dot(ab, ab) * np.dot(ac, ac) * np.dot(bc, bc) / (2 * denom))
        # Actually, let me use the standard formula
        # Circumcenter O = a + alpha*(b-a) + beta*(c-a)
        # where alpha, beta satisfy:
        # 2*alpha*|ab|^2 + 2*beta*ab.ac = |ab|^2
        # 2*alpha*ab.ac + 2*beta*|ac|^2 = |ac|^2
        
        ab_sq = np.dot(ab, ab)
        ac_sq = np.dot(ac, ac)
        ab_ac = np.dot(ab, ac)
        
        det = ab_sq * ac_sq - ab_ac * ab_ac
        if abs(det) < 1e-15:
            center = (a + b + c) / 3.0
            center = center / np.linalg.norm(center)
            r = np.arccos(np.clip(np.min([np.dot(center, a), np.dot(center, b), np.dot(center, c)]), -1, 1))
            max_r = max(max_r, r)
            continue
        
        alpha = (ab_sq * ac_sq - ab_ac * ab_ac) / (2 * det)  # This is wrong, let me redo
        # Correct formula:
        # alpha = (|ab|^2 * |ac|^2 - (ab.ac)^2) / (2 * (|ab|^2 * |ac|^2 - (ab.ac)^2)) ... no
        
        # Standard 2D circumcenter formula:
        # alpha = (|ac|^2 * |ab|^2 - (ab.ac)^2) / (2 * det) ... 
        # Let me just use:
        # O = a + (|ab|^2 * (c-a) - |ac|^2 * (b-a)) / (2 * cross(b-a, c-a) · n)
        # where n is the normal to the plane
        
        # Simpler: use the formula from Wikipedia for circumcenter of 3 points
        # D = 2*(ax*(by-cy) + bx*(cy-ay) + cx*(ay-by))
        # But in 3D, we work in the plane of the triangle
        
        # Let me use a different approach: the circumcenter is the point equidistant
        # from a, b, c. Solve for O = a + u*ab + v*ac:
        # |O - b|^2 = |O - a|^2 => 2*u*ab.ab + 2*v*ab.ac = ab.ab
        # |O - c|^2 = |O - a|^2 => 2*u*ab.ac + 2*v*ac.ac = ac.ac
        
        A_mat = np.array([[2*ab_sq, 2*ab_ac], [2*ab_ac, 2*ac_sq]])
        b_vec = np.array([ab_sq, ac_sq])
        
        try:
            uv = np.linalg.solve(A_mat, b_vec)
        except:
            center = (a + b + c) / 3.0
            center = center / np.linalg.norm(center)
            r = np.arccos(np.clip(np.min([np.dot(center, a), np.dot(center, b), np.dot(center, c)]), -1, 1))
            max_r = max(max_r, r)
            continue
        
        center_3d = a + uv[0]*ab + uv[1]*ac
        center_norm = np.linalg.norm(center_3d)
        if center_norm < 1e-15:
            center = (a + b + c) / 3.0
            center = center / np.linalg.norm(center)
        else:
            center = center_3d / center_norm
        
        # Spherical circumradius: max angular distance from center to vertices
        dots = np.clip([np.dot(center, a), np.dot(center, b), np.dot(center, c)], -1, 1)
        r = np.arccos(np.min(dots))
        max_r = max(max_r, r)
    
    return max_r


def _covering_radius_sample(pts, n_samples=100000):
    """Fallback: estimate covering radius by sampling."""
    pts = np.asarray(pts)
    # Generate random points on sphere
    rng = np.random.RandomState(42)
    theta = rng.uniform(0, 2*np.pi, n_samples)
    phi = np.arccos(1 - 2*rng.uniform(0, 1, n_samples))
    samples = np.column_stack([np.sin(phi)*np.cos(theta), np.sin(phi)*np.sin(theta), np.cos(phi)])
    
    # For each sample, find min distance to any point
    # Use dot products: cos(angle) = dot
    dots = samples @ pts.T  # (n_samples, N)
    cos_min = np.max(dots, axis=1)  # max dot = min angle
    angles = np.arccos(np.clip(cos_min, -1, 1))
    return np.max(angles)


def _soft_max_covering_radius(pts, temp=0.05):
    """Soft approximation of the covering radius using log-sum-exp."""
    pts = np.asarray(pts)
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    
    # Sample points on the sphere
    rng = np.random.RandomState(123)
    n_samples = 50000
    theta = rng.uniform(0, 2*np.pi, n_samples)
    phi = np.arccos(1 - 2*rng.uniform(0, 1, n_samples))
    samples = np.column_stack([np.sin(phi)*np.cos(theta), np.sin(phi)*np.sin(theta), np.cos(phi)])
    
    dots = samples @ pts.T
    # For each sample, the min angle is arccos(max dot)
    # Soft version: use -temp * log-sum-exp(-dots/temp) to approximate max dot
    neg_dots = -dots / temp
    max_neg = np.max(neg_dots, axis=1, keepdims=True)
    log_sum_exp = max_neg.squeeze() + np.log(np.sum(np.exp(neg_dots - max_neg), axis=1))
    soft_max_dot = -temp * log_sum_exp
    
    # The covering radius is approximately arccos(soft_max_dot)
    # But arccos is not smooth at 1, so let's work in cos space
    # Minimize the max angle = minimize arccos(min_dot) = maximize min_dot
    # So we want to maximize the soft max dot
    return -np.mean(soft_max_dot)  # Negate because we want to maximize


def construct():
    t_start = time.time()
    
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Initial covering radius
    best_pts = pts.copy()
    best_cr = _covering_radius(pts)
    
    # Gradient-based optimization with backtracking line search
    # Objective: minimize the covering radius
    # Use a smooth surrogate: the soft-max of the angular distances
    
    def objective(x):
        """Smooth objective: soft-max of covering radius."""
        x = x.reshape(N, 3)
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
        
        # Use convex hull based covering radius for accuracy
        # But for gradient, use a smooth approximation
        # Sample-based soft max
        rng = np.random.RandomState(7)
        n_samples = 20000
        theta = rng.uniform(0, 2*np.pi, n_samples)
        phi_s = np.arccos(1 - 2*rng.uniform(0, 1, n_samples))
        samples = np.column_stack([np.sin(phi_s)*np.cos(theta), np.sin(phi_s)*np.sin(theta), np.cos(phi_s)])
        
        dots = samples @ x.T  # (n_samples, N)
        # min angle for each sample = arccos(max dot)
        # soft max dot: -t * log(sum(exp(-dots/t)))
        t = 0.03
        neg = -dots / t
        max_neg = np.max(neg, axis=1, keepdims=True)
        lse = max_neg.squeeze() + np.log(np.sum(np.exp(neg - max_neg), axis=1))
        soft_max = -t * lse
        # We want to maximize soft_max (equivalent to minimizing covering radius)
        # So objective = -mean(soft_max)
        return -np.mean(soft_max)
    
    def gradient(x):
        """Gradient of the smooth objective."""
        x = x.reshape(N, 3)
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
        
        rng = np.random.RandomState(7)
        n_samples = 20000
        theta = rng.uniform(0, 2*np.pi, n_samples)
        phi_s = np.arccos(1 - 2*rng.uniform(0, 1, n_samples))
        samples = np.column_stack([np.sin(phi_s)*np.cos(theta), np.sin(phi_s)*np.sin(theta), np.cos(phi_s)])
        
        dots = samples @ x.T  # (n_samples, N)
        t = 0.03
        neg = -dots / t
        max_neg = np.max(neg, axis=1, keepdims=True)
        exp_neg = np.exp(neg - max_neg)
        sum_exp = np.sum(exp_neg, axis=1, keepdims=True)
        weights = exp_neg / sum_exp  # (n_samples, N) softmax weights
        
        # d/dx of -mean(soft_max) where soft_max = -t * lse
        # lse = max_neg + log(sum_exp)
        # d(soft_max)/d(dots[i,j]) = t * weights[i,j]
        # d(dots[i,j])/dx[j] = samples[i]
        # So d(soft_max)/dx[j] = sum_i t * weights[i,j] * samples[i]
        # d(objective)/dx[j] = -d(soft_max)/dx[j] = -t * sum_i weights[i,j] * samples[i]
        
        # grad[j] = -t * (weights.T @ samples)[j]
        grad_raw = -t * (weights.T @ samples)  # (N, 3)
        
        # Project gradient onto tangent space of sphere
        grad = grad_raw - (grad_raw * x).sum(axis=1, keepdims=True) * x
        
        return grad.flatten()
    
    # Optimize
    x = pts.flatten()
    lr = 0.01
    max_iters = 5000
    
    for it in range(max_iters):
        if time.time() > DEADLINE - 10:
            break
        
        g = gradient(x)
        if np.linalg.norm(g) < 1e-8:
            break
        
        # Backtracking line search
        alpha = lr
        f_current = objective(x)
        improved = False
        for _ in range(20):
            x_new = x - alpha * g
            # Normalize to sphere
            x_new = x_new.reshape(N, 3)
            x_new = x_new / np.linalg.norm(x_new, axis=1, keepdims=True)
            f_new = objective(x_new)
            if f_new < f_current - 1e-4 * alpha * np.dot(g, g):
                x = x_new.flatten()
                improved = True
                break
            alpha *= 0.5
        
        if not improved:
            # Try smaller step
            x_new = x - 0.001 * g
            x_new = x_new.reshape(N, 3)
            x_new = x_new / np.linalg.norm(x_new, axis=1, keepdims=True)
            x = x_new.flatten()
        
        # Check improvement in actual covering radius periodically
        if it % 50 == 0:
            x_check = x.reshape(N, 3)
            x_check = x_check / np.linalg.norm(x_check, axis=1, keepdims=True)
            cr = _covering_radius(x_check)
            if cr < best_cr - 1e-6:
                best_cr = cr
                best_pts = x_check.copy()
    
    # Final polish: use the best points found
    # Do a few more iterations with smaller steps
    x = best_pts.flatten()
    for it in range(500):
        if time.time() > DEADLINE - 5:
            break
        
        g = gradient(x)
        if np.linalg.norm(g) < 1e-10:
            break
        
        alpha = 0.001
        x_new = x - alpha * g
        x_new = x_new.reshape(N, 3)
        x_new = x_new / np.linalg.norm(x_new, axis=1, keepdims=True)
        x = x_new.flatten()
    
    x_final = x.reshape(N, 3)
    x_final = x_final / np.linalg.norm(x_final, axis=1, keepdims=True)
    
    # Compare with best
    cr_final = _covering_radius(x_final)
    if cr_final < best_cr:
        best_pts = x_final
    
    return best_pts
# EVOLVE-BLOCK-END
