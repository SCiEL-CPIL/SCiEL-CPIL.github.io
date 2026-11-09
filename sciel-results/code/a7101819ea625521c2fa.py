"""Rewritten program for sphere covering with 50 caps."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import itertools


def _covering_radius(points):
    """Compute the covering radius (in radians) for a set of points on the unit sphere.
    
    The covering radius is the maximum angular distance from any point on the sphere
    to its nearest point in the configuration. This equals the maximum circumradius
    of the spherical facets of the convex hull.
    
    We compute this by finding the convex hull, then for each facet (triangle on the
    convex hull), computing the circumcenter of that triangle projected onto the sphere,
    and checking the angular distance from that circumcenter to the three vertices.
    The maximum of these distances over all facets is the covering radius.
    """
    points = np.asarray(points)
    # Ensure points are on unit sphere
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    
    try:
        hull = ConvexHull(points)
    except Exception:
        # Fallback: compute max nearest-neighbor distance
        from scipy.spatial import cKDTree
        tree = cKDTree(points)
        # Sample many points on sphere to estimate covering radius
        rng = np.random.RandomState(42)
        n_samples = 100000
        # Generate random points on sphere
        gauss = rng.randn(n_samples, 3)
        gauss /= np.linalg.norm(gauss, axis=1, keepdims=True)
        d, _ = tree.query(gauss)
        # Angular distance: arccos(clamp(dot, -1, 1))
        dots = np.sum(gauss * points[np.argmin(d, axis=1)][:, None], axis=1)
        # Actually, let's just use the distance from tree
        # The Euclidean distance d relates to angular distance via: d = 2*sin(theta/2)
        # So theta = 2*arcsin(d/2)
        max_theta = 2 * np.arcsin(np.clip(d.max() / 2.0, 0, 1))
        return max_theta
    
    # For each facet of the convex hull, compute the spherical circumradius
    max_theta = 0.0
    for simplex in hull.simplices:
        p0, p1, p2 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
        
        # Compute the circumcenter of the triangle in 3D, then project to sphere
        # The circumcenter of a triangle on the unit sphere is the normalized cross product
        # of (p1-p0) and (p2-p0), but more carefully:
        # The circumcenter C satisfies |C - p0| = |C - p1| = |C - p2| and |C| = 1
        # C lies in the plane of the triangle and on the unit sphere.
        
        # Method: find the point on the unit sphere equidistant from p0, p1, p2
        # This is the normalized vector perpendicular to the plane through origin... no.
        # 
        # Better approach: The circumcenter of a spherical triangle can be found as follows.
        # Let v = p1 - p0, w = p2 - p0
        # The circumcenter in the plane is p0 + a*v + b*w where a, b solve:
        #   |p0 + a*v + b*w|^2 = |p0|^2 = 1  (on unit sphere)
        #   Actually, we want the point C on the unit sphere such that dot(C, p0) = dot(C, p1) = dot(C, p2)
        # This means C is orthogonal to (p1-p0) and (p2-p0):
        #   dot(C, p1-p0) = 0
        #   dot(C, p2-p0) = 0
        # So C is parallel to (p1-p0) x (p2-p0)
        
        n = np.cross(p1 - p0, p2 - p0)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-10:
            continue
        C = n / n_norm
        
        # Check which side: the circumcenter should be on the same side as the facet normal
        # The facet normal (outward) is n. The circumcenter C should satisfy dot(C, p0) > 0
        # for a proper spherical triangle (less than a hemisphere)
        # Actually, we need to be careful. The circumcenter of the spherical triangle
        # is the point on the sphere equidistant from all three vertices, on the "inside"
        # of the triangle. There are two such points (antipodal). We want the one closer
        # to the centroid of the triangle.
        
        centroid = (p0 + p1 + p2) / 3.0
        centroid /= np.linalg.norm(centroid)
        
        if np.dot(C, centroid) < 0:
            C = -C
        
        # The angular distance from C to p0 (or p1, p2) is the circumradius
        dot_val = np.clip(np.dot(C, p0), -1.0, 1.0)
        theta = np.arccos(dot_val)
        if theta > max_theta:
            max_theta = theta
    
    return max_theta


def _covering_radius_degrees(points):
    return np.degrees(_covering_radius(points))


def construct():
    """Construct 50 points on the unit sphere minimizing the covering radius."""
    
    # Start from a Fibonacci spiral configuration
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Normalize
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms
    
    # Optimize using gradient-free method to minimize covering radius
    # We use a local search / optimization approach
    
    best_pts = pts.copy()
    best_radius = _covering_radius(pts)
    
    # Use a simple local optimization: perturb points and accept if covering radius decreases
    # But this is slow. Instead, let's use a more structured approach.
    
    # Alternative: Use the fact that at the optimum, the covering radius is determined
    # by the largest facet circumradius. We can try to minimize the maximum circumradius
    # by moving points.
    
    # Let's try a coordinate descent / steepest descent approach on the sphere.
    # For each point, we can try to move it to reduce the maximum circumradius.
    
    # Actually, let's use a simpler but effective approach: 
    # 1. Start with Fibonacci spiral
    # 2. Use scipy.optimize to minimize the covering radius
    
    # The covering radius is not smooth, so we'll use a derivative-free method.
    # But N=50 means 150 parameters (with constraints). This is challenging.
    
    # Let's try a different approach: use the known best construction by Hardin, Sloane, Smith.
    # Since we don't have the exact coordinates, we'll use a heuristic optimization.
    
    # Use a simulated annealing / local search approach
    np.random.seed(42)
    
    current_pts = pts.copy()
    current_radius = best_radius
    T = 0.01  # Temperature
    T_min = 1e-6
    cooling = 0.999
    
    # Precompute for efficiency
    for iteration in range(5000):
        if T < T_min:
            break
        
        # Pick a random point to perturb
        idx = np.random.randint(N)
        original = current_pts[idx].copy()
        
        # Generate a random perturbation on the sphere
        # Use a small Gaussian perturbation, then re-normalize
        delta = np.random.randn(3) * 0.05
        new_pt = original + delta
        new_pt /= np.linalg.norm(new_pt)
        
        current_pts[idx] = new_pt
        new_radius = _covering_radius(current_pts)
        
        # Accept if better, or with probability exp(-(new-old)/T)
        if new_radius < current_radius:
            current_radius = new_radius
            if current_radius < best_radius:
                best_radius = current_radius
                best_pts = current_pts.copy()
        else:
            # Metropolis acceptance
            prob = np.exp(-(new_radius - current_radius) / T)
            if np.random.rand() < prob:
                current_radius = new_radius
            else:
                current_pts[idx] = original
        
        T *= cooling
    
    # Final polish: try small perturbations around the best solution
    for _ in range(2000):
        idx = np.random.randint(N)
        original = best_pts[idx].copy()
        delta = np.random.randn(3) * 0.01
        new_pt = original + delta
        new_pt /= np.linalg.norm(new_pt)
        
        best_pts[idx] = new_pt
        new_radius = _covering_radius(best_pts)
        
        if new_radius < best_radius:
            best_radius = new_radius
        else:
            best_pts[idx] = original
    
    # Ensure all points are on the unit sphere
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    best_pts = best_pts / norms
    
    return best_pts
# EVOLVE-BLOCK-END
