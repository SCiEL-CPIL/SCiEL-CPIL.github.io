"""Rewritten program for sphere covering with 130 points."""
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 130
DEADLINE_SECONDS = 140.0  # safety margin under 150s kill

def _compute_covering_radius(pts):
    """Compute the covering radius in degrees for a set of unit-sphere points."""
    pts = np.asarray(pts)
    if pts.shape[0] < 4:
        return np.pi
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi
    max_r = 0.0
    for simplex in hull.simplices:
        # simplex is a set of 3 indices (triangular facets in 3D)
        p0 = pts[simplex[0]]
        p1 = pts[simplex[1]]
        p2 = pts[simplex[2]]
        # Circumcenter of the triangle on the unit sphere
        # The circumcenter of a spherical triangle (on unit sphere) is the
        # normalized cross product of (p1-p0) x (p2-p0) ... actually we need
        # the point on the sphere equidistant from all three vertices.
        # For a triangle on the unit sphere, the circumcenter is along the
        # normal to the plane containing the three points, normalized.
        # But we need the one on the same side as the sphere surface.
        
        # Compute the circumcenter of the planar triangle, then project to sphere
        a = p1 - p0
        b = p2 - p0
        c = np.cross(a, b)
        if np.linalg.norm(c) < 1e-12:
            continue
        c = c / np.linalg.norm(c)
        
        # The circumcenter of the spherical triangle is the point on the sphere
        # in the direction of the normalized vector that is equidistant from all 3.
        # Actually, for points on the unit sphere, the circumcenter of the triangle
        # (as a set of 3 points) lies along the direction of the cross product
        # of two edge vectors, but we need the correct one.
        
        # Better approach: solve for the point x on the unit sphere such that
        # |x - p0| = |x - p1| = |x - p2|.
        # This means x is in the plane perpendicular to (p1-p0) passing through midpoint,
        # and similarly for (p2-p0).
        # The circumcenter is along the normal to the plane of the triangle.
        
        # The plane normal is c (computed above). The circumcenter on the sphere
        # is either c or -c (whichever is on the "outside" of the convex hull).
        # For a convex hull facet, the outward normal points away from the origin.
        # Since all points are on the unit sphere, the facet's circumcenter that
        # is farthest from the origin in the outward direction is what we want.
        
        # Actually, the covering radius for a facet is the maximum angular distance
        # from any point on the spherical triangle to its three vertices.
        # This is the circumradius of the spherical triangle.
        
        # The circumradius R satisfies: the angular distance from circumcenter to
        # each vertex is R. The circumcenter is at distance h from the origin
        # along the normal direction.
        
        # Let's compute: the circumcenter in 3D (not necessarily on the sphere)
        # of the triangle p0, p1, p2.
        # Standard formula:
        d = 2.0 * (p0[0] * (p1[1] * p2[2] - p1[2] * p2[1]) -
                   p0[1] * (p1[0] * p2[2] - p1[2] * p2[0]) +
                   p0[2] * (p1[0] * p2[1] - p1[1] * p2[0]))
        if abs(d) < 1e-12:
            continue
        
        ux = ((p0[0]**2 + p0[1]**2 + p0[2]**2) * (p1[1]*p2[2] - p1[2]*p2[1]) -
              (p1[0]**2 + p1[1]**2 + p1[2]**2) * (p0[1]*p2[2] - p0[2]*p2[1]) +
              (p2[0]**2 + p2[1]**2 + p2[2]**2) * (p0[1]*p1[2] - p0[2]*p1[1])) / d
        uy = ((p1[0]**2 + p1[1]**2 + p1[2]**2) * (p0[2]*p2[0] - p0[0]*p2[2]) -
              (p0[0]**2 + p0[1]**2 + p0[2]**2) * (p1[2]*p2[0] - p1[0]*p2[2]) +
              (p2[0]**2 + p2[1]**2 + p2[2]**2) * (p1[0]*p0[2] - p1[2]*p0[0])) / d
        uz = ((p0[0]**2 + p0[1]**2 + p0[2]**2) * (p1[0]*p2[1] - p1[1]*p2[0]) -
              (p1[0]**2 + p1[1]**2 + p1[2]**2) * (p0[0]*p2[1] - p0[1]*p2[0]) +
              (p2[0]**2 + p2[1]**2 + p2[2]**2) * (p0[0]*p1[1] - p0[1]*p1[0])) / d
        
        cc = np.array([ux, uy, uz])
        # The circumcenter of the spherical triangle is the projection of cc onto
        # the sphere, but we need the one that is "outside" the convex hull.
        # For a convex hull of points on the unit sphere, the circumcenter of a facet
        # that is relevant for the covering radius is the one farther from the origin
        # in the direction away from the hull interior.
        
        # The distance from origin to circumcenter:
        cc_norm = np.linalg.norm(cc)
        if cc_norm < 1e-12:
            continue
        
        # The angular circumradius is the angle between cc/cc_norm and any vertex.
        # But we need to be careful: the circumcenter on the sphere is cc/cc_norm
        # if cc is on the "outside" side. For a convex hull, the facet normal
        # points outward, and the circumcenter should be on the outside.
        
        # The circumradius in angular terms:
        # cos(R) = dot(cc_norm_unit, p0) where cc_norm_unit = cc / |cc|
        cc_unit = cc / cc_norm
        cos_R = np.clip(np.dot(cc_unit, p0), -1.0, 1.0)
        R = np.arccos(cos_R)
        
        # But we need to check if this circumcenter is actually the "empty cap" center.
        # For a convex hull facet, the circumcenter that matters is the one on the
        # side of the facet away from the origin (outward). Since all points are on
        # the unit sphere, the outward circumcenter is the one with positive dot
        # with the facet normal.
        
        # The facet normal (outward) is c (or -c). Let's check which direction.
        # For a convex hull, the outward normal points away from the centroid.
        centroid = (p0 + p1 + p2) / 3.0
        if np.dot(c, centroid) > 0:
            # c points toward the origin, so outward is -c
            outward_normal = -c
        else:
            outward_normal = c
        
        # The circumcenter on the sphere should be in the direction of outward_normal
        # if it's on the outside. Let's check the sign.
        if np.dot(cc_unit, outward_normal) < 0:
            cc_unit = -cc_unit
            cos_R = np.clip(np.dot(cc_unit, p0), -1.0, 1.0)
            R = np.arccos(cos_R)
        
        max_r = max(max_r, R)
    
    return max_r * 180.0 / np.pi


def _fibonacci_sphere(n):
    """Generate n points on the unit sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, seed=42):
    """Generate n random points on the unit sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def _refine(pts, time_budget=10.0):
    """Refine a configuration by local optimization to minimize covering radius."""
    pts = np.array(pts, dtype=float)
    # Normalize
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms
    
    start_time = time.time()
    
    # Use L-BFGS-B on the flattened array, with the objective being the covering radius
    # Since covering radius is non-smooth, we'll use a smooth approximation.
    
    def smooth_covering_radius(x, beta=10.0):
        """Smooth approximation of the covering radius using log-sum-exp."""
        pts = x.reshape(-1, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / (norms + 1e-12)
        
        # Compute covering radius via convex hull
        try:
            hull = ConvexHull(pts)
        except Exception:
            return np.pi
        
        max_cos = 0.0  # We want to maximize the minimum cos (i.e., minimize max angle)
        # Actually, we want to minimize the max circumradius.
        # The circumradius R for a facet satisfies cos(R) = dot(cc_unit, p_i).
        # To minimize R, we want to maximize cos(R) for the worst facet.
        # But this is still non-smooth.
        
        # Alternative: use the fact that the covering radius is the max over facets
        # of the circumradius. We can use a soft-max over the circumradii.
        
        circumradii = []
        for simplex in hull.simplices:
            p0 = pts[simplex[0]]
            p1 = pts[simplex[1]]
            p2 = pts[simplex[2]]
            
            d = 2.0 * (p0[0] * (p1[1] * p2[2] - p1[2] * p2[1]) -
                       p0[1] * (p1[0] * p2[2] - p1[2] * p2[0]) +
                       p0[2] * (p1[0] * p2[1] - p1[1] * p2[0]))
            if abs(d) < 1e-12:
                continue
            
            ux = ((p0[0]**2 + p0[1]**2 + p0[2]**2) * (p1[1]*p2[2] - p1[2]*p2[1]) -
                  (p1[0]**2 + p1[1]**2 + p1[2]**2) * (p0[1]*p2[2] - p0[2]*p2[1]) +
                  (p2[0]**2 + p2[1]**2 + p2[2]**2) * (p0[1]*p1[2] - p0[2]*p1[1])) / d
            uy = ((p1[0]**2 + p1[1]**2 + p1[2]**2) * (p0[2]*p2[0] - p0[0]*p2[2]) -
                  (p0[0]**2 + p0[1]**2 + p0[2]**2) * (p1[2]*p2[0] - p1[0]*p2[2]) +
                  (p2[0]**2 + p2[1]**2 + p2[2]**2) * (p1[0]*p0[2] - p1[2]*p0[0])) / d
            uz = ((p0[0]**2 + p0[1]**2 + p0[2]**2) * (p1[0]*p2[1] - p1[1]*p2[0]) -
                  (p1[0]**2 + p1[1]**2 + p1[2]**2) * (p0[0]*p2[1] - p0[1]*p2[0]) +
                  (p2[0]**2 + p2[1]**2 + p2[2]**2) * (p0[0]*p1[1] - p0[1]*p1[0])) / d
            
            cc = np.array([ux, uy, uz])
            cc_norm = np.linalg.norm(cc)
            if cc_norm < 1e-12:
                continue
            
            cc_unit = cc / cc_norm
            cos_R = np.clip(np.dot(cc_unit, p0), -1.0, 1.0)
            R = np.arccos(cos_R)
            circumradii.append(R)
        
        if not circumradii:
            return np.pi
        
        circumradii = np.array(circumradii)
        # Soft max
        beta_local = beta
        m = np.max(circumradii)
        soft_max = m + np.log(np.sum(np.exp(beta_local * (circumradii - m)))) / beta_local
        return soft_max
    
    # This approach is too slow. Let's use a simpler local refinement.
    # Use coordinate descent on the max facet.
    
    best_pts = pts.copy()
    best_cr = _compute_covering_radius(pts)
    
    while time.time() - start_time < time_budget:
        # Find the worst facet
        try:
            hull = ConvexHull(best_pts)
        except Exception:
            break
        
        worst_facet = None
        worst_R = 0.0
        for simplex in hull.simplices:
            p0 = best_pts[simplex[0]]
            p1 = best_pts[simplex[1]]
            p2 = best_pts[simplex[2]]
            
            d = 2.0 * (p0[0] * (p1[1] * p2[2] - p1[2] * p2[1]) -
                       p0[1] * (p1[0] * p2[2] - p1[2] * p2[0]) +
                       p0[2] * (p1[0] * p2[1] - p1[1] * p2[0]))
            if abs(d) < 1e-12:
                continue
            
            ux = ((p0[0]**2 + p0[1]**2 + p0[2]**2) * (p1[1]*p2[2] - p1[2]*p2[1]) -
                  (p1[0]**2 + p1[1]**2 + p1[2]**2) * (p0[1]*p2[2] - p0[2]*p2[1]) +
                  (p2[0]**2 + p2[1]**2 + p2[2]**2) * (p0[1]*p1[2] - p0[2]*p1[1])) / d
            uy = ((p1[0]**2 + p1[1]**2 + p1[2]**2) * (p0[2]*p2[0] - p0[0]*p2[2]) -
                  (p0[0]**2 + p0[1]**2 + p0[2]**2) * (p1[2]*p2[0] - p1[0]*p2[2]) +
                  (p2[0]**2 + p2[1]**2 + p2[2]**2) * (p1[0]*p0[2] - p1[2]*p0[0])) / d
            uz = ((p0[0]**2 + p0[1]**2 + p0[2]**2) * (p1[0]*p2[1] - p1[1]*p2[0]) -
                  (p1[0]**2 + p1[1]**2 + p1[2]**2) * (p0[0]*p2[1] - p0[1]*p2[0]) +
                  (p2[0]**2 + p2[1]**2 + p2[2]**2) * (p0[0]*p1[1] - p0[1]*p1[0])) / d
            
            cc = np.array([ux, uy, uz])
            cc_norm = np.linalg.norm(cc)
            if cc_norm < 1e-12:
                continue
            
            cc_unit = cc / cc_norm
            cos_R = np.clip(np.dot(cc_unit, p0), -1.0, 1.0)
            R = np.arccos(cos_R)
            
            if R > worst_R:
                worst_R = R
                worst_facet = simplex
        
        if worst_facet is None:
            break
        
        # Try to move each of the three vertices slightly toward the circumcenter
        # to reduce the circumradius
        p0 = best_pts[worst_facet[0]]
        p1 = best_pts[worst_facet[1]]
        p2 = best_pts[worst_facet[2]]
        
        d = 2.0 * (p0[0] * (p1[1] * p2[2] - p1[2] * p2[1]) -
                   p0[1] * (p1[0] * p2[2] - p1[2] * p2[0]) +
                   p0[2] * (p1[0] * p2[1] - p1[1] * p2[0]))
        if abs(d) < 1e-12:
            break
        
        ux = ((p0[0]**2 + p0[1]**2 + p0[2]**2) * (p1[1]*p2[2] - p1[2]*p2[1]) -
              (p1[0]**2 + p1[1]**2 + p1[2]**2) * (p0[1]*p2[2] - p0[2]*p2[1]) +
              (p2[0]**2 + p2[1]**2 + p2[2]**2) * (p0[1]*p1[2] - p0[2]*p1[1])) / d
        uy = ((p1[0]**2 + p1[1]**2 + p1[2]**2) * (p0[2]*p2[0] - p0[0]*p2[2]) -
              (p0[0]**2 + p0[1]**2 + p0[2]**2) * (p1[2]*p2[0] - p1[0]*p2[2]) +
              (p2[0]**2 + p2[1]**2 + p2[2]**2) * (p1[0]*p0[2] - p1[2]*p0[0])) / d
        uz = ((p0[0]**2 + p0[1]**2 + p0[2]**2) * (p1[0]*p2[1] - p1[1]*p2[0]) -
              (p1[0]**2 + p1[1]**2 + p1[2]**2) * (p0[0]*p2[1] - p0[1]*p2[0]) +
              (p2[0]**2 + p2[1]**2 + p2[2]**2) * (p0[0]*p1[1] - p0[1]*p1[0])) / d
        
        cc = np.array([ux, uy, uz])
        cc_norm = np.linalg.norm(cc)
        if cc_norm < 1e-12:
            break
        cc_unit = cc / cc_norm
        
        # Move each vertex toward the circumcenter
        improved = False
        for idx in worst_facet:
            p = best_pts[idx]
            # Move p toward cc_unit
            for step in [0.01, 0.005, 0.001, 0.05]:
                new_p = p + step * (cc_unit - p)
                new_p = new_p / np.linalg.norm(new_p)
                trial = best_pts.copy()
                trial[idx] = new_p
                trial_cr = _compute_covering_radius(trial)
                if trial_cr < best_cr - 1e-10:
                    best_pts = trial
                    best_cr = trial_cr
                    improved = True
                    break
        
        if not improved:
            break
    
    return best_pts, best_cr


def construct():
    start_time = time.time()
    
    # Start with Fibonacci sphere
    pts = _fibonacci_sphere(N)
    best_pts = pts.copy()
    best_cr = _compute_covering_radius(pts)
    
    # Try multiple random starts and refine
    rng = np.random.RandomState(12345)
    n_starts = 5
    
    for i in range(n_starts):
        if time.time() - start_time > DEADLINE_SECONDS - 30:
            break
        
        if i == 0:
            candidate = pts.copy()
        else:
            # Perturb the Fibonacci sphere
            candidate = pts + 0.1 * rng.randn(N, 3)
            norms = np.linalg.norm(candidate, axis=1, keepdims=True)
            candidate = candidate / norms
        
        # Refine
        refined, cr = _refine(candidate, time_budget=min(20.0, DEADLINE_SECONDS - (time.time() - start_time) - 10))
        if cr < best_cr:
            best_pts = refined
            best_cr = cr
    
    # Final polish with more time if available
    remaining = DEADLINE_SECONDS - (time.time() - start_time) - 5
    if remaining > 5:
        polished, cr = _refine(best_pts, time_budget=remaining)
        if cr < best_cr:
            best_pts = polished
            best_cr = cr
    
    # Ensure output is valid
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    best_pts = best_pts / norms
    
    return best_pts
