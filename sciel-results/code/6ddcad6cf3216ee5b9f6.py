"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130
import time
import numpy as np
from scipy.optimize import minimize

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time

def _covering_radius(points):
    """Compute the covering radius in degrees for a set of unit vectors."""
    # Use the convex hull to find the largest circumradius of facets
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi  # invalid, return large value
    
    max_r = 0.0
    n = len(points)
    
    # For each facet (simplex), compute the circumcenter and check if the cap is empty
    # The covering radius is the max over all points y on sphere of min distance to points.
    # This equals the max circumradius of the spherical Voronoi cells, which corresponds to
    # the max circumradius of the convex hull facets (since the farthest point from a facet's
    # vertices within the cap is the circumcenter of that facet's great circle).
    
    # Actually, the covering radius is determined by the farthest point from any configuration point.
    # This occurs at a vertex of the spherical Voronoi diagram, which corresponds to a circumcenter
    # of a facet of the convex hull.
    
    # For each facet, compute the circumcenter of the corresponding spherical triangle
    # The circumcenter in 3D is the normalized centroid direction? No, it's more complex.
    
    # Simpler approach: the covering radius is the max over all facets of the angular distance
    # from the facet's circumcenter to its vertices. For a spherical triangle with vertices a,b,c,
    # the circumcenter is the point equidistant from all three.
    
    # For a facet with vertices p1, p2, ..., pk (k=3 for generic), the circumcenter on the sphere
    # is the normalized solution to the system of equations.
    
    # Alternative: use the fact that the covering radius is the max over all points on the sphere
    # of the min distance. We can approximate this by checking the circumcenters of all facets.
    
    for simplex in hull.simplices:
        pts = points[simplex]  # shape (k, 3)
        k = len(pts)
        
        if k == 3:
            # Spherical triangle: find circumcenter
            # The circumcenter is the normalized vector in the direction of the intersection
            # of the three great circles' perpendicular bisectors.
            # For three points a, b, c on the unit sphere, the circumcenter is proportional to
            # (a x b) x (b x c) + ... actually, it's the normalized vector:
            # circumcenter = normalize(a * sin(2A) + b * sin(2B) + c * sin(2C))
            # where A, B, C are the angles opposite to a, b, c.
            # 
            # Simpler: the circumcenter is the normalized cross product of (b-a) and (c-a) 
            # projected... no.
            #
            # The circumcenter of a spherical triangle is the normalized vector:
            # c = a * (b·c) + b * (c·a) + c * (a·b) - (a·b·c) * (a+b+c)
            # This is getting complicated. Let me use a different approach.
            #
            # For three unit vectors a, b, c, the circumcenter (normalized) is:
            # n = (a × b) × (b × c) + ... 
            # Actually, the point equidistant from a, b, c satisfies:
            # x·a = x·b = x·c, |x| = 1
            # This means x is in the null space of [a-b; b-c] (as a linear system)
            # x·(a-b) = 0 and x·(b-c) = 0
            # So x is parallel to (a-b) × (b-c)
            
            v1 = pts[0] - pts[1]
            v2 = pts[1] - pts[2]
            x = np.cross(v1, v2)
            norm_x = np.linalg.norm(x)
            if norm_x < 1e-12:
                # Degenerate, use centroid
                cc = pts.mean(axis=0)
                cc = cc / np.linalg.norm(cc)
            else:
                cc = x / norm_x
            
            # The circumcenter could be on either side; pick the one that gives the 
            # correct (smaller) radius, i.e., the one on the same side as the facet normal
            # For the covering radius, we want the point on the sphere that is farthest
            # from all points, so we want the circumcenter that is on the "outside"
            # of the convex hull, i.e., in the direction of the facet normal.
            
            # The facet normal for the convex hull is outward-pointing
            # Let's compute the facet normal
            normal = np.cross(pts[1] - pts[0], pts[2] - pts[0])
            normal = normal / np.linalg.norm(normal)
            
            # Ensure cc is in the same hemisphere as the normal
            if np.dot(cc, normal) < 0:
                cc = -cc
            
            # Compute the angular distance from cc to the vertices
            dists = np.arccos(np.clip(pts @ cc, -1.0, 1.0))
            r = np.max(dists)
            
        elif k == 4:
            # Spherical tetrahedron: circumcenter is the normalized centroid? No.
            # For 4 points, the circumcenter satisfies x·pi = const for all i.
            # x·(p1-p2) = 0, x·(p1-p3) = 0, x·(p1-p4) = 0
            # x is in the null space of a 3x3 system
            A = np.array([pts[0]-pts[1], pts[0]-pts[2], pts[0]-pts[3]])
            # x is orthogonal to all rows of A, so x = cross of two rows
            v1 = A[0]
            v2 = A[1]
            x = np.cross(v1, v2)
            norm_x = np.linalg.norm(x)
            if norm_x < 1e-12:
                cc = pts.mean(axis=0)
                cc = cc / np.linalg.norm(cc)
            else:
                cc = x / norm_x
            
            normal = np.cross(pts[1] - pts[0], pts[2] - pts[0])
            normal = normal / np.linalg.norm(normal)
            if np.dot(cc, normal) < 0:
                cc = -cc
            
            dists = np.arccos(np.clip(pts @ cc, -1.0, 1.0))
            r = np.max(dists)
        else:
            continue
        
        max_r = max(max_r, r)
    
    return np.degrees(max_r)


def _soft_covering_radius(points, temperature=0.05):
    """Smooth approximation of the covering radius using log-sum-exp."""
    # Sample points on the sphere to approximate the max over y
    # Use a grid of points
    n_theta = 20
    n_phi = 40
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    TH, PH = np.meshgrid(theta, phi, indexing='ij')
    Y = np.column_stack([
        np.sin(TH) * np.cos(PH).ravel(),
        np.sin(TH) * np.sin(PH).ravel(),
        np.cos(TH).ravel()
    ])
    
    # Compute pairwise angular distances: Y[i] to points[j]
    # dots[i,j] = Y[i] @ points[j]
    dots = Y @ points.T  # shape (n_samples, N)
    dots = np.clip(dots, -1.0, 1.0)
    dists = np.arccos(dots)  # angular distances
    
    # For each y, min distance to any point
    min_dists = np.min(dists, axis=1)
    
    # Soft max: temperature * log(sum(exp(dists/temp)))
    # Use log-sum-exp for numerical stability
    max_d = np.max(min_dists)
    log_sum = max_d + np.log(np.sum(np.exp((min_dists - max_d) / temperature)))
    soft_max = temperature * log_sum
    
    return np.degrees(soft_max)


def _gradient_soft_covering_radius(points, temperature=0.05):
    """Gradient of the soft covering radius."""
    n_theta = 20
    n_phi = 40
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    TH, PH = np.meshgrid(theta, phi, indexing='ij')
    Y = np.column_stack([
        np.sin(TH) * np.cos(PH).ravel(),
        np.sin(TH) * np.sin(PH).ravel(),
        np.cos(TH).ravel()
    ])
    
    dots = Y @ points.T
    dots = np.clip(dots, -1.0, 1.0)
    dists = np.arccos(dots)
    min_dists = np.min(dists, axis=1)
    
    # Soft max gradient
    max_d = np.max(min_dists)
    exps = np.exp((min_dists - max_d) / temperature)
    weights = exps / np.sum(exps)
    
    # d(soft_max)/d(min_dists[i]) = weights[i]
    # d(min_dists[i])/d(points[j]) = ?
    # min_dists[i] = min_j dists[i,j]
    # This is nonsmooth, but we can use the argmin
    
    argmins = np.argmin(dists, axis=1)
    
    grad = np.zeros_like(points)
    for i in range(len(Y)):
        j = argmins[i]
        w = weights[i]
        # d(dist)/d(point[j]) where dist = arccos(Y[i] @ points[j])
        # = -Y[i] / sqrt(1 - (Y[i]@points[j])^2)
        dot_val = Y[i] @ points[j]
        dot_val = np.clip(dot_val, -1.0 + 1e-12, 1.0 - 1e-12)
        denom = np.sqrt(1.0 - dot_val * dot_val)
        if denom > 1e-12:
            grad[j] += -w * Y[i] / denom
    
    return np.degrees(grad)


def _project_to_sphere(points):
    """Project points back to the unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return points / norms


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5 second safety margin
    
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = points.copy()
    best_cr = _covering_radius(best_points)
    
    # Optimization using soft covering radius
    temperature = 0.05
    
    def objective(x):
        pts = _project_to_sphere(x.reshape(N, 3))
        return _soft_covering_radius(pts, temperature)
    
    def gradient(x):
        pts = _project_to_sphere(x.reshape(N, 3))
        g = _gradient_soft_covering_radius(pts, temperature)
        # Project gradient to tangent space
        g = g - np.sum(g * pts, axis=1, keepdims=True) * pts
        return g.ravel()
    
    # Use L-BFGS-B with gradient
    x0 = points.ravel()
    
    try:
        result = minimize(
            objective,
            x0,
            jac=gradient,
            method='L-BFGS-B',
            options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        if time.time() < deadline:
            opt_points = _project_to_sphere(result.x.reshape(N, 3))
            cr = _covering_radius(opt_points)
            if cr < best_cr:
                best_points = opt_points
                best_cr = cr
    except Exception:
        pass
    
    # Local refinement: iterate on the worst points
    if time.time() < deadline - 5.0:
        for iteration in range(50):
            if time.time() > deadline - 5.0:
                break
            
            cr = _covering_radius(best_points)
            
            # Find the worst facet (largest circumradius)
            try:
                hull = ConvexHull(best_points)
            except Exception:
                break
            
            worst_r = 0.0
            worst_facet = None
            worst_cc = None
            
            for simplex in hull.simplices:
                pts = best_points[simplex]
                k = len(pts)
                
                if k == 3:
                    v1 = pts[0] - pts[1]
                    v2 = pts[1] - pts[2]
                    x = np.cross(v1, v2)
                    norm_x = np.linalg.norm(x)
                    if norm_x < 1e-12:
                        continue
                    cc = x / norm_x
                    normal = np.cross(pts[1] - pts[0], pts[2] - pts[0])
                    normal = normal / np.linalg.norm(normal)
                    if np.dot(cc, normal) < 0:
                        cc = -cc
                    dists = np.arccos(np.clip(pts @ cc, -1.0, 1.0))
                    r = np.max(dists)
                    if r > worst_r:
                        worst_r = r
                        worst_facet = simplex
                        worst_cc = cc
                elif k == 4:
                    A = np.array([pts[0]-pts[1], pts[0]-pts[2], pts[0]-pts[3]])
                    x = np.cross(A[0], A[1])
                    norm_x = np.linalg.norm(x)
                    if norm_x < 1e-12:
                        continue
                    cc = x / norm_x
                    normal = np.cross(pts[1] - pts[0], pts[2] - pts[0])
                    normal = normal / np.linalg.norm(normal)
                    if np.dot(cc, normal) < 0:
                        cc = -cc
                    dists = np.arccos(np.clip(pts @ cc, -1.0, 1.0))
                    r = np.max(dists)
                    if r > worst_r:
                        worst_r = r
                        worst_facet = simplex
                        worst_cc = cc
            
            if worst_facet is None:
                break
            
            # Move the vertices of the worst facet toward the circumcenter
            step = 0.01
            for idx in worst_facet:
                best_points[idx] = best_points[idx] + step * worst_cc
                best_points[idx] /= np.linalg.norm(best_points[idx])
            
            new_cr = _covering_radius(best_points)
            if new_cr < cr:
                best_points = best_points.copy()
            else:
                # Try smaller step
                step = 0.001
                for idx in worst_facet:
                    best_points[idx] = best_points[idx] + step * worst_cc
                    best_points[idx] /= np.linalg.norm(best_points[idx])
                new_cr = _covering_radius(best_points)
                if new_cr < cr:
                    best_points = best_points.copy()
    
    # Final check
    final_cr = _covering_radius(best_points)
    
    return best_points
# EVOLVE-BLOCK-END
