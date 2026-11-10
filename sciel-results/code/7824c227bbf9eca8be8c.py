"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull, Delaunay
import time
import math


def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms


def _covering_radius(pts):
    """
    Compute the covering radius in degrees.
    The covering radius is the maximum over all points y on the sphere of the
    angular distance from y to the nearest point in pts.
    
    For a set of points on the sphere, the covering radius equals the largest
    circumradius of the facets of the convex hull that correspond to empty caps.
    We compute this via the Delaunay triangulation of the points on the sphere.
    """
    pts = _normalize(pts)
    
    # Compute convex hull
    try:
        hull = ConvexHull(pts)
    except Exception:
        return 180.0  # Invalid configuration
    
    # Get the facets (simplices) of the hull
    # For 3D points, facets are triangles
    simplices = hull.simplices
    
    # For each triangular facet, compute the circumradius of the spherical cap
    # The covering radius is the maximum over all facets of the angular distance
    # from the facet's circumcenter (on the sphere) to the nearest vertex.
    
    max_circumradius = 0.0
    for simplex in simplices:
        a, b, c = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
        
        # Compute the normal to the triangle (pointing outward)
        ab = b - a
        ac = c - a
        normal = np.cross(ab, ac)
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-12:
            continue
        normal = normal / norm_len
        
        # The circumcenter of the spherical triangle is in the direction of the normal
        # (for a convex hull facet, the outward normal points away from the origin)
        # The angular distance from the circumcenter to each vertex is the same
        # for a spherical triangle inscribed in a circle on the sphere.
        
        # Compute the angle between the normal and each vertex
        cos_a = np.dot(normal, a)
        cos_b = np.dot(normal, b)
        cos_c = np.dot(normal, c)
        
        # The circumradius (angular distance) is arccos(cos_a) where cos_a is the
        # same for all three vertices if the triangle is inscribed in a circle
        # centered at the normal direction.
        # Actually, for a spherical triangle, the circumradius R satisfies:
        # cos(R) = (cos(a) + cos(b) + cos(c)) / (1 + 2*cos(a)*cos(b)*cos(c)) / something...
        
        # Simpler approach: the covering radius for a facet is the maximum angular
        # distance from any point in the facet's Voronoi cell to the nearest site.
        # For a Delaunay triangle, this is the circumradius of the triangle on the sphere.
        
        # Compute side lengths (angular distances)
        d_ab = np.arccos(np.clip(np.dot(a, b), -1, 1))
        d_bc = np.arccos(np.clip(np.dot(b, c), -1, 1))
        d_ca = np.arccos(np.clip(np.dot(c, a), -1, 1))
        
        # Spherical circumradius formula:
        # R = arccos((cos(a)*cos(b)*cos(c) + cos(a) + cos(b) + cos(c)) / 
        #              (1 + 2*cos(a)*cos(b)*cos(c)))
        # where a, b, c are the side lengths.
        
        cos_a_side = math.cos(d_ab)
        cos_b_side = math.cos(d_bc)
        cos_c_side = math.cos(d_ca)
        
        numerator = cos_a_side * cos_b_side * cos_c_side + cos_a_side + cos_b_side + cos_c_side
        denominator = 1.0 + 2.0 * cos_a_side * cos_b_side * cos_c_side
        
        if abs(denominator) < 1e-15:
            continue
            
        cos_R = numerator / denominator
        cos_R = np.clip(cos_R, -1.0, 1.0)
        
        circumradius = np.arccos(cos_R)
        max_circumradius = max(max_circumradius, circumradius)
    
    return np.degrees(max_circumradius)


def _min_pairwise_distance(pts):
    """Compute the minimum pairwise angular distance."""
    pts = _normalize(pts)
    n = len(pts)
    min_dist = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            d = np.arccos(np.clip(np.dot(pts[i], pts[j]), -1, 1))
            min_dist = min(min_dist, d)
    return min_dist


def _objective(x):
    """
    Objective function for optimization.
    We want to maximize the minimum pairwise distance (Tammes proxy),
    which is equivalent to minimizing the negative of it.
    """
    pts = _normalize(x.reshape(N, 3))
    return -_min_pairwise_distance(pts)


def _spherical_kmeans_init(n_clusters, n_points, rng):
    """
    Initialize n_clusters points using a spherical K-means-like approach.
    Start with random points and iterate.
    """
    # Start with random points
    pts = rng.standard_normal((n_clusters, 3))
    pts = _normalize(pts)
    
    # Generate candidate points
    candidates = rng.standard_normal((n_points, 3))
    candidates = _normalize(candidates)
    
    # Iterate K-means
    for _ in range(10):
        # Assign each candidate to nearest cluster center
        dists = np.zeros((n_points, n_clusters))
        for k in range(n_clusters):
            dists[:, k] = np.arccos(np.clip(candidates @ pts[k], -1, 1))
        
        labels = np.argmin(dists, axis=1)
        
        # Update cluster centers
        new_pts = np.zeros_like(pts)
        for k in range(n_clusters):
            mask = labels == k
            if np.sum(mask) > 0:
                new_pts[k] = np.sum(candidates[mask], axis=0)
                norm = np.linalg.norm(new_pts[k])
                if norm > 1e-12:
                    new_pts[k] /= norm
                else:
                    new_pts[k] = pts[k]
            else:
                new_pts[k] = pts[k]
        
        pts = new_pts
    
    return pts


def construct():
    """
    Construct 50 points on the unit sphere minimizing the covering radius.
    Uses multi-start optimization with L-BFGS-B on the Tammes problem proxy.
    """
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin before 150s limit
    
    best_score = float('inf')
    best_pts = None
    
    rng = np.random.default_rng(42)
    
    # Multi-start optimization
    n_restarts = 20
    
    for restart in range(n_restarts):
        if time.time() > deadline - 30:
            break
        
        # Initialize
        if restart == 0:
            # Fibonacci spiral start
            i = np.arange(N) + 0.5
            z = 1.0 - 2.0 * i / N
            phi = np.pi * (1.0 + np.sqrt(5.0)) * i
            r = np.sqrt(1.0 - z * z)
            pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        else:
            # Spherical K-means initialization
            pts = _spherical_kmeans_init(N, 500, rng)
        
        # Optimize using L-BFGS-B
        x0 = pts.flatten()
        
        try:
            result = minimize(
                _objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12}
            )
        except Exception:
            continue
        
        if result.success or result.fun < best_score:
            pts_opt = _normalize(result.x.reshape(N, 3))
            score = _covering_radius(pts_opt)
            
            if score < best_score:
                best_score = score
                best_pts = pts_opt
        
        # Check time
        if time.time() > deadline - 60:
            break
    
    # Final polish: use the best configuration found
    if best_pts is None:
        # Fallback to Fibonacci
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        best_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    else:
        # Do a final high-precision polish
        x0 = best_pts.flatten()
        try:
            result = minimize(
                _objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-14}
            )
            if result.fun < best_score:
                best_pts = _normalize(result.x.reshape(N, 3))
        except Exception:
            pass
    
    return _normalize(best_pts)
# EVOLVE-BLOCK-END
