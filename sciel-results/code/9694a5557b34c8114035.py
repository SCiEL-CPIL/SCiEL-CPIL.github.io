"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def _normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v
    return v / n

def _covering_radius(points):
    """
    Compute the covering radius (in degrees) for a set of points on the unit sphere.
    The covering radius is the maximum angular distance from any point on the sphere
    to its nearest point in the set. This is equivalent to the largest circumradius
    of the facets of the convex hull of the points (each facet is a triangle on the sphere).
    """
    points = np.array(points)
    points = np.array([_normalize(p) for p in points])
    
    # Compute convex hull
    try:
        hull = ConvexHull(points)
    except Exception:
        # Fallback: if convex hull fails, use a simple grid-based estimate
        # This is a rough fallback
        theta = np.linspace(0, np.pi, 50)
        phi = np.linspace(0, 2*np.pi, 100)
        TT, PP = np.meshgrid(theta, phi)
        grid = np.column_stack([
            np.sin(TT) * np.cos(PP).ravel(),
            np.sin(TT) * np.sin(PP).ravel(),
            np.cos(TT).ravel()
        ])
        # For each grid point, find nearest point
        dists = []
        for g in grid:
            d = np.arccos(np.clip(np.dot(points, g), -1.0, 1.0))
            dists.append(np.min(d))
        return np.max(dists) * 180.0 / np.pi
    
    # For each facet (triangle) of the convex hull, compute the circumradius
    # The circumradius of a spherical triangle is the angular distance from the
    # circumcenter to any vertex. For a triangle on the unit sphere with vertices
    # a, b, c, the circumradius can be computed from the side lengths.
    
    max_radius = 0.0
    for simplex in hull.simplices:
        a, b, c = points[simplex[0]], points[simplex[1]], points[simplex[2]]
        
        # Compute side lengths (angular distances)
        ab = np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))
        bc = np.arccos(np.clip(np.dot(b, c), -1.0, 1.0))
        ca = np.arccos(np.clip(np.dot(c, a), -1.0, 1.0))
        
        # Circumradius formula for spherical triangle:
        # R = (a * b * c) / (4 * area) where area is the spherical excess
        # Or use: sin(R) = (sin(a/2) * sin(b/2) * sin(c/2)) / (sin(s/2) * sin(s-a)/2 * sin(s-b)/2 * sin(s-c)/2)
        # Simpler: use the formula R = (abc)/(4K) where K is the area of the triangle
        # Actually, for a spherical triangle with sides a, b, c, the circumradius R satisfies:
        # sin(R) = (sin(a) * sin(b) * sin(c)) / (4 * sin(s) * sin(s-a) * sin(s-b) * sin(s-c)) where s = (a+b+c)/2
        # But a more direct approach: the circumcenter is the point equidistant from a, b, c.
        
        # Let's use a different approach: the circumradius is the distance from the
        # circumcenter to any vertex. The circumcenter of a spherical triangle can be
        # found by solving for a point x such that x·a = x·b = x·c (since all distances
        # are equal, the dot products with the vertices are equal).
        
        # Set up: x·(a-b) = 0 and x·(b-c) = 0, with ||x|| = 1
        # This is a linear system in 3 variables with the constraint ||x||=1.
        
        A = np.array([a - b, b - c])
        # x is orthogonal to both (a-b) and (b-c), so x is parallel to (a-b) × (b-c)
        cross = np.cross(a - b, b - c)
        if np.linalg.norm(cross) < 1e-12:
            # Degenerate triangle
            R = max(ab, bc, ca) / 2.0
        else:
            x = cross / np.linalg.norm(cross)
            # Check which side; the circumcenter could be x or -x
            # The circumcenter should be on the same side as the triangle
            # Actually, for a spherical triangle, the circumcenter is the point
            # equidistant from all three vertices. Let's compute the distance from
            # x to a, and also from -x to a, and pick the one that gives a valid
            # circumradius (should be <= pi/2 for a cap).
            
            d1 = np.arccos(np.clip(np.dot(x, a), -1.0, 1.0))
            d2 = np.arccos(np.clip(np.dot(-x, a), -1.0, 1.0))
            
            # The circumradius is the smaller of the two (the cap should be small)
            R = min(d1, d2)
        
        max_radius = max(max_radius, R)
    
    return max_radius * 180.0 / np.pi

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = points.copy()
    best_radius = _covering_radius(points)
    
    # Iterative optimization: repeatedly find the worst facet, then optimize
    # the three vertices of that facet to reduce the circumradius.
    
    max_iter = 500
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        
        # Compute covering radius and identify the worst facet
        try:
            hull = ConvexHull(points)
        except Exception:
            break
        
        max_radius = 0.0
        worst_simplex = None
        for simplex in hull.simplices:
            a, b, c = points[simplex[0]], points[simplex[1]], points[simplex[2]]
            ab = np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))
            bc = np.arccos(np.clip(np.dot(b, c), -1.0, 1.0))
            ca = np.arccos(np.clip(np.dot(c, a), -1.0, 1.0))
            
            cross = np.cross(a - b, b - c)
            if np.linalg.norm(cross) < 1e-12:
                R = max(ab, bc, ca) / 2.0
            else:
                x = cross / np.linalg.norm(cross)
                d1 = np.arccos(np.clip(np.dot(x, a), -1.0, 1.0))
                d2 = np.arccos(np.clip(np.dot(-x, a), -1.0, 1.0))
                R = min(d1, d2)
            
            if R > max_radius:
                max_radius = R
                worst_simplex = simplex
        
        if worst_simplex is None:
            break
        
        # Optimize the three vertices of the worst simplex
        idx = list(worst_simplex)
        
        def objective(params):
            new_points = points.copy()
            new_points[idx[0]] = params[0:3]
            new_points[idx[1]] = params[3:6]
            new_points[idx[2]] = params[6:9]
            # Normalize
            new_points = np.array([_normalize(p) for p in new_points])
            return _covering_radius(new_points)
        
        x0 = np.concatenate([points[idx[0]], points[idx[1]], points[idx[2]]])
        
        try:
            result = minimize(objective, x0, method='Nelder-Mead', 
                             options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6})
            if result.fun < best_radius:
                new_points = points.copy()
                new_points[idx[0]] = result.x[0:3]
                new_points[idx[1]] = result.x[3:6]
                new_points[idx[2]] = result.x[6:9]
                new_points = np.array([_normalize(p) for p in new_points])
                best_points = new_points
                best_radius = result.fun
                points = new_points
        except Exception:
            pass
    
    # Final normalization
    best_points = np.array([_normalize(p) for p in best_points])
    return best_points
# EVOLVE-BLOCK-END
