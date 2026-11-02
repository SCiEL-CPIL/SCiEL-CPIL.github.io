"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

def _covering_radius(pts):
    """Compute the covering radius in degrees using the convex hull."""
    try:
        hull = ConvexHull(pts)
        # For each facet, compute the circumradius (in radians)
        # The covering radius is the max over all facets of the circumradius
        # For a spherical cap defined by a facet, the angular distance from
        # the facet center to any point on the cap is at most the circumradius
        
        # Get facet vertices
        facets = hull.simplices  # shape (F, 3) for 3D
        
        max_radius = 0.0
        for facet in facets:
            v1, v2, v3 = pts[facet[0]], pts[facet[1]], pts[facet[2]]
            # Compute the circumcenter of the triangle on the sphere
            # The circumcenter in 3D for a spherical triangle
            # Using the formula: circumcenter = (a*sin(2A) + b*sin(2B) + c*sin(2C)) / (2*sin(A)*sin(B)*sin(C))
            # But easier: find the point equidistant from all three vertices on the sphere
            
            # Method: solve for the point on the unit sphere equidistant from v1, v2, v3
            # This means the circumcenter lies in the plane of the triangle
            # The circumcenter in 3D (normalized) can be found by:
            # c = (|v2|^2*(v3-v1) x (v3-v2) + |v3|^2*(v1-v2) x (v1-v3) + |v1|^2*(v2-v1) x (v2-v3)) / (2 * |...|)
            # But since all are unit vectors, |vi|^2 = 1
            
            # Simpler approach: the circumcenter of the spherical triangle
            # is along the direction of the cross product of (v2-v1) and (v3-v1)
            # But that gives the normal to the plane, not the circumcenter
            
            # Better: use the fact that for three points on a unit sphere,
            # the circumcenter of the triangle (in 3D) projected onto the sphere
            # gives the center of the minimal cap containing the three points.
            
            # Compute the plane normal
            edge1 = v2 - v1
            edge2 = v3 - v1
            normal = np.cross(edge1, edge2)
            norm_normal = np.linalg.norm(normal)
            if norm_normal < 1e-12:
                continue
            normal = normal / norm_normal
            
            # The circumcenter in the plane of the triangle
            # For a triangle with vertices on a unit sphere, the circumcenter
            # (center of the circle through the three points) can be computed as:
            # c = v1 + t1*edge1 + t2*edge2
            # where t1, t2 satisfy:
            # |v1 + t1*e1 + t2*e2|^2 = 1
            # (v1 + t1*e1 + t2*e2) . e1 = 0.5*|e1|^2 (if e1 is the chord)
            # Actually, let's use a direct formula.
            
            # The circumcenter of three points on the unit sphere:
            # a = v1, b = v2, c = v3
            # The circumcenter O satisfies:
            # |O-a|^2 = |O-b|^2 = |O-c|^2 and |O| = 1
            # This gives:
            # 2*O.(b-a) = |b|^2 - |a|^2 = 0 (since all unit)
            # 2*O.(c-a) = |c|^2 - |a|^2 = 0
            # So O is perpendicular to (b-a) and (c-a), meaning O is along normal!
            # But O must be on the unit sphere and in the plane... 
            # Wait, O is in the plane of a,b,c, so O = a + s*(b-a) + t*(c-a)
            # And O . (b-a) = 0.5*|b-a|^2? No.
            # |O-a|^2 = |O-b|^2 => 2*O.(b-a) = |b|^2 - |a|^2 = 0
            # So O is perpendicular to b-a in the sense that O.(b-a) = 0? 
            # That can't be right since O is in the plane.
            
            # Let me redo: |O-a|^2 = |O-b|^2
            # O.O - 2O.a + a.a = O.O - 2O.b + b.b
            # -2O.a + 1 = -2O.b + 1
            # O.(b-a) = 0
            # Similarly O.(c-a) = 0
            # So O is perpendicular to both (b-a) and (c-a), hence O is parallel to normal.
            # O = ±normal (since |O|=1)
            # But O must be in the plane of a,b,c. The plane has normal 'normal',
            # so any point in the plane satisfies (x-a).normal = 0.
            # If O = normal, then (normal - a).normal = 1 - a.normal
            # For O to be in the plane: 1 - a.normal = 0, i.e., a.normal = 1
            # That's not generally true.
            
            # I think I made an error. Let me reconsider.
            # The circumcenter of the three points as seen from the origin...
            # Actually, for the covering radius, we need the largest angular distance
            # from any point on the sphere to the nearest configuration point.
            # This is determined by the Delaunay triangulation on the sphere.
            # The worst point is at the circumcenter of a Delaunay triangle.
            
            # For three points on the unit sphere, the circumcenter of the spherical
            # triangle (the point equidistant from all three on the sphere) can be
            # found as follows:
            # The three points define a plane. The line from the origin perpendicular
            # to this plane intersects the sphere at two points. One of these is the
            # circumcenter if the triangle is acute.
            
            # Actually, the circumcenter of a spherical triangle is the intersection
            # of the perpendicular bisectors of the sides (great circle arcs).
            
            # For our purpose, a simpler approach: the circumradius of the facet
            # is the angular distance from the facet's circumcenter to any vertex.
            # The covering radius is the max of these circumradii over all facets
            # (for the convex hull, assuming the hull is correct).
            
            # Let's compute the circumcenter properly.
            # For three unit vectors a, b, c, the circumcenter is:
            # O = (a + b + c) normalized? No, that's the centroid.
            
            # Correct formula: The circumcenter of triangle abc on the unit sphere
            # is given by:
            # O = (a*|b-c|^2 + b*|c-a|^2 + c*|a-b|^2) / |...|
            # This is the circumcenter in the plane of the triangle.
            
            d_bc = np.linalg.norm(b - c) ** 2
            d_ca = np.linalg.norm(c - a) ** 2
            d_ab = np.linalg.norm(a - b) ** 2
            
            O = (a * d_bc + b * d_ca + c * d_ab)
            O_norm = np.linalg.norm(O)
            if O_norm < 1e-12:
                continue
            O = O / O_norm
            
            # The circumradius is the angular distance from O to any vertex
            # e.g., angle between O and a
            cos_angle = np.clip(np.dot(O, a), -1.0, 1.0)
            radius = np.arccos(cos_angle)
            max_radius = max(max_radius, radius)
        
        return np.degrees(max_radius)
    except Exception:
        return np.inf


def _objective(pts_flat):
    """Objective function: covering radius in degrees."""
    pts = pts_flat.reshape(-1, 3)
    # Normalize
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    pts = pts / norms
    return _covering_radius(pts)


def _init_fibonacci():
    """Initialize with Fibonacci spiral points."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def construct():
    """Construct 130 points on the unit sphere minimizing the covering radius."""
    start_time = time.time()
    deadline = start_time + 140.0  # 10 second safety margin
    
    # Initialize
    best_pts = _init_fibonacci()
    best_radius = _covering_radius(best_pts)
    
    # Local optimization using gradient-based methods
    # We'll use the fact that the covering radius is a piecewise smooth function.
    # Each facet defines a region, and within each region, the objective is the
    # circumradius of that facet.
    
    # For local optimization, we can try to move each point to reduce the
    # circumradius of the facets it belongs to.
    
    # Simple approach: iterative local search
    # For each point, try small perturbations and keep the best.
    
    max_iters = 500
    step_size = 0.05
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        improved = False
        for i in range(N):
            if time.time() > deadline:
                break
            
            orig = best_pts[i].copy()
            best_local = best_pts[i].copy()
            best_local_radius = best_radius
            
            # Try perturbations in random directions
            for _ in range(10):
                # Random direction on the sphere
                rand_dir = np.random.randn(3)
                rand_dir = rand_dir / np.linalg.norm(rand_dir)
                
                # Move point slightly
                new_pt = best_pts[i] + step_size * rand_dir
                new_pt = new_pt / np.linalg.norm(new_pt)
                
                test_pts = best_pts.copy()
                test_pts[i] = new_pt
                
                test_radius = _covering_radius(test_pts)
                if test_radius < best_local_radius:
                    best_local_radius = test_radius
                    best_local = new_pt
            
            if best_local_radius < best_radius:
                best_pts[i] = best_local
                best_radius = best_local_radius
                improved = True
        
        if not improved:
            # Reduce step size
            step_size *= 0.9
            if step_size < 1e-4:
                break
    
    # Final normalization
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    best_pts = best_pts / norms
    
    return best_pts
# EVOLVE-BLOCK-END
