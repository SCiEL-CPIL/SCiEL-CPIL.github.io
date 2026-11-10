"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.spatial import ConvexHull

N = 130

# EVOLVE-BLOCK-START

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms


def _covering_radius(pts):
    """Compute the covering radius (in degrees) of a set of points on the unit sphere.
    
    The covering radius is the maximum over all points y on the sphere of the angular
    distance from y to its nearest point in pts. This equals the maximum circumradius
    of the facets of the convex hull of pts.
    """
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        # Fallback: use nearest-neighbor grid approach
        # Sample many points on sphere
        theta = np.linspace(0, np.pi, 200)
        phi = np.linspace(0, 2 * np.pi, 400)
        TH, PH = np.meshgrid(theta, phi, indexing='ij')
        y = np.column_stack([
            np.sin(TH) * np.cos(PH).ravel(),
            np.sin(TH) * np.sin(PH).ravel(),
            np.cos(TH).ravel()
        ])
        # For each y, find nearest pt
        # Use chunked computation to avoid memory issues
        max_dist = 0.0
        chunk = 50000
        for i in range(0, len(y), chunk):
            yc = y[i:i+chunk]
            # Compute angular distance: arccos(clamp(dot(y, pts), -1, 1))
            dots = yc @ pts.T  # shape (chunk, N)
            dots = np.clip(dots, -1.0, 1.0)
            dists = np.arccos(dots)
            max_dist = max(max_dist, np.max(dists))
        return np.degrees(max_dist)
    
    max_r = 0.0
    for simplex in hull.simplices:
        # simplex is a set of 3 indices (for 3D, facets are triangles)
        # Compute circumradius of the spherical triangle
        # The covering radius for a facet is the angular distance from the center of the
        # spherical cap (the point equidistant from the three vertices) to the vertices.
        # For a spherical triangle with vertices v1, v2, v3, the "circumcenter" in terms
        # of covering is the point on the sphere that maximizes the minimum distance to
        # the three vertices. For a flat triangle, this is the circumcenter. For spherical,
        # we need the spherical circumcenter.
        
        # Simple approach: the circumradius of the facet equals the maximum angular distance
        # from any point in the spherical triangle to its nearest vertex. For small triangles,
        # this is approximately the circumradius of the planar triangle formed by the three
        # points.
        
        # Better: compute the circumcenter of the three points on the sphere.
        # The circumcenter c satisfies |c - v1| = |c - v2| = |c - v3| in 3D (Euclidean),
        # and |c| = 1.
        
        v1, v2, v3 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
        
        # The circumcenter of three points on the unit sphere lies at the intersection of
        # the unit sphere with the plane equidistant from all three.
        # Equivalently, it's the normalization of the vector:
        # c ∝ (v1 - v2) × (v1 - v3) ... no, that's the normal to the plane through v1,v2,v3.
        
        # Actually, the circumcenter in 3D (Euclidean) of three points on the unit sphere:
        # The three points lie on a plane. The circumcenter of the triangle in that plane,
        # projected to the sphere, gives us what we want.
        
        # Let's compute the Euclidean circumcenter of the triangle v1,v2,v3.
        # Then normalize it to get the spherical circumcenter.
        
        # Circumcenter formula:
        # a = |v2 - v3|, b = |v1 - v3|, c = |v1 - v2|
        # Using the formula based on barycentric coordinates or the cross product method.
        
        d = 2.0 * (
            v1[0] * (v2[1] * v3[2] - v2[2] * v3[1]) -
            v1[1] * (v2[0] * v3[2] - v2[2] * v3[0]) +
            v1[2] * (v2[0] * v3[1] - v2[1] * v3[0])
        )
        if abs(d) < 1e-12:
            # Degenerate: use centroid
            c = (v1 + v2 + v3) / 3.0
            c = c / np.linalg.norm(c)
            r = np.max([
                np.arccos(np.clip(np.dot(c, v1), -1, 1)),
                np.arccos(np.clip(np.dot(c, v2), -1, 1)),
                np.arccos(np.clip(np.dot(c, v3), -1, 1))
            ])
            max_r = max(max_r, r)
            continue
        
        a2 = np.dot(v2 - v3, v2 - v3)
        b2 = np.dot(v1 - v3, v1 - v3)
        c2 = np.dot(v1 - v2, v1 - v2)
        
        # Circumcenter in the plane:
        # A = v1, B = v2, C = v3
        # D = 2*(A*(Bx*Cy - By*Cx) + B*(Cx*Ay - Cx*By) + C*(Bx*Ay - Bx*By))  -- wait, let me use a cleaner formula
        
        # Using the standard formula:
        # Let A=v1, B=v2, C=v3
        # D = 2*(Ax*(By-Cy) + Bx*(Cy-Ay) + Cx*(Ay-By))
        Ax, Ay, Az = v1
        Bx, By, Bz = v2
        Cx, Cy, Cz = v3
        
        D = 2.0 * (Ax * (By - Cy) + Bx * (Cy - Ay) + Cx * (Ay - By))
        if abs(D) < 1e-12:
            c = (v1 + v2 + v3) / 3.0
            c = c / np.linalg.norm(c)
            r = np.max([
                np.arccos(np.clip(np.dot(c, v1), -1, 1)),
                np.arccos(np.clip(np.dot(c, v2), -1, 1)),
                np.arccos(np.clip(np.dot(c, v3), -1, 1))
            ])
            max_r = max(max_r, r)
            continue
        
        # Ux = ((Ax^2+Ay^2+Az^2)*(By-Cy) + (Bx^2+By^2+Bz^2)*(Cy-Ay) + (Cx^2+Cy^2+Cz^2)*(Ay-By)) / D
        # Since all on unit sphere, Ax^2+Ay^2+Az^2 = 1, etc.
        Ux = ((By - Cy) + (Bx*Bx + By*By + Bz*Bz) * (Cy - Ay) + (Cx*Cx + Cy*Cy + Cz*Cz) * (Ay - By)) / D
        Uy = ((Az - Cz) + (Bx*Bx + By*By + Bz*Bz) * (Cz - Az) + (Cx*Cx + Cy*Cy + Cz*Cz) * (Az - Cz)) / D
        Uz = ((Ay - By) + (Bx*Bx + By*By + Bz*Bz) * (By - Ay) + (Cx*Cx + Cy*Cy + Cz*Cz) * (Ay - By)) / D
        
        # Wait, I'm making errors. Let me use a more robust approach.
        # The circumcenter of three points in 3D:
        # Solve: |c - v1|^2 = |c - v2|^2 = |c - v3|^2
        # => 2*c·(v2-v1) = |v2|^2 - |v1|^2 = 0 (since unit)
        # => 2*c·(v3-v1) = 0
        # So c is perpendicular to (v2-v1) and (v3-v1), and |c|=1.
        # That means c is along the cross product (v2-v1) × (v3-v1), normalized.
        
        n = np.cross(v2 - v1, v3 - v1)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-12:
            c = (v1 + v2 + v3) / 3.0
            c = c / np.linalg.norm(c)
            r = np.max([
                np.arccos(np.clip(np.dot(c, v1), -1, 1)),
                np.arccos(np.clip(np.dot(c, v2), -1, 1)),
                np.arccos(np.clip(np.dot(c, v3), -1, 1))
            ])
            max_r = max(max_r, r)
            continue
        
        c = n / n_norm
        # But wait, this gives a point equidistant from v1, v2, v3 in Euclidean distance,
        # but it might be on the "wrong" side of the sphere. We need the one that's
        # on the same side as the triangle (the one that's inside the spherical cap).
        # Check: the centroid should be closer to c than to -c.
        centroid = (v1 + v2 + v3) / 3.0
        if np.dot(c, centroid) < 0:
            c = -c
        
        r1 = np.arccos(np.clip(np.dot(c, v1), -1, 1))
        r2 = np.arccos(np.clip(np.dot(c, v2), -1, 1))
        r3 = np.arccos(np.clip(np.dot(c, v3), -1, 1))
        r = max(r1, r2, r3)
        max_r = max(max_r, r)
    
    return np.degrees(max_r)


def _hill_climb(pts, max_iter=5000, step_size=0.1, time_limit=None, start_time=None):
    """Local optimization via hill-climbing to minimize the covering radius."""
    if start_time is None:
        start_time = time.time()
    pts = _normalize(pts).copy()
    current_cr = _covering_radius(pts)
    
    for it in range(max_iter):
        if time_limit is not None and (time.time() - start_time) > time_limit:
            break
        
        # Pick a random point to move
        idx = np.random.randint(N)
        original = pts[idx].copy()
        
        # Generate a random perturbation on the sphere
        # Use a small random direction
        direction = np.random.randn(3)
        direction /= np.linalg.norm(direction)
        
        # Try several step sizes
        improved = False
        for step in [step_size, step_size * 0.5, step_size * 0.1]:
            # Move the point
            new_pts = pts.copy()
            new_pts[idx] = pts[idx] + step * direction
            new_pts[idx] = new_pts[idx] / np.linalg.norm(new_pts[idx])
            
            new_cr = _covering_radius(new_pts)
            if new_cr < current_cr:
                pts = new_pts
                current_cr = new_cr
                improved = True
                break
        
        # Also try random directions
        if not improved:
            for _ in range(3):
                direction = np.random.randn(3)
                direction /= np.linalg.norm(direction)
                step = step_size * 0.5
                new_pts = pts.copy()
                new_pts[idx] = pts[idx] + step * direction
                new_pts[idx] = new_pts[idx] / np.linalg.norm(new_pts[idx])
                new_cr = _covering_radius(new_pts)
                if new_cr < current_cr:
                    pts = new_pts
                    current_cr = new_cr
                    break
    
    return pts, current_cr


def _fibonacci_spiral(n):
    """Generate points on the unit sphere using the golden-angle spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 140s out of 150s budget
    
    # Start with Fibonacci spiral
    pts = _fibonacci_spiral(N)
    best_pts = pts.copy()
    best_cr = _covering_radius(pts)
    
    # Perform hill-climbing optimization
    pts, cr = _hill_climb(pts, max_iter=3000, step_size=0.15, time_limit=deadline - 10, start_time=start_time)
    if cr < best_cr:
        best_cr = cr
        best_pts = pts.copy()
    
    # If time remains, do more optimization with smaller steps
    if time.time() < deadline - 10:
        pts, cr = _hill_climb(best_pts, max_iter=3000, step_size=0.05, time_limit=deadline - 5, start_time=start_time)
        if cr < best_cr:
            best_cr = cr
            best_pts = pts.copy()
    
    # Final normalization
    best_pts = _normalize(best_pts)
    return best_pts

# EVOLVE-BLOCK-END
