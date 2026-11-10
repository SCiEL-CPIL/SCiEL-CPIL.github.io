"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize


def _covering_radius(pts):
    """Compute covering radius in radians. Returns (radius, worst_facet_index, worst_point)."""
    try:
        hull = ConvexHull(pts)
        # For each facet, compute the circumcenter (point on sphere farthest from all facet vertices
        # but within the facet's cap). The covering radius is the max over facets of the angular
        # distance from the facet's circumcenter to the nearest site.
        
        # Actually, the covering radius is the maximum over all points on the sphere of the distance
        # to the nearest site. This equals the maximum over all Voronoi cells of the distance to
        # the cell's farthest point. For a convex hull, the Voronoi cell of a vertex corresponds
        # to the region of the sphere closest to that vertex. The farthest point in a Voronoi cell
        # is the circumcenter of the corresponding dual facet (in the dual sense).
        
        # Simpler approach: for each facet of the convex hull, the circumcenter of that facet (projected
        # onto the unit sphere) is a candidate worst point. The covering radius is the max angular distance
        # from such circumcenters to the nearest site.
        
        # Compute circumcenters of each facet
        n_sites = len(pts)
        max_ang = 0.0
        worst_facet = -1
        worst_pt = None
        
        for fi in range(len(hull.simplices)):
            tri = hull.simplices[fi]
            a, b, c = pts[tri[0]], pts[tri[1]], pts[tri[2]]
            
            # Circumcenter of triangle abc on the unit sphere
            # The circumcenter is the point equidistant from a, b, c
            # Solve: |x|^2 = 1, x·a = x·b = x·c
            # This means x is in the direction of (a × b) × (b × c) normalized... 
            # Actually, the circumcenter of three points on a sphere is the intersection of the sphere
            # with the plane equidistant from all three. 
            # For three points on the unit sphere, the circumcenter is proportional to:
            # a * (|b|^2 |c|^2 - (b·c)^2) + b * (|a|^2 |c|^2 - (a·c)^2) + c * (|a|^2 |b|^2 - (a·b)^2)
            # But since all are unit, this simplifies.
            # Alternative: the circumcenter direction is (a - b) × (a - c) normalized? No.
            
            # Better: find x such that x·a = x·b = x·c and |x|=1
            # x·(a-b) = 0, x·(a-c) = 0, so x is parallel to (a-b) × (a-c)
            # But we need to check if this gives the correct circumcenter (it gives the normal to the plane through a,b,c)
            # The circumcenter of the spherical triangle is the point on the sphere equidistant from a, b, c.
            # It lies in the direction of the cross product of (b-a) and (c-a) only if the triangle is right...
            
            # Let me use a more robust method:
            # The circumcenter of three points on the unit sphere can be found as follows:
            # Let v = (b - a) × (c - a), normalized. This is the normal to the plane.
            # The circumcenter is v * cos(alpha) + w * sin(alpha) where w is in the plane...
            
            # Simpler: use the formula for circumcenter of three points in 3D projected to sphere.
            # The circumcenter of triangle abc (in 3D) projected to the unit sphere.
            # 3D circumcenter: solve |x - a|^2 = |x - b|^2 = |x - c|^2
            # This gives x = a + t1*(b-a) + t2*(c-a) where t1, t2 solve a linear system.
            
            ab = b - a
            ac = c - a
            bc = c - b
            
            # |x-a|^2 = |x-b|^2 => 2*x·(b-a) = |b|^2 - |a|^2 = 0 (since unit)
            # Wait, that's only if |a|=|b|=1. Yes they are unit.
            # So x·(b-a) = 0 and x·(c-a) = 0
            # So x is parallel to (b-a) × (c-a)
            
            v = np.cross(b - a, c - a)
            nv = np.linalg.norm(v)
            if nv < 1e-12:
                continue
            v = v / nv
            
            # Now x = ±v. We need the one that is on the same side as the origin... 
            # Actually for points on the unit sphere, the circumcenter of the spherical triangle
            # (the one inside the triangle) should be the one where x·a > 0 (if the triangle is small).
            # Let's check: if a, b, c are close together, the circumcenter should be near them, so x·a > 0.
            if v[0]*a[0] + v[1]*a[1] + v[2]*a[2] < 0:
                v = -v
            
            # Verify: is v actually equidistant? Due to numerical issues, let's recompute.
            # Actually, for unit vectors, x·(b-a)=0 means x·b = x·a, so yes equidistant.
            
            # Angular distance from v to nearest site
            dists = np.arccos(np.clip(pts @ v, -1.0, 1.0))
            d = np.min(dists)
            if d > max_ang:
                max_ang = d
                worst_facet = fi
                worst_pt = v.copy()
        
        return max_ang, worst_facet, worst_pt
    except Exception:
        return np.pi, -1, None


def _lloyd_step(pts, n_iter=50):
    """Perform Lloyd's algorithm iterations to minimize covering radius."""
    pts = pts.copy()
    for _ in range(n_iter):
        # For each site, find its Voronoi cell (region of sphere closest to it)
        # and move it to the centroid of that cell (weighted by area, approximated by
        # the mean of the facet circumcenters that are in its cell).
        
        # Simpler Lloyd: for each site, find the set of facet circumcenters that are
        # closest to it, and move the site toward their average.
        
        # Compute all facet circumcenters
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
        
        circumcenters = []
        for fi in range(len(hull.simplices)):
            tri = hull.simplices[fi]
            a, b, c = pts[tri[0]], pts[tri[1]], pts[tri[2]]
            v = np.cross(b - a, c - a)
            nv = np.linalg.norm(v)
            if nv < 1e-12:
                continue
            v = v / nv
            if np.dot(v, a) < 0:
                v = -v
            circumcenters.append(v)
        
        if not circumcenters:
            break
        
        circumcenters = np.array(circumcenters)
        
        # For each site, find circumcenters closest to it
        new_pts = pts.copy()
        for i in range(len(pts)):
            dists = np.linalg.norm(circumcenters - pts[i], axis=1)
            # Take the closest few circumcenters
            k = min(5, len(circumcenters))
            idx = np.argsort(dists)[:k]
            centroid = circumcenters[idx].mean(axis=0)
            nc = np.linalg.norm(centroid)
            if nc > 1e-10:
                centroid = centroid / nc
                # Blend to avoid large jumps
                new_pts[i] = 0.5 * pts[i] + 0.5 * centroid
                new_pts[i] /= np.linalg.norm(new_pts[i])
        
        pts = new_pts
    
    return pts


def _local_optimize(pts, n_iter=200):
    """Local optimization targeting the worst facet."""
    pts = pts.copy()
    for it in range(n_iter):
        radius, wf, wpt = _covering_radius(pts)
        if wf < 0:
            break
        
        # Get the three sites of the worst facet
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
        
        tri = hull.simplices[wf]
        a_idx, b_idx, c_idx = tri[0], tri[1], tri[2]
        
        # Move the three sites of the worst facet toward the worst point
        # to reduce the circumradius
        alpha = 0.05  # step size
        for idx in [a_idx, b_idx, c_idx]:
            # Move toward wpt
            new_pt = pts[idx] + alpha * (wpt - pts[idx])
            # Project back to sphere
            new_pt /= np.linalg.norm(new_pt)
            pts[idx] = new_pt
    
    return pts


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Leave 20s margin
    
    # Initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_pts = pts.copy()
    best_radius, _, _ = _covering_radius(pts)
    
    # Phase 1: Lloyd's algorithm for global improvement
    while time.time() < deadline - 30.0:
        pts_lloyd = _lloyd_step(pts, n_iter=20)
        r_lloyd, _, _ = _covering_radius(pts_lloyd)
        if r_lloyd < best_radius:
            best_radius = r_lloyd
            best_pts = pts_lloyd.copy()
            pts = pts_lloyd.copy()
        else:
            # If no improvement, try with fewer iterations
            pts_lloyd = _lloyd_step(pts, n_iter=5)
            r_lloyd, _, _ = _covering_radius(pts_lloyd)
            if r_lloyd < best_radius:
                best_radius = r_lloyd
                best_pts = pts_lloyd.copy()
                pts = pts_lloyd.copy()
            else:
                break
    
    # Phase 2: Local optimization targeting worst facets
    while time.time() < deadline - 10.0:
        pts_local = _local_optimize(pts, n_iter=50)
        r_local, _, _ = _covering_radius(pts_local)
        if r_local < best_radius:
            best_radius = r_local
            best_pts = pts_local.copy()
            pts = pts_local.copy()
        else:
            # Try smaller steps
            pts_local = _local_optimize(pts, n_iter=20)
            r_local, _, _ = _covering_radius(pts_local)
            if r_local < best_radius:
                best_radius = r_local
                best_pts = pts_local.copy()
                pts = pts_local.copy()
            else:
                break
    
    # Phase 3: Final fine-tuning with small perturbations
    while time.time() < deadline - 5.0:
        # Perturb the worst facet sites slightly
        radius, wf, wpt = _covering_radius(pts)
        if wf < 0:
            break
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
        tri = hull.simplices[wf]
        
        improved = False
        for idx in tri:
            # Try small random perturbations
            for _ in range(10):
                delta = np.random.randn(3) * 0.01
                new_pt = pts[idx] + delta
                new_pt /= np.linalg.norm(new_pt)
                trial = pts.copy()
                trial[idx] = new_pt
                r_trial, _, _ = _covering_radius(trial)
                if r_trial < best_radius:
                    best_radius = r_trial
                    best_pts = trial.copy()
                    pts = trial.copy()
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            break
    
    # Ensure output is valid
    pts_out = best_pts.copy()
    # Normalize
    norms = np.linalg.norm(pts_out, axis=1)
    pts_out = pts_out / norms[:, np.newaxis]
    
    # Check for coincident points
    for i in range(N):
        for j in range(i+1, N):
            if np.linalg.norm(pts_out[i] - pts_out[j]) < 1e-8:
                # Separate them slightly
                pts_out[j] += np.random.randn(3) * 0.01
                pts_out[j] /= np.linalg.norm(pts_out[j])
    
    return pts_out
# EVOLVE-BLOCK-END
