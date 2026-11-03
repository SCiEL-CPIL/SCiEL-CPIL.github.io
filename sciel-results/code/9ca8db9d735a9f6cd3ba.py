"""Covering the sphere with 130 equal caps."""
import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from scipy.optimize import minimize
import time

N = 130
TIME_LIMIT = 140.0  # safety margin before 150s kill


def _covering_radius(points):
    """Compute the covering radius in radians from the convex hull facets."""
    points = np.asarray(points, dtype=np.float64)
    # Ensure points are on unit sphere
    norms = np.linalg.norm(points, axis=1)
    points = points / norms[:, None]
    
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi  # fallback: worst case
    
    # For each triangular facet, compute the circumcenter and circumradius
    # The covering radius is the max over all facets of the angular distance
    # from the facet circumcenter to its three vertices (if circumcenter is on the sphere side)
    # Actually, for a spherical Voronoi cell, the farthest point is either a vertex of the cell
    # or the circumcenter of a facet if it's "outside" the cell.
    # 
    # The correct approach: for each facet (triangle) of the convex hull,
    # the farthest point in the corresponding spherical Voronoi cell is:
    # - If the facet's circumcenter (on the sphere) is inside the Voronoi cell, use circumradius
    # - Otherwise, the farthest point is one of the Voronoi vertices
    #
    # Simpler correct method: the covering radius equals the maximum over all
    # Voronoi vertices of the angular distance to the nearest point, AND
    # the maximum circumradius of hull facets whose circumcenter is in the "correct" hemisphere.
    #
    # Most robust: compute the spherical Delaunay/Voronoi and find max distance.
    # But simplest correct: for each hull facet, compute the point on the sphere
    # farthest from all three facet vertices (which is the circumcenter if the
    # triangle is acute, or the midpoint of the longest side if obtuse).
    
    max_r = 0.0
    for simplex in hull.simplices:
        v0, v1, v2 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
        # Circumcenter of triangle on unit sphere
        # The circumcenter is along the normal of the plane through v0,v1,v2
        # pointing toward the origin side... actually away from origin for convex hull.
        # For a convex hull of points on the unit sphere, the facet normal points outward.
        # The circumcenter on the sphere is in the direction of the normal.
        
        # Compute normal
        e1 = v1 - v0
        e2 = v2 - v0
        normal = np.cross(e1, e2)
        n = np.linalg.norm(normal)
        if n < 1e-12:
            continue
        normal = normal / n
        
        # The circumcenter on the unit sphere: project to sphere
        # It's the point c on the unit sphere equidistant from v0, v1, v2
        # c = normal / |normal| scaled... 
        # Actually, the circumcenter in 3D of the triangle is:
        # It lies on the line from origin through the normal direction.
        # For points on unit sphere, the circumcenter on the sphere is:
        # c = normal (already unit) if the triangle is "acute" on the sphere
        
        # Check: the circumradius is the angle between c and v0
        # But we need to verify c is the correct point (the one farthest from all points)
        
        # The angular circumradius:
        cos_r = np.dot(c := normal, v0)
        cos_r = np.clip(cos_r, -1.0, 1.0)
        r = np.arccos(cos_r)
        
        # But we must check that this circumcenter is actually the farthest point
        # in the Voronoi cell. If the triangle is obtuse, the farthest point
        # is the midpoint of the longest edge, not the circumcenter.
        
        # Check if circumcenter is valid: it should be farther from all three
        # vertices than any other point in the cell. A necessary condition is
        # that the circumcenter is in the same hemisphere as the facet normal
        # (which it is by construction) and that the angles at each vertex
        # of the spherical triangle are < 90 degrees (acute triangle).
        
        # Simpler check: verify that the circumcenter is actually farther from
        # the three vertices than the midpoints of the edges.
        # If not, use the midpoint of the longest edge.
        
        # Midpoints of edges
        m01 = (v0 + v1) / np.linalg.norm(v0 + v1)
        m12 = (v1 + v2) / np.linalg.norm(v1 + v2)
        m02 = (v0 + v2) / np.linalg.norm(v0 + v2)
        
        # Distance from circumcenter to each vertex
        d_c0 = np.arccos(np.clip(np.dot(c, v0), -1, 1))
        d_c1 = np.arccos(np.clip(np.dot(c, v1), -1, 1))
        d_c2 = np.arccos(np.clip(np.dot(c, v2), -1, 1))
        d_circum = max(d_c0, d_c1, d_c2)
        
        # Distance from midpoints to their respective opposite vertex
        # For edge v0-v1, the farthest point from v0 and v1 in that direction is m01
        # Check if m01 is farther from v2 than c is from v0,v1,v2
        d_m01_to_v2 = np.arccos(np.clip(np.dot(m01, v2), -1, 1))
        d_m01_to_v0 = np.arccos(np.clip(np.dot(m01, v0), -1, 1))
        d_m01_to_v1 = np.arccos(np.clip(np.dot(m01, v1), -1, 1))
        
        d_m12_to_v0 = np.arccos(np.clip(np.dot(m12, v0), -1, 1))
        d_m12_to_v1 = np.arccos(np.clip(np.dot(m12, v1), -1, 1))
        d_m12_to_v2 = np.arccos(np.clip(np.dot(m12, v2), -1, 1))
        
        d_m02_to_v1 = np.arccos(np.clip(np.dot(m02, v1), -1, 1))
        d_m02_to_v0 = np.arccos(np.clip(np.dot(m02, v0), -1, 1))
        d_m02_to_v2 = np.arccos(np.clip(np.dot(m02, v2), -1, 1))
        
        # The covering radius for this cell is the max of:
        # - circumradius (if circumcenter is valid)
        # - distances from midpoints to the third vertex
        # Actually, the correct farthest point in a spherical triangle's Voronoi cell
        # is either the circumcenter or one of the edge midpoints.
        
        candidate = max(d_circum, d_m01_to_v2, d_m12_to_v0, d_m02_to_v1)
        # Wait, this isn't quite right either. Let me think more carefully.
        #
        # The Voronoi cell on the sphere for a set of points: the farthest point
        # from the nearest site in a cell is at a vertex of the cell (where 3+ 
        # cells meet) or at the circumcenter of a face.
        #
        # For the convex hull approach: each hull facet corresponds to a Voronoi vertex
        # (a point equidistant from 3 sites). The covering radius is the max over
        # all such Voronoi vertices of their distance to the nearest site.
        #
        # So: for each hull facet, the circumcenter IS the Voronoi vertex, and its
        # distance to the nearest of the 3 sites is the circumradius.
        # But we also need to check that this Voronoi vertex is actually part of
        # the Voronoi diagram (i.e., the 3 sites are the 3 nearest sites at that point).
        
        # For a convex hull of points on the unit sphere, each hull facet's circumcenter
        # is a valid Voronoi vertex IF the circumcenter is in the correct position.
        # The key check: the circumcenter should be such that all other points are
        # farther away. But this is expensive.
        
        # Practical approach: just use the circumradius and take the max.
        # This gives an upper bound on the covering radius.
        
        if d_circum > max_r:
            max_r = d_circum
    
    return max_r


def _covering_radius_fast(points):
    """Faster covering radius using only circumradii of hull facets."""
    points = np.asarray(points, dtype=np.float64)
    norms = np.linalg.norm(points, axis=1)
    points = points / norms[:, None]
    
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi
    
    max_r = 0.0
    for simplex in hull.simplices:
        v0, v1, v2 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
        e1 = v1 - v0
        e2 = v2 - v0
        normal = np.cross(e1, e2)
        n = np.linalg.norm(normal)
        if n < 1e-12:
            continue
        normal = normal / n
        
        cos_r = np.clip(np.dot(normal, v0), -1.0, 1.0)
        r = np.arccos(cos_r)
        if r > max_r:
            max_r = r
    
    return max_r


def _refine_gradient(points, max_iters=500, lr=0.001):
    """Gradient-based refinement using a smooth approximation of the covering radius."""
    points = np.array(points, dtype=np.float64)
    n = len(points)
    
    best_points = points.copy()
    best_cr = _covering_radius_fast(points)
    
    for it in range(max_iters):
        # Compute covering radius and identify worst facet
        norms = np.linalg.norm(points, axis=1)
        points = points / norms[:, None]
        
        try:
            hull = ConvexHull(points)
        except Exception:
            break
        
        # Find the facet with maximum circumradius
        worst_idx = 0
        worst_r = 0.0
        worst_normal = None
        worst_vertices = None
        
        for i, simplex in enumerate(hull.simplices):
            v0, v1, v2 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
            e1 = v1 - v0
            e2 = v2 - v0
            normal = np.cross(e1, e2)
            nrm = np.linalg.norm(normal)
            if nrm < 1e-12:
                continue
            normal = normal / nrm
            cos_r = np.clip(np.dot(normal, v0), -1.0, 1.0)
            r = np.arccos(cos_r)
            if r > worst_r:
                worst_r = r
                worst_idx = i
                worst_normal = normal
                worst_vertices = simplex
        
        if worst_normal is None:
            break
        
        # Move the 3 vertices of the worst facet away from the circumcenter
        # Gradient: push vertices in the direction that reduces the circumradius
        # This means moving them toward the circumcenter direction (outward)
        # Actually, to reduce the circumradius, we want to make the triangle smaller
        # on the sphere, so move vertices toward each other or toward the center.
        
        # Simple heuristic: move each vertex slightly in the direction of the
        # centroid of the other two vertices (to shrink the triangle)
        vs = worst_vertices
        for vi in range(3):
            idx = vs[vi]
            other1 = vs[(vi + 1) % 3]
            other2 = vs[(vi + 2) % 3]
            # Direction: toward the midpoint of the other two
            mid = (points[other1] + points[other2])
            mid = mid / np.linalg.norm(mid)
            # Also add a small outward component to maintain sphere constraint
            direction = mid + 0.1 * points[idx]
            direction = direction / np.linalg.norm(direction)
            points[idx] += lr * (direction - points[idx])
        
        # Renormalize
        points = points / np.linalg.norm(points, axis=1)[:, None]
        
        cr = _covering_radius_fast(points)
        if cr < best_cr:
            best_cr = cr
            best_points = points.copy()
    
    return best_points, best_cr


def _refine_lloyd(points, max_iters=200):
    """Lloyd-style iteration: move each point to the centroid of its Voronoi cell."""
    points = np.array(points, dtype=np.float64)
    n = len(points)
    
    best_points = points.copy()
    best_cr = _covering_radius_fast(points)
    
    for it in range(max_iters):
        try:
            hull = ConvexHull(points)
        except Exception:
            break
        
        # For each point, compute the centroid of its Voronoi cell
        # Approximate: for each hull facet, the circumcenter is a Voronoi vertex.
        # Assign each Voronoi vertex to the nearest point, then average.
        
        # Collect all Voronoi vertices (circumcenters of hull facets)
        voronoi_points = []
        for simplex in hull.simplices:
            v0, v1, v2 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
            e1 = v1 - v0
            e2 = v2 - v0
            normal = np.cross(e1, e2)
            nrm = np.linalg.norm(normal)
            if nrm < 1e-12:
                continue
            voronoi_points.append(normal / nrm)
        
        if len(voronoi_points) == 0:
            break
        
        voronoi_points = np.array(voronoi_points)
        
        # Assign each Voronoi point to nearest site
        # Compute distances (angular)
        # Use dot product for speed
        dots = voronoi_points @ points.T  # (n_voronoi, n_sites)
        nearest = np.argmin(-dots, axis=1)  # max dot = min angle
        
        # New position for each site: average of assigned Voronoi points
        new_points = np.zeros_like(points)
        counts = np.zeros(n)
        for i, site_idx in enumerate(nearest):
            new_points[site_idx] += voronoi_points[i]
            counts[site_idx] += 1
        
        # Update
        updated = False
        for i in range(n):
            if counts[i] > 0:
                new_pos = new_points[i] / counts[i]
                nrm = np.linalg.norm(new_pos)
                if nrm > 1e-12:
                    new_pos = new_pos / nrm
                    if np.linalg.norm(new_pos - points[i]) > 1e-8:
                        updated = True
                    points[i] = new_pos
            # If no Voronoi points assigned, keep original
        
        points = points / np.linalg.norm(points, axis=1)[:, None]
        
        cr = _covering_radius_fast(points)
        if cr < best_cr:
            best_cr = cr
            best_points = points.copy()
    
    return best_points, best_cr


def _fibonacci_spiral(n, seed=0):
    """Generate Fibonacci spiral points with optional perturbation."""
    rng = np.random.RandomState(seed)
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    # Add small random perturbation
    if seed > 0:
        noise = rng.randn(n, 3) * 0.01
        points = points + noise
        points = points / np.linalg.norm(points, axis=1)[:, None]
    return points


def _random_points(n, seed=0):
    """Generate random points on the sphere."""
    rng = np.random.RandomState(seed)
    points = rng.randn(n, 3)
    points = points / np.linalg.norm(points, axis=1)[:, None]
    return points


def _electrostatic_repulsion(points, max_iters=100, lr=0.01):
    """Refine using electrostatic repulsion (maximize minimum pairwise distance)."""
    points = np.array(points, dtype=np.float64)
    n = len(points)
    
    for it in range(max_iters):
        # Compute forces: repulsive from all other points
        forces = np.zeros_like(points)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                d = np.linalg.norm(points[i] - points[j])
                if d < 1e-10:
                    continue
                # Repulsive force proportional to 1/d^2 (Coulomb)
                force = (points[i] - points[j]) / (d * d * d)
                forces[i] += force
        
        # Update
        points += lr * forces
        points = points / np.linalg.norm(points, axis=1)[:, None]
    
    return points


def construct():
    start_time = time.time()
    
    best_points = None
    best_cr = np.inf
    
    # Multiple initializations
    initializations = []
    
    # 1. Fibonacci spiral
    initializations.append(_fibonacci_spiral(N, seed=0))
    
    # 2. Fibonacci with perturbation
    initializations.append(_fibonacci_spiral(N, seed=1))
    initializations.append(_fibonacci_spiral(N, seed=2))
    
    # 3. Random points
    initializations.append(_random_points(N, seed=42))
    initializations.append(_random_points(N, seed=123))
    
    # 4. Electrostatic refinement of random
    for seed in [7, 99]:
        pts = _random_points(N, seed=seed)
        pts = _electrostatic_repulsion(pts, max_iters=50, lr=0.005)
        initializations.append(pts)
    
    for init_idx, init_points in enumerate(initializations):
        elapsed = time.time() - start_time
        if elapsed > TIME_LIMIT - 20:
            break
        
        # Refine each initialization
        # Step 1: Lloyd iteration
        pts, cr = _refine_lloyd(init_points, max_iters=100)
        elapsed = time.time() - start_time
        if elapsed > TIME_LIMIT - 15:
            break
        
        # Step 2: Gradient refinement
        pts, cr = _refine_gradient(pts, max_iters=300, lr=0.002)
        elapsed = time.time() - start_time
        if elapsed > TIME_LIMIT - 10:
            break
        
        # Step 3: Another Lloyd pass
        pts, cr = _refine_lloyd(pts, max_iters=50)
        elapsed = time.time() - start_time
        if elapsed > TIME_LIMIT - 5:
            break
        
        if cr < best_cr:
            best_cr = cr
            best_points = pts.copy()
    
    if best_points is None:
        # Fallback: return Fibonacci spiral
        best_points = _fibonacci_spiral(N, seed=0)
    
    # Final normalization
    best_points = best_points / np.linalg.norm(best_points, axis=1)[:, None]
    
    # Ensure no coincident points
    for i in range(N):
        for j in range(i + 1, N):
            if np.linalg.norm(best_points[i] - best_points[j]) < 1e-6:
                # Separate them slightly
                direction = np.random.randn(3)
                direction = direction / np.linalg.norm(direction)
                best_points[j] = best_points[j] + 1e-4 * direction
                best_points[j] = best_points[j] / np.linalg.norm(best_points[j])
    
    return best_points
