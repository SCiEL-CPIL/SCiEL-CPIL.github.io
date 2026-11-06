"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _normalize(points):
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return points / norms


def _covering_radius_deg(points):
    """Compute the covering radius in degrees using the convex hull facets approach."""
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(points)
        # For each facet (simplex), find the farthest point on that facet from origin
        # The covering radius is the max over facets of the circumradius
        # For a simplex facet with vertices v_0, ..., v_k, the circumcenter is the point
        # equidistant from all vertices. The covering radius for that facet is the distance
        # from origin to the circumcenter projected... actually for a facet of the convex hull
        # of points on the unit sphere, the relevant quantity is the angular radius of the
        # spherical cap centered at the facet's circumcenter (on the sphere) that contains
        # the facet.
        
        # Better approach: for each facet, compute the circumcenter in 3D.
        # The circumcenter of a simplex with vertices v_0,...,v_k satisfies
        # |c - v_i|^2 = |c - v_j|^2 for all i,j.
        # This gives: 2 c . (v_j - v_i) = |v_j|^2 - |v_i|^2
        # Since all |v_i| = 1, this simplifies to c . (v_j - v_i) = 0 for all i,j.
        # So the circumcenter is perpendicular to all edge vectors, meaning it lies along
        # the normal to the facet.
        
        # For a facet with vertices v_0, ..., v_k (k+1 vertices forming a k-simplex in 3D,
        # so k <= 2 for 3D convex hull facets which are triangles), the circumcenter
        # lies along the normal direction n = (v_1-v_0) x (v_2-v_0) (for triangles).
        # We need to find the scalar t such that c = t*n satisfies |c - v_i| = R for all i.
        
        max_radius = 0.0
        for simplex in hull.simplices:
            verts = points[simplex]
            # Compute normal
            v0 = verts[0]
            v1 = verts[1]
            v2 = verts[2]
            edge1 = v1 - v0
            edge2 = v2 - v0
            n = np.cross(edge1, edge2)
            nn = np.linalg.norm(n)
            if nn < 1e-15:
                continue
            n = n / nn
            
            # Find t such that |t*n - v0| = |t*n - v1| = |t*n - v2|
            # |t*n - v0|^2 = t^2 - 2t(n.v0) + |v0|^2 = t^2 - 2t(n.v0) + 1
            # |t*n - v1|^2 = t^2 - 2t(n.v1) + 1
            # So n.v0 = n.v1 = n.v2 (which they should be since n is normal to the plane)
            # The circumradius R satisfies R^2 = t^2 - 2t(n.v0) + 1
            # But we also need the circumcenter to be the point on the line through origin
            # in direction n that is equidistant from all vertices.
            # Actually the circumcenter c satisfies c . (v_i - v_j) = 0, so c is along n.
            # Let c = t * n. Then |c - v_i|^2 = t^2 - 2t(n.v_i) + 1.
            # For this to be the same for all i, we need n.v_i to be the same for all i,
            # which is true since n is perpendicular to the plane containing the vertices.
            # The circumradius squared is t^2 - 2t*h + 1 where h = n.v_0.
            # But we also know that the circumcenter is the unique point equidistant from
            # all vertices. For a triangle on the unit sphere, we can compute it directly.
            
            # Alternative: the angular distance from the facet center to the farthest vertex.
            # The facet center (centroid on the sphere) is n / |n| but that's not quite right.
            # Let's use: the covering radius for this facet is the max angular distance
            # from any point in the spherical triangle to the nearest of the 3 vertices.
            # This is the circumradius of the spherical triangle.
            
            # For a spherical triangle with side lengths a, b, c (angular distances),
            # the circumradius R satisfies: sin(R) = sin(a)sin(b)sin(c) / (2 * area_term)
            # This is complex. Let's use a simpler approach:
            # The circumcenter of the planar triangle projected onto the sphere.
            
            # Compute planar circumcenter of the 3 vertices (as points in 3D)
            # The planar circumcenter C satisfies |C - v_i| = R_planar for all i.
            # C lies in the plane of the triangle. The circumcenter in 3D is:
            # C = v0 + a*edge1 + b*edge2 where a,b solve:
            # 2*a*(edge1.edge1) + 2*b*(edge1.edge2) = 2*(v0-v0).edge1 = 0 ... no
            # |C-v0|^2 = |C-v1|^2 => 2C.(v1-v0) = |v1|^2 - |v0|^2 = 0 => C.edge1 = 0
            # Wait, that's not right either. Let me just compute directly.
            
            # For 3 points on unit sphere, the circumcenter of the spherical triangle
            # is along the direction n (normal to the plane). The angular circumradius
            # R satisfies cos(R) = n . v_i (same for all i since n is normal).
            # Wait, that's the angular distance from the pole n to each vertex, which is
            # the same for all vertices only if the triangle is isosceles... no.
            
            # Let me think again. The circumcenter of a spherical triangle is the point
            # on the sphere equidistant (angularly) from all three vertices.
            # Let c be the circumcenter (unit vector). Then c.v_i = cos(R) for all i.
            # So c is a unit vector with c.v_0 = c.v_1 = c.v_2 = cos(R).
            # This means c - cos(R)*v_i is perpendicular to v_i for each i... 
            # Actually c.v_i = cos(R) means the projection of c onto v_i is cos(R).
            # We have 3 equations: c.v_0 = c.v_1 = c.v_2 = cos(R), and |c|=1.
            # From c.v_0 = c.v_1: c.(v_0 - v_1) = 0, so c is perpendicular to v_0-v_1.
            # Similarly c.(v_0 - v_2) = 0.
            # So c is along (v_0-v_1) x (v_0-v_2) = normal direction.
            # So c = ±n/|n| (unit normal).
            # Then cos(R) = c.v_0 = ±n.v_0/|n|.
            # We want the smaller R, so we pick the sign that gives cos(R) > 0 if possible.
            
            cos_R = abs(n[0]*v0[0] + n[1]*v0[1] + n[2]*v0[2]) / nn
            cos_R = min(cos_R, 1.0)
            R = np.arccos(cos_R)
            max_radius = max(max_radius, R)
        
        return np.degrees(max_radius)
    except Exception:
        # Fallback: compute max over a grid
        return 90.0


def _covering_radius_grid(points, n_samples=2000):
    """Fallback covering radius using random sampling."""
    # Sample random points on sphere
    rng = np.random.default_rng(42)
    theta = rng.uniform(0, 2*np.pi, n_samples)
    phi = np.arccos(rng.uniform(-1, 1, n_samples))
    samples = np.column_stack([
        np.sin(phi)*np.cos(theta),
        np.sin(phi)*np.sin(theta),
        np.cos(phi)
    ])
    # Compute angular distance from each sample to nearest point
    # cos of angular distance = dot product
    dots = samples @ points.T  # shape (n_samples, N)
    max_cos = dots.max(axis=1)  # max dot product = min angular distance
    max_cos = np.clip(max_cos, -1.0, 1.0)
    min_cos = max_cos.min()
    min_cos = np.clip(min_cos, -1.0, 1.0)
    return np.degrees(np.arccos(min_cos))


def _objective(x):
    """Objective: covering radius in radians (to be minimized)."""
    points = _normalize(x.reshape(N, 3))
    # Quick check: are all points distinct?
    # Compute pairwise distances
    dots = points @ points.T
    np.fill_diagonal(dots, 0)
    if dots.max() > 1.0 - 1e-10:
        return 1.0  # coincident points penalty
    # Use convex hull method
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(points)
        max_R = 0.0
        for simplex in hull.simplices:
            verts = points[simplex]
            v0, v1, v2 = verts[0], verts[1], verts[2]
            edge1 = v1 - v0
            edge2 = v2 - v0
            n = np.cross(edge1, edge2)
            nn = np.linalg.norm(n)
            if nn < 1e-15:
                continue
            cos_R = abs(np.dot(n, v0)) / nn
            cos_R = min(cos_R, 1.0)
            R = np.arccos(cos_R)
            if R > max_R:
                max_R = R
        return max_R
    except Exception:
        return 1.0


def _rsgd_step(points, step_size, worst_weight=0.5):
    """Riemannian SGD step focusing on worst-case points."""
    # Compute the covering radius contribution from each facet
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(points)
    except Exception:
        return points
    
    # Compute gradient for each point
    grads = np.zeros_like(points)
    
    for simplex in hull.simplices:
        verts = points[simplex]
        v0, v1, v2 = verts[0], verts[1], verts[2]
        edge1 = v1 - v0
        edge2 = v2 - v0
        n = np.cross(edge1, edge2)
        nn = np.linalg.norm(n)
        if nn < 1e-15:
            continue
        n_unit = n / nn
        cos_R = abs(np.dot(n_unit, v0))
        cos_R = min(cos_R, 1.0)
        R = np.arccos(cos_R)
        
        # The circumcenter direction
        c_dir = n_unit if np.dot(n_unit, v0) > 0 else -n_unit
        
        # Gradient of R with respect to each vertex
        # R = arccos(|n.v0|/|n|), this is complex. Let's use a simpler approach:
        # Move each vertex slightly toward the circumcenter to reduce R.
        # The direction to move v_i is toward c_dir projected onto the tangent space at v_i.
        for idx in simplex:
            vi = points[idx]
            # Tangent direction toward c_dir
            tangent = c_dir - np.dot(c_dir, vi) * vi
            tnorm = np.linalg.norm(tangent)
            if tnorm > 1e-15:
                grads[idx] += (tangent / tnorm) * R
    
    # Normalize gradient and take step
    for i in range(N):
        gi = grads[i]
        gi_norm = np.linalg.norm(gi)
        if gi_norm > 1e-15:
            # Riemannian gradient: project to tangent space
            g_tangent = gi - np.dot(gi, points[i]) * points[i]
            g_tnorm = np.linalg.norm(g_tangent)
            if g_tnorm > 1e-15:
                points[i] = points[i] + step_size * g_tangent / g_tnorm
                # Renormalize
                points[i] = points[i] / np.linalg.norm(points[i])
    
    return points


def construct():
    t0 = time.time()
    deadline = t0 + 145.0  # safety margin
    
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    points = _normalize(points)
    
    best_points = points.copy()
    best_radius = _covering_radius_grid(points, 5000)
    
    # Phase 1: Riemannian SGD with annealing
    n_steps = 5000
    for step in range(n_steps):
        if time.time() > deadline - 30.0:
            break
        
        # Annealing schedule
        progress = step / n_steps
        step_size = 0.05 * (1.0 - progress) ** 2 + 0.001
        
        points = _rsgd_step(points, step_size)
        
        # Evaluate periodically
        if step % 100 == 0 or step == n_steps - 1:
            rad = _covering_radius_grid(points, 3000)
            if rad < best_radius:
                best_radius = rad
                best_points = points.copy()
    
    # Phase 2: Local refinement using scipy minimize on the smooth proxy
    if time.time() < deadline - 10.0:
        # Use the best points as starting point
        x0 = best_points.flatten()
        try:
            result = minimize(
                _objective, x0, method='L-BFGS-B',
                options={'maxiter': 200, 'gtol': 1e-8}
            )
            if result.fun < best_radius * np.pi / 180.0:
                new_points = _normalize(result.x.reshape(N, 3))
                new_rad = _covering_radius_grid(new_points, 5000)
                if new_rad < best_radius:
                    best_radius = new_rad
                    best_points = new_points
        except Exception:
            pass
    
    # Final validation
    best_points = _normalize(best_points)
    
    # Ensure no coincident points
    dots = best_points @ best_points.T
    np.fill_diagonal(dots, 0)
    if dots.max() > 1.0 - 1e-10:
        # Perturb coincident points
        for i in range(N):
            for j in range(i+1, N):
                if dots[i,j] > 1.0 - 1e-10:
                    best_points[j] += 0.01 * np.random.randn(3)
                    best_points[j] /= np.linalg.norm(best_points[j])
    
    return best_points
# EVOLVE-BLOCK-END
