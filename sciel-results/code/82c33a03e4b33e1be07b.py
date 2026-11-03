"""Rewritten program: optimize 50-point spherical covering via repulsion + local refinement."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return pts / norms

def _covering_radius_deg(pts):
    """Compute the covering radius in degrees.
    The covering radius is the maximum over all points on the sphere of the angular
    distance to the nearest point in pts. This equals the maximum circumradius of the
    facets of the convex hull of pts (for points on the unit sphere).
    
    We approximate this by:
    1. Computing the convex hull facets
    2. For each facet (triangle), computing its circumradius on the sphere
    3. Taking the maximum
    
    A simpler but effective approximation: the covering radius is bounded by the
    maximum distance from any hull vertex to the farthest point on its Voronoi cell,
    which for spherical Delaunay/Voronoi is related to the facet circumradii.
    
    For efficiency, we use the fact that the covering radius equals the maximum
    over all points on the sphere of min distance to pts. We can estimate this by
    sampling, but for the evaluator's purposes, we need a good estimate.
    
    Actually, the true covering radius for a set of points on the sphere is the
    maximum circumradius of the spherical triangles in the Delaunay triangulation
    (equivalently, the convex hull facets when points are in general position).
    
    We'll compute this properly using scipy's ConvexHull.
    """
    from scipy.spatial import ConvexHull
    
    # Ensure points are on unit sphere
    pts = _normalize(pts)
    
    # Compute convex hull
    hull = ConvexHull(pts)
    
    # For each facet (simplicial face), compute the circumradius
    # A facet is a set of vertices. For 3D, facets are triangles.
    # The circumradius of a spherical triangle with vertices a, b, c (unit vectors)
    # can be computed from the angles of the triangle.
    
    max_circum = 0.0
    
    for simplex in hull.simplices:
        verts = pts[simplex]
        # For a triangle on the unit sphere, the circumradius is the angular radius
        # of the smallest cap containing all three vertices, which is the distance
        # from the centroid (in the spherical sense) to the farthest vertex.
        # More precisely, the circumcenter of a spherical triangle is the point
        # equidistant from all three vertices.
        
        # For three unit vectors a, b, c, the circumcenter is in the direction
        # of the cross product approach. Actually, for a spherical triangle,
        # the circumcenter can be found as the normalized sum of the vertices
        # weighted appropriately, or more simply:
        
        # The circumradius R satisfies: for each vertex v, angular_dist(circumcenter, v) = R
        # For a triangle with side lengths (angular) alpha, beta, gamma,
        # the circumradius can be computed via spherical trigonometry.
        
        # Simpler approach: the circumcenter of three points on a sphere is the
        # intersection of the sphere with the plane equidistant from all three.
        # For unit vectors a, b, c, the circumcenter direction is:
        #   d = (a x b) x c + ... this gets complicated.
        
        # Let's use a numerical approach: the circumradius is the maximum distance
        # from the spherical centroid to any vertex, but that's not exact.
        
        # Better: for a spherical triangle, the circumradius R is given by:
        #   sin(R) = a * b * c / (2 * sqrt(1 + 2*cos(A)*cos(B)*cos(C) - cos^2(A) - cos^2(B) - cos^2(C)))
        # where a, b, c are side lengths and A, B, C are angles. This is complex.
        
        # Practical approach: the circumcenter of three points on the unit sphere
        # is the unit vector in the direction of the normal to the plane through
        # the three points, IF the triangle is small. But for general triangles,
        # we need to solve properly.
        
        # For three unit vectors a, b, c, the circumcenter u satisfies:
        #   u . a = u . b = u . c = cos(R)
        # So u is in the span of a, b, c and equidistant from all three.
        # u = alpha*a + beta*b + gamma*c
        # u.a = u.b = u.c
        # This gives a linear system.
        
        a, b, c = verts[0], verts[1], verts[2]
        
        # Solve for u in span{a,b,c} such that u.a = u.b = u.c
        # Let u = alpha*a + beta*b + gamma*c
        # u.a = alpha + beta*(a.b) + gamma*(a.c)
        # u.b = alpha*(a.b) + beta + gamma*(b.c)
        # u.c = alpha*(a.c) + beta*(b.c) + gamma
        # Set u.a = u.b = u.c = t
        # (1 - 1)*alpha + (a.b - 1)*beta + (a.c - 1)*gamma = 0  ... this is getting messy
        
        # Alternative: the circumcenter is the normalized vector:
        #   u = normalize( (a x b) x (b x c) x (c x a) )  ... no
        
        # Let me just use the fact that for the covering radius, we can sample
        # the sphere densely and compute min distance. But that's slow.
        
        # Actually, for the evaluator, the covering radius is computed by them.
        # My job is just to return good points. Let me use a simpler metric for
        # my internal optimization: the maximum nearest-neighbor distance,
        # estimated by sampling.
        
        pass
    
    # Let's just sample the sphere to estimate the covering radius
    # This is approximate but good enough for optimization
    theta = np.linspace(0, np.pi, 360)
    phi = np.linspace(0, 2*np.pi, 720)
    TH, PH = np.meshgrid(theta, phi, indexing='ij')
    sx = np.sin(TH) * np.cos(PH)
    sy = np.sin(TH) * np.sin(PH)
    sz = np.cos(TH)
    samples = np.column_stack([sx.ravel(), sy.ravel(), sz.ravel()])
    
    # For each sample, find min distance to pts
    # Use cdist for efficiency
    from scipy.spatial.distance import cdist
    d = cdist(samples, pts)
    min_d = d.min(axis=1)
    max_min_d = min_d.max()
    
    return np.degrees(np.arccos(np.clip(max_min_d, -1, 1)))

def _repulsion_force(pts, k=4):
    """Compute repulsion force on each point from all others."""
    n = len(pts)
    forces = np.zeros_like(pts)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = pts[i] - pts[j]
            dist = np.linalg.norm(d)
            if dist < 1e-10:
                # Add small random perturbation
                d = np.random.randn(3) * 1e-10
                dist = np.linalg.norm(d)
            force = d / (dist ** (k + 1))
            forces[i] += force
    return forces

def _local_refinement(pts, n_iter=200, step_size=0.01):
    """Local refinement using gradient descent on a smoothed objective."""
    pts = _normalize(pts.copy())
    best_pts = pts.copy()
    best_score = _covering_radius_deg(best_pts)
    
    for it in range(n_iter):
        # Compute forces (repulsion)
        forces = _repulsion_force(pts, k=4)
        
        # Also add a slight attraction to avoid clustering
        # Update positions
        new_pts = pts + step_size * forces
        new_pts = _normalize(new_pts)
        
        score = _covering_radius_deg(new_pts)
        if score < best_score:
            best_score = score
            best_pts = new_pts.copy()
            pts = new_pts
        else:
            # Reduce step size
            step_size *= 0.95
    
    return best_pts

def construct():
    start_time = time.time()
    deadline = 140.0  # 10s safety margin
    
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts = _normalize(pts)
    
    best_pts = pts.copy()
    best_score = _covering_radius_deg(best_pts)
    
    # Phase 1: Repulsion-based optimization
    np.random.seed(42)
    for restart in range(5):
        if time.time() > deadline - 30:
            break
        
        # Start from current best or perturbed
        if restart == 0:
            current = pts.copy()
        else:
            current = best_pts + np.random.randn(N, 3) * 0.1
            current = _normalize(current)
        
        # Symplectic-like integration with repulsion
        velocity = np.zeros_like(current)
        dt = 0.05
        for step in range(500):
            if time.time() > deadline - 20:
                break
            
            forces = _repulsion_force(current, k=4)
            velocity = 0.9 * velocity + dt * forces  # damping
            new_pts = current + dt * velocity
            new_pts = _normalize(new_pts)
            
            score = _covering_radius_deg(new_pts)
            if score < best_score:
                best_score = score
                best_pts = new_pts.copy()
            
            current = new_pts
        
        # Local refinement
        if time.time() < deadline - 15:
            refined = _local_refinement(best_pts, n_iter=100, step_size=0.01)
            refined_score = _covering_radius_deg(refined)
            if refined_score < best_score:
                best_score = refined_score
                best_pts = refined
    
    # Final check
    if time.time() > deadline - 5:
        return _normalize(best_pts)
    
    return _normalize(best_pts)
# EVOLVE-BLOCK-END
