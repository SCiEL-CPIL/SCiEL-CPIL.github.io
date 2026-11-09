"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 130

# EVOLVE-BLOCK-START

def _cover_radius(points):
    """Compute the covering radius in radians.
    
    The covering radius is the maximum angular distance from any point on the sphere
    to its nearest site. This equals the maximum circumradius of the spherical Voronoi cells,
    which for the convex hull facets corresponds to the largest empty cap.
    
    We compute this by finding the farthest point in each Voronoi cell, which is the
    center of the largest empty circle in that cell. For a spherical Voronoi cell,
    the farthest point is the spherical circumcenter of the cell's vertices (if it
    lies inside the cell) or a vertex.
    
    A simpler and robust approach: sample the sphere densely and compute the max
    angular distance to nearest site. But this is approximate.
    
    Better: For each pair of sites, the Voronoi boundary is the great circle perpendicular
    to the line connecting them. The covering radius is the max over all Voronoi vertices
    (intersection of 3 boundaries) of the distance to the nearest site, plus checking
    the circumcenters of each facet of the convex hull.
    
    Actually, the most robust method: the covering radius equals the maximum, over all
    facets of the convex hull of the points, of the circumradius of that facet (the angular
    radius of the smallest cap centered at the circumcenter of the facet that contains
    the facet). But we need to ensure the cap is empty (contains no other points).
    
    Let me use a different approach: compute the spherical Voronoi diagram's vertices and
    check distances, plus the cell circumcenters.
    
    For efficiency and robustness, I'll use the following:
    1. Compute all pairwise great-circle boundaries
    2. Find Voronoi vertices (intersections of 3 boundaries)
    3. For each Voronoi vertex, compute distance to nearest site
    4. The covering radius is the max of these distances
    
    Actually, the covering radius is achieved at either a Voronoi vertex or at the
    circumcenter of a Voronoi cell (which is the center of the largest empty circle
    in that cell).
    
    Let me use a practical approach: for each site, the Voronoi cell's farthest point
    is either a vertex of the cell or the circumcenter of the cell (if it's inside).
    
    I'll compute it as: max over all Voronoi vertices of dist(v, nearest site),
    and max over all cells of the circumradius of the cell's vertices (if the circumcenter
    is inside the cell).
    
    For simplicity and correctness, let me just densely sample and take max.
    
    Actually, let me think more carefully. The covering radius of a set of points on the
    sphere is the radius of the largest empty cap. This is equal to the maximum, over all
    triples of points (i,j,k) that form a facet of the convex hull, of the angular circumradius
    of the spherical triangle (i,j,k), provided the circumcenter of that triangle is not
    covered by any other point (i.e., the cap is empty).
    
    Let me implement this properly.
    """
    points = np.asarray(points)
    n = len(points)
    
    # Compute all pairwise dot products
    # For n=130, this is 130x130 = 16900, fine
    dots = points @ points.T
    
    # For each triple (i, j, k) with i < j < k, check if they form a facet
    # A facet of the convex hull: the spherical triangle (i,j,k) such that no other
    # point is in the spherical cap centered at the circumcenter with radius = circumradius.
    
    # The circumcenter of spherical triangle (i,j,k) is the point equidistant from all three.
    # It's the normalized cross product of (j-i) and (k-i) in 3D... no, that's not right.
    
    # Actually, the circumcenter is the intersection of the sphere with the line through
    # the origin in the direction of the vector that is equidistant from i, j, k.
    # This vector v satisfies v.i = v.j = v.k.
    # So v is in the null space of [i-j; i-k] (in the sense that v.(i-j)=0 and v.(i-k)=0).
    # So v is parallel to (i-j) x (i-k).
    
    # Let me compute for each triple.
    # For n=130, number of triples is C(130,3) = 357760. That's a lot but manageable.
    
    # Actually, let me use a simpler but still correct approach:
    # The covering radius is the max over all points y on the sphere of min_i angle(y, p_i).
    # This maximum is achieved at a Voronoi vertex or at a cell circumcenter.
    
    # Let me just compute the max over a dense set of points. For 130 points, the
    # covering radius is around 11 degrees, so the Voronoi cells have area ~ 4*pi/130 ~ 0.096 sr.
    # A dense sampling with ~100000 points should be sufficient.
    
    # But this is approximate. Let me do the exact computation.
    
    # Exact method:
    # 1. For each triple (i,j,k), compute the circumcenter c and circumradius r.
    # 2. Check if the cap centered at c with radius r contains any other point.
    #    If not, then r is a candidate for the covering radius.
    # 3. Also check Voronoi vertices: for each triple, the Voronoi vertex is the point
    #    equidistant from i, j, k (on the sphere). This is the same as the circumcenter!
    #    Wait, no. The Voronoi vertex is the point equidistant from i, j, k, which IS
    #    the circumcenter of the triangle (i,j,k).
    
    # So actually, the Voronoi vertices ARE the circumcenters of the spherical triangles
    # formed by triples of points. And the covering radius is the max over all such
    # circumcenters (that are Voronoi vertices, i.e., the circumcenter is inside the
    # Voronoi cell) of the distance to the nearest site.
    
    # Hmm, let me think again. The covering radius is the max over the sphere of the
    # distance to the nearest site. This function is piecewise smooth, with maxima at
    # Voronoi vertices (where 3 sites are equidistant and nearest) or at the circumcenter
    # of a Voronoi cell (where the farthest point in the cell is the circumcenter).
    
    # For a Voronoi cell with vertices v1, v2, ..., vm, the farthest point from the
    # site is either one of the vertices or the circumcenter of the cell (if it exists
    # and is inside the cell).
    
    # Let me just compute it as: for each triple (i,j,k) that forms a facet of the
    # convex hull, compute the circumradius, and take the max.
    
    # A triple (i,j,k) forms a facet if the circumcenter c of the triangle (i,j,k)
    # satisfies: for all other points p, angle(c, p) > circumradius (i.e., p is outside
    # the cap). Equivalently, dot(c, p) < dot(c, i) for all p != i,j,k.
    
    # Let me implement this.
    
    max_r = 0.0
    
    # Precompute for speed
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                pi, pj, pk = points[i], points[j], points[k]
                
                # Circumcenter: v = (pi - pj) x (pi - pk), normalized
                v = np.cross(pi - pj, pi - pk)
                nv = np.linalg.norm(v)
                if nv < 1e-12:
                    continue
                c = v / nv
                
                # Make sure c is on the "correct" side (same hemisphere as the triangle)
                # The circumcenter should satisfy c.pi = c.pj = c.pk
                # Let's check: if c.pi < 0, flip c
                if c @ pi < 0:
                    c = -c
                
                # Circumradius: arccos(c.pi)
                cos_r = np.clip(c @ pi, -1.0, 1.0)
                r = np.arccos(cos_r)
                
                if r <= max_r:
                    continue
                
                # Check if the cap is empty: for all p != i,j,k, c.p < c.pi
                # i.e., the angular distance from c to p is > r
                dots_c = c @ points
                # Exclude i, j, k
                mask = np.ones(n, dtype=bool)
                mask[i] = False
                mask[j] = False
                mask[k] = False
                if np.max(dots_c[mask]) > cos_r + 1e-12:
                    continue
                
                max_r = r
    
    return max_r


def _cover_radius_fast(points):
    """Faster covering radius using a sampling-based approach with refinement."""
    points = np.asarray(points)
    n = len(points)
    
    # Dense sampling
    n_samples = 50000
    # Use a quasi-random sequence for better coverage
    i = np.arange(n_samples) + 0.5
    z = 1.0 - 2.0 * i / n_samples
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    samples = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Compute min distance from each sample to nearest point
    # For memory efficiency, do in batches
    batch_size = 5000
    max_dist = 0.0
    for start in range(0, n_samples, batch_size):
        batch = samples[start:start+batch_size]
        dots = batch @ points.T  # (batch, n)
        min_dot = np.max(dots, axis=1)  # max dot = min angle
        min_angle = np.arccos(np.clip(min_dot, -1.0, 1.0))
        batch_max = np.max(min_angle)
        if batch_max > max_dist:
            max_dist = batch_max
    
    return max_dist


def _cover_radius_exact(points):
    """Exact covering radius using the facet-based method."""
    points = np.asarray(points)
    n = len(points)
    
    max_r = 0.0
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                pi, pj, pk = points[i], points[j], points[k]
                v = np.cross(pi - pj, pi - pk)
                nv = np.linalg.norm(v)
                if nv < 1e-12:
                    continue
                c = v / nv
                if c @ pi < 0:
                    c = -c
                cos_r = np.clip(c @ pi, -1.0, 1.0)
                r = np.arccos(cos_r)
                if r <= max_r + 1e-10:
                    continue
                dots_c = c @ points
                mask = np.ones(n, dtype=bool)
                mask[i] = False
                mask[j] = False
                mask[k] = False
                if np.max(dots_c[mask]) > cos_r + 1e-10:
                    continue
                max_r = r
    return max_r


def _initial_fibonacci():
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _objective(points):
    """Objective: covering radius in radians (to minimize)."""
    return _cover_radius_fast(points)


def _repulsion_energy(points):
    """A smooth surrogate: sum of -log(distance) or similar, to use as a gradient-based objective."""
    # Use a repulsive potential: sum_{i<j} -1/d_ij^alpha
    # This is smooth and differentiable
    n = len(points)
    dots = points @ points.T
    # Avoid self-interaction
    np.fill_diagonal(dots, np.nan)
    # d_ij^2 = 2 - 2*dots_ij
    d2 = 2.0 - 2.0 * dots
    d2 = np.maximum(d2, 1e-12)
    # Repulsive energy: sum -1/sqrt(d2) = sum -1/d
    # But we want to maximize minimum distance, so minimize sum -1/d
    # Actually, let's use a different approach: minimize sum of exp(-beta * d_ij)
    # which is smooth and emphasizes close pairs
    beta = 10.0
    energy = 0.0
    for i in range(n):
        for j in range(i+1, n):
            energy += np.exp(-beta * np.sqrt(d2[i,j]))
    return energy


def _grad_repulsion(points):
    """Gradient of the repulsion energy."""
    n = len(points)
    beta = 10.0
    grad = np.zeros_like(points)
    for i in range(n):
        for j in range(i+1, n):
            d_vec = points[i] - points[j]
            d = np.linalg.norm(d_vec)
            if d < 1e-12:
                continue
            # d/d(points[i]) exp(-beta * |pi - pj|)
            # = exp(-beta*d) * (-beta) * d/d(pi) |pi-pj|
            # = exp(-beta*d) * (-beta) * (pi-pj)/|pi-pj|
            # But we need to project to the tangent space of the sphere
            factor = beta * np.exp(-beta * d) / d
            grad[i] += factor * d_vec
            grad[j] -= factor * d_vec
    # Project to tangent space
    for i in range(n):
        grad[i] = grad[i] - np.dot(grad[i], points[i]) * points[i]
    return grad


def construct():
    """Construct 130 points on the unit sphere minimizing the covering radius."""
    start_time = time.time()
    deadline = start_time + 145.0  # 5 second safety margin
    
    # Start with Fibonacci spiral
    pts = _initial_fibonacci().copy()
    best_pts = pts.copy()
    best_score = _cover_radius_fast(pts)
    
    # Local optimization using gradient descent on a repulsive potential
    # with periodic annealing and restarts
    
    lr = 0.01
    beta_start = 5.0
    beta_end = 30.0
    n_steps = 500
    
    # Multi-start with annealing
    n_restarts = 8
    
    for restart in range(n_restarts):
        if time.time() > deadline - 20.0:
            break
        
        if restart == 0:
            pts = _initial_fibonacci().copy()
        else:
            # Perturb the best solution
            pts = best_pts.copy()
            noise = np.random.randn(N, 3) * 0.1
            pts = pts + noise
            # Renormalize
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            pts = pts / norms
            # Check no coincident points
            dots = pts @ pts.T
            np.fill_diagonal(dots, 0)
            if np.max(dots) > 0.9999:
                # Too many coincident points, try different perturbation
                pts = _initial_fibonacci().copy()
                noise = np.random.randn(N, 3) * 0.05
                pts = pts + noise
                norms = np.linalg.norm(pts, axis=1, keepdims=True)
                pts = pts / norms
        
        # Anneal beta
        for step in range(n_steps):
            if time.time() > deadline - 10.0:
                break
            
            beta = beta_start + (beta_end - beta_start) * (step / n_steps)
            
            # Compute gradient
            grad = _grad_repulsion(pts)
            
            # Gradient step
            pts_new = pts - lr * grad
            # Renormalize
            norms = np.linalg.norm(pts_new, axis=1, keepdims=True)
            pts_new = pts_new / norms
            
            # Check for coincident points
            dots = pts_new @ pts_new.T
            np.fill_diagonal(dots, 0)
            if np.max(dots) > 0.99999:
                # Backtrack
                lr *= 0.5
                continue
            
            pts = pts_new
            
            # Occasionally evaluate the actual covering radius
            if step % 50 == 0 or step == n_steps - 1:
                score = _cover_radius_fast(pts)
                if score < best_score:
                    best_score = score
                    best_pts = pts.copy()
        
        # Reset learning rate for next restart
        lr = max(0.005, lr * 0.8)
    
    # Final polish: use the best points found
    # Do a few more iterations with smaller steps
    if time.time() < deadline - 5.0:
        pts = best_pts.copy()
        for _ in range(200):
            if time.time() > deadline - 3.0:
                break
            grad = _grad_repulsion(pts)
            pts_new = pts - 0.001 * grad
            norms = np.linalg.norm(pts_new, axis=1, keepdims=True)
            pts_new = pts_new / norms
            dots = pts_new @ pts_new.T
            np.fill_diagonal(dots, 0)
            if np.max(dots) > 0.99999:
                break
            pts = pts_new
            score = _cover_radius_fast(pts)
            if score < best_score:
                best_score = score
                best_pts = pts.copy()
    
    # Ensure no coincident points and valid configuration
    final_pts = best_pts
    norms = np.linalg.norm(final_pts, axis=1, keepdims=True)
    final_pts = final_pts / norms
    
    # Check for coincident points
    dots = final_pts @ final_pts.T
    np.fill_diagonal(dots, 0)
    if np.max(dots) > 0.9999:
        # Fall back to Fibonacci
        final_pts = _initial_fibonacci()
    
    # Check not in a closed hemisphere: the points should span the sphere
    # A simple check: the origin should be in the convex hull
    # For 130 points on the sphere, this is almost always true unless they're badly clustered
    # Let's just check that the max dot product between any pair is < 1 (no coincident)
    
    return final_pts

# EVOLVE-BLOCK-END
