"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 130

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 130s safety margin out of 150s

    def fibonacci_spiral(n):
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(0, 1.0 - z * z))
        pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        return pts

    def normalize(pts):
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        return pts / norms

    def covering_radius(pts):
        """Compute the covering radius in radians using convex hull facets."""
        from scipy.spatial import ConvexHull
        try:
            hull = ConvexHull(pts)
        except Exception:
            return np.pi  # invalid or degenerate
        # The covering radius is the maximum circumradius of the hull facets
        # For each facet, find the circumcenter of the points defining it
        # and compute the angular distance from that circumcenter to the facet points
        max_radius = 0.0
        for simplex in hull.simplices:
            facet_pts = pts[simplex]
            # Compute circumcenter of the facet (on the unit sphere)
            # For a triangle on the unit sphere, the circumcenter is the normalized
            # sum of the vertex vectors (for acute triangles) or the point opposite
            # the longest side for obtuse triangles
            n_pts = len(simplex)
            if n_pts == 3:
                a, b, c = facet_pts
                # Check if the triangle is acute
                # Compute side lengths (chord lengths)
                ab = np.linalg.norm(a - b)
                bc = np.linalg.norm(b - c)
                ca = np.linalg.norm(c - a)
                # The circumcenter of a spherical triangle
                # For a triangle on the unit sphere, if it's acute, the circumcenter
                # is in the direction of a + b + c (normalized)
                # If obtuse, it's the antipode of the vertex opposite the longest side
                # Actually, for covering radius we need the point on the sphere
                # farthest from the facet, which is the circumcenter of the
                # spherical triangle
                # The circumcenter of a spherical triangle can be found as follows:
                # It's the intersection of the great circles perpendicular to the sides
                # A simpler approach: the circumcenter is the normalized cross product
                # approach or we can use the formula for the circumradius
                
                # For a spherical triangle with vertices a, b, c on the unit sphere,
                # the circumradius R satisfies:
                # cos(R) = (a·b + b·c + c·a - 1) / (2 * (a·b * b·c * c·a - ... ))
                # Actually, let's use a more direct approach:
                # The circumcenter O satisfies |O - a| = |O - b| = |O - c| and |O| = 1
                # This means O·a = O·b = O·c = cos(R)
                # So O is in the null space of [a; b; c] - cos(R) * ones
                # Equivalently, O is proportional to (a + b + c) for acute triangles
                
                s = a + b + c
                s_norm = np.linalg.norm(s)
                if s_norm < 1e-10:
                    # Degenerate, use the vertex with largest angle
                    # Fall back to max distance from vertices
                    for v in facet_pts:
                        # The farthest point from this facet vertex region
                        # is approximately the antipode
                        max_radius = max(max_radius, np.arccos(np.clip(-1.0, 1.0, np.dot(v, -v))))
                    continue
                
                O = s / s_norm
                # Check if O is on the correct side (same side as the interior)
                # The circumradius is the angular distance from O to any vertex
                cos_R = np.dot(O, a)
                cos_R = np.clip(cos_R, -1.0, 1.0)
                R = np.arccos(cos_R)
                
                # For obtuse triangles, the circumcenter is outside the triangle
                # and the covering radius is determined by the longest side
                # Check if the triangle is obtuse by checking if O is inside
                # The triangle is acute if the circumcenter is inside
                # A simpler check: if any angle is > 90 degrees
                # Angle at a: arccos(((b-a)·(c-a)) / (|b-a||c-a|)) but on sphere
                # Spherical angle at a: arccos(((b·c - (b·a)(a·c)) / sqrt((1-(b·a)^2)(1-(a·c)^2))))
                ba = np.clip(np.dot(b, a), -1.0, 1.0)
                ca = np.clip(np.dot(c, a), -1.0, 1.0)
                bc = np.clip(np.dot(b, c), -1.0, 1.0)
                denom = np.sqrt(max(0, (1 - ba**2) * (1 - ca**2)))
                if denom > 1e-10:
                    cos_angle_a = (bc - ba * ca) / denom
                    cos_angle_a = np.clip(cos_angle_a, -1.0, 1.0)
                    angle_a = np.arccos(cos_angle_a)
                    if angle_a > np.pi / 2:
                        # Obtuse at a, circumcenter is antipode of a
                        O = -a
                        cos_R = np.dot(O, b)
                        cos_R = np.clip(cos_R, -1.0, 1.0)
                        R = np.arccos(cos_R)
                        # But the covering radius is the max over the facet
                        # For an obtuse triangle, the farthest point is the midpoint of the longest side
                        # The longest side is opposite the obtuse angle
                        # So the farthest point from the facet is the midpoint of the longest side
                        # The angular distance from the midpoint to the endpoints is half the side length
                        # The longest side is bc (opposite to a)
                        mid = b + c
                        mid_norm = np.linalg.norm(mid)
                        if mid_norm > 1e-10:
                            mid = mid / mid_norm
                            # The covering radius contribution is the distance from mid to b (or c)
                            cos_half = np.dot(mid, b)
                            cos_half = np.clip(cos_half, -1.0, 1.0)
                            R = np.arccos(cos_half)
                        max_radius = max(max_radius, R)
                        continue
                
                max_radius = max(max_radius, R)
            else:
                # For non-triangular facets (shouldn't happen with ConvexHull in 3D)
                for v in facet_pts:
                    max_radius = max(max_radius, np.arccos(np.clip(-1.0, 1.0, -1.0)))
        return max_radius

    def covering_radius_fast(pts):
        """Faster covering radius using the fact that the covering radius is the
        maximum over hull facets of the circumradius."""
        from scipy.spatial import ConvexHull
        try:
            hull = ConvexHull(pts)
        except Exception:
            return np.pi
        
        max_radius = 0.0
        for simplex in hull.simplices:
            a, b, c = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
            s = a + b + c
            s_norm = np.linalg.norm(s)
            if s_norm < 1e-10:
                continue
            O = s / s_norm
            cos_R = np.clip(np.dot(O, a), -1.0, 1.0)
            R = np.arccos(cos_R)
            max_radius = max(max_radius, R)
        return max_radius

    # Initialize with Fibonacci spiral
    pts = normalize(fibonacci_spiral(N))
    best_pts = pts.copy()
    best_cr = covering_radius_fast(pts)

    # Riemannian gradient descent with dynamic worst-case focus
    # The objective is the max over all points on the sphere of the distance to the nearest point
    # This is a non-smooth function. We use a smooth approximation:
    # For each point y on the sphere, f(y) = min_i d(y, p_i)
    # The covering radius is max_y f(y)
    # We can approximate the gradient by focusing on the worst-case points
    
    def riemannian_sgd(pts, max_iters, deadline):
        """Riemannian SGD with annealing and worst-case focus."""
        pts = pts.copy()
        lr = 0.1
        best_cr = covering_radius_fast(pts)
        best_pts = pts.copy()
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Compute the covering radius and identify worst-case points
            cr = covering_radius_fast(pts)
            if cr < best_cr:
                best_cr = cr
                best_pts = pts.copy()
            
            # Compute pairwise distances
            # For efficiency, we'll sample points on the sphere and find the worst ones
            n_samples = 5000
            # Generate random points on the sphere
            u = np.random.randn(n_samples, 3)
            u = u / np.linalg.norm(u, axis=1, keepdims=True)
            
            # For each sample point, find the nearest configuration point
            # and compute the distance
            dists = np.zeros((n_samples, N))
            for i in range(N):
                dots = np.clip(u @ pts[i], -1.0, 1.0)
                dists[:, i] = np.arccos(dots)
            
            min_dists = np.min(dists, axis=1)
            nearest_idx = np.argmin(dists, axis=1)
            
            # Focus on the worst-case points (largest min distances)
            worst_threshold = np.percentile(min_dists, 95)
            worst_mask = min_dists > worst_threshold
            worst_pts = u[worst_mask]
            worst_nearest = nearest_idx[worst_mask]
            
            if len(worst_pts) == 0:
                break
            
            # Compute the gradient of the smooth approximation
            # For each worst-case point y, the gradient of d(y, p_i) with respect to p_i
            # is in the direction that moves p_i toward y
            # The Riemannian gradient on the sphere for moving p_i toward y is:
            # grad = (y - (y·p_i) p_i) / ||y - (y·p_i) p_i||
            
            grad = np.zeros_like(pts)
            for y, idx in zip(worst_pts, worst_nearest):
                dot = np.dot(y, pts[idx])
                dot = np.clip(dot, -1.0, 1.0)
                # The direction to move p_i is toward y
                # The Riemannian gradient direction is the component of y orthogonal to p_i
                orth = y - dot * pts[idx]
                orth_norm = np.linalg.norm(orth)
                if orth_norm > 1e-10:
                    grad[idx] += orth / orth_norm
            
            # Normalize the gradient
            grad_norms = np.linalg.norm(grad, axis=1, keepdims=True)
            grad_norms[grad_norms < 1e-12] = 1.0
            grad = grad / grad_norms
            
            # Update: move each point in the direction of the gradient
            # With annealing: reduce learning rate over time
            anneal_factor = 1.0 / (1.0 + 0.01 * it)
            step = lr * anneal_factor
            
            new_pts = pts + step * grad
            # Project back to the sphere
            new_pts = normalize(new_pts)
            
            # Check if the update improved the covering radius
            new_cr = covering_radius_fast(new_pts)
            if new_cr < cr:
                pts = new_pts
            else:
                # Try a smaller step
                new_pts = pts + step * 0.5 * grad
                new_pts = normalize(new_pts)
                new_cr = covering_radius_fast(new_pts)
                if new_cr < cr:
                    pts = new_pts
                else:
                    # Keep the old points
                    pass
            
            # Occasionally try a random perturbation to escape local minima
            if it % 50 == 49 and np.random.random() < 0.3:
                # Perturb a random subset of points
                n_perturb = max(1, N // 10)
                idxs = np.random.choice(N, n_perturb, replace=False)
                for idx in idxs:
                    # Random direction on the sphere
                    rand_dir = np.random.randn(3)
                    rand_dir = rand_dir / np.linalg.norm(rand_dir)
                    # Move the point
                    new_pt = pts[idx] + 0.1 * rand_dir
                    new_pt = new_pt / np.linalg.norm(new_pt)
                    test_pts = pts.copy()
                    test_pts[idx] = new_pt
                    test_cr = covering_radius_fast(test_pts)
                    if test_cr < cr:
                        pts = test_pts
                        cr = test_cr
                        if cr < best_cr:
                            best_cr = cr
                            best_pts = pts.copy()
        
        return best_pts, best_cr

    # Run Riemannian SGD
    max_iters = 5000
    pts, best_cr = riemannian_sgd(pts, max_iters, deadline)
    best_pts = pts

    # Try multiple restarts with different initializations
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Different initialization: rotate the Fibonacci spiral
        angle = np.random.uniform(0, 2 * np.pi)
        axis = np.random.randn(3)
        axis = axis / np.linalg.norm(axis)
        
        def rotate(pts, axis, angle):
            # Rodrigues' rotation formula
            c = np.cos(angle)
            s = np.sin(angle)
            k = axis
            K = np.array([[0, -k[2], k[1]],
                          [k[2], 0, -k[0]],
                          [-k[1], k[0], 0]])
            R = c * np.eye(3) + s * K + (1 - c) * np.outer(k, k)
            return pts @ R.T
        
        init_pts = normalize(rotate(fibonacci_spiral(N), axis, angle))
        
        # Also try a different spiral pattern
        if restart == 2:
            init_pts = normalize(fibonacci_spiral(N) * (-1))  # Antipodal
        
        pts2, cr2 = riemannian_sgd(init_pts, max_iters // 2, deadline)
        
        if cr2 < best_cr:
            best_cr = cr2
            best_pts = pts2

    # Final check: ensure no coincident points
    norms = np.linalg.norm(best_pts, axis=1)
    best_pts = best_pts / norms[:, None]
    
    # Check for coincident points
    for i in range(N):
        for j in range(i + 1, N):
            if np.linalg.norm(best_pts[i] - best_pts[j]) < 1e-6:
                # Separate them
                best_pts[j] = best_pts[i] + np.random.randn(3) * 1e-3
                best_pts[j] = best_pts[j] / np.linalg.norm(best_pts[j])
    
    return best_pts
# EVOLVE-BLOCK-END
