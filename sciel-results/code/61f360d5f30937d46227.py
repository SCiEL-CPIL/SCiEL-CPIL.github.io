"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    def fibonacci_spiral(n):
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(0, 1.0 - z * z))
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    def compute_facet_circumcenters(pts):
        """
        For each facet of the convex hull, compute the circumcenter (the point on the sphere
        equidistant from the facet's vertices) and the angular radius.
        Returns arrays of circumcenters (M, 3) and radii (M,).
        """
        try:
            hull = ConvexHull(pts)
        except Exception:
            return None, None, None

        simplices = hull.simplices
        M = len(simplices)
        circumcenters = np.zeros((M, 3))
        radii = np.zeros(M)
        facet_indices = np.zeros((M, 3), dtype=int)

        for k, simplex in enumerate(simplices):
            v1 = pts[simplex[0]]
            v2 = pts[simplex[1]]
            v3 = pts[simplex[2]]
            facet_indices[k] = simplex

            edge1 = v2 - v1
            edge2 = v3 - v1
            n = np.cross(edge1, edge2)
            norm_n = np.linalg.norm(n)
            if norm_n < 1e-12:
                circumcenters[k] = (v1 + v2 + v3) / 3.0
                circumcenters[k] /= np.linalg.norm(circumcenters[k])
                d = np.linalg.norm(circumcenters[k] - v1)
                radii[k] = 2.0 * np.arcsin(np.clip(d / 2.0, 0.0, 1.0))
                continue

            c = n / norm_n
            # Ensure outward pointing
            if np.dot(c, v1) < 0:
                c = -c

            circumcenters[k] = c
            d = np.linalg.norm(c - v1)
            radii[k] = 2.0 * np.arcsin(np.clip(d / 2.0, 0.0, 1.0))

        return circumcenters, radii, facet_indices

    def soft_max_objective(params, beta):
        """
        params: (N, 3) flat array of points on the sphere (not necessarily normalized).
        Returns (value, gradient) for the soft-max of facet circumradii.
        """
        pts = params.reshape(N, 3)
        # Normalize
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        pts_norm = pts / norms

        circumcenters, radii, _ = compute_facet_circumcenters(pts_norm)
        if circumcenters is None:
            return 1e6, np.zeros_like(params)

        # Soft max: f = (1/beta) * log(sum(exp(beta * radii)))
        # For numerical stability
        max_r = np.max(radii)
        if max_r < 1e-15:
            return 0.0, np.zeros_like(params)

        scaled = beta * (radii - max_r)
        # Clip for numerical stability
        scaled = np.clip(scaled, -100, 100)
        log_sum = np.log(np.sum(np.exp(scaled)))
        value = max_r + np.log(np.sum(np.exp(scaled))) / beta

        # Gradient via finite differences or analytical
        # For simplicity and speed, use a subgradient approximation:
        # The soft-max gradient weights each facet by its softmax probability.
        weights = np.exp(scaled) / np.sum(np.exp(scaled))
        
        # The gradient of the covering radius w.r.t. each point is a weighted sum
        # of the gradients of the individual facet radii.
        # For a facet with circumcenter c and vertices v1, v2, v3:
        # The radius r satisfies cos(r) = c . v1 = c . v2 = c . v3
        # dr/dv1 = -(c - (c.v1)v1) / sin(r)  (direction of movement on sphere)
        # More precisely, if we move v1 in direction u (tangent to sphere),
        # d(c.v1)/du = c . u, and dr = -d(c.v1)/sin(r)
        # So gradient of r w.r.t. v1 is -c / sin(r) projected to tangent space
        # = -(c - (c.v1)v1) / sin(r)
        
        grad = np.zeros((N, 3))
        sin_r = np.sin(radii)
        sin_r = np.maximum(sin_r, 1e-12)

        for k in range(len(radii)):
            w = weights[k]
            if w < 1e-10:
                continue
            c = circumcenters[k]
            simplex = _  # We lost this, need to recompute or store
            # Actually we need the facet indices. Let's restructure.
        
        # This approach is getting complex. Let's use a simpler subgradient:
        # Move each point away from its nearest circumcenter, weighted by proximity.
        # Or better: use the analytical gradient properly.
        
        # Let me recompute with stored indices
        return value, grad  # Placeholder - will fix below

    def soft_max_objective_v2(params, beta):
        """
        Proper soft-max objective with gradient.
        params: (N, 3) flat array.
        """
        pts = params.reshape(N, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        pts_norm = pts / norms

        circumcenters, radii, facet_indices = compute_facet_circumcenters(pts_norm)
        if circumcenters is None:
            return 1e6, np.zeros_like(params)

        max_r = np.max(radii)
        if max_r < 1e-15:
            return 0.0, np.zeros_like(params)

        scaled = beta * (radii - max_r)
        scaled = np.clip(scaled, -100, 100)
        exp_scaled = np.exp(scaled)
        sum_exp = np.sum(exp_scaled)
        weights = exp_scaled / sum_exp
        value = max_r + np.log(sum_exp) / beta

        # Gradient computation
        grad = np.zeros((N, 3))
        sin_r = np.sin(radii)
        sin_r = np.maximum(sin_r, 1e-12)

        for k in range(len(radii)):
            w = weights[k]
            if w < 1e-10:
                continue
            
            c = circumcenters[k]
            indices = facet_indices[k]
            
            for idx in indices:
                v = pts_norm[idx]
                # Gradient of radius r_k w.r.t. point v:
                # r_k = 2*arcsin(||c - v||/2), so dr/dv = -(c-v)/||c-v|| * ... 
                # Actually: cos(r) = c.v, so -sin(r)*dr = c.dv, dr = -c.dv/sin(r)
                # For tangent movement: gradient on sphere is -(c - (c.v)v)/sin(r)
                c_dot_v = np.dot(c, v)
                grad_vec = -(c - c_dot_v * v) / sin_r[k]
                grad[idx] += w * grad_vec

        return value, grad.ravel()

    def lloyd_iteration(pts, num_iters=50, step_size=0.1):
        """
        Lloyd-like iteration: move each point toward the circumcenter of its 
        "Voronoi cell" approximated by the nearest facet circumcenter.
        """
        for it in range(num_iters):
            if time.time() > deadline - 15:
                break
                
            circumcenters, radii, _ = compute_facet_circumcenters(pts)
            if circumcenters is None:
                break
            
            max_r = np.max(radii)
            if max_r < 1e-10:
                break

            # For each point, find the nearest circumcenter and move toward it
            # This is an approximation of Lloyd's algorithm
            new_pts = pts.copy()
            for i in range(N):
                # Find nearest circumcenter
                dists = np.linalg.norm(circumcenters - pts[i], axis=1)
                nearest_k = np.argmin(dists)
                target = circumcenters[nearest_k]
                
                # Move toward target
                dir_vec = target - pts[i]
                norm_dir = np.linalg.norm(dir_vec)
                if norm_dir > 1e-10:
                    dir_vec /= norm_dir
                    new_pts[i] = pts[i] + step_size * dir_vec
                    new_pts[i] /= np.linalg.norm(new_pts[i])
            
            # Check improvement
            _, new_radii, _ = compute_facet_circumcenters(new_pts)
            if new_radii is not None and np.max(new_radii) < max_r:
                pts = new_pts
            else:
                # Try smaller step
                step_size *= 0.5
                if step_size < 0.001:
                    break

        return pts

    def random_perturbation(pts, scale=0.05):
        """Add small random perturbation and renormalize."""
        perturbed = pts + np.random.randn(N, 3) * scale
        perturbed /= np.linalg.norm(perturbed, axis=1, keepdims=True)
        return perturbed

    def local_polish(pts, max_iters=100):
        """
        Local polishing using coordinate descent with adaptive step sizes.
        """
        current_pts = pts.copy()
        _, current_radii, _ = compute_facet_circumcenters(current_pts)
        if current_radii is None:
            return current_pts
        current_max = np.max(current_radii)

        step_sizes = np.ones(N) * 0.05

        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            improved = False
            for i in range(N):
                if time.time() > deadline - 5:
                    break
                
                # Try multiple random directions
                best_new = current_pts[i].copy()
                best_new_max = current_max

                for _ in range(8):
                    direction = np.random.randn(3)
                    direction /= np.linalg.norm(direction)
                    
                    new_p = current_pts[i] + step_sizes[i] * direction
                    new_p /= np.linalg.norm(new_p)
                    
                    trial_pts = current_pts.copy()
                    trial_pts[i] = new_p
                    
                    _, trial_radii, _ = compute_facet_circumcenters(trial_pts)
                    if trial_radii is not None:
                        trial_max = np.max(trial_radii)
                        if trial_max < best_new_max:
                            best_new_max = trial_max
                            best_new = new_p

                if best_new_max < current_max:
                    current_pts[i] = best_new
                    current_max = best_new_max
                    step_sizes[i] = min(step_sizes[i] * 1.2, 0.2)
                    improved = True
                else:
                    step_sizes[i] *= 0.7

            if not improved:
                # Try larger random perturbations
                for i in range(N):
                    if time.time() > deadline - 5:
                        break
                    for _ in range(3):
                        direction = np.random.randn(3)
                        direction /= np.linalg.norm(direction)
                        for scale in [0.02, 0.05, 0.1]:
                            new_p = current_pts[i] + scale * direction
                            new_p /= np.linalg.norm(new_p)
                            trial_pts = current_pts.copy()
                            trial_pts[i] = new_p
                            _, trial_radii, _ = compute_facet_circumcenters(trial_pts)
                            if trial_radii is not None and np.max(trial_radii) < current_max:
                                current_pts[i] = new_p
                                current_max = np.max(trial_radii)
                                improved = True
                                break
            if not improved:
                break

        return current_pts

    # Main optimization
    best_points = None
    best_radius = np.inf

    # Generate initial configurations
    initial_configs = []
    initial_configs.append(fibonacci_spiral(N))
    
    # Add some random initializations
    for _ in range(5):
        random_pts = np.random.randn(N, 3)
        random_pts /= np.linalg.norm(random_pts, axis=1, keepdims=True)
        initial_configs.append(random_pts)

    # Process each initial configuration
    for init_idx, init_pts in enumerate(initial_configs):
        if time.time() > deadline - 20:
            break

        current_pts = init_pts.copy()

        # Phase 1: Lloyd iteration to get a good starting point
        current_pts = lloyd_iteration(current_pts, num_iters=30, step_size=0.1)

        # Phase 2: LBFGS optimization on the sphere using soft-max surrogate
        if time.time() < deadline - 15:
            # Flatten for optimizer
            x0 = current_pts.ravel()
            
            # Use a moderate beta for the soft-max
            beta = 50.0
            
            result = minimize(
                soft_max_objective_v2,
                x0,
                args=(beta,),
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.success or result.fun < best_radius:
                candidate = result.x.reshape(N, 3)
                candidate /= np.linalg.norm(candidate, axis=1, keepdims=True)
                _, radii, _ = compute_facet_circumcenters(candidate)
                if radii is not None:
                    r = np.max(radii)
                    if r < best_radius:
                        best_radius = r
                        best_points = candidate.copy()

        # Phase 3: Additional Lloyd iterations
        if time.time() < deadline - 10:
            current_pts = lloyd_iteration(best_points if best_points is not None else current_pts, 
                                         num_iters=20, step_size=0.05)
            _, radii, _ = compute_facet_circumcenters(current_pts)
            if radii is not None:
                r = np.max(radii)
                if r < best_radius:
                    best_radius = r
                    best_points = current_pts.copy()

        # Phase 4: Local polish
        if time.time() < deadline - 5:
            polished = local_polish(best_points if best_points is not None else current_pts, max_iters=50)
            _, radii, _ = compute_facet_circumcenters(polished)
            if radii is not None:
                r = np.max(radii)
                if r < best_radius:
                    best_radius = r
                    best_points = polished.copy()

    # Final verification and normalization
    if best_points is None:
        best_points = fibonacci_spiral(N)
    
    best_points = best_points / np.linalg.norm(best_points, axis=1, keepdims=True)
    
    return best_points
# EVOLVE-BLOCK-END
