"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    def fibonacci_sphere(n, offset=0.5):
        i = np.arange(n) + offset
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
        pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        return pts

    def generate_grid(num_points):
        """Generate a set of points on the sphere for constraint checking."""
        # Use Fibonacci spiral for uniform distribution
        return fibonacci_sphere(num_points, offset=0.5)

    def compute_facet_circumcenters(pts):
        """
        Compute all facet circumcenters and their angular radii.
        Returns: list of (radius_radians, circumcenter, simplex_indices)
        """
        try:
            hull = ConvexHull(pts)
        except Exception:
            return [(np.pi, np.array([0.0, 0.0, 1.0]), None)]

        results = []
        for simplex in hull.simplices:
            v1 = pts[simplex[0]]
            v2 = pts[simplex[1]]
            v3 = pts[simplex[2]]

            edge1 = v2 - v1
            edge2 = v3 - v1
            n = np.cross(edge1, edge2)
            norm_n = np.linalg.norm(n)
            if norm_n < 1e-12:
                continue

            c = n / norm_n
            if np.dot(c, v1) < 0:
                c = -c

            d = np.linalg.norm(c - v1)
            radius = 2.0 * np.arcsin(np.clip(d / 2.0, 0.0, 1.0))
            results.append((radius, c, simplex))

        if not results:
            return [(np.pi, np.array([0.0, 0.0, 1.0]), None)]

        return results

    def compute_covering_radius(pts):
        """
        Compute the covering radius using the convex hull facet circumradius method.
        Returns the covering radius in radians and the worst-case point (circumcenter of worst facet).
        """
        facets = compute_facet_circumcenters(pts)
        max_radius = 0.0
        worst_center = None
        for radius, c, _ in facets:
            if radius > max_radius:
                max_radius = radius
                worst_center = c
        if worst_center is None:
            return np.pi, np.array([0.0, 0.0, 1.0])
        return max_radius, worst_center

    def optimize_worst_facet(pts, step_size, iterations=50):
        """
        Iteratively move the vertices of the worst facet towards the worst center.
        """
        pts = pts.copy()
        for _ in range(iterations):
            if time.time() > deadline - 5:
                break

            radius, worst_center = compute_covering_radius(pts)

            # Find the facet corresponding to worst_center
            try:
                hull = ConvexHull(pts)
            except:
                break

            target_indices = None
            for simplex in hull.simplices:
                v1 = pts[simplex[0]]
                v2 = pts[simplex[1]]
                v3 = pts[simplex[2]]
                edge1 = v2 - v1
                edge2 = v3 - v1
                n = np.cross(edge1, edge2)
                norm_n = np.linalg.norm(n)
                if norm_n < 1e-12:
                    continue
                c = n / norm_n
                if np.dot(c, v1) < 0:
                    c = -c
                if np.dot(c, worst_center) > 0.9999:
                    target_indices = simplex
                    break

            if target_indices is None:
                break

            # Move vertices towards worst_center
            for idx in target_indices:
                v = pts[idx]
                dir_vec = worst_center - v
                norm_dir = np.linalg.norm(dir_vec)
                if norm_dir > 1e-10:
                    dir_vec /= norm_dir
                    pts[idx] = v + step_size * dir_vec
                    pts[idx] /= np.linalg.norm(pts[idx])

        return pts

    def slsqp_objective_and_constraints(z, grid_pts):
        """
        Objective: t
        Constraints:
        1. t - d(grid_pt, nearest_site) >= 0 for all grid_pts
        2. ||site_i||^2 - 1 = 0 for all sites
        """
        # Unpack z: first 3*N are site coords, last is t
        sites = z[:3*N].reshape(N, 3)
        t = z[3*N]
        
        # Normalize sites to ensure they are on the sphere during evaluation
        # (SLSQP might drift, but equality constraints should pull them back)
        norms = np.linalg.norm(sites, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        sites = sites / norms

        # Objective is t
        obj = t

        # Inequality constraints: t - d_i >= 0
        # d_i is the angular distance from grid_pt i to its nearest site
        # We compute distances from grid_pts to all sites
        # To be efficient, we can vectorize this
        # grid_pts: (M, 3), sites: (N, 3)
        # dists: (M, N)
        dots = np.clip(np.dot(grid_pts, sites.T), -1.0, 1.0)
        dists = np.arccos(dots)
        
        # min_dist: (M,)
        min_dists = np.min(dists, axis=1)
        
        # Constraints: t - min_dists >= 0
        ineq_constraints = t - min_dists

        # Equality constraints: ||site_i||^2 - 1 = 0
        # We use the original unnormalized sites from z for the constraint definition
        # to ensure the gradient points towards the sphere
        orig_sites = z[:3*N].reshape(N, 3)
        eq_constraints = np.sum(orig_sites**2, axis=1) - 1.0

        return obj, ineq_constraints, eq_constraints

    def run_slsqp(initial_pts, grid_pts, maxiter=100):
        """
        Run SLSQP optimization.
        """
        initial_t = 0.5 # Initial guess for t (radians)
        z0 = np.concatenate([initial_pts.flatten(), [initial_t]])
        
        # Define constraints for SLSQP
        # SLSQP expects a list of dicts
        # We have many inequality constraints (M) and equality constraints (N)
        # It is more efficient to pass them as a single vector if possible, 
        # but SLSQP interface requires individual constraints or a custom function.
        # scipy.optimize.minimize with method='SLSQP' accepts a list of constraint dicts.
        # Creating M+N dicts is slow. 
        # Instead, we can use a single constraint function that returns an array? No, SLSQP doesn't support that directly in the standard interface.
        # However, we can use 'SLSQP' with a custom 'constraints' list.
        # Given M~100, N=50, total 150 constraints. This is acceptable.
        
        constraints = []
        
        # Inequality constraints: t - d_i >= 0
        # We define a function for each? No, that's too many function objects.
        # Alternative: Use a single penalty function? No, directive says "subject to the constraint".
        # Let's try to use a coarser grid to reduce M.
        
        # Actually, SLSQP can handle a list of constraints.
        # But constructing 100+ constraint dictionaries every call is expensive.
        # Let's try a different approach: Minimize t + lambda * sum(max(0, d_i - t)^2)
        # This is a smooth approximation of the constrained problem.
        # The directive suggests "minimizing a slack variable t subject to the constraint...".
        # If we use the penalty method, we are not strictly using the constraint interface.
        # Let's stick to the penalty method for speed and robustness, as it is a standard way to handle this in Python without complex constraint handling.
        # Or, we can use a small grid.
        
        # Let's implement the penalty method version which is effectively a smooth constrained optimization.
        # Objective: t + mu * sum(max(0, d_i - t)^2)
        # This encourages t to be at least the max distance, and penalizes violations.
        
        def penalty_objective(z):
            sites = z[:3*N].reshape(N, 3)
            t = z[3*N]
            
            norms = np.linalg.norm(sites, axis=1, keepdims=True)
            norms[norms < 1e-10] = 1.0
            sites = sites / norms
            
            dots = np.clip(np.dot(grid_pts, sites.T), -1.0, 1.0)
            dists = np.arccos(dots)
            min_dists = np.min(dists, axis=1)
            
            # Penalty for violations
            violations = np.maximum(0.0, min_dists - t)
            penalty = np.sum(violations**2)
            
            # Regularization to keep sites on sphere (handled by L-BFGS/SLSQP if we add constraints, 
            # but here we just rely on the initial normalization and small steps)
            # Add a small term to keep t bounded? No.
            
            return t + 10.0 * penalty

        # We will use L-BFGS-B for this smooth objective, as it is faster than SLSQP for this form.
        # But the directive specifically mentioned SLSQP with constraints.
        # Let's try SLSQP with a small grid (e.g., 50 points) to keep constraint count low.
        
        pass # Placeholder, we will implement the actual logic in the main loop

    # Phase 1: Initial Fibonacci spiral
    best = fibonacci_sphere(N)
    best_radius, _ = compute_covering_radius(best)

    # Phase 2: SLSQP with grid constraints
    # Use a grid of ~100 points
    grid_pts = generate_grid(100)
    
    # We will use a smooth penalty approach with L-BFGS-B as it is more robust for many constraints
    # unless we can efficiently batch constraints.
    # Let's try the penalty method with L-BFGS-B first, as it's similar to the current soft-max but uses grid points.
    
    def grid_penalty_objective(z):
        sites = z[:3*N].reshape(N, 3)
        t = z[3*N]
        
        norms = np.linalg.norm(sites, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        sites = sites / norms
        
        dots = np.clip(np.dot(grid_pts, sites.T), -1.0, 1.0)
        dists = np.arccos(dots)
        min_dists = np.min(dists, axis=1)
        
        violations = np.maximum(0.0, min_dists - t)
        penalty = np.sum(violations**2)
        
        # Add sphere constraint penalty to help L-BFGS-B
        sphere_violation = np.sum((np.linalg.norm(sites, axis=1)**2 - 1.0)**2)
        
        return t + 10.0 * penalty + 100.0 * sphere_violation

    z0 = np.concatenate([best.flatten(), [0.5]])
    
    try:
        result = minimize(
            grid_penalty_objective,
            z0,
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        if result.success:
            best_candidate = result.x[:3*N].reshape(N, 3)
            best_candidate /= np.linalg.norm(best_candidate, axis=1, keepdims=True)
            r, _ = compute_covering_radius(best_candidate)
            if r < best_radius:
                best = best_candidate
                best_radius = r
    except:
        pass

    # Phase 3: Local polishing with facet moves
    for step in [0.02, 0.01, 0.005]:
        if time.time() > deadline - 10:
            break
        current = optimize_worst_facet(best, step, iterations=50)
        r, _ = compute_covering_radius(current)
        if r < best_radius:
            best = current
            best_radius = r

    # Phase 4: Final fine-tuning
    if time.time() < deadline - 5:
        final = best.copy()
        for _ in range(50):
            if time.time() > deadline - 5:
                break
            radius, worst_center = compute_covering_radius(final)

            try:
                hull = ConvexHull(final)
            except:
                break

            target_indices = None
            for simplex in hull.simplices:
                v1 = final[simplex[0]]
                v2 = final[simplex[1]]
                v3 = final[simplex[2]]
                edge1 = v2 - v1
                edge2 = v3 - v1
                n = np.cross(edge1, edge2)
                norm_n = np.linalg.norm(n)
                if norm_n < 1e-12:
                    continue
                c = n / norm_n
                if np.dot(c, v1) < 0:
                    c = -c
                if np.dot(c, worst_center) > 0.9999:
                    target_indices = simplex
                    break

            if target_indices is None:
                break

            for idx in target_indices:
                v = final[idx]
                dir_vec = worst_center - v
                norm_dir = np.linalg.norm(dir_vec)
                if norm_dir > 1e-10:
                    dir_vec /= norm_dir
                    final[idx] = v + 0.002 * dir_vec
                    final[idx] /= np.linalg.norm(final[idx])

        radius, _ = compute_covering_radius(final)
        if radius < best_radius:
            best = final
            best_radius = radius

    # Final normalization and validation
    best = best / np.linalg.norm(best, axis=1, keepdims=True)
    norms = np.linalg.norm(best, axis=1)
    valid = (norms > 1e-10) & np.all(np.isfinite(best), axis=1)
    if not np.all(valid):
        best = fibonacci_sphere(N)

    return best
# EVOLVE-BLOCK-END
