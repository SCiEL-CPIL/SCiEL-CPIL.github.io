"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.spatial import ConvexHull

N = 50

# EVOLVE-BLOCK-START
def construct():
    """
    Cover the unit sphere with 50 equal caps.
    Combined Strategy:
    1. Start with a Fibonacci spiral.
    2. Fast vectorized Lloyd's algorithm for coarse convergence.
    3. Deterministic gradient-based refinement targeting the worst convex hull facets.
    4. Final local polish with small steps.
    """
    start_time = time.time()
    timeout = 110.0  # Safety margin before 120s limit

    def fibonacci_sphere(n):
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    def compute_exact_covering_radius(points):
        """
        Compute the exact covering radius using the convex hull.
        Returns the max circumradius in degrees and the indices of the worst facet.
        """
        try:
            hull = ConvexHull(points)
        except Exception:
            # Fallback if hull fails
            return float('inf'), None, None
        
        max_radius = 0.0
        worst_facet = None
        worst_cc = None
        
        # Iterate over all facets (simplices)
        for simplex in hull.simplices:
            v1, v2, v3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
            
            # Compute circumcenter of the triangle v1, v2, v3
            e1 = v2 - v1
            e2 = v3 - v1
            
            A = np.array([
                [np.dot(e1, e1), np.dot(e1, e2)],
                [np.dot(e1, e2), np.dot(e2, e2)]
            ])
            b_vec = np.array([0.5 * np.dot(e1, e1), 0.5 * np.dot(e2, e2)])
            
            det = np.linalg.det(A)
            if abs(det) < 1e-12:
                # Degenerate case
                side1 = np.linalg.norm(v2 - v1)
                side2 = np.linalg.norm(v3 - v2)
                side3 = np.linalg.norm(v1 - v3)
                max_side = max(side1, side2, side3)
                rad = np.arcsin(np.clip(max_side / 2.0, -1.0, 1.0))
                deg = np.degrees(rad)
                if deg > max_radius:
                    max_radius = deg
                    worst_facet = simplex
                    # Approximate CC for degenerate case (midpoint of longest side)
                    if max_side == side1:
                        worst_cc = (v1 + v2) / 2.0
                    elif max_side == side2:
                        worst_cc = (v2 + v3) / 2.0
                    else:
                        worst_cc = (v1 + v3) / 2.0
                continue
                
            try:
                params = np.linalg.solve(A, b_vec)
            except np.linalg.LinAlgError:
                continue
                
            circumcenter = v1 + params[0] * e1 + params[1] * e2
            norm_cc = np.linalg.norm(circumcenter)
            if norm_cc < 1e-10:
                continue
            circumcenter_norm = circumcenter / norm_cc
            
            dot_val = np.clip(np.dot(circumcenter_norm, v1), -1.0, 1.0)
            angular_radius = np.arccos(dot_val)
            deg = np.degrees(angular_radius)
            
            if deg > max_radius:
                max_radius = deg
                worst_facet = simplex
                worst_cc = circumcenter_norm

        return max_radius, worst_facet, worst_cc

    def vectorized_lloyd_iteration(points, test_points, step_size=1.0):
        """One iteration of Lloyd's algorithm (vectorized)."""
        dot_products = test_points @ points.T
        dot_products = np.clip(dot_products, -1.0, 1.0)
        nearest = np.argmax(dot_products, axis=1)

        sums = np.zeros((len(points), 3))
        np.add.at(sums, nearest, test_points)
        
        counts = np.bincount(nearest, minlength=len(points))
        
        counts_safe = np.where(counts == 0, 1, counts)
        
        centroids = sums / counts_safe[:, np.newaxis]
        
        norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        new_points = centroids / norms
        
        empty_mask = counts == 0
        if np.any(empty_mask):
            new_points[empty_mask] = points[empty_mask]

        result = (1 - step_size) * points + step_size * new_points
        norms = np.linalg.norm(result, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        return result / norms

    # Step 1: Initial configuration
    points = fibonacci_sphere(N)
    best_points = points.copy()
    best_score, _, _ = compute_exact_covering_radius(points)
    
    # Step 2: Fast Vectorized Lloyd's
    # Use a moderate number of points for speed
    num_lloyd_test = 20000
    lloyd_test_points = fibonacci_sphere(num_lloyd_test)
    
    step_size = 1.0
    lloyd_iterations = 50
    
    for i in range(lloyd_iterations):
        if time.time() - start_time > timeout * 0.2:
            break
            
        points = vectorized_lloyd_iteration(points, lloyd_test_points, step_size)
        
        # Evaluate periodically to track best
        if i % 5 == 0 or i == lloyd_iterations - 1:
            current_score, _, _ = compute_exact_covering_radius(points)
            if current_score < best_score:
                best_score = current_score
                best_points = points.copy()
        
        # Reduce step size slightly for stability in later iterations
        if i > 20:
            step_size *= 0.95

    # Step 3: Deterministic Gradient Refinement on Worst Facets
    # This phase targets the specific facets causing the largest covering radius.
    current_points = best_points.copy()
    current_score, worst_facet, worst_cc = compute_exact_covering_radius(current_points)
    
    # If the initial hull calculation failed or score is inf, restart from a random config
    if np.isinf(current_score):
        current_points = fibonacci_sphere(N) * 0.9 + np.random.normal(0, 0.1, (N, 3))
        current_points /= np.linalg.norm(current_points, axis=1, keepdims=True)
        current_score, worst_facet, worst_cc = compute_exact_covering_radius(current_points)

    step_size_grad = 0.05
    min_step_size = 0.001
    no_improve_count = 0
    max_no_improve = 20
    
    phase_end_time = start_time + timeout * 0.9
    
    while time.time() < phase_end_time:
        if no_improve_count > max_no_improve:
            # Reset step size and try a different perturbation strategy if stuck
            step_size_grad = max(min_step_size, step_size_grad * 0.5)
            if step_size_grad < min_step_size:
                break
            no_improve_count = 0
        
        if worst_facet is None or worst_cc is None:
            # If we can't find a worst facet, do a small random jitter
            idx = np.random.randint(N)
            noise = np.random.normal(0, 0.01, 3)
            new_point = current_points[idx] + noise
            new_point /= np.linalg.norm(new_point)
            trial_points = current_points.copy()
            trial_points[idx] = new_point
            trial_score, _, _ = compute_exact_covering_radius(trial_points)
            if trial_score < current_score:
                current_points = trial_points
                current_score = trial_score
                if current_score < best_score:
                    best_score = current_score
                    best_points = current_points.copy()
                    no_improve_count = 0
            else:
                no_improve_count += 1
            continue

        # Identify vertices of the worst facet
        v_indices = worst_facet
        # Get the circumcenter direction
        cc_dir = worst_cc
        
        # Compute gradient direction for each vertex in the worst facet
        # We want to move vertices towards the circumcenter to reduce the circumradius
        # Direction = projection of (CC - V) onto tangent plane at V
        trial_points = current_points.copy()
        
        moved_any = False
        for idx in v_indices:
            v = current_points[idx]
            # Vector from vertex to circumcenter
            to_cc = cc_dir - v
            # Project onto tangent plane: remove radial component
            radial_comp = np.dot(to_cc, v) * v
            tangent_dir = to_cc - radial_comp
            norm_td = np.linalg.norm(tangent_dir)
            
            if norm_td > 1e-10:
                tangent_dir = tangent_dir / norm_td
                # Move in this direction
                new_v = v + step_size_grad * tangent_dir
                new_v /= np.linalg.norm(new_v)
                trial_points[idx] = new_v
                moved_any = True
        
        if not moved_any:
            # If no movement was possible, try a small random perturbation of the worst facet vertices
            for idx in v_indices:
                noise = np.random.normal(0, 0.01, 3)
                new_v = current_points[idx] + noise
                new_v /= np.linalg.norm(new_v)
                trial_points[idx] = new_v
            moved_any = True

        if moved_any:
            trial_score, new_worst_facet, new_worst_cc = compute_exact_covering_radius(trial_points)
            
            if trial_score < current_score:
                current_points = trial_points
                current_score = trial_score
                worst_facet = new_worst_facet
                worst_cc = new_worst_cc
                no_improve_count = 0
                # Increase step size slightly if we are improving fast? Or keep constant?
                # Let's keep constant for stability, but maybe increase if we were stuck.
                if current_score < best_score:
                    best_score = current_score
                    best_points = current_points.copy()
            else:
                no_improve_count += 1
                step_size_grad *= 0.95
                if step_size_grad < min_step_size:
                    step_size_grad = min_step_size

    # Final check and return
    final_score, _, _ = compute_exact_covering_radius(best_points)
    if final_score < best_score:
        best_score = final_score

    # Ensure points are normalized
    best_points = best_points / np.linalg.norm(best_points, axis=1, keepdims=True)
    
    return best_points
# EVOLVE-BLOCK-END
