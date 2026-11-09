# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex 20-gon with equidistant vertices using turn-angle parameterization.
    
    Key structural change: parameterize by 19 turn angles (positive, summing to 2*pi),
    which guarantees convexity by construction. This avoids penalty-based convexity
    enforcement and reduces dimensionality.
    """
    start_time = time.time()
    deadline = start_time + 620.0  # Safety margin before 630s kill
    
    def compute_polygon_from_angles(turn_angles_19):
        """
        Given 19 turn angles (positive), compute the 20 vertices of a closed convex polygon.
        The 20th turn angle is determined by closure: 2*pi - sum(first 19).
        All 20 angles must be positive for strict convexity.
        """
        n = 20
        last_angle = 2.0 * np.pi - np.sum(turn_angles_19)
        if last_angle <= 1e-10:
            return None
        
        all_angles = np.append(turn_angles_19, last_angle)
        
        # Edge directions: cumulative sum of turn angles
        # Start with edge 0 in direction 0 (positive x-axis)
        edge_dirs = np.zeros(n)
        for i in range(1, n):
            edge_dirs[i] = edge_dirs[i-1] + all_angles[i-1]
        
        # Unit edge vectors
        ex = np.cos(edge_dirs)
        ey = np.sin(edge_dirs)
        
        # Solve for edge lengths l_0..l_19 such that polygon closes:
        # sum(l_i * e_i) = 0
        # This is a 2x20 system. We have 18 degrees of freedom for lengths.
        # Strategy: fix l_0 = 1, solve for l_1..l_19, then check non-negativity.
        # Actually, we need to find ANY positive solution.
        
        # The closure condition: A @ l = 0 where A is 2x20
        A = np.column_stack([ex, ey])
        
        # Use SVD to find the null space
        U, S, Vt = np.linalg.svd(A, full_matrices=False)
        # Null space vectors are the last rows of Vt
        # For a 2x20 matrix, null space has dimension 18
        null_space = Vt[2:, :].T  # 20x18
        
        # We need a positive vector in the null space.
        # Strategy: try to find l = alpha * null_space @ c for some c such that l > 0.
        # This is a linear programming problem, but let's use a simpler heuristic.
        
        # Alternative: parametrize 18 lengths freely, solve for the other 2.
        # Fix l_18 and l_19 as free, solve for l_0..l_17.
        # A[:, :18] @ l[:18] = -A[:, 18:] @ l[18:]
        
        A_left = A[:, :18]
        A_right = A[:, 18:]
        
        # Solve A_left @ l[:18] = -A_right @ l[18:]
        # For a given l[18:], l[:18] = -A_left^{-1} @ A_right @ l[18:]
        # But A_left is 2x18, not square. Use least squares.
        
        # Actually, let's use a different approach: 
        # Choose 18 edge lengths freely (positive), then the remaining 2 are determined.
        # But we need all 20 to be positive.
        
        # Simpler: use the fact that for a convex polygon with given turn angles,
        # there exists a unique (up to scale) set of positive edge lengths.
        # We can find it by solving the linear system with non-negativity constraints.
        
        # Use scipy's linprog or just try a simple approach:
        # Set all lengths to 1, check if it closes. If not, adjust.
        
        # Actually, the cleanest approach: the null space of A is 18-dimensional.
        # We want a vector in the null space with all positive entries.
        # This is equivalent to finding a point in the intersection of the null space
        # with the positive orthant.
        
        # Heuristic: try random combinations of null space basis vectors
        # and check if the result is positive.
        
        best_l = None
        rng_local = np.random.default_rng(12345)
        
        for attempt in range(100):
            # Random combination of null space basis
            c = rng_local.dirichlet(np.ones(18))
            l_candidate = null_space @ c
            # Check if all positive
            if np.all(l_candidate > 1e-10):
                best_l = l_candidate
                break
        
        if best_l is None:
            # Try with negative combinations
            for attempt in range(100):
                c = rng_local.dirichlet(np.ones(18))
                l_candidate = null_space @ c
                if np.all(-l_candidate > 1e-10):
                    best_l = -l_candidate
                    break
        
        if best_l is None:
            return None
        
        # Compute vertices
        vertices = np.zeros((n, 2))
        for i in range(1, n):
            vertices[i] = vertices[i-1] + best_l[i-1] * np.array([ex[i-1], ey[i-1]])
        
        return vertices
    
    def normalize_polygon(vertices):
        """Subtract centroid, divide by mean distance to centroid."""
        centroid = np.mean(vertices, axis=0)
        centered = vertices - centroid
        dists = np.sqrt(np.sum(centered ** 2, axis=1))
        mean_dist = np.mean(dists)
        if mean_dist <= 0.01:
            return None
        return centered / mean_dist
    
    def vertex_score(v, all_vertices):
        """Compute the score for a single vertex v given all other vertices."""
        n = len(all_vertices)
        # Distances from v to all other vertices
        diffs = all_vertices - v
        dists = np.sqrt(np.sum(diffs ** 2, axis=1))
        
        # Exclude self (distance 0)
        # Find index of v
        idx = np.argmin(dists)
        other_dists = np.delete(dists, idx)
        other_dists = np.sort(other_dists)
        
        # Check if any distance is < 0.05
        if np.any(other_dists < 0.05):
            return 0.0
        
        # Find k: first index minimizing d_{k+1} - d_k
        diffs_arr = np.diff(other_dists)
        k = np.argmin(diffs_arr)
        D = (other_dists[k] + other_dists[k + 1]) / 2.0
        
        if D < 0.1:
            return 0.0
        
        # Take four distances
        if k == 0:
            selected = other_dists[0:4]
        elif k == 17:
            selected = other_dists[15:19]
        else:
            selected = other_dists[k-1:k+3]
        
        # raw = mean of max(d/D, D/d) for these four
        ratios = np.maximum(selected / D, D / selected)
        raw = np.mean(ratios)
        
        return min(1.0, 1.0 / raw)
    
    def min_vertex_score(vertices):
        """Compute the minimum vertex score over all 20 vertices."""
        scores = []
        for i in range(20):
            s = vertex_score(vertices[i], vertices)
            scores.append(s)
        return min(scores)
    
    def objective(turn_angles_19):
        """
        Objective function for optimization.
        Returns negative of min vertex score (to minimize).
        Adds penalty if polygon is invalid.
        """
        vertices = compute_polygon_from_angles(turn_angles_19)
        if vertices is None:
            return 1000.0
        
        # Check for repeated points
        for i in range(20):
            for j in range(i + 1, 20):
                if np.linalg.norm(vertices[i] - vertices[j]) < 1e-6:
                    return 1000.0
        
        # Normalize
        normalized = normalize_polygon(vertices)
        if normalized is None:
            return 1000.0
        
        # Compute min vertex score
        score = min_vertex_score(normalized)
        return -score
    
    # Initial guess: regular 20-gon
    # For a regular 20-gon, all turn angles are equal: 2*pi/20 = pi/10
    regular_angle = 2.0 * np.pi / 20.0
    initial_angles = np.full(19, regular_angle)
    
    # Compute initial polygon and score
    initial_vertices = compute_polygon_from_angles(initial_angles)
    if initial_vertices is None:
        # Fallback to regular polygon
        angles = np.linspace(0.0, 2.0 * np.pi, 20, endpoint=False)
        initial_vertices = np.column_stack([np.cos(angles), np.sin(angles)])
    
    initial_normalized = normalize_polygon(initial_vertices)
    if initial_normalized is None:
        initial_normalized = initial_vertices
    
    best_vertices = initial_normalized.copy()
    best_score = min_vertex_score(best_vertices)
    
    # Multi-restart optimization using Nelder-Mead (handles non-smooth objectives)
    num_restarts = 30
    rng = np.random.default_rng(42)
    
    for restart in range(num_restarts):
        if time.time() > deadline - 30.0:
            break
        
        # Generate diverse initial point
        if restart == 0:
            init_angles = initial_angles.copy()
        else:
            # Perturb the initial angles
            # Keep them positive and sum to less than 2*pi
            noise_scale = 0.05 + 0.2 * (restart / num_restarts)
            init_angles = initial_angles + rng.normal(0, noise_scale, 19)
            # Ensure positivity
            init_angles = np.maximum(init_angles, 0.01)
            # Ensure sum < 2*pi
            current_sum = np.sum(init_angles)
            if current_sum >= 2.0 * np.pi:
                scale = (2.0 * np.pi - 0.01) / current_sum
                init_angles = init_angles * scale
        
        # Optimize using Nelder-Mead
        try:
            result = minimize(
                objective,
                init_angles,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            
            if result.fun < -best_score:
                candidate_angles = result.x
                candidate_vertices = compute_polygon_from_angles(candidate_angles)
                
                if candidate_vertices is not None:
                    # Check validity
                    valid = True
                    for i in range(20):
                        for j in range(i + 1, 20):
                            if np.linalg.norm(candidate_vertices[i] - candidate_vertices[j]) < 1e-6:
                                valid = False
                                break
                        if not valid:
                            break
                    
                    if valid:
                        candidate_normalized = normalize_polygon(candidate_vertices)
                        if candidate_normalized is not None:
                            candidate_score = min_vertex_score(candidate_normalized)
                            if candidate_score > best_score:
                                best_score = candidate_score
                                best_vertices = candidate_normalized.copy()
        except Exception:
            continue
    
    # Try differential evolution for broader search if time permits
    if time.time() < deadline - 60.0:
        try:
            bounds = [(0.01, 2.0 * np.pi / 3) for _ in range(19)]
            de_result = differential_evolution(
                objective,
                bounds,
                seed=42,
                maxiter=100,
                tol=1e-8,
                polish=True
            )
            
            if de_result.fun < -best_score:
                candidate_angles = de_result.x
                candidate_vertices = compute_polygon_from_angles(candidate_angles)
                
                if candidate_vertices is not None:
                    valid = True
                    for i in range(20):
                        for j in range(i + 1, 20):
                            if np.linalg.norm(candidate_vertices[i] - candidate_vertices[j]) < 1e-6:
                                valid = False
                                break
                        if not valid:
                            break
                    
                    if valid:
                        candidate_normalized = normalize_polygon(candidate_vertices)
                        if candidate_normalized is not None:
                            candidate_score = min_vertex_score(candidate_normalized)
                            if candidate_score > best_score:
                                best_score = candidate_score
                                best_vertices = candidate_normalized.copy()
        except Exception:
            pass
    
    return best_vertices


# EVOLVE-BLOCK-END
