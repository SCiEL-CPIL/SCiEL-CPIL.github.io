# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 13.

    Uses a parametrization based on a regular 13-gon with radial perturbations.
    Optimizes the radial amplitudes to maximize the ratio A_min / A_hull.

    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    start_time = time.time()
    deadline = start_time + 600.0
    safety_margin = 10.0

    # Base angles for a regular 13-gon
    base_angles = np.linspace(0, 2 * np.pi, n, endpoint=False)

    def compute_ratio(radii):
        """Compute A_min / A_hull for given radii."""
        # Ensure radii are positive
        radii = np.maximum(radii, 0.1)
        
        # Compute points
        x = radii * np.cos(base_angles)
        y = radii * np.sin(base_angles)
        points = np.column_stack((x, y))
        
        # Check for degeneracy
        if np.allclose(points[:, 0], points[0, 0]) or np.allclose(points[:, 1], points[0, 1]):
            return 0.0
        
        try:
            # Compute convex hull area
            hull = ConvexHull(points)
            A_hull = hull.volume
            if A_hull <= 0:
                return 0.0
        except Exception:
            return 0.0
        
        # Compute all triangle areas
        # Number of triangles: C(13, 3) = 286
        min_area = np.inf
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    p1 = points[i]
                    p2 = points[j]
                    p3 = points[k]
                    area = abs(p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1])) / 2.0
                    if area < min_area:
                        min_area = area
        
        if min_area <= 0:
            return 0.0
        
        return min_area / A_hull

    def compute_ratio_vectorized(radii):
        """Vectorized computation of A_min / A_hull for given radii."""
        radii = np.maximum(radii, 0.1)
        
        # Compute points
        x = radii * np.cos(base_angles)
        y = radii * np.sin(base_angles)
        points = np.column_stack((x, y))
        
        # Check for degeneracy
        if np.allclose(points[:, 0], points[0, 0]) or np.allclose(points[:, 1], points[0, 1]):
            return 0.0
        
        try:
            hull = ConvexHull(points)
            A_hull = hull.volume
            if A_hull <= 0:
                return 0.0
        except Exception:
            return 0.0
        
        # Vectorized triangle area computation
        # Get all combinations of 3 indices
        i_idx = np.repeat(np.arange(n), n * (n - 1) // 2)
        j_idx = np.tile(np.repeat(np.arange(n), n - 1), n)
        k_idx = []
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    k_idx.append(k)
        k_idx = np.array(k_idx)
        
        p1 = points[i_idx]
        p2 = points[j_idx]
        p3 = points[k_idx]
        
        areas = np.abs(p1[:, 0] * (p2[:, 1] - p3[:, 1]) + 
                       p2[:, 0] * (p3[:, 1] - p1[:, 1]) + 
                       p3[:, 0] * (p1[:, 1] - p2[:, 1])) / 2.0
        
        min_area = np.min(areas)
        if min_area <= 0:
            return 0.0
        
        return min_area / A_hull

    # Initial guess: all radii equal (regular polygon)
    best_radii = np.ones(n)
    best_score = compute_ratio_vectorized(best_radii)
    
    # Simulated annealing
    rng = np.random.default_rng(seed=42)
    T = 0.1
    T_min = 1e-6
    alpha = 0.995
    current_radii = best_radii.copy()
    current_score = best_score
    
    # Precompute index arrays for vectorized evaluation
    i_idx = np.repeat(np.arange(n), n * (n - 1) // 2)
    j_idx = np.tile(np.repeat(np.arange(n), n - 1), n)
    k_idx = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                k_idx.append(k)
    k_idx = np.array(k_idx)
    
    def fast_ratio(radii):
        """Fast ratio computation with precomputed indices."""
        radii = np.maximum(radii, 0.1)
        x = radii * np.cos(base_angles)
        y = radii * np.sin(base_angles)
        points = np.column_stack((x, y))
        
        if np.allclose(points[:, 0], points[0, 0]) or np.allclose(points[:, 1], points[0, 1]):
            return 0.0
        
        try:
            hull = ConvexHull(points)
            A_hull = hull.volume
            if A_hull <= 0:
                return 0.0
        except Exception:
            return 0.0
        
        p1 = points[i_idx]
        p2 = points[j_idx]
        p3 = points[k_idx]
        
        areas = np.abs(p1[:, 0] * (p2[:, 1] - p3[:, 1]) + 
                       p2[:, 0] * (p3[:, 1] - p1[:, 1]) + 
                       p3[:, 0] * (p1[:, 1] - p2[:, 1])) / 2.0
        
        min_area = np.min(areas)
        if min_area <= 0:
            return 0.0
        
        return min_area / A_hull

    best_score = fast_ratio(best_radii)
    current_score = best_score
    current_radii = best_radii.copy()
    
    no_improve_count = 0
    max_no_improve = 5000
    
    while T > T_min and time.time() < deadline - safety_margin:
        # Propose a move
        move_type = rng.random()
        if move_type < 0.3:
            # Perturb one radius
            idx = rng.integers(0, n)
            delta = rng.normal(0, 0.05)
            new_radii = current_radii.copy()
            new_radii[idx] += delta
        elif move_type < 0.6:
            # Perturb a pair of radii
            idx1, idx2 = rng.choice(n, size=2, replace=False)
            delta = rng.normal(0, 0.03, size=2)
            new_radii = current_radii.copy()
            new_radii[idx1] += delta[0]
            new_radii[idx2] += delta[1]
        else:
            # Small global perturbation
            new_radii = current_radii + rng.normal(0, 0.02, size=n)
        
        # Ensure radii stay positive
        new_radii = np.maximum(new_radii, 0.1)
        
        new_score = fast_ratio(new_radii)
        
        # Acceptance criterion
        if new_score >= current_score:
            current_radii = new_radii
            current_score = new_score
            no_improve_count = 0
            if new_score > best_score:
                best_score = new_score
                best_radii = new_radii.copy()
        else:
            delta_e = current_score - new_score
            if delta_e < 1e-10:
                current_radii = new_radii
                current_score = new_score
                no_improve_count = 0
                if new_score > best_score:
                    best_score = new_score
                    best_radii = new_radii.copy()
            else:
                accept_prob = np.exp(-delta_e / T)
                if rng.random() < accept_prob:
                    current_radii = new_radii
                    current_score = new_score
                no_improve_count += 1
        
        T *= alpha
        
        if no_improve_count > max_no_improve:
            # Restart with slightly different initial conditions
            current_radii = best_radii + rng.normal(0, 0.01, size=n)
            current_radii = np.maximum(current_radii, 0.1)
            current_score = fast_ratio(current_radii)
            no_improve_count = 0
            T = max(T, 0.01)
    
    # Final verification
    final_points = np.column_stack((best_radii * np.cos(base_angles), best_radii * np.sin(base_angles)))
    final_score = compute_ratio(final_points)
    
    if final_score <= 0:
        # Fallback to regular polygon
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        final_points = np.column_stack((np.cos(angles), np.sin(angles)))
    
    return final_points


# EVOLVE-BLOCK-END
