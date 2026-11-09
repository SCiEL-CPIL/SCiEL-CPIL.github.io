# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist

def min_max_dist_dim3_14() -> np.ndarray:
    start_time = time.time()
    timeout = 600
    n = 14

    def calculate_score(points):
        dists = pdist(points)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def project_to_sphere(points):
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1e-12, norms)
        return points / norms

    def optimize_energy(points, iterations=10000, lr=0.01, beta=4.0):
        """
        Minimize repulsive energy E = -sum d_ij^-beta on sphere.
        Vectorized: compute full distance matrix each iteration.
        """
        points = project_to_sphere(points.copy())
        n_pts = len(points)
        eye_mask = ~np.eye(n_pts, dtype=bool)

        for i in range(iterations):
            if time.time() - start_time > timeout - 10:
                break

            # Vectorized distance computation
            diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
            dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
            dist_matrix = np.sqrt(dist_sq)
            dist_matrix[~eye_mask] = np.where(dist_matrix[~eye_mask] < 1e-10, 1e-10, dist_matrix[~eye_mask])
            dist_matrix[eye_mask] = 1.0  # self distances, will be masked

            # Gradient of E = -sum_{i<j} d_ij^-beta
            # dE/dp_i = beta * sum_{j != i} d_ij^(-beta-2) * (p_i - p_j)
            # We compute for all i at once:
            # weights[i,j] = beta * d_ij^(-beta-2) for i != j, 0 otherwise
            weights = np.zeros_like(dist_matrix)
            weights[eye_mask] = beta * np.power(dist_matrix[eye_mask], -beta - 2)
            # Make symmetric: weights[i,j] should be beta * d_ij^(-beta-2)
            # For the gradient at point i: sum_j weights[i,j] * (p_i - p_j)
            # = sum_j weights[i,j] * diff[i,j]
            grad = np.einsum('ij,ijk->ik', weights, diff)

            # Project to tangent space
            for k in range(n_pts):
                p_k = points[k]
                g_k = grad[k]
                proj = np.dot(g_k, p_k)
                grad[k] = g_k - proj * p_k

            # Move against gradient (minimize energy)
            points = points - lr * grad
            points = project_to_sphere(points)

            # Decay learning rate
            if i % 200 == 0 and i > 0:
                lr *= 0.97

        return points

    def polish_min_dist(points, iterations=5000, lr=0.003):
        """
        Polish to maximize minimum pairwise distance.
        Only moves points involved in the closest pairs.
        """
        points = project_to_sphere(points.copy())
        n_pts = len(points)
        eye_mask = ~np.eye(n_pts, dtype=bool)

        for i in range(iterations):
            if time.time() - start_time > timeout - 5:
                break

            diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
            dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
            dist_matrix = np.sqrt(dist_sq)
            dist_matrix[eye_mask] = dist_matrix[eye_mask]
            dist_matrix[~eye_mask] = np.inf

            d_min = np.min(dist_matrix)

            # Find all pairs at minimum distance
            min_pairs = np.argwhere(dist_matrix <= d_min + 1e-12)

            grad = np.zeros_like(points)
            for idx in min_pairs:
                i_idx, j_idx = int(idx[0]), int(idx[1])
                vec = points[i_idx] - points[j_idx]
                d = np.linalg.norm(vec)
                if d > 1e-10:
                    unit = vec / d
                    grad[i_idx] += unit
                    grad[j_idx] -= unit

            # Project to tangent space
            for k in range(n_pts):
                p_k = points[k]
                g_k = grad[k]
                proj = np.dot(g_k, p_k)
                grad[k] = g_k - proj * p_k

            # Ascend on min distance (move points apart)
            points = points + lr * grad
            points = project_to_sphere(points)

            if i > 1000:
                lr *= 0.9995

        return points

    def fibonacci_sphere(n_pts, R=1.0):
        golden = (1 + np.sqrt(5)) / 2
        i = np.arange(n_pts)
        z = 1 - (2 * i + 1) / n_pts
        r = np.sqrt(np.maximum(0, 1 - z**2))
        theta = 2 * np.pi * i / golden
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        pts = np.column_stack([x, y, z])
        return pts * R

    best_score = 0.0
    best_points = None

    # Generate multiple initializations
    inits = []

    # 1. Fibonacci sphere
    inits.append(fibonacci_sphere(n))

    # 2. Random initializations with different seeds
    for seed in [42, 123, 456, 789]:
        np.random.seed(seed)
        inits.append(np.random.randn(n, 3))

    # 3. Spherical cap variations - start from fibonacci with perturbation
    np.random.seed(999)
    inits.append(fibonacci_sphere(n) + np.random.randn(n, 3) * 0.2)

    for init_idx, init in enumerate(inits):
        if time.time() - start_time > timeout - 30:
            break

        pts = project_to_sphere(init.copy())

        # Phase 1: Energy minimization to spread points
        pts1 = optimize_energy(pts, iterations=8000, lr=0.015, beta=4.0)
        s1 = calculate_score(pts1)
        if s1 > best_score:
            best_score = s1
            best_points = pts1.copy()

        # Phase 2: Polish minimum distance
        pts2 = polish_min_dist(pts1, iterations=4000, lr=0.003)
        s2 = calculate_score(pts2)
        if s2 > best_score:
            best_score = s2
            best_points = pts2.copy()

        # Phase 3: Another round of energy + polish from polished result
        if time.time() - start_time < timeout - 60:
            pts3 = optimize_energy(pts2, iterations=5000, lr=0.008, beta=6.0)
            pts4 = polish_min_dist(pts3, iterations=3000, lr=0.002)
            s4 = calculate_score(pts4)
            if s4 > best_score:
                best_score = s4
                best_points = pts4.copy()

    if best_points is None:
        best_points = fibonacci_sphere(n)

    return best_points

# EVOLVE-BLOCK-END
