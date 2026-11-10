# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize


def min_max_dist_dim3_14() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580  # safety margin before 630s kill

    n = 14
    d = 3

    def compute_score(points):
        """Compute (d_min/d_max)^2"""
        if points.shape != (n, d):
            return 0.0
        if not np.all(np.isfinite(points)):
            return 0.0
        dists = pdist(points)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-12:
            return 0.0
        return (d_min / d_max) ** 2

    def objective_ratio(points):
        """Negative of (d_min/d_max)^2 for minimization"""
        pts = points.reshape(n, d)
        dists = pdist(pts)
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-12:
            return 0.0
        return -(d_min / d_max) ** 2

    def objective_soft_min(points, beta=50.0):
        """Soft minimum of distances, negative for minimization"""
        pts = points.reshape(n, d)
        dists = pdist(pts)
        # Soft min using log-sum-exp
        scaled = -dists / beta
        max_scaled = np.max(scaled)
        log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
        soft_min = -beta * log_sum
        return -soft_min

    def objective_soft_ratio(points, beta=30.0):
        """Soft approximation of (d_min/d_max)^2"""
        pts = points.reshape(n, d)
        dists = pdist(pts)
        # Soft min
        scaled_min = -dists / beta
        max_scaled_min = np.max(scaled_min)
        log_sum_min = max_scaled_min + np.log(np.sum(np.exp(scaled_min - max_scaled_min)))
        soft_min = -beta * log_sum_min
        # Soft max
        scaled_max = dists / beta
        max_scaled_max = np.max(scaled_max)
        log_sum_max = max_scaled_max + np.log(np.sum(np.exp(scaled_max - max_scaled_max)))
        soft_max = beta * log_sum_max
        if soft_max < 1e-12:
            return 0.0
        return -(soft_min / soft_max) ** 2

    def icosahedron_initialization():
        """Generate icosahedron vertices and add 4 more points"""
        phi = (1 + np.sqrt(5)) / 2.0
        verts = np.array([
            [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
            [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
            [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
        ], dtype=float)
        # Normalize
        norms = np.linalg.norm(verts, axis=1, keepdims=True)
        verts = verts / norms
        # Add 4 more points: use face centers of icosahedron
        # Or just add points at positions that are roughly symmetric
        # Use the 12 icosahedron vertices plus 4 additional
        # For 14 points, we can use a truncated icosahedron or just add 4 more
        # Let's add 4 points at (1,1,1), (1,-1,-1), (-1,1,-1), (-1,-1,1) normalized
        extra = np.array([
            [1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]
        ], dtype=float)
        extra_norms = np.linalg.norm(extra, axis=1, keepdims=True)
        extra = extra / extra_norms
        points = np.vstack([verts, extra])
        # Normalize all to unit sphere
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms
        return points

    def dodecahedron_initialization():
        """Generate dodecahedron vertices"""
        phi = (1 + np.sqrt(5)) / 2.0
        verts = []
        # 8 vertices from cube
        for x in [-1, 1]:
            for y in [-1, 1]:
                for z in [-1, 1]:
                    verts.append([x, y, z])
        # 4 vertices from rectangle in xy plane
        for a in [-1/phi, 1/phi]:
            for b in [-phi, phi]:
                verts.append([0, a, b])
                verts.append([a, b, 0])
                verts.append([b, 0, a])
        verts = np.array(verts, dtype=float)
        norms = np.linalg.norm(verts, axis=1, keepdims=True)
        verts = verts / norms
        # Dodecahedron has 20 vertices, we need 14
        # Select 14 that are well-spread
        # Just take first 14
        points = verts[:14]
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms
        return points

    def random_initialization(seed):
        np.random.seed(seed)
        points = np.random.randn(n, d)
        # Normalize to unit sphere
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms
        return points

    def fibonacci_sphere(n_points):
        """Generate points on a sphere using Fibonacci lattice"""
        points = []
        golden_angle = np.pi * (3 - np.sqrt(5))
        for i in range(n_points):
            y = 1 - (i / (n_points - 1)) * 2
            radius = np.sqrt(1 - y * y)
            theta = golden_angle * i
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            points.append([x, y, z])
        return np.array(points)

    best_points = None
    best_score = 0.0

    # Generate diverse initializations
    initializations = []
    initializations.append(icosahedron_initialization())
    initializations.append(dodecahedron_initialization())
    initializations.append(fibonacci_sphere(n))
    for seed in [42, 123, 456, 789, 1000, 2024, 3141, 2718]:
        initializations.append(random_initialization(seed))

    for init_idx, init_pts in enumerate(initializations):
        if time.time() > deadline - 30:
            break

        # Phase 1: Optimize soft min distance
        try:
            result1 = minimize(
                objective_soft_min,
                init_pts.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if result1.success or result1.fun < 0:
                pts1 = result1.x.reshape(n, d)
                # Normalize to unit sphere
                norms = np.linalg.norm(pts1, axis=1, keepdims=True)
                norms = np.where(norms < 1e-10, 1.0, norms)
                pts1 = pts1 / norms
            else:
                pts1 = init_pts
        except:
            pts1 = init_pts

        # Phase 2: Optimize soft ratio
        if time.time() > deadline - 20:
            break
        try:
            result2 = minimize(
                objective_soft_ratio,
                pts1.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if result2.success or result2.fun < 0:
                pts2 = result2.x.reshape(n, d)
                # Normalize to unit sphere
                norms = np.linalg.norm(pts2, axis=1, keepdims=True)
                norms = np.where(norms < 1e-10, 1.0, norms)
                pts2 = pts2 / norms
            else:
                pts2 = pts1
        except:
            pts2 = pts1

        # Phase 3: Fine-tune with exact ratio objective
        if time.time() > deadline - 10:
            break
        try:
            result3 = minimize(
                objective_ratio,
                pts2.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if result3.success or result3.fun < 0:
                pts3 = result3.x.reshape(n, d)
                # Normalize to unit sphere
                norms = np.linalg.norm(pts3, axis=1, keepdims=True)
                norms = np.where(norms < 1e-10, 1.0, norms)
                pts3 = pts3 / norms
            else:
                pts3 = pts2
        except:
            pts3 = pts2

        score = compute_score(pts3)
        if score > best_score:
            best_score = score
            best_points = pts3.copy()

    if best_points is None:
        # Fallback
        best_points = icosahedron_initialization()

    return best_points


# EVOLVE-BLOCK-END
