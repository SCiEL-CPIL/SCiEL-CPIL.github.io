import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint

def min_max_dist_dim3_14() -> np.ndarray:
    n = 14
    d = 3
    start_time = time.time()
    deadline = start_time + 595  # safety margin before 630s kill

    def true_score(points):
        """Calculates (d_min / d_max)^2"""
        dists_sq = pdist(points, metric='sqeuclidean')
        if len(dists_sq) == 0:
            return 0.0
        d_min_sq = np.min(dists_sq)
        d_max_sq = np.max(dists_sq)
        if d_max_sq < 1e-12:
            return 0.0
        return (d_min_sq / d_max_sq) ** 2

    def normalize_rms(points):
        """Normalizes points to have RMS radius 1"""
        rms = np.sqrt(np.mean(np.sum(points ** 2, axis=1)))
        if rms < 1e-6:
            return points
        return points / rms

    def smooth_obj(x_flat):
        """Smooth surrogate: minimize sum of inverse squared distances"""
        pts = x_flat.reshape(n, d)
        pts = normalize_rms(pts)
        dists_sq = pdist(pts, metric='sqeuclidean')
        dists_sq = np.maximum(dists_sq, 1e-8)
        return np.sum(1.0 / dists_sq)

    def smooth_obj_grad(x_flat):
        """Gradient of smooth_obj"""
        pts = x_flat.reshape(n, d)
        pts = normalize_rms(pts)
        dists_sq = pdist(pts, metric='sqeuclidean')
        dists_sq = np.maximum(dists_sq, 1e-8)
        # Compute gradient numerically would be expensive; use finite differences
        # For L-BFGS-B, let's use analytic gradient approximation
        # Actually, let's just use the function without gradient and let scipy handle it
        # or provide a simple finite difference
        eps = 1e-7
        grad = np.zeros_like(x_flat)
        for i in range(len(x_flat)):
            x_plus = x_flat.copy()
            x_plus[i] += eps
            x_minus = x_flat.copy()
            x_minus[i] -= eps
            grad[i] = (smooth_obj(x_plus) - smooth_obj(x_minus)) / (2 * eps)
        return grad

    def min_dist_obj(x_flat, target_min_dist=0.0):
        """Objective to maximize minimum distance by minimizing negative min dist with constraint on max dist"""
        pts = x_flat.reshape(n, d)
        pts = normalize_rms(pts)
        dists_sq = pdist(pts, metric='sqeuclidean')
        d_min_sq = np.min(dists_sq)
        d_max_sq = np.max(dists_sq)
        if d_max_sq < 1e-12:
            return 1.0  # penalize
        # We want to maximize d_min_sq / d_max_sq
        # Minimize -d_min_sq / d_max_sq
        return -d_min_sq / d_max_sq

    # Generate initial candidates
    candidates = []

    # Candidate 1: Icosahedron vertices + 2 poles
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    ico = np.array([
        [0.0, 1.0, phi], [0.0, -1.0, phi], [0.0, 1.0, -phi], [0.0, -1.0, -phi],
        [1.0, phi, 0.0], [-1.0, phi, 0.0], [1.0, -phi, 0.0], [-1.0, -phi, 0.0],
        [phi, 0.0, 1.0], [-phi, 0.0, 1.0], [phi, 0.0, -1.0], [-phi, 0.0, -1.0]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    poles = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    candidates.append(np.vstack([ico, poles]))

    # Candidate 2: Regular dodecagon in xy-plane + 2 poles
    angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    dodec = np.column_stack([np.cos(angles), np.sin(angles), np.zeros(12)])
    candidates.append(np.vstack([dodec, poles]))

    # Candidate 3: Fibonacci sphere for 14 points
    def fibonacci_sphere(n_points):
        points = []
        golden_ratio = (1 + np.sqrt(5)) / 2
        for i in range(n_points):
            z = 1 - 2 * (i + 0.5) / n_points
            r = np.sqrt(1 - z * z)
            theta = np.pi * (1 + np.sqrt(5)) * i
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            points.append([x, y, z])
        return np.array(points)

    fib_pts = fibonacci_sphere(14)
    fib_pts = fib_pts / np.linalg.norm(fib_pts, axis=1, keepdims=True)
    candidates.append(fib_pts)

    # Candidate 4: Cuboctahedron-like (12 vertices) + 2 poles
    cube = np.array([
        [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
        [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]
    ], dtype=float)
    cube = cube / np.sqrt(3.0)
    octa = np.array([
        [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]
    ], dtype=float)
    all_pts = np.vstack([cube, octa])
    all_pts = normalize_rms(all_pts)
    candidates.append(all_pts)

    # Candidate 5: Random points on sphere (multiple seeds)
    rng = np.random.RandomState(42)
    for seed in range(10):
        rng = np.random.RandomState(seed)
        pts = rng.randn(n, d)
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        candidates.append(pts)

    # Candidate 6: Antipodal pairs (7 pairs)
    rng = np.random.RandomState(123)
    for seed in range(5):
        rng = np.random.RandomState(seed)
        pts_half = rng.randn(7, d)
        pts_half = pts_half / np.linalg.norm(pts_half, axis=1, keepdims=True)
        pts_full = np.vstack([pts_half, -pts_half])
        candidates.append(pts_full)

    best_score = 0.0
    best_points = None

    # Score all candidates first
    for cand in candidates:
        cand = normalize_rms(cand)
        s = true_score(cand)
        if s > best_score:
            best_score = s
            best_points = cand.copy()

    # Optimization phases
    for i, cand in enumerate(candidates):
        if time.time() > deadline - 30:
            break

        cand = normalize_rms(cand)

        try:
            # Phase 1: L-BFGS-B on smooth surrogate
            res1 = minimize(
                smooth_obj,
                cand.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            pts1 = normalize_rms(res1.x.reshape(n, d))

            # Phase 2: Nelder-Mead on true objective
            def nm_obj(x_flat):
                pts = x_flat.reshape(n, d)
                pts = normalize_rms(pts)
                return -true_score(pts)

            res2 = minimize(
                nm_obj,
                pts1.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-13, 'fatol': 1e-15,
                         'adaptive': True}
            )
            pts2 = normalize_rms(res2.x.reshape(n, d))

            s = true_score(pts2)
            if s > best_score:
                best_score = s
                best_points = pts2.copy()

        except Exception:
            continue

    # Additional local searches from best
    if best_points is not None and time.time() < deadline - 40:
        for _ in range(5):
            if time.time() > deadline - 15:
                break
            try:
                perturbed = best_points + np.random.randn(n, d) * 0.1
                perturbed = normalize_rms(perturbed)

                res1 = minimize(
                    smooth_obj,
                    perturbed.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                pts1 = normalize_rms(res1.x.reshape(n, d))

                def nm_obj(x_flat):
                    pts = x_flat.reshape(n, d)
                    pts = normalize_rms(pts)
                    return -true_score(pts)

                res2 = minimize(
                    nm_obj,
                    pts1.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 3000, 'xatol': 1e-13, 'fatol': 1e-15,
                             'adaptive': True}
                )
                pts2 = normalize_rms(res2.x.reshape(n, d))
                s = true_score(pts2)
                if s > best_score:
                    best_score = s
                    best_points = pts2.copy()
            except Exception:
                continue

    # Fallback
    if best_points is None:
        phi = (1.0 + np.sqrt(5.0)) / 2.0
        ico = np.array([
            [0.0, 1.0, phi], [0.0, -1.0, phi], [0.0, 1.0, -phi], [0.0, -1.0, -phi],
            [1.0, phi, 0.0], [-1.0, phi, 0.0], [1.0, -phi, 0.0], [-1.0, -phi, 0.0],
            [phi, 0.0, 1.0], [-phi, 0.0, 1.0], [phi, 0.0, -1.0], [-phi, 0.0, -1.0]
        ], dtype=float)
        ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
        best_points = np.vstack([ico, np.array([[0.0, 0.0, 1.0]]), np.array([[0.0, 0.0, -1.0]])])

    if not np.all(np.isfinite(best_points)):
        phi = (1.0 + np.sqrt(5.0)) / 2.0
        ico = np.array([
            [0.0, 1.0, phi], [0.0, -1.0, phi], [0.0, 1.0, -phi], [0.0, -1.0, -phi],
            [1.0, phi, 0.0], [-1.0, phi, 0.0], [1.0, -phi, 0.0], [-1.0, -phi, 0.0],
            [phi, 0.0, 1.0], [-phi, 0.0, 1.0], [phi, 0.0, -1.0], [-phi, 0.0, -1.0]
        ], dtype=float)
        ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
        best_points = np.vstack([ico, np.array([[0.0, 0.0, 1.0]]), np.array([[0.0, 0.0, -1.0]])])

    return best_points
