import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

def min_max_dist_dim3_14() -> np.ndarray:
    n = 14
    d = 3
    start_time = time.time()
    deadline = start_time + 590  # safety margin for 630s kill

    def score(pts):
        """Compute (d_min / d_max)^2 for the point configuration."""
        pts = np.asarray(pts, dtype=float)
        if pts.shape != (n, d) or not np.all(np.isfinite(pts)):
            return 0.0
        dists = pdist(pts)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-12:
            return 0.0
        return (d_min / d_max) ** 2

    def neg_score(pts):
        return -score(pts)

    def coulomb_energy(pts, k=1.0, eps=1e-6):
        """Smooth surrogate: minimize sum of 1/r^2 between all pairs."""
        pts = np.asarray(pts, dtype=float)
        if pts.shape != (n, d) or not np.all(np.isfinite(pts)):
            return 1e10
        dists = pdist(pts)
        dists = np.maximum(dists, eps)
        return k * np.sum(1.0 / (dists ** 2))

    def smooth_penalty(pts, target_ratio=1.0, power=4):
        """Smooth penalty that encourages equalizing all pairwise distances."""
        pts = np.asarray(pts, dtype=float)
        if pts.shape != (n, d) or not np.all(np.isfinite(pts)):
            return 1e10
        dists = pdist(pts)
        if len(dists) == 0:
            return 1e10
        d_mean = np.mean(dists)
        if d_mean < 1e-12:
            return 1e10
        # Penalty for deviation from uniform distance
        var = np.var(dists) / (d_mean ** 2)
        # Also penalize if min is too small relative to max
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-12:
            return 1e10
        ratio = d_min / d_max
        return var + (1.0 - ratio) ** power

    # Generate diverse initial configurations
    def fibonacci_sphere(n_pts, seed_offset=0):
        rng = np.random.RandomState(42 + seed_offset)
        indices = np.arange(0, n_pts, dtype=float) + 0.5
        phi_f = np.arccos(1 - 2 * indices / n_pts)
        theta = np.pi * (1 + 5**0.5) * indices
        # Add small random perturbation for diversity
        pert = rng.randn(n_pts, 3) * 0.02
        x = np.cos(theta) * np.sin(phi_f)
        y = np.sin(theta) * np.sin(phi_f)
        z = np.cos(phi_f)
        pts = np.column_stack([x, y, z]) + pert
        return pts

    def icosahedron_plus_2():
        phi = (1 + np.sqrt(5)) / 2
        ico12 = np.array([
            [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
            [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
        ], dtype=float)
        ico12 = ico12 / np.linalg.norm(ico12, axis=1, keepdims=True)
        # Add 2 points at origin and one more direction
        extra = np.array([[0, 0, 0], [0.5, 0.5, 0.5]])
        extra = extra / np.linalg.norm(extra, axis=1, keepdims=True)
        return np.vstack([ico12, extra])

    def random_config(rng, scale=1.0):
        return rng.randn(n, d) * scale

    def structured_config(rng):
        # Try to place points with some structure
        pts = np.zeros((n, d))
        # Place 12 points roughly on a sphere
        for i in range(12):
            pts[i] = rng.randn(3)
            pts[i] /= np.linalg.norm(pts[i])
        # Place 2 points inside
        pts[12] = rng.randn(3) * 0.3
        pts[13] = rng.randn(3) * 0.3
        return pts

    def icosahedron_perturbed(rng, seed_offset=0):
        """Generate icosahedron-based configs with perturbations."""
        base = icosahedron_plus_2()
        pert = rng.randn(n, d) * 0.05
        return base + pert

    # Collect initial candidates
    candidates = []
    rng = np.random.RandomState(12345)

    # Fibonacci sphere variants
    for i in range(5):
        candidates.append(fibonacci_sphere(n, seed_offset=i))

    # Icosahedron-based with perturbations
    for i in range(20):
        candidates.append(icosahedron_perturbed(rng, seed_offset=i))

    # Random configurations
    for i in range(8):
        scale = 0.5 + rng.rand()
        candidates.append(random_config(rng, scale))

    # Structured configurations
    for i in range(10):
        candidates.append(structured_config(rng))

    # Evaluate all candidates and keep best
    best_score = -np.inf
    best_points = None

    for i, cand in enumerate(candidates):
        if time.time() > deadline - 60:
            break
        s = score(cand)
        if s > best_score:
            best_score = s
            best_points = cand.copy()

    if best_points is None:
        best_points = candidates[0]

    # Now refine with multiple local optimization strategies
    # Strategy 1: Coulomb repulsion to spread points evenly
    if time.time() < deadline - 180:
        def coulomb_obj(x):
            pts = x.reshape(n, d)
            return coulomb_energy(pts)

        # Run Coulomb optimization from best point and several perturbations
        coulomb_starts = [best_points.copy()]
        for i in range(10):
            perturbed = best_points + rng.randn(n, d) * 0.2
            coulomb_starts.append(perturbed)

        for start in coulomb_starts:
            if time.time() > deadline - 150:
                break
            try:
                result = minimize(
                    coulomb_obj,
                    start.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
                pts = result.x.reshape(n, d)
                if np.all(np.isfinite(pts)):
                    s = score(pts)
                    if s > best_score:
                        best_score = s
                        best_points = pts.copy()
            except Exception:
                pass

    # Strategy 2: Smooth penalty optimization
    if time.time() < deadline - 120:
        def penalty_obj(x):
            pts = x.reshape(n, d)
            return smooth_penalty(pts)

        penalty_starts = [best_points.copy()]
        for i in range(10):
            perturbed = best_points + rng.randn(n, d) * 0.15
            penalty_starts.append(perturbed)

        for start in penalty_starts:
            if time.time() > deadline - 90:
                break
            try:
                result = minimize(
                    penalty_obj,
                    start.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
                pts = result.x.reshape(n, d)
                if np.all(np.isfinite(pts)):
                    s = score(pts)
                    if s > best_score:
                        best_score = s
                        best_points = pts.copy()
            except Exception:
                pass

    # Strategy 3: Nelder-Mead local refinement on the actual score
    def nelder_mead_refine(start_pts, max_iter=500):
        def obj(x):
            pts = x.reshape(n, d)
            return -score(pts)

        try:
            result = minimize(
                obj,
                start_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            pts = result.x.reshape(n, d)
            if np.all(np.isfinite(pts)):
                return pts
        except Exception:
            pass
        return None

    # Refine from best point and several random perturbations
    refine_starts = [best_points.copy()]
    for i in range(15):
        perturbed = best_points + rng.randn(n, d) * 0.1
        refine_starts.append(perturbed)

    for start in refine_starts:
        if time.time() > deadline - 30:
            break
        refined = nelder_mead_refine(start, max_iter=300)
        if refined is not None:
            s = score(refined)
            if s > best_score:
                best_score = s
                best_points = refined.copy()

    # Strategy 4: More targeted local search - adjust points to increase d_min
    # Focus on pairs that are too close
    if time.time() < deadline - 30:
        pts = best_points.copy()
        for iteration in range(50):
            if time.time() > deadline - 15:
                break
            dists = pdist(pts)
            d_min = np.min(dists)
            d_max = np.max(dists)
            if d_max < 1e-12:
                break
            current_score = (d_min / d_max) ** 2

            # Find the minimum distance pair
            min_idx = np.argmin(dists)
            # Convert linear index to (i, j)
            count = 0
            i_min = j_min = -1
            for i in range(n):
                for j in range(i + 1, n):
                    if count == min_idx:
                        i_min = i
                        j_min = j
                        break
                    count += 1
                if i_min != -1:
                    break

            # Find the maximum distance pair
            max_idx = np.argmax(dists)
            count = 0
            i_max = j_max = -1
            for i in range(n):
                for j in range(i + 1, n):
                    if count == max_idx:
                        i_max = i
                        j_max = j
                        break
                    count += 1
                if i_max != -1:
                    break

            # Try to increase d_min by moving i_min and j_min apart
            diff = pts[i_min] - pts[j_min]
            dist = np.linalg.norm(diff)
            if dist > 1e-12:
                direction = diff / dist
                step = 0.01
                pts[i_min] += direction * step
                pts[j_min] -= direction * step

            # Try to decrease d_max by moving i_max and j_max closer
            diff_max = pts[i_max] - pts[j_max]
            dist_max = np.linalg.norm(diff_max)
            if dist_max > 1e-12:
                direction_max = diff_max / dist_max
                step_max = 0.01
                pts[i_max] -= direction_max * step_max
                pts[j_max] += direction_max * step_max

            s_new = score(pts)
            if s_new > current_score:
                best_score = max(best_score, s_new)
                best_points = pts.copy()
            else:
                pts = best_points.copy()  # revert

    # Final normalization
    best_points = best_points - np.mean(best_points, axis=0)
    norm_max = np.max(np.linalg.norm(best_points, axis=1))
    if norm_max > 0:
        best_points = best_points / norm_max * 100

    if not np.all(np.isfinite(best_points)):
        best_points = np.zeros((n, d))

    return best_points
