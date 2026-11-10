# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

def min_max_dist_dim3_14() -> np.ndarray:
    n = 14
    d = 3
    start_time = time.time()
    deadline = start_time + 595.0  # safety margin before 630s kill

    def true_score(points):
        dists_sq = pdist(points, metric='sqeuclidean')
        if len(dists_sq) == 0:
            return 0.0
        d_min_sq = np.min(dists_sq)
        d_max_sq = np.max(dists_sq)
        if d_max_sq < 1e-15:
            return 0.0
        return d_min_sq / d_max_sq

    def normalize_rms(points):
        rms = np.sqrt(np.mean(np.sum(points**2, axis=1)))
        if rms < 1e-6:
            return points
        return points / rms

    def soft_max_dist_sq(dists_sq, k=50.0):
        m = np.max(dists_sq)
        if m < 1e-15:
            return 0.0
        shifted = dists_sq / k - m / k
        return k * (m / k + np.log(np.sum(np.exp(shifted))))

    def soft_min_dist_sq(dists_sq, k=50.0):
        m = np.min(dists_sq)
        if m < 1e-15:
            return 0.0
        shifted = -dists_sq / k + m / k
        return k * (-m / k - np.log(np.sum(np.exp(shifted))))

    def smooth_ratio_obj(x_flat, k=20.0):
        pts = x_flat.reshape(n, d)
        pts = normalize_rms(pts)
        dists_sq = pdist(pts, metric='sqeuclidean')
        d_max_sq = soft_max_dist_sq(dists_sq, k)
        d_min_sq = soft_min_dist_sq(dists_sq, k)
        if d_max_sq < 1e-10:
            return 0.0
        return -d_min_sq / d_max_sq

    def repulsion_obj(x_flat):
        pts = x_flat.reshape(n, d)
        pts = normalize_rms(pts)
        dists_sq = pdist(pts, metric='sqeuclidean')
        dists_sq = np.maximum(dists_sq, 1e-10)
        return np.sum(1.0 / dists_sq)

    def nm_obj(x_flat):
        pts = x_flat.reshape(n, d)
        pts = normalize_rms(pts)
        return -true_score(pts)

    best_score = -1.0
    best_points = None

    def _try_candidate(pts):
        nonlocal best_score, best_points
        if not np.all(np.isfinite(pts)):
            return
        pts = pts.reshape(n, d)
        score = true_score(pts)
        if score > best_score:
            best_score = score
            best_points = pts.copy()

    # Simulated Annealing refinement
    def anneal(initial_points, max_iters=40000, start_temp=0.05, end_temp=0.0005):
        nonlocal best_score, best_points
        current_points = initial_points.copy()
        current_score = true_score(current_points)
        if time.time() > deadline - 15:
            return current_points, current_score
        for i in range(max_iters):
            if time.time() > deadline - 15:
                break
            noise_scale = 0.04 * (1 - i / max_iters)
            perturbed = current_points + np.random.randn(n, d) * noise_scale
            perturbed = normalize_rms(perturbed)
            perturbed_score = true_score(perturbed)
            if perturbed_score > current_score:
                current_points = perturbed
                current_score = perturbed_score
            else:
                temp = start_temp * (end_temp / start_temp) ** (i / max_iters)
                if np.random.rand() < np.exp((perturbed_score - current_score) / temp):
                    current_points = perturbed
                    current_score = perturbed_score
            _try_candidate(current_points)
        return current_points, current_score

    np.random.seed(42)
    candidates = []

    # 1. Cuboctahedron (12) + 2 poles
    cuboct = np.array([
        [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0],
        [1, 0, 1], [1, 0, -1], [-1, 0, 1], [-1, 0, -1],
        [0, 1, 1], [0, 1, -1], [0, -1, 1], [0, -1, -1]
    ], dtype=float)
    cuboct = cuboct / np.sqrt(2.0)
    poles = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    candidates.append(np.vstack([cuboct, poles]))

    # 2. Icosahedron (12) + 2 poles
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    ico = np.array([
        [0.0, 1.0, phi], [0.0, -1.0, phi], [0.0, 1.0, -phi], [0.0, -1.0, -phi],
        [1.0, phi, 0.0], [-1.0, phi, 0.0], [1.0, -phi, 0.0], [-1.0, -phi, 0.0],
        [phi, 0.0, 1.0], [-phi, 0.0, 1.0], [phi, 0.0, -1.0], [-phi, 0.0, -1.0]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    candidates.append(np.vstack([ico, poles]))

    # 3. Dodecagon in xy-plane + 2 poles
    angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    dodec = np.column_stack([np.cos(angles), np.sin(angles), np.zeros(12)])
    candidates.append(np.vstack([dodec, poles]))

    # 4. Two hexagons at different z levels + 2 poles
    for z_off in [0.3, 0.5, 0.7]:
        ang1 = np.linspace(0, 2 * np.pi, 6, endpoint=False)
        ang2 = ang1 + np.pi / 6
        hex1 = np.column_stack([np.cos(ang1), np.sin(ang1), np.ones(6) * z_off])
        hex2 = np.column_stack([np.cos(ang2), np.sin(ang2), np.ones(6) * (-z_off)])
        all_pts = np.vstack([hex1, hex2, poles])
        candidates.append(normalize_rms(all_pts))

    # 5. Cube (8) + octahedron (6)
    cube = np.array([
        [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
        [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]
    ], dtype=float)
    cube = cube / np.sqrt(3.0)
    octa = np.array([
        [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]
    ], dtype=float)
    candidates.append(normalize_rms(np.vstack([cube, octa])))

    # 6. Antipodal pairs (7 + 7)
    for seed in range(8):
        rng = np.random.RandomState(seed)
        pts_upper = rng.randn(7, 3)
        pts_upper[:, 2] = np.abs(pts_upper[:, 2]) + 0.3
        pts_upper = pts_upper / np.linalg.norm(pts_upper, axis=1, keepdims=True)
        pts_lower = -pts_upper
        all_pts = np.vstack([pts_upper, pts_lower])
        candidates.append(normalize_rms(all_pts))

    # 7. Tilted dodecagon + poles
    for tilt in [0.1, 0.2, 0.3, 0.4, 0.5, 0.7]:
        dodec_t = np.column_stack([np.cos(angles), np.sin(angles), np.zeros(12)])
        c, s = np.cos(tilt), np.sin(tilt)
        R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        dodec_t = dodec_t @ R.T
        all_pts = np.vstack([dodec_t, poles])
        candidates.append(normalize_rms(all_pts))

    # 8. Cuboctahedron with tilted poles
    for pole_angle in [0.1, 0.2, 0.3, 0.5]:
        p1 = np.array([np.sin(pole_angle), 0.0, np.cos(pole_angle)])
        p2 = -p1
        all_pts = np.vstack([cuboct, p1.reshape(1, 3), p2.reshape(1, 3)])
        candidates.append(normalize_rms(all_pts))

    # 9. Icosahedron with moved poles
    for z_pole in [0.6, 0.8, 1.0, 1.3, 1.5]:
        p1 = np.array([0.0, 0.0, z_pole])
        p2 = np.array([0.0, 0.0, -z_pole])
        all_pts = np.vstack([ico, p1.reshape(1, 3), p2.reshape(1, 3)])
        candidates.append(normalize_rms(all_pts))

    # 10. Random starts
    for seed in range(10):
        rng = np.random.RandomState(seed + 100)
        rp = rng.randn(n, d)
        rp /= np.linalg.norm(rp, axis=1, keepdims=True)
        candidates.append(rp)

    # Evaluate all candidates
    for cand in candidates:
        _try_candidate(cand)

    # Optimize each candidate with Nelder-Mead (smooth ratio obj), then refine
    for i, init_pts in enumerate(candidates):
        if time.time() > deadline - 25:
            break
        x0 = init_pts.flatten()
        try:
            res = minimize(smooth_ratio_obj, x0, method='Nelder-Mead',
                           options={'maxiter': 2000, 'maxfev': 6000, 'xatol': 1e-8, 'fatol': 1e-10})
            candidate = res.x.reshape(n, d)
            _try_candidate(candidate)

            # Nelder-Mead on raw score from the smooth result
            if time.time() < deadline - 30:
                x1 = candidate.flatten()
                res2 = minimize(nm_obj, x1, method='Nelder-Mead',
                                options={'maxiter': 1500, 'maxfev': 5000, 'xatol': 1e-8, 'fatol': 1e-10})
                candidate2 = res2.x.reshape(n, d)
                _try_candidate(candidate2)
        except Exception:
            pass

    # Global diversification: repulsion (Fekete-like) from best point
    if time.time() < deadline - 40 and best_points is not None:
        try:
            x0 = best_points.flatten()
            res = minimize(repulsion_obj, x0, method='Nelder-Mead',
                           options={'maxiter': 2000, 'maxfev': 6000, 'xatol': 1e-8, 'fatol': 1e-10})
            candidate = res.x.reshape(n, d)
            _try_candidate(candidate)
        except Exception:
            pass

    # Final Simulated Annealing refinement from the best point found
    if time.time() < deadline - 20 and best_points is not None:
        sa_result, _ = anneal(best_points, max_iters=30000, start_temp=0.03, end_temp=0.0003)
        _try_candidate(sa_result)

    # Fallback
    if best_points is None:
        best_points = candidates[0]

    best_points = best_points.reshape(n, d)
    if not np.all(np.isfinite(best_points)):
        best_points = candidates[0]

    return best_points

# EVOLVE-BLOCK-END
