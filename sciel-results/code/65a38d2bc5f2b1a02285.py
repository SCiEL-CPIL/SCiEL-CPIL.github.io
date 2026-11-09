# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

_DEADLINE_SECONDS = 630.0
_SAFETY_MARGIN = 30.0


def _score(points: np.ndarray) -> float:
    """Compute (d_min / d_max)^2, with safeguards for degenerate cases."""
    n = len(points)
    if n < 2:
        return 0.0
    dists = pdist(points)
    if len(dists) == 0:
        return 0.0
    d_min = np.min(dists)
    d_max = np.max(dists)
    if not (np.isfinite(d_min) and np.isfinite(d_max)):
        return 0.0
    if d_max <= 1e-15:
        return 0.0
    if d_min <= 1e-15:
        return 0.0
    ratio = d_min / d_max
    return float(ratio * ratio)


def _soft_score(points: np.ndarray, alpha: float = 100.0) -> float:
    """
    Smooth surrogate for (d_min / d_max)^2 using log-sum-exp.
    """
    n = len(points)
    if n < 2:
        return 0.0
    dists = pdist(points)
    if len(dists) == 0:
        return 0.0
    if not np.all(np.isfinite(dists)):
        return 0.0
    if np.min(dists) <= 1e-15 or np.max(dists) <= 1e-15:
        return 0.0

    d = dists
    a = alpha

    neg_ad = -a * d
    max_neg = np.max(neg_ad)
    soft_min = -(max_neg + np.log(np.sum(np.exp(neg_ad - max_neg)))) / a

    ad = a * d
    max_ad = np.max(ad)
    soft_max = (max_ad + np.log(np.sum(np.exp(ad - max_ad)))) / a

    if soft_max <= 1e-15:
        return 0.0
    ratio = soft_min / soft_max
    return float(ratio * ratio)


def _soft_score_grad(points: np.ndarray, alpha: float = 100.0) -> tuple:
    """
    Smooth surrogate with gradient for L-BFGS-B.
    Returns (score, gradient w.r.t. Cartesian coords).
    """
    n = len(points)
    if n < 2:
        return 0.0, np.zeros_like(points)
    
    dists = pdist(points)
    if len(dists) == 0:
        return 0.0, np.zeros_like(points)
    if not np.all(np.isfinite(dists)):
        return 0.0, np.zeros_like(points)
    if np.min(dists) <= 1e-15 or np.max(dists) <= 1e-15:
        return 0.0, np.zeros_like(points)

    d = dists
    a = alpha
    
    neg_ad = -a * d
    max_neg = np.max(neg_ad)
    exp_neg = np.exp(neg_ad - max_neg)
    sum_exp_neg = np.sum(exp_neg)
    soft_min = -(max_neg + np.log(sum_exp_neg)) / a
    
    ad = a * d
    max_ad = np.max(ad)
    exp_ad = np.exp(ad - max_ad)
    sum_exp_ad = np.sum(exp_ad)
    soft_max = (max_ad + np.log(sum_exp_ad)) / a

    if soft_max <= 1e-15:
        return 0.0, np.zeros_like(points)
    
    ratio = soft_min / soft_max
    score = ratio * ratio
    
    w_min = exp_neg / sum_exp_neg
    w_max = exp_ad / sum_exp_ad
    
    d_score_dd = 2.0 * ratio / soft_max * (-w_min - ratio * w_max)
    
    grad = np.zeros_like(points)
    idx = np.triu_indices(n, k=1)
    for k in range(len(dists)):
        i, j = idx[0][k], idx[1][k]
        diff = points[i] - points[j]
        d_k = d[k]
        if d_k < 1e-15:
            continue
        coeff = d_score_dd[k]
        grad[i] += coeff * diff / d_k
        grad[j] -= coeff * diff / d_k
    
    return score, grad


def _sphere_to_cartesian(sphere_params: np.ndarray) -> np.ndarray:
    """Convert (theta, phi) pairs to unit sphere Cartesian coordinates."""
    theta = sphere_params[0::2]
    phi = sphere_params[1::2]
    sin_phi = np.sin(phi)
    x = sin_phi * np.cos(theta)
    y = sin_phi * np.sin(theta)
    z = np.cos(phi)
    return np.column_stack([x, y, z])


def _cartesian_to_sphere(points: np.ndarray) -> np.ndarray:
    """Convert unit sphere Cartesian coordinates to (theta, phi) pairs."""
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    r = np.sqrt(x**2 + y**2 + z**2)
    r_safe = np.where(r > 1e-15, r, 1.0)
    theta = np.arctan2(y, x)
    phi = np.arccos(np.clip(z / r_safe, -1.0, 1.0))
    params = np.zeros((2 * len(points),))
    params[0::2] = theta
    params[1::2] = phi
    return params


def _initial_icosahedron() -> np.ndarray:
    """Return 12 vertices of a regular icosahedron on the unit sphere."""
    phi = (1 + np.sqrt(5)) / 2
    verts = [
        [-1, phi, 0],
        [1, phi, 0],
        [-1, -phi, 0],
        [1, -phi, 0],
        [0, -1, phi],
        [0, 1, phi],
        [0, -1, -phi],
        [0, 1, -phi],
        [phi, 0, -1],
        [phi, 0, 1],
        [-phi, 0, -1],
        [-phi, 0, 1],
    ]
    pts = np.array(verts, dtype=np.float64)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms
    return pts


def _tetrahedron() -> np.ndarray:
    """Return 4 vertices of a regular tetrahedron on the unit sphere."""
    pts = np.array([
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
    ], dtype=np.float64)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms
    return pts


def _cube() -> np.ndarray:
    """Return 8 vertices of a cube on the unit sphere."""
    pts = np.array([
        [-1, -1, -1],
        [-1, -1, 1],
        [-1, 1, -1],
        [-1, 1, 1],
        [1, -1, -1],
        [1, -1, 1],
        [1, 1, -1],
        [1, 1, 1],
    ], dtype=np.float64)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms
    return pts


def _octahedron() -> np.ndarray:
    """Return 6 vertices of a regular octahedron on the unit sphere."""
    pts = np.array([
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, -1],
    ], dtype=np.float64)
    return pts


def _random_sphere_points(n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate n random points uniformly on the unit sphere."""
    pts = np.zeros((n, 3))
    for i in range(n):
        theta = rng.uniform(0, 2 * np.pi)
        z = rng.uniform(-1, 1)
        r = np.sqrt(1 - z**2)
        pts[i] = [r * np.cos(theta), r * np.sin(theta), z]
    return pts


def _generate_initial_guesses() -> list:
    """Generate diverse initial guesses as Cartesian points on the unit sphere."""
    guesses = []
    rng = np.random.default_rng(42)

    # 1. Icosahedron + 2 points placed to maximize minimum distance
    ico = _initial_icosahedron()
    pts = ico.copy()
    for _ in range(2):
        best_candidate = None
        best_min_dist = -1
        for _ in range(12000):
            candidate = _random_sphere_points(1, rng)[0]
            dists = np.linalg.norm(pts - candidate, axis=1)
            min_dist = np.min(dists)
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_candidate = candidate.copy()
        if best_candidate is not None:
            pts = np.vstack([pts, best_candidate])
    guesses.append(pts)

    # 2. Cube + tetrahedron + 2 more
    cube = _cube()
    tetra = _tetrahedron()
    combined = np.vstack([cube, tetra])
    for _ in range(2):
        best_candidate = None
        best_min_dist = -1
        for _ in range(8000):
            candidate = _random_sphere_points(1, rng)[0]
            dists = np.linalg.norm(combined - candidate, axis=1)
            min_dist = np.min(dists)
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_candidate = candidate.copy()
        if best_candidate is not None:
            combined = np.vstack([combined, best_candidate])
    guesses.append(combined)

    # 3. Octahedron + 8 cube vertices
    octa = _octahedron()
    cube = _cube()
    guesses.append(np.vstack([octa, cube]))

    # 4. 2 poles + 12 equatorial
    pts = np.zeros((14, 3))
    pts[0] = [0, 0, 1]
    pts[1] = [0, 0, -1]
    for i in range(12):
        angle = 2 * np.pi * i / 12
        pts[i + 2] = [np.cos(angle), np.sin(angle), 0]
    guesses.append(pts)

    # 5. 2 poles + 6 at z=0.5 + 6 at z=-0.5
    pts = np.zeros((14, 3))
    pts[0] = [0, 0, 1]
    pts[1] = [0, 0, -1]
    r = np.sqrt(1 - 0.25)
    for i in range(6):
        angle = 2 * np.pi * i / 6
        pts[i + 2] = [r * np.cos(angle), r * np.sin(angle), 0.5]
    for i in range(6):
        angle = 2 * np.pi * i / 6 + np.pi / 6
        pts[i + 8] = [r * np.cos(angle), r * np.sin(angle), -0.5]
    guesses.append(pts)

    # 6. Random
    guesses.append(_random_sphere_points(14, rng))

    # 7. Icosahedron + 2 random
    ico2 = _initial_icosahedron()
    extra = _random_sphere_points(2, rng)
    guesses.append(np.vstack([ico2, extra]))

    # 8. Fibonacci sphere (golden angle) - well-distributed 14 points
    n = 14
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    fib_pts = np.zeros((n, 3))
    for i in range(n):
        z = 1.0 - 2.0 * i / (n - 1)
        r = np.sqrt(1.0 - z**2)
        theta = golden_angle * i
        fib_pts[i] = [r * np.cos(theta), r * np.sin(theta), z]
    guesses.append(fib_pts)

    # 9. Icosahedron with 2 points at optimal positions (symmetric placement)
    ico3 = _initial_icosahedron()
    extra1 = np.array([0.0, 0.0, 1.0])
    extra2 = np.array([0.0, 0.0, -1.0])
    for extra in [extra1, extra2]:
        min_d = np.min(np.linalg.norm(ico3 - extra, axis=1))
        if min_d < 0.3:
            for angle in np.linspace(0, 2*np.pi, 20):
                rot = np.array([[np.cos(angle), -np.sin(angle), 0],
                               [np.sin(angle), np.cos(angle), 0],
                               [0, 0, 1]])
                rotated = rot @ extra
                min_d_rot = np.min(np.linalg.norm(ico3 - rotated, axis=1))
                if min_d_rot > min_d:
                    extra = rotated
                    min_d = min_d_rot
    guesses.append(np.vstack([ico3, extra1, extra2]))

    return guesses


def _nelder_mead_sphere(sphere_params: np.ndarray, deadline: float,
                        max_iter: int = 5000, use_soft: bool = True,
                        alpha: float = 100.0) -> np.ndarray:
    """Optimize in spherical coordinate space using Nelder-Mead."""
    start = time.time()

    if use_soft:
        def objective(x):
            if time.time() > deadline:
                return 0.0
            pts = _sphere_to_cartesian(x)
            return -_soft_score(pts, alpha)
    else:
        def objective(x):
            if time.time() > deadline:
                return 0.0
            pts = _sphere_to_cartesian(x)
            return -_score(pts)

    try:
        result = minimize(
            objective,
            sphere_params,
            method='Nelder-Mead',
            options={
                'maxiter': max_iter,
                'xatol': 1e-9,
                'fatol': 1e-11,
                'adaptive': True
            }
        )
        if result.success or result.fun < 0:
            return result.x
        else:
            return sphere_params
    except Exception:
        return sphere_params


def _lbfgs_polish_sphere(sphere_params: np.ndarray, deadline: float,
                          alpha: float = 200.0) -> np.ndarray:
    """Polish using L-BFGS-B on Cartesian with sphere constraint via penalty."""
    start = time.time()
    pts = _sphere_to_cartesian(sphere_params)
    
    def objective(x):
        if time.time() > deadline:
            return 0.0
        p = x.reshape(-1, 3)
        norms = np.linalg.norm(p, axis=1)
        penalty = 100.0 * np.sum((norms - 1.0)**2)
        score = _soft_score(p, alpha)
        return -score + penalty
    
    x0 = pts.flatten()
    try:
        result = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            options={
                'maxiter': 2000,
                'ftol': 1e-14,
                'gtol': 1e-12
            }
        )
        if result.success or result.fun < 0:
            best_pts = result.x.reshape(-1, 3)
            norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
            norms_safe = np.where(norms > 1e-15, norms, 1.0)
            best_pts = best_pts / norms_safe
            return _cartesian_to_sphere(best_pts)
        else:
            return sphere_params
    except Exception:
        return sphere_params


def _refine_sphere(sphere_params: np.ndarray, deadline: float) -> np.ndarray:
    """Coordinate descent refinement in spherical coordinates."""
    best_params = sphere_params.copy()
    best_score = _score(_sphere_to_cartesian(best_params))
    start = time.time()

    n_params = len(best_params)
    step_sizes = np.full(n_params, 0.05)

    no_improve_count = 0
    threshold = 1e-12

    for iteration in range(5000):
        if time.time() > deadline:
            break

        improved = False
        for i in range(n_params):
            if time.time() > deadline:
                return best_params

            step = step_sizes[i]
            params_plus = best_params.copy()
            params_plus[i] += step
            score_plus = _score(_sphere_to_cartesian(params_plus))

            params_minus = best_params.copy()
            params_minus[i] -= step
            score_minus = _score(_sphere_to_cartesian(params_minus))

            if score_plus >= score_minus:
                best_new_score = score_plus
                best_new_params = params_plus
            else:
                best_new_score = score_minus
                best_new_params = params_minus

            if best_new_score > best_score + threshold:
                best_params = best_new_params
                best_score = best_new_score
                improved = True
                step_sizes[i] *= 1.05
            else:
                step_sizes[i] *= 0.7

        if not improved:
            no_improve_count += 1
            if no_improve_count > 10:
                break
        else:
            no_improve_count = 0

    return best_params


def _simulated_annealing(points: np.ndarray, deadline: float, rng: np.random.Generator) -> np.ndarray:
    """Simulated annealing on the sphere to escape local optima."""
    current = points.copy()
    current_score = _score(current)
    best = current.copy()
    best_score = current_score
    
    T = 0.1
    cooling = 0.999
    n = len(current)
    
    while time.time() < deadline and T > 1e-8:
        # Pick a random point to perturb
        idx = rng.integers(0, n)
        new_pt = current[idx].copy()
        # Random direction
        direction = rng.normal(0, 1, 3)
        direction = direction / (np.linalg.norm(direction) + 1e-15)
        new_pt = new_pt + T * 0.1 * direction
        # Normalize back to sphere
        norm = np.linalg.norm(new_pt)
        if norm < 1e-15:
            new_pt = direction
            norm = 1.0
        new_pt = new_pt / norm
        
        new_points = current.copy()
        new_points[idx] = new_pt
        new_score = _score(new_points)
        
        delta = new_score - current_score
        if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-15)):
            current = new_points
            current_score = new_score
            if current_score > best_score:
                best = current.copy()
                best_score = current_score
        
        T *= cooling
    
    return best


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates.
    """
    start_time = time.time()
    deadline = start_time + _DEADLINE_SECONDS - _SAFETY_MARGIN

    # Generate initial guesses
    guesses = _generate_initial_guesses()

    best_points = guesses[0].copy()
    best_score = _score(best_points)

    # Phase 1: Nelder-Mead from each initial guess
    for i, guess in enumerate(guesses):
        if time.time() > deadline:
            break

        params = _cartesian_to_sphere(guess)
        opt_params = _nelder_mead_sphere(params, deadline, max_iter=3000, use_soft=True, alpha=100.0)
        opt_pts = _sphere_to_cartesian(opt_params)
        opt_score = _score(opt_pts)

        if opt_score > best_score:
            best_score = opt_score
            best_points = opt_pts

    # Phase 2: L-BFGS-B polish
    if time.time() < deadline:
        params = _cartesian_to_sphere(best_points)
        lbfgs_result = _lbfgs_polish_sphere(params, deadline, alpha=200.0)
        lbfgs_pts = _sphere_to_cartesian(lbfgs_result)
        lbfgs_score = _score(lbfgs_pts)
        if lbfgs_score > best_score:
            best_score = lbfgs_score
            best_points = lbfgs_pts

    # Phase 3: Coordinate descent refinement
    if time.time() < deadline:
        params = _cartesian_to_sphere(best_points)
        refined = _refine_sphere(params, deadline)
        refined_pts = _sphere_to_cartesian(refined)
        refined_score = _score(refined_pts)
        if refined_score > best_score:
            best_score = refined_score
            best_points = refined_pts

    # Phase 4: Simulated annealing to escape local optima
    rng = np.random.default_rng(12345)
    if time.time() < deadline:
        sa_result = _simulated_annealing(best_points, deadline, rng)
        sa_score = _score(sa_result)
        if sa_score > best_score:
            best_score = sa_score
            best_points = sa_result

    # Phase 5: Final Nelder-Mead polish after SA
    if time.time() < deadline:
        params = _cartesian_to_sphere(best_points)
        final_params = _nelder_mead_sphere(params, deadline, max_iter=5000, use_soft=False)
        final_pts = _sphere_to_cartesian(final_params)
        final_score = _score(final_pts)
        if final_score > best_score:
            best_score = final_score
            best_points = final_pts

    # Phase 6: Final L-BFGS-B polish
    if time.time() < deadline:
        params = _cartesian_to_sphere(best_points)
        lbfgs_final = _lbfgs_polish_sphere(params, deadline, alpha=500.0)
        lbfgs_final_pts = _sphere_to_cartesian(lbfgs_final)
        lbfgs_final_score = _score(lbfgs_final_pts)
        if lbfgs_final_score > best_score:
            best_score = lbfgs_final_score
            best_points = lbfgs_final_pts

    # Phase 7: Final coordinate descent
    if time.time() < deadline:
        params = _cartesian_to_sphere(best_points)
        final2 = _refine_sphere(params, deadline)
        final2_pts = _sphere_to_cartesian(final2)
        final2_score = _score(final2_pts)
        if final2_score > best_score:
            best_score = final2_score
            best_points = final2_pts

    # Validate
    best_points = np.asarray(best_points, dtype=np.float64)

    if best_points.shape != (14, 3):
        ico = _initial_icosahedron()
        extra = _random_sphere_points(2, np.random.default_rng(0))
        best_points = np.vstack([ico, extra])

    if not np.all(np.isfinite(best_points)):
        ico = _initial_icosahedron()
        extra = _random_sphere_points(2, np.random.default_rng(0))
        best_points = np.vstack([ico, extra])

    # Normalize to unit sphere
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    norms_safe = np.where(norms > 1e-15, norms, 1.0)
    best_points = best_points / norms_safe

    return best_points


# EVOLVE-BLOCK-END
