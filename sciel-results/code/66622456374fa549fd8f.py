"""Search-based solver: many varied candidates, keep best within time budget."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize


def _log_energy(pts):
    """Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|)."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, 1.0)
    iu = np.triu_indices(N, k=1)
    return -math.fsum(np.log(dist[iu]).tolist())


def _energy_and_gradient(pts):
    """Compute energy and gradient of log energy w.r.t. points."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    iu = np.triu_indices(N, k=1)
    energy = -math.fsum(np.log(dist[iu]).tolist())
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    grad = -np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    return energy, grad


def _project_to_sphere(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms


def _fibonacci_points(n):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_points(n, rng):
    """Generate random points on unit sphere."""
    pts = rng.standard_normal((n, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms


def _sobol_like_points(n, rng):
    """Generate quasi-random points with controlled spacing."""
    pts = rng.standard_normal((n, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms


def _lgbfgs_optimize(pts, deadline, maxiter=500):
    """Optimize using L-BFGS-B with spherical constraint via projection."""
    pts = pts.copy()
    pts = _project_to_sphere(pts)

    def objective_and_grad(flat_pts):
        p = flat_pts.reshape(N, 3)
        p = _project_to_sphere(p)
        e, g = _energy_and_gradient(p)
        g_tangent = g - np.sum(g * p, axis=1, keepdims=True) * p
        return e, g_tangent.flatten()

    x0 = pts.flatten()
    try:
        # First pass: coarse optimization
        result = minimize(
            objective_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 40}
        )
        if time.time() > deadline:
            return pts, _log_energy(pts)
        
        opt_pts = _project_to_sphere(result.x.reshape(N, 3))
        
        # Second pass: fine-tune from improved starting point
        if time.time() < deadline - 5.0:
            x1 = opt_pts.flatten()
            result2 = minimize(
                objective_and_grad,
                x1,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': maxiter // 2, 'ftol': 1e-15, 'gtol': 1e-13, 'maxls': 30}
            )
            if time.time() < deadline:
                opt_pts = _project_to_sphere(result2.x.reshape(N, 3))
        
        opt_energy = _log_energy(opt_pts)
        return opt_pts, opt_energy
    except Exception:
        return pts, _log_energy(pts)


def _newton_polish(pts, max_iter=100, deadline=None):
    """High-precision polish using gradient descent with adaptive step size."""
    pts = pts.copy()
    pts = _project_to_sphere(pts)
    current_energy = _log_energy(pts)
    best_pts = pts.copy()
    best_energy = current_energy

    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break

        e, grad = _energy_and_gradient(pts)
        # Project gradient to tangent
        grad_tangent = grad - np.sum(grad * pts, axis=1, keepdims=True) * pts

        # Check convergence
        grad_norm = np.linalg.norm(grad_tangent)
        if grad_norm < 1e-13:
            break

        # Adaptive step size based on gradient magnitude
        step = 1.0 / (1.0 + grad_norm * 50)
        
        # Line search: backtracking with Armijo condition
        alpha = 1.0
        new_pts = None
        new_energy = None
        for _ in range(30):
            candidate = pts - alpha * step * grad_tangent
            candidate = _project_to_sphere(candidate)
            cand_energy = _log_energy(candidate)
            
            # Armijo condition: sufficient decrease
            if cand_energy < current_energy - 1e-4 * alpha * grad_norm * grad_norm:
                new_pts = candidate
                new_energy = cand_energy
                break
            alpha *= 0.5
            if alpha < 1e-8:
                break

        if new_pts is not None and new_energy < current_energy:
            pts = new_pts
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_pts = pts.copy()
        else:
            # Try smaller fixed step
            small_step = step * 0.01
            candidate = pts - small_step * grad_tangent
            candidate = _project_to_sphere(candidate)
            cand_energy = _log_energy(candidate)
            if cand_energy < current_energy - 1e-15:
                pts = candidate
                current_energy = cand_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_pts = pts.copy()
            else:
                break

    return best_pts, best_energy


def _perturbed_fibonacci(n, scale, rng):
    """Fibonacci points with random perturbation."""
    fib = _fibonacci_points(n)
    perturbed = fib + scale * rng.standard_normal((n, 3))
    return _project_to_sphere(perturbed)


def _antipodal_pairs(n, rng):
    """Start with pairs of near-antipodal points, then perturb."""
    base = rng.standard_normal((n // 2 + n % 2, 3))
    norms = np.linalg.norm(base, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    base = base / norms

    if n % 2 == 0:
        pts = np.vstack([base, -base])
    else:
        extra = rng.standard_normal((1, 3))
        extra = extra / np.linalg.norm(extra)
        pts = np.vstack([base, -base, extra])

    # Small perturbation to break symmetry
    pts = pts + 0.01 * rng.standard_normal((n, 3))
    return _project_to_sphere(pts)


def _hexagonal_grid_like(n):
    """Generate points inspired by hexagonal packing on sphere."""
    pts = _fibonacci_points(n)
    # Apply a rotation to break any alignment
    theta = np.pi / 7
    rot = np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ])
    pts = pts @ rot.T
    return pts


def construct():
    deadline = time.time() + 145.0
    rng = np.random.default_rng(42)

    best_pts = None
    best_energy = float('inf')

    # Generate diverse initializations
    initializations = []

    # 1. Fibonacci base
    fib_pts = _fibonacci_points(N)
    initializations.append(fib_pts)

    # 2. Random points with different seeds
    for seed in [1, 2, 3, 7, 13, 42, 99, 123]:
        r = np.random.default_rng(seed)
        initializations.append(_random_points(N, r))

    # 3. Perturbed Fibonacci with various scales
    for scale in [0.02, 0.05, 0.1, 0.15, 0.25]:
        initializations.append(_perturbed_fibonacci(N, scale, rng))

    # 4. Antipodal pairs
    initializations.append(_antipodal_pairs(N, rng))

    # 5. Hexagonal-like
    initializations.append(_hexagonal_grid_like(N))

    # 6. More random variations
    for _ in range(8):
        initializations.append(_random_points(N, rng))

    # 7. Clustered random (points in groups)
    n_clusters = 5
    for _ in range(3):
        centers = rng.standard_normal((n_clusters, 3))
        centers = centers / np.linalg.norm(centers, axis=1, keepdims=True)
        pts = []
        per_cluster = N // n_clusters
        for i in range(n_clusters):
            cluster = centers[i:i+1] + 0.3 * rng.standard_normal((per_cluster, 3))
            pts.append(cluster)
        pts = np.vstack(pts)
        if len(pts) < N:
            extra = rng.standard_normal((N - len(pts), 3))
            pts = np.vstack([pts, extra])
        pts = pts[:N]
        initializations.append(_project_to_sphere(pts))

    total_inits = len(initializations)

    for idx, init_pts in enumerate(initializations):
        if time.time() > deadline - 25.0:
            break

        # Phase 1: L-BFGS-B optimization
        lgbfgs_pts, lgbfgs_energy = _lgbfgs_optimize(init_pts, deadline - 10.0, maxiter=300)

        if lgbfgs_energy < best_energy:
            best_energy = lgbfgs_energy
            best_pts = lgbfgs_pts.copy()

        # Phase 2: Newton polish for high precision
        if time.time() < deadline - 15.0:
            newton_pts, newton_energy = _newton_polish(lgbfgs_pts, max_iter=200, deadline=deadline - 5.0)

            if newton_energy < best_energy:
                best_energy = newton_energy
                best_pts = newton_pts.copy()

        # Phase 3: If time remains, do a second L-BFGS pass from perturbed result
        if time.time() < deadline - 20.0 and idx < total_inits // 2:
            perturbed = newton_pts + 0.05 * rng.standard_normal((N, 3))
            perturbed = _project_to_sphere(perturbed)
            lgbfgs2_pts, lgbfgs2_energy = _lgbfgs_optimize(perturbed, deadline - 10.0, maxiter=200)
            if lgbfgs2_energy < best_energy:
                best_energy = lgbfgs2_energy
                best_pts = lgbfgs2_pts.copy()

    if best_pts is None:
        best_pts = fib_pts

    # Final high-precision polish with multiple passes
    if time.time() < deadline - 10.0:
        # First polish pass
        polished_pts, polished_energy = _newton_polish(best_pts, max_iter=500, deadline=deadline - 5.0)
        if polished_energy < best_energy:
            best_energy = polished_energy
            best_pts = polished_pts

    # Second polish pass if time permits
    if time.time() < deadline - 5.0:
        polished_pts2, polished_energy2 = _newton_polish(best_pts, max_iter=300, deadline=deadline - 2.0)
        if polished_energy2 < best_energy:
            best_energy = polished_energy2
            best_pts = polished_pts2

    # One more L-BFGS pass with tighter tolerances
    if time.time() < deadline - 3.0:
        final_pts, final_energy = _lgbfgs_optimize(best_pts, deadline - 1.0, maxiter=800)
        if final_energy < best_energy:
            best_energy = final_energy
            best_pts = final_pts

    # Final ultra-precise polish
    if time.time() < deadline - 1.0:
        final_polish_pts, final_polish_energy = _newton_polish(best_pts, max_iter=200, deadline=deadline - 0.5)
        if final_polish_energy < best_energy:
            best_energy = final_polish_energy
            best_pts = final_polish_pts

    best_pts = _project_to_sphere(best_pts)

    # Validate: check for very close points
    dists = np.linalg.norm(best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :], axis=2)
    np.fill_diagonal(dists, 1.0)
    if np.min(dists) < 1e-9:
        best_pts = _project_to_sphere(fib_pts)

    return best_pts
# EVOLVE-BLOCK-END
