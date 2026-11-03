import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 130

def compute_covering_radius_exact(points):
    """
    Computes the exact covering radius using ConvexHull Voronoi vertices.
    """
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi

    faces = hull.simplices
    voronoi_vertices = []

    for face in faces:
        a = points[face[0]]
        b = points[face[1]]
        c = points[face[2]]
        n = np.cross(b - a, c - a)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-10:
            continue
        d = np.dot(n, a)
        t = d / (n_norm ** 2)
        o = t * n
        o_norm = np.linalg.norm(o)
        if o_norm < 1e-10:
            continue
        o = o / o_norm
        voronoi_vertices.append(o)

    if not voronoi_vertices:
        return np.pi

    voronoi_vertices = np.array(voronoi_vertices)
    max_dist = 0.0
    for p in points:
        dots = np.dot(voronoi_vertices, p)
        dots = np.clip(dots, -1.0, 1.0)
        dists = np.arccos(dots)
        min_dist = np.min(dists)
        if min_dist > max_dist:
            max_dist = min_dist
    return max_dist


def repulsive_energy_and_grad(x, sigma=0.15, power=4):
    """
    Compute repulsive energy and gradient for points on unit sphere.
    Energy = sum_{i<j} (1 - dot(p_i, p_j))^power
    This penalizes close pairs (small angular distance).
    """
    pts = x.reshape(N, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    pts = pts / norms

    # Compute all pairwise dot products: (N, N)
    dots = np.dot(pts, pts.T)
    dots = np.clip(dots, -1.0, 1.0)

    # Energy: sum over i < j of (1 - dots[i,j])^power
    # Use upper triangle
    iu = np.triu_indices(N, k=1)
    vals = 1.0 - dots[iu]
    vals = np.maximum(vals, 1e-15)
    energy = np.sum(vals ** power)

    # Gradient: dE/dp_i = sum_{j != i} power * (1 - dots[i,j])^(power-1) * (-p_j + dots[i,j]*p_i)
    # Actually: d/dp_i (1 - p_i·p_j)^power = power*(1-p_i·p_j)^(power-1) * (-p_j)
    # But we need to account for normalization. Since pts are on sphere, we project gradient.
    # For simplicity, compute gradient in ambient space and let L-BFGS-B handle it.
    # dE/dp_i = sum_{j != i, j != i} power * (1 - dots[i,j])^(power-1) * (-p_j)
    #         + sum_{j != i} power * (1 - dots[i,j])^(power-1) * dots[i,j] * p_i  (from normalization)
    # Actually for constrained optimization on sphere, we can just use the ambient gradient
    # and the optimizer will handle it. Let's compute the ambient gradient.

    # grad[i] = sum_{j != i} power * (1 - dots[i,j])^(power-1) * (-p_j)
    # But also the derivative of the normalization constraint.
    # Simpler: just use the pairwise gradient without normalization correction,
    # since L-BFGS-B works in R^3 and we re-normalize.

    # Compute gradient matrix
    # For each i: grad_i = sum_{j != i} power * (1 - dots[i,j])^(power-1) * (-p_j)
    # Vectorized:
    # diff[i,j] = (1 - dots[i,j])^(power-1) for i != j
    diff = 1.0 - dots
    diff = np.maximum(diff, 1e-15)
    diff = diff ** (power - 1)
    # Zero out diagonal
    np.fill_diagonal(diff, 0.0)

    # grad[i] = -power * sum_j diff[i,j] * p_j
    grad = -power * np.dot(diff, pts)  # (N, 3)

    # Flatten gradient
    grad_flat = grad.flatten()

    return energy, grad_flat


def repulsive_energy_only(x, sigma=0.15, power=4):
    e, _ = repulsive_energy_and_grad(x, sigma, power)
    return e


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill

    def elapsed():
        return time.time() - start_time

    # Generate multiple diverse initial configurations
    def fibonacci_spiral(n, offset=0.5):
        i = np.arange(n) + offset
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(0, 1.0 - z * z))
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    def random_points(n, seed=42):
        rng = np.random.RandomState(seed)
        pts = rng.randn(n, 3)
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        return pts

    def jittered_fibonacci(n, jitter=0.1, seed=42):
        base = fibonacci_spiral(n)
        rng = np.random.RandomState(seed)
        noise = rng.randn(n, 3) * jitter
        pts = base + noise
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        return pts

    # Create diverse initial configurations
    initializations = [
        fibonacci_spiral(N),
        random_points(N, seed=1),
        jittered_fibonacci(N, jitter=0.15, seed=2),
        random_points(N, seed=3),
        jittered_fibonacci(N, jitter=0.3, seed=4),
        random_points(N, seed=5),
    ]

    best_points = None
    best_energy = np.inf
    best_exact_radius = np.pi

    # Phase 1: Optimize from each initialization
    for init_idx, init_pts in enumerate(initializations):
        if elapsed() > deadline - 15.0:
            break

        init_flat = init_pts.flatten()
        try:
            res = minimize(
                repulsive_energy_only,
                init_flat,
                jac=lambda x: repulsive_energy_and_grad(x)[1],
                method='L-BFGS-B',
                options={'maxiter': 300, 'maxfun': 3000, 'ftol': 1e-14, 'gtol': 1e-10}
            )

            new_points = res.x.reshape(N, 3)
            new_points = new_points / np.linalg.norm(new_points, axis=1, keepdims=True)

            # Check for validity
            norms = np.linalg.norm(new_points, axis=1)
            if np.any(norms < 1e-8) or not np.all(np.isfinite(new_points)):
                continue

            # Compute exact covering radius
            exact_r = compute_covering_radius_exact(new_points)

            if exact_r < best_exact_radius:
                best_exact_radius = exact_r
                best_points = new_points.copy()
                best_energy = res.fun

        except Exception:
            pass

    # Phase 2: Further refinement from the best configuration found
    if best_points is not None and elapsed() < deadline - 20.0:
        # Try multiple refinement passes with different power parameters
        for power in [6, 8, 10]:
            if elapsed() > deadline - 15.0:
                break

            try:
                res2 = minimize(
                    lambda x: repulsive_energy_only(x, power=power),
                    best_points.flatten(),
                    jac=lambda x: repulsive_energy_and_grad(x, power=power)[1],
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'maxfun': 2000, 'ftol': 1e-14, 'gtol': 1e-10}
                )

                new_points = res2.x.reshape(N, 3)
                new_points = new_points / np.linalg.norm(new_points, axis=1, keepdims=True)

                norms = np.linalg.norm(new_points, axis=1)
                if np.any(norms < 1e-8) or not np.all(np.isfinite(new_points)):
                    continue

                exact_r = compute_covering_radius_exact(new_points)
                if exact_r < best_exact_radius:
                    best_exact_radius = exact_r
                    best_points = new_points.copy()

            except Exception:
                pass

    # Phase 3: Local search around best configuration with small perturbations
    if best_points is not None and elapsed() < deadline - 10.0:
        rng = np.random.RandomState(123)
        for trial in range(10):
            if elapsed() > deadline - 8.0:
                break

            noise = rng.randn(N, 3) * 0.03
            perturbed = best_points + noise
            perturbed = perturbed / np.linalg.norm(perturbed, axis=1, keepdims=True)

            try:
                res3 = minimize(
                    repulsive_energy_only,
                    perturbed.flatten(),
                    jac=lambda x: repulsive_energy_and_grad(x)[1],
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'maxfun': 1000, 'ftol': 1e-14, 'gtol': 1e-10}
                )

                new_points = res3.x.reshape(N, 3)
                new_points = new_points / np.linalg.norm(new_points, axis=1, keepdims=True)

                norms = np.linalg.norm(new_points, axis=1)
                if np.any(norms < 1e-8) or not np.all(np.isfinite(new_points)):
                    continue

                exact_r = compute_covering_radius_exact(new_points)
                if exact_r < best_exact_radius:
                    best_exact_radius = exact_r
                    best_points = new_points.copy()

            except Exception:
                pass

    # Fallback: if we never found a valid configuration, use Fibonacci
    if best_points is None:
        best_points = fibonacci_spiral(N)

    # Final normalization and validation
    best_points = best_points / np.linalg.norm(best_points, axis=1, keepdims=True)

    return best_points
