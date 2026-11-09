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
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    iu = np.triu_indices(N, k=1)
    # log(1/d) = -0.5 * log(d^2)
    return -0.5 * math.fsum(np.log(dist_sq[iu]).tolist())


def _energy_and_gradient(pts):
    """Compute energy and gradient of log energy w.r.t. points."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    iu = np.triu_indices(N, k=1)
    energy = -0.5 * math.fsum(np.log(dist_sq[iu]).tolist())
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    # grad of -log(|xi-xj|) = (xi-xj)/|xi-xj|^2 = diff / dist_sq
    # Energy = -sum log(dist) = -0.5 sum log(dist_sq)
    # dE/dxi = -sum_{j!=i} (xi-xj)/dist_sq_ij * (-1) ... let me re-derive
    # E = sum_{i<j} log(1/|xi-xj|) = -sum_{i<j} log|xi-xj|
    # dE/dxi = -sum_{j!=i} (1/|xi-xj|) * (xi-xj)/|xi-xj| = -sum_{j!=i} (xi-xj)/|xi-xj|^2
    # = -sum_{j!=i} diff[i,j] / dist_sq[i,j]
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
        result = minimize(
            objective_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-13, 'maxls': 40}
        )
        if time.time() > deadline:
            return pts, _log_energy(pts)
        opt_pts = _project_to_sphere(result.x.reshape(N, 3))
        opt_energy = _log_energy(opt_pts)
        return opt_pts, opt_energy
    except Exception:
        return pts, _log_energy(pts)


def _newton_polish(pts, max_iter=200, deadline=None):
    """Newton-style refinement with adaptive step size and backtracking."""
    pts = pts.copy()
    pts = _project_to_sphere(pts)
    current_energy = _log_energy(pts)
    best_pts = pts.copy()
    best_energy = current_energy

    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break

        e, grad = _energy_and_gradient(pts)
        grad_tangent = grad - np.sum(grad * pts, axis=1, keepdims=True) * pts

        grad_norm = np.linalg.norm(grad_tangent)
        if grad_norm < 1e-14:
            break

        # Adaptive step with line search
        step = 1.0 / (1.0 + grad_norm * 5)
        alpha = 1.0
        new_pts = pts
        new_energy = current_energy
        for _ in range(50):
            new_pts = pts - alpha * step * grad_tangent
            new_pts = _project_to_sphere(new_pts)
            new_energy = _log_energy(new_pts)
            if new_energy < current_energy:
                break
            alpha *= 0.5
            if alpha < 1e-12:
                break

        if new_energy < current_energy - 1e-16:
            pts = new_pts
            current_energy = new_energy
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

    pts = pts + 0.01 * rng.standard_normal((n, 3))
    return _project_to_sphere(pts)


def _hexagonal_grid_like(n):
    """Generate points inspired by hexagonal packing on sphere."""
    pts = _fibonacci_points(n)
    theta = np.pi / 7
    rot = np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ])
    pts = pts @ rot.T
    return pts


def _lattice_initialization(rng, n_lattice=5):
    """
    Generate points from a 3D lattice structure, then project to sphere.
    Uses FCC-like lattice for better packing.
    """
    # FCC lattice vectors
    a = 1.0
    fcc_basis = np.array([
        [0, 0, 0],
        [0, 0.5, 0.5],
        [0.5, 0, 0.5],
        [0.5, 0.5, 0]
    ]) * a

    # Generate lattice points in a cube
    lattice_pts = []
    for i in range(-n_lattice, n_lattice + 1):
        for j in range(-n_lattice, n_lattice + 1):
            for k in range(-n_lattice, n_lattice + 1):
                for basis in fcc_basis:
                    pt = np.array([i, j, k]) + basis
                    if np.linalg.norm(pt) <= 1.5:
                        lattice_pts.append(pt)

    lattice_pts = np.array(lattice_pts)
    
    # If not enough points, fall back to random
    if len(lattice_pts) < N:
        return _random_points(N, rng)

    # Select N points from lattice, preferring those near unit sphere
    norms = np.linalg.norm(lattice_pts, axis=1)
    # Weight by closeness to unit sphere
    weights = np.exp(-5.0 * (norms - 1.0) ** 2)
    weights /= weights.sum()

    # Sample N points
    indices = rng.choice(len(lattice_pts), size=N, replace=False, p=weights)
    pts = lattice_pts[indices]

    # Add small perturbation and project
    pts = pts + 0.05 * rng.standard_normal((N, 3))
    return _project_to_sphere(pts)


def _symmetric_lattice_init(rng):
    """
    Generate points using a parameterized FCC-like lattice optimized for symmetry.
    We define the lattice by a scaling factor and a small anisotropy vector.
    """
    # Base FCC lattice points in a larger box
    max_coord = 8
    fcc_basis = np.array([
        [0, 0, 0],
        [0, 0.5, 0.5],
        [0.5, 0, 0.5],
        [0.5, 0.5, 0]
    ])
    
    lattice_pts = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                for basis in fcc_basis:
                    pt = np.array([i, j, k]) + basis
                    lattice_pts.append(pt)
                    
    lattice_pts = np.array(lattice_pts)
    
    # To make it "symmetric", enforce antipodal symmetry
    half_n = N // 2
    # Pick half_n points from the lattice
    indices = rng.choice(len(lattice_pts), size=half_n, replace=False)
    pts_half = lattice_pts[indices]
    
    # Add small random perturbation to avoid singularities
    pts_half = pts_half + 0.1 * rng.standard_normal((half_n, 3))
    
    # Create antipodal pairs
    pts = np.vstack([pts_half, -pts_half])
    
    # Project to sphere
    pts = _project_to_sphere(pts)
    
    return pts


def _simulated_annealing(pts, deadline, rng, max_iter=500):
    """
    Simulated annealing to escape local minima.
    """
    pts = _project_to_sphere(pts.copy())
    current_energy = _log_energy(pts)
    best_pts = pts.copy()
    best_energy = current_energy

    T = 0.1
    cooling = 0.995

    for it in range(max_iter):
        if time.time() > deadline:
            break

        # Random perturbation
        scale = T * 0.1
        new_pts = pts + scale * rng.standard_normal((N, 3))
        new_pts = _project_to_sphere(new_pts)
        new_energy = _log_energy(new_pts)

        delta = new_energy - current_energy
        if delta < 0 or rng.random() < math.exp(-delta / T):
            pts = new_pts
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_pts = pts.copy()

        T *= cooling
        if T < 1e-6:
            break

    return best_pts, best_energy


def construct():
    deadline = time.time() + 145.0
    rng = np.random.default_rng(42)

    best_pts = None
    best_energy = float('inf')

    # Generate diverse initializations - focus on quality over quantity
    initializations = []

    # 1. Fibonacci base (most promising)
    fib_pts = _fibonacci_points(N)
    initializations.append(fib_pts)

    # 2. A few random points with different seeds
    for seed in [1, 7, 42, 99]:
        r = np.random.default_rng(seed)
        initializations.append(_random_points(N, r))

    # 3. Perturbed Fibonacci with a few scales
    for scale in [0.03, 0.1, 0.2]:
        initializations.append(_perturbed_fibonacci(N, scale, rng))

    # 4. Antipodal pairs
    initializations.append(_antipodal_pairs(N, rng))

    # 5. Hexagonal-like
    initializations.append(_hexagonal_grid_like(N))

    # 6. One lattice-based initialization
    initializations.append(_lattice_initialization(rng, n_lattice=5))

    # 7. Symmetric lattice initialization
    initializations.append(_symmetric_lattice_init(rng))

    # 8. A couple more random variations
    for _ in range(4):
        initializations.append(_random_points(N, rng))

    total_inits = len(initializations)

    for idx, init_pts in enumerate(initializations):
        if time.time() > deadline - 30.0:
            break

        # Phase 1: L-BFGS-B optimization with more iterations for early candidates
        maxiter_lbfgs = 500 if idx < 3 else 300
        lgbfgs_pts, lgbfgs_energy = _lgbfgs_optimize(init_pts, deadline - 10.0, maxiter=maxiter_lbfgs)

        if lgbfgs_energy < best_energy:
            best_energy = lgbfgs_energy
            best_pts = lgbfgs_pts.copy()

        # Phase 2: Newton polish
        if time.time() < deadline - 20.0:
            newton_pts, newton_energy = _newton_polish(lgbfgs_pts, max_iter=300, deadline=deadline - 5.0)

            if newton_energy < best_energy:
                best_energy = newton_energy
                best_pts = newton_pts.copy()
        else:
            newton_pts = lgbfgs_pts
            newton_energy = lgbfgs_energy

        # Phase 3: Simulated annealing for first few candidates to escape local minima
        if time.time() < deadline - 35.0 and idx < 5:
            sa_pts, sa_energy = _simulated_annealing(newton_pts, deadline - 25.0, rng, max_iter=300)
            if sa_energy < best_energy:
                best_energy = sa_energy
                best_pts = sa_pts.copy()
            # Polish after SA
            if time.time() < deadline - 15.0:
                polished, pol_e = _newton_polish(sa_pts, max_iter=150, deadline=deadline - 5.0)
                if pol_e < best_energy:
                    best_energy = pol_e
                    best_pts = polished.copy()

        # Phase 4: Second L-BFGS pass from perturbed result for top candidates
        if time.time() < deadline - 30.0 and idx < total_inits // 3:
            perturbed = newton_pts + 0.05 * rng.standard_normal((N, 3))
            perturbed = _project_to_sphere(perturbed)
            lgbfgs2_pts, lgbfgs2_energy = _lgbfgs_optimize(perturbed, deadline - 15.0, maxiter=300)
            if lgbfgs2_energy < best_energy:
                best_energy = lgbfgs2_energy
                best_pts = lgbfgs2_pts.copy()

    if best_pts is None:
        best_pts = fib_pts

    # Final high-precision polish - invest remaining time here
    if time.time() < deadline - 5.0:
        polished_pts, polished_energy = _newton_polish(best_pts, max_iter=500, deadline=deadline - 2.0)
        if polished_energy < best_energy:
            best_energy = polished_energy
            best_pts = polished_pts

    # One more L-BFGS pass with tight tolerances
    if time.time() < deadline - 3.0:
        final_pts, final_energy = _lgbfgs_optimize(best_pts, deadline - 1.0, maxiter=1000)
        if final_energy < best_energy:
            best_energy = final_energy
            best_pts = final_pts

    # Final Newton polish after L-BFGS
    if time.time() < deadline - 1.0:
        polished_pts, polished_energy = _newton_polish(best_pts, max_iter=200, deadline=deadline - 0.5)
        if polished_energy < best_energy:
            best_energy = polished_energy
            best_pts = polished_pts

    best_pts = _project_to_sphere(best_pts)

    # Validate: check for very close points
    dists = np.linalg.norm(best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :], axis=2)
    np.fill_diagonal(dists, 1.0)
    if np.min(dists) < 1e-9:
        best_pts = _project_to_sphere(fib_pts)

    return best_pts
# EVOLVE-BLOCK-END
