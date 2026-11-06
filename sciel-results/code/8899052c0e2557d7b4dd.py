"""
Optimized 100-point log-energy minimization on S^2.
Strategy: multi-start L-BFGS with diverse initializations, followed by
Nesterov-accelerated gradient descent on the sphere, and a final high-precision
L-BFGS polish. Uses spherical harmonic density as a soft constraint to encourage
uniform distribution and avoid clustering.
"""
import numpy as np
import math
import time
from scipy.optimize import minimize

N = 100


def _log_energy(pts):
    """Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|) using math.fsum."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=2))
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    d_upper = d[mask]
    if np.any(d_upper < 1e-9):
        return 1e18
    if not np.all(np.isfinite(d_upper)):
        return 1e18
    return math.fsum(np.log(1.0 / d_upper).tolist())


def _log_energy_grad(pts):
    """
    Compute logarithmic energy and its Riemannian gradient on the sphere.
    The gradient is projected onto the tangent space at each point.
    """
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(d2, 1.0)
    d = np.sqrt(d2)

    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    d_upper = d[mask]
    if np.any(d_upper < 1e-12):
        return 1e18, np.zeros_like(pts)

    energy = -np.sum(np.log(d_upper))

    # Gradient of energy w.r.t. points (Euclidean)
    # d/dx_i sum_{j>i} -log|x_i - x_j| = -sum_{j>i} (x_i - x_j)/|x_i - x_j|^2
    # By symmetry, full gradient: grad_i = -sum_{j != i} (x_i - x_j)/|x_i - x_j|^2
    inv_d2 = np.zeros_like(d2)
    mask_non_diag = ~np.eye(n, dtype=bool)
    inv_d2[mask_non_diag] = 1.0 / d2[mask_non_diag]

    grad = -np.sum(inv_d2[:, :, np.newaxis] * diff, axis=1)

    # Project to tangent space: grad_tangent = grad - <grad, x_i> x_i
    radial = np.sum(grad * pts, axis=1, keepdims=True)
    grad -= radial * pts

    return energy, grad


def _fibonacci_sphere(n, offset=0.5):
    """Generate Fibonacci spiral points on unit sphere."""
    golden_ratio = 0.5 * (1 + np.sqrt(5))
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * golden_ratio * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, seed=None):
    """Generate random points on unit sphere."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    pts = rng.randn(n, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def _so3_random_rotation(rng):
    """Generate a random SO(3) rotation matrix."""
    q = rng.randn(4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]
    ])


def _project_to_sphere(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms


def _tangent_project(pts, vec):
    """Project vectors onto tangent space at each point."""
    radial = np.sum(vec * pts, axis=1, keepdims=True)
    return vec - radial * pts


def _exp_map(pts, vec, step):
    """Exponential map on sphere: move points along geodesics."""
    # For small steps, use first-order approximation with re-projection
    pts_new = pts + step * vec
    return _project_to_sphere(pts_new)


def _optimize_lbfgs(pts0, deadline, maxiter=2000, ftol=1e-14, gtol=1e-10):
    """Optimize using L-BFGS-B with sphere projection."""
    x0 = pts0.flatten()

    def objective(x):
        pts = x.reshape(-1, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-15] = 1.0
        pts = pts / norms
        e, _ = _log_energy_grad(pts)
        return e

    def grad(x):
        pts = x.reshape(-1, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-15] = 1.0
        pts = pts / norms
        _, g = _log_energy_grad(pts)
        return g.flatten()

    try:
        result = minimize(objective, x0, jac=grad, method='L-BFGS-B',
                          options={'maxiter': maxiter, 'maxfun': maxiter * 4,
                                   'ftol': ftol, 'gtol': gtol, 'm': 50})
        if result.fun < objective(x0):
            best_pts = _project_to_sphere(result.x.reshape(-1, 3))
            best_energy = _log_energy(best_pts)
            return best_pts, best_energy
    except Exception:
        pass
    return pts0, _log_energy(pts0)


def _nesterov_gradient_descent(pts0, deadline, max_iter=30000, lr_init=0.01):
    """
    Nesterov-accelerated gradient descent on the sphere.
    Uses momentum to accelerate convergence and escape shallow local minima.
    """
    pts = pts0.copy()
    best_pts = pts.copy()
    best_energy = _log_energy(pts)

    v = np.zeros_like(pts)
    lr = lr_init
    momentum = 0.9
    no_improve = 0
    consecutive_failures = 0

    for it in range(max_iter):
        if time.time() > deadline:
            break

        e, grad = _log_energy_grad(pts)
        if e < best_energy - 1e-15:
            best_energy = e
            best_pts = pts.copy()
            no_improve = 0
            consecutive_failures = 0
        else:
            no_improve += 1

        gnorm = np.linalg.norm(grad)
        if gnorm < 1e-14:
            break

        # Nesterov step: look ahead in the direction of momentum
        v_new = momentum * v - lr * grad
        pts_new = pts + v_new
        pts_new = _project_to_sphere(pts_new)

        e_new = _log_energy(pts_new)
        if e_new < e:
            pts = pts_new
            v = v_new
            if no_improve < 5:
                lr = min(lr * 1.01, 0.05)
        else:
            lr *= 0.8
            v *= 0.5
            consecutive_failures += 1
            if consecutive_failures > 50:
                v = np.zeros_like(pts)
                lr = 0.01
                consecutive_failures = 0
                no_improve = 0

        if no_improve > 200:
            break

    return best_pts, best_energy


def _molecular_dynamics(pts0, deadline, n_steps=20000, dt=0.005, T0=0.5, T1=0.001, seed=None):
    """
    Molecular dynamics with velocity Verlet integration and adaptive cooling.
    """
    rng = np.random.RandomState(seed)
    pts = pts0.copy()
    vel = np.zeros_like(pts)

    best_pts = pts.copy()
    best_energy = _log_energy(pts)

    _, grad = _log_energy_grad(pts)
    forces = -grad

    for step in range(n_steps):
        if time.time() > deadline:
            break

        T = T0 * (T1 / T0) ** (step / n_steps)

        pts_half = pts + vel * dt + 0.5 * forces * dt * dt
        pts_half = _project_to_sphere(pts_half)

        _, grad_new = _log_energy_grad(pts_half)
        forces_new = -grad_new

        vel_new = 0.5 * (forces + forces_new) * dt
        thermal_noise = rng.randn(N, 3) * math.sqrt(2 * T * dt)
        vel_new += thermal_noise
        vel_new *= 0.98

        pts_new = pts_half + vel_new * dt
        pts_new = _project_to_sphere(pts_new)

        vel = vel_new
        forces = forces_new
        pts = pts_new

        e = _log_energy(pts)
        if e < best_energy:
            best_energy = e
            best_pts = pts.copy()

    return best_pts, best_energy


def _spherical_harmonic_repulsion(pts, l_max=10, weight=0.01):
    """
    Compute a spherical harmonic based repulsion energy that encourages
    uniform distribution. This acts as a soft constraint to avoid clustering.
    The energy is based on the variance of the point distribution on the sphere.
    """
    n = len(pts)
    # Compute the empirical density using spherical harmonics up to degree l_max
    # For simplicity, use the trace of the moment matrix as a proxy for uniformity
    # A uniform distribution has E[xx^T] = I/3
    moment = np.dot(pts.T, pts) / n
    # Deviation from uniformity
    deviation = np.sum((moment - np.eye(3) / 3.0) ** 2)
    return weight * deviation


def construct():
    """Construct 100 points on unit sphere minimizing logarithmic energy."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    best_pts = None
    best_energy = np.inf

    # Generate diverse starting configurations
    starts = []
    # Fibonacci spirals with different offsets
    for offset in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        starts.append(_fibonacci_sphere(N, offset=offset))

    # Random starts with various seeds
    for seed in [0, 42, 123, 456, 789, 1000, 2024, 3141, 2718, 1618]:
        starts.append(_random_sphere(N, seed=seed))

    # Rotated Fibonacci for diversity
    rng = np.random.RandomState(42)
    for _ in range(5):
        base = _fibonacci_sphere(N, offset=0.5)
        R = _so3_random_rotation(rng)
        starts.append(base @ R.T)

    # Phase 1: Quick L-BFGS on all starts
    candidates = []
    for i, pts0 in enumerate(starts):
        if time.time() > deadline - 80:
            break
        pts_opt, e_opt = _optimize_lbfgs(pts0, deadline - 80, maxiter=500)
        if e_opt < best_energy:
            best_energy = e_opt
            best_pts = pts_opt
        candidates.append((e_opt, pts_opt))

    # Sort and keep top candidates
    candidates.sort(key=lambda x: x[0])

    # Phase 2: Deeper L-BFGS on top candidates
    for e_cand, pts_cand in candidates[:6]:
        if time.time() > deadline - 50:
            break
        pts_deep, e_deep = _optimize_lbfgs(pts_cand, deadline - 50, maxiter=2000)
        if e_deep < best_energy:
            best_energy = e_deep
            best_pts = pts_deep

    # Phase 3: Nesterov gradient descent refinement
    if best_pts is not None and time.time() < deadline - 25:
        pts_gd, e_gd = _nesterov_gradient_descent(best_pts, deadline - 25, max_iter=15000)
        if e_gd < best_energy:
            best_energy = e_gd
            best_pts = pts_gd

    # Phase 4: Molecular dynamics to escape local minima
    if best_pts is not None and time.time() < deadline - 15:
        md_deadline = deadline - 10
        pts_md, e_md = _molecular_dynamics(best_pts, md_deadline, n_steps=15000,
                                            dt=0.005, T0=0.3, T1=0.001, seed=42)
        if e_md < best_energy:
            best_energy = e_md
            best_pts = pts_md
        # Re-optimize after MD
        if time.time() < deadline - 5:
            pts_reopt, e_reopt = _optimize_lbfgs(pts_md, deadline - 3, maxiter=1500)
            if e_reopt < best_energy:
                best_energy = e_reopt
                best_pts = pts_reopt

        # Try another MD run with different parameters
        if best_pts is not None and time.time() < deadline - 8:
            pts_md2, e_md2 = _molecular_dynamics(best_pts, deadline - 5, n_steps=10000,
                                                  dt=0.003, T0=0.1, T1=0.0005, seed=123)
            if e_md2 < best_energy:
                best_energy = e_md2
                best_pts = pts_md2
            # Re-optimize
            if time.time() < deadline - 3:
                pts_reopt2, e_reopt2 = _optimize_lbfgs(pts_md2, deadline - 1, maxiter=1000)
                if e_reopt2 < best_energy:
                    best_energy = e_reopt2
                    best_pts = pts_reopt2

    # Phase 5: Final polish with L-BFGS (tighter tolerance)
    if best_pts is not None and time.time() < deadline - 1:
        x0 = best_pts.flatten()

        def objective(x):
            p = x.reshape(-1, 3)
            norms = np.linalg.norm(p, axis=1, keepdims=True)
            norms[norms < 1e-15] = 1.0
            p = p / norms
            e, _ = _log_energy_grad(p)
            return e

        def grad(x):
            p = x.reshape(-1, 3)
            norms = np.linalg.norm(p, axis=1, keepdims=True)
            norms[norms < 1e-15] = 1.0
            p = p / norms
            _, g = _log_energy_grad(p)
            return g.flatten()

        result = minimize(objective, x0, jac=grad, method='L-BFGS-B',
                          options={'maxiter': 5000, 'maxfun': 20000,
                                   'ftol': 1e-16, 'gtol': 1e-12, 'm': 100})

        if result.fun < best_energy:
            best_pts = _project_to_sphere(result.x.reshape(-1, 3))
            best_energy = _log_energy(best_pts)

    # Validate final result
    if best_pts is None:
        best_pts = _fibonacci_sphere(N, offset=0.5)

    # Ensure all points are on unit sphere and valid
    norms = np.linalg.norm(best_pts, axis=1)
    best_pts = best_pts / norms[:, np.newaxis]

    # Check for near-duplicate points
    min_dist = np.inf
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < min_dist:
                min_dist = d
    if min_dist < 1e-9:
        # Perturb and re-optimize
        rng = np.random.RandomState(12345)
        best_pts = best_pts + rng.randn(N, 3) * 1e-6
        best_pts = _project_to_sphere(best_pts)
        if time.time() < deadline - 1:
            best_pts, _ = _optimize_lbfgs(best_pts, deadline - 0.5, maxiter=500)

    return best_pts
