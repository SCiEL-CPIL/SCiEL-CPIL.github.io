"""
Logarithmic energy minimization of 150 points on the unit sphere.

Strategy:
  - Multi-start basin-hopping search (Fibonacci, geodesic, random, perturbed).
  - Riemannian optimization: exponential-map steps and tangent-space L-BFGS
    polish that respect the sphere geometry exactly.
  - Velocity-Verlet and Adam phases for basin finding and refinement.
  - Local perturbation restarts using Riemannian perturbations.
  - Final high-precision polish pass.
"""

import numpy as np
import math
import time
from scipy.optimize import minimize

N = 150

# EVOLVE-BLOCK-START

# ---------------------------------------------------------------------------
# Energy and gradient
# ---------------------------------------------------------------------------
def _energy_and_grad(pts):
    """
    Logarithmic energy and tangent-space gradient.
    pts: (N, 3) unit vectors.
    E = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    grad_i (Cartesian) = sum_{j!=i} (x_j - x_i) / |x_i - x_j|^2, projected to tangent.
    """
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist2, 1.0)

    iu = np.triu_indices(n, k=1)
    dists = np.sqrt(dist2[iu])
    energy = -math.fsum(np.log(dists).tolist())

    inv_dist2 = 1.0 / dist2
    np.fill_diagonal(inv_dist2, 0.0)
    row_sums = np.sum(inv_dist2, axis=1)
    grad = inv_dist2 @ pts - pts * row_sums[:, np.newaxis]

    # Project to tangent space
    dot = np.sum(grad * pts, axis=1, keepdims=True)
    grad = grad - dot * pts
    return energy, grad


def _energy_only(pts):
    """Compute only the energy (fsum for precision)."""
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist2, 1.0)
    iu = np.triu_indices(n, k=1)
    dists = np.sqrt(dist2[iu])
    return -math.fsum(np.log(dists).tolist())


def _project_to_sphere(pts):
    """Normalize points to the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return pts / norms


# ---------------------------------------------------------------------------
# Riemannian (exponential map) utilities
# ---------------------------------------------------------------------------
def _exp_map_tangent(pts, tangent):
    """
    Exponential map on the sphere: move pts along tangent vectors.
    For a unit vector x and tangent v, the exponential map is
        exp_x(v) = cos(|v|) x + sin(|v|) (v/|v|)
    which is equivalent to rotating x by angle |v| about the axis v/|v|.
    This respects the sphere geometry exactly.
    """
    norm = np.linalg.norm(tangent, axis=1, keepdims=True)
    # Avoid division by zero
    safe_norm = np.where(norm > 1e-15, norm, 1.0)
    u = tangent / safe_norm
    c = np.cos(norm)
    s = np.sin(norm)
    # Where norm ~ 0, c ~ 1, s ~ 0, so result ~ x
    out = c * pts + s * u
    return _project_to_sphere(out)


def _log_map_tangent(pts, target):
    """
    Logarithmic map: tangent vector at pts that maps to target on the sphere.
    v = theta * (target - cos(theta) x) / sin(theta), theta = arccos(x . target).
    """
    cos_theta = np.sum(pts * target, axis=1, keepdims=True)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    sin_theta = np.sin(theta)
    # Avoid division by zero
    safe_sin = np.where(sin_theta > 1e-12, sin_theta, 1.0)
    v = (theta / safe_sin) * (target - cos_theta * pts)
    # Zero out where theta ~ 0
    v = np.where((theta > 1e-12)[:, np.newaxis], v, 0.0)
    return v


# ---------------------------------------------------------------------------
# Riemannian L-BFGS polish (optimizes in tangent space)
# ---------------------------------------------------------------------------
def _riemannian_lbfgs_polish(pts, deadline, maxiter=400, ftol=1e-16, gtol=1e-13):
    """
    Polish a configuration using L-BFGS-B on tangent-space coordinates.
    We represent the configuration as a flat tangent vector t (size 3N) and
    define f(t) = energy(exp(pts, t)), grad_f(t) = grad of energy at exp(pts,t)
    pulled back (which for the sphere exponential map with the metric is just
    the tangent gradient, since the exponential map is an isometry in the
    infinitesimal sense for our purposes here).

    We use a trust-region-like approach: start from t=0 (current config),
    optimize t, and if it improves, set pts = exp(pts, t_best) and repeat.
    """
    pts = _project_to_sphere(pts.copy())
    best_pts = pts.copy()
    best_energy = _energy_only(pts)

    def fun(t):
        # t is flat tangent vector (3N,). Interpret as (N,3) tangent.
        tv = t.reshape(N, 3)
        # Make sure tangent is actually tangent (subtract radial part)
        dot = np.sum(tv * pts, axis=1, keepdims=True)
        tv = tv - dot * pts
        new_pts = _exp_map_tangent(pts, tv)
        e, g = _energy_and_grad(new_pts)
        # Gradient of f w.r.t. t: since exp is local isometry, g (tangent) is
        # the gradient in t-space directly.
        return e, g.flatten()

    t0 = np.zeros(3 * N)
    try:
        res = minimize(fun, t0, method='L-BFGS-B', jac=True,
                       options={'maxiter': maxiter, 'ftol': ftol,
                                'gtol': gtol, 'maxls': 30})
        if time.time() < deadline:
            tv = res.x.reshape(N, 3)
            dot = np.sum(tv * pts, axis=1, keepdims=True)
            tv = tv - dot * pts
            new_pts = _exp_map_tangent(pts, tv)
            new_energy = _energy_only(new_pts)
            if new_energy < best_energy and np.all(np.isfinite(new_pts)):
                best_energy = new_energy
                best_pts = new_pts.copy()
    except Exception:
        pass

    return best_pts, best_energy


# ---------------------------------------------------------------------------
# Cartesian L-BFGS polish (fallback / final)
# ---------------------------------------------------------------------------
def _lbfgs_polish(pts, deadline, maxiter=500, ftol=1e-16, gtol=1e-14):
    """L-BFGS-B polish using flat Cartesian coordinates with projection."""
    pts = _project_to_sphere(pts.copy())
    best_pts = pts.copy()
    best_energy = _energy_only(pts)

    def fun(x):
        p = _project_to_sphere(x.reshape(N, 3))
        e, g = _energy_and_grad(p)
        return e, g.flatten()

    x0 = pts.flatten()
    try:
        res = minimize(fun, x0, method='L-BFGS-B', jac=True,
                       options={'maxiter': maxiter, 'ftol': ftol,
                                'gtol': gtol, 'maxls': 30})
        p_res = _project_to_sphere(res.x.reshape(N, 3))
        e_res = _energy_only(p_res)
        if e_res < best_energy and np.all(np.isfinite(p_res)):
            best_energy = e_res
            best_pts = p_res.copy()
    except Exception:
        pass
    return best_pts, best_energy


# ---------------------------------------------------------------------------
# Initial configurations
# ---------------------------------------------------------------------------
def _fibonacci_spiral(n, offset=0.0, twist=1.0):
    i = np.arange(n) + 0.5 + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i * twist
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _geodesic_dome(n, seed=0):
    """
    Generate points via recursive subdivision of an icosahedron.
    Provides a different structure than the Fibonacci spiral.
    """
    rng = np.random.RandomState(seed)
    # Start with icosahedron vertices
    phi_g = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1, phi_g, 0], [1, phi_g, 0], [-1, -phi_g, 0], [1, -phi_g, 0],
        [0, -1, phi_g], [0, 1, phi_g], [0, -1, -phi_g], [0, 1, -phi_g],
        [phi_g, 0, -1], [phi_g, 0, 1], [-phi_g, 0, -1], [-phi_g, 0, 1]
    ], dtype=float)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)

    # Edges of icosahedron
    edges = [(0,11),(0,5),(0,1),(0,7),(0,10),(1,5),(5,11),(11,10),(10,7),
             (7,1),(1,6),(6,10),(2,11),(2,5),(2,9),(2,4),(2,8),(3,5),
             (3,11),(3,4),(3,9),(3,6),(4,9),(5,4),(11,4),(10,7),(6,1),
             (8,4),(9,8),(8,5),(6,9),(7,2),(8,3)]

    # Recursively subdivide until we have enough points
    while len(verts) < n:
        new_verts = []
        for i, j in edges:
            mid = (verts[i] + verts[j]) / 2.0
            mid /= np.linalg.norm(mid)
            new_verts.append(mid)
        verts = np.vstack([verts, np.array(new_verts)])

    # If we have more than n, randomly select n
    if len(verts) > n:
        idx = rng.choice(len(verts), size=n, replace=False)
        verts = verts[idx]

    return verts


def _random_points(n, seed=42):
    """Generate random points on the sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def _golden_spiral(n):
    """Generate points using the golden angle for a different spiral pattern."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = 2.399963229728653 * i  # Golden angle in radians
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


# ---------------------------------------------------------------------------
# Optimizers
# ---------------------------------------------------------------------------
def _velocity_verlet_sphere(points, deadline, dt=0.005, n_steps=5000, damping=0.98):
    """
    Velocity-Verlet integration on the sphere using Riemannian exponential map.
    """
    pts = _project_to_sphere(points.copy())
    vel = np.zeros_like(pts)
    best_pts = pts.copy()
    best_energy = _energy_only(pts)

    # Initial force
    _, grad0 = _energy_and_grad(pts)
    force = -grad0

    for step in range(n_steps):
        if time.time() > deadline - 0.5:
            break

        # Velocity-Verlet half step
        vel_half = vel + 0.5 * dt * force
        # Project velocity to tangent space
        dot_v = np.sum(vel_half * pts, axis=1, keepdims=True)
        vel_half = vel_half - dot_v * pts

        # Position update using exponential map
        pts_new = _exp_map_tangent(pts, dt * vel_half)

        # New force
        _, grad_new = _energy_and_grad(pts_new)
        force_new = -grad_new
        # Project force to tangent space
        dot_f = np.sum(force_new * pts_new, axis=1, keepdims=True)
        force_new = force_new - dot_f * pts_new

        # Velocity update with damping
        vel = (vel_half + 0.5 * dt * force_new) * damping
        dot_v2 = np.sum(vel * pts_new, axis=1, keepdims=True)
        vel = vel - dot_v2 * pts_new

        pts = pts_new
        force = force_new

        energy = _energy_only(pts)
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()

        # Adaptive damping or step size if energy increases
        if step > 100 and energy > best_energy * 1.05:
            dt *= 0.9
            damping *= 0.99

    return best_pts, best_energy


def _adam_optimize(pts0, max_iter=3000, lr_start=0.01, lr_end=0.001,
                   beta1=0.9, beta2=0.999, eps=1e-8, deadline=None, check_interval=20):
    """
    Adam optimization on the sphere with learning rate decay and adaptive step.
    Uses Riemannian exponential map for updates.
    """
    pts = _project_to_sphere(pts0.copy())
    best_pts = pts.copy()
    best_energy = _energy_only(pts)

    m = np.zeros_like(pts)
    v = np.zeros_like(pts)

    # Cosine decay for learning rate
    def lr_schedule(t):
        return lr_end + 0.5 * (lr_start - lr_end) * (1 + math.cos(math.pi * t / max_iter))

    # Adaptive learning rate multiplier
    adaptive_lr_mult = 1.0

    for t in range(1, max_iter + 1):
        if deadline is not None and (t % check_interval == 0 or t == 1):
            if time.time() > deadline:
                break

        energy, grad = _energy_and_grad(pts)

        # Adam update
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad * grad)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        update = m_hat / (np.sqrt(v_hat) + eps)

        lr = lr_schedule(t) * adaptive_lr_mult

        # Step in tangent direction using exponential map
        pts_new = _exp_map_tangent(pts, -lr * update)

        if not np.all(np.isfinite(pts_new)):
            adaptive_lr_mult *= 0.5
            continue

        energy_new = _energy_only(pts_new)

        # Adaptive learning rate adjustment
        if energy_new < best_energy:
            best_energy = energy_new
            best_pts = pts_new.copy()
            # Increase adaptive multiplier if we're making good progress
            if t > 100:
                adaptive_lr_mult = min(adaptive_lr_mult * 1.1, 3.0)
        else:
            # Decrease adaptive multiplier if we're not improving
            adaptive_lr_mult = max(adaptive_lr_mult * 0.8, 0.1)

        pts = pts_new

    return best_pts, best_energy


def _riemannian_perturb(pts, rng, scale=0.1, n_perturb=10):
    """
    Create a perturbed version of pts by moving n_perturb points along
    random tangent directions using the exponential map.
    """
    p = pts.copy()
    idx = rng.choice(N, size=n_perturb, replace=False)
    # Generate random tangent vectors at the selected points
    tangents = np.zeros((n_perturb, 3))
    for i, idx_i in enumerate(idx):
        # Random direction
        rand_dir = rng.randn(3)
        # Make tangent to sphere at p[idx_i]
        dot = np.dot(rand_dir, p[idx_i])
        rand_dir -= dot * p[idx_i]
        norm = np.linalg.norm(rand_dir)
        if norm > 1e-15:
            rand_dir /= norm
        tangents[i] = rand_dir * (rng.randn() * scale)
    # Apply exponential map to selected points
    p[idx] = _exp_map_tangent(p[idx], tangents)
    return _project_to_sphere(p)


def _local_perturb_restart(pts, rng, perturb_scale=0.05, n_perturb=10):
    """
    Create perturbed versions of pts by randomly moving n_perturb points
    using Riemannian perturbations.
    Returns list of perturbed point arrays.
    """
    results = []
    for _ in range(5):
        p = _riemannian_perturb(pts, rng, scale=perturb_scale, n_perturb=n_perturb)
        results.append(p)
    return results


def _final_grad_polish(pts, max_iter=5000, tol=1e-15, deadline=None):
    """
    High-precision gradient descent with backtracking line search on the sphere.
    Uses exact tangent-space projection and adaptive step sizes.
    """
    pts = _project_to_sphere(pts.copy())
    best_pts = pts.copy()
    best_energy = _energy_only(pts)

    # Compute initial gradient
    _, grad = _energy_and_grad(pts)

    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break

        # Check convergence
        grad_norm = np.linalg.norm(grad)
        if grad_norm < tol:
            break

        # Backtracking line search
        step = 0.1
        for _ in range(20):
            # Move in negative gradient direction using exponential map
            pts_new = _exp_map_tangent(pts, -step * grad)

            if not np.all(np.isfinite(pts_new)):
                step *= 0.5
                continue

            energy_new = _energy_only(pts_new)

            if energy_new < best_energy:
                best_energy = energy_new
                best_pts = pts_new.copy()

            # Armijo condition
            if energy_new <= best_energy - 1e-4 * step * grad_norm * grad_norm:
                break

            step *= 0.5

        # Update
        pts = _exp_map_tangent(pts, -step * grad)

        # Recompute gradient
        _, grad = _energy_and_grad(pts)

    return best_pts, best_energy


def _lbfgs_tight_polish(pts, deadline, maxiter=1000):
    """
    Ultra-tight L-BFGS-B polish.
    """
    def fun(x):
        p = _project_to_sphere(x.reshape(N, 3))
        e, g = _energy_and_grad(p)
        return e, g.flatten()

    x0 = pts.flatten()
    try:
        res = minimize(fun, x0, method='L-BFGS-B', jac=True,
                       options={'maxiter': maxiter, 'maxfun': maxiter * 50,
                                'ftol': 1e-18, 'gtol': 1e-14, 'maxls': 60})
        p_res = _project_to_sphere(res.x.reshape(N, 3))
        e_check = _energy_only(p_res)
        if e_check < _energy_only(pts):
            return p_res
    except Exception:
        pass
    return pts


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin for 150s limit

    # Define multiple initializations
    # 1. Fibonacci spiral
    init1 = _fibonacci_spiral(N)
    # 2. Fibonacci spiral with different twist
    init2 = _fibonacci_spiral(N, twist=1.5)
    # 3. Random points
    init3 = _random_points(N, seed=123)
    # 4. Random points different seed
    init4 = _random_points(N, seed=456)
    # 5. Perturbed Fibonacci
    rng = np.random.RandomState(999)
    init5 = _project_to_sphere(_fibonacci_spiral(N) + rng.randn(N, 3) * 0.1)
    # 6. Geodesic dome
    init6 = _geodesic_dome(N, seed=7)
    # 7. Golden spiral
    init7 = _golden_spiral(N)

    inits = [init1, init2, init3, init4, init5, init6, init7]

    best_pts_global = None
    best_energy_global = float('inf')

    # Time per restart
    time_per_restart = 30.0
    num_restarts = 4

    for i in range(num_restarts):
        current_deadline = start_time + (i + 1) * time_per_restart
        if current_deadline > deadline - 30.0:  # Leave 30s for perturbation restarts + final polish
            current_deadline = deadline - 30.0

        pts_start = inits[i % len(inits)]

        # Phase 1: Velocity Verlet (fast basin finding)
        vv_deadline = current_deadline - 12.0
        if time.time() < vv_deadline:
            pts_vv, e_vv = _velocity_verlet_sphere(pts_start, vv_deadline, dt=0.005, n_steps=2500)
        else:
            pts_vv, e_vv = pts_start, _energy_only(pts_start)

        # Phase 2: Adam (refine)
        adam_deadline = current_deadline - 4.0
        if time.time() < adam_deadline:
            pts_adam, e_adam = _adam_optimize(pts_vv, max_iter=1200, lr_start=0.005, lr_end=0.0005, deadline=adam_deadline)
        else:
            pts_adam, e_adam = pts_vv, e_vv

        # Phase 3: Riemannian L-BFGS (polish)
        if time.time() < current_deadline - 1.0:
            pts_lbfgs, e_lbfgs = _riemannian_lbfgs_polish(pts_adam, current_deadline - 1.0, maxiter=300)
        else:
            pts_lbfgs, e_lbfgs = pts_adam, e_adam

        if e_lbfgs < best_energy_global:
            best_energy_global = e_lbfgs
            best_pts_global = pts_lbfgs.copy()

    # Local Perturbation Restart Phase
    # Try perturbing the best solution to escape local minima
    if best_pts_global is not None and time.time() < deadline - 10.0:
        perturb_deadline = deadline - 10.0
        rng_perturb = np.random.RandomState(42)

        # Try several perturbations with varying scales and point counts
        for scale, n_pert in [(0.1, 12), (0.06, 8), (0.03, 5), (0.015, 3)]:
            if time.time() > perturb_deadline - 5.0:
                break

            perturbations = _local_perturb_restart(best_pts_global, rng_perturb,
                                                   perturb_scale=scale, n_perturb=n_pert)
            for pert_pts in perturbations:
                if time.time() > perturb_deadline - 2.0:
                    break

                # Quick Adam refinement
                if time.time() < perturb_deadline - 3.0:
                    pts_ref, e_ref = _adam_optimize(pert_pts, max_iter=400,
                                                     lr_start=0.005, lr_end=0.0005,
                                                     deadline=perturb_deadline - 3.0)
                    # Quick Riemannian L-BFGS polish
                    if time.time() < perturb_deadline - 1.0:
                        pts_pol, e_pol = _riemannian_lbfgs_polish(pts_ref, perturb_deadline - 1.0, maxiter=150)
                    else:
                        pts_pol, e_pol = pts_ref, e_ref

                    if e_pol < best_energy_global:
                        best_energy_global = e_pol
                        best_pts_global = pts_pol.copy()

    # Final Polish Phase
    # Use remaining time to polish the global best
    remaining_deadline = deadline - 2.0

    if best_pts_global is None:
        best_pts_global = _fibonacci_spiral(N)
        best_energy_global = _energy_only(best_pts_global)

    # Riemannian L-BFGS with more iterations
    if time.time() < remaining_deadline - 5.0:
        best_pts_global, _ = _riemannian_lbfgs_polish(best_pts_global, remaining_deadline - 5.0, maxiter=2000)
        best_energy_global = _energy_only(best_pts_global)

    # High-precision gradient descent polish
    if time.time() < remaining_deadline - 2.0:
        best_pts_global, _ = _final_grad_polish(best_pts_global, max_iter=3000,
                                                 tol=1e-16, deadline=remaining_deadline - 2.0)
        best_energy_global = _energy_only(best_pts_global)

    # One more standard L-BFGS with different settings
    if time.time() < remaining_deadline - 0.5:
        best_pts_global, _ = _lbfgs_polish(best_pts_global, remaining_deadline - 0.5, maxiter=300)
        best_energy_global = _energy_only(best_pts_global)

    best_pts_global = _project_to_sphere(best_pts_global)

    # Validation
    if not np.all(np.isfinite(best_pts_global)):
        return _fibonacci_spiral(N)

    diff = best_pts_global[:, np.newaxis, :] - best_pts_global[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    iu = np.triu_indices(N, k=1)
    if np.min(dists[iu]) < 1e-9:
        return _fibonacci_spiral(N)

    return best_pts_global

# EVOLVE-BLOCK-END
