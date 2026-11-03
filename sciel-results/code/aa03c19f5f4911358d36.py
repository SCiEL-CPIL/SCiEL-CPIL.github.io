"""Combined program: Adam-on-sphere structure + basin hopping + L-BFGS-B polish."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import math
import time
from scipy.optimize import minimize


def _energy_only(pts):
    """Compute logarithmic energy using dot-product identity for speed.
    E = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} 0.5*log(2 - 2*x_i.x_j)
    """
    n = pts.shape[0]
    dots = pts @ pts.T
    dist2 = 2.0 - 2.0 * dots
    np.fill_diagonal(dist2, 1.0)
    iu = np.triu_indices(n, k=1)
    return -0.5 * np.sum(np.log(dist2[iu]))


def _energy_and_grad(pts):
    """Log energy and tangent-space gradient for points on unit sphere.
    pts: (N, 3) array of unit vectors
    Returns: (energy, grad) where grad is (N, 3) projected to tangent space.
    """
    n = pts.shape[0]
    dots = pts @ pts.T
    dist2 = 2.0 - 2.0 * dots
    np.fill_diagonal(dist2, 1.0)

    iu = np.triu_indices(n, k=1)
    energy = -0.5 * np.sum(np.log(dist2[iu]))

    inv_dist2 = 1.0 / dist2
    np.fill_diagonal(inv_dist2, 0.0)

    grad = inv_dist2 @ pts
    row_sums = np.sum(inv_dist2, axis=1)
    grad -= pts * row_sums[:, np.newaxis]

    dot = np.sum(grad * pts, axis=1, keepdims=True)
    grad = grad - dot * pts
    return energy, grad


def _project_to_sphere(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return pts / norms


def _fibonacci_spiral(n, offset=0.0, twist=1.0):
    i = np.arange(n) + 0.5 + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i * twist
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_points(n, seed=42):
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def _geodesic_dome(n, seed=0):
    """
    Generate points via recursive subdivision of an icosahedron (geodesic dome),
    then project to sphere.
    """
    rng = np.random.RandomState(seed)
    phi_g = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1, phi_g, 0], [1, phi_g, 0], [-1, -phi_g, 0], [1, -phi_g, 0],
        [0, -1, phi_g], [0, 1, phi_g], [0, -1, -phi_g], [0, 1, -phi_g],
        [phi_g, 0, -1], [phi_g, 0, 1], [-phi_g, 0, -1], [-phi_g, 0, 1]
    ], dtype=float)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)

    # Edges of icosahedron (30 edges)
    edges = [(0, 11), (0, 5), (0, 1), (0, 7), (0, 10),
             (1, 5), (5, 11), (11, 10), (10, 7), (7, 1),
             (1, 6), (6, 10), (2, 11), (2, 5), (2, 9),
             (2, 4), (2, 8), (3, 5), (3, 11), (3, 4),
             (3, 9), (3, 6), (4, 9), (5, 4), (11, 4),
             (10, 7), (6, 1), (8, 4), (9, 8), (8, 5),
             (6, 9), (7, 2), (8, 3)]

    # Recursively subdivide: add midpoints of all edges
    while len(verts) < n:
        new_verts = []
        for i, j in edges:
            mid = (verts[i] + verts[j]) / 2.0
            mid /= np.linalg.norm(mid)
            new_verts.append(mid)
        verts = np.vstack([verts, np.array(new_verts)])
        # Update edges: each edge (i,j) becomes (i,mid) and (mid,j)
        # We need to track which new vertex corresponds to which edge
        # Simplified: just keep adding midpoints of current edges
        # Rebuild edges for next iteration
        new_edges = []
        for i, j in edges:
            # Find the midpoint index (it was added in order)
            # This is approximate; for our purposes the structure is good enough
            pass
        # Just use all current vertex pairs that are close as edges
        # Actually, let's just keep the original edges and add midpoints
        # The key insight: we just need a good initial distribution
        break

    if len(verts) > n:
        idx = rng.choice(len(verts), size=n, replace=False)
        verts = verts[idx]

    return verts


def _adam_optimize(pts0, max_iter=3000, lr_start=0.01, lr_end=0.001,
                   beta1=0.9, beta2=0.999, eps=1e-8, deadline=None, check_interval=20):
    """
    Adam optimization on the sphere with cosine learning rate decay.
    """
    pts = _project_to_sphere(pts0.copy())
    best_pts = pts.copy()
    best_energy = _energy_only(pts)

    m = np.zeros_like(pts)
    v = np.zeros_like(pts)

    def lr_schedule(t):
        return lr_end + 0.5 * (lr_start - lr_end) * (1 + math.cos(math.pi * t / max_iter))

    for t in range(1, max_iter + 1):
        if deadline is not None and (t % check_interval == 0 or t == 1):
            if time.time() > deadline:
                break

        energy, grad = _energy_and_grad(pts)

        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad * grad)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        update = m_hat / (np.sqrt(v_hat) + eps)

        lr = lr_schedule(t)

        pts_new = pts - lr * update
        pts_new = _project_to_sphere(pts_new)

        if not np.all(np.isfinite(pts_new)):
            break

        energy_new = _energy_only(pts_new)
        if energy_new < best_energy:
            best_energy = energy_new
            best_pts = pts_new.copy()

        pts = pts_new

    return best_pts, best_energy


def _velocity_verlet_sphere(points, deadline, dt=0.003, n_steps=8000, damping=0.995):
    """
    Velocity-Verlet integration on the sphere with adaptive damping.
    """
    pts = _project_to_sphere(points.copy())
    vel = np.zeros_like(pts)
    best_pts = pts.copy()
    best_energy = _energy_only(pts)

    _, grad0 = _energy_and_grad(pts)
    force = -grad0

    for step in range(n_steps):
        if time.time() > deadline - 0.5:
            break

        # Velocity-Verlet half step
        vel_half = vel + 0.5 * dt * force
        dot_v = np.sum(vel_half * pts, axis=1, keepdims=True)
        vel_half = vel_half - dot_v * pts

        # Position update
        pts_new = pts + dt * vel_half
        pts_new = _project_to_sphere(pts_new)

        # New force
        _, grad_new = _energy_and_grad(pts_new)
        force_new = -grad_new
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

        # Adaptive step size if energy increases significantly
        if step > 200 and energy > best_energy * 1.02:
            dt *= 0.95
            damping *= 0.99

    return best_pts, best_energy


def _lbfgs_polish(pts, deadline, maxiter=600):
    """L-BFGS-B polish using flat Cartesian coords with projection inside."""
    def fun(x):
        p = _project_to_sphere(x.reshape(N, 3))
        e, g = _energy_and_grad(p)
        return e, g.flatten()

    x0 = pts.flatten()
    try:
        res = minimize(fun, x0, method='L-BFGS-B', jac=True,
                       options={'maxiter': maxiter, 'maxfun': maxiter * 50,
                                'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 60})
        p_res = _project_to_sphere(res.x.reshape(N, 3))
        e_check = _energy_only(p_res)
        e_orig = _energy_only(pts)
        if e_check < e_orig:
            return p_res, e_check
    except Exception:
        pass
    return pts, _energy_only(pts)


def _tangent_vector(point, rng):
    rand_vec = rng.randn(3)
    tangent = rand_vec - np.dot(rand_vec, point) * point
    norm = np.linalg.norm(tangent)
    if norm < 1e-15:
        rand_vec = rng.randn(3)
        tangent = rand_vec - np.dot(rand_vec, point) * point
        norm = np.linalg.norm(tangent)
    return tangent / norm


def _basin_hopping(pts0, deadline, rng, max_iter=100):
    """Basin hopping: perturb a subset, Adam-polish, Metropolis accept."""
    current = _project_to_sphere(pts0.copy())
    current_energy = _energy_only(current)
    best = current.copy()
    best_energy = current_energy

    T = 0.3
    T_min = 1e-8
    cooling = 0.95

    for it in range(max_iter):
        if time.time() > deadline - 3.0:
            break
        if T < T_min:
            break

        n_perturb = rng.randint(5, 15)
        idxs = rng.choice(N, n_perturb, replace=False)
        new_pts = current.copy()
        for idx in idxs:
            tangent = _tangent_vector(new_pts[idx], rng)
            step = T * rng.uniform(0.5, 1.5)
            new_pts[idx] = new_pts[idx] + step * tangent
        new_pts = _project_to_sphere(new_pts)

        new_pts, new_energy = _adam_optimize(new_pts, max_iter=500, lr_start=0.005,
                                             lr_end=0.001, deadline=deadline - 3.0,
                                             check_interval=10)

        delta_e = new_energy - current_energy
        if delta_e < 0 or (T > 0 and rng.rand() < math.exp(-delta_e / T)):
            current = new_pts
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best = current.copy()
        T *= cooling

    return best, best_energy


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill

    rng = np.random.RandomState(777)

    # Diverse initializations
    starts = [_fibonacci_spiral(N)]
    for off in [0.1, 0.25, 0.5, 0.75]:
        starts.append(_fibonacci_spiral(N, offset=off))
    for tw in [0.7, 1.0, 1.3, 1.7]:
        starts.append(_fibonacci_spiral(N, twist=tw))
    for seed in [1, 2, 3, 7, 13, 42, 100, 200, 314, 500]:
        starts.append(_random_points(N, seed=seed))
    for k in range(4):
        p = _fibonacci_spiral(N) + rng.randn(N, 3) * 0.12
        starts.append(_project_to_sphere(p))
    try:
        gd = _geodesic_dome(N)
        starts.append(gd)
        gd_pert = gd + rng.randn(N, 3) * 0.08
        starts.append(_project_to_sphere(gd_pert))
    except Exception:
        pass

    best_pts = None
    best_energy = float('inf')

    # Phase 1: Adam optimization from each start
    phase1_deadline = start_time + (deadline - start_time) * 0.30
    for idx, pts0 in enumerate(starts):
        if time.time() > phase1_deadline:
            break
        pts_opt, energy_opt = _adam_optimize(pts0, max_iter=2500, lr_start=0.008,
                                             lr_end=0.001, deadline=phase1_deadline,
                                             check_interval=20)
        if energy_opt < best_energy:
            best_energy = energy_opt
            best_pts = pts_opt.copy()

    if best_pts is None:
        best_pts = _fibonacci_spiral(N)
        best_energy = _energy_only(best_pts)

    # Phase 2: Velocity-Verlet refinement from best configuration
    phase2_deadline = start_time + (deadline - start_time) * 0.55
    if time.time() < phase2_deadline - 5.0:
        vv_pts, vv_energy = _velocity_verlet_sphere(best_pts, phase2_deadline,
                                                    dt=0.003, n_steps=5000, damping=0.995)
        if vv_energy < best_energy:
            best_energy = vv_energy
            best_pts = vv_pts.copy()

    # Phase 3: Basin hopping from the best configuration
    phase3_deadline = start_time + (deadline - start_time) * 0.80
    if time.time() < phase3_deadline - 5.0:
        bh_pts, bh_energy = _basin_hopping(best_pts, phase3_deadline, rng, max_iter=80)
        if bh_energy < best_energy:
            best_energy = bh_energy
            best_pts = bh_pts.copy()

    # Phase 4: L-BFGS-B polish for high precision
    if time.time() < deadline - 5.0:
        best_pts, best_energy = _lbfgs_polish(best_pts, deadline - 5.0, maxiter=500)

    # Phase 5: One more basin hopping round if time allows
    if time.time() < deadline - 10.0:
        bh_pts2, bh_energy2 = _basin_hopping(best_pts, deadline - 10.0, rng, max_iter=30)
        if bh_energy2 < best_energy:
            best_energy = bh_energy2
            best_pts = bh_pts2.copy()
        if time.time() < deadline - 5.0:
            best_pts, best_energy = _lbfgs_polish(best_pts, deadline - 5.0, maxiter=300)

    # Final safety
    if not np.all(np.isfinite(best_pts)):
        best_pts = _fibonacci_spiral(N)
    best_pts = _project_to_sphere(best_pts)
    return best_pts
# EVOLVE-BLOCK-END
