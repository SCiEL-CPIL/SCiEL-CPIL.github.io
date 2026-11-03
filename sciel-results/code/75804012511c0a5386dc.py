"""Combined program: Adam-on-sphere structure + L-BFGS-B/basin-hopping refinement."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import math
import time
from scipy.optimize import minimize


def _energy_and_grad(pts):
    """Log energy and tangent-space gradient for points on unit sphere.
    pts: (N, 3) array of unit vectors
    Returns: (energy, grad) where grad is (N, 3) projected to tangent space.
    """
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist2, 1.0)

    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(np.sqrt(dist2[iu])))

    inv_dist2 = 1.0 / dist2
    np.fill_diagonal(inv_dist2, 0.0)

    # grad[i] = sum_{j != i} (x_j - x_i) / dist_ij^2
    grad = inv_dist2 @ pts
    row_sums = np.sum(inv_dist2, axis=1)
    grad -= pts * row_sums[:, np.newaxis]

    # Project to tangent space
    dot = np.sum(grad * pts, axis=1, keepdims=True)
    grad = grad - dot * pts
    return energy, grad


def _energy_only(pts):
    """Compute just the energy (fast path, no gradient)."""
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist2, 1.0)
    iu = np.triu_indices(n, k=1)
    return -np.sum(np.log(np.sqrt(dist2[iu])))


def _project_to_sphere(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return pts / norms


def _fibonacci_spiral(n, offset=0.0, twist=1.0):
    i = np.arange(n) + 0.5 + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i * twist
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_points(n, seed=42):
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def _geodesic_dome(n_target, subdivisions=6):
    """Generate ~n_target points on the sphere via a subdivided icosahedron.
    Returns exactly n_target points (duplicates removed, extras trimmed)."""
    t = (1.0 + np.sqrt(5.0)) / 2.0
    # 12 vertices of icosahedron
    verts = np.array([
        [-1,  t, 0], [1,  t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1,  t], [0, 1,  t], [0, -1, -t], [0, 1, -t],
        [ t, 0, -1], [ t, 0, 1], [-t, 0, -1], [-t, 0, 1],
    ], dtype=float)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)

    # 20 faces (triangles) of the icosahedron
    faces = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
    ]

    def subdivide(face, sub):
        """Recursively subdivide a triangular face `sub` times."""
        if sub == 0:
            return [face]
        a, b, c = face
        ab = _midpoint_on_sphere(verts[a], verts[b])
        bc = _midpoint_on_sphere(verts[b], verts[c])
        ca = _midpoint_on_sphere(verts[c], verts[a])
        return (subdivide((a, ab, ca), sub - 1) +
                subdivide((ab, b, bc), sub - 1) +
                subdivide((ca, bc, c), sub - 1) +
                subdivide((ab, bc, ca), sub - 1))

    def _midpoint_on_sphere(p1, p2):
        m = (p1 + p2) / 2.0
        return m / np.linalg.norm(m)

    # Collect all unique vertices from subdivided faces
    all_faces = []
    for f in faces:
        all_faces.extend(subdivide(f, subdivisions))

    # Gather unique vertices
    seen = set()
    unique_verts = []
    for f in all_faces:
        for idx in f:
            if idx not in seen:
                seen.add(idx)
                unique_verts.append(verts[idx])

    pts = np.array(unique_verts)
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)

    # If we have more than n_target, thin them out uniformly
    if len(pts) > n_target:
        # Sort by angle for deterministic selection
        order = np.argsort(np.arctan2(pts[:, 1], pts[:, 0]) +
                           np.arccos(pts[:, 2]) * 0.1)
        step = len(pts) / n_target
        selected = [int(round(i * step)) % len(pts) for i in range(n_target)]
        selected = sorted(set(selected))
        if len(selected) < n_target:
            # Fill remaining with random indices
            remaining = [i for i in range(len(pts)) if i not in set(selected)]
            rng = np.random.RandomState(123)
            extra = rng.choice(remaining, n_target - len(selected), replace=False)
            selected = sorted(set(selected) | set(extra))
        pts = pts[selected[:n_target]]
    elif len(pts) < n_target:
        # Pad with random points
        rng = np.random.RandomState(456)
        extra = rng.randn(n_target - len(pts), 3)
        extra = extra / np.linalg.norm(extra, axis=1, keepdims=True)
        pts = np.vstack([pts, extra])

    return pts[:n_target]


def _adam_optimize(pts0, max_iter=4000, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8,
                   deadline=None, check_interval=25):
    """Adam optimization on the sphere using tangent-space gradient."""
    pts = _project_to_sphere(pts0.copy())
    best_pts = pts.copy()
    best_energy = _energy_only(pts)

    m = np.zeros_like(pts)
    v = np.zeros_like(pts)

    start_time = time.time()

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

        # Step in tangent direction then re-project
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


def _lbfgs_polish(pts, deadline, maxiter=600):
    """L-BFGS-B polish using flat Cartesian coords with projection inside."""
    def fun(x):
        p = _project_to_sphere(x.reshape(N, 3))
        e, g = _energy_and_grad(p)
        return e, g.flatten()

    x0 = pts.flatten()
    try:
        res = minimize(fun, x0, method='L-BFGS-B', jac=True,
                       options={'maxiter': maxiter, 'maxfun': maxiter * 40,
                                'ftol': 1e-18, 'gtol': 1e-14, 'maxls': 50,
                                'iprint': -1})
        p_res = _project_to_sphere(res.x.reshape(N, 3))
        e_check = _energy_only(p_res)
        if e_check < _energy_only(pts):
            return p_res
    except Exception:
        pass
    return pts


def _tangent_vector(point, rng):
    rand_vec = rng.randn(3)
    tangent = rand_vec - np.dot(rand_vec, point) * point
    norm = np.linalg.norm(tangent)
    if norm < 1e-15:
        rand_vec = rng.randn(3)
        tangent = rand_vec - np.dot(rand_vec, point) * point
        norm = np.linalg.norm(tangent)
    return tangent / norm


def _basin_hopping(pts0, deadline, rng, max_iter=80):
    """Basin hopping: perturb a subset, Adam-polish, Metropolis accept."""
    current = _project_to_sphere(pts0.copy())
    current_energy = _energy_only(current)
    best = current.copy()
    best_energy = current_energy

    T = 0.15
    T_min = 1e-7
    cooling = 0.97

    for it in range(max_iter):
        if time.time() > deadline - 2.0:
            break
        if T < T_min:
            break

        n_perturb = rng.randint(3, 10)
        idxs = rng.choice(N, n_perturb, replace=False)
        new_pts = current.copy()
        for idx in idxs:
            tangent = _tangent_vector(new_pts[idx], rng)
            new_pts[idx] = new_pts[idx] + T * tangent
        new_pts = _project_to_sphere(new_pts)

        # Short Adam polish on the perturbed config
        new_pts, new_energy = _adam_optimize(new_pts, max_iter=800, lr=0.005,
                                             deadline=deadline - 2.0, check_interval=10)

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
    deadline = start_time + 135.0  # 15s safety margin

    rng = np.random.RandomState(777)

    # Diverse initializations
    starts = [_fibonacci_spiral(N)]
    for off in [0.1, 0.3, 0.5, 0.7]:
        starts.append(_fibonacci_spiral(N, offset=off))
    for tw in [0.8, 1.3, 2.0]:
        starts.append(_fibonacci_spiral(N, twist=tw))
    for seed in [1, 2, 3, 7, 13, 42, 100, 200]:
        starts.append(_random_points(N, seed=seed))
    # Perturbed Fibonacci
    for k in range(3):
        p = _fibonacci_spiral(N) + rng.randn(N, 3) * 0.15
        starts.append(_project_to_sphere(p))
    # Geodesic dome starts (uniform spacing enforced by construction)
    for sub in [5, 6]:
        try:
            dome = _geodesic_dome(N, subdivisions=sub)
            if dome is not None and len(dome) == N and np.all(np.isfinite(dome)):
                starts.append(dome)
        except Exception:
            pass
    # Perturbed geodesic dome
    try:
        dome0 = _geodesic_dome(N, subdivisions=5)
        if dome0 is not None and len(dome0) == N:
            p = dome0 + rng.randn(N, 3) * 0.1
            starts.append(_project_to_sphere(p))
    except Exception:
        pass

    best_pts = None
    best_energy = float('inf')

    # Phase 1: Adam optimization from each start
    for idx, pts0 in enumerate(starts):
        if time.time() > deadline - 40.0:
            break
        pts_opt, energy_opt = _adam_optimize(pts0, max_iter=3000, lr=0.005,
                                             deadline=deadline - 40.0,
                                             check_interval=25)
        if energy_opt < best_energy:
            best_energy = energy_opt
            best_pts = pts_opt.copy()

    if best_pts is None:
        best_pts = _fibonacci_spiral(N)
        best_energy = _energy_only(best_pts)

    # Phase 2: basin hopping from the best configuration
    if time.time() < deadline - 25.0:
        bh_pts, bh_energy = _basin_hopping(best_pts, deadline - 25.0, rng,
                                           max_iter=60)
        if bh_energy < best_energy:
            best_energy = bh_energy
            best_pts = bh_pts.copy()

    # Phase 3: L-BFGS-B polish for high precision
    if time.time() < deadline - 12.0:
        best_pts = _lbfgs_polish(best_pts, deadline - 12.0, maxiter=600)
        best_energy = _energy_only(best_pts)

    # Phase 4: Final fine-tune with very small Adam steps for last bits of energy
    if time.time() < deadline - 3.0:
        fine_pts, fine_energy = _adam_optimize(
            best_pts, max_iter=500, lr=0.001,
            deadline=deadline - 3.0, check_interval=10)
        if fine_energy < best_energy:
            best_energy = fine_energy
            best_pts = fine_pts

    # Final safety
    if not np.all(np.isfinite(best_pts)):
        best_pts = _fibonacci_spiral(N)
    best_pts = _project_to_sphere(best_pts)
    return best_pts
# EVOLVE-BLOCK-END
