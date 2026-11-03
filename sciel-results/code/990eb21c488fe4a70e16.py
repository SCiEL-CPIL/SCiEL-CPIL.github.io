"""Logarithmic energy minimization of 150 points on the unit sphere.
Uses geodesic dome initialization, Adam on sphere, basin hopping, and L-BFGS-B polish."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import math
import time
from scipy.optimize import minimize


def _energy_only(pts):
    """Compute logarithmic energy using dot-product identity.
    E = sum_{i<j} log(1/|x_i - x_j|) = -0.5 * sum_{i<j} log(2 - 2*x_i.x_j)
    """
    n = pts.shape[0]
    dots = pts @ pts.T
    dist2 = 2.0 - 2.0 * dots
    np.fill_diagonal(dist2, 1.0)
    iu = np.triu_indices(n, k=1)
    return -0.5 * np.sum(np.log(dist2[iu]))


def _energy_and_grad(pts):
    """Log energy and tangent-space gradient for points on unit sphere."""
    n = pts.shape[0]
    dots = pts @ pts.T
    dist2 = 2.0 - 2.0 * dots
    np.fill_diagonal(dist2, 1.0)

    iu = np.triu_indices(n, k=1)
    energy = -0.5 * np.sum(np.log(dist2[iu]))

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


def _geodesic_dome(n_target):
    """Generate points via subdivided icosahedron, then select n_target most evenly spaced."""
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1,  t, 0], [1,  t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1,  t], [0, 1,  t], [0, -1, -t], [0, 1, -t],
        [ t, 0, -1], [ t, 0, 1], [-t, 0, -1], [-t, 0, 1],
    ], dtype=float)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)

    faces = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
    ]

    # Subdivide icosahedron 4 times to get 12*4^4 = 3072 vertices
    # Each subdivision: for each face, add midpoints and subdivide into 4 faces
    for _ in range(4):
        new_faces = []
        mid_cache = {}
        for face in faces:
            a, b, c = face
            for edge in [(a, b), (b, c), (c, a)]:
                key = (min(edge), max(edge))
                if key not in mid_cache:
                    m = (verts[edge[0]] + verts[edge[1]]) / 2.0
                    m /= np.linalg.norm(m)
                    mid_cache[key] = len(verts)
                    verts = np.vstack([verts, m])
            ab = mid_cache[(min(a, b), max(a, b))]
            bc = mid_cache[(min(b, c), max(b, c))]
            ca = mid_cache[(min(c, a), max(c, a))]
            new_faces.extend([(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)])
        faces = new_faces

    # Now we have ~3072 points. Select n_target most evenly spaced.
    # Use a greedy farthest-point sampling to pick n_target points.
    if len(verts) <= n_target:
        return verts
    selected = [0]  # Start with vertex 0
    for _ in range(n_target - 1):
        # Compute min distance from each unselected point to selected points
        best_idx = -1
        best_min_dist = -1
        for i in range(len(verts)):
            if i in selected:
                continue
            min_d = min(np.linalg.norm(verts[i] - verts[j]) for j in selected)
            if min_d > best_min_dist:
                best_min_dist = min_d
                best_idx = i
        selected.append(best_idx)

    return verts[selected]


def _adam_optimize(pts0, max_iter=5000, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8,
                   deadline=None, check_interval=20):
    """Adam optimization on the sphere using tangent-space gradient."""
    pts = _project_to_sphere(pts0.copy())
    best_pts = pts.copy()
    best_energy = _energy_only(pts)

    m = np.zeros_like(pts)
    v = np.zeros_like(pts)

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


def _lbfgs_polish(pts, maxiter=600):
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


def _basin_hopping(pts0, deadline, rng, max_iter=120):
    """Basin hopping: perturb a subset, Adam-polish, Metropolis accept."""
    current = _project_to_sphere(pts0.copy())
    current_energy = _energy_only(current)
    best = current.copy()
    best_energy = current_energy

    T = 0.25
    T_min = 1e-8
    cooling = 0.96

    for it in range(max_iter):
        if time.time() > deadline - 3.0:
            break
        if T < T_min:
            break

        # Perturb a random subset with larger steps
        n_perturb = rng.randint(5, 15)
        idxs = rng.choice(N, n_perturb, replace=False)
        new_pts = current.copy()
        for idx in idxs:
            tangent = _tangent_vector(new_pts[idx], rng)
            step = T * rng.uniform(0.5, 1.5)
            new_pts[idx] = new_pts[idx] + step * tangent
        new_pts = _project_to_sphere(new_pts)

        # Short Adam polish on the perturbed config
        new_pts, new_energy = _adam_optimize(new_pts, max_iter=600, lr=0.005,
                                             deadline=deadline - 3.0, check_interval=10)

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
    # Perturbed Fibonacci
    for k in range(4):
        p = _fibonacci_spiral(N) + rng.randn(N, 3) * 0.12
        starts.append(_project_to_sphere(p))
    # Geodesic dome based
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
    phase1_deadline = start_time + (deadline - start_time) * 0.35
    for idx, pts0 in enumerate(starts):
        if time.time() > phase1_deadline:
            break
        pts_opt, energy_opt = _adam_optimize(pts0, max_iter=3000, lr=0.005,
                                             deadline=phase1_deadline,
                                             check_interval=20)
        if energy_opt < best_energy:
            best_energy = energy_opt
            best_pts = pts_opt.copy()

    if best_pts is None:
        best_pts = _fibonacci_spiral(N)
        best_energy = _energy_only(best_pts)

    # Phase 2: basin hopping from the best configuration
    phase2_deadline = start_time + (deadline - start_time) * 0.80
    if time.time() < phase2_deadline - 10.0:
        bh_pts, bh_energy = _basin_hopping(best_pts, phase2_deadline, rng,
                                           max_iter=100)
        if bh_energy < best_energy:
            best_energy = bh_energy
            best_pts = bh_pts.copy()

    # Phase 3: L-BFGS-B polish for high precision
    if time.time() < deadline - 5.0:
        best_pts, best_energy = _lbfgs_polish(best_pts, maxiter=500)

    # Phase 4: One more basin hopping round if time allows
    if time.time() < deadline - 10.0:
        bh_pts2, bh_energy2 = _basin_hopping(best_pts, deadline - 10.0, rng,
                                             max_iter=30)
        if bh_energy2 < best_energy:
            best_energy = bh_energy2
            best_pts = bh_pts2.copy()
        if time.time() < deadline - 5.0:
            best_pts, best_energy = _lbfgs_polish(best_pts, maxiter=300)

    # Final safety
    if not np.all(np.isfinite(best_pts)):
        best_pts = _fibonacci_spiral(N)
    best_pts = _project_to_sphere(best_pts)
    return best_pts
# EVOLVE-BLOCK-END
