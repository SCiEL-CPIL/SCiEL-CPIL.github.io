import numpy as np
import time
import math
from scipy.optimize import minimize

N = 150


def _log_energy(points):
    """Compute log(1/d) energy for points on unit sphere using math.fsum."""
    n = len(points)
    iu = np.triu_indices(n, k=1)
    diff = points[iu[0]] - points[iu[1]]
    dists_sq = np.sum(diff * diff, axis=-1)
    dists = np.sqrt(np.maximum(dists_sq, 1e-30))
    log_dists = np.log(dists)
    return -math.fsum(log_dists.tolist())


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
    dists = np.sqrt(np.maximum(dist2[iu], 1e-30))
    energy = -math.fsum(np.log(dists).tolist())

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
    dists = np.sqrt(np.maximum(dist2[iu], 1e-30))
    return -math.fsum(np.log(dists).tolist())


def _project_to_sphere(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return pts / norms


def _fibonacci_sphere(n):
    """Generate Fibonacci sphere points."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_uniform_sphere(n, rng):
    """Generate random uniform points on sphere."""
    points = rng.randn(n, 3)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    return points / norms


def _subdivide_icosahedron(subdivisions):
    """
    Create a geodesic dome by subdividing an icosahedron.
    Returns vertices on the unit sphere.
    """
    phi = (1.0 + math.sqrt(5.0)) / 2.0

    verts = np.array([
        [-1,  phi, 0], [ 1,  phi, 0], [-1, -phi, 0], [ 1, -phi, 0],
        [ 0, -1,  phi], [ 0,  1,  phi], [ 0, -1, -phi], [ 0,  1, -phi],
        [ phi, 0, -1], [ phi, 0,  1], [-phi, 0, -1], [-phi, 0,  1]
    ], dtype=float)

    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms

    faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ]

    for _ in range(subdivisions):
        new_faces = []
        mid_cache = {}

        for face in faces:
            a, b, c = face

            def get_mid(p1, p2):
                key = (min(p1, p2), max(p1, p2))
                if key not in mid_cache:
                    mid = (verts[p1] + verts[p2]) / 2.0
                    mid = mid / np.linalg.norm(mid)
                    verts = np.vstack([verts, mid])
                    mid_cache[key] = len(verts) - 1
                return mid_cache[key]

            ab = get_mid(a, b)
            bc = get_mid(b, c)
            ca = get_mid(c, a)

            new_faces.extend([
                [a, ab, ca],
                [b, bc, ab],
                [c, ca, bc],
                [ab, bc, ca]
            ])

        faces = new_faces

    return verts


def _geodesic_dome_init(n_target, subdivisions=3):
    """
    Generate a geodesic dome with approximately n_target points.
    """
    best_verts = None
    best_diff = float('inf')

    for subdiv in range(1, 6):
        verts = _subdivide_icosahedron(subdiv)
        n = len(verts)
        diff = abs(n - n_target)
        if diff < best_diff:
            best_diff = diff
            best_verts = verts

    if len(best_verts) > n_target:
        rng = np.random.RandomState(42)
        idxs = rng.choice(len(best_verts), n_target, replace=False)
        return best_verts[idxs]
    elif len(best_verts) < n_target:
        remaining = n_target - len(best_verts)
        rng = np.random.RandomState(42)
        extra = _random_uniform_sphere(remaining, rng)
        return np.vstack([best_verts, extra])
    else:
        return best_verts


def _md_init(n, steps=2000, dt=0.01, cooling=0.99, seed=42):
    """
    Initialize points using simple Molecular Dynamics with repulsive forces.
    """
    rng = np.random.RandomState(seed)
    points = _random_uniform_sphere(n, rng)
    velocities = np.zeros_like(points)

    for step in range(steps):
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists_sq = np.sum(diff * diff, axis=-1)
        np.fill_diagonal(dists_sq, 1.0)

        inv_dists_cubed = 1.0 / (dists_sq * np.sqrt(dists_sq))
        np.fill_diagonal(inv_dists_cubed, 0.0)

        forces = np.sum(diff * inv_dists_cubed[:, :, np.newaxis], axis=1)

        velocities = velocities * cooling + forces * dt
        points = points + velocities * dt

        points = _project_to_sphere(points)
        radial_vel = np.sum(velocities * points, axis=1, keepdims=True)
        velocities = velocities - radial_vel * points

    return points


def _adam_optimize(pts0, max_iter=4000, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8,
                   deadline=None, check_interval=25):
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


def _lbfgs_polish(pts, deadline, maxiter=600, tol=1e-14):
    """L-BFGS-B polish using flat Cartesian coords with projection inside."""
    def fun(x):
        p = _project_to_sphere(x.reshape(N, 3))
        e, g = _energy_and_grad(p)
        return e, g.flatten()

    x0 = pts.flatten()
    try:
        res = minimize(fun, x0, method='L-BFGS-B', jac=True,
                       options={'maxiter': maxiter, 'maxfun': maxiter * 40,
                                'ftol': tol, 'gtol': tol, 'maxls': 50,
                                'iprint': -1})
        p_res = _project_to_sphere(res.x.reshape(N, 3))
        e_check = _log_energy(p_res)
        e_orig = _log_energy(pts)
        if e_check < e_orig:
            return p_res, e_check
    except Exception:
        pass
    return pts, _log_energy(pts)


def _local_minimize(points, deadline, maxiter=4000):
    """Run Adam then L-BFGS-B to find local minimum."""
    # Phase A: Adam optimization
    p_adam, e_adam = _adam_optimize(points, max_iter=maxiter, lr=0.01, deadline=deadline)

    # Phase B: L-BFGS-B polish on the Adam result
    p_final, e_final = _lbfgs_polish(p_adam, deadline, maxiter=300)

    # Return the best of the two
    if e_final <= e_adam:
        return p_final, e_final
    return p_adam, e_adam


def _basin_hopping(init_points, deadline, rng, max_iter=60):
    """Basin hopping with Adam+L-BFGS local minimization."""
    current_points, current_energy = _local_minimize(init_points, deadline, maxiter=3000)
    best_points = current_points.copy()
    best_energy = current_energy

    T = 0.1
    T_min = 1e-6
    cooling = 0.97

    for it in range(max_iter):
        if time.time() > deadline - 2.0:
            break
        if T < T_min:
            break

        n_perturb = rng.randint(2, 10)
        idxs = rng.choice(N, n_perturb, replace=False)

        new_points = current_points.copy()
        for idx in idxs:
            rand_vec = rng.randn(3)
            tangent = rand_vec - np.dot(rand_vec, new_points[idx]) * new_points[idx]
            norm_t = np.linalg.norm(tangent)
            if norm_t > 1e-15:
                tangent /= norm_t
            else:
                continue
            step_size = T * rng.uniform(0.5, 1.5)
            new_pt = new_points[idx] + step_size * tangent
            new_points[idx] = new_pt / np.linalg.norm(new_pt)

        new_points, new_energy = _local_minimize(new_points, deadline, maxiter=3000)

        delta_e = new_energy - current_energy
        if delta_e < 0 or (T > 0 and rng.rand() < math.exp(-delta_e / T)):
            current_points = new_points
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_points = current_points.copy()

        T *= cooling

    return best_points, best_energy


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    rng = np.random.RandomState(42)

    # Generate diverse initializations
    inits = []

    # 1. Geodesic dome (primary - best topological structure)
    try:
        geodesic_pts = _geodesic_dome_init(N, subdivisions=3)
        inits.append(geodesic_pts)
    except Exception:
        pass

    # 2. Fibonacci sphere
    inits.append(_fibonacci_sphere(N))

    # 3. Random points (more seeds for diversity)
    for seed in range(5):
        r = np.random.RandomState(seed)
        inits.append(_random_uniform_sphere(N, r))

    # 4. MD initialized points
    for seed in range(3):
        md_pts = _md_init(N, steps=1000, seed=seed)
        inits.append(md_pts)

    # Best so far
    best_points = inits[0].copy()
    best_energy = _log_energy(best_points)

    # Phase 1: Local minimize all inits (Adam + L-BFGS)
    local_min_results = []
    for init in inits:
        if time.time() > deadline - 15.0:
            break
        p_res, e_res = _local_minimize(init, deadline, maxiter=4000)
        local_min_results.append((e_res, p_res))
        if e_res < best_energy:
            best_energy = e_res
            best_points = p_res.copy()

    # Sort local minima by energy
    local_min_results.sort(key=lambda x: x[0])

    # Phase 2: Basin hopping from the best candidates
    num_bh_runs = min(4, len(local_min_results))
    for i in range(num_bh_runs):
        if time.time() > deadline - 10.0:
            break
        bh_rng = np.random.RandomState(i * 100 + 123)
        bh_points, bh_energy = _basin_hopping(local_min_results[i][1], deadline, bh_rng, max_iter=50)
        if bh_energy < best_energy:
            best_energy = bh_energy
            best_points = bh_points.copy()

    # Phase 3: Final high-precision polish
    if time.time() < deadline - 2.0:
        # Adam with very small learning rate for fine adjustment
        p_fine, e_fine = _adam_optimize(best_points, max_iter=2000, lr=0.003, deadline=deadline)
        if e_fine < best_energy:
            best_energy = e_fine
            best_points = p_fine.copy()

        # Final L-BFGS-B with tightest tolerance
        p_final, e_final = _lbfgs_polish(best_points, deadline, maxiter=1000, tol=1e-16)
        if e_final < best_energy:
            best_energy = e_final
            best_points = p_final.copy()

    best_points = _project_to_sphere(best_points)

    # Safety checks
    if not np.all(np.isfinite(best_points)):
        return _fibonacci_sphere(N)

    diff = best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    iu = np.triu_indices(N, k=1)
    if np.min(dists[iu]) < 1e-9:
        return _fibonacci_sphere(N)

    return best_points
