"""Rewritten program for logarithmic energy minimization of 150 points on sphere."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize


def _log_energy(points):
    """Compute log(1/d) energy for points on unit sphere using math.fsum."""
    n = len(points)
    iu = np.triu_indices(n, k=1)
    diff = points[iu[0]] - points[iu[1]]
    dists_sq = np.sum(diff * diff, axis=-1)
    dists = np.sqrt(dists_sq)
    log_dists = np.log(dists)
    return -math.fsum(log_dists.tolist())


def _energy_and_grad(x):
    """Energy and gradient for L-BFGS-B. x is flat array of shape (N*3,)."""
    points = x.reshape(N, 3)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    points = points / norms

    n = len(points)
    iu = np.triu_indices(n, k=1)
    diff = points[iu[0]] - points[iu[1]]
    dists_sq = np.sum(diff * diff, axis=-1)

    weights = 1.0 / dists_sq

    grads = np.zeros((n, 3))
    np.add.at(grads, iu[0], weights[:, np.newaxis] * diff)
    np.add.at(grads, iu[1], -weights[:, np.newaxis] * diff)

    dot_products = np.sum(grads * points, axis=1, keepdims=True)
    grads_tangent = grads - dot_products * points

    energy = -math.fsum(np.log(np.sqrt(dists_sq)).tolist())
    return energy, grads_tangent.flatten()


def _project_to_sphere(points):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms


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


def _tangent_vector(point, rng):
    """Generate a random unit tangent vector at point on sphere."""
    rand_vec = rng.randn(3)
    tangent = rand_vec - np.dot(rand_vec, point) * point
    norm = np.linalg.norm(tangent)
    if norm < 1e-15:
        rand_vec = rng.randn(3)
        tangent = rand_vec - np.dot(rand_vec, point) * point
        norm = np.linalg.norm(tangent)
    return tangent / norm


def _geodesic_dome(n_target, rng, seed=42):
    """Generate points using subdivided icosahedron (geodesic dome) approach."""
    # Start with icosahedron vertices
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    # Faces of icosahedron
    faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ]
    
    # Subdivide each face into smaller triangles
    def subdivide_face(v0, v1, v2, level):
        """Recursively subdivide a triangular face."""
        if level == 0:
            return [v0, v1, v2]
        
        # Midpoints
        m01 = (v0 + v1) / 2.0
        m01 = m01 / np.linalg.norm(m01)
        m12 = (v1 + v2) / 2.0
        m12 = m12 / np.linalg.norm(m12)
        m20 = (v2 + v0) / 2.0
        m20 = m20 / np.linalg.norm(m20)
        
        # Recursively subdivide the 4 sub-triangles
        points = []
        points.extend(subdivide_face(v0, m01, m20, level - 1))
        points.extend(subdivide_face(m01, v1, m12, level - 1))
        points.extend(subdivide_face(m20, m12, v2, level - 1))
        points.extend(subdivide_face(m01, m12, m20, level - 1))
        return points
    
    # Determine subdivision level
    # icosahedron has 20 faces
    # After k subdivisions, we have 20 * 4^k faces
    # Number of vertices ≈ 10 * 4^k + 2
    level = 0
    while True:
        n_vertices = 10 * (4 ** level) + 2
        if n_vertices >= n_target:
            break
        level += 1
    
    # Generate all vertices from subdivision
    all_points = []
    for face in faces:
        v0, v1, v2 = verts[face[0]], verts[face[1]], verts[face[2]]
        subdivided = subdivide_face(v0, v1, v2, level)
        all_points.extend(subdivided)
    
    # Remove duplicates
    all_points = np.array(all_points)
    # Use a simple deduplication based on rounding
    unique_points = []
    seen = set()
    for p in all_points:
        key = tuple(np.round(p, 10))
        if key not in seen:
            seen.add(key)
            unique_points.append(p)
    
    unique_points = np.array(unique_points)
    
    # If we have more than n_target, select n_target points
    if len(unique_points) > n_target:
        # Select evenly spaced subset
        indices = np.linspace(0, len(unique_points) - 1, n_target, dtype=int)
        unique_points = unique_points[indices]
    
    # If we have fewer, add random points
    while len(unique_points) < n_target:
        p = rng.randn(3)
        p = p / np.linalg.norm(p)
        unique_points = np.vstack([unique_points, p])
    
    return unique_points[:n_target]


def _local_minimize(points, deadline, maxiter=500):
    """Run L-BFGS-B to find local minimum."""
    x0 = points.flatten()
    try:
        res = minimize(
            _energy_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': maxiter, 'maxfun': maxiter * 20, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 40}
        )
        p_res = _project_to_sphere(res.x.reshape(N, 3))
        e_check = _log_energy(p_res)
        return p_res, e_check
    except Exception:
        return points, _log_energy(points)


def _gradient_descent_sphere(points, deadline, lr=0.1, maxiter=200):
    """Simple gradient descent on the sphere with adaptive step size."""
    current = points.copy()
    current_energy = _log_energy(current)
    best = current.copy()
    best_energy = current_energy

    for it in range(maxiter):
        if time.time() > deadline:
            break
        n = len(current)
        diff = current[:, np.newaxis, :] - current[np.newaxis, :, :]
        dists_sq = np.sum(diff * diff, axis=-1)
        np.fill_diagonal(dists_sq, 1.0)

        weights = np.zeros_like(dists_sq)
        mask = ~np.eye(n, dtype=bool)
        weights[mask] = 1.0 / dists_sq[mask]
        sum_weights = np.sum(weights, axis=1)
        weighted_sum = weights @ current
        grads = weighted_sum - current * sum_weights[:, np.newaxis]
        dot_products = np.sum(grads * current, axis=1, keepdims=True)
        grads_tangent = grads - dot_products * current

        new_points = current - lr * grads_tangent
        new_points = _project_to_sphere(new_points)
        new_energy = _log_energy(new_points)

        if new_energy < current_energy:
            current = new_points
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best = current.copy()
            lr *= 1.05
        else:
            lr *= 0.5
            if lr < 1e-8:
                break

    return best, best_energy


def _basin_hopping_single(init_points, deadline, rng, max_iter=200):
    """Single basin hopping trajectory with adaptive perturbation."""
    current_points, current_energy = _local_minimize(init_points, deadline)
    best_points = current_points.copy()
    best_energy = current_energy

    T = 0.1
    T_min = 1e-7
    cooling = 0.985

    for it in range(max_iter):
        if time.time() > deadline - 3.0:
            break
        if T < T_min:
            break

        if rng.rand() < 0.15:
            n_perturb = rng.randint(10, 21)
            idxs = rng.choice(N, n_perturb, replace=False)
            step_scale = rng.uniform(0.3, 0.8)
        else:
            n_perturb = rng.randint(2, 9)
            idxs = rng.choice(N, n_perturb, replace=False)
            step_scale = rng.uniform(0.05, 0.3)

        new_points = current_points.copy()
        for idx in idxs:
            tangent = _tangent_vector(new_points[idx], rng)
            step_size = T * step_scale
            new_point = new_points[idx] + step_size * tangent
            new_point = new_point / np.linalg.norm(new_point)
            new_points[idx] = new_point

        new_points, new_energy = _local_minimize(new_points, deadline, maxiter=200)

        delta_e = new_energy - current_energy
        if delta_e < 0 or (T > 0 and rng.rand() < math.exp(-delta_e / T)):
            current_points = new_points
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_points = current_points.copy()

        T *= cooling

    return best_points, best_energy


def _newton_polish(points, deadline, maxiter=100):
    """Newton-type refinement on the sphere using Hessian approximation."""
    current = points.copy()
    current_energy = _log_energy(current)
    best = current.copy()
    best_energy = current_energy
    
    # Use a simplified Newton method with line search
    lr = 0.01
    for it in range(maxiter):
        if time.time() > deadline:
            break
        
        n = len(current)
        # Compute gradient
        diff = current[:, np.newaxis, :] - current[np.newaxis, :, :]
        dists_sq = np.sum(diff * diff, axis=-1)
        np.fill_diagonal(dists_sq, 1.0)
        
        weights = np.zeros_like(dists_sq)
        mask = ~np.eye(n, dtype=bool)
        weights[mask] = 1.0 / dists_sq[mask]
        sum_weights = np.sum(weights, axis=1)
        weighted_sum = weights @ current
        grads = weighted_sum - current * sum_weights[:, np.newaxis]
        dot_products = np.sum(grads * current, axis=1, keepdims=True)
        grads_tangent = grads - dot_products * current
        
        # Approximate Hessian using finite differences of gradient
        # Use a simplified approach: scale gradient by estimated curvature
        grad_norms = np.linalg.norm(grads_tangent, axis=1)
        max_grad = np.max(grad_norms) if np.max(grad_norms) > 0 else 1e-15
        
        # Adaptive step size based on gradient magnitude
        step_scale = lr / (max_grad + 1e-10)
        
        new_points = current - step_scale * grads_tangent
        new_points = _project_to_sphere(new_points)
        new_energy = _log_energy(new_points)
        
        if new_energy < current_energy:
            current = new_points
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best = current.copy()
            lr *= 1.1
        else:
            lr *= 0.5
            if lr < 1e-12:
                break
    
    return best, best_energy


def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    rng = np.random.RandomState(12345)

    # Generate diverse initializations
    inits = []

    # Geodesic dome (icosahedron-based)
    dome_points = _geodesic_dome(N, rng, seed=42)
    inits.append(dome_points)
    
    # Fibonacci sphere
    inits.append(_fibonacci_sphere(N))

    # Random uniform points with different seeds
    for seed in range(8):
        r = np.random.RandomState(seed * 7 + 1)
        inits.append(_random_uniform_sphere(N, r))

    # Perturbed Fibonacci
    for k in range(4):
        perturbed = _fibonacci_sphere(N) + rng.randn(N, 3) * 0.2
        inits.append(_project_to_sphere(perturbed))

    # Perturbed geodesic dome
    for k in range(4):
        perturbed = dome_points + rng.randn(N, 3) * 0.15
        inits.append(_project_to_sphere(perturbed))

    # Rotated Fibonacci
    for angle in [0.1, 0.3, 0.5, 0.7]:
        theta = angle
        rot_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)]
        ])
        fib = _fibonacci_sphere(N)
        rotated = fib @ rot_matrix.T
        inits.append(_project_to_sphere(rotated))

    # Best-so-far tracking
    best_points = inits[0].copy()
    best_energy = _log_energy(best_points)

    # Phase 1: Local minimize all initializations
    for idx, init in enumerate(inits):
        if time.time() > deadline - 35.0:
            break
        p_res, e_res = _local_minimize(init, deadline)
        if e_res < best_energy:
            best_energy = e_res
            best_points = p_res.copy()

    # Phase 2: Basin hopping from best initializations
    scored_inits = []
    for idx, init in enumerate(inits):
        e = _log_energy(init)
        scored_inits.append((e, idx, init))
    scored_inits.sort()

    for i in range(min(6, len(scored_inits))):
        if time.time() > deadline - 25.0:
            break
        bh_rng = np.random.RandomState(i * 100 + 500)
        bh_points, bh_energy = _basin_hopping_single(
            scored_inits[i][2], deadline, bh_rng, max_iter=120
        )
        if bh_energy < best_energy:
            best_energy = bh_energy
            best_points = bh_points.copy()

    # Phase 3: Newton polish from best
    if time.time() < deadline - 15.0:
        newton_points, newton_energy = _newton_polish(best_points, deadline, maxiter=80)
        if newton_energy < best_energy:
            best_energy = newton_energy
            best_points = newton_points.copy()

    # Phase 4: Final high-precision local minimize
    if time.time() < deadline - 8.0:
        final_points, final_energy = _local_minimize(best_points, deadline, maxiter=500)
        if final_energy < best_energy:
            best_energy = final_energy
            best_points = final_points.copy()

        if time.time() < deadline - 3.0:
            try:
                x0 = best_points.flatten()
                res = minimize(
                    _energy_and_grad,
                    x0,
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 2000, 'maxfun': 20000, 'ftol': 1e-18, 'gtol': 1e-14, 'maxls': 50}
                )
                p_res = _project_to_sphere(res.x.reshape(N, 3))
                e_check = _log_energy(p_res)
                if e_check < best_energy:
                    best_energy = e_check
                    best_points = p_res.copy()
            except Exception:
                pass

    # Phase 5: Gradient descent polish
    if time.time() < deadline - 2.0:
        gd_points, gd_energy = _gradient_descent_sphere(
            best_points, deadline, lr=0.005, maxiter=150
        )
        if gd_energy < best_energy:
            best_energy = gd_energy
            best_points = gd_points.copy()

    # Final Newton polish
    if time.time() < deadline - 1.0:
        newton2_points, newton2_energy = _newton_polish(best_points, deadline, maxiter=50)
        if newton2_energy < best_energy:
            best_energy = newton2_energy
            best_points = newton2_points.copy()

    best_points = _project_to_sphere(best_points)

    if not np.all(np.isfinite(best_points)):
        return _fibonacci_sphere(N)

    diff = best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    iu = np.triu_indices(N, k=1)
    if np.min(dists[iu]) < 1e-9:
        return _fibonacci_sphere(N)

    return best_points
# EVOLVE-BLOCK-END
