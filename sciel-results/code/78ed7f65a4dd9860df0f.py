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

def _energy_and_grad(x):
    """Energy and gradient for L-BFGS-B. x is flat array of shape (N*3,)."""
    points = x.reshape(N, 3)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    points = points / norms

    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dists_sq, 1.0)
    
    inv_dists_sq = 1.0 / dists_sq
    np.fill_diagonal(inv_dists_sq, 0.0)
    
    sum_inv_dists_sq = np.sum(inv_dists_sq, axis=1)
    weighted_points = inv_dists_sq @ points
    
    grad = weighted_points - points * sum_inv_dists_sq[:, np.newaxis]
    
    dot_products = np.sum(grad * points, axis=1, keepdims=True)
    grad_tangent = grad - dot_products * points
    
    iu = np.triu_indices(n, k=1)
    dists = np.sqrt(np.maximum(dists_sq[iu], 1e-30))
    energy = -math.fsum(np.log(dists).tolist())
    
    return energy, grad_tangent.flatten()

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
    
    return verts, faces

def _geodesic_dome_init(n_target):
    """
    Generate a geodesic dome with approximately n_target points.
    """
    best_verts = None
    best_diff = float('inf')
    
    for subdiv in range(1, 6):
        verts, faces = _subdivide_icosahedron(subdiv)
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

def _laplacian_smooth(points, steps=50, alpha=0.5):
    """
    Apply Laplacian smoothing to points on the sphere.
    """
    for _ in range(steps):
        k = 10
        smoothed = np.zeros_like(points)
        for i in range(N):
            dists = np.linalg.norm(points - points[i], axis=1)
            dists[i] = np.inf
            nn_idx = np.argsort(dists)[:k]
            smoothed[i] = np.mean(points[nn_idx], axis=0)
            norm = np.linalg.norm(smoothed[i])
            if norm > 1e-15:
                smoothed[i] /= norm
        points = (1 - alpha) * points + alpha * smoothed
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-15)
        points = points / norms
    return points

def _local_minimize(points, deadline, maxiter=500, tol=1e-14):
    """Run L-BFGS-B to find local minimum."""
    points = _laplacian_smooth(points, steps=20, alpha=0.3)
    
    x0 = points.flatten()
    try:
        res = minimize(
            _energy_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': maxiter, 'maxfun': maxiter * 20, 'ftol': tol, 'gtol': tol, 'maxls': 40}
        )
        p_res = _project_to_sphere(res.x.reshape(N, 3))
        e_check = _log_energy(p_res)
        return p_res, e_check
    except Exception:
        return points, _log_energy(points)

def _basin_hopping(init_points, deadline, rng, max_iter=100):
    """Basin hopping with L-BFGS-B local minimization."""
    current_points, current_energy = _local_minimize(init_points, deadline, maxiter=200)
    best_points = current_points.copy()
    best_energy = current_energy
    
    T = 0.15
    T_min = 1e-6
    cooling = 0.95
    
    for it in range(max_iter):
        if time.time() > deadline - 2.0:
            break
        if T < T_min:
            break
            
        n_perturb = rng.randint(3, 12)
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
            step_size = T * rng.uniform(0.5, 2.0)
            new_pt = new_points[idx] + step_size * tangent
            new_points[idx] = new_pt / np.linalg.norm(new_pt)
        
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

def _so2_rotate(pts, rng):
    """Apply random SO(3) rotation to points."""
    u = rng.standard_normal(3)
    v = rng.standard_normal(3)
    w = rng.standard_normal(3)
    u = u / np.linalg.norm(u)
    v = v / np.linalg.norm(v)
    w = np.cross(u, v)
    w = w / np.linalg.norm(w)
    R = np.column_stack([u, v, w])
    return pts @ R.T

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    rng = np.random.RandomState(42)
    
    inits = []
    
    try:
        geodesic_pts = _geodesic_dome_init(N)
        inits.append(geodesic_pts)
    except Exception:
        pass
    
    inits.append(_fibonacci_sphere(N))
    
    for seed in range(3):
        r = np.random.RandomState(seed)
        inits.append(_random_uniform_sphere(N, r))
        
    for seed in range(2):
        md_pts = _md_init(N, steps=1000, seed=seed)
        inits.append(md_pts)

    best_points = inits[0].copy()
    best_energy = _log_energy(best_points)
    
    local_min_results = []
    for init in inits:
        if time.time() > deadline - 10.0:
            break
        p_res, e_res = _local_minimize(init, deadline, maxiter=500)
        local_min_results.append((e_res, p_res))
        if e_res < best_energy:
            best_energy = e_res
            best_points = p_res.copy()

    local_min_results.sort(key=lambda x: x[0])
    
    num_bh_runs = min(4, len(local_min_results))
    for i in range(num_bh_runs):
        if time.time() > deadline - 15.0:
            break
        bh_rng = np.random.RandomState(i * 100 + 123)
        bh_points, bh_energy = _basin_hopping(local_min_results[i][1], deadline, bh_rng, max_iter=60)
        if bh_energy < best_energy:
            best_energy = bh_energy
            best_points = bh_points.copy()

    # Add SO(3) rotations of the best point as new starting points for basin hopping
    if time.time() < deadline - 20.0:
        for rot_seed in range(3):
            if time.time() > deadline - 15.0:
                break
            rot_rng = np.random.RandomState(rot_seed * 777 + 11)
            rotated = _so2_rotate(best_points, rot_rng)
            # Local minimize the rotated version
            p_res, e_res = _local_minimize(rotated, deadline, maxiter=300)
            if e_res < best_energy:
                best_energy = e_res
                best_points = p_res.copy()
            # Basin hopping from the rotated version
            bh_rng = np.random.RandomState(rot_seed * 333 + 42)
            bh_points, bh_energy = _basin_hopping(p_res, deadline, bh_rng, max_iter=40)
            if bh_energy < best_energy:
                best_energy = bh_energy
                best_points = bh_points.copy()

    if time.time() < deadline - 2.0:
        final_points, final_energy = _local_minimize(best_points, deadline, maxiter=3000, tol=1e-16)
        if final_energy < best_energy:
            best_energy = final_energy
            best_points = final_points.copy()

    best_points = _project_to_sphere(best_points)

    if not np.all(np.isfinite(best_points)):
        return _fibonacci_sphere(N)

    diff = best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    iu = np.triu_indices(N, k=1)
    if np.min(dists[iu]) < 1e-9:
        return _fibonacci_sphere(N)

    return best_points
