"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize


def _log_energy(points):
    """Compute log(1/d) energy for points on unit sphere."""
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, 1.0)
    iu = np.triu_indices(n, k=1)
    log_dists = np.log(dists[iu])
    return -math.fsum(log_dists)


def _energy_and_grad(x_flat):
    """Energy and gradient for L-BFGS-B. x_flat is shape (N*3,)."""
    n = N
    points = x_flat.reshape(n, 3)
    
    # Compute pairwise squared distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dists_sq, 1.0)
    
    # Energy = -sum_{i<j} log(d_ij) = -0.5 * sum_{i!=j} log(d_ij)
    # Using dists_sq to avoid sqrt: log(d) = 0.5 * log(d^2)
    log_dists_sq = np.log(dists_sq)
    iu = np.triu_indices(n, k=1)
    energy = -0.5 * np.sum(log_dists_sq[iu])
    
    # Gradient of E = -sum_{i<j} log(d_ij) w.r.t. x_i:
    # dE/dx_i = -sum_{j!=i} (x_i - x_j) / d_ij^2
    # So gradient = -[points[i]*sum_j w_ij - sum_j w_ij*points[j]]
    weights = np.zeros_like(dists_sq)
    mask = ~np.eye(n, dtype=bool)
    weights[mask] = 1.0 / dists_sq[mask]
    
    sum_weights = np.sum(weights, axis=1)
    weighted_sum = weights @ points
    grads = -(points * sum_weights[:, np.newaxis] - weighted_sum)
    
    # Project to tangent space (remove radial component)
    dot_products = np.sum(grads * points, axis=1, keepdims=True)
    grads_tangent = grads - dot_products * points
    
    grad_flat = grads_tangent.flatten()
    
    return energy, grad_flat


def _optimize_lbfgsb(points, maxiter=2000, tol=1e-10):
    """Optimize using L-BFGS-B on the sphere."""
    x_flat = points.flatten()
    
    result = minimize(
        _energy_and_grad,
        x_flat,
        jac=True,
        method='L-BFGS-B',
        options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': tol, 'maxfun': 100000}
    )
    
    opt_points = _project_to_sphere(result.x.reshape(N, 3))
    energy = _log_energy(opt_points)
    return opt_points, energy


def _fibonacci_sphere(n):
    """Generate Fibonacci sphere points."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _geodesic_dome(n):
    """Generate points via subdivided icosahedron (geodesic dome)."""
    # Icosahedron vertices
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    
    # Icosahedron faces
    faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ]
    
    # Subdivide faces to get approximately n points
    # For n=150, we need subdivision such that total vertices ≈ 150
    # After k subdivisions: V = 10 * 4^k + 2
    # k=0: 12, k=1: 42, k=2: 162
    # 162 is close to 150, so use k=2 and then select 150
    
    def subdivide_face(v1, v2, v3, depth):
        """Recursively subdivide a triangular face."""
        if depth == 0:
            return [v1, v2, v3]
        
        m12 = (v1 + v2) / 2.0
        m23 = (v2 + v3) / 2.0
        m31 = (v3 + v1) / 2.0
        m12 /= np.linalg.norm(m12)
        m23 /= np.linalg.norm(m23)
        m31 /= np.linalg.norm(m31)
        
        verts = []
        verts.extend(subdivide_face(v1, m12, m31, depth - 1))
        verts.extend(subdivide_face(m12, v2, m23, depth - 1))
        verts.extend(subdivide_face(m31, m23, v3, depth - 1))
        verts.extend(subdivide_face(m12, m23, m31, depth - 1))
        return verts
    
    all_verts = []
    for face in faces:
        v1, v2, v3 = verts[face[0]], verts[face[1]], verts[face[2]]
        all_verts.extend(subdivide_face(v1, v2, v3, 2))
    
    # Deduplicate
    unique_verts = []
    seen = set()
    for v in all_verts:
        key = tuple(np.round(v, 10))
        if key not in seen:
            seen.add(key)
            unique_verts.append(v)
    
    unique_verts = np.array(unique_verts)
    unique_verts /= np.linalg.norm(unique_verts, axis=1, keepdims=True)
    
    # If we have more than n, select n points (first n after sorting by angle)
    if len(unique_verts) >= n:
        return unique_verts[:n]
    else:
        # Pad with Fibonacci points
        fib = _fibonacci_sphere(n - len(unique_verts))
        return np.vstack([unique_verts, fib])


def _random_uniform_sphere(n, seed=42):
    """Generate random uniform points on sphere."""
    rng = np.random.RandomState(seed)
    points = rng.randn(n, 3)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    return points / norms


def _project_to_sphere(points):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms


def _basin_hopping_single(points, maxiter=50, T=0.05, seed=42):
    """Basin hopping: random perturbations followed by local minimization."""
    rng = np.random.RandomState(seed)
    current = points.copy()
    current_energy = _log_energy(current)
    best = current.copy()
    best_energy = current_energy
    
    for it in range(maxiter):
        # Random perturbation
        n_perturb = rng.randint(1, 5)
        idx = rng.choice(N, size=n_perturb, replace=False)
        perturbation = rng.randn(N, 3) * T
        perturbation[idx] *= 2.0  # Larger perturbation for selected points
        
        new_points = _project_to_sphere(current + perturbation)
        
        # Local minimize from perturbed point
        opt_points, opt_energy = _optimize_lbfgsb(new_points, maxiter=1000, tol=1e-9)
        
        delta = opt_energy - current_energy
        if delta < 0 or rng.rand() < np.exp(-delta / max(T, 1e-15)):
            current = opt_points
            current_energy = opt_energy
            if opt_energy < best_energy:
                best_energy = opt_energy
                best = opt_points.copy()
        
        # Gradually decrease temperature
        T *= 0.95
    
    return best, best_energy


def _multi_point_move(points, n_moves=5, step_size=0.05, seed=42):
    """Try moving small groups of points to find better configurations."""
    rng = np.random.RandomState(seed)
    current = points.copy()
    current_energy = _log_energy(current)
    best = current.copy()
    best_energy = current_energy
    
    for _ in range(n_moves * 10):
        # Select a small group of points to move
        group_size = rng.randint(1, 4)
        idx = rng.choice(N, size=group_size, replace=False)
        
        # Generate new positions for these points
        new_positions = rng.randn(group_size, 3)
        new_positions /= np.linalg.norm(new_positions, axis=1, keepdims=True)
        
        trial = current.copy()
        trial[idx] = new_positions
        trial = _project_to_sphere(trial)
        
        trial_energy = _log_energy(trial)
        
        if trial_energy < current_energy:
            current = trial
            current_energy = trial_energy
            if trial_energy < best_energy:
                best_energy = trial_energy
                best = trial.copy()
    
    return best, best_energy


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_points = None
    best_energy = float('inf')
    
    # Strategy 1: Geodesic dome + L-BFGS-B
    if time.time() < deadline - 60:
        points = _geodesic_dome(N)
        opt_points, energy = _optimize_lbfgsb(points, maxiter=3000, tol=1e-10)
        if energy < best_energy:
            best_energy = energy
            best_points = opt_points.copy()
    
    # Strategy 2: Fibonacci sphere + L-BFGS-B
    if time.time() < deadline - 50:
        points = _fibonacci_sphere(N)
        opt_points, energy = _optimize_lbfgsb(points, maxiter=3000, tol=1e-10)
        if energy < best_energy:
            best_energy = energy
            best_points = opt_points.copy()
    
    # Strategy 3: Multiple random starts with L-BFGS-B
    n_random_starts = 6
    for seed in range(n_random_starts):
        if time.time() > deadline - 40:
            break
        
        points = _random_uniform_sphere(N, seed=seed * 100 + 42)
        opt_points, energy = _optimize_lbfgsb(points, maxiter=2000, tol=1e-9)
        if energy < best_energy:
            best_energy = energy
            best_points = opt_points.copy()
    
    # Strategy 4: Basin hopping from best solution
    if best_points is not None and time.time() < deadline - 30:
        bh_points, bh_energy = _basin_hopping_single(
            best_points, maxiter=30, T=0.03, seed=123
        )
        if bh_energy < best_energy:
            best_energy = bh_energy
            best_points = bh_points.copy()
        
        # Polish with L-BFGS-B after basin hopping
        if time.time() < deadline - 15:
            opt_points, energy = _optimize_lbfgsb(best_points, maxiter=3000, tol=1e-11)
            if energy < best_energy:
                best_energy = energy
                best_points = opt_points.copy()
    
    # Strategy 5: Multi-point move exploration
    if best_points is not None and time.time() < deadline - 12:
        mp_points, mp_energy = _multi_point_move(
            best_points, n_moves=10, step_size=0.05, seed=456
        )
        if mp_energy < best_energy:
            best_energy = mp_energy
            best_points = mp_points.copy()
        
        # Polish after multi-point move
        if time.time() < deadline - 8:
            opt_points, energy = _optimize_lbfgsb(best_points, maxiter=3000, tol=1e-11)
            if energy < best_energy:
                best_energy = energy
                best_points = opt_points.copy()
    
    # Strategy 6: Second basin hopping round with different seed
    if best_points is not None and time.time() < deadline - 10:
        bh2_points, bh2_energy = _basin_hopping_single(
            best_points, maxiter=20, T=0.02, seed=789
        )
        if bh2_energy < best_energy:
            best_energy = bh2_energy
            best_points = bh2_points.copy()
    
    # Final high-precision polish
    if best_points is not None and time.time() < deadline - 3:
        opt_points, energy = _optimize_lbfgsb(best_points, maxiter=5000, tol=1e-12)
        if energy < best_energy:
            best_energy = energy
            best_points = opt_points.copy()
    
    # Ensure valid output
    if best_points is None or not np.all(np.isfinite(best_points)):
        best_points = _fibonacci_sphere(N)
    
    best_points = _project_to_sphere(best_points)
    
    return best_points
# EVOLVE-BLOCK-END
