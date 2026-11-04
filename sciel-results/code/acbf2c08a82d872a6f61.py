"""Combined program for logarithmic energy minimization of 150 points on sphere."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize


def _log_energy(points):
    """Compute log(1/d) energy for points on unit sphere using math.fsum."""
    n = len(points)
    if n < 2:
        return 0.0
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dists_sq, 1.0)
    iu = np.triu_indices(n, k=1)
    log_dists = 0.5 * np.log(dists_sq[iu])
    return -math.fsum(log_dists.tolist())


def _fibonacci_sphere(n):
    """Generate Fibonacci sphere points."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


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


def _energy_and_grad(x):
    """Energy and gradient for L-BFGS-B. x is flat array of shape (N*3,)."""
    points = x.reshape(N, 3)
    points = _project_to_sphere(points)
    
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(dists_sq, 1.0)
    
    inv_dists_sq = np.zeros_like(dists_sq)
    mask = ~np.eye(n, dtype=bool)
    inv_dists_sq[mask] = 1.0 / dists_sq[mask]
    
    sum_inv = np.sum(inv_dists_sq, axis=1)
    weighted_sum = inv_dists_sq @ points
    
    grads = weighted_sum - points * sum_inv[:, np.newaxis]
    
    dot_products = np.einsum('ij,ij->i', grads, points)
    grads_tangent = grads - dot_products[:, np.newaxis] * points
    
    iu = np.triu_indices(n, k=1)
    energy = -0.5 * np.sum(np.log(dists_sq[iu]))
    
    return energy, grads_tangent.flatten()


def _greedy_init(n, seed=42):
    """
    Generate initial points using a greedy algorithm.
    Place points one by one to minimize the incremental energy.
    """
    rng = np.random.RandomState(seed)
    points = []
    
    p = rng.randn(3)
    p = p / np.linalg.norm(p)
    points.append(p)
    
    for i in range(n - 1):
        candidates = rng.randn(1000, 3)
        candidates = candidates / np.linalg.norm(candidates, axis=1, keepdims=True)
        
        best_candidate = None
        best_incremental_energy = float('inf')
        
        for c in candidates:
            dists = np.linalg.norm(points - c, axis=1)
            dists = np.maximum(dists, 1e-15)
            cost = np.sum(np.log(1.0 / dists))
            
            if cost < best_incremental_energy:
                best_incremental_energy = cost
                best_candidate = c
        
        points.append(best_candidate)
    
    return np.array(points)


def _local_minimize(points, deadline, maxiter=3000):
    """Run L-BFGS-B to find local minimum."""
    x0 = points.flatten()
    try:
        res = minimize(
            _energy_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': maxiter, 'maxfun': maxiter * 10, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 100}
        )
        p_res = _project_to_sphere(res.x.reshape(N, 3))
        e_check = _log_energy(p_res)
        return p_res, e_check
    except Exception:
        return points, _log_energy(points)


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


def _riemannian_line_search(points, direction, energy_func, max_iter=50):
    """
    Riemannian line search with Armijo condition.
    direction is a tangent vector at each point.
    Returns the step size alpha.
    """
    c1 = 1e-4
    rho = 0.5
    alpha = 1.0
    f0 = energy_func(points)
    g = np.einsum('ij,ij->i', direction, points)  # should be ~0 for tangent
    
    for _ in range(max_iter):
        # Move along geodesic (exp map)
        new_points = points.copy()
        for i in range(N):
            d = direction[i]
            dn = np.linalg.norm(d)
            if dn < 1e-15:
                continue
            d_unit = d / dn
            step = alpha * dn
            # Spherical interpolation: cos(step)*p + sin(step)*d_unit
            new_points[i] = np.cos(step) * points[i] + np.sin(step) * d_unit
            new_points[i] = new_points[i] / np.linalg.norm(new_points[i])
        
        f1 = energy_func(new_points)
        # Armijo condition
        if f1 <= f0 + c1 * alpha * np.sum(direction * (energy_func(points + alpha * direction) - f0)):
            return alpha
        alpha *= rho
    
    return alpha


def _basin_hopping_single(init_points, deadline, rng, max_iter=40):
    """
    Basin hopping with Metropolis acceptance:
    1. Local minimize
    2. Perturb random subset with adaptive step size
    3. Local minimize
    4. Accept if better (or Metropolis)
    """
    current_points, current_energy = _local_minimize(init_points, deadline)
    best_points = current_points.copy()
    best_energy = current_energy

    T = 0.05
    T_min = 1e-12
    cooling = 0.95
    base_step = 0.1

    for it in range(max_iter):
        if time.time() > deadline - 5.0:
            break
        if T < T_min:
            break

        progress = it / max_iter
        step_scale = base_step * (1.0 - 0.7 * progress)
        
        n_perturb = max(1, int(rng.randint(1, 4) * (1.0 + 0.5 * progress)))
        idxs = rng.choice(N, n_perturb, replace=False)

        new_points = current_points.copy()
        for idx in idxs:
            tangent = _tangent_vector(new_points[idx], rng)
            step_size = step_scale * (0.5 + 0.5 * rng.rand())
            new_point = new_points[idx] + step_size * tangent
            new_point = new_point / np.linalg.norm(new_point)
            new_points[idx] = new_point

        new_points, new_energy = _local_minimize(new_points, deadline, maxiter=1500)

        delta_e = new_energy - current_energy
        if delta_e < 0 or (T > 0 and rng.rand() < math.exp(-delta_e / T)):
            current_points = new_points
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_points = current_points.copy()

        T *= cooling

    return best_points, best_energy


def _parallel_tempering_step(replicas, energies, deadline, rng, maxiter=2000):
    """
    Parallel tempering step:
    - Each replica has points and energy
    - Try to swap adjacent replicas based on Metropolis criterion
    - Returns updated replicas and energies
    """
    n_rep = len(replicas)
    temps = [0.01 * (2.0 ** i) for i in range(n_rep)]
    
    for i in range(n_rep - 1):
        if time.time() > deadline - 5.0:
            break
        
        T1, T2 = temps[i], temps[i+1]
        
        delta_beta = (energies[i] / T2 + energies[i+1] / T1 - energies[i] / T1 - energies[i+1] / T2)
        
        if delta_beta <= 0 or rng.rand() < math.exp(-delta_beta):
            replicas[i], replicas[i+1] = replicas[i+1].copy(), replicas[i].copy()
            energies[i], energies[i+1] = energies[i+1], energies[i]
    
    return replicas, energies


def _polish_high_precision(points, deadline):
    """
    Multi-stage high-precision polish using progressively tighter tolerances.
    """
    best = points.copy()
    best_e = _log_energy(best)
    
    # Stage 1: standard polish
    if time.time() < deadline - 3.0:
        try:
            res = minimize(
                _energy_and_grad,
                best.flatten(),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 5000, 'maxfun': 50000, 'ftol': 1e-15, 'gtol': 1e-14, 'maxls': 100}
            )
            p_res = _project_to_sphere(res.x.reshape(N, 3))
            e = _log_energy(p_res)
            if e < best_e:
                best = p_res
                best_e = e
        except Exception:
            pass
    
    # Stage 2: tighter
    if time.time() < deadline - 2.0:
        try:
            res = minimize(
                _energy_and_grad,
                best.flatten(),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 3000, 'maxfun': 30000, 'ftol': 1e-16, 'gtol': 1e-15, 'maxls': 50}
            )
            p_res = _project_to_sphere(res.x.reshape(N, 3))
            e = _log_energy(p_res)
            if e < best_e:
                best = p_res
                best_e = e
        except Exception:
            pass
    
    # Stage 3: ultra-precise
    if time.time() < deadline - 1.0:
        try:
            res = minimize(
                _energy_and_grad,
                best.flatten(),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 2000, 'maxfun': 20000, 'ftol': 1e-16, 'gtol': 1e-16, 'maxls': 30}
            )
            p_res = _project_to_sphere(res.x.reshape(N, 3))
            e = _log_energy(p_res)
            if e < best_e:
                best = p_res
                best_e = e
        except Exception:
            pass
    
    return best


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    rng = np.random.RandomState(42)
    
    best_points = None
    best_energy = float('inf')
    
    # Generate diverse initializations
    inits = [
        _fibonacci_sphere(N),
        _random_uniform_sphere(N, seed=1),
        _random_uniform_sphere(N, seed=2),
        _random_uniform_sphere(N, seed=3),
        _random_uniform_sphere(N, seed=4),
        _random_uniform_sphere(N, seed=5),
        _random_uniform_sphere(N, seed=6),
        _random_uniform_sphere(N, seed=7),
        _random_uniform_sphere(N, seed=8),
        _greedy_init(N, seed=10),
        _greedy_init(N, seed=20),
    ]
    
    # First pass: local minimize each initialization
    for idx, init in enumerate(inits):
        if time.time() > deadline - 20.0:
            break
        
        p_res, e_res = _local_minimize(init, deadline, maxiter=2500)
        
        if e_res < best_energy:
            best_energy = e_res
            best_points = p_res.copy()
    
    # Second pass: basin hopping from the best few initializations
    candidates = []
    for idx, init in enumerate(inits[:5]):
        if time.time() > deadline - 25.0:
            break
        p_res, e_res = _local_minimize(init, deadline, maxiter=1500)
        candidates.append((p_res, e_res))
    
    candidates.sort(key=lambda x: x[1])
    
    for ci in range(min(2, len(candidates))):
        if time.time() < deadline - 15.0:
            bh_points, bh_energy = _basin_hopping_single(candidates[ci][0], deadline, rng, max_iter=25)
            if bh_energy < best_energy:
                best_energy = bh_energy
                best_points = bh_points.copy()
    
    # Third pass: parallel tempering with multiple replicas
    if time.time() < deadline - 20.0 and best_points is not None:
        n_rep = 5
        replicas = []
        energies = []
        
        for r in range(n_rep):
            if r == 0:
                rep = best_points.copy()
            else:
                rep = best_points.copy()
                n_pert = rng.randint(2, 6 + r)
                idxs = rng.choice(N, n_pert, replace=False)
                for idx in idxs:
                    tangent = _tangent_vector(rep[idx], rng)
                    step = 0.01 * (1 + 0.3 * r)
                    rep[idx] = rep[idx] + step * tangent
                    rep[idx] = rep[idx] / np.linalg.norm(rep[idx])
                rep = _project_to_sphere(rep)
            
            e = _log_energy(rep)
            replicas.append(rep)
            energies.append(e)
        
        n_pt_iters = 30
        for it in range(n_pt_iters):
            if time.time() > deadline - 8.0:
                break
            
            for r in range(n_rep):
                if time.time() > deadline - 8.0:
                    break
                replicas[r], energies[r] = _local_minimize(replicas[r], deadline, maxiter=800)
            
            replicas, energies = _parallel_tempering_step(replicas, energies, deadline, rng)
            
            for r in range(n_rep):
                if energies[r] < best_energy:
                    best_energy = energies[r]
                    best_points = replicas[r].copy()
        
        # Final Polish on the best replica from PT
        if best_points is not None and time.time() < deadline - 5.0:
            best_points = _polish_high_precision(best_points, deadline)
            best_energy = _log_energy(best_points)
    
    # Final high-precision polish
    if best_points is not None:
        best_points = _polish_high_precision(best_points, deadline)
        best_energy = _log_energy(best_points)
    
    if best_points is None:
        best_points = _fibonacci_sphere(N)
    
    best_points = _project_to_sphere(best_points)
    
    if not np.all(np.isfinite(best_points)):
        return _fibonacci_sphere(N)
    
    # Check for points too close
    diff = best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :]
    dists_sq = np.einsum('ijk,ijk->ij', diff, diff)
    iu = np.triu_indices(N, k=1)
    if np.min(dists_sq[iu]) < (1e-9) ** 2:
        return _fibonacci_sphere(N)
    
    return best_points
# EVOLVE-BLOCK-END
