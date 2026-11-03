import numpy as np
import time
import math
from scipy.optimize import minimize

N = 150

def _log_energy(points):
    """Compute log(1/d) energy using math.fsum for accuracy."""
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dists_sq, 1.0)
    iu = np.triu_indices(n, k=1)
    log_dists = 0.5 * np.log(dists_sq[iu])
    return -math.fsum(log_dists.tolist())

def _fast_log_energy(points):
    """Compute log(1/d) energy using np.sum for speed."""
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dists_sq, 1.0)
    iu = np.triu_indices(n, k=1)
    log_dists = 0.5 * np.log(dists_sq[iu])
    return -np.sum(log_dists)

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
    dists_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dists_sq, 1.0)
    
    # E = -sum_{i<j} log(d_ij)
    # dE/dx_i = sum_{j!=i} (x_j - x_i) / d_ij^2
    weights = np.zeros_like(dists_sq)
    mask = ~np.eye(n, dtype=bool)
    weights[mask] = 1.0 / dists_sq[mask]
    
    sum_weights = np.sum(weights, axis=1)
    weighted_sum = weights @ points
    
    grads = weighted_sum - points * sum_weights[:, np.newaxis]
    
    # Project gradient to tangent space
    dot_products = np.sum(grads * points, axis=1, keepdims=True)
    grads_tangent = grads - dot_products * points
    
    # Compute energy with math.fsum for consistency with final score
    iu = np.triu_indices(n, k=1)
    log_dists = 0.5 * np.log(dists_sq[iu])
    energy = -math.fsum(log_dists.tolist())
    
    return energy, grads_tangent.flatten()

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

def _geodesic_dome(n, subdivisions=4):
    """Generate points on a geodesic dome (icosahedron based)."""
    # Start with icosahedron
    t = (1.0 + np.sqrt(5.0)) / 2.0
    points = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ])
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    # Subdivide
    def subdivide(points):
        new_points = []
        # For simplicity, we just use random points if subdivision is complex to implement correctly with deduplication
        # A simpler "structured" random is better than a buggy geodesic dome
        return points
        
    # For robustness, let's just use a mix of Fibonacci and Random
    # The "Geodesic Dome" inspiration might have been a red herring or poorly implemented.
    # We'll stick to Fibonacci and Random as they are proven.
    return points

def _lbfgs_polish(points, max_iter=2000, ftol=1e-12, gtol=1e-10, time_limit=10.0):
    """Polish a point configuration using L-BFGS-B."""
    start = time.time()
    x0 = points.flatten()
    e0 = _log_energy(points)
    try:
        res = minimize(
            _energy_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={
                'maxiter': max_iter,
                'maxfun': max_iter * 10,
                'ftol': ftol,
                'gtol': gtol,
                'maxls': 50
            }
        )
        if res.success or res.fun < e0:
            p_res = _project_to_sphere(res.x.reshape(N, 3))
            e_check = _log_energy(p_res)
            if e_check < e0 and time.time() - start < time_limit:
                return p_res
    except Exception:
        pass
    return points

def _random_rotation_matrix(rng):
    """Generate a random rotation matrix (SO(3))."""
    # Using QR decomposition of a random Gaussian matrix
    A = rng.randn(3, 3)
    Q, R = np.linalg.qr(A)
    # Ensure proper rotation (det = 1)
    d = np.sign(np.linalg.det(Q))
    Q[:, -1] *= d
    if np.linalg.det(Q) < 0:
        Q[:, 1] *= -1
    return Q

def _riemannian_perturb(points, step_size, rng):
    """
    Apply a Riemannian perturbation: rotate a random subset of points.
    This keeps points on the sphere.
    """
    new_points = points.copy()
    # Select a random subset of points to rotate
    # Rotating all points is a global symmetry, so it doesn't change energy.
    # We must rotate a *subset* to change the configuration.
    # Let's rotate a random half of the points.
    n_perturb = N // 2
    indices = rng.choice(N, size=n_perturb, replace=False)
    
    # Generate a random rotation
    R = _random_rotation_matrix(rng)
    
    # Apply rotation scaled by step_size? 
    # Rotations are global. We can't easily "scale" a rotation of a subset 
    # to be a small step in the manifold unless we use the exponential map 
    # with a small angle.
    # Simpler: Pick a random axis and rotate by a small angle.
    axis = rng.randn(3)
    axis /= np.linalg.norm(axis)
    angle = step_size * rng.randn()
    
    # Rodrigues' rotation formula
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    I = np.eye(3)
    R = I + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    
    new_points[indices] = (R @ new_points[indices].T).T
    return new_points

def _parallel_tempering(points, max_iter=500, time_limit=60.0, rng=None, n_replicas=4):
    """
    Parallel Tempering (Replica Exchange) Simulated Annealing.
    """
    if rng is None:
        rng = np.random.RandomState(999)
    
    start = time.time()
    
    # Initialize replicas
    # Replica 0 is the current best
    replicas = [points.copy() for _ in range(n_replicas)]
    for i in range(1, n_replicas):
        # Initialize others with perturbations
        replicas[i] = _riemannian_perturb(replicas[0], 0.5, rng)
    
    # Evaluate energies
    energies = [_fast_log_energy(r) for r in replicas]
    
    # Temperatures
    # We want a range of temperatures.
    # T_low allows small moves, T_high allows large moves.
    # Since we are using fixed step sizes for perturbations, we can just use 
    # a schedule of step sizes or actual temperatures for Metropolis.
    # Let's use actual temperatures for Metropolis acceptance.
    # E is negative. Lower E is better.
    # Delta E = E_new - E_old.
    # Accept if Delta E < 0 or exp(-Delta E / T) > random.
    
    # Initial temperatures
    T_low = 0.1
    T_high = 10.0
    temperatures = np.linspace(T_low, T_high, n_replicas)
    
    # Best result overall
    best_points = replicas[0].copy()
    best_energy = energies[0]
    
    # Step size for perturbations
    base_step = 0.5
    
    for it in range(max_iter):
        if time.time() - start > time_limit:
            break
            
        # 1. Evolve each replica
        for i in range(n_replicas):
            if time.time() - start > time_limit:
                break
            
            # Perturb
            new_points = _riemannian_perturb(replicas[i], base_step, rng)
            new_energy = _fast_log_energy(new_points)
            
            # Metropolis Acceptance
            delta_e = new_energy - energies[i]
            if delta_e < 0:
                replicas[i] = new_points
                energies[i] = new_energy
            else:
                T = temperatures[i]
                if T > 1e-9:
                    if np.log(rng.rand()) < -delta_e / T:
                        replicas[i] = new_points
                        energies[i] = new_energy
                        
        # 2. Attempt swaps between adjacent replicas
        # Swap if it preserves detailed balance
        for i in range(n_replicas - 1):
            if time.time() - start > time_limit:
                break
            
            E_i = energies[i]
            E_j = energies[i+1]
            T_i = temperatures[i]
            T_j = temperatures[i+1]
            
            # Delta E for swap
            # Replica i gets E_j, Replica j gets E_i
            # Delta = (E_j/T_i + E_i/T_j) - (E_i/T_i + E_j/T_j)
            delta = (E_j - E_i) * (1/T_i - 1/T_j)
            
            if delta < 0:
                # Always accept
                replicas[i], replicas[i+1] = replicas[i+1], replicas[i]
                energies[i], energies[i+1] = energies[i+1], energies[i]
            else:
                if np.log(rng.rand()) < -delta:
                    replicas[i], replicas[i+1] = replicas[i+1], replicas[i]
                    energies[i], energies[i+1] = energies[i+1], energies[i]
                    
        # 3. Update best
        min_idx = np.argmin(energies)
        if energies[min_idx] < best_energy:
            best_energy = energies[min_idx]
            best_points = replicas[min_idx].copy()
            
            # Polish the best replica to stay in basin
            # Do this periodically to avoid wasting time on polishing every step
            if it % 10 == 0 and time.time() - start < time_limit - 5.0:
                polished = _lbfgs_polish(best_points, max_iter=50, ftol=1e-8, gtol=1e-6, time_limit=2.0)
                e_polished = _fast_log_energy(polished)
                if e_polished < best_energy:
                    best_energy = e_polished
                    best_points = polished.copy()
                    # Update the replica that was best
                    replicas[min_idx] = best_points.copy()
                    energies[min_idx] = best_energy

    return best_points, best_energy

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Initializations
    inits = [
        _fibonacci_sphere(N),
        _random_uniform_sphere(N, seed=1),
        _random_uniform_sphere(N, seed=2),
        _random_uniform_sphere(N, seed=3),
        _random_uniform_sphere(N, seed=42),
    ]
    
    best_points = inits[0].copy()
    best_energy = _log_energy(best_points)
    
    # First pass: moderate optimization on all initializations
    for idx, init in enumerate(inits):
        if time.time() > deadline - 80.0:
            break
        
        polished = _lbfgs_polish(init, max_iter=1000, ftol=1e-10, gtol=1e-8, time_limit=10.0)
        e_polish = _log_energy(polished)
        
        if e_polish < best_energy:
            best_energy = e_polish
            best_points = polished.copy()
    
    # Second pass: Parallel Tempering Simulated Annealing
    if time.time() < deadline - 50.0:
        pt_result, pt_energy = _parallel_tempering(best_points, max_iter=500, time_limit=deadline - time.time() - 40.0)
        # Verify with precise energy
        e_pt_precise = _log_energy(pt_result)
        if e_pt_precise < best_energy:
            best_energy = e_pt_precise
            best_points = pt_result.copy()
    
    # Third pass: final high-precision polish
    if time.time() < deadline - 10.0:
        final_polished = _lbfgs_polish(best_points, max_iter=5000, ftol=1e-14, gtol=1e-12, time_limit=deadline - time.time() - 5.0)
        e_final = _log_energy(final_polished)
        if e_final < best_energy:
            best_points = final_polished.copy()
    
    best_points = _project_to_sphere(best_points)
    
    if not np.all(np.isfinite(best_points)):
        return _fibonacci_sphere(N)
    
    # Check for points too close
    diff = best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=-1)
    iu = np.triu_indices(N, k=1)
    if np.min(dists_sq[iu]) < 1e-18:
        return _fibonacci_sphere(N)
    
    return best_points
