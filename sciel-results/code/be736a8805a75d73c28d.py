"""Optimize logarithmic energy of 150 points on the unit sphere using Riemannian optimization."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize
from scipy.spatial.distance import cdist


def _energy_and_grad(pts):
    """Compute logarithmic energy and gradient on the sphere.
    
    E = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log|xi-xj|
    
    Gradient: dE/dx_i = sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    """
    n = len(pts)
    # Compute pairwise differences
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
    dist_sq = np.sum(diff * diff, axis=2)  # (n, n)
    
    # Avoid division by zero on diagonal
    np.fill_diagonal(dist_sq, 1.0)
    
    # inv_dist_sq[i,j] = 1/|xi-xj|^2, with diagonal = 0
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    # Gradient: grad[i] = sum_j (pts[i] - pts[j]) / |pts[i]-pts[j]|^2
    # = sum_j diff[i,j] * inv_dist_sq[i,j]
    grad = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    
    # Energy: sum_{i<j} log(1/|xi-xj|) = -sum_{i<j} log|xi-xj|
    # Use upper triangle only
    iu = np.triu_indices(n, k=1)
    d = np.sqrt(dist_sq[iu])
    # Filter out any very small distances (shouldn't happen)
    mask = d > 1e-15
    d = d[mask]
    if len(d) == 0:
        energy = 0.0
    else:
        energy = -np.sum(np.log(d))
    
    return energy, grad


def _energy_only(pts):
    """Compute just the energy (for speed during optimization)."""
    energy, _ = _energy_and_grad(pts)
    return energy


def _project_to_sphere(pts):
    """Project points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms


def _fibonacci_sphere(n, seed=None):
    """Generate Fibonacci spiral points on sphere."""
    if seed is not None:
        rng = np.random.RandomState(seed)
        offset = rng.uniform(0, 1)
    else:
        offset = 0.0
    i = np.arange(n) + 0.5 + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return pts


def _random_sphere(n, seed=None):
    """Generate random points on sphere."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
    pts = rng.randn(n, 3)
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    return pts


def _icosahedron_based(n, seed=None):
    """Generate points based on icosahedron subdivision for better initial spread."""
    # Start with icosahedron vertices
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    if n <= 12:
        return verts[:n]
    
    # Use Fibonacci as fallback for simplicity, but with better initial distribution
    return _fibonacci_sphere(n, seed=seed)


def _riemannian_lbfgs(pts, time_limit=30.0, tol=1e-10):
    """Riemannian L-BFGS optimization using scipy's minimize with custom constraints.
    
    We use a retraction-based approach: optimize in R^3 with projection to sphere.
    """
    start_time = time.time()
    n = len(pts)
    
    # Flatten points for scipy
    x0 = pts.flatten()
    
    def objective(x):
        if time.time() - start_time > time_limit:
            raise Exception("Time limit exceeded")
        p = x.reshape(n, 3)
        p = _project_to_sphere(p)
        e, _ = _energy_and_grad(p)
        return e
    
    def gradient(x):
        if time.time() - start_time > time_limit:
            raise Exception("Time limit exceeded")
        p = x.reshape(n, 3)
        p = _project_to_sphere(p)
        _, g = _energy_and_grad(p)
        # Project gradient to tangent space
        g_tangent = g - np.sum(g * p, axis=1, keepdims=True) * p
        return g_tangent.flatten()
    
    try:
        result = minimize(
            objective, x0, jac=gradient,
            method='L-BFGS-B',
            options={'maxiter': 5000, 'ftol': tol, 'gtol': 1e-8, 'maxfun': 10000}
        )
        best_pts = _project_to_sphere(result.x.reshape(n, 3))
    except Exception:
        best_pts = _project_to_sphere(pts.copy())
    
    return best_pts


def _adam_optimize(pts, lr=0.01, max_iter=5000, time_limit=30.0, tol=1e-10):
    """Adam optimizer on the sphere with adaptive learning rate."""
    n = len(pts)
    pts = _project_to_sphere(pts.copy())
    
    m = np.zeros_like(pts)  # First moment
    v = np.zeros_like(pts)  # Second moment
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    
    prev_energy = _energy_only(pts)
    start_time = time.time()
    
    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        energy, grad = _energy_and_grad(pts)
        
        # Project gradient to tangent space
        grad_tangent = grad - np.sum(grad * pts, axis=1, keepdims=True) * pts
        
        # Adam update
        m = beta1 * m + (1 - beta1) * grad_tangent
        v = beta2 * v + (1 - beta2) * (grad_tangent ** 2)
        m_hat = m / (1 - beta1 ** (it + 1))
        v_hat = v / (1 - beta2 ** (it + 1))
        
        pts = pts - lr * m_hat / (np.sqrt(v_hat) + eps)
        pts = _project_to_sphere(pts)
        
        new_energy = _energy_only(pts)
        
        if abs(prev_energy - new_energy) < tol:
            break
        prev_energy = new_energy
    
    return pts


def _parallel_tempering(initial_pts, time_limit=100.0, n_replicas=4):
    """
    Parallel Tempering (Replica Exchange) optimization.
    Uses multiple replicas at different temperatures to escape local minima.
    """
    n = len(initial_pts)
    start_time = time.time()
    
    # Initialize replicas
    # Replica 0 is the "cold" one (T=0 effectively, only accepts improvements)
    # Others are "hot"
    temps = [0.0, 0.05, 0.1, 0.2]
    if n_replicas > len(temps):
        # Extend temps if needed, but 4 is usually enough for N=150
        extra = [0.2 * (1 + i) for i in range(n_replicas - len(temps))]
        temps.extend(extra)
    temps = temps[:n_replicas]
    
    replicas = []
    for i in range(n_replicas):
        if i == 0:
            replicas.append(_project_to_sphere(initial_pts.copy()))
        else:
            # Perturb initial points for diversity
            rng = np.random.RandomState(42 + i)
            perturb = rng.randn(n, 3) * 0.1
            replicas.append(_project_to_sphere(initial_pts + perturb))
            
    energies = [_energy_only(r) for r in replicas]
    best_pts = replicas[0].copy()
    best_energy = min(energies)
    
    # Precompute some stats for perturbation size
    # Typical distance between points on sphere for N=150 is ~ 2/sqrt(N) ~ 0.16
    # Perturbation magnitude should be comparable to this scale
    base_perturb_scale = 0.05 
    
    # Main loop
    # We estimate cost per step: N_replicas * (Energy Calc + Gradient? No, just Energy for SA/PT steps)
    # Energy calc is O(N^2). For N=150, 150*150 = 22500 ops. Very fast.
    # We can do many steps.
    
    steps = 0
    # Target: run until time_limit is reached
    # Let's run in chunks to check time
    
    while time.time() - start_time < time_limit:
        # 1. Perform a Metropolis step for each replica
        new_energies = energies[:]
        for i in range(n_replicas):
            T = temps[i]
            # Generate perturbation
            # Scale perturbation slightly with temperature to allow larger jumps in hot replicas
            scale = base_perturb_scale * (1.0 + 0.5 * T) 
            rng = np.random.RandomState(steps * 100 + i)
            perturb = rng.randn(n, 3) * scale
            candidate = _project_to_sphere(replicas[i] + perturb)
            
            new_e = _energy_only(candidate)
            delta_e = new_e - energies[i]
            
            if T == 0.0:
                if new_e < energies[i]:
                    replicas[i] = candidate
                    energies[i] = new_e
                    new_energies[i] = new_e
            else:
                # Metropolis criterion
                if delta_e < 0 or np.random.rand() < np.exp(-delta_e / T):
                    replicas[i] = candidate
                    energies[i] = new_e
                    new_energies[i] = new_e
        
        # 2. Attempt swaps between adjacent replicas
        # Sort by temperature? Temps are fixed. Swap adjacent in index (which corresponds to T order if we defined them sorted)
        # Our temps list is [0, 0.05, 0.1, 0.2]. So index 0 is coldest.
        for i in range(n_replicas - 1):
            T1 = temps[i]
            T2 = temps[i+1]
            E1 = new_energies[i]
            E2 = new_energies[i+1]
            
            # Delta = (E1/T2 - E2/T1) - (E1/T1 - E2/T2) = E1(1/T2 - 1/T1) - E2(1/T1 - 1/T2)
            # = (E1 - E2) * (1/T2 - 1/T1)
            # If T1 < T2, 1/T2 < 1/T1, so (1/T2 - 1/T1) is negative.
            # Let's just use the standard formula:
            # Accept swap with prob exp(- (E1/T2 - E1/T1 - E2/T1 + E2/T2))
            # = exp( (E2 - E1) * (1/T2 - 1/T1) )
            
            if T1 == 0.0 or T2 == 0.0:
                # Handle T=0 carefully. If T1=0, replica 0 is fixed to best so far? 
                # No, in PT, T=0 means it only accepts improvements. Swapping with T=0 is tricky.
                # Usually, one doesn't swap with T=0 or handles it as limit.
                # Let's just skip swaps involving T=0 to be safe and efficient, 
                # or treat T=0 as very small epsilon.
                # Actually, if T1=0, E1 is the global best of replica 0.
                # Swapping a high energy state from T2 into T1 would be rejected by T1's next step if it's worse.
                # But T1's step already happened.
                # Let's just use epsilon for T=0 in swap calculation.
                T1_eff = 1e-6
                T2_eff = T2
            else:
                T1_eff = T1
                T2_eff = T2
                
            delta_swap = (E2 - E1) * (1.0/T2_eff - 1.0/T1_eff)
            
            if delta_swap < 0 or np.random.rand() < np.exp(-delta_swap):
                # Swap
                replicas[i], replicas[i+1] = replicas[i+1], replicas[i]
                energies[i], energies[i+1] = energies[i+1], energies[i]
                new_energies[i], new_energies[i+1] = new_energies[i+1], new_energies[i]
        
        # Update best
        for i in range(n_replicas):
            if energies[i] < best_energy:
                best_energy = energies[i]
                best_pts = replicas[i].copy()
        
        steps += 1
        
        # Time check every 100 steps
        if steps % 100 == 0:
             if time.time() - start_time > time_limit:
                 break

    return best_pts


def _final_polish(pts, time_limit=15.0):
    """Final high-precision polish with very tight tolerance."""
    n = len(pts)
    pts = _project_to_sphere(pts.copy())
    
    start_time = time.time()
    prev_energy = _energy_only(pts)
    
    # Use L-BFGS with very tight tolerance
    x0 = pts.flatten()
    
    def objective(x):
        if time.time() - start_time > time_limit:
            raise Exception("Time limit exceeded")
        p = x.reshape(n, 3)
        p = _project_to_sphere(p)
        e, _ = _energy_and_grad(p)
        return e
    
    def gradient(x):
        if time.time() - start_time > time_limit:
            raise Exception("Time limit exceeded")
        p = x.reshape(n, 3)
        p = _project_to_sphere(p)
        _, g = _energy_and_grad(p)
        g_tangent = g - np.sum(g * p, axis=1, keepdims=True) * p
        return g_tangent.flatten()
    
    try:
        result = minimize(
            objective, x0, jac=gradient,
            method='L-BFGS-B',
            options={'maxiter': 10000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 20000}
        )
        polished = _project_to_sphere(result.x.reshape(n, 3))
    except Exception:
        polished = pts
    
    return polished


def _velocity_verlet(pts, time_limit=20.0):
    """Velocity Verlet integration on the sphere for optimization."""
    n = len(pts)
    pts = _project_to_sphere(pts.copy())
    vel = np.zeros_like(pts)
    
    dt = 0.01
    prev_energy = _energy_only(pts)
    start_time = time.time()
    
    for it in range(5000):
        if time.time() - start_time > time_limit:
            break
        
        _, grad = _energy_and_grad(pts)
        grad_tangent = grad - np.sum(grad * pts, axis=1, keepdims=True) * pts
        
        # Update velocity
        vel = 0.99 * vel - dt * grad_tangent
        
        # Update position
        pts = pts + dt * vel
        pts = _project_to_sphere(pts)
        
        new_energy = _energy_only(pts)
        if abs(prev_energy - new_energy) < 1e-12:
            break
        prev_energy = new_energy
    
    return pts


def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    best_pts = None
    best_energy = float('inf')
    
    def try_candidate(pts):
        nonlocal best_pts, best_energy
        if time.time() - start_time > deadline - 5.0:
            return
        pts = _project_to_sphere(pts)
        if not np.all(np.isfinite(pts)):
            return
        # Check min distance
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)
        if np.min(dist) < 1e-9:
            return
        e = _energy_only(pts)
        if e < best_energy:
            best_energy = e
            best_pts = pts.copy()

    # Strategy:
    # 1. Generate a few good initial guesses (Fibonacci, Random)
    # 2. Run Parallel Tempering on each for a significant chunk of time
    # 3. Polish the best result with L-BFGS
    
    initializations = [
        _fibonacci_sphere(N),
        _random_sphere(N, seed=42),
        _random_sphere(N, seed=123)
    ]
    
    # Allocate time: 
    # 3 initializations.
    # Let's give each PT run ~ 40s. Total 120s.
    # Remaining 20s for polishing.
    
    pt_time_budget = 40.0
    
    for i, init_pts in enumerate(initializations):
        if time.time() - start_time > deadline - 25.0:
            break
            
        remaining_for_pt = min(pt_time_budget, deadline - 25.0 - (time.time() - start_time))
        if remaining_for_pt < 5.0:
            continue
            
        try:
            # Run Parallel Tempering
            pt_result = _parallel_tempering(init_pts, time_limit=remaining_for_pt, n_replicas=4)
            try_candidate(pt_result)
            
            # If we have time, do a quick local polish on the PT result before moving to next init
            if time.time() - start_time < deadline - 15.0:
                polish_t = min(5.0, deadline - 15.0 - (time.time() - start_time))
                if polish_t > 1.0:
                    polished = _riemannian_lbfgs(pt_result, time_limit=polish_t)
                    try_candidate(polished)
        except Exception:
            pass

    # Final Global Polish on the absolute best found
    if best_pts is not None and time.time() - start_time < deadline - 5.0:
        polish_time = min(15.0, deadline - 5.0 - (time.time() - start_time))
        if polish_time > 1.0:
            try:
                final_polished = _final_polish(best_pts, time_limit=polish_time)
                try_candidate(final_polished)
            except Exception:
                pass

    # Fallback
    if best_pts is None:
        best_pts = _fibonacci_sphere(N)
    
    best_pts = _project_to_sphere(best_pts)
    
    # Final validation
    if not np.all(np.isfinite(best_pts)):
        best_pts = _fibonacci_sphere(N)
        best_pts = _project_to_sphere(best_pts)
        
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    if np.min(dist) < 1e-9:
        best_pts = _fibonacci_sphere(N)
        best_pts = _project_to_sphere(best_pts)

    return best_pts
# EVOLVE-BLOCK-END
