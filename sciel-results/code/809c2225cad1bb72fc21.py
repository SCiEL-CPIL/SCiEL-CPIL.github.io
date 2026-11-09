"""Morse cluster global minima finder using basin hopping with structured seeds."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from concurrent.futures import ThreadPoolExecutor, as_completed

# Global time management
TOTAL_TIME_BUDGET = 290.0  # Leave margin for 330s kill
INSTANCE_TIME_BUDGET = TOTAL_TIME_BUDGET / 6.0
_start_global = time.time()


def _morse_energy(pos, rho, penalty_lambda=100.0):
    """Compute Morse energy for all pairs with penalty for close contacts."""
    n = pos.shape[0]
    if n < 2:
        return 0.0
    # Pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    # Mask out self-distances
    mask = ~np.eye(n, dtype=bool)
    dist = dist[mask]
    # Morse potential: exp(rho*(1-r))*(exp(rho*(1-r)) - 2)
    x = rho * (1.0 - dist)
    exp_x = np.exp(x)
    energy = np.sum(exp_x * (exp_x - 2.0))
    
    # Penalty for r < 0.5
    violations = np.maximum(0.0, 0.5 - dist)
    penalty = penalty_lambda * np.sum(violations ** 2)
    
    return energy + penalty


def _morse_gradient(pos, rho, penalty_lambda=100.0):
    """Compute gradient of Morse energy with penalty."""
    n = pos.shape[0]
    if n < 2:
        return np.zeros_like(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1) + 1e-15)
    mask = ~np.eye(n, dtype=bool)
    
    # Morse gradient
    x = rho * (1.0 - dist)
    exp_x = np.exp(x)
    dE_dr_morse = -2.0 * rho * exp_x * (exp_x - 1.0)
    
    # Penalty gradient: d/dr (lambda * max(0, 0.5-r)^2) = -2*lambda*(0.5-r) if r<0.5 else 0
    # Note: derivative of (0.5-r)^2 is -2(0.5-r).
    # So dE_pen/dr = -2 * penalty_lambda * (0.5 - dist) * (dist < 0.5)
    dE_dr_pen = np.where(dist < 0.5, -2.0 * penalty_lambda * (0.5 - dist), 0.0)
    
    dE_dr = dE_dr_morse + dE_dr_pen
    
    # Set diagonal to 0
    dE_dr = np.where(mask, dE_dr, 0.0)
    
    # Vectorized gradient calculation
    # grad_i = sum_j dE_dr_ij * (pos_i - pos_j) / r_ij
    # (pos_i - pos_j) / r_ij = diff_ij / dist_ij
    inv_dist = np.where(dist > 1e-10, 1.0 / dist, 0.0)
    # dE_dr is (n, n), diff is (n, n, 3), inv_dist is (n, n)
    # We want to sum over j (axis 1) for each i (axis 0)
    # force_ij = dE_dr_ij * diff_ij * inv_dist_ij
    # Broadcast dE_dr and inv_dist to (n, n, 1)
    forces = dE_dr[:, :, np.newaxis] * diff * inv_dist[:, :, np.newaxis]
    grad = np.sum(forces, axis=1)
    
    return grad


def _local_minimize(pos, rho, maxiter=200):
    """Local minimization using L-BFGS-B with distance constraint handling."""
    n = pos.shape[0]
    
    def objective_and_grad(x):
        p = x.reshape(-1, 3)
        e = _morse_energy(p, rho)
        g = _morse_gradient(p, rho).flatten()
        return e, g
    
    # Use L-BFGS-B
    try:
        res = minimize(
            objective_and_grad,
            pos.flatten(),
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'gtol': 1e-8}
        )
        return res.fun, res.x.reshape(-1, 3)
    except Exception:
        return _morse_energy(pos, rho), pos


def _enforce_min_distance(pos, min_dist=0.5):
    """Push apart atoms that are too close."""
    n = pos.shape[0]
    for _ in range(50):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                d = pos[i] - pos[j]
                r = np.linalg.norm(d)
                if r < min_dist and r > 1e-10:
                    push = (min_dist - r) / r * 0.5
                    pos[i] += d * push
                    pos[j] -= d * push
                    moved = True
                elif r <= 1e-10:
                    # Random nudge
                    pos[i] += np.random.randn(3) * 0.01
                    moved = True
        if not moved:
            break
    return pos


def _generate_fcc_seed(n, spacing=1.0):
    """Generate FCC-like initial configuration."""
    if n == 1:
        return np.array([[0.0, 0.0, 0.0]])
    
    # Generate FCC lattice points
    max_coord = int(np.ceil(n ** (1/3.0))) + 3
    points = []
    
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                # FCC: simple cubic + face centers
                base = np.array([i, j, k]) * spacing
                # Add face centers
                for dx, dy, dz in [(0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)]:
                    p = base + np.array([dx, dy, dz]) * spacing
                    if np.linalg.norm(p) <= max_coord * spacing:
                        points.append(p)
    
    points = np.array(points)
    
    # Select n closest to center
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)[:n]
    seed = points[idx].copy()
    
    # Center it
    seed -= seed.mean(axis=0)
    return seed


def _generate_hcp_seed(n, spacing=1.0):
    """Generate HCP-like initial configuration."""
    if n == 1:
        return np.array([[0.0, 0.0, 0.0]])
    
    a = spacing
    c = np.sqrt(8.0/3.0) * a
    max_coord = int(np.ceil(n ** (1/3.0))) + 3
    points = []
    
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                # HCP: hexagonal layers
                x = a * (i + 0.5 * (j % 2))
                y = a * np.sqrt(3.0)/2.0 * j
                z = c * k
                p = np.array([x, y, z])
                if np.linalg.norm(p) <= max_coord * a:
                    points.append(p)
    
    points = np.array(points)
    
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)[:n]
    seed = points[idx].copy()
    seed -= seed.mean(axis=0)
    return seed


def _generate_random_seed(n, spacing=1.0, rng=None):
    """Generate random compact seed."""
    if rng is None:
        rng = np.random.default_rng()
    radius = spacing * n ** (1.0/3.0) * 0.8
    points = []
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            # Check min distance
            if all(np.linalg.norm(p - q) >= 0.6 * spacing for q in points):
                points.append(p)
        if len(points) < n and len(points) > 0 and len(points) % 100 == 0:
            # Relax packing constraint slightly
            radius *= 1.05
    return np.array(points)


def _basin_hopping(pos, rho, n, time_limit, rng):
    """Basin hopping with structured kicks."""
    current_energy, current_pos = _local_minimize(pos, rho)
    best_energy = current_energy
    best_pos = current_pos.copy()
    
    t_start = time.time()
    temp = 0.1 * max(1.0, abs(current_energy) / n)
    
    while time.time() - t_start < time_limit:
        # Generate kick
        kick_size = temp * 0.3
        kick = rng.normal(0, kick_size, (n, 3))
        trial_pos = current_pos + kick
        trial_pos = _enforce_min_distance(trial_pos, 0.5)
        
        trial_energy, trial_pos = _local_minimize(trial_pos, rho)
        
        delta_e = trial_energy - current_energy
        if delta_e < 0 or rng.random() < np.exp(-delta_e / temp):
            current_energy = trial_energy
            current_pos = trial_pos.copy()
            if current_energy < best_energy:
                best_energy = current_energy
                best_pos = current_pos.copy()
        
        # Cool down
        temp *= 0.95
        if temp < 1e-6 * max(1.0, abs(best_energy) / n):
            break
    
    return best_energy, best_pos


def construct(instance):
    """Find low-energy Morse cluster configuration."""
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind", 10))
    
    t_start = time.time()
    time_remaining = lambda: INSTANCE_TIME_BUDGET - (time.time() - t_start)
    
    # Generate diverse initial seeds
    rng = np.random.default_rng(hash((n, rho)) % 2**32)
    
    seeds = []
    seeds.append(_generate_fcc_seed(n))
    seeds.append(_generate_hcp_seed(n))
    for _ in range(2):
        seeds.append(_generate_random_seed(n, rng=rng))
    
    # Run basin hopping from each seed in parallel
    best_energy = np.inf
    best_pos = seeds[0].copy()
    
    n_seeds = len(seeds)
    per_seed_time = max(1.0, time_remaining() / n_seeds)
    
    # Use ThreadPoolExecutor for parallel search
    # Note: GIL might limit true parallelism for CPU-bound tasks, but scipy/numpy release GIL.
    # Also, I/O or system calls might benefit.
    # However, for pure CPU bound NumPy, threads might not speed up computation,
    # but they allow overlapping if there are any blocking calls.
    # Given the constraint of 4 threads, we use 4 workers.
    
    results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for i, seed in enumerate(seeds):
            # Create a unique RNG for each thread to avoid contention
            seed_rng = np.random.default_rng(hash((n, rho, i)) % 2**32)
            seed_pos = _enforce_min_distance(seed, 0.5)
            futures.append(executor.submit(_basin_hopping, seed_pos, rho, n, per_seed_time, seed_rng))
        
        for future in as_completed(futures):
            try:
                e, pos = future.result()
                if e < best_energy:
                    best_energy = e
                    best_pos = pos.copy()
            except Exception as e:
                pass # Ignore errors in individual threads
    
    # Final polish
    if time_remaining() > 2.0:
        best_energy, best_pos = _local_minimize(best_pos, rho, maxiter=200)
        best_pos = _enforce_min_distance(best_pos, 0.5)
    
    return best_pos
# EVOLVE-BLOCK-END
