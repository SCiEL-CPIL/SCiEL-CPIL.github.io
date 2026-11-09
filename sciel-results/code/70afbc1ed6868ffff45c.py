"""Morse cluster global minima search with integrated Basin Hopping and L-BFGS."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as opt

# Global deadline shared across all six construct() calls
_START_TIME = time.time()
_TOTAL_BUDGET = 310.0  # 20s safety margin before 330s kill

def _time_left():
    return _TOTAL_BUDGET - (time.time() - _START_TIME)

def _morse_energy(pos, rho):
    """Compute total Morse energy."""
    n = len(pos)
    if n < 2:
        return 0.0
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    iu = np.triu_indices(n, k=1)
    d = dist[iu]
    # Avoid log(0) or exp issues if d is very small, though constraints should prevent this
    d = np.maximum(d, 1e-6)
    e = np.exp(rho * (1.0 - d))
    return float(np.sum(e * (e - 2.0)))

def _morse_gradient(pos, rho):
    """Compute gradient of Morse energy."""
    n = len(pos)
    grad = np.zeros_like(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    iu = np.triu_indices(n, k=1)
    d = dist[iu]
    d = np.maximum(d, 1e-6)
    
    # Force magnitude: dE/dr = rho * exp(rho(1-r)) * (exp(rho(1-r)) - 1) * exp(rho(1-r)) 
    # Let u = rho(1-r). E = exp(u)(exp(u)-2). dE/dr = -rho * exp(u)(exp(u)-1)
    # Wait, dE/dr = dE/du * du/dr. dE/du = exp(u)(exp(u)-2) + exp(u)(exp(u)) = exp(u)(2exp(u)-2) = 2exp(u)(exp(u)-1).
    # du/dr = -rho.
    # So dE/dr = -2 * rho * exp(rho(1-r)) * (exp(rho(1-r)) - 1)
    
    e = np.exp(rho * (1.0 - d))
    force_mag = -2.0 * rho * e * (e - 1.0)
    
    # force vector on j due to i is -force_mag * (r_j - r_i) / |r_j - r_i|
    # grad_i = sum_j force_mag * (r_i - r_j) / d_ij
    # grad_j = sum_i force_mag * (r_j - r_i) / d_ij
    
    # We have diff[i,j] = pos[i] - pos[j]
    # unit_vec = diff / d
    # contribution to grad[i] is force_mag * unit_vec
    # contribution to grad[j] is -force_mag * unit_vec
    
    # Vectorized accumulation
    # We only have iu indices.
    # diff[iu] is shape (M, 3)
    # force_mag is shape (M,)
    # unit_vec = diff[iu] / d[:, np.newaxis]
    
    unit_vec = diff[iu] / d[:, np.newaxis]
    contrib = force_mag[:, np.newaxis] * unit_vec
    
    # Add to grad[i]
    np.add.at(grad, iu[0], contrib)
    # Subtract from grad[j]
    np.add.at(grad, iu[1], -contrib)
    
    return grad

def _build_fcc_cluster(n, spacing=1.0):
    """Build an FCC-like compact cluster."""
    points = []
    for i in range(-6, 7):
        for j in range(-6, 7):
            for k in range(-6, 7):
                if (i + j + k) % 2 == 0:
                    p = np.array([i, j, k]) * spacing * 0.7071
                    points.append(p)
    
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)[:n]
    return points[idx].copy()

def _build_bcc_cluster(n, spacing=1.0):
    """Build a BCC-like compact cluster."""
    points = []
    for i in range(-5, 6):
        for j in range(-5, 6):
            for k in range(-5, 6):
                p = np.array([i, j, k]) * spacing * 0.577
                points.append(p)
                pc = np.array([i + 0.5, j + 0.5, k + 0.5]) * spacing * 0.577
                points.append(pc)
    
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)[:n]
    return points[idx].copy()

def _build_icosahedral_shells(n, spacing=1.0):
    """Build cluster using icosahedral shell geometry."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    ico12 = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    norm_ico = np.linalg.norm(ico12[0])
    ico12 /= norm_ico
    
    points = [np.zeros(3)]
    
    shells = [
        (1.0, 12),
        (1.633, 30),
        (2.0, 56),
        (2.449, 92),
        (2.828, 140),
    ]
    
    for radius, count in shells:
        if len(points) >= n:
            break
        # Use 12 vertices rotated/scaled for first shell, then random for others
        if count == 12:
            for v in ico12:
                if len(points) >= n:
                    break
                points.append(v * radius)
        else:
            # Fibonacci sphere for better distribution
            golden = (1.0 + np.sqrt(5.0)) / 2.0
            for i in range(count):
                if len(points) >= n:
                    break
                z = 1.0 - 2.0 * i / max(1, count - 1)
                r = np.sqrt(1.0 - z * z)
                theta = 2.0 * np.pi * i / golden
                p = np.array([r * np.cos(theta), r * np.sin(theta), z]) * radius
                points.append(p)
                
    # Fill remaining
    shell_radius = 3.2
    while len(points) < n:
        rng = np.random.default_rng(int(shell_radius * 1000))
        added = 0
        for _ in range(200):
            if len(points) >= n:
                break
            p = rng.normal(0, 1, 3)
            p = p / np.linalg.norm(p) * shell_radius
            if all(np.linalg.norm(p - q) >= 0.5 * spacing for q in points):
                points.append(p)
                added += 1
        if added == 0:
            shell_radius += 0.4
        else:
            shell_radius += 0.25
    
    return np.array(points[:n])

def _build_random_compact(n, spacing=1.0, seed=42):
    """Build a random compact cluster."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0/3.0) * 0.85
    points = []
    max_attempts = 10000
    for _ in range(max_attempts):
        if len(points) >= n:
            break
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.65 * spacing for q in points):
                points.append(p)
    
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    
    return np.array(points[:n])

def _ensure_min_distance(pos, min_d=0.5, max_iter=50):
    """Ensure all pairs are at least min_d apart by nudging."""
    pos = pos.copy()
    n = len(pos)
    for _ in range(max_iter):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < min_d:
                    diff = pos[i] - pos[j]
                    if np.linalg.norm(diff) < 1e-10:
                        diff = np.array([1.0, 0, 0])
                    diff /= np.linalg.norm(diff)
                    shift = (min_d - d) * 0.5 + 0.01
                    pos[i] += diff * shift
                    pos[j] -= diff * shift
                    moved = True
        if not moved:
            break
    return pos

def _local_relax(pos, rho, time_limit=1.0):
    """Local relaxation using L-BFGS-B."""
    n = len(pos)
    x0 = pos.flatten()
    
    def obj(x):
        p = x.reshape(n, 3)
        return _morse_energy(p, rho)
        
    def grad(x):
        p = x.reshape(n, 3)
        return _morse_gradient(p, rho).flatten()

    # Check time before starting
    start_t = time.time()
    try:
        result = opt.minimize(
            obj, x0, jac=grad, method='L-BFGS-B',
            options={'maxiter': 200, 'gtol': 1e-5, 'disp': False}
        )
        return result.x.reshape(n, 3), result.fun
    except Exception:
        return pos, _morse_energy(pos, rho)

def _basin_hopping(pos, rho, time_budget, rng):
    """Basin hopping loop."""
    best_pos = pos.copy()
    best_energy = _morse_energy(pos, rho)
    current_pos = pos.copy()
    current_energy = best_energy
    
    start_time = time.time()
    step_scale = 0.5
    beta = 10.0 # Initial beta, might need tuning
    
    while time.time() - start_time < time_budget:
        # Perturb
        noise = rng.normal(0, step_scale, pos.shape)
        trial_pos = current_pos + noise
        trial_pos = _ensure_min_distance(trial_pos, 0.5)
        
        # Local relax
        trial_pos, trial_energy = _local_relax(trial_pos, rho, time_limit=0.5)
        
        # Metropolis acceptance
        delta_e = trial_energy - current_energy
        if delta_e < 0 or rng.random() < np.exp(-delta_e * beta):
            current_pos = trial_pos
            current_energy = trial_energy
            
            if trial_energy < best_energy:
                best_energy = trial_energy
                best_pos = trial_pos.copy()
                # Decay step scale slightly on success? Or keep it.
                # Let's decay beta to allow larger jumps? No, usually beta increases in BH to converge.
                # But we want to explore. Let's keep beta constant or decay step_scale.
                step_scale *= 0.95
        else:
            step_scale *= 1.05 # Increase if rejected? Or keep.
            
        # Safety check
        if _time_left() < 5.0:
            break
            
    return best_pos, best_energy

def construct(instance):
    """Find low-energy Morse cluster configuration."""
    size = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind"))
    
    # Check time budget
    tl = _time_left()
    if tl < 5.0:
        # Very little time left, just return a quick result
        pos = _build_random_compact(size, spacing=1.0, seed=size)
        return _ensure_min_distance(pos)
    
    # Generate diverse starting configurations
    start_configs = []
    
    builders = [
        lambda: _build_fcc_cluster(size, 1.0),
        lambda: _build_bcc_cluster(size, 1.0),
        lambda: _build_icosahedral_shells(size, 1.0),
        lambda: _build_random_compact(size, 1.0, seed=size),
        lambda: _build_random_compact(size, 1.0, seed=size + 100),
        lambda: _build_fcc_cluster(size, 1.0) + np.random.default_rng(size).normal(0, 0.1, (size, 3)),
    ]
    
    for builder in builders:
        try:
            cfg = builder()
            if len(cfg) == size:
                start_configs.append(_ensure_min_distance(cfg))
        except Exception:
            pass
    
    if not start_configs:
        start_configs = [_build_random_compact(size, spacing=1.0, seed=size)]
    
    best_energy = float('inf')
    best_pos = start_configs[0].copy()
    
    # Determine how many basin hopping runs we can afford
    # Estimate time per BH run. Local relax takes ~0.1-0.5s. BH loop runs until time_budget.
    # We allocate time to each start config.
    n_starts = len(start_configs)
    time_per_start = (tl - 10.0) / n_starts # 10s safety margin
    
    for i in range(n_starts):
        tl = _time_left()
        if tl < 5.0:
            break
        
        # If time is very tight, just do a local relax
        if time_per_start < 2.0:
            config = start_configs[i]
            try:
                relaxed, energy = _local_relax(config, rho, time_limit=1.0)
                if energy < best_energy:
                    best_energy = energy
                    best_pos = relaxed.copy()
            except Exception:
                pass
        else:
            config = start_configs[i]
            rng = np.random.default_rng(i * 1000 + size)
            try:
                bh_pos, bh_energy = _basin_hopping(config, rho, time_per_start * 0.9, rng)
                if bh_energy < best_energy:
                    best_energy = bh_energy
                    best_pos = bh_pos.copy()
            except Exception:
                pass
    
    # Final verification: ensure minimum distance constraint
    pos = _ensure_min_distance(best_pos)
    
    return pos
# EVOLVE-BLOCK-END
