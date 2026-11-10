"""Morse cluster global minima: Multi-start basin hopping with structural templates."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

try:
    from autoresearch_suite.chem_clusters import relax as _cc_relax, morse_energy_gradient
    HAS_CHEM_CLUSTERS = True
except ImportError:
    HAS_CHEM_CLUSTERS = False

# Deadline guard: evaluator kills at 330s, use 300s as hard limit with margin
TOTAL_DEADLINE = 290.0
START_TIME = None
_deadline_lock = threading.Lock()


def _morse_energy_and_grad(positions, rho):
    """Compute Morse energy and gradient for a cluster using vectorized ops."""
    n = positions.shape[0]
    # Compute all pair distances
    d = squareform(pdist(positions))
    np.fill_diagonal(d, 1.0)  # avoid self-interaction
    r = np.where(d > 0, d, 1.0)
    
    # Morse potential: V(r) = exp(rho*(1-r)) * (exp(rho*(1-r)) - 2)
    x = rho * (1.0 - r)
    ex = np.exp(x)
    v = ex * (ex - 2.0)
    energy = 0.5 * np.sum(v)
    
    # Gradient: dV/dr = rho * exp(rho*(1-r)) * (2*exp(rho*(1-r)) - 3)
    dvdr = rho * ex * (2.0 * ex - 3.0)
    
    # Vectorized gradient computation
    factor = np.zeros_like(r)
    mask = r > 1e-10
    factor[mask] = -dvdr[mask] / r[mask]
    
    # grad[i] = sum_j factor[i,j] * (pos[i] - pos[j])
    row_sum = factor.sum(axis=1)  # shape (n,)
    grad = positions * row_sum[:, np.newaxis] - factor @ positions
    
    return energy, grad


def _morse_energy_only(positions, rho):
    """Compute Morse energy only (for speed in some contexts)."""
    n = positions.shape[0]
    d = squareform(pdist(positions))
    np.fill_diagonal(d, 1.0)
    r = np.where(d > 0, d, 1.0)
    x = rho * (1.0 - r)
    ex = np.exp(x)
    v = ex * (ex - 2.0)
    return 0.5 * np.sum(v)


def _generate_bcc(n):
    """Generate BCC-like cluster of n atoms."""
    a = 1.0
    max_coord = int(np.ceil((n / 2.0) ** (1/3)) + 2)
    
    candidates = []
    for ix in range(-max_coord, max_coord + 1):
        for iy in range(-max_coord, max_coord + 1):
            for iz in range(-max_coord, max_coord + 1):
                candidates.append(np.array([ix * a, iy * a, iz * a]))
                candidates.append(np.array([(ix + 0.5) * a, (iy + 0.5) * a, (iz + 0.5) * a]))
    
    candidates = np.array(candidates)
    dists = np.linalg.norm(candidates, axis=1)
    idx = np.argsort(dists)
    positions = candidates[idx[:n]]
    positions -= positions.mean(axis=0)
    return positions


def _generate_fcc(n):
    """Generate FCC-like cluster of n atoms."""
    a = 1.0
    max_coord = int(np.ceil((n / 4.0) ** (1/3)) + 2)
    
    candidates = []
    for ix in range(-max_coord, max_coord + 1):
        for iy in range(-max_coord, max_coord + 1):
            for iz in range(-max_coord, max_coord + 1):
                candidates.append(np.array([ix * a, iy * a, iz * a]))
                candidates.append(np.array([(ix + 0.5) * a, (iy + 0.5) * a, iz * a]))
                candidates.append(np.array([(ix + 0.5) * a, iy * a, (iz + 0.5) * a]))
                candidates.append(np.array([ix * a, (iy + 0.5) * a, (iz + 0.5) * a]))
    
    candidates = np.array(candidates)
    dists = np.linalg.norm(candidates, axis=1)
    idx = np.argsort(dists)
    positions = candidates[idx[:n]]
    positions -= positions.mean(axis=0)
    return positions


def _generate_hcp(n):
    """Generate HCP-like cluster of n atoms."""
    a = 1.0
    c = a * np.sqrt(8.0 / 3.0)
    
    candidates = []
    max_coord = int(np.ceil((n / 2.0) ** (1/3)) + 2)
    
    for layer in range(-max_coord, max_coord + 1):
        z = layer * c / 2.0
        for ix in range(-max_coord, max_coord + 1):
            for iy in range(-max_coord, max_coord + 1):
                if layer % 2 == 0:
                    x = ix * a
                    y = iy * a * np.sqrt(3.0) / 2.0
                else:
                    x = (ix + 0.5) * a
                    y = iy * a * np.sqrt(3.0) / 2.0
                candidates.append(np.array([x, y, z]))
    
    candidates = np.array(candidates)
    dists = np.linalg.norm(candidates, axis=1)
    idx = np.argsort(dists)
    positions = candidates[idx[:n]]
    positions -= positions.mean(axis=0)
    return positions


def _generate_icosahedral(n):
    """Generate icosahedral shell structure - known ground state for short-range Morse."""
    t = (1.0 + np.sqrt(5.0)) / 2.0
    ico_vertices = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    ico_vertices /= np.linalg.norm(ico_vertices[0])
    
    positions = [np.zeros(3)]
    
    # Shell 1: icosahedron (12 atoms)
    for v in ico_vertices:
        if len(positions) >= n:
            break
        positions.append(1.0 * v)
    
    # Shell 2: 30 atoms at edge midpoints
    if len(positions) < n:
        target_dot = -1.0 / t
        shell2_radius = np.sqrt(2.0 * (1.0 + 1.0/t))
        edge_mids = []
        for i in range(12):
            for j in range(i+1, 12):
                dot = np.dot(ico_vertices[i], ico_vertices[j])
                if abs(dot - target_dot) < 1e-6:
                    mid = ico_vertices[i] + ico_vertices[j]
                    mid /= np.linalg.norm(mid)
                    edge_mids.append(mid)
        
        for v in edge_mids:
            if len(positions) >= n:
                break
            positions.append(shell2_radius * v)
    
    # Shell 3: 20 atoms at face centers
    if len(positions) < n:
        face_centers = []
        for i in range(12):
            for j in range(i+1, 12):
                for k in range(j+1, 12):
                    d_ij = np.dot(ico_vertices[i], ico_vertices[j])
                    d_jk = np.dot(ico_vertices[j], ico_vertices[k])
                    d_ik = np.dot(ico_vertices[i], ico_vertices[k])
                    if (abs(d_ij - target_dot) < 1e-6 and 
                        abs(d_jk - target_dot) < 1e-6 and
                        abs(d_ik - target_dot) < 1e-6):
                        center = ico_vertices[i] + ico_vertices[j] + ico_vertices[k]
                        center /= np.linalg.norm(center)
                        face_centers.append(center)
        
        shell3_radius = 1.0515
        for v in face_centers:
            if len(positions) >= n:
                break
            positions.append(shell3_radius * v)
    
    # If still need more atoms, add points on outer shell
    if len(positions) < n:
        rng = np.random.default_rng(42)
        for _ in range(n - len(positions)):
            theta = rng.uniform(0, 2 * np.pi)
            phi = np.arccos(rng.uniform(-1, 1))
            p = 1.8 * np.array([np.sin(phi) * np.cos(theta),
                                np.sin(phi) * np.sin(theta),
                                np.cos(phi)])
            positions.append(p)
    
    result = np.array(positions[:n])
    result -= result.mean(axis=0)
    return result


def _generate_shell(n):
    """Generate concentric shell structure."""
    positions = [np.zeros(3)]
    shell_radii = [1.0, 1.8, 2.5, 3.2, 4.0, 4.8, 5.5]
    rng = np.random.default_rng(42)
    
    for radius in shell_radii:
        if len(positions) >= n:
            break
        remaining = n - len(positions)
        for _ in range(remaining):
            theta = rng.uniform(0, 2 * np.pi)
            phi = np.arccos(rng.uniform(-1, 1))
            p = radius * np.array([np.sin(phi) * np.cos(theta),
                                    np.sin(phi) * np.sin(theta),
                                    np.cos(phi)])
            positions.append(p)
    
    return np.array(positions[:n])


def _generate_random(n, seed=0):
    """Generate random cluster with minimum spacing."""
    rng = np.random.default_rng(seed)
    max_radius = np.sqrt(n) * 0.8
    positions = []
    attempts = 0
    while len(positions) < n and attempts < n * 100:
        attempts += 1
        p = rng.uniform(-max_radius, max_radius, size=3)
        if len(positions) == 0:
            positions.append(p)
        else:
            arr = np.array(positions)
            dists = np.linalg.norm(arr - p, axis=1)
            if np.min(dists) > 0.5:
                positions.append(p)
    
    # If we didn't get enough, fill with random points
    while len(positions) < n:
        p = rng.uniform(-max_radius, max_radius, size=3)
        positions.append(p)
    
    result = np.array(positions[:n])
    result -= result.mean(axis=0)
    return result


def _relax_lbfgs(positions, rho, maxiter=200):
    """Relax positions using L-BFGS-B."""
    if HAS_CHEM_CLUSTERS:
        try:
            result = _cc_relax(positions, 'morse', rho)
            if result is not None and isinstance(result, np.ndarray) and result.shape == positions.shape:
                e = _morse_energy_only(result, rho)
                return result, e
        except Exception:
            pass
    
    def fun_and_grad(x):
        pos = x.reshape(-1, 3)
        e, g = _morse_energy_and_grad(pos, rho)
        return e, g
    
    result = minimize(fun_and_grad, positions.flatten(), method='L-BFGS-B',
                      jac=True, options={'maxiter': maxiter, 'gtol': 1e-8})
    return result.x.reshape(-1, 3), result.fun


def _basin_hopping(positions, rho, n_iter=50, step_size=0.3, seed=0, time_limit=None):
    """Basin hopping with Metropolis acceptance: local minimize + random perturbation."""
    rng = np.random.default_rng(seed)
    best_pos = positions.copy()
    best_energy = _morse_energy_only(best_pos, rho)
    
    # First, relax the initial position
    best_pos, best_energy = _relax_lbfgs(best_pos, rho, maxiter=100)
    
    current_pos = best_pos.copy()
    current_energy = best_energy
    
    # Temperature schedule for Metropolis
    T0 = 0.5
    T_final = 0.01
    n_total = max(n_iter, 1)
    
    for i in range(n_iter):
        if time_limit is not None and time.time() > time_limit:
            break
        
        T = T0 * (T_final / T0) ** (i / n_total)
        
        perturbation = rng.normal(0, step_size, size=current_pos.shape)
        new_pos = current_pos + perturbation
        
        new_pos, new_energy = _relax_lbfgs(new_pos, rho, maxiter=50)
        
        delta_e = new_energy - current_energy
        if delta_e < 0 or rng.random() < np.exp(-delta_e / T):
            current_pos = new_pos
            current_energy = new_energy
        
        if new_energy < best_energy:
            best_energy = new_energy
            best_pos = new_pos.copy()
        
        if i % 10 == 0 and i > 0:
            step_size *= 0.92
    
    return best_pos, best_energy


def _run_single_bh(init_pos, rho, n_iter, step_size, seed, time_limit):
    """Run a single basin hopping chain. Returns (positions, energy)."""
    try:
        pos, energy = _basin_hopping(init_pos, rho, n_iter=n_iter, 
                                      step_size=step_size, seed=seed,
                                      time_limit=time_limit)
        return pos, energy
    except Exception:
        return None, np.inf


def construct(instance):
    """Find low-energy Morse cluster configurations using parallel multi-start basin hopping."""
    global START_TIME
    if START_TIME is None:
        START_TIME = time.time()
    
    size = instance["n"]
    rho = instance.get("rho", instance.get("kind", 10))
    
    # Check time budget
    elapsed = time.time() - START_TIME
    if elapsed > TOTAL_DEADLINE - 5:
        return _generate_bcc(size)
    
    deadline = START_TIME + TOTAL_DEADLINE
    
    # Determine which structural templates to use based on rho
    # Short range (rho=3,6): icosahedral motifs dominate
    # Long range (rho=10): crystalline motifs dominate
    if rho <= 6:
        # Short/medium range: icosahedral first, then crystalline
        generators = [
            ('ico', _generate_icosahedral),
            ('fcc', _generate_fcc),
            ('bcc', _generate_bcc),
            ('hcp', _generate_hcp),
            ('shell', _generate_shell),
        ]
    else:
        # Long range: crystalline first
        generators = [
            ('fcc', _generate_fcc),
            ('bcc', _generate_bcc),
            ('hcp', _generate_hcp),
            ('ico', _generate_icosahedral),
            ('shell', _generate_shell),
        ]
    
    # Generate initial configurations
    inits = []
    for name, gen_func in generators:
        try:
            init_pos = gen_func(size)
            # Quick relax
            relaxed_pos, relaxed_energy = _relax_lbfgs(init_pos, rho, maxiter=80)
            inits.append((name, relaxed_pos, relaxed_energy))
        except Exception:
            pass
    
    # Add some random starts for diversity
    for rs in range(3):
        try:
            rand_pos = _generate_random(size, seed=size + rs * 7 + 1)
            relaxed_pos, relaxed_energy = _relax_lbfgs(rand_pos, rho, maxiter=60)
            inits.append((f'rand{rs}', relaxed_pos, relaxed_energy))
        except Exception:
            pass
    
    # Sort by initial energy (best first)
    inits.sort(key=lambda x: x[2])
    
    best_pos = inits[0][1].copy() if inits else _generate_bcc(size)
    best_energy = inits[0][2] if inits else np.inf
    
    # Allocate time for parallel basin hopping
    remaining = deadline - time.time()
    if remaining < 10:
        # Not enough time, just return best relaxed
        _ensure_min_distance(best_pos, size)
        return best_pos
    
    # Determine number of parallel chains and iterations per chain
    # Use up to 4 threads
    n_chains = min(len(inits), 4)
    time_per_chain = remaining * 0.8 / n_chains
    
    # Create tasks for parallel execution
    tasks = []
    for idx in range(n_chains):
        name, init_pos, init_energy = inits[idx]
        # More iterations for better initial configs
        n_iter = max(10, int(time_per_chain / 0.15))  # ~0.15s per iteration estimate
        n_iter = min(n_iter, 60)
        step_size = 0.25 if idx == 0 else 0.3
        seed = size + idx * 100
        time_limit = time.time() + time_per_chain
        tasks.append((init_pos, rho, n_iter, step_size, seed, time_limit))
    
    # Run in parallel
    results = []
    with ThreadPoolExecutor(max_workers=min(4, n_chains)) as executor:
        futures = {
            executor.submit(_run_single_bh, *task): i 
            for i, task in enumerate(tasks)
        }
        for future in as_completed(futures):
            pos, energy = future.result()
            if pos is not None and energy < best_energy:
                best_energy = energy
                best_pos = pos.copy()
    
    # If we have time left, do more basin hopping from the best found
    remaining = deadline - time.time()
    if remaining > 15 and best_pos is not None:
        try:
            time_limit = deadline - 5
            bh_pos, bh_energy = _basin_hopping(best_pos, rho, n_iter=30, step_size=0.25, 
                                                seed=size + 1000, time_limit=time_limit)
            if bh_energy < best_energy:
                best_energy = bh_energy
                best_pos = bh_pos.copy()
        except Exception:
            pass
    
    # Final relaxation
    if best_pos is not None and time.time() < deadline - 5:
        try:
            best_pos, best_energy = _relax_lbfgs(best_pos, rho, maxiter=150)
        except Exception:
            pass
    
    # Ensure minimum distance constraint (0.5 apart)
    if best_pos is not None:
        _ensure_min_distance(best_pos, size)
    
    if best_pos is None:
        best_pos = _generate_bcc(size)
    
    return best_pos


def _ensure_min_distance(positions, n):
    """Ensure all pairs are at least 0.5 apart."""
    for iteration in range(10):  # Multiple passes to handle cascading fixes
        changed = False
        d = squareform(pdist(positions))
        np.fill_diagonal(d, 1.0)
        for i in range(n):
            for j in range(i + 1, n):
                if d[i, j] < 0.5 and d[i, j] > 1e-10:
                    direction = (positions[i] - positions[j]) / d[i, j]
                    shift = (0.5 - d[i, j]) / 2.0 + 0.01
                    positions[i] += direction * shift
                    positions[j] -= direction * shift
                    changed = True
        if not changed:
            break
