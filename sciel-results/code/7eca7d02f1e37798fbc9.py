# EVOLVE-BLOCK-START
import numpy as np
import time
import concurrent.futures
import math
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

TOTAL_DEADLINE = 295.0
_START_TIME = time.time()

def _time_left():
    return max(0.0, TOTAL_DEADLINE - (time.time() - _START_TIME))

def _morse_energy_fast(positions, rho):
    """Vectorized Morse energy."""
    n = len(positions)
    if n < 2:
        return 0.0
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    x = rho * (1.0 - dists)
    ex = np.exp(x)
    e = ex * (ex - 2.0)
    iu = np.triu_indices(n, k=1)
    return np.sum(e[iu])

def _morse_gradient_fast(positions, rho):
    """Vectorized Morse gradient."""
    n = len(positions)
    if n < 2:
        return np.zeros_like(positions)
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    x = rho * (1.0 - dists)
    ex = np.exp(x)
    # dE/dr_ij = 2*rho*ex*(ex-1)
    # dE/dpos_i = sum_j dE/dr_ij * (pos_i - pos_j)/r_ij
    dE_dr = 2.0 * rho * ex * (ex - 1.0)
    np.fill_diagonal(dE_dr, 0.0)
    # Avoid division by zero
    safe_dists = np.where(dists > 1e-10, dists, 1e-10)
    grad = np.zeros_like(positions)
    for i in range(n):
        for j in range(i+1, n):
            factor = dE_dr[i, j] / safe_dists[i, j]
            grad[i] += factor * diff[i, j]
            grad[j] -= factor * diff[i, j]
    return grad

def _enforce_min_distance(positions, min_dist=0.5):
    """Nudge atoms apart if they are too close."""
    n = len(positions)
    positions = positions.copy()
    for _ in range(10):
        moved = False
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(positions[i] - positions[j])
                if d < min_dist:
                    if d < 1e-10:
                        diff = np.array([0.01, 0, 0])
                    else:
                        diff = (positions[i] - positions[j]) / d
                    shift = (min_dist - d + 0.01) / 2.0
                    positions[i] += diff * shift
                    positions[j] -= diff * shift
                    moved = True
        if not moved:
            break
    return positions

def _make_fcc_cluster(n, spacing=1.0):
    side = int(np.ceil(n ** (1/3))) + 2
    points = []
    a = spacing * np.sqrt(2)
    basis = np.array([[0,0,0], [0.5,0.5,0], [0.5,0,0.5], [0,0.5,0.5]]) * a
    for ix in range(side):
        for iy in range(side):
            for iz in range(side):
                for b in basis:
                    pt = np.array([ix, iy, iz]) * a + b
                    points.append(pt)
    points = np.array(points)
    center = points.mean(axis=0)
    points -= center
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)[:n]
    return points[idx]

def _make_bcc_cluster(n, spacing=1.0):
    side = int(np.ceil(n ** (1/3))) + 2
    points = []
    a = spacing * (2.0 / np.sqrt(3.0))
    basis = np.array([[0,0,0], [0.5,0.5,0.5]]) * a
    for ix in range(side):
        for iy in range(side):
            for iz in range(side):
                for b in basis:
                    pt = np.array([ix, iy, iz]) * a + b
                    points.append(pt)
    points = np.array(points)
    center = points.mean(axis=0)
    points -= center
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)[:n]
    return points[idx]

def _make_hcp_cluster(n, spacing=1.0):
    a = spacing
    c = a * np.sqrt(8/3)
    points = []
    radius = int(np.ceil(np.sqrt(n / 3))) + 2
    for layer in range(2 * radius + 1):
        z = (layer - radius) * c
        for q in range(-radius, radius + 1):
            for r in range(max(-radius, -q - radius), min(radius, -q + radius) + 1):
                x = a * (q + r/2)
                y = a * (np.sqrt(3)/2) * r
                if layer % 2 == 0:
                    points.append([x, y, z])
                else:
                    points.append([x + a/2, y + a*np.sqrt(3)/6, z])
    points = np.array(points)
    center = points.mean(axis=0)
    points -= center
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)[:n]
    return points[idx]

def _make_icosahedron_vertices():
    """Return 12 vertices of a regular icosahedron."""
    phi = (1 + np.sqrt(5)) / 2
    v = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    # Normalize to unit sphere
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    v = v / norms
    return v

def _make_icosahedral_cluster(n, spacing=1.0):
    """Build an icosahedral cluster: 1 center + 12 shell + more shells."""
    if n == 1:
        return np.array([[0.0, 0.0, 0.0]])
    
    vertices = _make_icosahedron_vertices()
    # Scale so nearest neighbor distance ~ spacing
    # Edge length of icosahedron with circumradius R: edge = R * 4/phi
    # We want edge ~ spacing, so R = spacing * phi / 4
    R1 = spacing * (1 + np.sqrt(5)) / 4.0
    
    points = [np.array([0.0, 0.0, 0.0])]
    for v in vertices:
        points.append(v * R1)
    
    # If we need more atoms, add a second shell (octahedral directions)
    if n > 13:
        octa = np.array([
            [1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1]
        ], dtype=float)
        # Scale octahedral shell
        R2 = R1 * np.sqrt(2) * 0.95  # slightly inside to avoid overlap
        for v in octa:
            if len(points) >= n:
                break
            points.append(v * R2)
    
    # If still need more, add jittered random points near the surface
    if len(points) < n:
        rng = np.random.default_rng(123)
        while len(points) < n:
            # Pick a random direction
            d = rng.normal(0, 1, 3)
            d = d / np.linalg.norm(d)
            # Place at radius slightly larger than first shell
            r = R1 * (1.0 + 0.1 * rng.random())
            points.append(d * r)
    
    points = np.array(points[:n])
    return points

def _make_decahedral_cluster(n, spacing=1.0):
    """Build a decahedral cluster (twinned fcc)."""
    # Start with fcc and create twinned structure
    base = _make_fcc_cluster(n, spacing)
    # Apply a twinning operation: reflect across a plane and merge
    # Simple approach: create two fcc fragments joined at a twin plane
    half = n // 2
    frag1 = _make_fcc_cluster(half, spacing)
    frag2 = _make_fcc_cluster(n - half, spacing)
    # Reflect frag2 and offset
    frag2 = frag2 * np.array([1, 1, -1])
    frag2 += np.array([0, 0, 0.5])
    combined = np.vstack([frag1, frag2])
    # Center
    combined -= combined.mean(axis=0)
    return combined[:n]

def _make_random_compact(n, rng, spacing=1.0):
    radius = spacing * n ** (1/3) * 1.2
    points = []
    attempts = 0
    while len(points) < n and attempts < 5000:
        attempts += 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if len(points) == 0:
                points.append(p)
            else:
                dists = np.linalg.norm(np.array(points) - p, axis=1)
                if np.min(dists) >= 0.8 * spacing:
                    points.append(p)
    if len(points) < n:
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            points.append(p)
    return np.array(points)

def _make_spherical_shell(n, spacing=1.0):
    points = []
    if n == 1:
        return np.array([[0,0,0]])
    golden = (1 + np.sqrt(5)) / 2
    for i in range(n):
        z = 1 - 2 * (i + 0.5) / n
        r = np.sqrt(1 - z*z)
        phi = 2 * np.pi * i / golden
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        points.append([x, y, z])
    points = np.array(points)
    scale = 1.5 * (n ** (1/3)) / 3.0 
    points *= scale
    return points

def _lbfgs_refine(pos, rho, max_iter=100, time_budget=5.0):
    """Refine a configuration using L-BFGS-B with explicit gradient."""
    x0 = pos.flatten()
    
    def objective(x):
        positions = x.reshape(-1, 3)
        return _morse_energy_fast(positions, rho)
    
    def gradient(x):
        positions = x.reshape(-1, 3)
        return _morse_gradient_fast(positions, rho).flatten()
    
    try:
        start_t = time.time()
        result = minimize(
            objective, x0,
            jac=gradient,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'maxfun': 500, 'ftol': 1e-10, 'gtol': 1e-5}
        )
        elapsed = time.time() - start_t
        if elapsed > time_budget:
            return np.inf, pos
        
        if result.success or result.fun < np.inf:
            refined = result.x.reshape(-1, 3)
            refined = _enforce_min_distance(refined, 0.5)
            final_e = _morse_energy_fast(refined, rho)
            return final_e, refined
    except Exception:
        pass
    return np.inf, pos

def _simulated_annealing(pos, rho, n_atoms, t_init=50.0, t_final=0.1, n_iter=3000,
                          move_scale=0.3, time_budget=40.0, rng=None):
    """
    Simulated annealing with L-BFGS local minimization.
    Uses Metropolis acceptance: accept worse states with probability exp(-dE/T).
    Periodically runs L-BFGS to escape shallow local minima.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    current_pos = pos.copy()
    current_e = _morse_energy_fast(current_pos, rho)
    best_pos = current_pos.copy()
    best_e = current_e
    
    n = n_atoms
    # Temperature schedule: geometric cooling
    cooling = (t_final / t_init) ** (1.0 / max(n_iter, 1))
    T = t_init
    
    # Track for periodic L-BFGS
    lbfqs_interval = max(20, n_iter // 15)  # every ~200 iters
    lbfqs_count = 0
    
    start_time = time.time()
    no_improve = 0
    max_no_improve = 500
    
    for it in range(n_iter):
        # Time check
        elapsed = time.time() - start_time
        if elapsed > time_budget:
            break
        if _time_left() < 10.0:
            break
        
        # Generate a random perturbation
        # Move 1-3 random atoms by a random displacement
        n_moves = 1 + rng.integers(0, 3)
        idx = rng.integers(0, n, size=n_moves)
        displacement = rng.normal(0, move_scale, size=(n_moves, 3))
        
        new_pos = current_pos.copy()
        new_pos[idx] += displacement
        
        # Enforce minimum distance
        new_pos = _enforce_min_distance(new_pos, 0.5)
        
        new_e = _morse_energy_fast(new_pos, rho)
        
        # Metropolis acceptance
        dE = new_e - current_e
        if dE < 0 or (T > 1e-10 and rng.random() < math.exp(-dE / T)):
            current_pos = new_pos
            current_e = new_e
            no_improve = 0
            
            if current_e < best_e:
                best_e = current_e
                best_pos = current_pos.copy()
        else:
            no_improve += 1
        
        # Cool down
        T *= cooling
        
        # Periodic L-BFGS refinement to escape local minima
        if (it + 1) % lbfqs_interval == 0:
            e_refined, pos_refined = _lbfgs_refine(current_pos, rho, max_iter=80, time_budget=3.0)
            if e_refined < current_e:
                current_pos = pos_refined
                current_e = e_refined
                if current_e < best_e:
                    best_e = current_e
                    best_pos = current_pos.copy()
                no_improve = 0
            lbfqs_count += 1
        
        # If stuck too long, do a larger perturbation
        if no_improve > max_no_improve:
            # Big kick: perturb many atoms
            kick_idx = rng.integers(0, n, size=max(3, n // 4))
            current_pos[kick_idx] += rng.normal(0, 0.5, size=(len(kick_idx), 3))
            current_pos = _enforce_min_distance(current_pos, 0.5)
            current_e = _morse_energy_fast(current_pos, rho)
            no_improve = 0
            # Reset temperature slightly to allow escape
            T = max(T, 5.0)
        
        # Reduce move scale as we cool
        if T < 5.0:
            move_scale = 0.15
        elif T < 20.0:
            move_scale = 0.25
    
    return best_e, best_pos

def construct(instance):
    """
    Find low-energy Morse cluster configurations using simulated annealing
    with L-BFGS local minimization from diverse seeds.
    """
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind", 10))
    
    best_energy = np.inf
    best_positions = None
    
    rng = np.random.default_rng(42)
    
    initial_configs = []
    
    # Physically-motivated seeds
    initial_configs.append(_make_icosahedral_cluster(n))
    initial_configs.append(_make_decahedral_cluster(n))
    initial_configs.append(_make_fcc_cluster(n))
    initial_configs.append(_make_bcc_cluster(n))
    initial_configs.append(_make_hcp_cluster(n))
    initial_configs.append(_make_spherical_shell(n))
    
    # Random compact
    initial_configs.append(_make_random_compact(n, rng))
    
    # Jittered versions of key seeds
    for base in initial_configs[:3]:
        jitter = rng.normal(0, 0.15, size=base.shape)
        initial_configs.append(base + jitter)
    
    # More random starts
    for i in range(2):
        initial_configs.append(_make_random_compact(n, rng))
    
    # Shuffle
    rng.shuffle(initial_configs)
    
    time_limit = _time_left()
    if time_limit < 5:
        for pos in initial_configs:
            e = _morse_energy_fast(pos, rho)
            if e < best_energy:
                best_energy = e
                best_positions = pos
        return _enforce_min_distance(best_positions, 0.5)
    
    # Phase 1: Quick L-BFGS relaxation of all seeds to get baseline
    for pos in initial_configs:
        if _time_left() < 10.0:
            break
        e, refined = _lbfgs_refine(pos, rho, max_iter=80, time_budget=3.0)
        if e < best_energy:
            best_energy = e
            best_positions = refined.copy()
    
    if best_positions is None:
        best_positions = initial_configs[0]
        best_energy = _morse_energy_fast(best_positions, rho)
    
    # Phase 2: Simulated annealing from the best seed and a few others
    # Use remaining time for SA runs
    sa_time_budget = min(_time_left() * 0.7, 200.0)
    
    # Pick top 3 seeds for SA
    seed_energies = []
    for pos in initial_configs:
        e = _morse_energy_fast(pos, rho)
        seed_energies.append((e, pos))
    seed_energies.sort(key=lambda x: x[0])
    
    # Run SA from top seeds, distributing time
    n_sa_runs = min(3, len(seed_energies))
    time_per_sa = sa_time_budget / n_sa_runs
    
    for i in range(n_sa_runs):
        if _time_left() < 15.0:
            break
        
        e_seed, pos_seed = seed_energies[i]
        
        # Scale n_iter based on available time
        # Estimate: each SA iteration takes ~0.5ms for small n, more for large
        est_time_per_iter = 0.001 * (n / 40.0)  # rough estimate
        est_iters = int(time_per_sa / (est_time_per_iter * 2))  # factor 2 for L-BFGS overhead
        est_iters = max(500, min(est_iters, 10000))
        
        e_sa, pos_sa = _simulated_annealing(
            pos_seed, rho, n,
            t_init=30.0, t_final=0.05,
            n_iter=est_iters,
            move_scale=0.3,
            time_budget=time_per_sa,
            rng=rng
        )
        
        if e_sa < best_energy:
            best_energy = e_sa
            best_positions = pos_sa.copy()
    
    # Phase 3: Final L-BFGS polish on best result
    if _time_left() > 5.0:
        e_final, pos_final = _lbfgs_refine(best_positions, rho, max_iter=200, time_budget=min(_time_left() - 3.0, 10.0))
        if e_final < best_energy:
            best_energy = e_final
            best_positions = pos_final
    
    best_positions = _enforce_min_distance(best_positions, 0.5)
    return best_positions
# EVOLVE-BLOCK-END
