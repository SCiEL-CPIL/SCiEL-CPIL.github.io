"""Sutton-Chen cluster global minima via simulated annealing + diverse geometric seeds."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import concurrent.futures
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_START_TIME = time.time()
_DEADLINE = 320.0  # safety margin before 330s kill


def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)


def _icosa_shell(n, scale=1.0, seed=0):
    """Generate an icosahedral shell cluster with some randomization."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms

    rng = np.random.default_rng(seed)
    pts = [np.zeros(3)]

    shell_radii = [1.0, 1.55, 2.1, 2.65, 3.2, 3.75, 4.3, 4.85, 5.4]
    for r in shell_radii:
        if len(pts) >= n:
            break
        for v in verts:
            if len(pts) >= n:
                break
            p = r * v * scale
            ok = True
            for q in pts:
                if np.linalg.norm(p - q) < 0.38 * scale:
                    ok = False
                    break
            if ok:
                pts.append(p)

    # Fill remaining with random points on sphere shells
    while len(pts) < n:
        idx = min(len(pts) // 12, len(shell_radii) - 1)
        r = shell_radii[idx] * scale
        theta = rng.uniform(0, 2 * np.pi)
        cos_phi = rng.uniform(-1, 1)
        sin_phi = np.sqrt(1 - cos_phi**2)
        p = r * np.array([sin_phi * np.cos(theta), sin_phi * np.sin(theta), cos_phi])
        ok = True
        for q in pts:
            if np.linalg.norm(p - q) < 0.38 * scale:
                ok = False
                break
        if ok:
            pts.append(p)
        else:
            pts.append(p)

    return np.array(pts[:n])


def _fcc_cluster(n, scale=1.0, seed=0):
    """Generate an FCC-like cluster."""
    pts = []
    a = scale * 0.72
    max_idx = 7
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                for offset in [(0, 0, 0), (0, a/2, a/2), (a/2, 0, a/2), (a/2, a/2, 0)]:
                    p = np.array([i*a + offset[0], j*a + offset[1], k*a + offset[2]])
                    if np.linalg.norm(p) < 5.0 * scale:
                        pts.append(p)

    pts = np.array(pts)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)[:n]
    return pts[idx]


def _hcp_cluster(n, scale=1.0, seed=0):
    """Generate an HCP-like cluster."""
    pts = []
    a = scale * 0.72
    c = a * np.sqrt(8.0/3.0)
    max_idx = 6
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                if k % 2 == 0:
                    p = np.array([i*a, j*a*np.sqrt(3)/2, k*c/2])
                else:
                    p = np.array([i*a + a/2, j*a*np.sqrt(3)/2 + a*np.sqrt(3)/4, k*c/2])
                if np.linalg.norm(p) < 4.5 * scale:
                    pts.append(p)

    pts = np.array(pts)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)[:n]
    return pts[idx]


def _bcc_cluster(n, scale=1.0, seed=0):
    """Generate a BCC-like cluster."""
    pts = []
    a = scale * 0.75
    max_idx = 6
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                p = np.array([i*a, j*a, k*a])
                if np.linalg.norm(p) < 4.5 * scale:
                    pts.append(p)
                # Body center
                pc = np.array([i*a + a/2, j*a + a/2, k*a + a/2])
                if np.linalg.norm(pc) < 4.5 * scale:
                    pts.append(pc)

    pts = np.array(pts)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)[:n]
    return pts[idx]


def _random_compact(n, scale=1.0, seed=0):
    """Random compact cluster with minimum distance."""
    rng = np.random.default_rng(seed)
    radius = scale * (n ** (1.0/3.0)) * 0.9 + 1.0
    pts = []
    attempts = 0
    max_attempts = 200 * n
    while len(pts) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            ok = True
            for q in pts:
                if np.linalg.norm(p - q) < 0.45 * scale:
                    ok = False
                    break
            if ok:
                pts.append(p)
        attempts += 1

    while len(pts) < n:
        p = rng.uniform(-radius, radius, 3)
        pts.append(p)

    return np.array(pts[:n])


def _deformed_icosahedral(n, scale=1.0, seed=0):
    """Icosahedral with random deformations."""
    base = _icosa_shell(n, scale=scale, seed=seed)
    rng = np.random.default_rng(seed + 1000)
    noise = rng.normal(0, 0.15, base.shape)
    return base + noise


def _simulated_annealing(pos, kind, max_iters=300, t0=1.0, t_end=0.01, seed=0):
    """Simulated annealing with adaptive step size."""
    rng = np.random.default_rng(seed)
    energy, pos = relax(pos.copy(), 'sutton_chen', kind)
    best_energy = energy
    best_pos = pos.copy()

    n_atoms = pos.shape[0]
    # Adaptive step size based on cluster size
    base_step = 0.3 + 0.1 * np.sqrt(n_atoms) / 5.0

    for it in range(max_iters):
        if _time_left() < 5:
            break

        # Temperature schedule
        alpha = t_end / t0
        T = t0 * (alpha ** (it / max(1, max_iters - 1)))

        # Step size decreases with temperature
        step = base_step * (0.5 + 0.5 * (T / t0))

        # Generate perturbation
        delta = rng.normal(0, step, pos.shape)
        new_pos = pos + delta

        # Relax the perturbed structure
        new_energy, new_pos = relax(new_pos, 'sutton_chen', kind)

        # Acceptance criterion
        if new_energy < energy:
            accept = True
        else:
            dE = new_energy - energy
            if T > 1e-10:
                accept = rng.random() < np.exp(-dE / T)
            else:
                accept = False

        if accept:
            energy = new_energy
            pos = new_pos
            if energy < best_energy:
                best_energy = energy
                best_pos = pos.copy()

    return best_energy, best_pos


def _basin_hopping(pos, kind, max_iters=40, step_size=0.4, seed=0):
    """Basin hopping with Metropolis acceptance."""
    rng = np.random.default_rng(seed)
    energy, pos = relax(pos.copy(), 'sutton_chen', kind)
    best_energy = energy
    best_pos = pos.copy()

    for it in range(max_iters):
        if _time_left() < 5:
            break

        delta = rng.normal(0, step_size, pos.shape)
        new_pos = pos + delta
        new_energy, new_pos = relax(new_pos, 'sutton_chen', kind)

        if new_energy < energy:
            energy = new_energy
            pos = new_pos
            if energy < best_energy:
                best_energy = energy
                best_pos = pos.copy()
        else:
            dE = new_energy - energy
            if energy < 0 and dE < 5.0:
                if rng.random() < np.exp(-dE / 0.08):
                    pos = new_pos
                    energy = new_energy

    return best_energy, best_pos


def _worker(args):
    """Worker function for parallel evaluation of a seed."""
    pos, kind, method, seed, max_iters = args
    try:
        if method == 'sa':
            e, p = _simulated_annealing(pos, kind, max_iters=max_iters, seed=seed)
        else:
            e, p = _basin_hopping(pos, kind, max_iters=max_iters, seed=seed)
        if np.all(np.isfinite(p)) and e < np.inf:
            # Validate minimum distance
            from scipy.spatial.distance import pdist
            min_d = pdist(p)
            if len(min_d) > 0 and min_d.min() > 0.35:
                return e, p
            else:
                return np.inf, p
        return np.inf, p
    except Exception:
        return np.inf, p


def construct(instance):
    """Multi-method global optimization for Sutton-Chen clusters."""
    n = instance["n"]
    kind = instance.get("kind", "12-6")

    # Generate diverse starting structures
    seeds_list = []

    # Icosahedral variants
    for s in range(3):
        seeds_list.append(('icosa', _icosa_shell(n, scale=1.0, seed=s), s))
    # FCC
    seeds_list.append(('fcc', _fcc_cluster(n, scale=1.0), 100))
    # HCP
    seeds_list.append(('hcp', _hcp_cluster(n, scale=1.0), 200))
    # BCC
    seeds_list.append(('bcc', _bcc_cluster(n, scale=1.0), 300))
    # Deformed icosahedral
    for s in range(2):
        seeds_list.append(('def_icosa', _deformed_icosahedral(n, scale=1.0, seed=s), 400 + s))
    # Random compact
    for s in range(4):
        seeds_list.append(('rand', _random_compact(n, scale=1.0, seed=s), 500 + s))

    # Determine compute budget per seed
    # Total time budget ~320s, leave 20s for final relaxation
    # Use parallel workers for speed
    n_seeds = len(seeds_list)
    # Allocate iterations based on available time
    # Each relax call is roughly 0.1-0.5s depending on n
    # Estimate: for n~40-80, each relax ~0.2s, so ~5s per 25 iterations
    # With 320s and n_seeds seeds, each gets ~320/n_seeds seconds
    # But we run in parallel with up to 4 threads

    # First pass: quick relaxation of all seeds to get initial energies
    best_energy = np.inf
    best_pos = None

    for name, pos, seed in seeds_list:
        if _time_left() < 10:
            break
        try:
            energy, pos_relaxed = relax(pos.copy(), 'sutton_chen', kind)
            if np.all(np.isfinite(pos_relaxed)) and energy < best_energy:
                best_energy = energy
                best_pos = pos_relaxed.copy()
        except Exception:
            pass

    # Second pass: run SA and basin hopping on the best seeds in parallel
    # Pick top seeds by initial energy
    seed_energies = []
    for name, pos, seed in seeds_list:
        if _time_left() < 15:
            break
        try:
            energy, pos_relaxed = relax(pos.copy(), 'sutton_chen', kind)
            if np.all(np.isfinite(pos_relaxed)) and energy < np.inf:
                seed_energies.append((energy, pos_relaxed, name, seed))
        except Exception:
            pass

    # Sort by energy and take top seeds
    seed_energies.sort(key=lambda x: x[0])
    top_seeds = seed_energies[:min(6, len(seed_energies))]

    # Run SA and basin hopping on top seeds
    tasks = []
    for energy, pos, name, seed in top_seeds:
        if _time_left() < 20:
            break
        # SA with moderate iterations
        tasks.append((pos.copy(), kind, 'sa', seed, 150))
        # Basin hopping
        tasks.append((pos.copy(), kind, 'bh', seed + 1000, 30))

    # Run tasks in parallel (up to 4 threads)
    if tasks and _time_left() > 30:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(_worker, task): task for task in tasks}
            for future in concurrent.futures.as_completed(futures):
                if _time_left() < 10:
                    break
                try:
                    e, p = future.result(timeout=60)
                    if e < best_energy and np.all(np.isfinite(p)):
                        best_energy = e
                        best_pos = p.copy()
                except Exception:
                    pass

    # Final refinement: basin hopping from the best structure with smaller steps
    if best_pos is not None and _time_left() > 15:
        try:
            e, p = _basin_hopping(best_pos.copy(), kind, max_iters=40, step_size=0.2, seed=9999)
            if e < best_energy and np.all(np.isfinite(p)):
                best_energy = e
                best_pos = p.copy()
        except Exception:
            pass

    # One more SA refinement from best
    if best_pos is not None and _time_left() > 15:
        try:
            e, p = _simulated_annealing(best_pos.copy(), kind, max_iters=200, t0=0.5, t_end=0.005, seed=8888)
            if e < best_energy and np.all(np.isfinite(p)):
                best_energy = e
                best_pos = p.copy()
        except Exception:
            pass

    # Validate final result
    if best_pos is None or not np.all(np.isfinite(best_pos)):
        best_pos = _random_compact(n, scale=1.0, seed=7777)
        try:
            _, best_pos = relax(best_pos, 'sutton_chen', kind)
        except Exception:
            pass

    # Final check: minimum distance
    try:
        from scipy.spatial.distance import pdist
        min_d = pdist(best_pos)
        if len(min_d) > 0 and min_d.min() < 0.35:
            # Nudge atoms apart
            for _ in range(100):
                for i in range(len(best_pos)):
                    for j in range(i+1, len(best_pos)):
                        d = np.linalg.norm(best_pos[i] - best_pos[j])
                        if d < 0.35 and d > 1e-10:
                            direction = (best_pos[i] - best_pos[j]) / d
                            best_pos[i] += direction * 0.01
                            best_pos[j] -= direction * 0.01
                # Re-check
                min_d2 = pdist(best_pos)
                if len(min_d2) > 0 and min_d2.min() >= 0.35:
                    break
    except Exception:
        pass

    return best_pos
# EVOLVE-BLOCK-END
