"""Morse cluster global minima search with parallel basin hopping and diverse seeds."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

# ---------------------------------------------------------------------------
# Global time management (evaluator kills at 330 s wall-clock)
# ---------------------------------------------------------------------------
_GLOBAL_START = time.time()
_GLOBAL_DEADLINE = _GLOBAL_START + 320.0  # 10 s safety margin


def _time_left():
    return max(0.0, _GLOBAL_DEADLINE - time.time())


# ---------------------------------------------------------------------------
# Helper: Compute energy using standard formula (for validation/fallback)
# ---------------------------------------------------------------------------
def _morse_energy(pos, rho):
    n = pos.shape[0]
    diff = pos[:, None, :] - pos[None, :, :]
    r = np.sqrt(np.sum(diff * diff, axis=2))
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    r = r[mask]
    if r.size == 0:
        return 0.0
    u = np.exp(rho * (1.0 - r))
    return float(np.sum(u * (u - 2.0)))


# ---------------------------------------------------------------------------
# Local minimization using autoresearch_suite.relax
# ---------------------------------------------------------------------------
def _local_min(pos, rho, timeout=5.0):
    """
    Relax the structure using the provided relax function.
    Returns (positions, energy).
    """
    try:
        # relax function signature: relax(positions, 'morse', rho)
        # It returns the relaxed positions.
        relaxed_pos = relax(pos, 'morse', rho)
        # Ensure it's a numpy array of shape (n, 3)
        relaxed_pos = np.array(relaxed_pos)
        e = _morse_energy(relaxed_pos, rho)
        return relaxed_pos, e
    except Exception as e:
        # Fallback to original positions if relax fails
        return pos, _morse_energy(pos, rho)


# ---------------------------------------------------------------------------
# Seed generators
# ---------------------------------------------------------------------------
def _fcc_seed(n, spacing=1.0, rng=None):
    """Generate n atoms in FCC shell arrangement."""
    if rng is None:
        rng = np.random.default_rng(0)
    # FCC lattice points: all integer triples (i,j,k) with i+j+k even,
    # plus face centers. Simpler: generate a cubic grid of FCC sites.
    pts = []
    maxc = int(np.ceil((n * 2.0) ** (1.0 / 3.0))) + 3
    for i in range(-maxc, maxc + 1):
        for j in range(-maxc, maxc + 1):
            for k in range(-maxc, maxc + 1):
                if (i + j + k) % 2 == 0:
                    pts.append([i, j, k])
    pts = np.array(pts, dtype=float)
    # Center
    pts -= pts.mean(axis=0)
    # Sort by radius, take nearest n
    radii = np.linalg.norm(pts, axis=1)
    order = np.argsort(radii)
    pts = pts[order[:n]] * (spacing * 0.577)  # FCC nearest neighbor ~ sqrt(2)/2 * a
    return pts


def _hcp_seed(n, spacing=1.0, rng=None):
    """Generate n atoms in HCP-like layered arrangement."""
    if rng is None:
        rng = np.random.default_rng(1)
    pts = []
    # Layer 1: hexagonal packing
    layer = 0
    while len(pts) < n:
        # hexagonal layer with radius growing
        R = int(np.ceil(np.sqrt(max(1, n - len(pts)) / 3.0)))
        for i in range(-R, R + 1):
            for j in range(-R, R + 1):
                if i * i + j * j <= R * R:
                    x = i + 0.5 * (j % 2)
                    y = j * 0.866
                    z = layer * 0.816
                    pts.append([x, y, z])
        layer += 1
    pts = np.array(pts, dtype=float)
    pts -= pts.mean(axis=0)
    radii = np.linalg.norm(pts, axis=1)
    order = np.argsort(radii)
    pts = pts[order[:n]] * (spacing * 0.5)
    return pts


def _icosahedral_seed(n, spacing=1.0, rng=None):
    """Generate n atoms in icosahedral shell arrangement."""
    if rng is None:
        rng = np.random.default_rng(2)
    pts = [[0.0, 0.0, 0.0]]
    # 12 icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
    ], dtype=float)
    ico /= np.linalg.norm(ico, axis=1, keepdims=True)
    pts.extend(ico.tolist())
    # Second shell: 30 vertices (edges midpoints of icosahedron, normalized)
    if n > 13:
        verts = ico
        edges = []
        for a in range(12):
            for b in range(a + 1, 12):
                d = np.linalg.norm(verts[a] - verts[b])
                if 0.9 < d < 1.1:
                    mid = (verts[a] + verts[b]) / 2.0
                    mid /= np.linalg.norm(mid)
                    edges.append(mid)
        pts.extend([p.tolist() for p in edges])
    pts = np.array(pts, dtype=float)
    pts *= spacing * 0.5
    if len(pts) < n:
        # fill with random points on a sphere
        extra = n - len(pts)
        random_pts = rng.normal(size=(extra, 3))
        random_pts /= np.linalg.norm(random_pts, axis=1, keepdims=True)
        random_pts *= spacing * 0.7
        pts = np.vstack([pts, random_pts])
    else:
        pts = pts[:n]
    return pts


def _random_compact(n, spacing=1.0, rng=None):
    """Random compact cluster with minimum separation."""
    if rng is None:
        rng = np.random.default_rng(3)
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    points = []
    attempts = 0
    while len(points) < n and attempts < 20000:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.5 * spacing for q in points):
                points.append(p)
        attempts += 1
    # If we couldn't place all, add remaining at random
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    return np.array(points[:n], dtype=float)


# ---------------------------------------------------------------------------
# Basin hopping worker (for parallel execution)
# ---------------------------------------------------------------------------
def _basin_hop_worker(args):
    """
    Worker function for parallel basin hopping.
    args: (seed_pos, rho, n, budget, seed_id)
    """
    seed_pos, rho, n, budget, seed_id = args
    rng = np.random.default_rng(seed_id * 12345 + 6789)
    
    best_pos = seed_pos.copy()
    best_e = _morse_energy(seed_pos, rho)
    
    # Initial relaxation
    best_pos, best_e = _local_min(best_pos, rho)
    
    cur_pos = best_pos.copy()
    cur_e = best_e
    
    t0 = time.time()
    temp = 1.0
    kick_scale = 0.3
    step = 0
    
    while (time.time() - t0) < budget and _time_left() > 5.0:
        # Perturb
        kick = kick_scale * rng.normal(size=cur_pos.shape)
        new_pos = cur_pos + kick
        
        # Local minimize
        min_pos, min_e = _local_min(new_pos, rho)
        
        # Metropolis accept
        if min_e < cur_e:
            cur_pos, cur_e = min_pos, min_e
            if min_e < best_e:
                best_pos, best_e = min_pos, min_e
        else:
            delta = min_e - cur_e
            if delta < 0 or rng.random() < np.exp(-delta / max(temp, 1e-8)):
                cur_pos, cur_e = min_pos, min_e
        
        step += 1
        # Anneal temperature
        temp = max(0.05, 1.0 * (0.95 ** step))
        # Reduce kick scale slowly
        kick_scale = max(0.05, 0.3 * (0.98 ** step))
        
    return best_pos, best_e


# ---------------------------------------------------------------------------
# Main construct
# ---------------------------------------------------------------------------
def construct(instance):
    n = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind"))
    spacing = 1.0

    # Per-instance budget: distribute remaining time, but cap
    remaining = _time_left()
    inst_budget = min(remaining * 0.9, 50.0)  # cap per instance at 50 s
    inst_budget = max(inst_budget, 3.0)
    
    # If time is very low, just relax a single good seed
    if inst_budget < 5.0:
        seed = _fcc_seed(n, spacing)
        pos, e = _local_min(seed, rho)
        return pos

    rng = np.random.default_rng(n * 1000 + int(rho))

    # Build diverse candidate seeds
    seeds = []
    try:
        seeds.append(_fcc_seed(n, spacing, rng))
    except Exception:
        pass
    try:
        seeds.append(_hcp_seed(n, spacing, rng))
    except Exception:
        pass
    try:
        seeds.append(_icosahedral_seed(n, spacing, rng))
    except Exception:
        pass
    # Random compact seeds
    for k in range(2):
        try:
            seeds.append(_random_compact(n, spacing, rng))
        except Exception:
            pass
    
    if not seeds:
        seeds.append(np.random.default_rng(n).normal(size=(n, 3)))

    # Prepare parallel tasks
    # We will run basin hopping from each seed in parallel
    # Number of workers: up to 4 (task constraint)
    num_workers = min(4, len(seeds))
    
    # Adjust budget per worker
    # We want to use most of inst_budget
    # If we run N workers in parallel, each gets inst_budget time (wall clock)
    # But we need to account for overhead.
    # Let's give each worker a budget of inst_budget * 0.9 to be safe.
    worker_budget = inst_budget * 0.8
    
    tasks = []
    for i, seed in enumerate(seeds[:num_workers]):
        # Pass a copy of the seed
        tasks.append((seed.copy(), rho, n, worker_budget, i))
        
    best_pos = None
    best_e = np.inf
    
    # Use ProcessPoolExecutor for true parallelism
    # Note: mp.get_context('fork') is safer on Linux, 'spawn' on Windows.
    # We'll use the default which is usually fine, or specify fork if available.
    try:
        ctx = mp.get_context('fork')
    except:
        ctx = mp.get_context()
        
    with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as executor:
        futures = []
        for task in tasks:
            futures.append(executor.submit(_basin_hop_worker, task))
            
        for future in as_completed(futures):
            if _time_left() < 2.0:
                break
            try:
                pos, e = future.result()
                if e < best_e:
                    best_e = e
                    best_pos = pos
            except Exception as e:
                pass

    # If no result from parallel (e.g. timeout or error), fallback to sequential relaxation of first seed
    if best_pos is None:
        seed = seeds[0]
        best_pos, best_e = _local_min(seed, rho)
        
    # Enforce minimum separation 0.5 (light cleanup)
    best_pos = _enforce_min_sep(best_pos, 0.5, rng)
    return best_pos


def _enforce_min_sep(pos, min_d, rng):
    """Nudge atoms that are too close apart."""
    n = pos.shape[0]
    for _ in range(20):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                d = pos[i] - pos[j]
                r = np.linalg.norm(d)
                if r < min_d:
                    if r < 1e-8:
                        d = rng.normal(size=3)
                        r = np.linalg.norm(d)
                    push = (min_d - r) * 0.5 * (d / r)
                    pos[i] += push
                    pos[j] -= push
                    moved = True
        if not moved:
            break
    return pos
# EVOLVE-BLOCK-END
