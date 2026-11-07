"""Program for Sutton-Chen TM cluster global minima search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_START_TIME = time.time()
_DEADLINE = 325.0  # Safety margin before 330s kill

_lock = threading.Lock()
_shared_best = {"energy": np.inf, "positions": None}


def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)


def _is_valid(positions, min_dist=0.35):
    """Check if positions are valid (finite, no atoms closer than min_dist)."""
    if not np.all(np.isfinite(positions)):
        return False
    n = len(positions)
    for i in range(n - 1):
        d = np.linalg.norm(positions[i + 1:] - positions[i], axis=1)
        if np.any(d < min_dist):
            return False
    return True


def _generate_fcc(n):
    """Generate n atoms in FCC-like arrangement (closest to origin)."""
    max_r = int(np.ceil(n ** (1 / 3))) + 4
    candidates = []
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            for k in range(-max_r, max_r + 1):
                if (i + j + k) % 2 == 0:
                    p = np.array([i, j, k], dtype=float) / np.sqrt(2.0)
                    candidates.append((np.dot(p, p), p))
    candidates.sort(key=lambda x: x[0])
    pts = [p for _, p in candidates[:n]]
    return np.array(pts)


def _generate_bcc(n):
    """Generate n atoms in BCC-like arrangement."""
    max_r = int(np.ceil(n ** (1 / 3))) + 4
    candidates = []
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            for k in range(-max_r, max_r + 1):
                p1 = np.array([i, j, k], dtype=float)
                candidates.append((np.dot(p1, p1), p1))
                p2 = p1 + 0.5
                candidates.append((np.dot(p2, p2), p2))
    candidates.sort(key=lambda x: x[0])
    pts = [p for _, p in candidates[:n]]
    return np.array(pts)


def _generate_icosahedral(n):
    """Generate icosahedral shell structure."""
    phi = (1 + np.sqrt(5)) / 2
    v = [
        np.array([0, 1, phi]), np.array([0, 1, -phi]),
        np.array([0, -1, phi]), np.array([0, -1, -phi]),
        np.array([1, phi, 0]), np.array([1, -phi, 0]),
        np.array([-1, phi, 0]), np.array([-1, -phi, 0]),
        np.array([phi, 0, 1]), np.array([phi, 0, -1]),
        np.array([-phi, 0, 1]), np.array([-phi, 0, -1])
    ]
    v = [p / np.linalg.norm(p) for p in v]
    
    pts = [np.zeros(3)]
    if n > 1:
        pts.extend(v[:min(12, n - 1)])
    
    if n > 13:
        # Shell 2: 42 atoms (30 edge midpoints + 12 face centers)
        edges = []
        for i in range(12):
            for j in range(i + 1, 12):
                d = np.linalg.norm(v[i] - v[j])
                if 1.5 < d < 2.5:
                    edges.append((i, j))
        
        shell2_pts = []
        for i, j in edges[:30]:
            mid = (v[i] + v[j]) / 2
            mid = mid * 2.0 / np.linalg.norm(mid)
            shell2_pts.append(mid)
        
        faces = []
        for i in range(12):
            for j in range(i + 1, 12):
                for k in range(j + 1, 12):
                    d1 = np.linalg.norm(v[i] - v[j])
                    d2 = np.linalg.norm(v[j] - v[k])
                    d3 = np.linalg.norm(v[i] - v[k])
                    if 1.5 < d1 < 2.5 and 1.5 < d2 < 2.5 and 1.5 < d3 < 2.5:
                        center = (v[i] + v[j] + v[k]) / 3
                        center = center * 2.0 / np.linalg.norm(center)
                        faces.append(center)
        
        shell2_pts.extend(faces[:12])
        
        unique_pts = []
        for p in shell2_pts:
            if not any(np.linalg.norm(p - q) < 0.1 for q in unique_pts):
                unique_pts.append(p)
        
        pts.extend(unique_pts[:min(42, n - len(pts))])
    
    if n > 55:
        # Shell 3: use random points on sphere of radius 3
        rng = np.random.default_rng(123)
        remaining = min(92, n - len(pts))
        for _ in range(remaining * 5):
            if len(pts) >= n:
                break
            p = rng.normal(size=3)
            p = p * 3.0 / np.linalg.norm(p)
            if all(np.linalg.norm(p - q) > 0.5 for q in pts):
                pts.append(p)
    
    return np.array(pts[:n])


def _generate_random(n, seed):
    """Generate random positions in a sphere."""
    rng = np.random.default_rng(seed)
    r = 0.4 * np.sqrt(n)
    pts = rng.normal(size=(n, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    pts = pts / norms * r * rng.random((n, 1)) ** (1 / 3)
    return pts


def _generate_deformed(n, seed, base_type='fcc'):
    """Generate deformed template with random perturbations."""
    rng = np.random.default_rng(seed)
    if base_type == 'fcc':
        pts = _generate_fcc(n)
    elif base_type == 'bcc':
        pts = _generate_bcc(n)
    else:
        pts = _generate_icosahedral(n)
    # Add random deformation
    sigma = 0.05 + 0.1 * rng.random()
    pts = pts + rng.normal(0, sigma, (n, 3))
    return pts


def _basin_hopping_worker(positions, kind, n_trials, sigma_start, worker_id):
    """
    Basin hopping worker: perturb, relax, keep if lower energy.
    Runs in a thread and updates shared best.
    """
    rng = np.random.default_rng(54321 + worker_id * 7919 + len(positions))
    n = len(positions)
    
    try:
        e, current = relax(positions, 'sutton_chen', kind)
    except Exception:
        return
    
    if not (np.isfinite(e) and _is_valid(current)):
        return
    
    local_best_e = e
    local_best_pos = current.copy()
    
    for i in range(n_trials):
        if _time_left() < 5.0:
            break
        
        sigma = sigma_start * (0.85 ** i)
        pos = current + rng.normal(0, sigma, (n, 3))
        
        if not _is_valid(pos):
            continue
        
        try:
            e_new, relaxed = relax(pos, 'sutton_chen', kind)
            if np.isfinite(e_new) and _is_valid(relaxed):
                if e_new < local_best_e:
                    local_best_e = e_new
                    local_best_pos = relaxed.copy()
                    # Update shared best
                    with _lock:
                        if e_new < _shared_best["energy"]:
                            _shared_best["energy"] = e_new
                            _shared_best["positions"] = relaxed.copy()
                # Accept for basin hopping
                current = relaxed
        except Exception:
            continue
    
    # Final check on local best
    with _lock:
        if local_best_e < _shared_best["energy"]:
            _shared_best["energy"] = local_best_e
            _shared_best["positions"] = local_best_pos


def _generate_starts(n, kind, n_starts=12):
    """
    Generate diverse starting configurations for multi-start basin hopping.
    Mix of templates, deformed templates, and random.
    """
    starts = []
    
    # Template starts
    templates = [_generate_fcc, _generate_bcc, _generate_icosahedral]
    for gen in templates:
        try:
            pts = gen(n)
            if _is_valid(pts):
                starts.append(pts)
        except Exception:
            pass
    
    # Deformed template starts
    rng = np.random.default_rng(42)
    for i in range(min(4, n_starts - len(starts))):
        base_type = ['fcc', 'bcc', 'ico'][i % 3]
        try:
            pts = _generate_deformed(n, seed=100 + i, base_type=base_type)
            if _is_valid(pts):
                starts.append(pts)
        except Exception:
            pass
    
    # Random starts
    for i in range(min(4, n_starts - len(starts))):
        try:
            pts = _generate_random(n, seed=200 + i * 13 + n)
            if _is_valid(pts):
                starts.append(pts)
        except Exception:
            pass
    
    # Pad with more random if needed
    i = 0
    while len(starts) < n_starts:
        try:
            pts = _generate_random(n, seed=300 + i * 7 + n)
            if _is_valid(pts):
                starts.append(pts)
        except Exception:
            pass
        i += 1
        if i > 20:
            break
    
    return starts[:n_starts]


def construct(instance):
    n = instance["n"]
    kind = instance["kind"]
    
    # Reset shared state
    with _lock:
        _shared_best["energy"] = np.inf
        _shared_best["positions"] = None
    
    # Phase 1: Quick template relaxation to get initial best
    generators = [_generate_fcc, _generate_bcc, _generate_icosahedral]
    initial_best = None
    
    for gen in generators:
        if _time_left() < 10.0:
            break
        try:
            base = gen(n)
            if not _is_valid(base):
                base = base * 1.3
            if _is_valid(base):
                energy, relaxed = relax(base, 'sutton_chen', kind)
                if np.isfinite(energy) and _is_valid(relaxed):
                    with _lock:
                        if energy < _shared_best["energy"]:
                            _shared_best["energy"] = energy
                            _shared_best["positions"] = relaxed.copy()
                    initial_best = relaxed
        except Exception:
            continue
    
    # Phase 2: Concurrent multi-start basin hopping
    if _time_left() > 30.0:
        starts = _generate_starts(n, kind, n_starts=12)
        
        # Determine trials per worker based on time budget
        time_budget = max(10.0, _time_left() - 15.0)
        # Estimate: each BH trial takes ~0.5-2s for n~50-80
        # With 4 threads, we can do ~time_budget/1.5 * 4 trials total
        est_trials_per_worker = max(8, int(time_budget / 1.5))
        
        # Launch concurrent workers
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for i, start in enumerate(starts):
                if _time_left() < 5.0:
                    break
                futures.append(executor.submit(
                    _basin_hopping_worker, start, kind, est_trials_per_worker, 0.15, i
                ))
            
            # Wait for all to complete (or timeout)
            for f in as_completed(futures, timeout=max(5.0, _time_left() - 5.0)):
                try:
                    f.result()
                except Exception:
                    pass
    
    # Phase 3: Polish the best found with more basin hopping
    with _lock:
        best_energy = _shared_best["energy"]
        best_pos = _shared_best["positions"]
    
    if best_pos is not None and _time_left() > 20.0:
        # Additional basin hopping from the best
        rng = np.random.default_rng(99999 + n)
        n_polish = max(10, int(_time_left() / 2.0))
        current = best_pos.copy()
        for i in range(n_polish):
            if _time_left() < 5.0:
                break
            sigma = 0.12 * (0.88 ** i)
            pos = current + rng.normal(0, sigma, (n, 3))
            if not _is_valid(pos):
                continue
            try:
                e_new, relaxed = relax(pos, 'sutton_chen', kind)
                if np.isfinite(e_new) and _is_valid(relaxed):
                    if e_new < best_energy:
                        best_energy = e_new
                        best_pos = relaxed.copy()
                        with _lock:
                            _shared_best["energy"] = best_energy
                            _shared_best["positions"] = best_pos.copy()
                    current = relaxed
            except Exception:
                continue
    
    # Fallback
    if best_pos is None:
        best_pos = _generate_fcc(n)
        if not _is_valid(best_pos):
            best_pos = _generate_bcc(n)
        if not _is_valid(best_pos):
            rng2 = np.random.default_rng(999)
            best_pos = rng2.uniform(-3, 3, (n, 3))
            for attempt in range(50):
                if _is_valid(best_pos):
                    break
                best_pos = rng2.uniform(-3, 3, (n, 3))
    
    return best_pos

# EVOLVE-BLOCK-END
