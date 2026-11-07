"""Sutton-Chen transition-metal cluster global minima: growth + basin-hopping hybrid."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import cdist
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_START_TIME = None
_TIME_LIMIT = 295.0

def _elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME

def _time_left():
    return _TIME_LIMIT - _elapsed()

def _min_dist(positions):
    d = cdist(positions, positions)
    np.fill_diagonal(d, np.inf)
    return float(np.min(d))

def _valid(pos):
    if pos is None or len(pos) == 0:
        return False
    if not np.all(np.isfinite(pos)):
        return False
    return _min_dist(pos) >= 0.35

# --- Structural seeds ---
def _ico_shells():
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
    ], dtype=float)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if d < 0.8:
                mid = (verts[i] + verts[j]) / 2.0
                mid = mid / np.linalg.norm(mid)
                edges.append(mid)
    edges_u = []
    for e in edges:
        if not any(np.allclose(e, u, atol=1e-6) for u in edges_u):
            edges_u.append(e)
    
    faces = []
    for i in range(12):
        for j in range(i+1, 12):
            for k in range(j+1, 12):
                d1 = np.linalg.norm(verts[i] - verts[j])
                d2 = np.linalg.norm(verts[j] - verts[k])
                d3 = np.linalg.norm(verts[k] - verts[i])
                if d1 < 0.8 and d2 < 0.8 and d3 < 0.8:
                    fc = (verts[i] + verts[j] + verts[k]) / 3.0
                    fc = fc / np.linalg.norm(fc)
                    faces.append(fc)
    faces_u = []
    for f in faces:
        if not any(np.allclose(f, u, atol=1e-6) for f in faces_u):
            faces_u.append(f)
    
    return verts, edges_u, faces_u

def _make_mackay(n, spacing=0.72):
    verts, edges, faces = _ico_shells()
    r1 = spacing * 0.6
    r2 = r1 * np.sqrt(2.0)
    r3 = r1 * np.sqrt(3.0)
    r4 = r1 * 2.0
    points = [np.zeros(3)]
    for v in verts:
        points.append(v * r1)
    for e in edges:
        points.append(e * r2)
    for f in faces:
        points.append(f * r3)
    if n <= len(points):
        return np.array(points[:n])
    remaining = n - len(points)
    golden = (1.0 + np.sqrt(5.0)) / 2.0
    for i in range(remaining):
        y = 1.0 - (i + 0.5) / remaining * 2.0
        r = np.sqrt(max(0.0, 1.0 - y * y))
        theta = 2.0 * np.pi * golden * i
        p = np.array([r * np.cos(theta), y, r * np.sin(theta)])
        points.append(p * r4)
    return np.array(points[:n])

def _make_fcc(n, spacing=0.72):
    a = spacing
    r_max = spacing * n ** (1.0/3.0) * 0.85
    max_idx = int(r_max / (a * 0.5)) + 1
    candidates = []
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                if (i + j + k) % 2 == 0:
                    p = np.array([i, j, k]) * a * 0.5
                    if np.linalg.norm(p) <= r_max:
                        candidates.append(p)
                else:
                    p = (np.array([i, j, k]) + np.array([1, 1, 1])) * a * 0.25
                    if np.linalg.norm(p) <= r_max:
                        candidates.append(p)
    if not candidates:
        return None
    candidates = np.array(candidates)
    d2 = np.sum(candidates**2, axis=1)
    order = np.argsort(d2)
    return candidates[order[:n]]

def _make_bcc(n, spacing=0.72):
    a = spacing
    r_max = spacing * n ** (1.0/3.0) * 0.85
    max_idx = int(r_max / a) + 1
    candidates = []
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                p = np.array([i, j, k]) * a
                if np.linalg.norm(p) <= r_max:
                    candidates.append(p)
                p2 = (np.array([i, j, k]) + np.array([0.5, 0.5, 0.5])) * a
                if np.linalg.norm(p2) <= r_max:
                    candidates.append(p2)
    if not candidates:
        return None
    candidates = np.array(candidates)
    d2 = np.sum(candidates**2, axis=1)
    order = np.argsort(d2)
    return candidates[order[:n]]

def _make_random_compact(n, spacing=0.72, seed=None, min_dist=0.5):
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0/3.0) * 0.85
    points = []
    max_attempts = 3000
    for _ in range(max_attempts):
        if len(points) >= n:
            break
        batch_size = min(100, n - len(points))
        batch = rng.uniform(-radius, radius, (batch_size, 3))
        batch = batch[np.linalg.norm(batch, axis=1) <= radius]
        for p in batch:
            if len(points) >= n:
                break
            if not points:
                points.append(p)
            else:
                arr = np.array(points)
                dists = np.linalg.norm(arr - p, axis=1)
                if np.min(dists) >= min_dist:
                    points.append(p)
    if len(points) < n:
        for _ in range(max_attempts):
            if len(points) >= n:
                break
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) > radius:
                continue
            if not points:
                points.append(p)
            else:
                arr = np.array(points)
                dists = np.linalg.norm(arr - p, axis=1)
                if np.min(dists) >= 0.36:
                    points.append(p)
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    return np.array(points[:n])

def _make_deformed(base_positions, spacing=0.72, seed=None, scale=0.05):
    rng = np.random.default_rng(seed)
    positions = base_positions.copy()
    noise = rng.normal(0, scale * spacing, positions.shape)
    positions += noise
    for _ in range(3):
        d = cdist(positions, positions)
        np.fill_diagonal(d, np.inf)
        mask = d < 0.36
        if not np.any(mask):
            break
        for i in range(len(positions)):
            close = np.where(mask[i])[0]
            for j in close:
                if j <= i:
                    continue
                dist = d[i, j]
                if dist > 0:
                    direction = (positions[i] - positions[j]) / dist
                    push = (0.36 - dist) / 2.0 + 0.01
                    positions[i] += direction * push
                    positions[j] -= direction * push
    return positions

# --- Growth algorithm: build cluster by adding atoms to interstitial sites ---
def _growth_build(n, potential, kind, spacing=0.72, max_iter_per_step=30, rng=None):
    """
    Sequential growth: start from small cluster, add one atom at a time
    at the best interstitial site, relaxing after each addition.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    
    # Start with a small seed (2-3 atoms)
    start_n = min(3, n)
    if start_n == 1:
        positions = np.array([[0.0, 0.0, 0.0]])
    elif start_n == 2:
        positions = np.array([[spacing*0.5, 0, 0], [-spacing*0.5, 0, 0]])
    else:
        positions = np.array([[0, 0, 0], [spacing*0.6, 0, 0], [spacing*0.3, spacing*0.52, 0]])
    
    best_e, best_pos = relax(positions, potential, kind)
    
    for target_n in range(start_n + 1, n + 1):
        if _time_left() < 15.0:
            break
        
        # Generate candidate interstitial sites
        n_candidates = 20
        candidates = []
        
        # Candidate 1: center of mass
        com = np.mean(best_pos, axis=0)
        candidates.append(com)
        
        # Candidates: random points near existing atoms
        for _ in range(n_candidates - 1):
            # Pick a random existing atom and add a random offset
            idx = rng.integers(0, len(best_pos))
            offset = rng.normal(0, spacing * 0.5, 3)
            candidates.append(best_pos[idx] + offset)
        
        # Also try points on a sphere around the cluster
        for _ in range(5):
            theta = rng.uniform(0, 2*np.pi)
            phi = rng.uniform(-np.pi/2, np.pi/2)
            r = spacing * 0.8 * (target_n ** 0.33)
            p = np.array([r*np.cos(phi)*np.cos(theta), r*np.cos(phi)*np.sin(theta), r*np.sin(phi)])
            candidates.append(p)
        
        candidates = np.array(candidates)
        
        # Evaluate each candidate by adding it and relaxing
        best_candidate_e = np.inf
        best_candidate_pos = None
        
        for cand in candidates:
            if _time_left() < 10.0:
                break
            new_pos = np.vstack([best_pos, cand.reshape(1, 3)])
            # Check minimum distance
            if _min_dist(new_pos) < 0.3:
                continue
            try:
                e, relaxed = relax(new_pos, potential, kind)
                if np.isfinite(e) and e < best_candidate_e:
                    if _min_dist(relaxed) >= 0.35:
                        best_candidate_e = e
                        best_candidate_pos = relaxed
            except Exception:
                continue
        
        if best_candidate_pos is not None:
            best_e, best_pos = best_candidate_e, best_candidate_pos
        else:
            # Fallback: just add a random point
            offset = rng.normal(0, spacing * 0.5, 3)
            idx = rng.integers(0, len(best_pos))
            new_pos = np.vstack([best_pos, (best_pos[idx] + offset).reshape(1, 3)])
            try:
                best_e, best_pos = relax(new_pos, potential, kind)
            except Exception:
                break
    
    return best_e, best_pos

# --- Basin hopping ---
def _basin_hopping(start_pos, potential, kind, spacing, max_iter, t0, t1, rng,
                   best_energy, best_positions, kick_scale=0.15):
    cur_e, cur_pos = relax(start_pos, potential, kind)
    if not np.isfinite(cur_e):
        return best_energy, best_positions
    if cur_e < best_energy and _min_dist(cur_pos) >= 0.35:
        best_energy = cur_e
        best_positions = cur_pos.copy()
    
    for it in range(max_iter):
        if _time_left() < 3.0:
            break
        T = t0 * (t1 / t0) ** (it / max(1, max_iter - 1))
        noise = rng.normal(0, kick_scale * spacing, cur_pos.shape)
        kicked = cur_pos + noise
        try:
            new_e, new_pos = relax(kicked, potential, kind)
        except Exception:
            continue
        if not np.isfinite(new_e):
            continue
        if new_e < best_energy and _min_dist(new_pos) >= 0.35:
            best_energy = new_e
            best_positions = new_pos.copy()
        if new_e < cur_e:
            cur_e, cur_pos = new_e, new_pos
        else:
            delta = new_e - cur_e
            if rng.uniform() < np.exp(-delta / max(T, 1e-10)):
                cur_e, cur_pos = new_e, new_pos
    return best_energy, best_positions

def construct(instance):
    global _START_TIME
    _START_TIME = time.time()
    
    n = instance["n"]
    kind = instance.get("kind", "12-6")
    potential = instance.get("potential", "sutton_chen")
    spacing = 0.72
    
    best_energy = np.inf
    best_positions = None
    rng = np.random.default_rng(n + 42)
    
    def _try_relax(positions):
        nonlocal best_energy, best_positions
        if _time_left() < 2.0:
            return
        try:
            energy, relaxed = relax(positions, potential, kind)
            if np.isfinite(energy) and energy < best_energy:
                if _min_dist(relaxed) >= 0.35:
                    best_energy = energy
                    best_positions = relaxed.copy()
        except Exception:
            pass
    
    # Phase 1: Growth algorithm (deterministic-ish, exploits structural hierarchy)
    if _time_left() > 60.0:
        try:
            growth_e, growth_pos = _growth_build(n, potential, kind, spacing, max_iter_per_step=20, rng=rng)
            if np.isfinite(growth_e) and growth_e < best_energy and _min_dist(growth_pos) >= 0.35:
                best_energy = growth_e
                best_positions = growth_pos
        except Exception:
            pass
    
    # Phase 2: Generate diverse structural seeds
    seeds = []
    
    try:
        mackay = _make_mackay(n, spacing)
        if mackay is not None and len(mackay) == n:
            seeds.append(("mackay", mackay))
    except Exception:
        pass
    
    try:
        fcc = _make_fcc(n, spacing)
        if fcc is not None and len(fcc) == n:
            seeds.append(("fcc", fcc))
    except Exception:
        pass
    
    try:
        bcc = _make_bcc(n, spacing)
        if bcc is not None and len(bcc) == n:
            seeds.append(("bcc", bcc))
    except Exception:
        pass
    
    for seed_val in [n, n + 1, n + 2, n + 3]:
        try:
            rnd = _make_random_compact(n, spacing, seed=seed_val, min_dist=0.5)
            if len(rnd) == n:
                seeds.append((f"rand_{seed_val}", rnd))
        except Exception:
            pass
    
    if seeds:
        for k in range(3):
            try:
                deformed = _make_deformed(seeds[0][1], spacing, seed=n + 10 + k, scale=0.04)
                seeds.append((f"deform_mackay_{k}", deformed))
            except Exception:
                pass
    
    # If growth gave us a good starting point, add it as a seed
    if best_positions is not None and _valid(best_positions):
        seeds.append(("growth", best_positions.copy()))
    
    # Relax all seeds first
    for label, pos in seeds:
        if _time_left() < 4.0:
            break
        _try_relax(pos)
    
    # Phase 3: Basin hopping from best seeds
    if best_positions is not None and _time_left() > 20.0:
        # Prioritize seeds that already gave good energies
        seed_energies = []
        for label, pos in seeds:
            if _time_left() < 4.0:
                break
            try:
                e, _ = relax(pos, potential, kind)
                if np.isfinite(e):
                    seed_energies.append((e, label, pos))
            except Exception:
                pass
        
        seed_energies.sort(key=lambda x: x[0])
        # Take top seeds for basin hopping
        top_seeds = [(label, pos) for e, label, pos in seed_energies[:5]]
        
        for label, start_pos in top_seeds:
            if _time_left() < 10.0:
                break
            max_iter = 50
            t0 = 0.5
            t1 = 0.01
            best_energy, best_positions = _basin_hopping(
                start_pos, potential, kind, spacing,
                max_iter, t0, t1, rng, best_energy, best_positions
            )
    
    # Phase 4: Final polish
    if best_positions is not None and np.all(np.isfinite(best_positions)) and _time_left() > 5.0:
        try:
            energy, relaxed = relax(best_positions, potential, kind)
            if np.isfinite(energy) and _min_dist(relaxed) >= 0.35:
                best_positions = relaxed
        except Exception:
            pass
    
    # Fallback
    if best_positions is None or not np.all(np.isfinite(best_positions)):
        positions = np.zeros((n, 3))
        for i in range(n):
            theta = i * 2.399
            r = spacing * 0.3 * np.sqrt(i)
            positions[i] = [r * np.cos(theta), r * np.sin(theta), 0.1 * i * 0.05]
        try:
            energy, relaxed = relax(positions, potential, kind)
            if np.isfinite(energy):
                best_positions = relaxed
        except Exception:
            best_positions = positions
    
    if not np.all(np.isfinite(best_positions)):
        positions = np.zeros((n, 3))
        side = int(np.ceil(n ** (1/3)))
        for i in range(n):
            ix = i % side
            iy = (i // side) % side
            iz = i // (side * side)
            positions[i] = [ix * 0.5, iy * 0.5, iz * 0.5]
        best_positions = positions
    
    return best_positions
# EVOLVE-BLOCK-END
