"""Parallel Basin Hopping with diverse initial structures for Sutton-Chen TM clusters."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import multiprocessing as mp
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_START_TIME = time.time()
_DEADLINE = 290.0  # Safety margin: evaluator kills at 330s

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def _is_valid(positions, min_dist=0.35):
    """Check that all coordinates are finite and pairwise distances >= min_dist."""
    if not np.all(np.isfinite(positions)):
        return False
    n = len(positions)
    if n <= 1:
        return True
    dists = np.linalg.norm(positions[:, np.newaxis, :] - positions[np.newaxis, :, :], axis=2)
    np.fill_diagonal(dists, np.inf)
    return np.all(dists >= min_dist)

def _generate_fcc(n):
    """Generate a cluster of n atoms in an FCC-like structure."""
    max_r = int(np.ceil(n ** (1/3))) + 3
    candidates = []
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            for k in range(-max_r, max_r + 1):
                if (i + j + k) % 2 == 0:
                    p = np.array([i, j, k]) * (1.0 / np.sqrt(2.0))
                    candidates.append((np.linalg.norm(p), p))
    candidates.sort(key=lambda x: x[0])
    pts = [p for _, p in candidates[:n]]
    return np.array(pts)

def _generate_bcc(n):
    """Generate a cluster of n atoms in a BCC-like structure."""
    max_r = int(np.ceil(n ** (1/3))) + 3
    candidates = []
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            for k in range(-max_r, max_r + 1):
                p1 = np.array([i, j, k])
                p2 = np.array([i + 0.5, j + 0.5, k + 0.5])
                candidates.append((np.linalg.norm(p1), p1))
                candidates.append((np.linalg.norm(p2), p2))
    candidates.sort(key=lambda x: x[0])
    pts = [p for _, p in candidates[:n]]
    return np.array(pts)

def _generate_mackay_icosahedron(n):
    """Generate a Mackay icosahedral cluster (1, 12, 42, 120, ...)."""
    phi = (1 + np.sqrt(5)) / 2
    # 12 vertices of icosahedron
    v = [
        np.array([0, 1, phi]), np.array([0, 1, -phi]),
        np.array([0, -1, phi]), np.array([0, -1, -phi]),
        np.array([1, phi, 0]), np.array([1, -phi, 0]),
        np.array([-1, phi, 0]), np.array([-1, -phi, 0]),
        np.array([phi, 0, 1]), np.array([phi, 0, -1]),
        np.array([-phi, 0, 1]), np.array([-phi, 0, -1])
    ]
    v = [p / np.linalg.norm(p) for p in v]
    
    pts = [np.zeros(3)]  # Center
    if n > 1:
        pts.extend(v[:min(n-1, 12)])
    
    if n > 13:
        # Second shell: 30 edge midpoints + 20 face centers
        # Edge midpoints (30 edges of icosahedron)
        edges = [
            (0,1),(0,2),(0,4),(0,5),
            (1,3),(1,4),(1,7),(1,8),
            (2,3),(2,6),(2,9),(2,10),
            (3,7),(3,8),
            (4,5),(4,8),(4,11),
            (5,9),(5,10),(5,11),
            (6,7),(6,9),(6,10),
            (7,11),
            (8,11),
            (9,10)
        ]
        edge_pts = []
        for a, b in edges:
            mid = (v[a] + v[b]) / 2.0
            edge_pts.append(mid / np.linalg.norm(mid))
        
        # Face centers (20 faces)
        faces = [
            (0,1,4),(0,1,8),(0,2,5),(0,2,9),(0,4,11),(0,5,11),
            (1,3,4),(1,3,7),(1,4,8),(1,7,8),
            (2,3,5),(2,3,6),(2,5,9),(2,6,10),
            (3,6,7),(3,7,11),
            (4,5,11),(4,8,11),
            (5,9,11),(5,10,11),
            (6,9,10),(6,10,11),
            (7,9,10),(7,10,11)
        ]
        face_pts = []
        for a, b, c in faces:
            fc = (v[a] + v[b] + v[c]) / 3.0
            face_pts.append(fc / np.linalg.norm(fc))
        
        shell2 = edge_pts + face_pts
        remaining = n - len(pts)
        pts.extend(shell2[:remaining])
    
    if n > 55:
        # Third shell: 120 points (approximate with sphere sampling)
        remaining = n - len(pts)
        # Use golden spiral for even distribution
        for i in range(remaining):
            y = 1.0 - 2.0 * (i + 0.5) / remaining
            r = np.sqrt(1.0 - y*y)
            theta = 2.399963229728653 * i  # golden angle
            p = np.array([r*np.cos(theta), r*np.sin(theta), y]) * 2.6
            pts.append(p)
    
    return np.array(pts[:n])

def _generate_snowball(n, rng, seed_offset=0):
    """Generate a snowball cluster by sequentially adding atoms at best positions."""
    rng_local = np.random.default_rng(seed_offset)
    pts = [np.zeros(3)]
    if n <= 1:
        return np.array(pts)
    
    # First shell: 12 atoms in icosahedral arrangement
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
    pts.extend(v[:min(n-1, 12)])
    
    while len(pts) < n:
        # Try random positions and pick the one that minimizes energy contribution
        best_p = None
        best_e = np.inf
        for _ in range(50):
            theta = rng_local.uniform(0, 2*np.pi)
            cos_phi = rng_local.uniform(-1, 1)
            sin_phi = np.sqrt(1 - cos_phi**2)
            r = rng_local.uniform(1.5, 3.0)
            p = r * np.array([sin_phi*np.cos(theta), sin_phi*np.sin(theta), cos_phi])
            
            # Quick energy estimate: sum of pair interactions with existing atoms
            if len(pts) == 0:
                e = 0.0
            else:
                pts_arr = np.array(pts)
                dists = np.linalg.norm(pts_arr - p, axis=1)
                dists = dists[dists > 0.1]
                if len(dists) == 0:
                    e = np.inf
                else:
                    e = np.sum(dists**-12) - 144.41 * np.sqrt(np.sum(dists**-6))
            
            if e < best_e:
                best_e = e
                best_p = p
        
        if best_p is not None:
            # Check minimum distance
            pts_arr = np.array(pts)
            dists = np.linalg.norm(pts_arr - best_p, axis=1)
            if np.min(dists) > 0.35:
                pts.append(best_p)
            else:
                # Slightly perturb
                pts.append(best_p + rng_local.normal(0, 0.1, 3))
    
    return np.array(pts[:n])

def _generate_random(n, rng, seed_offset=0):
    """Generate a random cluster with minimum distance enforcement."""
    rng_local = np.random.default_rng(seed_offset)
    for _ in range(200):
        pts = rng_local.uniform(-3, 3, (n, 3))
        if _is_valid(pts):
            return pts
    # Last resort: FCC
    return _generate_fcc(n)

def _bh_chain(pos, kind, n_steps, rng, deadline):
    """Run a basin hopping chain with adaptive step size. Returns (best_energy, best_pos)."""
    best_energy = np.inf
    best_pos = pos.copy()
    
    try:
        energy, pos = relax(pos, 'sutton_chen', kind)
        if not np.isfinite(energy) or not _is_valid(pos):
            return np.inf, best_pos
        if energy < best_energy:
            best_energy = energy
            best_pos = pos.copy()
    except Exception:
        return np.inf, best_pos
    
    step_size = 0.3
    consecutive_rejects = 0
    
    for step in range(n_steps):
        if time.time() > deadline:
            break
        
        # Adaptive step size
        if consecutive_rejects > 10:
            step_size = min(step_size * 1.3, 1.0)
            consecutive_rejects = 0
        elif step % 20 == 0:
            step_size = max(step_size * 0.9, 0.1)
        
        noise = rng.normal(0, step_size, pos.shape)
        perturbed = pos + noise
        
        try:
            new_energy, new_pos = relax(perturbed, 'sutton_chen', kind)
            if not np.isfinite(new_energy) or not _is_valid(new_pos):
                continue
        except Exception:
            continue
        
        if new_energy < energy:
            pos = new_pos
            energy = new_energy
            step_size = max(step_size * 0.95, 0.1)
            consecutive_rejects = 0
            if energy < best_energy:
                best_energy = energy
                best_pos = pos.copy()
        else:
            # Metropolis acceptance with cooling temperature
            T = max(0.01, 0.1 * np.exp(-step / max(n_steps, 1)))
            if rng.uniform() < np.exp(-(new_energy - energy) / T):
                pos = new_pos
                energy = new_energy
                consecutive_rejects = 0
            else:
                consecutive_rejects += 1
    
    return best_energy, best_pos

def _worker_task(args):
    """Worker function for parallel BH chains."""
    pos, kind, n_steps, seed, deadline = args
    rng = np.random.default_rng(seed)
    try:
        energy, best_pos = _bh_chain(pos, kind, n_steps, rng, deadline)
        return (energy, best_pos)
    except Exception:
        return (np.inf, pos)

def construct(instance):
    n = instance["n"]
    kind = instance["kind"]
    
    best_energy = np.inf
    best_pos = None
    
    deadline = _START_TIME + _DEADLINE
    
    # Generate diverse initial structures
    initial_structures = []
    
    # FCC
    try:
        fcc = _generate_fcc(n)
        if _is_valid(fcc):
            initial_structures.append(fcc)
    except Exception:
        pass
    
    # BCC
    try:
        bcc = _generate_bcc(n)
        if _is_valid(bcc):
            initial_structures.append(bcc)
    except Exception:
        pass
    
    # Mackay icosahedron
    try:
        ico = _generate_mackay_icosahedron(n)
        if _is_valid(ico):
            initial_structures.append(ico)
    except Exception:
        pass
    
    # Snowball with different seeds
    for seed in [1, 2, 3]:
        try:
            sb = _generate_snowball(n, np.random.default_rng(0), seed)
            if _is_valid(sb):
                initial_structures.append(sb)
        except Exception:
            pass
    
    # Random with different seeds
    for seed in [10, 20]:
        try:
            rnd = _generate_random(n, np.random.default_rng(0), seed)
            if _is_valid(rnd):
                initial_structures.append(rnd)
        except Exception:
            pass
    
    # If no valid structures, use FCC as fallback
    if not initial_structures:
        initial_structures = [_generate_fcc(n)]
    
    # Determine number of steps per chain based on time budget
    n_structures = len(initial_structures)
    n_workers = min(4, n_structures)
    
    # Estimate time per relaxation call (~0.1-0.5s for n~80)
    # Budget: ~280s for search, leave 10s for overhead
    time_budget = min(_time_left() - 10.0, 280.0)
    if time_budget < 5.0:
        time_budget = 5.0
    
    # Each worker gets time_budget / n_workers seconds
    worker_time = time_budget / n_workers
    # Estimate steps: ~0.3s per BH step (relax + overhead)
    n_steps_per_chain = max(5, int(worker_time / 0.4))
    
    # Prepare tasks for parallel execution
    tasks = []
    for i, struct in enumerate(initial_structures[:n_workers]):
        seed = 1000 + i * 7 + n
        tasks.append((struct, kind, n_steps_per_chain, seed, deadline))
    
    # Run parallel BH chains
    try:
        with mp.Pool(processes=n_workers) as pool:
            results = pool.map(_worker_task, tasks)
        
        for energy, pos in results:
            if energy < best_energy and np.isfinite(energy) and _is_valid(pos):
                best_energy = energy
                best_pos = pos
    except Exception:
        # Fallback: sequential BH
        for i, struct in enumerate(initial_structures[:3]):
            if _time_left() < 5.0:
                break
            rng = np.random.default_rng(1000 + i * 7 + n)
            energy, pos = _bh_chain(struct, kind, n_steps_per_chain, rng, deadline)
            if energy < best_energy and np.isfinite(energy) and _is_valid(pos):
                best_energy = energy
                best_pos = pos
    
    # If still no valid result, do a final sequential attempt
    if best_pos is None:
        for struct in initial_structures[:2]:
            if _time_left() < 3.0:
                break
            rng = np.random.default_rng(9999)
            energy, pos = _bh_chain(struct, kind, 10, rng, deadline)
            if energy < best_energy and np.isfinite(energy) and _is_valid(pos):
                best_energy = energy
                best_pos = pos
    
    # Final fallback
    if best_pos is None:
        best_pos = _generate_fcc(n)
        if not _is_valid(best_pos):
            best_pos = _generate_bcc(n)
        if not _is_valid(best_pos):
            rng = np.random.default_rng(42)
            for _ in range(100):
                best_pos = rng.uniform(-2, 2, (n, 3))
                if _is_valid(best_pos):
                    break
    
    return best_pos

# EVOLVE-BLOCK-END
