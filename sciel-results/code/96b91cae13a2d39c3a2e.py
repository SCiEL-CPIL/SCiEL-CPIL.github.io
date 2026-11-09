# EVOLVE-BLOCK-START
import numpy as np
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax
from autoresearch_suite.chem_clusters import dzugutov_energy_gradient

# Global deadline tracking
_START_TIME = None
_HARD_LIMIT = 288.0
_INSTANCE_ORDER = [38, 45, 50, 55, 57, 60]
_INSTANCE_INDEX = {n: i for i, n in enumerate(_INSTANCE_ORDER)}

# Dzugutov potential parameters
A_DZ = 5.82
B_DZ = 1.28
a_DZ = 1.87
b_DZ = 1.94
c_DZ = 1.10
d_DZ = 0.27
m_DZ = 16


def _dzugutov_energy(positions):
    """Compute Dzugutov potential energy exactly."""
    n = len(positions)
    if n < 2:
        return 0.0
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            rij = np.linalg.norm(positions[i] - positions[j])
            if rij < 1e-10:
                continue
            if rij < a_DZ:
                v = A_DZ * (rij ** (-m_DZ) - B_DZ) * np.exp(c_DZ / (rij - a_DZ))
            elif rij < b_DZ:
                v = B_DZ * np.exp(d_DZ / (rij - b_DZ))
            else:
                v = 0.0
            total += v
    return total


def _get_icosahedron_13():
    """Return the 13-atom icosahedral cluster (known global minimum core)."""
    phi = (1 + np.sqrt(5)) / 2
    # 12 vertices of icosahedron
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    # Normalize to unit radius
    verts /= np.linalg.norm(verts[0])
    # Scale to typical Dzugutov equilibrium (~1.8-2.0)
    scale = 1.9
    verts *= scale
    # Center at origin
    center = np.zeros(3)
    return np.vstack([center, verts])


def _icosahedron_edges(verts):
    """Find edges of icosahedron from 12 vertex positions."""
    edges = []
    dists = []
    for i in range(12):
        for j in range(i + 1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            dists.append((d, i, j))
    dists.sort()
    min_dist = dists[0][0]
    for d, i, j in dists:
        if d < min_dist * 1.01:
            edges.append((i, j))
    return edges


def _icosahedron_faces(verts, edges):
    """Find triangular faces of icosahedron."""
    connected = set()
    for i, j in edges:
        connected.add((i, j))
        connected.add((j, i))
    faces = []
    for i in range(12):
        for j in range(i + 1, 12):
            if (i, j) in connected:
                for k in range(j + 1, 12):
                    if (j, k) in connected and (i, k) in connected:
                        faces.append((i, j, k))
    return faces


def _build_hierarchical_seed(n_atoms, seed_idx, scale=1.9):
    """
    Build a hierarchical polytetrahedral seed:
    - Core: 13-atom icosahedron
    - Shell 2: 30 edge-midpoint positions (Mackay 43)
    - Shell 3: 20 face-center positions (Mackay 63)
    - Extra: random surface positions
    """
    core = _get_icosahedron_13()  # 13 atoms, first is center
    center = core[0]
    verts = core[1:]  # 12 vertices
    edges = _icosahedron_edges(verts)
    faces = _icosahedron_faces(verts, edges)
    
    rng = np.random.default_rng(seed_idx * 7919 + n_atoms * 13)
    
    positions = [center.copy()]
    
    # Shell 1: 12 vertices
    for v in verts:
        if len(positions) >= n_atoms:
            break
        p = v + rng.normal(0, 0.03 * scale, 3)
        positions.append(p)
    
    remaining = n_atoms - len(positions)
    if remaining <= 0:
        return np.array(positions[:n_atoms])
    
    # Shell 2: 30 edge midpoints
    for i, j in edges:
        if remaining <= 0:
            break
        mid = (verts[i] + verts[j]) / 2.0
        # Push outward to maintain polytetrahedral geometry
        mid_dir = mid / (np.linalg.norm(mid) + 1e-10)
        r_mid = np.linalg.norm(mid) * 1.15
        p = mid_dir * r_mid + rng.normal(0, 0.04 * scale, 3)
        positions.append(p)
        remaining -= 1
    
    if remaining <= 0:
        return np.array(positions[:n_atoms])
    
    # Shell 3: 20 face centers
    for i, j, k in faces:
        if remaining <= 0:
            break
        fc = (verts[i] + verts[j] + verts[k]) / 3.0
        fc_dir = fc / (np.linalg.norm(fc) + 1e-10)
        r_fc = np.linalg.norm(fc) * 1.25
        p = fc_dir * r_fc + rng.normal(0, 0.05 * scale, 3)
        positions.append(p)
        remaining -= 1
    
    # Extra atoms: random on a slightly larger sphere
    if remaining > 0:
        r_outer = scale * 1.5
        for _ in range(remaining):
            theta = rng.uniform(0, 2 * np.pi)
            cos_phi = rng.uniform(-1, 1)
            sin_phi = np.sqrt(1 - cos_phi ** 2)
            p = r_outer * np.array([sin_phi * np.cos(theta), sin_phi * np.sin(theta), cos_phi])
            p = p + rng.normal(0, 0.1 * scale, 3)
            positions.append(p)
    
    return np.array(positions[:n_atoms])


def _build_icosahedral_shell_seed(n_atoms, seed_idx, scale=1.9):
    """
    Alternative: icosahedral shell with more randomness.
    """
    core = _get_icosahedron_13()
    center = core[0]
    verts = core[1:]
    edges = _icosahedron_edges(verts)
    faces = _icosahedron_faces(verts, edges)
    
    rng = np.random.default_rng(seed_idx * 104729 + n_atoms)
    
    positions = [center.copy()]
    for v in verts:
        if len(positions) >= n_atoms:
            break
        p = v + rng.normal(0, 0.06 * scale, 3)
        positions.append(p)
    
    remaining = n_atoms - len(positions)
    if remaining <= 0:
        return np.array(positions[:n_atoms])
    
    # Mix of edge mids and face centers
    candidates = []
    for i, j in edges:
        mid = (verts[i] + verts[j]) / 2.0
        candidates.append(mid * 1.1)
    for i, j, k in faces:
        fc = (verts[i] + verts[j] + verts[k]) / 3.0
        candidates.append(fc * 1.2)
    
    idxs = rng.permutation(len(candidates))
    for idx in idxs:
        if remaining <= 0:
            break
        p = candidates[idx] + rng.normal(0, 0.1 * scale, 3)
        valid = True
        for q in positions:
            if np.linalg.norm(p - q) < 0.5:
                valid = False
                break
        if valid:
            positions.append(p)
            remaining -= 1
    
    if remaining > 0:
        r_outer = scale * 1.6
        for _ in range(remaining):
            theta = rng.uniform(0, 2 * np.pi)
            cos_phi = rng.uniform(-1, 1)
            sin_phi = np.sqrt(1 - cos_phi ** 2)
            p = r_outer * np.array([sin_phi * np.cos(theta), sin_phi * np.sin(theta), cos_phi])
            p = p + rng.normal(0, 0.15 * scale, 3)
            positions.append(p)
    
    return np.array(positions[:n_atoms])


def _build_random_compact_seed(n_atoms, seed_idx, scale=1.9):
    """Random compact cluster with minimum distance constraint."""
    rng = np.random.default_rng(seed_idx * 31 + n_atoms)
    radius = 0.8 * n_atoms ** (1.0 / 3.0) * scale
    points = []
    max_attempts = 5000
    attempts = 0
    while len(points) < n_atoms and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.55 for q in points):
                points.append(p)
        attempts += 1
    while len(points) < n_atoms:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    return np.array(points[:n_atoms])


def _generate_seeds(n_atoms, n_seeds):
    """Generate diverse seeds for n_atoms atoms."""
    seeds = []
    
    # Hierarchical polytetrahedral seeds (primary, best for Dzugutov)
    n_hier = min(6, n_seeds)
    for s in range(n_hier):
        try:
            seeds.append(_build_hierarchical_seed(n_atoms, seed_idx=s))
        except Exception:
            pass
    
    # Icosahedral shell with more randomness
    n_shell = min(4, max(0, n_seeds - n_hier))
    for s in range(n_shell):
        try:
            seeds.append(_build_icosahedral_shell_seed(n_atoms, seed_idx=s + 100))
        except Exception:
            pass
    
    # Random compact for diversity
    n_rand = min(4, max(0, n_seeds - n_hier - n_shell))
    for s in range(n_rand):
        try:
            seeds.append(_build_random_compact_seed(n_atoms, seed_idx=s + 200))
        except Exception:
            pass
    
    # Fallback
    if not seeds:
        rng = np.random.default_rng(n_atoms)
        radius = 0.8 * n_atoms ** (1.0 / 3.0) * 1.9
        points = []
        while len(points) < n_atoms:
            p = rng.uniform(-radius, radius, 3)
            points.append(p)
        seeds.append(np.array(points))
    
    return seeds[:n_seeds]


def _gradient_optimize(positions, potential, parameter, deadline, max_time=5.0):
    """
    Use L-BFGS-B with the provided gradient for fast local minimization.
    Falls back to relax() if gradient method fails.
    """
    t0 = time.time()
    
    def objective(x):
        if time.time() > deadline:
            return float('inf')
        try:
            pos = x.reshape(-1, 3)
            e, _ = dzugutov_energy_gradient(pos)
            return e
        except Exception:
            return float('inf')
    
    def gradient(x):
        if time.time() > deadline:
            return np.zeros_like(x)
        try:
            pos = x.reshape(-1, 3)
            _, g = dzugutov_energy_gradient(pos)
            return g.flatten()
        except Exception:
            return np.zeros_like(x)
    
    x0 = positions.flatten()
    
    try:
        result = minimize(
            objective, x0, jac=gradient,
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8,
                     'maxfun': 500, 'maxls': 50}
        )
        if result.success or result.fun < float('inf'):
            best_pos = result.x.reshape(-1, 3)
            # Verify with relax for safety
            try:
                e, pos = relax(best_pos.copy(), potential, parameter)
                return e, pos
            except Exception:
                e = _dzugutov_energy(best_pos)
                return e, best_pos
    except Exception:
        pass
    
    # Fallback to relax
    try:
        return relax(positions.copy(), potential, parameter)
    except Exception:
        return float('inf'), positions


def _basin_hopping(positions, potential, parameter, rng, max_iters, kick_start, kick_end, deadline):
    """
    Basin hopping with adaptive cooling using gradient-based local minimization.
    """
    try:
        best_energy, best_positions = _gradient_optimize(
            positions.copy(), potential, parameter, deadline, max_time=3.0
        )
    except Exception:
        return float('inf'), positions
    
    current_energy = best_energy
    current_positions = best_positions.copy()
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        t = it / max(1, max_iters - 1)
        kick_scale = kick_start * (kick_end / kick_start) ** t
        
        kick = rng.normal(0, kick_scale, size=current_positions.shape)
        new_positions = current_positions + kick
        
        try:
            new_energy, new_relaxed = _gradient_optimize(
                new_positions, potential, parameter, deadline, max_time=2.0
            )
        except Exception:
            continue
        
        if new_energy < current_energy:
            current_energy = new_energy
            current_positions = new_relaxed
            if new_energy < best_energy:
                best_energy = new_energy
                best_positions = new_relaxed.copy()
        else:
            delta_e = new_energy - current_energy
            temp = 0.06 * (1.0 - t) + 0.005
            if delta_e < 0.3 and rng.random() < np.exp(-delta_e / temp):
                current_energy = new_energy
                current_positions = new_relaxed
    
    return best_energy, best_positions


def construct(instance):
    """
    Multi-stage search:
    Stage 1: Fast gradient-based multi-start local minimization.
    Stage 2: Basin hopping from top-K relaxed configurations.
    Stage 3: Final gradient polish on best.
    """
    global _START_TIME
    
    if _START_TIME is None:
        _START_TIME = time.time()
    
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    elapsed = time.time() - _START_TIME
    remaining_total = max(0.0, _HARD_LIMIT - elapsed)
    
    idx = _INSTANCE_INDEX.get(size, 0)
    n_rem = len(_INSTANCE_ORDER) - idx
    
    if n_rem > 0:
        call_budget = remaining_total / n_rem
    else:
        call_budget = remaining_total
    
    call_budget = max(call_budget, 2.0)
    call_deadline = time.time() + call_budget
    
    # Stage 1: Generate seeds and do fast gradient-based relaxations
    n_seeds = min(20, max(10, int(call_budget * 3)))
    seeds = _generate_seeds(size, n_seeds)
    
    # Relax all seeds in parallel using gradient-based optimization
    relaxed_results = []  # list of (energy, positions)
    
    def do_optimize(seed):
        if time.time() > call_deadline:
            return (float('inf'), None)
        try:
            e, pos = _gradient_optimize(seed.copy(), potential, parameter, call_deadline, max_time=3.0)
            if e < float('inf'):
                return (e, pos)
        except Exception:
            pass
        return (float('inf'), None)
    
    n_threads = min(4, len(seeds))
    
    if n_threads == 1:
        for s in seeds:
            e, pos = do_optimize(s)
            if pos is not None:
                relaxed_results.append((e, pos))
    else:
        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = {executor.submit(do_optimize, s): i for i, s in enumerate(seeds)}
            for future in as_completed(futures):
                try:
                    timeout = max(0.1, call_deadline - time.time())
                    e, pos = future.result(timeout=timeout)
                    if pos is not None:
                        relaxed_results.append((e, pos))
                except Exception:
                    continue
    
    if not relaxed_results:
        # Fallback: return first seed
        return seeds[0]
    
    # Sort by energy
    relaxed_results.sort(key=lambda x: x[0])
    
    best_energy = relaxed_results[0][0]
    best_positions = relaxed_results[0][1].copy()
    
    # Check remaining time for Stage 2
    remaining_for_bh = call_deadline - time.time()
    
    if remaining_for_bh < 3.0:
        return best_positions
    
    # Stage 2: Basin hopping from top-K relaxed configs
    K = min(5, len(relaxed_results))
    
    # Allocate time for BH: use up to 70% of remaining
    bh_deadline = time.time() + min(remaining_for_bh * 0.7, 20.0)
    
    # Determine max iterations based on time
    max_iters = min(25, max(5, int(remaining_for_bh * 0.4)))
    
    kick_start = 0.35
    kick_end = 0.06
    
    def run_bh_chain(start_pos, chain_id):
        if time.time() > bh_deadline:
            return (float('inf'), None)
        rng = np.random.default_rng(chain_id * 1000 + size)
        e, pos = _basin_hopping(
            start_pos, potential, parameter, rng,
            max_iters, kick_start, kick_end, bh_deadline
        )
        return (e, pos)
    
    n_bh_threads = min(4, K)
    
    if n_bh_threads == 1:
        e, pos = run_bh_chain(relaxed_results[0][1], 0)
        if pos is not None and e < best_energy:
            best_energy = e
            best_positions = pos
    else:
        with ThreadPoolExecutor(max_workers=n_bh_threads) as executor:
            futures = {
                executor.submit(run_bh_chain, relaxed_results[i][1], i): i
                for i in range(K)
            }
            for future in as_completed(futures):
                try:
                    timeout = max(0.1, bh_deadline - time.time())
                    e, pos = future.result(timeout=timeout)
                    if pos is not None and e < best_energy:
                        best_energy = e
                        best_positions = pos
                except Exception:
                    continue
    
    # Stage 3: Final gradient polish
    remaining_final = call_deadline - time.time()
    if remaining_final > 1.0:
        try:
            final_e, final_pos = _gradient_optimize(
                best_positions.copy(), potential, parameter, call_deadline, max_time=2.0
            )
            if final_e < best_energy:
                best_energy = final_e
                best_positions = final_pos
        except Exception:
            pass
    
    return best_positions
# EVOLVE-BLOCK-END
