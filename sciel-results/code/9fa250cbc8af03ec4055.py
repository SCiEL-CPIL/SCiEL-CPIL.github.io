"""Sutton-Chen cluster global minima: fragment-growth + parallel basin hopping."""
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_DEADLINE_SEC = 300.0
_N_THREADS = 4


def _time_left(start_time):
    return _DEADLINE_SEC - (time.time() - start_time)


def _min_pairwise_dist(positions):
    """Return the minimum pairwise distance in positions (n,3)."""
    n = len(positions)
    if n < 2:
        return np.inf
    # Use broadcasting in chunks to limit memory
    min_d = np.inf
    chunk = max(1, min(n, 200))
    for i in range(0, n, chunk):
        block = positions[i:i+chunk]
        # distances from each atom in block to all atoms after it
        for k in range(len(block)):
            gi = i + k
            diffs = positions[gi+1:] - block[k]
            if len(diffs) > 0:
                d = np.min(np.linalg.norm(diffs, axis=1))
                if d < min_d:
                    min_d = d
    return min_d


def _icosahedron_13():
    phi = (1 + np.sqrt(5)) / 2
    atoms = np.array([
        [0, 0, 0],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
    ], dtype=float)
    atoms = atoms / np.linalg.norm(atoms[1])
    atoms *= 0.72
    return atoms


def _random_compact(n, spacing=0.72, seed=None):
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0/3.0) * 0.85
    points = []
    attempts = 0
    while len(points) < n and attempts < n * 800:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.55 * spacing for q in points):
                points.append(p)
        attempts += 1
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    return np.array(points[:n])


def _fcc_cluster(n, seed=0):
    rng = np.random.default_rng(seed)
    side = int(np.ceil(n ** (1.0/3.0))) + 2
    atoms = []
    for i in range(-side, side+1):
        for j in range(-side, side+1):
            for k in range(-side, side+1):
                for off in [(0,0,0),(0.5,0.5,0),(0.5,0,0.5),(0,0.5,0.5)]:
                    p = np.array([i+off[0], j+off[1], k+off[2]], dtype=float)
                    if np.linalg.norm(p) <= side * 0.85:
                        atoms.append(p)
    atoms = np.array(atoms)
    if len(atoms) < n:
        while len(atoms) < n:
            atoms = np.vstack([atoms, rng.uniform(-side, side, 3)])
    else:
        # Shuffle then pick n nearest to center for diversity
        order = rng.permutation(len(atoms))
        atoms = atoms[order]
        dists = np.linalg.norm(atoms, axis=1)
        idx = np.argsort(dists)[:n]
        atoms = atoms[idx]
    atoms -= atoms.mean(axis=0)
    return atoms


def _hcp_cluster(n, seed=1):
    rng = np.random.default_rng(seed)
    a = 1.0
    c = 1.633 * a
    rows = int(np.ceil(n ** 0.5)) + 1
    atoms = []
    for layer in range(rows + 2):
        z = layer * c / 2
        ox, oy = (0, 0) if layer % 2 == 0 else (a/2, a*np.sqrt(3)/6)
        for i in range(-rows, rows+1):
            for j in range(-rows, rows+1):
                x = i*a + ox + (j % 2) * a/2
                y = j*a*np.sqrt(3)/2 + oy
                p = np.array([x, y, z])
                if np.linalg.norm(p) <= rows * a * 0.95:
                    atoms.append(p)
    atoms = np.array(atoms)
    if len(atoms) < n:
        while len(atoms) < n:
            atoms = np.vstack([atoms, rng.uniform(-rows*a, rows*a, 3)])
    else:
        order = rng.permutation(len(atoms))
        atoms = atoms[order]
        dists = np.linalg.norm(atoms, axis=1)
        idx = np.argsort(dists)[:n]
        atoms = atoms[idx]
    atoms -= atoms.mean(axis=0)
    return atoms


def _local_optimize(positions, kind):
    try:
        energy, new_positions = relax(positions, 'sutton_chen', kind)
        if np.isfinite(energy) and np.all(np.isfinite(new_positions)):
            return float(energy), new_positions
    except Exception:
        pass
    return np.inf, positions


def _fragment_grow(start_positions, kind, target_n, seed, deadline_check):
    """Grow from start_positions to target_n by adding one atom at a time."""
    rng = np.random.default_rng(seed)
    current = start_positions.copy()
    current = current - current.mean(axis=0)
    e, current = _local_optimize(current, kind)
    if not np.isfinite(e):
        return np.inf, current

    while len(current) < target_n:
        if deadline_check() < 10:
            break
        # Place new atom near surface: pick a random existing atom, offset outward
        idx = rng.integers(0, len(current))
        centroid = current.mean(axis=0)
        outward = current[idx] - centroid
        out_norm = np.linalg.norm(outward)
        if out_norm < 1e-6:
            outward = rng.normal(0, 1, 3)
            out_norm = np.linalg.norm(outward)
        outward /= out_norm
        # New atom position: existing atom + outward * spacing + small random jitter
        spacing = 0.72
        new_atom = current[idx] + outward * spacing * rng.uniform(0.8, 1.3) + rng.normal(0, 0.15, 3)
        candidate = np.vstack([current, new_atom])
        # Relax
        e_new, pos_new = _local_optimize(candidate, kind)
        if np.isfinite(e_new):
            current = pos_new
        else:
            # Try a couple more placements
            for _ in range(3):
                new_atom = current[idx] + rng.normal(0, 0.5, 3)
                candidate = np.vstack([current, new_atom])
                e_new, pos_new = _local_optimize(candidate, kind)
                if np.isfinite(e_new):
                    current = pos_new
                    break
    if len(current) < target_n:
        # Pad with random atoms if growth stalled
        while len(current) < target_n:
            new_atom = current[rng.integers(0, len(current))] + rng.normal(0, 0.4, 3)
            current = np.vstack([current, new_atom])
        e, current = _local_optimize(current, kind)
    return e, current


def _basin_hopping_chain(start_pos, kind, target_n, seed, max_time, deadline_check):
    """Run a basin hopping chain from start_pos for up to max_time seconds."""
    rng = np.random.default_rng(seed)
    current_energy, current_pos = _local_optimize(start_pos, kind)
    if not np.isfinite(current_energy):
        return current_energy, current_pos

    best_energy = current_energy
    best_pos = current_pos.copy()
    T = 0.3
    perturbation = 0.25
    chain_start = time.time()

    while True:
        remaining = deadline_check()
        if remaining < 5:
            break
        if time.time() - chain_start > max_time:
            break

        direction = rng.normal(0, 1, current_pos.shape)
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction /= norm
        perturbed = current_pos + perturbation * direction

        e_new, pos_new = _local_optimize(perturbed, kind)
        if not np.isfinite(e_new):
            T *= 0.98
            continue

        accept = False
        if e_new < current_energy:
            accept = True
        else:
            delta = e_new - current_energy
            if T > 1e-6 and rng.random() < np.exp(-delta / T):
                accept = True

        if accept:
            current_energy = e_new
            current_pos = pos_new
            if e_new < best_energy:
                best_energy = e_new
                best_pos = pos_new.copy()
            T *= 0.99
            perturbation = min(0.5, perturbation * 1.002)
        else:
            T *= 0.995

    return best_energy, best_pos


def _recombine(pos_a, pos_b, n, seed):
    """Create a hybrid start by mixing two clusters."""
    rng = np.random.default_rng(seed)
    # Take n atoms: randomly choose from union, with some mixing
    combined = np.vstack([pos_a, pos_b])
    order = rng.permutation(len(combined))
    # Pick n atoms, but ensure some from each
    n_a = n // 2
    n_b = n - n_a
    a_order = rng.permutation(len(pos_a))[:n_a]
    b_order = rng.permutation(len(pos_b))[:n_b]
    mixed = np.vstack([pos_a[a_order], pos_b[b_order]])
    # Add small random perturbation
    mixed += rng.normal(0, 0.1, mixed.shape)
    return mixed


def construct(instance):
    start_time = time.time()
    size = instance["n"]
    kind = instance["kind"]

    def deadline_check():
        return _time_left(start_time)

    best_energy = np.inf
    best_positions = None

    def update_best(e, p):
        nonlocal best_energy, best_positions
        if e < best_energy and np.all(np.isfinite(p)):
            best_energy = e
            best_positions = p.copy()

    # Phase 1: Generate diverse initial clusters and relax them
    starts = []
    starts.append(_icosahedron_13() if size >= 13 else _random_compact(size, seed=0))
    starts.append(_fcc_cluster(size, seed=0))
    starts.append(_fcc_cluster(size, seed=1))
    starts.append(_hcp_cluster(size, seed=0))
    starts.append(_hcp_cluster(size, seed=1))
    for s_seed in range(6):
        starts.append(_random_compact(size, seed=s_seed))

    for s in starts:
        if deadline_check() < 20:
            break
        s = s - s.mean(axis=0)
        if not np.all(np.isfinite(s)):
            continue
        e, p = _local_optimize(s, kind)
        if np.isfinite(e):
            update_best(e, p)

    if best_positions is None:
        rng = np.random.default_rng(size)
        radius = 0.72 * size ** (1.0/3.0)
        return rng.uniform(-radius, radius, (size, 3))

    # Phase 2: Fragment growth from icosahedron core
    if deadline_check() > 30 and size > 13:
        core = _icosahedron_13()
        e_grow, p_grow = _fragment_grow(core, kind, size, seed=100, deadline_check=deadline_check)
        if np.isfinite(e_grow):
            update_best(e_grow, p_grow)

    # Phase 3: Parallel basin hopping chains
    if deadline_check() > 15:
        n_chains = _N_THREADS
        # Create chain starts: best position + perturbations, plus recombined starts
        chain_starts = []
        for i in range(n_chains):
            if i == 0:
                cs = best_positions + np.random.default_rng(200+i).normal(0, 0.15, best_positions.shape)
            elif i == 1 and len(starts) >= 2:
                cs = _recombine(starts[0], starts[1], size, seed=300+i)
            else:
                base = starts[i % len(starts)]
                cs = base + np.random.default_rng(400+i).normal(0, 0.2, base.shape)
            chain_starts.append(cs)

        remaining = deadline_check()
        max_time_per_chain = remaining * 0.7 / n_chains  # each chain gets a share

        with ThreadPoolExecutor(max_workers=n_chains) as executor:
            futures = {}
            for i, cs in enumerate(chain_starts):
                fut = executor.submit(_basin_hopping_chain, cs, kind, size, 500+i, max_time_per_chain, deadline_check)
                futures[fut] = i
            for fut in as_completed(futures):
                if deadline_check() < 5:
                    break
                try:
                    e_chain, p_chain = fut.result()
                    if np.isfinite(e_chain):
                        update_best(e_chain, p_chain)
                except Exception:
                    pass

    # Phase 4: If time remains, do more basin hopping from the best
    if deadline_check() > 15:
        rng = np.random.default_rng(777)
        current_energy = best_energy
        current_pos = best_positions.copy()
        T = 0.2
        perturbation = 0.2
        phase4_start = time.time()
        while deadline_check() > 5 and time.time() - phase4_start < 20:
            direction = rng.normal(0, 1, current_pos.shape)
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction /= norm
            perturbed = current_pos + perturbation * direction
            e_new, pos_new = _local_optimize(perturbed, kind)
            if not np.isfinite(e_new):
                T *= 0.98
                continue
            accept = False
            if e_new < current_energy:
                accept = True
            else:
                delta = e_new - current_energy
                if T > 1e-6 and rng.random() < np.exp(-delta / T):
                    accept = True
            if accept:
                current_energy = e_new
                current_pos = pos_new
                if e_new < best_energy:
                    best_energy = e_new
                    best_positions = pos_new.copy()
                T *= 0.99
                perturbation = min(0.5, perturbation * 1.002)
            else:
                T *= 0.995

    # Final validation: ensure no atoms closer than 0.35
    if best_positions is not None:
        min_d = _min_pairwise_dist(best_positions)
        if min_d < 0.35:
            rng = np.random.default_rng(12345)
            for _ in range(100):
                candidate = best_positions + rng.normal(0, 0.08, best_positions.shape)
                e_cand, p_cand = _local_optimize(candidate, kind)
                if np.isfinite(e_cand):
                    min_d2 = _min_pairwise_dist(p_cand)
                    if min_d2 >= 0.35:
                        best_positions = p_cand
                        break
            else:
                # Last resort: spread atoms more
                best_positions = best_positions * 1.1 + rng.normal(0, 0.1, best_positions.shape)
                e_cand, p_cand = _local_optimize(best_positions, kind)
                if np.isfinite(e_cand):
                    best_positions = p_cand

    return best_positions
