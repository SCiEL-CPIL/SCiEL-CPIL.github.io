"""Rewritten program for Sutton-Chen transition-metal cluster global minima.
Uses a model-and-optimize approach with Metropolis Basin Hopping.
"""
import time
import numpy as np
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_DEADLINE_SEC = 300.0

def _time_left(start_time):
    return _DEADLINE_SEC - (time.time() - start_time)

def _random_compact(n, spacing=0.72, seed=None):
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    points = []
    attempts = 0
    while len(points) < n and attempts < n * 500:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            # Check distance
            if all(np.linalg.norm(p - q) >= 0.6 * spacing for q in points):
                points.append(p)
        attempts += 1
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    return np.array(points[:n])

def _fcc_cluster(n):
    atoms = []
    side = int(np.ceil(n ** (1.0 / 3.0))) + 2
    for i in range(-side, side + 1):
        for j in range(-side, side + 1):
            for k in range(-side, side + 1):
                for offset in [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]:
                    p = np.array([i + offset[0], j + offset[1], k + offset[2]])
                    if np.linalg.norm(p) <= side * 0.8:
                        atoms.append(p)
    atoms = np.array(atoms)
    if len(atoms) < n:
        rng = np.random.default_rng(42)
        while len(atoms) < n:
            atoms = np.vstack([atoms, rng.uniform(-side, side, 3)])
    else:
        dists = np.linalg.norm(atoms, axis=1)
        idx = np.argsort(dists)[:n]
        atoms = atoms[idx]
    atoms -= atoms.mean(axis=0)
    return atoms

def _bcc_cluster(n):
    atoms = []
    side = int(np.ceil(n ** (1.0 / 3.0))) + 2
    for i in range(-side, side + 1):
        for j in range(-side, side + 1):
            for k in range(-side, side + 1):
                p = np.array([i, j, k])
                if np.linalg.norm(p) <= side * 0.8:
                    atoms.append(p)
                p2 = np.array([i + 0.5, j + 0.5, k + 0.5])
                if np.linalg.norm(p2) <= side * 0.8:
                    atoms.append(p2)
    atoms = np.array(atoms)
    if len(atoms) < n:
        rng = np.random.default_rng(43)
        while len(atoms) < n:
            atoms = np.vstack([atoms, rng.uniform(-side, side, 3)])
    else:
        dists = np.linalg.norm(atoms, axis=1)
        idx = np.argsort(dists)[:n]
        atoms = atoms[idx]
    atoms -= atoms.mean(axis=0)
    return atoms

def _hcp_cluster(n):
    atoms = []
    a = 1.0
    c = 1.633 * a
    rows = int(np.ceil(n ** 0.5))
    for layer in range(0, rows + 2):
        z = layer * c / 2
        if layer % 2 == 0:
            offset_x, offset_y = 0, 0
        else:
            offset_x, offset_y = a / 2, a * np.sqrt(3) / 6
        for i in range(-rows, rows + 1):
            for j in range(-rows, rows + 1):
                x = i * a + offset_x + (j % 2) * a / 2
                y = j * a * np.sqrt(3) / 2 + offset_y
                p = np.array([x, y, z])
                if np.linalg.norm(p) <= rows * a * 0.9:
                    atoms.append(p)
    atoms = np.array(atoms)
    if len(atoms) < n:
        rng = np.random.default_rng(44)
        while len(atoms) < n:
            atoms = np.vstack([atoms, rng.uniform(-rows * a, rows * a, 3)])
    else:
        dists = np.linalg.norm(atoms, axis=1)
        idx = np.argsort(dists)[:n]
        atoms = atoms[idx]
    atoms -= atoms.mean(axis=0)
    return atoms

def _icosahedral_shell(n):
    phi = (1 + np.sqrt(5)) / 2
    ico_atoms = np.array([
        [0, 0, 0],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
    ])
    ico_atoms = ico_atoms / np.linalg.norm(ico_atoms[1])
    ico_atoms *= 0.72
    
    if n <= 13:
        dists = np.linalg.norm(ico_atoms, axis=1)
        idx = np.argsort(dists)[:n]
        return ico_atoms[idx].copy()
    
    atoms = [ico_atoms.copy()]
    current_n = 13
    shell_radius = 1.4
    rng = np.random.default_rng(45)
    while current_n < n:
        theta = rng.uniform(0, 2 * np.pi)
        z = rng.uniform(-1, 1)
        r = np.sqrt(1 - z * z)
        p = shell_radius * np.array([r * np.cos(theta), r * np.sin(theta), z])
        atoms.append(p)
        current_n += 1
    all_atoms = np.vstack(atoms)
    all_atoms -= all_atoms.mean(axis=0)
    return all_atoms

def _local_optimize(positions, kind, max_iters=5000, tol=1e-10):
    try:
        energy, new_positions = relax(positions, 'sutton_chen', kind)
        if np.isfinite(energy) and np.all(np.isfinite(new_positions)):
            return energy, new_positions
    except Exception:
        pass
    return np.inf, positions

def construct(instance):
    start_time = time.time()
    size = instance["n"]
    kind = instance["kind"]
    
    best_energy = np.inf
    best_positions = None
    
    # 1. Diverse Starts
    starts = []
    starts.append(_fcc_cluster(size))
    starts.append(_bcc_cluster(size))
    starts.append(_hcp_cluster(size))
    starts.append(_icosahedral_shell(size))
    for seed in range(8):
        starts.append(_random_compact(size, spacing=0.72, seed=seed))
    
    # Relax starts
    for s in starts:
        if _time_left(start_time) < 10:
            break
        s = s - s.mean(axis=0)
        if not np.all(np.isfinite(s)):
            continue
        e, p = _local_optimize(s, kind)
        if e < best_energy:
            best_energy = e
            best_positions = p.copy()

    if best_positions is None:
        rng = np.random.default_rng(size)
        radius = 0.72 * size ** (1.0 / 3.0)
        return rng.uniform(-radius, radius, (size, 3))

    # 2. Metropolis Basin Hopping
    remaining_time = _time_left(start_time)
    if remaining_time > 15:
        rng = np.random.default_rng(999)
        current_energy = best_energy
        current_pos = best_positions.copy()
        
        T = 0.5
        perturbation = 0.2
        
        # Estimate max iterations
        # Assume 0.2s per relax
        max_iters = int(remaining_time * 5)
        
        for it in range(max_iters):
            if _time_left(start_time) < 5:
                break
            
            direction = rng.normal(0, 1, current_pos.shape)
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction /= norm
            perturbed = current_pos + perturbation * direction
            
            e_new, pos_new = _local_optimize(perturbed, kind)
            
            if not np.isfinite(e_new):
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
                perturbation = min(0.5, perturbation * 1.001)
            else:
                T *= 0.995

    # Final validation
    if best_positions is not None:
        min_dist = np.inf
        for i in range(len(best_positions)):
            for j in range(i + 1, len(best_positions)):
                d = np.linalg.norm(best_positions[i] - best_positions[j])
                if d < min_dist:
                    min_dist = d
        if min_dist < 0.35:
            rng = np.random.default_rng(12345)
            for _ in range(50):
                candidate = best_positions + rng.normal(0, 0.05, best_positions.shape)
                min_d = np.inf
                for i in range(len(candidate)):
                    for j in range(i + 1, len(candidate)):
                        d = np.linalg.norm(candidate[i] - candidate[j])
                        if d < min_d:
                            min_d = d
                if min_d >= 0.35:
                    best_positions = candidate
                    break
    return best_positions
