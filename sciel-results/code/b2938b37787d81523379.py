"""Rewritten program: lattice-based multi-start with basin hopping for Morse clusters."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

_DEADLINE = time.time() + 315.0

def _time_left():
    return _DEADLINE - time.time()

def _morse_energy(pos, rho):
    """Compute total Morse energy."""
    n = len(pos)
    if n < 2:
        return 0.0
    d = pdist(pos)
    x = rho * (1.0 - d)
    e = np.exp(x) * (np.exp(x) - 2.0)
    return float(np.sum(e))

def _min_dist_ok(pos, min_d=0.5):
    if len(pos) < 2:
        return True
    d = pdist(pos)
    return np.min(d) >= min_d - 1e-6

def _fix_min_dist(pos, min_d=0.5, rng=None, max_iter=100):
    """Separate atoms that are too close."""
    if rng is None:
        rng = np.random.default_rng()
    pos = pos.copy()
    for _ in range(max_iter):
        d = pdist(pos)
        idx = np.argwhere(np.triu(np.ones((len(pos), len(pos)), dtype=bool), k=1))
        pairs = np.argwhere(d < min_d - 1e-8)
        if len(pairs) == 0:
            break
        for i, j in pairs:
            if i >= len(pos) or j >= len(pos):
                continue
            diff = pos[j] - pos[i]
            dist = np.linalg.norm(diff)
            if dist < 1e-10:
                diff = rng.standard_normal(3)
                dist = np.linalg.norm(diff)
            diff = diff / dist
            push = (min_d - dist) * 0.5 + 0.01
            pos[i] -= diff * push
            pos[j] += diff * push
    return pos

def _fcc_cluster(n, rng=None):
    """Generate FCC-like cluster with nearest-neighbor distance ~1."""
    if rng is None:
        rng = np.random.default_rng()
    # FCC basis vectors
    basis = np.array([
        [0, 0, 0],
        [0.5, 0.5, 0],
        [0.5, 0, 0.5],
        [0, 0.5, 0.5],
    ])
    # Generate points in a cubic region
    side = max(2, int(np.ceil(n ** (1/3.0))))
    points = []
    for ix in range(side + 1):
        for iy in range(side + 1):
            for iz in range(side + 1):
                for b in basis:
                    p = np.array([ix, iy, iz]) + b
                    points.append(p)
                if len(points) >= n + 10:
                    break
                if len(points) >= n + 10:
                    break
                if len(points) >= n + 10:
                    break
        if len(points) >= n + 10:
            break
        if len(points) >= n + 10:
            break
    
    points = np.array(points)
    # Center
    points -= points.mean(axis=0)
    # Keep n closest to center
    dists = np.linalg.norm(points, axis=1)
    order = np.argsort(dists)
    points = points[order[:n]]
    # Scale so nearest neighbor ~ 1
    if len(points) > 1:
        d = pdist(points)
        d = d[d > 1e-10]
        if len(d) > 0:
            min_d = np.min(d)
            if min_d > 0:
                points *= 1.0 / min_d
    return points

def _bcc_cluster(n, rng=None):
    """Generate BCC-like cluster."""
    if rng is None:
        rng = np.random.default_rng()
    basis = np.array([
        [0, 0, 0],
        [0.5, 0.5, 0.5],
    ])
    side = max(2, int(np.ceil(n ** (1/3.0))))
    points = []
    for ix in range(side + 1):
        for iy in range(side + 1):
            for iz in range(side + 1):
                for b in basis:
                    p = np.array([ix, iy, iz]) + b
                    points.append(p)
                if len(points) >= n + 10:
                    break
                if len(points) >= n + 10:
                    break
                if len(points) >= n + 10:
                    break
        if len(points) >= n + 10:
            break
        if len(points) >= n + 10:
            break
    
    points = np.array(points)
    points -= points.mean(axis=0)
    dists = np.linalg.norm(points, axis=1)
    order = np.argsort(dists)
    points = points[order[:n]]
    if len(points) > 1:
        d = pdist(points)
        d = d[d > 1e-10]
        if len(d) > 0:
            min_d = np.min(d)
            if min_d > 0:
                points *= 1.0 / min_d
    return points

def _hcp_cluster(n, rng=None):
    """Generate HCP-like cluster."""
    if rng is None:
        rng = np.random.default_rng()
    a = 1.0  # nearest neighbor distance
    c = np.sqrt(8.0 / 3.0) * a  # ideal HCP c/a ratio
    basis = np.array([
        [0, 0, 0],
        [a, 0, 0],
        [a/2, a*np.sqrt(3)/2, 0],
        [a/2, a*np.sqrt(3)/2, c],
        [a, a*np.sqrt(3)/2, c],
        [a*1.5, a*np.sqrt(3)/2, c],
    ])
    # HCP lattice vectors
    v1 = np.array([a, 0, 0])
    v2 = np.array([a/2, a*np.sqrt(3)/2, 0])
    v3 = np.array([0, 0, c])
    
    points = []
    side = max(2, int(np.ceil(n ** (1/3.0))))
    for ix in range(side + 1):
        for iy in range(side + 1):
            for iz in range(side + 1):
                for b in basis:
                    p = ix * v1 + iy * v2 + iz * v3 + b
                    points.append(p)
                if len(points) >= n + 10:
                    break
                if len(points) >= n + 10:
                    break
                if len(points) >= n + 10:
                    break
        if len(points) >= n + 10:
            break
        if len(points) >= n + 10:
            break
    
    points = np.array(points)
    points -= points.mean(axis=0)
    dists = np.linalg.norm(points, axis=1)
    order = np.argsort(dists)
    points = points[order[:n]]
    if len(points) > 1:
        d = pdist(points)
        d = d[d > 1e-10]
        if len(d) > 0:
            min_d = np.min(d)
            if min_d > 0:
                points *= 1.0 / min_d
    return points

def _random_cluster(n, rng, min_d=0.55):
    """Random cluster with minimum distance."""
    radius = n ** (1/3.0) * 0.7
    points = []
    max_attempts = 5000
    attempts = 0
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        ok = True
        for q in points:
            if np.linalg.norm(p - q) < min_d:
                ok = False
                break
        if ok:
            points.append(p)
        attempts += 1
    while len(points) < n:
        p = rng.uniform(-radius * 2, radius * 2, 3)
        points.append(p)
    return np.array(points, dtype=float)

def _relax_and_check(pos, rho, potential='morse'):
    """Relax and ensure minimum distance."""
    try:
        energy, new_pos = relax(pos, potential, rho)
        if not _min_dist_ok(new_pos):
            new_pos = _fix_min_dist(new_pos, min_d=0.5)
            energy = _morse_energy(new_pos, rho)
        return energy, new_pos
    except Exception:
        return _morse_energy(pos, rho), pos

def _basin_hop(pos, rho, rng, n_steps=20, T0=0.5, T1=0.05):
    """Simple basin hopping: perturb, relax, accept if better or with probability."""
    current_pos = pos.copy()
    current_energy = _morse_energy(pos, rho)
    best_pos = pos.copy()
    best_energy = current_energy
    
    for step in range(n_steps):
        if _time_left() < 5:
            break
        T = T0 * (T1 / T0) ** (step / max(1, n_steps - 1))
        # Perturb
        sigma = 0.2 + 0.3 * (1 - step / max(1, n_steps))
        perturbed = current_pos + sigma * rng.standard_normal((len(pos), 3))
        # Relax
        e_new, new_pos = _relax_and_check(perturbed, rho)
        # Accept
        delta_e = e_new - current_energy
        if delta_e < 0 or (T > 0 and rng.random() < np.exp(-delta_e / T)):
            current_pos = new_pos
            current_energy = e_new
            if e_new < best_energy:
                best_energy = e_new
                best_pos = new_pos.copy()
    
    return best_energy, best_pos

def construct(instance):
    global _DEADLINE
    _DEADLINE = time.time() + 315.0
    
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind"))
    potential = instance["potential"]
    
    rng = np.random.default_rng(n * 12345 + int(rho) * 6789)
    
    best_energy = np.inf
    best_pos = None
    
    # Generate diverse initial configurations
    generators = [
        lambda: _fcc_cluster(n, rng),
        lambda: _bcc_cluster(n, rng),
        lambda: _hcp_cluster(n, rng),
        lambda: _random_cluster(n, rng, min_d=0.55),
        lambda: _random_cluster(n, rng, min_d=0.6),
        lambda: _random_cluster(n, rng, min_d=0.65),
    ]
    
    # For larger n, also try perturbed lattice versions
    if n > 40:
        generators.extend([
            lambda: _fcc_cluster(n, rng) + 0.05 * rng.standard_normal((n, 3)),
            lambda: _bcc_cluster(n, rng) + 0.05 * rng.standard_normal((n, 3)),
        ])
    
    n_starts = min(len(generators), max(6, 30 - n // 10))
    
    for i in range(n_starts):
        if _time_left() < 10:
            break
        try:
            pos = generators[i]()
            if len(pos) != n:
                pos = pos[:n] if len(pos) > n else np.vstack([pos, rng.uniform(-1, 1, (n - len(pos), 3))])
            pos = _fix_min_dist(pos, min_d=0.5, rng=rng)
            energy, new_pos = _relax_and_check(pos, rho, potential)
            if energy < best_energy:
                best_energy = energy
                best_pos = new_pos.copy()
        except Exception:
            continue
    
    if best_pos is None:
        best_pos = _random_cluster(n, rng, min_d=0.55)
        best_pos = _fix_min_dist(best_pos, min_d=0.5, rng=rng)
        best_energy, best_pos = _relax_and_check(best_pos, rho, potential)
    
    # Phase 2: basin hopping from the best configuration
    if _time_left() > 15:
        # Determine how many basin hopping steps we can afford
        # Each step involves a relax call which takes time proportional to n
        est_time_per_step = 0.01 * n / 10  # rough estimate
        max_bh_steps = min(50, int((_time_left() - 10) / max(0.1, est_time_per_step)))
        max_bh_steps = max(5, max_bh_steps)
        
        # Run basin hopping
        bh_energy, bh_pos = _basin_hop(best_pos, rho, rng, n_steps=max_bh_steps)
        if bh_energy < best_energy:
            best_energy = bh_energy
            best_pos = bh_pos.copy()
    
    # Phase 3: if still time, try more random restarts
    if _time_left() > 10:
        for i in range(20):
            if _time_left() < 5:
                break
            sigma = 0.3 * (1.0 - i / 20.0)
            perturbed = best_pos + sigma * rng.standard_normal((n, 3))
            perturbed = _fix_min_dist(perturbed, min_d=0.5, rng=rng)
            energy, new_pos = _relax_and_check(perturbed, rho, potential)
            if energy < best_energy:
                best_energy = energy
                best_pos = new_pos.copy()
    
    # Final validation
    if len(best_pos) != n:
        best_pos = np.random.uniform(-1, 1, (n, 3))
        best_pos = _fix_min_dist(best_pos, min_d=0.5, rng=rng)
        _, best_pos = _relax_and_check(best_pos, rho, potential)
    
    return best_pos
# EVOLVE-BLOCK-END
