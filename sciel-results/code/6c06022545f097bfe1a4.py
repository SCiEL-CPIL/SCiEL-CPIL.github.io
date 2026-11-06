"""Morse cluster global minima: structured seeds + basin hopping with adaptive cooling."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

_START_TIME = time.time()
_HARD_DEADLINE = _START_TIME + 285.0  # 15s safety margin before 300s kill


def _time_left():
    return _HARD_DEADLINE - time.time()


def _morse_energy(pos, rho):
    """Compute Morse energy exactly."""
    n = len(pos)
    if n < 2:
        return 0.0
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, 1e10)
    # Only compute for valid pairs (distance > 0.5 to avoid numerical issues)
    mask = dist > 0.5
    dist_safe = np.where(mask, dist, 1e10)
    x = rho * (1.0 - dist_safe)
    e = np.exp(x)
    pair_e = e * (e - 2.0)
    # Sum upper triangle
    total = 0.5 * np.sum(pair_e) - 0.5 * np.trace(pair_e)
    return float(total)


def _fcc_seed(n, spacing=1.0):
    """Generate an FCC-like compact cluster."""
    # Generate points in a cubic grid with FCC offsets
    points = []
    # FCC lattice: points at (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5) * spacing
    offsets = [
        [0, 0, 0],
        [0.5, 0.5, 0],
        [0.5, 0, 0.5],
        [0, 0.5, 0.5],
    ]
    # Determine grid size
    side = int(np.ceil(n ** (1.0/3.0)) + 2)
    for i in range(-side, side+1):
        for j in range(-side, side+1):
            for k in range(-side, side+1):
                for off in offsets:
                    p = np.array([i + off[0], j + off[1], k + off[2]]) * spacing
                    if len(points) < n:
                        # Check minimum distance
                        valid = True
                        for q in points:
                            if np.linalg.norm(p - q) < 0.5 * spacing:
                                valid = False
                                break
                        if valid:
                            points.append(p)
                    if len(points) >= n:
                        break
                if len(points) >= n:
                    break
            if len(points) >= n:
                break
        if len(points) >= n:
            break
    if len(points) < n:
        # Fill with random points
        while len(points) < n:
            p = np.random.uniform(-2*spacing, 2*spacing, 3)
            valid = True
            for q in points:
                if np.linalg.norm(p - q) < 0.5 * spacing:
                    valid = False
                    break
            if valid:
                points.append(p)
    return np.array(points[:n])


def _bcc_seed(n, spacing=1.0):
    """Generate a BCC-like compact cluster."""
    points = []
    offsets = [
        [0, 0, 0],
        [0.5, 0.5, 0.5],
    ]
    side = int(np.ceil(n ** (1.0/3.0)) + 2)
    for i in range(-side, side+1):
        for j in range(-side, side+1):
            for k in range(-side, side+1):
                for off in offsets:
                    p = np.array([i + off[0], j + off[1], k + off[2]]) * spacing
                    if len(points) < n:
                        valid = True
                        for q in points:
                            if np.linalg.norm(p - q) < 0.5 * spacing:
                                valid = False
                                break
                        if valid:
                            points.append(p)
                    if len(points) >= n:
                        break
                if len(points) >= n:
                    break
            if len(points) >= n:
                break
        if len(points) >= n:
            break
    if len(points) < n:
        while len(points) < n:
            p = np.random.uniform(-2*spacing, 2*spacing, 3)
            valid = True
            for q in points:
                if np.linalg.norm(p - q) < 0.5 * spacing:
                    valid = False
                    break
            if valid:
                points.append(p)
    return np.array(points[:n])


def _icosahedral_seed(n, spacing=1.0):
    """Generate an icosahedral-like cluster (good for medium/high rho)."""
    # Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
    ])
    ico_verts = ico_verts / np.linalg.norm(ico_verts[0]) * spacing * 0.5
    
    points = [np.zeros(3)]  # center
    # First shell: 12 vertices
    for v in ico_verts:
        if len(points) < n:
            points.append(v.copy())
    
    # Second shell: 30 points (edge midpoints scaled)
    if len(points) < n:
        # Generate edge midpoints of icosahedron
        edges = []
        for i in range(12):
            for j in range(i+1, 12):
                d = np.linalg.norm(ico_verts[i] - ico_verts[j])
                if d < spacing * 1.2:  # close pairs
                    mid = (ico_verts[i] + ico_verts[j]) / 2
                    # Scale to second shell radius
                    r = np.linalg.norm(mid)
                    if r > 1e-6:
                        mid = mid / r * spacing * 0.85
                        edges.append(mid)
        for e in edges:
            if len(points) < n:
                valid = True
                for q in points:
                    if np.linalg.norm(e - q) < 0.5 * spacing:
                        valid = False
                        break
                if valid:
                    points.append(e.copy())
    
    # Fill remaining with random points near the cluster
    if len(points) < n:
        radius = spacing * 1.2
        while len(points) < n:
            p = np.random.uniform(-radius, radius, 3)
            valid = True
            for q in points:
                if np.linalg.norm(p - q) < 0.5 * spacing:
                    valid = False
                    break
            if valid:
                points.append(p)
    
    return np.array(points[:n])


def _random_compact_seed(n, spacing=1.0, rng=None):
    """Generate a random compact cluster."""
    if rng is None:
        rng = np.random.default_rng()
    radius = spacing * n ** (1.0/3.0) * 0.7
    points = []
    attempts = 0
    while len(points) < n and attempts < n * 200:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            valid = True
            for q in points:
                if np.linalg.norm(p - q) < 0.5 * spacing:
                    valid = False
                    break
            if valid:
                points.append(p)
        attempts += 1
    # Fill any remaining
    while len(points) < n:
        p = rng.uniform(-radius*1.5, radius*1.5, 3)
        valid = True
        for q in points:
            if np.linalg.norm(p - q) < 0.5 * spacing:
                valid = False
                break
        if valid:
            points.append(p)
    return np.array(points[:n])


def _basin_hopping(pos, rho, n_iterations=200, rng=None):
    """Basin hopping: perturb, relax, accept/reject."""
    if rng is None:
        rng = np.random.default_rng()
    
    current_pos = pos.copy()
    current_energy = _morse_energy(current_pos, rho)
    
    # Relax initial position
    energy, current_pos = relax(current_pos, 'morse', rho)
    current_energy = energy
    
    best_pos = current_pos.copy()
    best_energy = current_energy
    
    # Adaptive perturbation size
    perturb_size = 0.3
    n_accepted = 0
    n_rejected = 0
    
    for i in range(n_iterations):
        if _time_left() < 2.0:
            break
        
        # Perturb
        noise = rng.normal(0, perturb_size, current_pos.shape)
        perturbed = current_pos + noise
        
        # Relax
        try:
            new_energy, new_pos = relax(perturbed, 'morse', rho)
        except:
            new_energy = current_energy + 1.0
            new_pos = perturbed
        
        # Metropolis accept/reject
        delta = new_energy - current_energy
        if delta < 0 or rng.random() < np.exp(-delta / 0.5):
            current_pos = new_pos
            current_energy = new_energy
            n_accepted += 1
            if current_energy < best_energy:
                best_energy = current_energy
                best_pos = current_pos.copy()
        else:
            n_rejected += 1
        
        # Adaptive cooling
        total = n_accepted + n_rejected
        if total > 20:
            accept_rate = n_accepted / total
            if accept_rate < 0.1:
                perturb_size *= 0.8
            elif accept_rate > 0.7:
                perturb_size *= 1.2
            perturb_size = max(0.05, min(1.0, perturb_size))
    
    return best_pos, best_energy


def construct(instance):
    """Find low-energy Morse cluster configuration."""
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    rho = parameter if parameter else 10.0
    
    rng = np.random.default_rng(size + 42)
    
    # Generate diverse starting seeds
    seeds = []
    
    # Icosahedral (good for medium/high rho)
    if rho >= 6:
        seeds.append(_icosahedral_seed(size, 1.0))
    
    # FCC (good for low rho)
    if rho <= 6:
        seeds.append(_fcc_seed(size, 1.0))
    
    # BCC
    seeds.append(_bcc_seed(size, 1.0))
    
    # Random compact
    for _ in range(3):
        seeds.append(_random_compact_seed(size, 1.0, rng))
    
    # Run basin hopping on each seed, tracking best
    best_pos = None
    best_energy = np.inf
    
    for seed_idx, seed in enumerate(seeds):
        if _time_left() < 5.0:
            break
        
        # Determine iterations based on time budget
        remaining = _time_left()
        n_iter = max(10, int(remaining * 15))  # ~15 iterations per second
        n_iter = min(n_iter, 300)
        
        try:
            pos, energy = _basin_hopping(seed, rho, n_iterations=n_iter, rng=rng)
            if energy < best_energy:
                best_energy = energy
                best_pos = pos.copy()
        except Exception:
            continue
    
    # If we didn't find anything valid, return the first seed relaxed
    if best_pos is None:
        try:
            _, best_pos = relax(seeds[0], 'morse', rho)
        except:
            best_pos = seeds[0]
    
    # Final relaxation
    try:
        _, final_pos = relax(best_pos, 'morse', rho)
        return final_pos
    except:
        return best_pos
# EVOLVE-BLOCK-END
