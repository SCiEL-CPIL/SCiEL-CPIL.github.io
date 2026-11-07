"""Sutton-Chen transition-metal cluster global minima - multi-start basin hopping."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_DEADLINE = 320.0  # safety margin before 330s kill
_start_time = None


def _get_time():
    return time.time() - _start_time if _start_time is not None else 0.0


def _sutton_chen_energy(positions, kind):
    """Compute Sutton-Chen energy."""
    n_exp, m_exp, c = _kind_params(kind)
    n = len(positions)
    if n < 2:
        return 0.0
    # Pair distances
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    # Avoid division by zero
    mask = dist > 1e-12
    dist_safe = np.where(mask, dist, 1.0)
    
    # r^-n_exp and r^-m_exp
    inv_r_n = np.where(mask, dist_safe ** (-n_exp), 0.0)
    inv_r_m = np.where(mask, dist_safe ** (-m_exp), 0.0)
    
    # Sum over j != i for each i
    sum_n = np.sum(inv_r_n, axis=1)  # sum_j r_ij^-n_exp for each i
    sum_m = np.sum(inv_r_m, axis=1)  # sum_j r_ij^-m_exp for each i
    
    # Energy: sum_i [1/2 sum_j r^-n - c*sqrt(sum_j r^-m)]
    e_rep = 0.5 * np.sum(sum_n)
    e_att = -c * np.sum(np.sqrt(np.maximum(sum_m, 0.0)))
    return e_rep + e_att


def _kind_params(kind):
    if kind == '12-6':
        return 12, 6, 144.41
    elif kind == '9-6':
        return 9, 6, 39.432
    elif kind == '10-8':
        return 10, 8, 34.408
    else:
        return 12, 6, 144.41


def _generate_fcc_start(n, rng, scale=1.0):
    """Generate a truncated FCC-like cluster."""
    points = []
    # FCC lattice vectors
    a = 1.0  # lattice constant in reduced units
    # Generate FCC points within a sphere
    # For FCC, atoms at (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5) * a per unit cell
    # We'll generate points on an FCC lattice and keep those within a sphere
    
    # Estimate radius: for FCC packing, n ≈ (4/3)πR³ / (a³/2) * 2... 
    # Actually FCC has 4 atoms per unit cell of volume a³, so density = 4/a³
    # n ≈ 4 * (4/3)πR³ / a³ => R ≈ (3n/(16π))^(1/3) * a
    R = (3.0 * n / (16.0 * np.pi)) ** (1.0/3.0) * a * 1.2
    
    # Generate candidate points on FCC lattice
    max_coord = int(np.ceil(R / a)) + 2
    candidates = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                # Four FCC positions per unit cell
                for di, dj, dk in [(0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)]:
                    x = (i + di) * a
                    y = (j + dj) * a
                    z = (k + dk) * a
                    r = np.sqrt(x*x + y*y + z*z)
                    if r <= R:
                        candidates.append([x, y, z])
    
    candidates = np.array(candidates)
    if len(candidates) < n:
        # Add random points if not enough FCC points
        while len(candidates) < n:
            p = rng.uniform(-R, R, 3)
            if np.linalg.norm(p) <= R:
                candidates = np.vstack([candidates, p.reshape(1, 3)])
    
    # Select n points closest to center (or randomly)
    if len(candidates) > n:
        # Sort by distance from center and take n closest, with some randomness
        dists = np.linalg.norm(candidates, axis=1)
        # Add small noise to break ties and allow variety
        dists_noisy = dists + rng.uniform(0, 0.1, len(candidates))
        indices = np.argsort(dists_noisy)[:n]
        points = candidates[indices]
    else:
        points = candidates
    
    return points * scale


def _generate_random_compact(n, rng, spacing=0.72):
    """Generate a random compact cluster."""
    radius = spacing * n ** (1.0/3.0) * 0.9
    points = []
    attempts = 0
    while len(points) < n and attempts < n * 100:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.6 * spacing for q in points):
                points.append(p)
            attempts += 1
    # If not enough points, add more randomly
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    return np.array(points)


def _basin_hopping_step(positions, kind, rng, step_size=0.1):
    """Perform one basin hopping step: perturb then relax."""
    # Perturb
    perturbed = positions + rng.normal(0, step_size, positions.shape)
    # Relax
    energy, relaxed = relax(perturbed, 'sutton_chen', kind)
    return energy, relaxed


def construct(instance):
    global _start_time
    _start_time = time.time()
    
    n = instance["n"]
    kind = instance["kind"]
    
    best_energy = np.inf
    best_positions = None
    
    rng = np.random.default_rng(hash((kind, n)) % 2**32)
    
    # Generate diverse starting points
    starts = []
    
    # FCC-based starts
    for scale in [0.9, 1.0, 1.1]:
        start = _generate_fcc_start(n, rng, scale=scale)
        starts.append(start)
    
    # Random compact starts
    for _ in range(3):
        start = _generate_random_compact(n, rng)
        starts.append(start)
    
    # Spherical shell start
    try:
        # Place atoms on concentric shells
        points = []
        # Inner core
        if n >= 1:
            points.append(rng.uniform(-0.5, 0.5, 3) * 0.1)
        # Shell 1
        shell1_r = 0.72
        shell1_n = min(12, n - len(points))
        if shell1_n > 0:
            # Approximate icosahedral arrangement
            phi = (1 + np.sqrt(5)) / 2
            ico_vertices = np.array([
                [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
                [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
                [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
            ])
            ico_vertices = ico_vertices / np.linalg.norm(ico_vertices, axis=1, keepdims=True) * shell1_r
            points.extend(ico_vertices[:shell1_n])
        # Shell 2
        remaining = n - len(points)
        if remaining > 0:
            shell2_r = 1.44
            # Random points on sphere
            for _ in range(remaining):
                theta = rng.uniform(0, 2*np.pi)
                z = rng.uniform(-1, 1)
                r_xy = np.sqrt(1 - z*z)
                p = shell2_r * np.array([r_xy*np.cos(theta), r_xy*np.sin(theta), z])
                points.append(p)
        starts.append(np.array(points[:n]))
    except:
        pass
    
    # Relax all starts
    for i, start in enumerate(starts):
        if _get_time() > _DEADLINE - 30:
            break
        try:
            energy, positions = relax(start, 'sutton_chen', kind)
            if np.isfinite(energy) and energy < best_energy:
                best_energy = energy
                best_positions = positions.copy()
        except:
            pass
    
    # Basin hopping from best position
    if best_positions is not None:
        current_energy = best_energy
        current_positions = best_positions.copy()
        
        # Adaptive step size
        step_size = 0.08
        no_improve_count = 0
        max_bh_steps = 200
        
        for step in range(max_bh_steps):
            if _get_time() > _DEADLINE - 15:
                break
            
            # Perturb
            perturbed = current_positions + rng.normal(0, step_size, current_positions.shape)
            
            # Relax
            try:
                energy, relaxed = relax(perturbed, 'sutton_chen', kind)
            except:
                continue
            
            if not np.isfinite(energy):
                continue
            
            # Metropolis acceptance
            if energy < current_energy:
                current_energy = energy
                current_positions = relaxed
                no_improve_count = 0
                if energy < best_energy:
                    best_energy = energy
                    best_positions = relaxed.copy()
            else:
                # Accept with Metropolis probability (simplified: always accept small increases)
                delta_e = energy - current_energy
                if delta_e < 0.1:  # Small increase, accept to escape local minima
                    current_energy = energy
                    current_positions = relaxed
                no_improve_count += 1
            
            # Adapt step size
            if no_improve_count > 20:
                step_size *= 0.9
                no_improve_count = 0
            elif step < 20:
                step_size = max(0.05, step_size * 0.95)
    
    # Final validation: check minimum distance
    if best_positions is not None:
        diff = best_positions[:, np.newaxis, :] - best_positions[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        min_dist = np.min(dist)
        if min_dist < 0.35:
            # Try to fix by spreading out
            # Simple fix: scale positions
            center = np.mean(best_positions, axis=0)
            best_positions = (best_positions - center) * 1.1 + center
            # Re-check
            diff = best_positions[:, np.newaxis, :] - best_positions[np.newaxis, :, :]
            dist = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dist, np.inf)
            if np.min(dist) < 0.35:
                # Fallback: use a random valid configuration
                best_positions = _generate_random_compact(n, rng, spacing=0.8)
    
    return best_positions
# EVOLVE-BLOCK-END
