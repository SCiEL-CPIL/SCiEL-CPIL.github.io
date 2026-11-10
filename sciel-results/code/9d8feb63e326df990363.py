"""Morse cluster global minima: basin hopping with diverse crystalline and shell starts."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient


def _morse_energy(pos, rho):
    """Compute Morse energy E = sum_{i<j} exp(rho(1-r_ij))(exp(rho(1-r_ij)) - 2)."""
    n = len(pos)
    if n < 2:
        return 0.0
    diff = pos[:, None, :] - pos[None, :, :]
    r = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(r, np.inf)
    iu = np.triu_indices(n, k=1)
    r_ij = r[iu]
    x = rho * (1.0 - r_ij)
    e = np.exp(x) * (np.exp(x) - 2.0)
    return float(np.sum(e))


def _is_valid(pos, min_dist=0.5):
    """Check if all pairwise distances are at least min_dist."""
    n = len(pos)
    if n < 2:
        return True
    diff = pos[:, None, :] - pos[None, :, :]
    r = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(r, np.inf)
    return np.min(r) >= min_dist


def _generate_fcc(n, spacing=1.0):
    """Generate an FCC-like cluster of n atoms."""
    # FCC lattice vectors
    a = spacing
    lattice = []
    # Generate FCC points in a cube
    for i in range(-2, 3):
        for j in range(-2, 3):
            for k in range(-2, 3):
                # FCC basis: (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)
                for bx, by, bz in [(0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)]:
                    p = np.array([i, j, k]) * a + np.array([bx, by, bz]) * a
                    lattice.append(p)
    
    lattice = np.array(lattice)
    # Center the cluster
    center = np.mean(lattice, axis=0)
    lattice = lattice - center
    
    # Sort by distance from center and take n
    dists = np.linalg.norm(lattice, axis=1)
    order = np.argsort(dists)
    points = lattice[order[:n]]
    
    # If we don't have enough, add random points
    if len(points) < n:
        remaining = n - len(points)
        max_r = np.max(dists) + 0.5
        while remaining > 0:
            p = np.random.uniform(-max_r, max_r, 3)
            if all(np.linalg.norm(p - q) >= 0.8 * spacing for q in points):
                points = np.vstack([points, p])
                remaining -= 1
            else:
                break
        # If still not enough, just add random
        while len(points) < n:
            points = np.vstack([points, np.random.uniform(-max_r, max_r, 3)])
    
    return points[:n]


def _generate_bcc(n, spacing=1.0):
    """Generate a BCC-like cluster of n atoms."""
    a = spacing
    lattice = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            for k in range(-2, 3):
                # BCC basis: (0,0,0), (0.5,0.5,0.5)
                for bx, by, bz in [(0,0,0), (0.5,0.5,0.5)]:
                    p = np.array([i, j, k]) * a + np.array([bx, by, bz]) * a
                    lattice.append(p)
    
    lattice = np.array(lattice)
    center = np.mean(lattice, axis=0)
    lattice = lattice - center
    
    dists = np.linalg.norm(lattice, axis=1)
    order = np.argsort(dists)
    points = lattice[order[:n]]
    
    if len(points) < n:
        remaining = n - len(points)
        max_r = np.max(dists) + 0.5
        while remaining > 0:
            p = np.random.uniform(-max_r, max_r, 3)
            if all(np.linalg.norm(p - q) >= 0.8 * spacing for q in points):
                points = np.vstack([points, p])
                remaining -= 1
            else:
                break
        while len(points) < n:
            points = np.vstack([points, np.random.uniform(-max_r, max_r, 3)])
    
    return points[:n]


def _generate_hcp(n, spacing=1.0):
    """Generate an HCP-like cluster of n atoms."""
    a = spacing
    c = a * np.sqrt(2.0 / 3.0)
    lattice = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            for k in range(-2, 3):
                # HCP basis
                for layer in [0, 1]:
                    if layer == 0:
                        bx, by = 0, 0
                    else:
                        bx, by = 0.5, 0.5 / np.sqrt(3)
                    p = np.array([i + bx, j + by, k]) * a
                    p[2] = k * c
                    lattice.append(p)
    
    lattice = np.array(lattice)
    center = np.mean(lattice, axis=0)
    lattice = lattice - center
    
    dists = np.linalg.norm(lattice, axis=1)
    order = np.argsort(dists)
    points = lattice[order[:n]]
    
    if len(points) < n:
        remaining = n - len(points)
        max_r = np.max(dists) + 0.5
        while remaining > 0:
            p = np.random.uniform(-max_r, max_r, 3)
            if all(np.linalg.norm(p - q) >= 0.8 * spacing for q in points):
                points = np.vstack([points, p])
                remaining -= 1
            else:
                break
        while len(points) < n:
            points = np.vstack([points, np.random.uniform(-max_r, max_r, 3)])
    
    return points[:n]


def _generate_shell(n, spacing=1.0):
    """Generate a spherical shell-like cluster of n atoms."""
    points = [np.zeros(3)]
    if n == 1:
        return np.array(points)
    
    # Add atoms on concentric shells
    shell_radii = [1.0, 1.8, 2.6, 3.4, 4.2]
    for radius in shell_radii:
        if len(points) >= n:
            break
        # Fibonacci sphere for even distribution
        num_on_shell = min(n - len(points), int(4 * np.pi * radius ** 2 / (spacing ** 2)))
        num_on_shell = max(1, num_on_shell)
        
        for i in range(num_on_shell):
            if len(points) >= n:
                break
            # Fibonacci sphere
            phi = (1 + np.sqrt(5)) / 2
            theta = 2 * np.pi * i / num_on_shell
            cos_theta = 1 - (2 * i + 1) / num_on_shell
            sin_theta = np.sqrt(1 - cos_theta ** 2)
            p = radius * np.array([sin_theta * np.cos(theta), sin_theta * np.sin(theta), cos_theta])
            
            # Check minimum distance
            if all(np.linalg.norm(p - q) >= 0.8 * spacing for q in points):
                points.append(p)
    
    # Fill remaining with random
    while len(points) < n:
        p = np.random.uniform(-5, 5, 3)
        if all(np.linalg.norm(p - q) >= 0.8 * spacing for q in points):
            points.append(p)
        else:
            # Just add it, relax will fix
            points.append(p)
    
    return np.array(points[:n])


def _generate_random_compact(n, spacing=1.0, rng=None):
    """Generate a random compact cluster."""
    if rng is None:
        rng = np.random.default_rng()
    
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    points = []
    max_attempts = 10000
    attempts = 0
    
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.8 * spacing for q in points):
                points.append(p)
        attempts += 1
    
    # If not enough, add random points
    while len(points) < n:
        points.append(rng.uniform(-radius, radius, 3))
    
    return np.array(points[:n])


def _basin_hopping(pos, rho, max_steps=200, perturb_scale=0.3, rng=None):
    """Basin hopping: perturb, relax, accept if lower energy."""
    if rng is None:
        rng = np.random.default_rng()
    
    current_energy, current_pos = relax(pos.copy(), 'morse', rho)
    best_energy = current_energy
    best_pos = current_pos.copy()
    
    for step in range(max_steps):
        # Perturb
        scale = perturb_scale * (1.0 + 0.5 * np.sin(step * 0.1))
        perturbed = current_pos + rng.normal(0, scale, size=current_pos.shape)
        
        # Relax
        new_energy, new_pos = relax(perturbed, 'morse', rho)
        
        # Metropolis accept (always accept for simplicity in this context)
        if new_energy < current_energy:
            current_energy = new_energy
            current_pos = new_pos
            if new_energy < best_energy:
                best_energy = new_energy
                best_pos = new_pos.copy()
        
        # Early stop if we've been stuck
        if step > 50 and best_energy == current_energy and step % 20 == 0:
            # Try a larger perturbation
            large_perturbed = current_pos + rng.normal(0, perturb_scale * 2, size=current_pos.shape)
            large_energy, large_pos = relax(large_perturbed, 'morse', rho)
            if large_energy < current_energy:
                current_energy = large_energy
                current_pos = large_pos
                if large_energy < best_energy:
                    best_energy = large_energy
                    best_pos = large_pos.copy()
    
    return best_energy, best_pos


def construct(instance):
    """Find low-energy Morse cluster arrangements using basin hopping with diverse starts."""
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    rho = parameter if isinstance(parameter, (int, float)) else 3
    
    start_time = time.time()
    deadline = start_time + 290  # Safety margin before 300s limit
    
    spacing = 1.0
    rng = np.random.default_rng(size + 42)
    
    # Generate diverse starting configurations
    starts = []
    
    # FCC
    starts.append(_generate_fcc(size, spacing))
    # BCC
    starts.append(_generate_bcc(size, spacing))
    # HCP
    starts.append(_generate_hcp(size, spacing))
    # Shell
    starts.append(_generate_shell(size, spacing))
    
    # Random compact starts
    for i in range(4):
        starts.append(_generate_random_compact(size, spacing, rng))
    
    # Initial relax of all starts
    best_energy = np.inf
    best_pos = None
    
    for start in starts:
        if time.time() > deadline:
            break
        
        energy, pos = relax(start, 'morse', parameter)
        if energy < best_energy:
            best_energy = energy
            best_pos = pos.copy()
    
    if best_pos is None:
        # Fallback: return the first relaxed start
        best_pos = starts[0]
        best_energy, best_pos = relax(best_pos, 'morse', parameter)
    
    # Basin hopping from the best start and a few others
    bh_starts = [best_pos]
    # Add a couple of other promising starts
    for start in starts[:3]:
        if start is not best_pos:
            e, p = relax(start, 'morse', parameter)
            if e < best_energy * 1.1:  # Within 10% of best
                bh_starts.append(p)
    
    for bh_start in bh_starts:
        if time.time() > deadline - 30:  # Reserve 30s for final steps
            break
        
        # Run basin hopping with decreasing perturbation
        for scale in [0.4, 0.3, 0.2, 0.15]:
            if time.time() > deadline - 10:
                break
            
            energy, pos = _basin_hopping(bh_start, parameter, max_steps=100, 
                                          perturb_scale=scale, rng=rng)
            if energy < best_energy:
                best_energy = energy
                best_pos = pos.copy()
        
        # Update bh_start to best found
        bh_start = best_pos
    
    # Final: try a few more random perturbations from best
    if time.time() < deadline - 5:
        for _ in range(5):
            if time.time() > deadline - 2:
                break
            perturbed = best_pos + rng.normal(0, 0.1, size=best_pos.shape)
            energy, pos = relax(perturbed, 'morse', parameter)
            if energy < best_energy:
                best_energy = energy
                best_pos = pos.copy()
    
    # Ensure validity: check minimum distance
    if not _is_valid(best_pos, 0.5):
        # Try to fix by relaxing again
        best_energy, best_pos = relax(best_pos, 'morse', parameter)
        if not _is_valid(best_pos, 0.5):
            # Last resort: return a valid random configuration
            valid_pos = _generate_random_compact(size, spacing, rng)
            _, valid_pos = relax(valid_pos, 'morse', parameter)
            best_pos = valid_pos
    
    return best_pos
# EVOLVE-BLOCK-END
