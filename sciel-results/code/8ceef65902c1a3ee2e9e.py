"""Morse cluster global minima: basin hopping with structured seeds and gradient-based local relaxation."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

_START_TIME = time.time()
_HARD_DEADLINE = _START_TIME + 280.0  # 20s safety margin before 300s kill


def _time_left():
    return _HARD_DEADLINE - time.time()


def _morse_energy(pos, rho):
    """Compute Morse energy exactly as defined in the task."""
    n = len(pos)
    if n < 2:
        return 0.0
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, 1e10)
    # Only compute for valid pairs (i < j)
    iu = np.triu_indices(n, 1)
    d = dist[iu]
    val = np.exp(rho * (1.0 - d))
    e = val * (val - 2.0)
    return float(np.sum(e))


def _morse_gradient(pos, rho):
    """Compute Morse gradient. Returns (energy, gradient)."""
    n = len(pos)
    if n < 2:
        return 0.0, np.zeros_like(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, 1e10)
    
    iu = np.triu_indices(n, 1)
    d = dist[iu]
    
    # Avoid numerical issues with very small distances
    d_safe = np.where(d > 1e-6, d, 1e-6)
    
    val = np.exp(rho * (1.0 - d_safe))
    e = val * (val - 2.0)
    
    # Gradient: dE/dr = rho * val * (2*val - 1) ... wait let me recompute
    # E = exp(rho(1-r)) * (exp(rho(1-r)) - 2)
    # Let u = rho(1-r), so E = exp(u)(exp(u) - 2) = exp(2u) - 2exp(u)
    # dE/dr = dE/du * du/dr = (2exp(2u) - 2exp(u)) * (-rho)
    #      = -rho * 2 * exp(u) * (exp(u) - 1)
    #      = -2*rho*val*(val - 1)
    # dE/dpos_i = dE/dr * (pos_i - pos_j)/r
    # But we need to be careful with the sign and the vector
    
    # For pair (i,j), force on i is in direction (pos_i - pos_j)/r * dE/dr
    # Actually, dE/dpos_i = sum_j dE/d(r_ij) * (pos_i - pos_j)/r_ij
    # where dE/dr = -2*rho*val*(val-1)
    
    de_dr = -2.0 * rho * val * (val - 1.0)
    
    # Build gradient
    grad = np.zeros_like(pos)
    for k in range(len(iu[0])):
        i, j = iu[0][k], iu[1][k]
        rij = diff[i, j]
        dr = de_dr[k] / d_safe[k]
        grad[i] += dr * rij
        grad[j] -= dr * rij
    
    return float(np.sum(e)), grad


def _generate_fcc(n):
    """Generate FCC lattice points."""
    points = []
    a = 1.0
    # Generate enough lattice points
    for x in range(-4, 5):
        for y in range(-4, 5):
            for z in range(-4, 5):
                # FCC: (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0) in units of a
                for dx, dy, dz in [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]:
                    p = np.array([x + dx, y + dy, z + dz]) * a
                    points.append(p)
    points = np.array(points)
    # Sort by distance from origin
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    points = points[idx]
    # Center
    points = points - np.mean(points[:n], axis=0)
    return points[:n]


def _generate_bcc(n):
    """Generate BCC lattice points."""
    points = []
    a = 1.0
    for x in range(-4, 5):
        for y in range(-4, 5):
            for z in range(-4, 5):
                for dx, dy, dz in [(0, 0, 0), (0.5, 0.5, 0.5)]:
                    p = np.array([x + dx, y + dy, z + dz]) * a
                    points.append(p)
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    points = points[idx]
    points = points - np.mean(points[:n], axis=0)
    return points[:n]


def _generate_icosahedral(n):
    """Generate icosahedral shell points."""
    points = []
    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2
    # Icosahedron vertices
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
    ])
    # Normalize to unit sphere
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    points.append(np.zeros(3))  # center
    
    # Add shells
    shell_radius = 1.0
    for shell in range(1, 10):
        r = shell_radius * shell
        # Generate more points on the icosahedron directions
        for v in verts:
            p = v * r
            points.append(p)
        # Add intermediate points
        for i in range(len(verts)):
            for j in range(i + 1, len(verts)):
                mid = (verts[i] + verts[j]) / 2
                mid_norm = np.linalg.norm(mid)
                if mid_norm > 1e-6:
                    mid = mid / mid_norm
                    points.append(mid * r)
    
    points = np.array(points)
    if len(points) < n:
        # Fill with random points
        rng = np.random.default_rng(42)
        while len(points) < n:
            p = rng.normal(0, 1, 3)
            p = p / np.linalg.norm(p) * shell_radius * (len(points) // 20 + 1)
            points = np.vstack([points, p])
    
    # Center
    points = points[:n]
    points = points - np.mean(points, axis=0)
    return points


def _generate_random_compact(n, rng):
    """Generate random compact cluster."""
    radius = 1.0 * n ** (1.0 / 3.0) * 0.8
    points = []
    for _ in range(n):
        for attempt in range(100):
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                if all(np.linalg.norm(p - q) >= 0.55 for q in points):
                    points.append(p)
                    break
        else:
            # Force add
            p = rng.uniform(-radius, radius, 3)
            points.append(p)
    return np.array(points)


def _basin_hopping(pos, rho, n, rng, max_iters=50, kick_scale=0.3):
    """Basin hopping with gradient-based local relaxation."""
    best_pos = pos.copy()
    best_energy = _morse_energy(pos, rho)
    
    for it in range(max_iters):
        if _time_left() < 5.0:
            break
        
        # Kick
        kick = rng.normal(0, kick_scale, pos.shape)
        new_pos = pos + kick
        
        # Local relaxation using gradient descent
        local_pos = new_pos.copy()
        for gd_iter in range(100):
            if _time_left() < 3.0:
                break
            e, grad = _morse_gradient(local_pos, rho)
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-8:
                break
            # Adaptive step size
            step = 0.1 / (1.0 + grad_norm)
            new_local = local_pos - step * grad
            new_e = _morse_energy(new_local, rho)
            if new_e < e:
                local_pos = new_local
            else:
                # Try smaller step
                step = 0.01 / (1.0 + grad_norm)
                new_local = local_pos - step * grad
                new_e = _morse_energy(new_local, rho)
                if new_e < e:
                    local_pos = new_local
                else:
                    break
        
        # Also try the provided relax function for better local optimization
        e_relaxed, relaxed_pos = relax(local_pos, "morse", rho)
        if e_relaxed < _morse_energy(local_pos, rho):
            local_pos = relaxed_pos
            local_energy = e_relaxed
        else:
            local_energy = _morse_energy(local_pos, rho)
        
        if local_energy < best_energy:
            best_energy = local_energy
            best_pos = local_pos.copy()
        
        pos = local_pos
        # Reduce kick scale over time
        kick_scale *= 0.95
    
    return best_pos, best_energy


def construct(instance):
    """Find global minimum of Morse cluster using basin hopping with multiple seeds."""
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind", 3))
    rng = np.random.default_rng(n * 1000 + int(rho))
    
    # Generate diverse initial structures
    seeds = []
    
    # FCC
    try:
        seeds.append(_generate_fcc(n))
    except:
        pass
    
    # BCC
    try:
        seeds.append(_generate_bcc(n))
    except:
        pass
    
    # Icosahedral
    try:
        seeds.append(_generate_icosahedral(n))
    except:
        pass
    
    # Random compact (multiple)
    for i in range(3):
        try:
            seeds.append(_generate_random_compact(n, rng))
        except:
            pass
    
    # Filter valid seeds (minimum distance constraint)
    valid_seeds = []
    for s in seeds:
        if len(s) == n:
            # Check minimum distance
            min_dist = np.inf
            for i in range(n):
                for j in range(i + 1, n):
                    d = np.linalg.norm(s[i] - s[j])
                    if d < min_dist:
                        min_dist = d
            if min_dist >= 0.5:
                valid_seeds.append(s)
            else:
                # Try to fix by scaling
                scale = 0.5 / min_dist * 1.01
                s_scaled = s * scale
                # Recheck
                min_dist2 = np.inf
                for i in range(n):
                    for j in range(i + 1, n):
                        d = np.linalg.norm(s_scaled[i] - s_scaled[j])
                        if d < min_dist2:
                            min_dist2 = d
                if min_dist2 >= 0.5:
                    valid_seeds.append(s_scaled)
    
    if not valid_seeds:
        # Fallback: random
        valid_seeds.append(_generate_random_compact(n, rng))
    
    # Run basin hopping on each seed
    best_pos = None
    best_energy = np.inf
    
    for seed in valid_seeds:
        if _time_left() < 10.0:
            break
        
        pos, energy = _basin_hopping(seed, rho, n, rng, max_iters=30, kick_scale=0.3)
        
        if energy < best_energy:
            best_energy = energy
            best_pos = pos
    
    # Final polish with relax
    if best_pos is not None and _time_left() > 5.0:
        final_energy, final_pos = relax(best_pos, "morse", rho)
        if final_energy < best_energy:
            best_pos = final_pos
            best_energy = final_energy
    
    return best_pos
# EVOLVE-BLOCK-END
