"""Morse cluster global minima finder - DIRECT structural approach.

Strategy: For Morse potentials, the optimal arrangement depends on rho:
- Low rho (long range): compact, dense packing (FCC/HCP-like)
- Medium rho: FCC-like with some icosahedral character
- High rho (short range): icosahedral shells, low coordination

We build explicit structural templates appropriate for each rho, then
relax them with multiple optimization methods and random restarts.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient


def _morse_energy(pos, rho):
    """Compute Morse energy for a configuration."""
    n = len(pos)
    if n < 2:
        return 0.0
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    upper = np.triu_indices(n, 1)
    r = dists[upper]
    val = rho * (1.0 - r)
    exp_val = np.exp(val)
    e = np.sum(exp_val * (exp_val - 2.0))
    return float(e)


def _morse_energy_gradient(pos, rho):
    """Compute gradient of Morse energy (vectorized)."""
    n = len(pos)
    grad = np.zeros_like(pos)
    if n < 2:
        return grad
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    upper = np.triu_indices(n, 1)
    r = dists[upper]
    val = rho * (1.0 - r)
    exp_val = np.exp(val)
    # dE/dr = -2 * exp(2*rho*(1-r))
    pair_force = -2.0 * exp_val * exp_val
    
    # Vectorized gradient computation
    for i in range(n):
        for j in range(i + 1, n):
            rij = pos[i] - pos[j]
            ri = dists[i, j]
            if ri > 1e-10:
                f = pair_force[j - i + i * (n - 1) - i * (i + 1) // 2]
                grad[i] += f * rij / ri
                grad[j] -= f * rij / ri
    return grad


def _min_dist_check(pos, min_d=0.5):
    """Check minimum pairwise distance."""
    n = len(pos)
    if n < 2:
        return True
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    return np.min(dists) >= min_d - 1e-9


def _ensure_min_dist(pos, min_d=0.51, max_iter=200):
    """Separate atoms that are too close."""
    pos = pos.copy()
    n = len(pos)
    for _ in range(max_iter):
        min_dist = np.inf
        min_pair = (0, 1)
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < min_dist:
                    min_dist = d
                    min_pair = (i, j)
        if min_dist >= min_d:
            break
        i, j = min_pair
        rij = pos[i] - pos[j]
        r = np.linalg.norm(rij)
        if r < 1e-10:
            rij = np.array([1.0, 0, 0])
            r = 1.0
        push = (min_d - r + 0.01) / 2.0
        pos[i] += push * rij / r
        pos[j] -= push * rij / r
    return pos


def _fcc_points(n, spacing=1.0):
    """Generate FCC lattice points, keeping n closest to origin."""
    pts = []
    a = spacing
    max_r = int(np.ceil(n ** (1/3))) + 4
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            for k in range(-max_r, max_r + 1):
                for dx, dy, dz in [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]:
                    p = np.array([i + dx, j + dy, k + dz]) * a
                    pts.append(p)
    pts = np.array(pts)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)[:n]
    return pts[idx].copy()


def _hcp_points(n, spacing=1.0):
    """Generate HCP lattice points, keeping n closest to origin."""
    pts = []
    a = spacing
    c_over_a = np.sqrt(8.0 / 3.0)
    max_r = int(np.ceil(n ** (1/3))) + 4
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            for k in range(-max_r, max_r + 1):
                p1 = np.array([i, j, k * c_over_a]) * a
                p2 = np.array([i + 1/3.0, j + 2/3.0, (k + 0.5) * c_over_a]) * a
                pts.append(p1)
                pts.append(p2)
    pts = np.array(pts)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)[:n]
    return pts[idx].copy()


def _bcc_points(n, spacing=1.0):
    """Generate BCC lattice points, keeping n closest to origin."""
    pts = []
    a = spacing * np.sqrt(3.0) / 2.0
    max_r = int(np.ceil(n ** (1/3))) + 4
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            for k in range(-max_r, max_r + 1):
                p1 = np.array([i, j, k]) * a
                p2 = np.array([i + 0.5, j + 0.5, k + 0.5]) * a
                pts.append(p1)
                pts.append(p2)
    pts = np.array(pts)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)[:n]
    return pts[idx].copy()


def _icosahedral_cluster(n, spacing=1.0):
    """Generate icosahedral cluster with shells."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    edge_len = 2.0
    scale = spacing / edge_len
    verts = verts * scale
    
    pts = [np.zeros(3)]
    
    if n > 1:
        for v in verts:
            pts.append(v.copy())
    
    if n > 13:
        faces = [
            [0, 4, 8], [0, 9, 4], [0, 1, 5], [0, 5, 10],
            [0, 10, 9], [0, 8, 7], [0, 7, 11], [0, 11, 9],
            [1, 6, 8], [1, 8, 4], [1, 2, 3], [1, 3, 5],
            [2, 6, 7], [2, 7, 3], [2, 3, 11], [2, 11, 6],
            [3, 7, 11], [3, 11, 5], [3, 5, 2], [4, 9, 8]
        ]
        for f in faces:
            center = (verts[f[0]] + verts[f[1]] + verts[f[2]]) / 3.0
            r = np.linalg.norm(center)
            if r > 1e-10:
                center = center / r * r * 1.15
            pts.append(center)
    
    if n > 33:
        edges = set()
        for i in range(12):
            for j in range(i + 1, 12):
                d = np.linalg.norm(verts[i] - verts[j])
                if d < 1.5 * scale:
                    mid = (verts[i] + verts[j]) / 2.0
                    r = np.linalg.norm(mid)
                    if r > 1e-10:
                        mid = mid / r * r * 1.3
                    pts.append(mid)
    
    if n > 63:
        verts2 = verts * 2.0
        for v in verts2:
            pts.append(v.copy())
        if n > 75:
            for f in faces:
                center = (verts2[f[0]] + verts2[f[1]] + verts2[f[2]]) / 3.0
                pts.append(center)
    
    pts = np.array(pts[:n])
    return pts


def _spherical_shell_cluster(n, spacing=1.0):
    """Distribute n points on/near a sphere using Fibonacci sphere."""
    shell_sizes = []
    remaining = n
    shell_radii = []
    
    if n <= 13:
        shell_sizes = [min(n, 13)]
        shell_radii = [0.0]
        remaining = n - shell_sizes[0]
    else:
        s1 = min(13, n)
        shell_sizes.append(s1)
        shell_radii.append(0.0)
        remaining -= s1
        if remaining > 0:
            s2 = min(20, remaining)
            shell_sizes.append(s2)
            shell_radii.append(spacing)
            remaining -= s2
            if remaining > 0:
                shell_sizes.append(remaining)
                shell_radii.append(spacing * 2.0)
    
    pts = []
    for size, radius in zip(shell_sizes, shell_radii):
        if size == 0:
            continue
        if radius < 1e-10:
            pts.append(np.zeros(3))
            phi = (1 + np.sqrt(5)) / 2
            verts = np.array([
                [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
                [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
                [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
            ], dtype=float)
            verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
            edge_len = 2.0
            scale = spacing / edge_len
            for v in verts[:size - 1]:
                pts.append(v * scale)
        else:
            golden_angle = np.pi * (3.0 - np.sqrt(5.0))
            for i in range(size):
                z = 1.0 - 2.0 * (i + 0.5) / size
                r = np.sqrt(max(0, 1.0 - z * z))
                theta = golden_angle * i
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                pts.append(np.array([x, y, z]) * radius)
    
    pts = np.array(pts[:n])
    return pts


def _random_dense(n, rng, spacing=1.0):
    """Generate random dense packing."""
    pts = []
    radius = spacing * (n ** (1/3)) * 0.8
    min_d = 0.55 * spacing
    max_attempts = 50000
    attempts = 0
    while len(pts) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            valid = True
            for q in pts:
                if np.linalg.norm(p - q) < min_d:
                    valid = False
                    break
            if valid:
                pts.append(p)
        attempts += 1
    
    while len(pts) < n:
        p = rng.uniform(-radius, radius, 3)
        pts.append(p)
    
    return np.array(pts)


def _perturb_and_repack(base, n, rng, scale=0.2):
    """Perturb a base cluster to n atoms."""
    if len(base) >= n:
        centroid = base.mean(axis=0)
        dists = np.linalg.norm(base - centroid, axis=1)
        idx = np.argsort(dists)[:n]
        pts = base[idx].copy()
    else:
        pts = base.copy()
        centroid = pts.mean(axis=0)
        while len(pts) < n:
            i = rng.integers(len(pts))
            new_p = pts[i] + rng.normal(0, scale, 3)
            pts = np.vstack([pts, new_p])
    
    pts = pts + rng.normal(0, 0.03, pts.shape)
    return pts


def _local_relax(pos, rho, max_iter=1000, tol=1e-12):
    """Gradient-based local relaxation with adaptive step size."""
    pos = pos.copy()
    n = len(pos)
    step = 0.01
    
    for it in range(max_iter):
        e_old = _morse_energy(pos, rho)
        grad = morse_energy_gradient(pos, rho)
        
        norm_grad = np.linalg.norm(grad)
        if norm_grad < tol:
            break
        
        max_step = 0.1
        step_size = min(step, max_step / (norm_grad + 1e-10))
        
        new_pos = pos - step_size * grad
        new_e = _morse_energy(new_pos, rho)
        
        if new_e < e_old:
            pos = new_pos
        else:
            step *= 0.5
            if step < 1e-10:
                break
    
    return pos, _morse_energy(pos, rho)


def _scipy_relax(pos, rho, method='BFGS', maxiter=500):
    """Use scipy.optimize.minimize for relaxation."""
    n = len(pos)
    x0 = pos.flatten()
    
    def objective(x):
        p = x.reshape(n, 3)
        return _morse_energy(p, rho)
    
    def gradient(x):
        p = x.reshape(n, 3)
        return morse_energy_gradient(p, rho).flatten()
    
    try:
        result = minimize(objective, x0, jac=gradient, method=method, 
                          options={'maxiter': maxiter, 'gtol': 1e-10})
        if result.success or result.fun < _morse_energy(pos, rho):
            return result.x.reshape(n, 3), result.fun
    except Exception:
        pass
    
    return pos, _morse_energy(pos, rho)


def construct(instance):
    """Find low-energy Morse cluster configuration."""
    size = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind", 10))
    
    start_time = time.time()
    deadline = start_time + 285.0
    
    rng = np.random.default_rng(size * 13 + int(rho) * 7 + 42)
    
    best_energy = np.inf
    best_pos = None
    
    # Generate structurally appropriate initial configurations
    initial_configs = []
    
    if rho <= 3:
        structures = ['fcc', 'hcp', 'bcc', 'fcc_pert', 'hcp_pert', 'random1', 'random2']
    elif rho <= 6:
        structures = ['fcc', 'hcp', 'ico', 'fcc_pert', 'hcp_pert', 'random1', 'random2', 'ico_pert']
    else:
        structures = ['ico', 'fcc', 'hcp', 'ico_pert', 'fcc_pert', 'random1', 'random2', 'shell']
    
    for struct in structures:
        if time.time() > deadline - 20:
            break
        try:
            if struct == 'fcc':
                pos = _fcc_points(size)
            elif struct == 'hcp':
                pos = _hcp_points(size)
            elif struct == 'bcc':
                pos = _bcc_points(size)
            elif struct == 'ico':
                pos = _icosahedral_cluster(size)
            elif struct == 'shell':
                pos = _spherical_shell_cluster(size)
            elif struct == 'fcc_pert':
                base = _fcc_points(max(size, 15))
                pos = _perturb_and_repack(base, size, rng, 0.3)
            elif struct == 'hcp_pert':
                base = _hcp_points(max(size, 15))
                pos = _perturb_and_repack(base, size, rng, 0.3)
            elif struct == 'ico_pert':
                base = _icosahedral_cluster(max(size, 13))
                pos = _perturb_and_repack(base, size, rng, 0.25)
            elif struct == 'random1':
                pos = _random_dense(size, rng)
            elif struct == 'random2':
                pos = _random_dense(size, rng)
            else:
                continue
            
            pos = _ensure_min_dist(pos)
            
            if _min_dist_check(pos):
                initial_configs.append((struct, pos))
        except Exception:
            continue
    
    # Add extra random restarts
    n_extra_random = min(6, size // 8 + 3)
    for i in range(n_extra_random):
        if time.time() > deadline - 20:
            break
        try:
            pos = _random_dense(size, rng)
            pos = _ensure_min_dist(pos)
            if _min_dist_check(pos):
                initial_configs.append((f'rand_extra_{i}', pos))
        except Exception:
            continue
    
    # Relax each configuration with multiple methods
    for name, pos in initial_configs:
        if time.time() > deadline - 30:
            break
        try:
            # Method 1: Built-in relax
            try:
                energy, relaxed_pos = relax(pos, potential, rho)
                if _min_dist_check(relaxed_pos):
                    if energy < best_energy:
                        best_energy = energy
                        best_pos = relaxed_pos.copy()
            except Exception:
                pass
            
            # Method 2: Local gradient relax
            try:
                relaxed_pos, energy = _local_relax(pos, rho, max_iter=500)
                if _min_dist_check(relaxed_pos):
                    if energy < best_energy:
                        best_energy = energy
                        best_pos = relaxed_pos.copy()
            except Exception:
                pass
            
            # Method 3: Scipy BFGS
            if time.time() < deadline - 35:
                try:
                    relaxed_pos, energy = _scipy_relax(pos, rho, method='BFGS', maxiter=300)
                    if _min_dist_check(relaxed_pos):
                        if energy < best_energy:
                            best_energy = energy
                            best_pos = relaxed_pos.copy()
                except Exception:
                    pass
                    
        except Exception:
            continue
    
    # Additional random restarts with time remaining
    if best_pos is not None and time.time() < deadline - 40:
        n_more = min(10, int((deadline - time.time()) / 6))
        for i in range(n_more):
            if time.time() > deadline - 35:
                break
            try:
                pos = _random_dense(size, rng)
                pos = _ensure_min_dist(pos)
                if _min_dist_check(pos):
                    try:
                        energy, relaxed_pos = relax(pos, potential, rho)
                        if _min_dist_check(relaxed_pos):
                            if energy < best_energy:
                                best_energy = energy
                                best_pos = relaxed_pos.copy()
                    except Exception:
                        pass
                    try:
                        relaxed_pos, energy = _local_relax(pos, rho, max_iter=300)
                        if _min_dist_check(relaxed_pos):
                            if energy < best_energy:
                                best_energy = energy
                                best_pos = relaxed_pos.copy()
                    except Exception:
                        pass
            except Exception:
                continue
    
    # Final refinement: high-precision polish of best_pos
    if best_pos is not None and time.time() < deadline - 15:
        try:
            refined_pos, refined_e = _local_relax(best_pos, rho, max_iter=500, tol=1e-14)
            if _min_dist_check(refined_pos) and refined_e < best_energy:
                best_energy = refined_e
                best_pos = refined_pos.copy()
        except Exception:
            pass
        
        # Try scipy with tight tolerance
        if time.time() < deadline - 10:
            try:
                refined_pos, refined_e = _scipy_relax(best_pos, rho, method='BFGS', maxiter=200)
                if _min_dist_check(refined_pos) and refined_e < best_energy:
                    best_energy = refined_e
                    best_pos = refined_pos.copy()
            except Exception:
                pass
    
    # Final fallback: if no valid result, create a simple valid config
    if best_pos is None:
        best_pos = _spherical_shell_cluster(size)
        best_pos = _ensure_min_dist(best_pos)
        if not _min_dist_check(best_pos):
            radius = 0.8 * size ** (1/3)
            best_pos = rng.uniform(-radius, radius, (size, 3))
            valid = [best_pos[0]]
            for i in range(1, size):
                attempts = 0
                while attempts < 2000:
                    p = best_pos[i]
                    if all(np.linalg.norm(p - v) >= 0.5 for v in valid):
                        valid.append(p)
                        break
                    p = rng.uniform(-radius, radius, 3)
                    best_pos[i] = p
                    attempts += 1
                else:
                    valid.append(best_pos[i])
            best_pos = np.array(valid[:size])
            best_pos = _ensure_min_dist(best_pos)
    
    return best_pos
# EVOLVE-BLOCK-END
