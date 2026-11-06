"""Morse cluster global minima finder - Vectorized gradient + L-BFGS-B + stochastic search.

Strategy:
1. Generate structurally appropriate initial configurations (FCC/HCP for long-range, icosahedral for short-range).
2. Use fully vectorized gradient computation for speed.
3. Relax with L-BFGS-B (robust, handles large systems well).
4. Kick-and-relax stochastic search to escape local minima.
5. Strict time management for 300s total budget across 6 instances.
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


def _morse_gradient(pos, rho):
    """Compute gradient of Morse energy (fully vectorized).
    
    U(r) = exp(rho*(1-r)) * (exp(rho*(1-r)) - 2)
    Let x = rho*(1-r), U = exp(x)*(exp(x)-2)
    dU/dr = -2*rho*exp(x)*(exp(x)-1)
    
    Force on atom i from atom j: F_ij = -dU/dr * (r_i - r_j)/|r_i - r_j|
    grad_i = sum_j dU/dr_ij * (r_i - r_j)/r_ij
           = sum_j [dU/dr_ij / r_ij] * r_i - sum_j [dU/dr_ij / r_ij] * r_j
    """
    n = len(pos)
    grad = np.zeros_like(pos)
    if n < 2:
        return grad
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    
    exp_val = np.exp(rho * (1.0 - dists))
    # dU/dr = -2*rho*exp(x)*(exp(x)-1)
    pair_force = -2.0 * rho * exp_val * (exp_val - 1.0)
    np.fill_diagonal(pair_force, 0.0)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        W = pair_force / dists
    W = np.where(dists > 1e-10, W, 0.0)
    
    # grad_i = sum_j W[i,j] * (pos[i] - pos[j]) = W[i,:].sum() * pos[i] - W[i,:] @ pos
    Ws = W.sum(axis=1)
    grad = Ws[:, np.newaxis] * pos - W @ pos
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


def _ensure_min_dist(pos, min_d=0.51, max_iter=500):
    """Separate atoms that are too close."""
    pos = pos.copy()
    n = len(pos)
    for _ in range(max_iter):
        if not _min_dist_check(pos, min_d):
            diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff ** 2, axis=-1))
            np.fill_diagonal(dists, np.inf)
            idx_i, idx_j = np.unravel_index(np.argmin(dists), dists.shape)
            if idx_i > idx_j:
                idx_i, idx_j = idx_j, idx_i
            rij = pos[idx_i] - pos[idx_j]
            r = np.linalg.norm(rij)
            if r < 1e-10:
                rij = np.array([1.0, 0, 0])
                r = 1.0
            push = (min_d - r + 0.01) / 2.0
            pos[idx_i] += push * rij / r
            pos[idx_j] -= push * rij / r
        else:
            break
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
    """Generate BCC lattice points."""
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


def _fibonacci_sphere(n, radius=1.0):
    """Distribute n points on a sphere using Fibonacci spiral."""
    if n == 1:
        return np.array([[0.0, 0.0, radius]])
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    pts = np.zeros((n, 3))
    for i in range(n):
        z = 1.0 - 2.0 * (i + 0.5) / n
        r = np.sqrt(max(0, 1.0 - z * z))
        theta = golden_angle * i
        pts[i] = np.array([r * np.cos(theta), r * np.sin(theta), z]) * radius
    return pts


def _spherical_shell_cluster(n, spacing=1.0):
    """Distribute n points in concentric shells using Fibonacci sphere."""
    pts = []
    if n <= 1:
        pts.append(np.zeros(3))
        return np.array(pts[:n])
    
    # Center atom
    pts.append(np.zeros(3))
    remaining = n - 1
    
    # Shell 1: icosahedral vertices
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    edge_len = 2.0
    scale1 = spacing / edge_len
    
    s1 = min(12, remaining)
    for v in verts[:s1]:
        pts.append(v * scale1)
    remaining -= s1
    
    # Shell 2: Fibonacci sphere
    if remaining > 0:
        radius2 = spacing * 1.2
        s2 = min(remaining, 20)
        fib_pts = _fibonacci_sphere(s2, radius2)
        for p in fib_pts:
            pts.append(p)
        remaining -= s2
    
    # Shell 3 and beyond: larger Fibonacci spheres
    shell_idx = 3
    while remaining > 0:
        radius = spacing * 0.5 * (shell_idx + 1)
        s = min(remaining, 30)
        fib_pts = _fibonacci_sphere(s, radius)
        for p in fib_pts:
            pts.append(p)
        remaining -= s
        shell_idx += 1
    
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
        while len(pts) < n:
            i = rng.integers(len(pts))
            new_p = pts[i] + rng.normal(0, scale, 3)
            pts = np.vstack([pts, new_p])
    
    pts = pts + rng.normal(0, 0.03, pts.shape)
    return pts


def _lbfgs_relax(pos, rho, maxiter=1000, gtol=1e-10):
    """L-BFGS-B relaxation with analytic gradient."""
    n = len(pos)
    x0 = pos.flatten()
    
    def objective(x):
        p = x.reshape(n, 3)
        return _morse_energy(p, rho)
    
    def gradient(x):
        p = x.reshape(n, 3)
        return _morse_gradient(p, rho).flatten()
    
    try:
        result = minimize(objective, x0, jac=gradient, method='L-BFGS-B',
                          options={'maxiter': maxiter, 'gtol': gtol, 'maxls': 50})
        if result.fun < _morse_energy(pos, rho):
            return result.x.reshape(n, 3), result.fun
    except Exception:
        pass
    
    return pos, _morse_energy(pos, rho)


def _local_relax(pos, rho, max_iter=2000, tol=1e-12):
    """Gradient-based local relaxation with adaptive step size."""
    pos = pos.copy()
    n = len(pos)
    step = 0.01
    
    for it in range(max_iter):
        e_old = _morse_energy(pos, rho)
        grad = _morse_gradient(pos, rho)
        
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


def construct(instance):
    """Find low-energy Morse cluster configuration."""
    size = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind", 10))
    
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin
    
    rng = np.random.default_rng(size * 13 + int(rho) * 7 + 42)
    
    best_energy = np.inf
    best_pos = None
    
    def update_best(pos, energy):
        nonlocal best_energy, best_pos
        if _min_dist_check(pos) and energy < best_energy:
            best_energy = energy
            best_pos = pos.copy()
    
    # Generate structurally appropriate initial configurations
    initial_configs = []
    
    if rho <= 3:
        structures = ['fcc', 'hcp', 'bcc', 'fcc_pert', 'hcp_pert', 'random1', 'random2', 'shell']
    elif rho <= 6:
        structures = ['fcc', 'hcp', 'ico', 'fcc_pert', 'hcp_pert', 'random1', 'random2', 'ico_pert', 'shell']
    else:
        structures = ['ico', 'fcc', 'hcp', 'ico_pert', 'fcc_pert', 'random1', 'random2', 'shell']
    
    for struct in structures:
        if time.time() > deadline - 30:
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
    n_extra_random = min(8, size // 6 + 4)
    for i in range(n_extra_random):
        if time.time() > deadline - 30:
            break
        try:
            pos = _random_dense(size, rng)
            pos = _ensure_min_dist(pos)
            if _min_dist_check(pos):
                initial_configs.append((f'rand_{i}', pos))
        except Exception:
            continue
    
    # Relax each configuration
    for name, pos in initial_configs:
        if time.time() > deadline - 40:
            break
        try:
            # Primary: L-BFGS-B (fast, robust)
            try:
                relaxed_pos, energy = _lbfgs_relax(pos, rho, maxiter=500, gtol=1e-10)
                update_best(relaxed_pos, energy)
            except Exception:
                pass
            
            # Secondary: built-in relax
            if time.time() < deadline - 45:
                try:
                    energy, relaxed_pos = relax(pos, potential, rho)
                    update_best(relaxed_pos, energy)
                except Exception:
                    pass
                    
        except Exception:
            continue
    
    # Kick-and-relax stochastic search
    if best_pos is not None and time.time() < deadline - 50:
        n_kicks = min(20, int((deadline - time.time()) / 5))
        kick_scale = 0.15
        
        for i in range(n_kicks):
            if time.time() > deadline - 45:
                break
            try:
                # Kick the best position
                kicked = best_pos + rng.normal(0, kick_scale, (size, 3))
                kicked = _ensure_min_dist(kicked)
                
                if not _min_dist_check(kicked):
                    continue
                
                # Relax with L-BFGS-B
                try:
                    relaxed_pos, energy = _lbfgs_relax(kicked, rho, maxiter=300, gtol=1e-10)
                    update_best(relaxed_pos, energy)
                except Exception:
                    pass
                
                # Occasionally try built-in relax too
                if i % 3 == 0 and time.time() < deadline - 50:
                    try:
                        energy, relaxed_pos = relax(kicked, potential, rho)
                        update_best(relaxed_pos, energy)
                    except Exception:
                        pass
                        
            except Exception:
                continue
    
    # Additional random restarts with time remaining
    if time.time() < deadline - 50:
        n_more = min(15, int((deadline - time.time()) / 4))
        for i in range(n_more):
            if time.time() > deadline - 45:
                break
            try:
                pos = _random_dense(size, rng)
                pos = _ensure_min_dist(pos)
                if _min_dist_check(pos):
                    try:
                        relaxed_pos, energy = _lbfgs_relax(pos, rho, maxiter=300, gtol=1e-10)
                        update_best(relaxed_pos, energy)
                    except Exception:
                        pass
            except Exception:
                continue
    
    # Final refinement: high-precision polish
    if best_pos is not None and time.time() < deadline - 20:
        try:
            refined_pos, refined_e = _lbfgs_relax(best_pos, rho, maxiter=1000, gtol=1e-12)
            update_best(refined_pos, refined_e)
        except Exception:
            pass
        
        if time.time() < deadline - 15:
            try:
                refined_pos, refined_e = _local_relax(best_pos, rho, max_iter=500, tol=1e-14)
                update_best(refined_pos, refined_e)
            except Exception:
                pass
    
    # Final fallback
    if best_pos is None:
        best_pos = _spherical_shell_cluster(size)
        best_pos = _ensure_min_dist(best_pos)
        if not _min_dist_check(best_pos):
            radius = 0.8 * size ** (1/3)
            best_pos = rng.uniform(-radius, radius, (size, 3))
            best_pos = _ensure_min_dist(best_pos)
    
    return best_pos
# EVOLVE-BLOCK-END
