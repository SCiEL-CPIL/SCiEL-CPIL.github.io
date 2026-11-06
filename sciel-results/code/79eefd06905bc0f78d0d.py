"""Morse cluster global minima finder with multi-start optimization and global time management."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

# Global time tracking across all six instances
_GLOBAL_START = None
_GLOBAL_DEADLINE = None


def _get_global_deadline():
    """Get or initialize the global deadline."""
    global _GLOBAL_START, _GLOBAL_DEADLINE
    if _GLOBAL_START is None:
        _GLOBAL_START = time.time()
        _GLOBAL_DEADLINE = _GLOBAL_START + 285.0  # 285s safety margin out of 330s
    return _GLOBAL_DEADLINE


def _time_remaining():
    """Get remaining time in seconds."""
    return _get_global_deadline() - time.time()


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
    """Compute Morse energy gradient for a configuration."""
    n = len(pos)
    grad = np.zeros_like(pos)
    if n < 2:
        return grad
    for i in range(n):
        for j in range(i + 1, n):
            rij = pos[i] - pos[j]
            r = np.linalg.norm(rij)
            if r < 1e-12:
                continue
            val = rho * (1.0 - r)
            exp_val = np.exp(val)
            # dE/dr = rho * exp_val * (2*exp_val - 2) * (-1) ... let me derive
            # E = exp(rho(1-r)) * (exp(rho(1-r)) - 2)
            # Let u = rho(1-r), E = exp(u)(exp(u)-2)
            # dE/du = exp(u)(exp(u)-2) + exp(u)*exp(u) = exp(u)(2exp(u)-2)
            # du/dr = -rho
            # dE/dr = -rho * exp(u)(2exp(u)-2)
            # dE/dri = dE/dr * (ri - rj)/r
            dEdr = -rho * exp_val * (2.0 * exp_val - 2.0)
            grad[i] += dEdr * rij / r
            grad[j] -= dEdr * rij / r
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


def _fix_min_dist(pos, min_d=0.5, max_iters=100):
    """Try to fix minimum distance violations by moving atoms apart."""
    n = len(pos)
    if n < 2:
        return pos
    for _ in range(max_iters):
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        min_d_actual = np.min(dists)
        if min_d_actual >= min_d - 1e-9:
            break
        # Find the pair with minimum distance
        i, j = np.unravel_index(np.argmin(dists), dists.shape)
        rij = pos[i] - pos[j]
        r = np.linalg.norm(rij)
        if r < 1e-12:
            pos[i] += np.array([0.01, 0.01, 0.01])
        else:
            direction = rij / r
            needed = (min_d - r) / 2.0 + 0.01
            pos[i] += direction * needed
            pos[j] -= direction * needed
    return pos


def _generate_fcc(n, spacing=1.0):
    """Generate an FCC-like cluster of n atoms."""
    pts = []
    a = spacing
    max_r = int(np.ceil(n ** (1/3))) + 3
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            for k in range(-max_r, max_r + 1):
                for dx, dy, dz in [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]:
                    p = np.array([i + dx, j + dy, k + dz]) * a
                    pts.append(p)
    pts = np.array(pts)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)[:n]
    return pts[idx]


def _generate_bcc(n, spacing=1.0):
    """Generate a BCC-like cluster of n atoms."""
    pts = []
    a = spacing
    max_r = int(np.ceil(n ** (1/3))) + 3
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            for k in range(-max_r, max_r + 1):
                pts.append(np.array([i, j, k]) * a)
                pts.append(np.array([i + 0.5, j + 0.5, k + 0.5]) * a)
    pts = np.array(pts)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)[:n]
    return pts[idx]


def _generate_hcp(n, spacing=1.0):
    """Generate an HCP-like cluster of n atoms."""
    pts = []
    a = spacing
    c = a * np.sqrt(8.0 / 3.0)
    max_r = int(np.ceil(n ** (1/3))) + 3
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            for k in range(-max_r, max_r + 1):
                p1 = np.array([i, j, k])
                p2 = np.array([i + 1/3, j + 2/3, k + 0.5])
                p1[2] *= c / a
                p2[2] *= c / a
                pts.append(p1 * a)
                pts.append(p2 * a)
    pts = np.array(pts)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)[:n]
    return pts[idx]


def _generate_spherical_shell(n, spacing=1.0):
    """Generate atoms on a spherical shell with some interior."""
    rng = np.random.default_rng(n * 13 + 7)
    pts = []
    # Center atom
    if n > 0:
        pts.append(np.zeros(3))
    # First shell: 12 atoms (icosahedron)
    if n > 1:
        phi = (1 + np.sqrt(5)) / 2
        verts = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
        ], dtype=float)
        verts = verts / np.linalg.norm(verts, axis=1, keepdims=True) * spacing
        for v in verts:
            if len(pts) < n:
                pts.append(v.copy())
    # Second shell: face centers
    if n > 13:
        phi = (1 + np.sqrt(5)) / 2
        verts = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
        ], dtype=float)
        verts = verts / np.linalg.norm(verts, axis=1, keepdims=True) * spacing
        # Add points on edges and faces
        for i in range(12):
            for j in range(i + 1, 12):
                if len(pts) >= n:
                    break
                d = np.linalg.norm(verts[i] - verts[j])
                if d < 1.6 * spacing:
                    mid = (verts[i] + verts[j]) / 2.0
                    mid = mid / np.linalg.norm(mid) * spacing * 0.87
                    pts.append(mid)
    # Fill remaining with random points on sphere
    while len(pts) < n:
        theta = rng.uniform(0, 2 * np.pi)
        z = rng.uniform(-1, 1)
        r_xy = np.sqrt(1 - z * z)
        p = np.array([r_xy * np.cos(theta), r_xy * np.sin(theta), z]) * spacing * 1.5
        pts.append(p)
    return np.array(pts[:n])


def _generate_random_compact(n, rng, spacing=1.0):
    """Generate a random compact cluster."""
    radius = spacing * (n / (4/3 * np.pi * 0.6)) ** (1/3) * 1.3
    pts = []
    min_d = 0.55
    attempts = 0
    max_attempts = 80000
    while len(pts) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= min_d for q in pts):
                pts.append(p)
        attempts += 1
    while len(pts) < n:
        p = rng.uniform(-radius, radius, 3)
        pts.append(p)
    return np.array(pts)


def _generate_random_sphere(n, rng, spacing=1.0):
    """Generate random points within a sphere."""
    radius = spacing * n ** (1/3) * 1.2
    pts = rng.uniform(-radius, radius, (n, 3))
    # Keep only points within sphere
    norms = np.linalg.norm(pts, axis=1)
    mask = norms <= radius
    if mask.sum() < n:
        # Resample
        while mask.sum() < n:
            extra = rng.uniform(-radius, radius, (n - mask.sum(), 3))
            pts = np.vstack([pts, extra])
            norms = np.linalg.norm(pts, axis=1)
            mask = norms <= radius
        pts = pts[mask]
    return pts[:n]


def _optimize_relax(pos, potential, rho, max_time=15.0):
    """Optimize using the provided relax function."""
    start = time.time()
    try:
        energy, relaxed = relax(pos, potential, rho)
        if time.time() - start < max_time and _time_remaining() > 5:
            # Try a second round of relaxation
            energy2, relaxed2 = relax(relaxed, potential, rho)
            if energy2 < energy:
                energy, relaxed = energy2, relaxed2
        return energy, relaxed
    except Exception:
        return _morse_energy(pos, rho), pos


def _optimize_lbfgs(pos, rho, max_time=15.0):
    """Optimize using L-BFGS-B from scipy."""
    from scipy.optimize import minimize
    
    def objective(x):
        p = x.reshape(-1, 3)
        return _morse_energy(p, rho)
    
    def gradient(x):
        p = x.reshape(-1, 3)
        return _morse_energy_gradient(p, rho).flatten()
    
    x0 = pos.flatten()
    n = len(pos)
    
    # Set bounds to keep atoms within a reasonable region
    radius = 2.0 * n ** (1/3)
    bounds = [(-radius, radius)] * (3 * n)
    
    try:
        result = minimize(
            objective, x0, jac=gradient,
            method='L-BFGS-B', bounds=bounds,
            options={'maxiter': 500, 'maxfun': 10000, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        if result.success or result.fun < _morse_energy(pos, rho):
            return float(result.fun), result.x.reshape(-1, 3)
        else:
            return _morse_energy(pos, rho), pos
    except Exception:
        return _morse_energy(pos, rho), pos


def _optimize_bfgs(pos, rho, max_time=10.0):
    """Optimize using BFGS from scipy."""
    from scipy.optimize import minimize
    
    def objective(x):
        p = x.reshape(-1, 3)
        return _morse_energy(p, rho)
    
    def gradient(x):
        p = x.reshape(-1, 3)
        return _morse_energy_gradient(p, rho).flatten()
    
    x0 = pos.flatten()
    
    try:
        result = minimize(
            objective, x0, jac=gradient,
            method='BFGS',
            options={'maxiter': 300, 'gtol': 1e-8}
        )
        if result.success or result.fun < _morse_energy(pos, rho):
            return float(result.fun), result.x.reshape(-1, 3)
        else:
            return _morse_energy(pos, rho), pos
    except Exception:
        return _morse_energy(pos, rho), pos


def _polish(pos, rho, potential, max_time=5.0):
    """Final polish: small perturbations and re-relaxation."""
    best_e = _morse_energy(pos, rho)
    best_p = pos.copy()
    rng = np.random.default_rng(12345)
    
    for _ in range(3):
        if _time_remaining() < 5:
            break
        perturbed = pos + rng.normal(0, 0.02, pos.shape)
        try:
            e, relaxed = relax(perturbed, potential, rho)
            if e < best_e and _min_dist_check(relaxed):
                best_e = e
                best_p = relaxed.copy()
        except Exception:
            pass
    
    return best_e, best_p


def construct(instance):
    """Find low-energy Morse cluster configuration."""
    size = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind"))
    
    deadline = _get_global_deadline()
    rng = np.random.default_rng(size * 7 + 42)
    
    best_energy = np.inf
    best_pos = None
    
    # Generate diverse initial configurations
    initial_configs = []
    
    # 1. FCC-based
    try:
        fcc = _generate_fcc(size, spacing=1.0)
        if _min_dist_check(fcc):
            initial_configs.append(('fcc', fcc))
    except Exception:
        pass
    
    # 2. BCC-based
    try:
        bcc = _generate_bcc(size, spacing=1.0)
        if _min_dist_check(bcc):
            initial_configs.append(('bcc', bcc))
    except Exception:
        pass
    
    # 3. HCP-based
    try:
        hcp = _generate_hcp(size, spacing=1.0)
        if _min_dist_check(hcp):
            initial_configs.append(('hcp', hcp))
    except Exception:
        pass
    
    # 4. Spherical shell
    try:
        sph = _generate_spherical_shell(size, spacing=1.0)
        if _min_dist_check(sph):
            initial_configs.append(('sph', sph))
    except Exception:
        pass
    
    # 5. Random compact (multiple variants)
    for i in range(4):
        try:
            rc = _generate_random_compact(size, rng, spacing=1.0)
            if _min_dist_check(rc):
                initial_configs.append((f'rand_compact{i}', rc))
            else:
                fixed = _fix_min_dist(rc)
                if _min_dist_check(fixed):
                    initial_configs.append((f'rand_compact{i}_fix', fixed))
        except Exception:
            pass
    
    # 6. Random sphere
    for i in range(2):
        try:
            rs = _generate_random_sphere(size, rng, spacing=1.0)
            if _min_dist_check(rs):
                initial_configs.append((f'rand_sphere{i}', rs))
            else:
                fixed = _fix_min_dist(rs)
                if _min_dist_check(fixed):
                    initial_configs.append((f'rand_sphere{i}_fix', fixed))
        except Exception:
            pass
    
    # 7. Perturbed FCC
    try:
        fcc_base = _generate_fcc(max(size, 20), spacing=1.0)
        pert = fcc_base[:size].copy() + rng.normal(0, 0.15, (min(size, len(fcc_base)), 3))
        if len(pert) < size:
            extra = rng.uniform(-2, 2, (size - len(pert), 3))
            pert = np.vstack([pert, extra])
        if _min_dist_check(pert):
            initial_configs.append(('fcc_pert', pert))
        else:
            fixed = _fix_min_dist(pert)
            if _min_dist_check(fixed):
                initial_configs.append(('fcc_pert_fix', fixed))
    except Exception:
        pass
    
    # Sort initial configs by initial energy (best first)
    def _config_energy(name_pos):
        try:
            return _morse_energy(name_pos[1], rho)
        except Exception:
            return np.inf
    
    initial_configs.sort(key=_config_energy)
    
    # Relax each initial configuration using multiple methods
    for name, pos in initial_configs:
        if _time_remaining() < 10:
            break
        
        # Method 1: relax function
        try:
            e1, p1 = _optimize_relax(pos, potential, rho, max_time=min(15.0, _time_remaining() * 0.3))
            if _min_dist_check(p1) and e1 < best_energy:
                best_energy = e1
                best_pos = p1.copy()
        except Exception:
            pass
        
        if _time_remaining() < 10:
            break
        
        # Method 2: L-BFGS-B
        try:
            e2, p2 = _optimize_lbfgs(pos, rho, max_time=min(15.0, _time_remaining() * 0.3))
            if _min_dist_check(p2) and e2 < best_energy:
                best_energy = e2
                best_pos = p2.copy()
        except Exception:
            pass
        
        if _time_remaining() < 10:
            break
        
        # Method 3: BFGS (only for smaller clusters due to cost)
        if size <= 50:
            try:
                e3, p3 = _optimize_bfgs(pos, rho, max_time=min(10.0, _time_remaining() * 0.2))
                if _min_dist_check(p3) and e3 < best_energy:
                    best_energy = e3
                    best_pos = p3.copy()
            except Exception:
                pass
    
    # If we have time, do additional random restarts
    if best_pos is not None and _time_remaining() > 20:
        n_extra = min(15, int(_time_remaining() / 8))
        for i in range(n_extra):
            if _time_remaining() < 10:
                break
            try:
                rp = _generate_random_compact(size, rng, spacing=1.0)
                if not _min_dist_check(rp):
                    rp = _fix_min_dist(rp)
                if not _min_dist_check(rp):
                    continue
                
                # Use the best optimization method
                e, relaxed = _optimize_relax(rp, potential, rho, max_time=min(10.0, _time_remaining() * 0.4))
                if _min_dist_check(relaxed) and e < best_energy:
                    best_energy = e
                    best_pos = relaxed.copy()
            except Exception:
                continue
    
    # Final polish
    if best_pos is not None and _time_remaining() > 5:
        try:
            e_polish, p_polish = _polish(best_pos, rho, potential, max_time=min(5.0, _time_remaining() * 0.5))
            if _min_dist_check(p_polish) and e_polish < best_energy:
                best_energy = e_polish
                best_pos = p_polish.copy()
        except Exception:
            pass
    
    # Final fallback: if no valid result, create a simple valid config
    if best_pos is None:
        # Create a linear chain
        spacing = 0.6
        best_pos = np.zeros((size, 3))
        for i in range(size):
            best_pos[i] = [i * spacing, 0, 0]
        if not _min_dist_check(best_pos):
            # Spread them out more
            radius = 0.6 * size ** (1/3) * 1.5
            valid = [best_pos[0]]
            for i in range(1, size):
                attempts = 0
                while attempts < 2000:
                    p = rng.uniform(-radius, radius, 3)
                    if all(np.linalg.norm(p - v) >= 0.5 for v in valid):
                        valid.append(p)
                        break
                    attempts += 1
                else:
                    valid.append(rng.uniform(-radius, radius, 3))
            best_pos = np.array(valid[:size])
    
    return best_pos
# EVOLVE-BLOCK-END
