"""Morse cluster global minima search with symmetry-adapted reduced-coordinate optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient


def _morse_energy(pos, rho):
    """Compute Morse potential energy: sum_{i<j} exp(rho(1-r_ij))(exp(rho(1-r_ij))-2)."""
    n = len(pos)
    if n < 2:
        return 0.0
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, 1.0)
    x = rho * (1.0 - dists)
    ex = np.exp(x)
    energy = np.sum(ex * (ex - 2.0)) / 2.0
    return energy


def _generate_fcc_cluster(n, rng):
    """Generate an FCC cluster of n atoms."""
    basis = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5]
    ])
    side = int(np.ceil((0.25 * n) ** (1.0/3.0))) + 2
    pts = []
    for i in range(-side, side + 1):
        for j in range(-side, side + 1):
            for k in range(-side, side + 1):
                for b in basis:
                    p = np.array([i, j, k]) + b
                    pts.append(p)
    pts = np.array(pts)
    pts -= np.mean(pts, axis=0)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)[:n]
    cluster = pts[idx]
    cluster += rng.normal(0, 0.05, size=cluster.shape)
    return cluster


def _generate_hcp_cluster(n, rng):
    """Generate an HCP cluster of n atoms."""
    a = 1.0
    c = np.sqrt(8.0/3.0) * a
    pts = []
    radius = np.sqrt(n / (np.pi * 0.866))
    num_layers = int(np.ceil(2 * radius / (c/2))) + 2
    
    for layer_idx in range(-num_layers//2, num_layers//2 + 1):
        z = layer_idx * (c / 2.0)
        if layer_idx % 2 == 0:
            offset = np.array([0.0, 0.0])
        else:
            offset = np.array([a/2.0, a*np.sqrt(3)/6.0])
        for i in range(-int(radius), int(radius) + 1):
            for j in range(-int(radius), int(radius) + 1):
                x = a * (i + j/2.0) + offset[0]
                y = a * (np.sqrt(3)/2.0 * j) + offset[1]
                p = np.array([x, y, z])
                if np.linalg.norm(p[:2]) <= radius:
                    pts.append(p)
                    
    if len(pts) < n:
        return _generate_fcc_cluster(n, rng)
        
    pts = np.array(pts)
    pts -= np.mean(pts, axis=0)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)[:n]
    cluster = pts[idx]
    cluster += rng.normal(0, 0.05, size=cluster.shape)
    return cluster


def _generate_icosahedron(n, rng):
    """Generate a Mackay icosahedron cluster."""
    phi = (1 + np.sqrt(5)) / 2.0
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    norm = np.linalg.norm(verts, axis=1)
    verts = verts / norm[:, np.newaxis]
    
    points = [np.zeros(3)]
    points.extend(verts)
    
    if n <= 13:
        pts = np.array(points[:n])
        return pts + rng.normal(0, 0.01, size=pts.shape)
    
    # Generate shell 2 (42 atoms at distance 2)
    # Midpoints of edges scaled to distance 2
    edge_mids = []
    for i in range(12):
        for j in range(i+1, 12):
            mid = (verts[i] + verts[j]) / 2.0
            dist = np.linalg.norm(mid)
            if dist > 1e-6:
                scaled = mid / dist * 2.0
                edge_mids.append(scaled)
    
    # Remove duplicates (midpoints of same edge from different vertex pairs)
    unique_mids = []
    for m in edge_mids:
        is_dup = False
        for um in unique_mids:
            if np.linalg.norm(m - um) < 0.1:
                is_dup = True
                break
        if not is_dup:
            unique_mids.append(m)
    
    points.extend(unique_mids)
    
    if n <= len(points):
        pts = np.array(points[:n])
        return pts + rng.normal(0, 0.01, size=pts.shape)
    
    # Generate shell 3 (92 atoms at distance 3) - approximate with random points on sphere
    remaining = n - len(points)
    for _ in range(remaining):
        theta = rng.uniform(0, 2*np.pi)
        z = rng.uniform(-1, 1)
        r = np.sqrt(1 - z*z)
        p = np.array([r*np.cos(theta), r*np.sin(theta), z]) * 3.0
        points.append(p)
    
    pts = np.array(points[:n])
    return pts + rng.normal(0, 0.01, size=pts.shape)


def _generate_deformed_icosahedron(n, rng, deform_scale=0.3):
    """Generate a deformed icosahedron cluster."""
    base = _generate_icosahedron(n, rng)
    # Apply radial deformation
    norms = np.linalg.norm(base, axis=1, keepdims=True)
    norms[norms < 1e-6] = 1.0
    radial_noise = rng.normal(1.0, deform_scale, size=(n, 1))
    base = base / norms * radial_noise
    return base


def _generate_deformed_fcc(n, rng, deform_scale=0.2):
    """Generate a deformed FCC cluster."""
    base = _generate_fcc_cluster(n, rng)
    norms = np.linalg.norm(base, axis=1, keepdims=True)
    norms[norms < 1e-6] = 1.0
    radial_noise = rng.normal(1.0, deform_scale, size=(n, 1))
    base = base / norms * radial_noise
    return base


def _optimize(pos, rho, method='relax', maxiter=2000):
    """Optimize a configuration using the specified method."""
    try:
        if method == 'relax':
            energy, positions = relax(pos, 'morse', rho)
            return energy, positions
        elif method == 'lfgs':
            def fun_and_grad(x):
                pos_2d = x.reshape(-1, 3)
                e, g = morse_energy_gradient(pos_2d, rho)
                return e, g.reshape(-1)
            result = minimize(fun_and_grad, pos.flatten(), method='L-BFGS-B',
                             jac=True, options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-10})
            positions = result.x.reshape(-1, 3)
            return result.fun, positions
        elif method == 'bfgs':
            def fun_and_grad(x):
                pos_2d = x.reshape(-1, 3)
                e, g = morse_energy_gradient(pos_2d, rho)
                return e, g.reshape(-1)
            result = minimize(fun_and_grad, pos.flatten(), method='BFGS',
                             jac=True, options={'maxiter': maxiter, 'gtol': 1e-10})
            positions = result.x.reshape(-1, 3)
            return result.fun, positions
        else:
            return _morse_energy(pos, rho), pos
    except Exception:
        return _morse_energy(pos, rho), pos


def _fix_min_distance(pos, min_dist=0.5):
    """Fix any atoms that are too close together."""
    n = len(pos)
    for _ in range(10):
        fixed = False
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < min_dist:
                    dir_vec = pos[i] - pos[j]
                    if np.linalg.norm(dir_vec) < 1e-10:
                        dir_vec = np.array([1.0, 0.0, 0.0])
                    dir_vec = dir_vec / np.linalg.norm(dir_vec)
                    shift = (min_dist - d) / 2 * 1.01
                    pos[i] += dir_vec * shift
                    pos[j] -= dir_vec * shift
                    fixed = True
        if not fixed:
            break
    return pos


def construct(instance):
    """Find global minimum of Morse cluster."""
    size = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind", 6))
    
    start_time = time.time()
    deadline = start_time + 290  # Safety margin before 330s kill
    
    best_energy = np.inf
    best_positions = None
    
    rng = np.random.default_rng(size)
    
    # Generate diverse initial configurations with structural priors
    inits = []
    
    # 1. FCC cluster
    try:
        inits.append(_generate_fcc_cluster(size, rng))
    except Exception:
        pass
    
    # 2. HCP cluster
    try:
        inits.append(_generate_hcp_cluster(size, rng))
    except Exception:
        pass
    
    # 3. Icosahedron
    try:
        inits.append(_generate_icosahedron(size, rng))
    except Exception:
        pass
    
    # 4. Deformed icosahedron (multiple scales)
    for scale in [0.1, 0.2, 0.3]:
        try:
            inits.append(_generate_deformed_icosahedron(size, rng, deform_scale=scale))
        except Exception:
            pass
    
    # 5. Deformed FCC (multiple scales)
    for scale in [0.1, 0.2]:
        try:
            inits.append(_generate_deformed_fcc(size, rng, deform_scale=scale))
        except Exception:
            pass
    
    # 6. Random packed inits
    for _ in range(4):
        try:
            radius = 1.0 * size ** (1.0/3.0) * 0.9
            points = []
            min_dist = 0.55
            attempts = 0
            max_attempts = 30000
            while len(points) < size and attempts < max_attempts:
                p = rng.uniform(-radius, radius, 3)
                if np.linalg.norm(p) <= radius:
                    if all(np.linalg.norm(p - q) >= min_dist for q in points):
                        points.append(p)
                attempts += 1
            if len(points) < size:
                min_dist = 0.51
                attempts = 0
                while len(points) < size and attempts < max_attempts:
                    p = rng.uniform(-radius, radius, 3)
                    if np.linalg.norm(p) <= radius:
                        if all(np.linalg.norm(p - q) >= min_dist for q in points):
                            points.append(p)
                    attempts += 1
            while len(points) < size:
                p = rng.uniform(-radius, radius, 3)
                points.append(p)
            inits.append(np.array(points[:size]))
        except Exception:
            pass
    
    # Optimize each initial configuration
    for i, init in enumerate(inits):
        if time.time() > deadline:
            break
        
        # Method 1: built-in relax
        e1, p1 = _optimize(init, rho, method='relax')
        if e1 < best_energy:
            best_energy = e1
            best_positions = p1
        
        # Method 2: L-BFGS-B
        if time.time() > deadline:
            break
        e2, p2 = _optimize(init, rho, method='lfgs', maxiter=1000)
        if e2 < best_energy:
            best_energy = e2
            best_positions = p2
        
        # Method 3: BFGS
        if time.time() > deadline:
            break
        e3, p3 = _optimize(init, rho, method='bfgs', maxiter=1000)
        if e3 < best_energy:
            best_energy = e3
            best_positions = p3
    
    # Final polish: take best and refine with tighter tolerance
    if best_positions is not None and time.time() < deadline - 10:
        e, p = _optimize(best_positions, rho, method='relax')
        if e < best_energy:
            best_energy = e
            best_positions = p
        
        if time.time() < deadline - 5:
            e, p = _optimize(best_positions, rho, method='lfgs', maxiter=3000)
            if e < best_energy:
                best_energy = e
                best_positions = p
        
        if time.time() < deadline - 2:
            e, p = _optimize(best_positions, rho, method='bfgs', maxiter=3000)
            if e < best_energy:
                best_energy = e
                best_positions = p
    
    # Ensure minimum distance constraint
    if best_positions is not None:
        best_positions = _fix_min_distance(best_positions, min_dist=0.5)
        best_energy = _morse_energy(best_positions, rho)
    
    if best_positions is None:
        best_positions = _generate_fcc_cluster(size, rng)
    
    return best_positions
# EVOLVE-BLOCK-END
