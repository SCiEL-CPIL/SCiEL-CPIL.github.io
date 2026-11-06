"""Morse cluster global minima finder using multi-start local optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient


def _morse_energy_and_grad(pos, rho):
    """Compute Morse energy and gradient for all pairs."""
    n = len(pos)
    if n < 2:
        return 0.0, np.zeros_like(pos)
    
    # Pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(r, 1.0)  # avoid division by zero on diagonal
    
    # Morse potential: V(r) = exp(rho*(1-r)) * (exp(rho*(1-r)) - 2)
    x = rho * (1.0 - r)
    ex = np.exp(x)
    v = ex * (ex - 2.0)
    
    # Gradient: dV/dr * dr/dpos
    # dV/dr = -rho * ex * (2*ex - 2) = -2*rho*ex*(ex-1)
    dvdr = -2.0 * rho * ex * (ex - 1.0)
    
    # Only compute for i < j to avoid double counting
    iu = np.triu_indices(n, k=1)
    dvdr_iu = dvdr[iu]
    r_iu = r[iu]
    diff_iu = diff[iu]
    
    # Gradient contribution: dvdr * (pos_i - pos_j) / r
    grad = np.zeros_like(pos)
    for idx in range(len(iu[0])):
        i, j = iu[0][idx], iu[1][idx]
        factor = dvdr_iu[idx] / r_iu[idx]
        grad[i] += factor * diff_iu[idx]
        grad[j] -= factor * diff_iu[idx]
    
    energy = np.sum(v[iu])
    return energy, grad


def _generate_icosahedral_shell(n, radius_scale=1.0):
    """Generate atoms on icosahedral shells."""
    # Icosahedron vertices (12)
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    
    # Octahedron vertices (6) for inner shell
    octa = np.array([
        [1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1]
    ], dtype=float)
    
    points = []
    
    # Start with center if n >= 1
    if n >= 1:
        points.append([0, 0, 0])
        n -= 1
    
    # Add octahedron shell (6 atoms) at radius ~0.7
    if n >= 6:
        for v in octa:
            points.append(v * 0.7 * radius_scale)
        n -= 6
    
    # Add icosahedron shell (12 atoms) at radius ~1.2
    if n >= 12:
        for v in ico:
            points.append(v * 1.2 * radius_scale)
        n -= 12
    
    # Add second icosahedron shell (12 atoms) at radius ~1.8
    if n >= 12:
        for v in ico:
            points.append(v * 1.8 * radius_scale)
        n -= 12
    
    # Add third shell (20 atoms - dodecahedron-like) at radius ~2.4
    if n >= 20:
        # Use icosahedron rotated
        angle = np.pi / 5
        R = np.array([
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)]
        ])
        for v in ico:
            points.append((R @ v) * 2.4 * radius_scale)
        n -= 12
    
    # Fill remaining with random points on sphere
    rng = np.random.default_rng(42)
    while n > 0:
        p = rng.normal(size=3)
        p = p / np.linalg.norm(p) * (2.4 + 0.5 * rng.random()) * radius_scale
        points.append(p)
        n -= 1
    
    return np.array(points[:int(np.sum(np.array([len(p) for p in [points]])))])[:len(points)]


def _generate_compact_cluster(n, rng, spacing=1.0):
    """Generate a compact random cluster with minimum separation."""
    points = []
    radius = spacing * n**(1.0/3.0) * 0.8
    attempts = 0
    while len(points) < n and attempts < n * 100:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.6 * spacing for q in points):
                points.append(p)
        attempts += 1
    
    # If we couldn't place all atoms, add them randomly
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    
    return np.array(points)


def _generate_fcc_cluster(n, spacing=1.0):
    """Generate an FCC-like cluster."""
    points = []
    # FCC lattice vectors
    a = spacing
    # Generate points in a cubic box and select nearest to center
    max_coord = int(np.ceil(n**(1.0/3.0))) + 2
    coords = []
    for i in range(-max_coord, max_coord+1):
        for j in range(-max_coord, max_coord+1):
            for k in range(-max_coord, max_coord+1):
                # FCC: add face-centered positions
                base = np.array([i, j, k]) * a
                face_centers = [
                    base + np.array([0.5, 0.5, 0]) * a,
                    base + np.array([0.5, 0, 0.5]) * a,
                    base + np.array([0, 0.5, 0.5]) * a,
                ]
                coords.append(base)
                coords.extend(face_centers)
    
    coords = np.array(coords)
    # Sort by distance from origin
    dists = np.linalg.norm(coords, axis=1)
    idx = np.argsort(dists)
    points = coords[idx[:n]]
    
    return points


def _generate_hcp_cluster(n, spacing=1.0):
    """Generate an HCP-like cluster."""
    points = []
    a = spacing
    c = a * np.sqrt(8.0/3.0)
    max_coord = int(np.ceil(n**(1.0/3.0))) + 2
    
    coords = []
    for layer in range(-max_coord, max_coord+1):
        z = layer * c
        for i in range(-max_coord, max_coord+1):
            for j in range(-max_coord, max_coord+1):
                if layer % 2 == 0:
                    x = i * a
                    y = j * a * np.sqrt(3.0)/2.0
                else:
                    x = (i + 0.5) * a
                    y = j * a * np.sqrt(3.0)/2.0
                coords.append([x, y, z])
    
    coords = np.array(coords)
    dists = np.linalg.norm(coords, axis=1)
    idx = np.argsort(dists)
    points = coords[idx[:n]]
    
    return points


def _local_minimize(pos, rho, max_iter=5000):
    """Local minimization using L-BFGS-B."""
    def objective_and_grad(x):
        p = x.reshape(-1, 3)
        e, g = _morse_energy_and_grad(p, rho)
        return e, g.flatten()
    
    result = minimize(
        objective_and_grad,
        pos.flatten(),
        jac=True,
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10}
    )
    return result.fun, result.x.reshape(-1, 3)


def _relax_with_fallback(pos, potential, parameter, max_iter=3000):
    """Try local relaxation, fall back to scipy if it fails."""
    try:
        energy, positions = relax(pos, potential, parameter)
        return energy, positions
    except Exception:
        # Fallback to our own minimizer
        e, p = _local_minimize(pos, parameter, max_iter=max_iter)
        return e, p


def construct(instance):
    """Find global minimum for Morse cluster using multi-start local optimization."""
    start_time = time.time()
    deadline = 280.0  # Leave 50s safety margin for 330s kill
    
    size = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind", 6))
    
    rng = np.random.default_rng(size)
    
    best_energy = np.inf
    best_positions = None
    
    # Generate diverse initial configurations
    initial_configs = []
    
    # 1. FCC-like
    try:
        fcc = _generate_fcc_cluster(size)
        if len(fcc) == size:
            initial_configs.append(fcc)
    except:
        pass
    
    # 2. HCP-like
    try:
        hcp = _generate_hcp_cluster(size)
        if len(hcp) == size:
            initial_configs.append(hcp)
    except:
        pass
    
    # 3. Icosahedral shells
    try:
        ico = _generate_icosahedral_shell(size)
        if len(ico) == size:
            initial_configs.append(ico)
    except:
        pass
    
    # 4-8. Random compact clusters with different seeds
    for seed in [size, size+1, size+2, size+3, size+4]:
        rng_local = np.random.default_rng(seed)
        compact = _generate_compact_cluster(size, rng_local)
        initial_configs.append(compact)
    
    # 9. Spherical shell
    theta = np.linspace(0, 2*np.pi, size, endpoint=False)
    phi_s = np.arccos(1 - 2*np.arange(size)/size)
    r_shell = size**(1.0/3.0) * 0.9
    shell = np.column_stack([
        r_shell * np.sin(phi_s) * np.cos(theta),
        r_shell * np.sin(phi_s) * np.sin(theta),
        r_shell * np.cos(phi_s)
    ])
    initial_configs.append(shell)
    
    # 10. Two-layer structure
    if size >= 10:
        n1 = size // 2
        n2 = size - n1
        layer1 = _generate_compact_cluster(n1, np.random.default_rng(size+10))
        layer2 = _generate_compact_cluster(n2, np.random.default_rng(size+11))
        layer2[:, 2] += 1.5  # offset in z
        two_layer = np.vstack([layer1, layer2])
        initial_configs.append(two_layer)
    
    # Try each configuration
    for i, config in enumerate(initial_configs):
        if time.time() - start_time > deadline:
            break
        
        if len(config) != size:
            continue
        
        # Ensure minimum separation
        min_sep = 0.5
        for _ in range(10):
            dists = np.sqrt(np.sum((config[:, np.newaxis, :] - config[np.newaxis, :, :])**2, axis=-1))
            np.fill_diagonal(dists, np.inf)
            if np.min(dists) >= min_sep:
                break
            # Nudge atoms that are too close
            close_pairs = np.argwhere(dists < min_sep)
            for idx in close_pairs[:5]:
                i_idx, j_idx = idx[0], idx[1]
                if i_idx > j_idx:
                    i_idx, j_idx = j_idx, i_idx
                diff = config[i_idx] - config[j_idx]
                norm = np.linalg.norm(diff)
                if norm < 1e-10:
                    diff = rng.normal(size=3)
                    norm = np.linalg.norm(diff)
                direction = diff / norm
                push = (min_sep - norm + 0.1) / 2.0
                config[i_idx] += push * direction
                config[j_idx] -= push * direction
        
        # Local minimization
        try:
            energy, positions = _relax_with_fallback(config.copy(), potential, rho, max_iter=2000)
        except Exception:
            continue
        
        if energy < best_energy:
            best_energy = energy
            best_positions = positions.copy()
    
    # If we have time, do additional random restarts
    if best_positions is not None and time.time() - start_time < deadline - 10:
        for restart in range(20):
            if time.time() - start_time > deadline:
                break
            
            rng_local = np.random.default_rng(size + 100 + restart)
            config = _generate_compact_cluster(size, rng_local)
            
            # Add some noise to best solution for basin hopping-like exploration
            if restart % 2 == 1 and best_positions is not None:
                config = best_positions + rng_local.normal(scale=0.3, size=(size, 3))
            
            try:
                energy, positions = _relax_with_fallback(config.copy(), potential, rho, max_iter=1500)
            except Exception:
                continue
            
            if energy < best_energy:
                best_energy = energy
                best_positions = positions.copy()
    
    # Final fallback: if we still don't have a valid solution
    if best_positions is None:
        best_positions = _generate_compact_cluster(size, np.random.default_rng(0))
        try:
            _, best_positions = relax(best_positions, potential, rho)
        except:
            pass
    
    return best_positions
# EVOLVE-BLOCK-END
