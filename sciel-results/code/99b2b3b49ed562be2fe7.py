# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient


def _morse_energy(positions, rho):
    """Compute exact Morse energy efficiently."""
    n = len(positions)
    if n < 2:
        return 0.0
    iu = np.triu_indices(n, k=1)
    diff = positions[iu[0]] - positions[iu[1]]
    r = np.sqrt(np.sum(diff * diff, axis=1))
    x = rho * (1.0 - r)
    ex = np.exp(x)
    e = ex * (ex - 2.0)
    return float(np.sum(e))


def _morse_gradient(positions, rho):
    """Compute Morse gradient efficiently."""
    n = len(positions)
    if n < 2:
        return np.zeros_like(positions)
    grad = np.zeros_like(positions)
    iu = np.triu_indices(n, k=1)
    diff = positions[iu[0]] - positions[iu[1]]
    r = np.sqrt(np.sum(diff * diff, axis=1))
    x = rho * (1.0 - r)
    ex = np.exp(x)
    # d/dr [exp(rho(1-r))(exp(rho(1-r))-2)] = -rho*exp(rho(1-r))*(exp(rho(1-r))-1)
    de_dr = -rho * ex * (ex - 1.0)
    # grad_i = sum_j de_dr * (r_i - r_j)/r_ij
    for k in range(len(iu[0])):
        i, j = iu[0][k], iu[1][k]
        if r[k] < 1e-10:
            continue
        factor = de_dr[k] / r[k]
        grad[i] += factor * diff[k]
        grad[j] -= factor * diff[k]
    return grad


def _generate_fcc(n, spacing=1.0):
    """Generate an FCC-like cluster of n atoms."""
    points = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            for k in range(-4, 5):
                for dx, dy, dz in [(0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)]:
                    p = np.array([i + dx, j + dy, k + dz]) * spacing
                    points.append(p)
    points = np.array(points)
    center = np.mean(points, axis=0)
    points -= center
    dists = np.linalg.norm(points, axis=1)
    order = np.argsort(dists)
    points = points[order[:n]]
    return points


def _generate_bcc(n, spacing=1.0):
    """Generate a BCC-like cluster of n atoms."""
    points = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            for k in range(-4, 5):
                for dx, dy, dz in [(0,0,0), (0.5,0.5,0.5)]:
                    p = np.array([i + dx, j + dy, k + dz]) * spacing
                    points.append(p)
    points = np.array(points)
    center = np.mean(points, axis=0)
    points -= center
    dists = np.linalg.norm(points, axis=1)
    order = np.argsort(dists)
    points = points[order[:n]]
    return points


def _generate_hcp(n, spacing=1.0):
    """Generate an HCP-like cluster of n atoms."""
    points = []
    a = spacing
    c = spacing * np.sqrt(8.0 / 3.0)
    for layer in range(-5, 6):
        z = layer * c / 2.0
        offset = 0.0 if layer % 2 == 0 else 1.0
        for i in range(-5, 6):
            for j in range(-5, 6):
                x = (i + 0.5 * (j % 2)) * a
                y = j * a * np.sqrt(3.0) / 2.0
                points.append(np.array([x, y, z]))
                if offset == 0:
                    points.append(np.array([x + a * np.sqrt(3.0) / 6.0, 
                                            y + a / 2.0, z + c / 6.0]))
    points = np.array(points)
    center = np.mean(points, axis=0)
    points -= center
    dists = np.linalg.norm(points, axis=1)
    order = np.argsort(dists)
    points = points[order[:n]]
    return points


def _generate_shell(n, spacing=1.0):
    """Generate a spherical shell-like cluster using Fibonacci sphere."""
    if n <= 1:
        return np.zeros((n, 3))
    points = [np.zeros(3)]
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    radius = spacing * (n ** (1/3.0) * 0.6)
    for i in range(1, n):
        z = 1.0 - 2.0 * i / (n - 1)
        rad = np.sqrt(max(0, 1.0 - z * z))
        theta = golden_angle * i
        x = np.cos(theta) * rad
        y = np.sin(theta) * rad
        points.append(np.array([x, y, z]) * radius)
    return np.array(points)


def _generate_icosahedral(n, spacing=1.0):
    """Generate icosahedral-like cluster (good for small clusters)."""
    phi = (1 + np.sqrt(5)) / 2
    # Icosahedron vertices
    verts = []
    for sx in [-1, 1]:
        for sy in [-1, 1]:
            for sz in [-1, 1]:
                verts.append(np.array([sx, sy*phi, sz*phi]))
                verts.append(np.array([sx*phi, sz, sy]))
                verts.append(np.array([sz*phi, sx, sy]))
    verts = np.array(verts)
    verts /= np.linalg.norm(verts[0])
    verts *= spacing * 0.8
    
    # Add inner shells
    all_points = [np.zeros(3)]
    # First shell (12 vertices of icosahedron)
    for v in verts:
        all_points.append(v)
    
    # Second shell (30 edges + 20 face centers)
    # Edge midpoints
    icos_edges = [
        (0,1),(0,2),(0,3),(0,4),
        (1,2),(1,5),(1,6),(1,7),
        (2,3),(2,5),(2,8),(2,9),
        (3,4),(3,7),(3,8),(3,10),
        (4,5),(4,6),(4,9),(4,10),
        (5,6),(5,8),(5,11),(5,12),
        (6,7),(6,10),(6,11),(6,13),
        (7,8),(7,10),(7,12),(7,13),
        (8,9),(8,11),(8,13),(8,14),
        (9,10),(9,12),(9,14),(9,15),
        (11,12),(11,13),(11,14),
        (12,13),(12,15),
        (13,14),(13,15),
        (14,15)
    ]
    # Just add more points radially
    current_count = len(all_points)
    for i in range(n - current_count):
        # Add points on a sphere with radius increasing
        angle1 = np.arccos(1 - 2*(i+1)/max(1, n-current_count))
        angle2 = 2 * np.pi * i / 7
        r = spacing * (1.5 + 0.3 * (i % 3))
        p = r * np.array([np.sin(angle1)*np.cos(angle2), 
                         np.sin(angle1)*np.sin(angle2), 
                         np.cos(angle1)])
        all_points.append(p)
    
    return np.array(all_points[:n])


def _relax_l_bfgs(positions, rho, max_iter=500, tol=1e-10):
    """Relax using L-BFGS-B with tight tolerance."""
    def fun(x):
        pos = x.reshape(-1, 3)
        return _morse_energy(pos, rho)
    
    def jac(x):
        pos = x.reshape(-1, 3)
        return _morse_gradient(pos, rho).flatten()
    
    x0 = positions.flatten()
    result = minimize(fun, x0, jac=jac, method='L-BFGS-B', 
                      options={'maxiter': max_iter, 'gtol': tol, 'maxls': 50,
                               'ftol': 1e-12})
    final_pos = result.x.reshape(-1, 3)
    final_energy = _morse_energy(final_pos, rho)
    return final_energy, final_pos


def _polish(positions, rho, max_iter=2000, tol=1e-12):
    """High-precision polish pass."""
    def fun(x):
        pos = x.reshape(-1, 3)
        return _morse_energy(pos, rho)
    
    def jac(x):
        pos = x.reshape(-1, 3)
        return _morse_gradient(pos, rho).flatten()
    
    x0 = positions.flatten()
    result = minimize(fun, x0, jac=jac, method='L-BFGS-B', 
                      options={'maxiter': max_iter, 'gtol': tol, 'maxls': 100,
                               'ftol': 1e-14})
    final_pos = result.x.reshape(-1, 3)
    final_energy = _morse_energy(final_pos, rho)
    return final_energy, final_pos


def _random_packed(n, seed, min_dist=0.55):
    """Generate random packed configuration."""
    rng = np.random.default_rng(seed)
    radius = 1.0 * n ** (1.0 / 3.0) * 0.8
    positions = []
    attempts = 0
    while len(positions) < n and attempts < 10000:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            valid = True
            for q in positions:
                if np.linalg.norm(p - q) < min_dist:
                    valid = False
                    break
            if valid:
                positions.append(p)
        attempts += 1
    if len(positions) < n:
        # Fill remaining with jittered positions
        while len(positions) < n:
            p = rng.uniform(-radius, radius, 3)
            positions.append(p)
    return np.array(positions)


def construct(instance):
    """Find low-energy Morse cluster configurations."""
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before 330s kill
    
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind", 3))
    
    # Generate diverse starting configurations
    generators = [
        (_generate_fcc, "fcc"),
        (_generate_bcc, "bcc"),
        (_generate_hcp, "hcp"),
        (_generate_shell, "shell"),
        (_generate_icosahedral, "ico"),
    ]
    
    best_energy = np.inf
    best_positions = None
    tried_configs = []
    
    # Try each generator with relaxation
    for gen_func, name in generators:
        if time.time() > deadline - 10:
            break
        
        try:
            positions = gen_func(n, spacing=1.0)
            energy, relaxed = _relax_l_bfgs(positions, rho, max_iter=400, tol=1e-10)
            
            if energy < best_energy:
                best_energy = energy
                best_positions = relaxed
            tried_configs.append((energy, relaxed.copy(), name))
        except Exception:
            continue
    
    # Random restarts
    n_random = max(3, min(8, int((deadline - time.time()) / 25)))
    for i in range(n_random):
        if time.time() > deadline - 15:
            break
        try:
            positions = _random_packed(n, seed=42 + i, min_dist=0.55)
            energy, relaxed = _relax_l_bfgs(positions, rho, max_iter=300, tol=1e-10)
            
            if energy < best_energy:
                best_energy = energy
                best_positions = relaxed
            tried_configs.append((energy, relaxed.copy(), f"rand{i}"))
        except Exception:
            continue
    
    # Perturbations of best solution
    if best_positions is not None and time.time() < deadline - 20:
        rng = np.random.default_rng(n + 100)
        n_perturb = max(2, min(6, int((deadline - time.time()) / 20)))
        for i in range(n_perturb):
            if time.time() > deadline - 10:
                break
            try:
                # Vary noise scale
                scale = 0.2 + 0.15 * i
                noise = rng.normal(0, scale, size=best_positions.shape)
                perturbed = best_positions + noise
                energy, relaxed = _relax_l_bfgs(perturbed, rho, max_iter=250, tol=1e-10)
                
                if energy < best_energy:
                    best_energy = energy
                    best_positions = relaxed
            except Exception:
                continue
    
    # Final high-precision polish on the best solution
    if best_positions is not None and time.time() < deadline - 5:
        try:
            energy, polished = _polish(best_positions, rho, max_iter=3000, tol=1e-13)
            if energy < best_energy:
                best_energy = energy
                best_positions = polished
        except Exception:
            pass
    
    # Ensure minimum distance constraint is satisfied (0.5 apart)
    if best_positions is not None:
        # Check and fix any close pairs
        for _ in range(10):
            min_dist_found = np.inf
            for i in range(len(best_positions)):
                for j in range(i+1, len(best_positions)):
                    d = np.linalg.norm(best_positions[i] - best_positions[j])
                    if d < min_dist_found:
                        min_dist_found = d
            if min_dist_found >= 0.5:
                break
            # Nudge atoms apart
            for i in range(len(best_positions)):
                for j in range(i+1, len(best_positions)):
                    d = np.linalg.norm(best_positions[i] - best_positions[j])
                    if d < 0.5 and d > 1e-10:
                        direction = (best_positions[i] - best_positions[j]) / d
                        push = (0.5 - d + 0.01) / 2
                        best_positions[i] += direction * push
                        best_positions[j] -= direction * push
            # Re-relax after fixing distances
            try:
                energy, relaxed = _relax_l_bfgs(best_positions, rho, max_iter=100, tol=1e-10)
                if energy < best_energy + 0.1:  # Allow small energy increase for validity
                    best_positions = relaxed
                    best_energy = energy
            except Exception:
                pass
    
    return best_positions
# EVOLVE-BLOCK-END
