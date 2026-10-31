"""
Multi-start global optimization for LJ38 with lattice distortion optimization.
Strategy:
1. Generate diverse initial topologies (FCC, BCC, Icosahedral, Random).
2. For structured topologies, optimize a global distortion parameter to minimize energy.
3. Perform local relaxation (L-BFGS) on all candidates.
4. Return the configuration with the lowest energy.
"""

N = 38

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

DEADLINE = 145.0  # Safety margin before 150s kill


def _lj_energy_and_grad(pos):
    """
    Compute LJ energy and gradient for N atoms.
    pos: (N, 3) array
    Returns (energy, grad) where grad is (N, 3).
    """
    n = pos.shape[0]
    diff = pos[np.newaxis, :, :] - pos[:, np.newaxis, :]  # diff[i,j] = pos[j] - pos[i]
    r2 = np.sum(diff * diff, axis=2)  # (N, N)
    np.fill_diagonal(r2, 1.0)
    
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    
    pair_energy = 4.0 * (inv_r12 - inv_r6)
    np.fill_diagonal(pair_energy, 0.0)
    
    energy = 0.5 * np.sum(pair_energy)
    
    r = np.sqrt(r2)
    np.fill_diagonal(r, 1.0)
    coeff = 24.0 * (inv_r6 - 2.0 * inv_r12) / r  # (N, N)
    np.fill_diagonal(coeff, 0.0)
    
    grad = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    
    return energy, grad


def _lj_energy_only(pos):
    """Fast energy-only computation."""
    diff = pos[np.newaxis, :, :] - pos[:, np.newaxis, :]
    r2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(r2, 1.0)
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    pair_energy = 4.0 * (inv_r12 - inv_r6)
    np.fill_diagonal(pair_energy, 0.0)
    return 0.5 * np.sum(pair_energy)


def _check_validity(pos, min_dist=0.5):
    """Check minimum distance constraint."""
    diff = pos[np.newaxis, :, :] - pos[:, np.newaxis, :]
    r2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(r2, np.inf)
    min_r2 = np.min(r2)
    return min_r2 >= min_dist * min_dist


def _make_fcc_cluster(distortion=1.0):
    """
    Construct FCC cluster with optional radial distortion.
    distortion < 1.0 compresses, > 1.0 expands.
    """
    s = 1.0 / math.sqrt(2.0) * distortion
    max_c = 5
    points = []
    for i in range(-max_c, max_c + 1):
        for j in range(-max_c, max_c + 1):
            for k in range(-max_c, max_c + 1):
                if (i + j + k) % 2 == 0:
                    x, y, z = i * s, j * s, k * s
                    r2 = x*x + y*y + z*z
                    points.append((x, y, z, r2))
    
    points.sort(key=lambda p: p[3])
    selected = np.array([(p[0], p[1], p[2]) for p in points[:N]], dtype=float)
    com = np.mean(selected, axis=0)
    selected -= com
    return selected


def _make_bcc_cluster(distortion=1.0):
    """
    Construct BCC cluster.
    """
    s = 1.0 / math.sqrt(3.0) * distortion
    max_c = 5
    points = []
    for i in range(-max_c, max_c + 1):
        for j in range(-max_c, max_c + 1):
            for k in range(-max_c, max_c + 1):
                # BCC: (i,j,k) and (i+0.5, j+0.5, k+0.5)
                # Simpler: just use integer grid and check parity or use two sublattices
                # For simplicity, let's just use a cubic grid and select nearest.
                # Actually, BCC lattice points are (i,j,k) where i,j,k integers, plus (i+0.5, j+0.5, k+0.5).
                # Let's generate both.
                pass
    
    # Simpler BCC generation:
    points = []
    for i in range(-max_c, max_c + 1):
        for j in range(-max_c, max_c + 1):
            for k in range(-max_c, max_c + 1):
                # Corner
                x, y, z = i * s, j * s, k * s
                points.append((x, y, z, x*x + y*y + z*z))
                # Center
                xc, yc, zc = (i + 0.5) * s, (j + 0.5) * s, (k + 0.5) * s
                points.append((xc, yc, zc, xc*xc + yc*yc + zc*zc))
                
    # Remove duplicates if any (unlikely with float, but safe)
    points.sort(key=lambda p: p[3])
    # Take unique closest
    selected = np.array([(p[0], p[1], p[2]) for p in points[:N]], dtype=float)
    com = np.mean(selected, axis=0)
    selected -= com
    return selected


def _make_icosahedral_cluster(distortion=1.0):
    """
    Construct icosahedral cluster.
    Layer 1: 1 atom (center)
    Layer 2: 12 atoms (vertices of icosahedron)
    Layer 3: 20 atoms (centers of faces)
    Layer 4: 5 atoms (partial outer shell)
    Total: 38
    """
    # Golden ratio
    phi = (1 + math.sqrt(5)) / 2
    # Icosahedron vertices (radius 1)
    # (0, ±1, ±phi), (±1, ±phi, 0), (±phi, 0, ±1)
    verts = []
    for x in [0, 1, phi]:
        for y in [0, 1, phi]:
            for z in [0, 1, phi]:
                if x == 0:
                    if y != 0 and z != 0:
                        verts.append((x, y, z))
                        verts.append((x, -y, z))
                        verts.append((x, y, -z))
                        verts.append((x, -y, -z))
                elif y == 0:
                    if x != 0 and z != 0:
                        verts.append((x, y, z))
                        verts.append((x, -y, z))
                        verts.append((x, y, -z))
                        verts.append((x, -y, -z))
                elif z == 0:
                    if x != 0 and y != 0:
                        verts.append((x, y, z))
                        verts.append((x, -y, z))
                        verts.append((x, y, -z))
                        verts.append((x, -y, -z))
    
    # Normalize to unit radius
    verts = np.array(verts)
    norms = np.linalg.norm(verts, axis=1)
    verts /= norms[:, np.newaxis]
    
    # Scale to desired radius (approx 1.0 for NN distance)
    r1 = 1.0 * distortion
    layer1 = np.array([[0.0, 0.0, 0.0]])
    layer2 = verts * r1
    
    # Face centers of icosahedron
    # There are 20 faces.
    # We can approximate by taking midpoints of edges? No, face centers.
    # Let's just generate a dense set of points on the sphere and pick the best 20?
    # Or use known coordinates.
    # For simplicity, let's just use a second shell of 20 atoms at radius r2 ~ 1.7
    # and a third shell of 5 atoms at radius r3 ~ 2.5
    # This is a rough icosahedral structure.
    
    # Better: Use the 12 vertices, then add 20 atoms at the centers of the triangular faces.
    # Finding face centers programmatically is complex.
    # Alternative: Just use random points on sphere for outer shells?
    # Let's stick to a simpler "magic number" structure:
    # 1 + 12 + 20 + 5 = 38
    # We will place 20 atoms at radius 1.7 * distortion and 5 at 2.5 * distortion.
    # To make it deterministic, we'll use a fixed set of directions.
    
    # Directions for 20 atoms (approximate icosahedron face centers)
    # We can use the vertices of a dodecahedron?
    # Let's just use a grid of points on the sphere and select 20 well-distributed ones.
    # Or, simpler: just relax a random icosahedral-like structure.
    
    # For this implementation, let's just generate 37 random points on shells and relax.
    # But we want a deterministic start.
    
    # Let's use the 12 vertices, then 20 points at radius 1.7, then 5 at 2.5.
    # We'll use a simple spiral distribution for the outer shells.
    
    def sphere_points(n, radius):
        # Fibonacci sphere
        points = []
        golden_angle = math.pi * (3 - math.sqrt(5))
        for i in range(n):
            y = 1 - (i / (n - 1)) * 2
            r = math.sqrt(1 - y*y)
            theta = golden_angle * i
            x = math.cos(theta) * r
            z = math.sin(theta) * r
            points.append([x * radius, y * radius, z * radius])
        return np.array(points)
    
    layer3 = sphere_points(20, 1.7 * distortion)
    layer4 = sphere_points(5, 2.5 * distortion)
    
    cluster = np.vstack([layer1, layer2, layer3, layer4])
    com = np.mean(cluster, axis=0)
    cluster -= com
    return cluster


def _make_random_cluster(seed=42):
    """
    Generate a random cluster with minimum distance constraint.
    """
    rng = np.random.default_rng(seed)
    pos = rng.normal(0, 1.5, (N, 3))
    
    # Enforce min distance
    for _ in range(1000):
        diff = pos[np.newaxis, :, :] - pos[:, np.newaxis, :]
        r2 = np.sum(diff * diff, axis=2)
        np.fill_diagonal(r2, 1.0)
        r = np.sqrt(r2)
        if np.min(r) >= 0.5:
            break
        idx = np.unravel_index(np.argmin(r2), r2.shape)
        i, j = idx
        direction = pos[j] - pos[i]
        norm = np.linalg.norm(direction)
        if norm < 1e-10:
            direction = np.array([1.0, 0.0, 0.0])
            norm = 1.0
        direction /= norm
        push = (0.5 - norm) / 2.0 + 0.01
        pos[i] -= direction * push
        pos[j] += direction * push
        
    return pos


def _local_relax(pos):
    """
    Local minimization using L-BFGS with analytical gradient.
    """
    def fun(x):
        p = x.reshape(N, 3)
        e, g = _lj_energy_and_grad(p)
        return e, g.flatten()
    
    try:
        result = minimize(
            fun, pos.flatten(), method='L-BFGS-B',
            jac=True,
            options={'maxiter': 300, 'gtol': 1e-10, 'ftol': 1e-14, 'maxls': 50}
        )
        relaxed = result.x.reshape(N, 3)
        return relaxed
    except Exception:
        return pos


def _optimize_distortion(gen_func, initial_dist=1.0):
    """
    Optimize the distortion parameter for a given generator function.
    """
    def objective(dist):
        try:
            pos = gen_func(dist)
            if not _check_validity(pos):
                return 1e20
            return _lj_energy_only(pos)
        except Exception:
            return 1e20
    
    result = minimize(
        objective, [initial_dist], method='Nelder-Mead',
        options={'maxiter': 50, 'xatol': 1e-4, 'fatol': 1e-8}
    )
    return result.x[0]


def construct():
    """
    Multi-start global optimization for LJ38.
    """
    start_time = time.time()
    
    candidates = []
    
    # 1. FCC with optimized distortion
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 30:
        print("Optimizing FCC distortion...")
        dist_fcc = _optimize_distortion(_make_fcc_cluster, 1.0)
        pos_fcc = _make_fcc_cluster(dist_fcc)
        pos_fcc = _local_relax(pos_fcc)
        if _check_validity(pos_fcc):
            e_fcc = _lj_energy_only(pos_fcc)
            candidates.append((e_fcc, pos_fcc))
            print(f"FCC relaxed energy: {e_fcc}")
    
    # 2. BCC with optimized distortion
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 30:
        print("Optimizing BCC distortion...")
        dist_bcc = _optimize_distortion(_make_bcc_cluster, 1.0)
        pos_bcc = _make_bcc_cluster(dist_bcc)
        pos_bcc = _local_relax(pos_bcc)
        if _check_validity(pos_bcc):
            e_bcc = _lj_energy_only(pos_bcc)
            candidates.append((e_bcc, pos_bcc))
            print(f"BCC relaxed energy: {e_bcc}")
    
    # 3. Icosahedral with optimized distortion
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 30:
        print("Optimizing Icosahedral distortion...")
        dist_ico = _optimize_distortion(_make_icosahedral_cluster, 1.0)
        pos_ico = _make_icosahedral_cluster(dist_ico)
        pos_ico = _local_relax(pos_ico)
        if _check_validity(pos_ico):
            e_ico = _lj_energy_only(pos_ico)
            candidates.append((e_ico, pos_ico))
            print(f"Icosahedral relaxed energy: {e_ico}")
    
    # 4. Random starts
    for seed in [1, 2, 3]:
        elapsed = time.time() - start_time
        if elapsed < DEADLINE - 15:
            print(f"Relaxing random start {seed}...")
            pos_rand = _make_random_cluster(seed)
            pos_rand = _local_relax(pos_rand)
            if _check_validity(pos_rand):
                e_rand = _lj_energy_only(pos_rand)
                candidates.append((e_rand, pos_rand))
                print(f"Random {seed} relaxed energy: {e_rand}")
    
    # Select best candidate
    if not candidates:
        # Fallback: return FCC without distortion optimization
        pos = _make_fcc_cluster(1.0)
        pos = _local_relax(pos)
        if not _check_validity(pos):
            # Last resort: return a simple grid
            pos = np.zeros((N, 3))
            for i in range(N):
                pos[i] = [i % 5, (i // 5) % 5, i // 25] * 1.1
            pos = _local_relax(pos)
        return pos
    
    candidates.sort(key=lambda x: x[0])
    best_energy, best_pos = candidates[0]
    
    # Final check
    if not _check_validity(best_pos):
        # Try to fix
        for _ in range(100):
            diff = best_pos[np.newaxis, :, :] - best_pos[:, np.newaxis, :]
            r2 = np.sum(diff * diff, axis=2)
            np.fill_diagonal(r2, 1.0)
            r = np.sqrt(r2)
            if np.min(r) >= 0.5:
                break
            idx = np.unravel_index(np.argmin(r2), r2.shape)
            i, j = idx
            direction = best_pos[j] - best_pos[i]
            norm = np.linalg.norm(direction)
            if norm < 1e-10:
                direction = np.array([1.0, 0.0, 0.0])
                norm = 1.0
            direction /= norm
            push = (0.5 - norm) / 2.0 + 0.01
            best_pos[i] -= direction * push
            best_pos[j] += direction * push
            
    return best_pos

# EVOLVE-BLOCK-END
