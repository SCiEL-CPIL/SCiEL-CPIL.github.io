"""
Topological Annealing for LJ38 Global Minimum.
Strategy:
1. Construct explicit topological candidates: FCC Truncated Octahedron, Icosahedral, BCC, and a dense Random set.
2. Apply a multi-stage relaxation protocol:
   a. Coarse relaxation (L-BFGS, loose tol) to settle into a basin.
   b. Fine relaxation (L-BFGS, tight tol) to converge to the basin minimum.
   c. Ultra-fine polish (L-BFGS, extreme tol) to squeeze out numerical residuals.
3. Use a strict time budget to ensure completion within 150s.
4. Return the lowest energy configuration found.
"""

N = 38
DEADLINE = 145.0  # Safety margin before 150s kill

import numpy as np
import time
import math
from scipy.optimize import minimize

def _lj_energy_and_grad(pos):
    """
    Compute LJ energy and gradient for N atoms.
    pos: (N, 3) array
    Returns (energy, grad) where grad is (N, 3).
    """
    n = pos.shape[0]
    # diff[i,j] = pos[j] - pos[i]
    diff = pos[np.newaxis, :, :] - pos[:, np.newaxis, :]
    r2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(r2, 1.0)
    
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    
    pair_energy = 4.0 * (inv_r12 - inv_r6)
    np.fill_diagonal(pair_energy, 0.0)
    
    energy = 0.5 * np.sum(pair_energy)
    
    r = np.sqrt(r2)
    np.fill_diagonal(r, 1.0)
    # Gradient of 4*(r^-12 - r^-6) wrt pos_i
    # dV/dr = 4*(-12*r^-13 + 6*r^-7) = 24*(r^-7 - 2*r^-13)
    # Force on i from j: -dV/dr * (r_vec / r)
    # r_vec = pos_j - pos_i = diff[i,j]
    # coeff = 24 * (inv_r6 - 2*inv_r12) / r
    coeff = 24.0 * (inv_r6 - 2.0 * inv_r12) / r
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

def _make_fcc_truncated_octahedron():
    """
    Construct the FCC truncated octahedron cluster (38 atoms).
    This is the known global minimum structure for LJ38.
    Structure:
    - 1 center
    - 12 vertices of truncated octahedron? No, it's a slice of FCC.
    - FCC lattice constant a such that nearest neighbor distance is ~1.09 (LJ min).
    - Actually, for LJ, optimal distance is 2^(1/6) ~ 1.122.
    - Let's generate FCC points and select the 38 closest to origin.
    """
    # FCC lattice vectors: a/2 * (1,1,0), a/2*(1,0,1), a/2*(0,1,1)
    # To get NN distance ~ 1.122, set a/sqrt(2) = 1.122 => a = 1.122 * sqrt(2) ~ 1.586
    # Let's use a scale factor s. NN distance = s * sqrt(2)/2 * 2? No.
    # FCC points: (i,j,k) * a/2 where i+j+k is even.
    # NN distance is a/sqrt(2).
    # We want a/sqrt(2) ~ 1.122. So a ~ 1.586.
    # Let's just generate a dense FCC grid and scale it.
    
    points = []
    max_c = 5
    # Generate integer coordinates for FCC
    for i in range(-max_c, max_c + 1):
        for j in range(-max_c, max_c + 1):
            for k in range(-max_c, max_c + 1):
                if (i + j + k) % 2 == 0:
                    # Scale by 1/sqrt(2) to make NN distance 1.0 initially, then relaxation will expand to 1.122
                    x, y, z = i / math.sqrt(2), j / math.sqrt(2), k / math.sqrt(2)
                    r2 = x*x + y*y + z*z
                    points.append((x, y, z, r2))
    
    points.sort(key=lambda p: p[3])
    selected = np.array([(p[0], p[1], p[2]) for p in points[:N]], dtype=float)
    
    # Center of mass
    com = np.mean(selected, axis=0)
    selected -= com
    
    return selected

def _make_icosahedral_cluster():
    """
    Construct an icosahedral cluster (1+12+20+5=38).
    This is a strong local minimum competitor.
    """
    phi = (1 + math.sqrt(5)) / 2
    # 12 vertices of icosahedron
    verts = []
    for x in [0, 1, phi]:
        for y in [0, 1, phi]:
            for z in [0, 1, phi]:
                if x == 0:
                    if y != 0 and z != 0:
                        verts.extend([(x, y, z), (x, -y, z), (x, y, -z), (x, -y, -z)])
                elif y == 0:
                    if x != 0 and z != 0:
                        verts.extend([(x, y, z), (x, -y, z), (x, y, -z), (x, -y, -z)])
                elif z == 0:
                    if x != 0 and y != 0:
                        verts.extend([(x, y, z), (x, -y, z), (x, y, -z), (x, -y, -z)])
    
    verts = np.array(verts)
    norms = np.linalg.norm(verts, axis=1)
    verts /= norms[:, np.newaxis]
    
    # Scale to radius ~ 1.0
    r1 = 1.0
    layer1 = np.array([[0.0, 0.0, 0.0]])
    layer2 = verts * r1
    
    # 20 face centers (approximate with Fibonacci sphere for simplicity, or use known coords)
    # For a robust start, let's use a Fibonacci sphere for the outer 25 atoms (20+5)
    # and let relaxation sort it out.
    # Actually, better to use the 12 vertices and then add 26 random points on a sphere of radius 1.8
    # and let relaxation handle it.
    
    def sphere_points(n, radius, seed=0):
        rng = np.random.default_rng(seed)
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
    
    # 20 atoms at radius 1.7, 5 atoms at radius 2.5
    layer3 = sphere_points(20, 1.7, seed=1)
    layer4 = sphere_points(5, 2.5, seed=2)
    
    cluster = np.vstack([layer1, layer2, layer3, layer4])
    com = np.mean(cluster, axis=0)
    cluster -= com
    return cluster

def _make_bcc_cluster():
    """
    Construct BCC cluster.
    """
    s = 1.0 / math.sqrt(3.0)
    max_c = 5
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
                
    points.sort(key=lambda p: p[3])
    selected = np.array([(p[0], p[1], p[2]) for p in points[:N]], dtype=float)
    com = np.mean(selected, axis=0)
    selected -= com
    return selected

def _make_random_cluster(seed=42):
    """
    Generate a random cluster with minimum distance constraint.
    """
    rng = np.random.default_rng(seed)
    pos = rng.normal(0, 1.5, (N, 3))
    
    # Enforce min distance
    for _ in range(2000):
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

def _relax_stage(pos, gtol, ftol, maxiter, maxls):
    """
    Perform one stage of relaxation.
    """
    def fun(x):
        p = x.reshape(N, 3)
        e, g = _lj_energy_and_grad(p)
        return e, g.flatten()
    
    try:
        result = minimize(
            fun, pos.flatten(), method='L-BFGS-B',
            jac=True,
            options={'maxiter': maxiter, 'gtol': gtol, 'ftol': ftol, 'maxls': maxls}
        )
        relaxed = result.x.reshape(N, 3)
        return relaxed
    except Exception:
        return pos

def _topological_anneal(pos):
    """
    Multi-stage relaxation protocol.
    Stage 1: Coarse (gtol=1e-6, ftol=1e-10, maxiter=100)
    Stage 2: Fine (gtol=1e-10, ftol=1e-14, maxiter=200)
    Stage 3: Polish (gtol=1e-13, ftol=1e-15, maxiter=100)
    """
    start_time = time.time()
    
    # Stage 1: Coarse
    if time.time() - start_time < DEADLINE - 20:
        pos = _relax_stage(pos, gtol=1e-6, ftol=1e-10, maxiter=100, maxls=50)
    
    # Stage 2: Fine
    if time.time() - start_time < DEADLINE - 10:
        pos = _relax_stage(pos, gtol=1e-10, ftol=1e-14, maxiter=200, maxls=50)
    
    # Stage 3: Polish
    if time.time() - start_time < DEADLINE - 5:
        pos = _relax_stage(pos, gtol=1e-13, ftol=1e-15, maxiter=100, maxls=20)
        
    return pos

def construct():
    """
    Multi-start global optimization using Topological Annealing.
    """
    start_time = time.time()
    candidates = []
    
    # Define topologies
    topologies = {
        "FCC": _make_fcc_truncated_octahedron,
        "ICO": _make_icosahedral_cluster,
        "BCC": _make_bcc_cluster,
    }
    
    # Add random starts
    for seed in [1, 2, 3, 4, 5]:
        topologies[f"RAND_{seed}"] = lambda s=seed: _make_random_cluster(s)
        
    for name, gen_func in topologies.items():
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 15:
            break
            
        try:
            pos = gen_func()
            if not _check_validity(pos):
                continue
                
            pos_relaxed = _topological_anneal(pos)
            
            if _check_validity(pos_relaxed):
                e = _lj_energy_only(pos_relaxed)
                candidates.append((e, pos_relaxed))
                print(f"{name} relaxed energy: {e}")
        except Exception as ex:
            print(f"Error in {name}: {ex}")
            
    if not candidates:
        # Fallback
        pos = _make_fcc_truncated_octahedron()
        pos = _relax_stage(pos, 1e-10, 1e-14, 200, 50)
        return pos
        
    candidates.sort(key=lambda x: x[0])
    best_energy, best_pos = candidates[0]
    
    # Final validity check and fix if necessary
    if not _check_validity(best_pos):
        # Simple fix
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
