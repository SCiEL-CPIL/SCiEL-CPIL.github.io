"""Dzugutov cluster global minima: divide-and-conquer polytetrahedral assembly."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

# Dzugutov potential parameters
A_P = 5.82
B_P = 1.28
a_P = 1.87
b_P = 1.94
c_P = 1.10
d_P = 0.27
m_P = 16

_START_TIME = time.time()
_DEADLINE = 330.0  # evaluator kills at 330s; keep margin


def _dzugutov_energy(positions):
    """Compute Dzugutov potential energy for a cluster (positions: (n,3) array)."""
    n = len(positions)
    if n < 2:
        return 0.0
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(r, np.inf)
    iu = np.triu_indices(n, k=1)
    r_ij = r[iu]
    # V(r) = A(r^-m - B)*exp(c/(r-a)) for r < a
    #      + B*exp(d/(r-b)) for r < b
    e1 = np.zeros_like(r_ij)
    mask_a = r_ij < a_P
    if np.any(mask_a):
        ra = r_ij[mask_a]
        e1[mask_a] = A_P * (ra ** (-m_P) - B_P) * np.exp(c_P / (ra - a_P))
    e2 = np.zeros_like(r_ij)
    mask_b = r_ij < b_P
    if np.any(mask_b):
        rb = r_ij[mask_b]
        e2[mask_b] = B_P * np.exp(d_P / (rb - b_P))
    return float(np.sum(e1 + e2))


def _icosahedron(center, radius, orientation=None):
    """Generate 12 vertices of a regular icosahedron centered at `center` with circumradius `radius`."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    # Standard icosahedron vertices (circumradius = phi)
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
    ], dtype=float)
    norm = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norm
    verts *= radius
    if orientation is not None:
        verts = verts @ orientation.T
    verts += center
    return verts


def _relax_cluster(positions, max_iter=500, tol=1e-10):
    """Relax a cluster using scipy L-BFGS-B with the Dzugutov gradient."""
    positions = np.array(positions, dtype=float)
    n = len(positions)

    def obj(x):
        pos = x.reshape(n, 3)
        # Enforce minimum distance by penalty
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        d = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(d, np.inf)
        iu = np.triu_indices(n, k=1)
        d_ij = d[iu]
        penalty = 0.0
        if np.any(d_ij < 0.5):
            penalty = 100.0 * np.sum((0.5 - d_ij[d_ij < 0.5]) ** 2)
        return _dzugutov_energy(pos) + penalty

    def grad(x):
        pos = x.reshape(n, 3)
        e, g = dzugutov_energy_gradient(pos)
        return g.flatten()

    try:
        res = minimize(obj, positions.flatten(), jac=grad, method='L-BFGS-B',
                       options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-8})
        return res.x.reshape(n, 3), float(obj(res.x))
    except Exception:
        return positions, _dzugutov_energy(positions)


def _build_icosahedron_cluster(n_atoms, radius=1.0):
    """Build a cluster of n_atoms atoms using icosahedral packing."""
    if n_atoms <= 1:
        return np.array([[0.0, 0.0, 0.0]])
    if n_atoms == 13:
        # 13-atom icosahedron: 1 center + 12 vertices
        center = np.array([[0.0, 0.0, 0.0]])
        verts = _icosahedron(np.zeros(3), radius)
        return np.vstack([center, verts])
    # For larger clusters, use multiple icosahedra or icosahedral shells
    # Start with a 13-atom icosahedron and add atoms in icosahedral positions
    base = _build_icosahedron_cluster(13, radius)
    if n_atoms == 13:
        return base
    # Add atoms in a second icosahedral shell
    remaining = n_atoms - 13
    shell_radius = radius * 1.8
    shell_atoms = _icosahedron(np.zeros(3), shell_radius)
    # Add more shells if needed
    all_atoms = [base]
    count = 13
    shell_idx = 1
    while count < n_atoms:
        r_shell = radius * (1.0 + 0.8 * shell_idx)
        shell = _icosahedron(np.zeros(3), r_shell)
        needed = min(12, n_atoms - count)
        all_atoms.append(shell[:needed])
        count += needed
        shell_idx += 1
    return np.vstack(all_atoms)


def _combine_clusters(clusters, rng):
    """Combine multiple sub-clusters by optimizing their relative positions and orientations."""
    # Place clusters at initial positions
    all_positions = []
    offsets = []
    for i, cl in enumerate(clusters):
        # Initial offset: place clusters in a line with some spread
        offset = np.array([i * 3.0, 0.0, 0.0])
        offsets.append(offset)
        all_positions.append(cl + offset)
    
    combined = np.vstack(all_positions)
    n_total = len(combined)
    n_clusters = len(clusters)
    
    # Optimize: relative positions of cluster centers + orientations
    # Simplification: just optimize all atom positions with a good initial guess
    return combined


def _assemble_polytetrahedral(n_atoms, rng):
    """Assemble a polytetrahedral cluster of n_atoms atoms."""
    # Strategy: build from icosahedral sub-units
    # For 38-60 atoms, use 2-5 icosahedra (13 atoms each)
    
    # Determine number of icosahedra
    n_ico = max(1, n_atoms // 13)
    remainder = n_atoms - n_ico * 13
    
    # Build icosahedra
    clusters = []
    for i in range(n_ico):
        ico = _build_icosahedron_cluster(13, radius=1.0)
        clusters.append(ico)
    
    if remainder > 0:
        # Add remaining atoms as a smaller icosahedron or random
        if remainder <= 13:
            rem_cluster = _build_icosahedron_cluster(remainder, radius=0.9)
            clusters.append(rem_cluster)
    
    # Place icosahedra in a polytetrahedral arrangement
    # Icosahedra pack around a central one
    if len(clusters) == 1:
        return clusters[0]
    
    # Central icosahedron at origin
    center_ico = clusters[0]
    all_positions = [center_ico]
    
    # Surrounding icosahedra at icosahedral positions
    if len(clusters) > 1:
        phi = (1.0 + np.sqrt(5.0)) / 2.0
        # 12 icosahedral directions
        dirs = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
        ], dtype=float)
        dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
        
        # Distance between icosahedron centers
        center_dist = 3.2
        
        for i, cl in enumerate(clusters[1:], start=1):
            # Random orientation for each icosahedron
            angle = rng.uniform(0, 2 * np.pi)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rot = np.array([
                [cos_a, -sin_a, 0],
                [sin_a, cos_a, 0],
                [0, 0, 1]
            ])
            cl_rotated = cl @ rot.T
            offset = dirs[i % 12] * center_dist
            all_positions.append(cl_rotated + offset)
    
    combined = np.vstack(all_positions)
    return combined


def _basin_hopping_step(positions, rng, step_size=0.5):
    """Perform a single basin hopping step: perturb, then relax."""
    n = len(positions)
    # Random perturbation
    noise = rng.normal(0, step_size, (n, 3))
    perturbed = positions + noise
    # Relax
    relaxed, energy = _relax_cluster(perturbed, max_iter=300, tol=1e-9)
    return relaxed, energy


def construct(instance):
    """Construct a low-energy Dzugutov cluster using polytetrahedral assembly + basin hopping."""
    n = instance["n"]
    potential = instance["potential"]
    
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    rng = np.random.default_rng(n + 42)
    
    # Deadline check
    def time_remaining():
        return _DEADLINE - (time.time() - _START_TIME)
    
    # Generate multiple diverse starting structures
    candidates = []
    
    # Strategy 1: Polytetrahedral assembly
    try:
        poly = _assemble_polytetrahedral(n, rng)
        if len(poly) == n:
            relaxed_poly, e_poly = _relax_cluster(poly, max_iter=500, tol=1e-10)
            candidates.append((e_poly, relaxed_poly))
    except Exception:
        pass
    
    # Strategy 2: Icosahedral seed with random perturbations
    for trial in range(5):
        if time_remaining() < 30:
            break
        try:
            # Build an icosahedral-based seed
            seed = _build_icosahedron_cluster(n, radius=1.0)
            if len(seed) != n:
                # Pad or trim
                if len(seed) < n:
                    extra = rng.normal(0, 1, (n - len(seed), 3))
                    seed = np.vstack([seed, extra])
                else:
                    seed = seed[:n]
            
            # Random perturbation
            perturbed = seed + rng.normal(0, 0.3, (n, 3))
            relaxed, e = _relax_cluster(perturbed, max_iter=400, tol=1e-10)
            candidates.append((e, relaxed))
        except Exception:
            continue
    
    # Strategy 3: Random compact starts (fallback)
    for trial in range(3):
        if time_remaining() < 30:
            break
        try:
            radius = 1.1 * n ** (1.0 / 3.0) * 0.75
            points = []
            while len(points) < n:
                p = rng.uniform(-radius, radius, 3)
                if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.8 * 1.1 for q in points):
                    points.append(p)
            relaxed, e = _relax_cluster(np.array(points), max_iter=400, tol=1e-10)
            candidates.append((e, relaxed))
        except Exception:
            continue
    
    if not candidates:
        # Last resort: simple icosahedral seed
        seed = _build_icosahedron_cluster(n, radius=1.0)
        if len(seed) != n:
            if len(seed) < n:
                extra = rng.normal(0, 1, (n - len(seed), 3))
                seed = np.vstack([seed, extra])
            else:
                seed = seed[:n]
        return seed
    
    # Select best candidate
    candidates.sort(key=lambda x: x[0])
    best_energy, best_positions = candidates[0]
    
    # Basin hopping from the best candidate
    if time_remaining() > 60:
        current = best_positions.copy()
        current_e = best_energy
        step_size = 0.5
        max_bh_steps = 20
        for step in range(max_bh_steps):
            if time_remaining() < 30:
                break
            try:
                new_pos, new_e = _basin_hopping_step(current, rng, step_size)
                if new_e < current_e:
                    current = new_pos
                    current_e = new_e
                    # Accept, optionally reduce step size
                    step_size *= 0.95
                else:
                    # Reject, increase step size
                    step_size *= 1.1
                    step_size = min(step_size, 1.5)
            except Exception:
                break
        
        if current_e < best_energy:
            best_positions = current
            best_energy = current_e
    
    return best_positions
# EVOLVE-BLOCK-END
