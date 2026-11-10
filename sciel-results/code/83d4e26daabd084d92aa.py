"""Dzugutov cluster global minima: hierarchical assembly + basin hopping."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient
import time

# Dzugutov potential parameters
A_POT = 5.82
B_POT = 1.28
a_POT = 1.87
b_POT = 1.94
c_POT = 1.10
d_POT = 0.27
m_POT = 16


def _dzugutov_energy(positions):
    """Compute Dzugutov potential energy."""
    n = len(positions)
    if n < 2:
        return 0.0
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(r, np.inf)
    iu = np.triu_indices(n, k=1)
    r_ij = r[iu]
    
    # V(r) = A(r^-m - B)exp(c/(r-a)) for r < a
    #     + B exp(d/(r-b)) for r < b
    # Note: the second term applies for r < b, which includes r < a.
    # So for r < a: V = A(r^-m - B)exp(c/(r-a)) + B exp(d/(r-b))
    # For a <= r < b: V = B exp(d/(r-b))
    
    energy = 0.0
    mask_a = r_ij < a_POT
    mask_b = (r_ij >= a_POT) & (r_ij < b_POT)
    
    if np.any(mask_a):
        ra = r_ij[mask_a]
        energy += np.sum(A_POT * (ra ** (-m_POT) - B_POT) * np.exp(c_POT / (ra - a_POT)) + B_POT * np.exp(d_POT / (ra - b_POT)))
    if np.any(mask_b):
        rb = r_ij[mask_b]
        energy += np.sum(B_POT * np.exp(d_POT / (rb - b_POT)))
    
    return float(energy)


def _dzugutov_gradient(positions):
    """Compute gradient of Dzugutov energy."""
    n = len(positions)
    if n < 2:
        return np.zeros_like(positions)
    
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(r, np.inf)
    
    grad = np.zeros_like(positions)
    iu = np.triu_indices(n, k=1)
    r_ij = r[iu]
    i_idx, j_idx = iu
    
    # For pairs with r < a:
    # V = A(r^-m - B)exp(c/(r-a)) + B exp(d/(r-b))
    # dV/dr = A(-m*r^(-m-1) - c*(r^-m - B)/(r-a)^2)exp(c/(r-a)) + B*(-d)/(r-b)^2 * exp(d/(r-b))
    #        = A*(-m*r^(-m-1) - c*(r^-m - B)/(r-a)^2)*exp(c/(r-a)) - B*d/(r-b)^2 * exp(d/(r-b))
    
    mask_a = r_ij < a_POT
    mask_b = (r_ij >= a_POT) & (r_ij < b_POT)
    
    if np.any(mask_a):
        ra = r_ij[mask_a]
        ia = i_idx[mask_a]
        ja = j_idx[mask_a]
        # dV/dr
        term1 = A_POT * (-m_POT * ra ** (-m_POT - 1) - c_POT * (ra ** (-m_POT) - B_POT) / (ra - a_POT) ** 2) * np.exp(c_POT / (ra - a_POT))
        term2 = -B_POT * d_POT / (ra - b_POT) ** 2 * np.exp(d_POT / (ra - b_POT))
        dVdr = term1 + term2
        # grad_i += dVdr * (pos_i - pos_j) / r
        # grad_j -= dVdr * (pos_i - pos_j) / r
        unit_vec = diff[ia, ja] / ra[:, np.newaxis]
        grad[ia] += dVdr[:, np.newaxis] * unit_vec
        grad[ja] -= dVdr[:, np.newaxis] * unit_vec
    
    if np.any(mask_b):
        rb = r_ij[mask_b]
        ib = i_idx[mask_b]
        jb = j_idx[mask_b]
        dVdr = -B_POT * d_POT / (rb - b_POT) ** 2 * np.exp(d_POT / (rb - b_POT))
        unit_vec = diff[ib, jb] / rb[:, np.newaxis]
        grad[ib] += dVdr[:, np.newaxis] * unit_vec
        grad[jb] -= dVdr[:, np.newaxis] * unit_vec
    
    return grad


def _energy_and_grad(positions):
    """Combined energy and gradient for scipy."""
    e = _dzugutov_energy(positions)
    g = _dzugutov_gradient(positions)
    return e, g


def _local_minimize(positions, maxiter=200):
    """Local minimization using L-BFGS-B."""
    def fun(x):
        pos = x.reshape(-1, 3)
        e, g = _energy_and_grad(pos)
        return e, g.flatten()
    
    x0 = positions.flatten()
    res = minimize(fun, x0, jac=True, method='L-BFGS-B', options={'maxiter': maxiter, 'gtol': 1e-8})
    pos = res.x.reshape(-1, 3)
    e = _dzugutov_energy(pos)
    return e, pos


def _make_icosahedron(center, radius=1.0, rng=None):
    """Generate 12 vertices of a regular icosahedron plus center (13 atoms)."""
    if rng is None:
        rng = np.random.default_rng()
    
    # Regular icosahedron vertices (normalized to unit circumradius)
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    # Normalize to unit circumradius
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    verts = verts * radius
    
    # Add center
    center = np.array(center, dtype=float)
    ico = np.vstack([center, center + verts])
    return ico


def _make_decahedron(center, radius=1.0, rng=None):
    """Generate a Mackay decahedron (55 atoms: 1+12+42)."""
    if rng is None:
        rng = np.random.default_rng()
    
    # Layer 0: center
    # Layer 1: 12 icosahedron vertices at distance r1
    # Layer 2: 42 vertices (20 face centers + 22 edge midpoints... actually complex)
    # Simplified: use icosahedral shell + additional atoms
    
    phi = (1 + np.sqrt(5)) / 2
    verts_ico = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts_ico, axis=1, keepdims=True)
    verts_ico = verts_ico / norms
    
    center = np.array(center, dtype=float)
    atoms = [center]
    
    # Shell 1: 12 atoms
    r1 = radius * 1.0
    atoms.extend(center + verts_ico * r1)
    
    # Shell 2: 42 atoms - approximate with points on icosahedral surface
    # For simplicity, place 42 atoms in a slightly larger shell with icosahedral symmetry
    # This is a rough approximation
    r2 = radius * 1.8
    # Generate 42 points with icosahedral symmetry
    # 20 face centers
    from scipy.spatial import ConvexHull
    hull = ConvexHull(verts_ico)
    faces = hull.simplices
    face_centers = verts_ico[faces].mean(axis=1)
    fc_norms = np.linalg.norm(face_centers, axis=1, keepdims=True)
    face_centers = face_centers / fc_norms * r2
    atoms.extend(center + face_centers)
    
    # 22 more atoms - place along edges
    edges = set()
    for f in faces:
        for i in range(3):
            e = tuple(sorted([f[i], f[(i+1)%3]]))
            edges.add(e)
    edge_mids = []
    for e in edges:
        mid = (verts_ico[e[0]] + verts_ico[e[1]]) / 2
        mid_norm = np.linalg.norm(mid)
        if mid_norm > 1e-10:
            mid = mid / mid_norm * r2
            edge_mids.append(mid)
    # We need 22 more, but we have 30 edges. Take first 22
    atoms.extend(center + np.array(edge_mids[:22]))
    
    return np.array(atoms)


def _assemble_clusters(small_clusters, rng):
    """Combine small clusters into a larger one by placing them near each other."""
    all_atoms = []
    for cluster in small_clusters:
        all_atoms.extend(cluster)
    return np.array(all_atoms)


def _basin_hopping(positions, n_iter=50, step_size=0.5, rng=None):
    """Basin hopping algorithm."""
    if rng is None:
        rng = np.random.default_rng()
    
    best_energy, best_pos = _local_minimize(positions)
    current_energy, current_pos = best_energy, best_pos
    
    for _ in range(n_iter):
        # Random perturbation
        perturbation = rng.normal(0, step_size, size=current_pos.shape)
        new_pos = current_pos + perturbation
        
        # Local minimize
        new_energy, new_pos_min = _local_minimize(new_pos, maxiter=100)
        
        # Metropolis accept
        if new_energy < current_energy:
            current_energy, current_pos = new_energy, new_pos_min
        else:
            # Accept with some probability (simplified: always accept lower, reject higher for speed)
            # For speed, just keep best
            pass
        
        if new_energy < best_energy:
            best_energy, best_pos = new_energy, new_pos_min
    
    return best_energy, best_pos


def construct(instance):
    """Find low-energy Dzugutov cluster structures."""
    size = instance["n"]
    start_time = time.time()
    deadline = start_time + 280  # Safety margin
    
    rng = np.random.default_rng(size)
    
    # Generate multiple diverse starting structures
    candidates = []
    
    # Strategy 1: Icosahedral-based seeds
    for trial in range(3):
        # Build from icosahedral units
        if size <= 13:
            ico = _make_icosahedron([0, 0, 0], radius=1.0, rng=rng)
            if len(ico) == size:
                candidates.append(ico)
            else:
                # Take first 'size' atoms
                candidates.append(ico[:size])
        elif size <= 30:
            # Two icosahedra
            ico1 = _make_icosahedron([-1.5, 0, 0], radius=0.9, rng=rng)
            ico2 = _make_icosahedron([1.5, 0, 0], radius=0.9, rng=rng)
            combined = np.vstack([ico1, ico2])
            candidates.append(combined[:size])
        else:
            # Multiple icosahedra
            n_icos = max(1, size // 13)
            all_atoms = []
            for i in range(n_icos):
                angle = 2 * np.pi * i / n_icos
                center = [2.0 * np.cos(angle), 2.0 * np.sin(angle), 0]
                ico = _make_icosahedron(center, radius=0.8, rng=rng)
                all_atoms.append(ico)
            combined = np.vstack(all_atoms)
            candidates.append(combined[:size])
    
    # Strategy 2: Random compact clusters
    for trial in range(3):
        radius = 1.1 * size ** (1.0 / 3.0) * 0.75
        points = []
        while len(points) < size:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.6 for q in points):
                points.append(p)
        candidates.append(np.array(points))
    
    # Strategy 3: Decahedron-based for larger clusters
    if size >= 40:
        for trial in range(2):
            dec = _make_decahedron([0, 0, 0], radius=1.0, rng=rng)
            if len(dec) >= size:
                candidates.append(dec[:size])
            else:
                # Add random atoms
                extra = rng.uniform(-2, 2, size=(size - len(dec), 3))
                combined = np.vstack([dec, extra])
                candidates.append(combined)
    
    # Relax each candidate and keep the best
    best_energy = np.inf
    best_positions = None
    
    for i, cand in enumerate(candidates):
        if time.time() > deadline - 30:
            break
        
        # Local relaxation
        try:
            e, pos = _local_minimize(cand, maxiter=150)
            if e < best_energy:
                best_energy = e
                best_positions = pos
        except Exception:
            continue
    
    # Basin hopping on the best candidate if time permits
    if best_positions is not None and time.time() < deadline - 60:
        try:
            bh_energy, bh_pos = _basin_hopping(best_positions, n_iter=30, step_size=0.4, rng=rng)
            if bh_energy < best_energy:
                best_energy = bh_energy
                best_positions = bh_pos
        except Exception:
            pass
    
    # Final check: ensure minimum distance constraint
    if best_positions is not None:
        # Verify and fix if needed
        diff = best_positions[:, np.newaxis, :] - best_positions[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=-1))
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        if min_dist < 0.5:
            # Scale up
            scale = 0.5 / min_dist * 1.01
            best_positions = best_positions * scale
    
    return best_positions
# EVOLVE-BLOCK-END
