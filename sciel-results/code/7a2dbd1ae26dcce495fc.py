"""Morse cluster global minima via structured seeds + L-BFGS-B with explicit gradient."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient


def _morse_energy_and_gradient(pos, rho):
    """Compute Morse energy and gradient simultaneously (vectorized)."""
    n = pos.shape[0]
    if n < 2:
        return 0.0, np.zeros_like(pos)
    
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    dist = np.sqrt(dist_sq + 1e-30)
    np.fill_diagonal(dist, np.inf)
    np.fill_diagonal(dist_sq, np.inf)
    
    iu = np.triu_indices(n, 1)
    r = dist[iu]
    
    val = rho * (1.0 - r)
    exp_val = np.exp(val)
    energy = np.sum(exp_val * (exp_val - 2.0))
    
    # Gradient: dE/dr_ij = rho * exp(2*rho*(1-r_ij)) * (1 - 2*exp(rho*(1-r_ij))) * (-1)
    # Actually: E_ij = e^v (e^v - 2) where v = rho(1-r)
    # dE_ij/dr = e^v * (e^v - 2)' * dv/dr = e^v * e^v * dv/dr = e^{2v} * (-rho)
    # Wait: d/dr [e^v(e^v-2)] = e^v * e^v * v' = e^{2v} * v'
    # v' = dv/dr = -rho
    # So dE/dr = -rho * e^{2v}
    # And dE/dpos_i = sum_j dE/dr_ij * d r_ij / d pos_i = sum_j (-rho * e^{2v_ij}) * (pos_i - pos_j)/r_ij
    
    grad = np.zeros_like(pos)
    # dE/dr_ij for each pair
    dEdr = -rho * np.exp(2.0 * val)
    # For each pair (i,j), contribution to grad[i] += dEdr_ij * (pos_i - pos_j)/r_ij
    # and grad[j] += dEdr_ij * (pos_j - pos_i)/r_ij = -dEdr_ij * (pos_i - pos_j)/r_ij
    
    # Vectorized: for each pair, diff[i,j] = pos[i] - pos[j]
    diff_iu = diff[iu]  # shape (n_pairs, 3)
    r_iu = r[:, np.newaxis]  # shape (n_pairs, 1)
    
    # unit vectors from i to j
    unit = diff_iu / r_iu  # (pos_i - pos_j)/r_ij
    
    # dE/dpos_i for each i: sum over j of dEdr_ij * unit_ij
    # dE/dpos_j for each j: sum over i of -dEdr_ij * unit_ij
    
    # Build gradient via accumulation
    i_idx, j_idx = iu
    np.add.at(grad, i_idx, dEdr[:, np.newaxis] * unit)
    np.add.at(grad, j_idx, -dEdr[:, np.newaxis] * unit)
    
    return float(energy), grad


def _enforce_min_distance(pos, min_dist=0.5, max_iter=100):
    """Push apart atoms that are too close."""
    pos = pos.copy()
    n = len(pos)
    for _ in range(max_iter):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < min_dist and d > 1e-10:
                    push = (min_dist - d + 0.01) / 2.0
                    direction = (pos[i] - pos[j]) / d
                    pos[i] += push * direction
                    pos[j] -= push * direction
                    moved = True
                elif d <= 1e-10:
                    # Overlapping, push apart randomly
                    direction = np.random.randn(3)
                    direction /= np.linalg.norm(direction)
                    pos[i] += 0.1 * direction
                    pos[j] -= 0.1 * direction
                    moved = True
        if not moved:
            break
    return pos


def _make_icosahedral_cluster(n, seed=0):
    """Create an icosahedral cluster with proper shell structure."""
    rng = np.random.default_rng(seed)
    
    # Icosahedron vertices (12 atoms) - regular icosahedron with edge ~1
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
    ], dtype=float)
    # Scale so nearest neighbor distance is ~1
    # Edge length of this icosahedron: distance between [0,1,phi] and [1,phi,0]
    # = sqrt(1 + (phi-1)^2 + phi^2) = sqrt(1 + (1/phi)^2 + phi^2)
    # phi^2 = phi+1, (1/phi)^2 = 1/(phi+1) = phi-1
    # = sqrt(1 + phi-1 + phi+1) = sqrt(1+2phi+1) = sqrt(2+2phi) = sqrt(2(1+phi))
    edge = np.sqrt(2.0 * (1.0 + phi))
    verts /= edge  # Now edge length = 1
    
    points = list(verts)
    
    if n <= 12:
        return np.array(points[:n])
    
    # Add second shell (30 atoms) - vertices of a truncated icosahedron or similar
    # For simplicity, add atoms on a sphere at radius ~1.6
    shell_r = 1.6
    remaining = n - len(points)
    
    # Use Fibonacci sphere for uniform distribution
    for k in range(remaining):
        # Fibonacci sphere point
        golden = (1.0 + np.sqrt(5.0)) / 2.0
        y = 1.0 - 2.0 * (k + 0.5) / remaining
        r_y = np.sqrt(1.0 - y * y)
        theta = 2.0 * np.pi * k / golden
        p = shell_r * np.array([r_y * np.cos(theta), r_y * np.sin(theta), y])
        points.append(p)
    
    return np.array(points[:n])


def _make_decahedral_cluster(n, seed=0):
    """Create a decahedral cluster (FCC-based, 5-fold symmetry)."""
    rng = np.random.default_rng(seed)
    
    # FCC lattice with 5-fold axis
    # Generate FCC points and select those forming a decahedral shape
    a = 1.0  # lattice parameter, nearest neighbor = a/sqrt(2)
    points = []
    
    # Generate FCC points in a cube
    for i in range(-4, 5):
        for j in range(-4, 5):
            for k in range(-4, 5):
                # FCC: corners + face centers
                p = a * np.array([i, j, k])
                points.append(p.copy())
                if i < 4:
                    points.append(a * np.array([i + 0.5, j, k + 0.5]))
                    points.append(a * np.array([i + 0.5, j + 0.5, k]))
                if j < 4:
                    points.append(a * np.array([i, j + 0.5, k + 0.5]))
    
    points = np.array(points)
    
    # Remove duplicates
    unique = []
    for p in points:
        if not any(np.linalg.norm(p - q) < 0.05 for q in unique):
            unique.append(p)
    unique = np.array(unique)
    
    # Sort by distance from origin
    dists = np.linalg.norm(unique, axis=1)
    idx = np.argsort(dists)
    unique = unique[idx]
    
    # Scale so nearest neighbor ~ 1
    if len(unique) >= 2:
        # Find typical nearest neighbor distance
        from scipy.spatial import cKDTree
        tree = cKDTree(unique[:min(50, len(unique))])
        dd, _ = tree.query(unique[:min(50, len(unique))], k=2)
        nn_dists = dd[:, 1]
        nn_dists = nn_dists[nn_dists > 0.1]
        if len(nn_dists) > 0:
            scale = 1.0 / np.median(nn_dists)
            unique *= scale
    
    if len(unique) >= n:
        return unique[:n].copy()
    else:
        # Pad with points on a sphere
        while len(unique) < n:
            theta = rng.uniform(0, 2 * np.pi)
            phi_a = rng.uniform(0, np.pi)
            r = 3.0
            p = r * np.array([np.sin(phi_a) * np.cos(theta), np.sin(phi_a) * np.sin(theta), np.cos(phi_a)])
            unique = np.vstack([unique, [p]])
        return unique[:n].copy()


def _make_fcc_cluster(n, seed=0):
    """Create an FCC cluster fragment."""
    rng = np.random.default_rng(seed)
    
    a = np.sqrt(2.0)  # so nearest neighbor = 1
    points = []
    
    for i in range(-5, 6):
        for j in range(-5, 6):
            for k in range(-5, 6):
                p = a * np.array([i, j, k])
                points.append(p.copy())
                if i < 5:
                    points.append(a * np.array([i + 0.5, j, k + 0.5]))
                    points.append(a * np.array([i + 0.5, j + 0.5, k]))
                if j < 5:
                    points.append(a * np.array([i, j + 0.5, k + 0.5]))
    
    points = np.array(points)
    
    # Remove duplicates
    unique = []
    for p in points:
        if not any(np.linalg.norm(p - q) < 0.05 for q in unique):
            unique.append(p)
    unique = np.array(unique)
    
    # Sort by distance from origin
    dists = np.linalg.norm(unique, axis=1)
    idx = np.argsort(dists)
    unique = unique[idx]
    
    if len(unique) >= n:
        return unique[:n].copy()
    else:
        while len(unique) < n:
            p = rng.uniform(-5, 5, 3)
            unique = np.vstack([unique, [p]])
        return unique[:n].copy()


def _make_hcp_cluster(n, seed=0):
    """Create an HCP cluster fragment."""
    rng = np.random.default_rng(seed)
    
    a = 1.0  # nearest neighbor
    c = a * np.sqrt(8.0 / 3.0)  # ideal c/a ratio
    points = []
    
    # HCP: ABAB stacking
    for layer in range(-4, 5):
        z = layer * c
        for i in range(-5, 6):
            for j in range(-5, 6):
                if layer % 2 == 0:
                    x = a * (i + 0.5 * (j % 2))
                    y = a * j * np.sqrt(3.0) / 2.0
                else:
                    x = a * (i + 0.5)
                    y = a * (j + 0.5) * np.sqrt(3.0) / 2.0
                points.append(np.array([x, y, z]))
    
    points = np.array(points)
    
    # Remove duplicates
    unique = []
    for p in points:
        if not any(np.linalg.norm(p - q) < 0.05 for q in unique):
            unique.append(p)
    unique = np.array(unique)
    
    # Sort by distance from origin
    dists = np.linalg.norm(unique, axis=1)
    idx = np.argsort(dists)
    unique = unique[idx]
    
    if len(unique) >= n:
        return unique[:n].copy()
    else:
        while len(unique) < n:
            p = rng.uniform(-5, 5, 3)
            unique = np.vstack([unique, [p]])
        return unique[:n].copy()


def _make_random_compact(n, seed=0):
    """Create a random compact cluster."""
    rng = np.random.default_rng(seed)
    radius = (n / (4.0 / 3.0 * np.pi * 0.5**3)) ** (1.0 / 3.0) * 0.8
    points = []
    for _ in range(n):
        for _ in range(200):
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.5 for q in points):
                points.append(p)
                break
        else:
            p = rng.uniform(-radius, radius, 3)
            points.append(p)
    return np.array(points)


def _make_spherical_shell(n, seed=0):
    """Atoms on a spherical shell."""
    rng = np.random.default_rng(seed)
    radius = (n / (4.0 * np.pi)) ** 0.5
    points = []
    for _ in range(n):
        theta = rng.uniform(0, 2 * np.pi)
        phi_a = np.arccos(2.0 * rng.uniform(0, 1) - 1.0)
        p = radius * np.array([np.sin(phi_a) * np.cos(theta), np.sin(phi_a) * np.sin(theta), np.cos(phi_a)])
        points.append(p)
    return np.array(points)


def _lbfgs_optimize(pos, rho, time_budget=10.0, maxiter=500):
    """Run L-BFGS-B optimization with explicit gradient."""
    start = time.time()
    
    def objective(x):
        pos_2d = x.reshape(-1, 3)
        e, _ = _morse_energy_and_gradient(pos_2d, rho)
        return e
    
    def gradient(x):
        pos_2d = x.reshape(-1, 3)
        _, g = _morse_energy_and_gradient(pos_2d, rho)
        return g.ravel()
    
    x0 = pos.ravel()
    
    try:
        result = minimize(
            objective, x0, jac=gradient, method='L-BFGS-B',
            options={'maxiter': maxiter, 'maxfun': 2000, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        if result.success or result.fun < objective(x0):
            return result.x.reshape(-1, 3), result.fun
    except:
        pass
    
    return pos, _morse_energy_and_gradient(pos, rho)[0]


def _optimize_with_restarts(pos, rho, n_restarts=5, time_budget=20.0):
    """Optimize with multiple perturbation restarts."""
    start = time.time()
    best_pos = pos.copy()
    best_energy = _morse_energy_and_gradient(pos, rho)[0]
    
    # First, optimize the original
    pos_opt, e_opt = _lbfgs_optimize(pos, rho, time_budget=time_budget / 2)
    if e_opt < best_energy:
        best_energy = e_opt
        best_pos = pos_opt.copy()
    
    # Perturbation restarts
    rng = np.random.default_rng(42)
    for i in range(n_restarts):
        if time.time() - start > time_budget * 0.8:
            break
        
        # Random perturbation
        magnitude = 0.1 + 0.1 * i
        perturbed = best_pos + rng.uniform(-magnitude, magnitude, best_pos.shape)
        perturbed = _enforce_min_distance(perturbed)
        
        pos_opt, e_opt = _lbfgs_optimize(perturbed, rho, time_budget=time_budget / (n_restarts + 1))
        if e_opt < best_energy:
            best_energy = e_opt
            best_pos = pos_opt.copy()
    
    return best_pos, best_energy


def construct(instance):
    """Find the lowest-energy arrangement for a Morse cluster."""
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind", 3.0))
    potential = instance.get("potential", "morse")
    
    start_time = time.time()
    deadline = start_time + 290  # Safety margin (300s total, 10s buffer)
    
    def time_left():
        return deadline - time.time()
    
    # Generate diverse initial seeds
    seeds = []
    
    # Icosahedral
    for s in [0, 1, 2]:
        try:
            seeds.append(('ico', _make_icosahedral_cluster(n, seed=s)))
        except:
            pass
    
    # Decahedral
    for s in [0, 1]:
        try:
            seeds.append(('deca', _make_decahedral_cluster(n, seed=s)))
        except:
            pass
    
    # FCC
    for s in [0, 1]:
        try:
            seeds.append(('fcc', _make_fcc_cluster(n, seed=s)))
        except:
            pass
    
    # HCP
    for s in [0, 1]:
        try:
            seeds.append(('hcp', _make_hcp_cluster(n, seed=s)))
        except:
            pass
    
    # Random compact
    for s in range(6):
        try:
            seeds.append(('rand', _make_random_compact(n, seed=s)))
        except:
            pass
    
    # Spherical shell
    for s in [0, 1]:
        try:
            seeds.append(('shell', _make_spherical_shell(n, seed=s)))
        except:
            pass
    
    # Sort seeds by initial energy (quick eval)
    seed_energies = []
    for name, pos in seeds:
        e = _morse_energy_and_gradient(pos, rho)[0]
        seed_energies.append((e, name, pos))
    seed_energies.sort(key=lambda x: x[0])
    
    # Optimize each seed
    best_pos = None
    best_energy = np.inf
    
    # Allocate time: top seeds get more time
    n_seeds = len(seed_energies)
    base_time = 15.0  # seconds per seed
    
    for rank, (e_init, name, pos) in enumerate(seed_energies):
        if time_left() < 30:
            break
        
        # Time allocation: top seeds get more
        if rank < 3:
            t_budget = min(base_time, time_left() * 0.3)
            n_restarts = 5
        elif rank < 6:
            t_budget = min(base_time * 0.7, time_left() * 0.2)
            n_restarts = 3
        else:
            t_budget = min(base_time * 0.4, time_left() * 0.1)
            n_restarts = 2
        
        try:
            pos_opt, e_opt = _optimize_with_restarts(pos, rho, n_restarts=n_restarts, time_budget=t_budget)
            if e_opt < best_energy:
                best_energy = e_opt
                best_pos = pos_opt.copy()
        except:
            continue
    
    # If we have time, do more random restarts from the best
    if best_pos is not None and time_left() > 30:
        rng = np.random.default_rng(123)
        remaining = time_left() - 20
        n_extra = min(10, int(remaining / 5))
        
        for i in range(n_extra):
            if time_left() < 25:
                break
            try:
                # Random perturbation of best
                magnitude = 0.2 + 0.1 * (i % 3)
                perturbed = best_pos + rng.uniform(-magnitude, magnitude, best_pos.shape)
                perturbed = _enforce_min_distance(perturbed)
                
                pos_opt, e_opt = _lbfgs_optimize(perturbed, rho, time_budget=5.0)
                if e_opt < best_energy:
                    best_energy = e_opt
                    best_pos = pos_opt.copy()
            except:
                continue
    
    # Final: also try the library's relax as a cross-check
    if best_pos is not None and time_left() > 15:
        try:
            e_relax, pos_relax = relax(best_pos, potential, rho)
            if e_relax < best_energy:
                best_energy = e_relax
                best_pos = pos_relax.copy()
        except:
            pass
    
    if best_pos is None:
        # Fallback
        best_pos = _make_random_compact(n, seed=0)
    
    # Ensure minimum distance
    best_pos = _enforce_min_distance(best_pos)
    
    return best_pos
# EVOLVE-BLOCK-END
