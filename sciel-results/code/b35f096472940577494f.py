"""Improved Dzugutov cluster optimizer with better polytetrahedral seeding."""

import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax
import autoresearch_suite.chem_clusters as chem_clusters


# Global deadline guard
TOTAL_TIME_LIMIT = 320.0  # Leave 10s safety margin for 330s kill
_start_time = None
_best_energies = {}  # cache best per n


def _elapsed():
    if _start_time is None:
        return 0.0
    return time.time() - _start_time


def _remaining():
    return max(0.0, TOTAL_TIME_LIMIT - _elapsed())


def _dzugutov_energy_gradient(positions):
    """Compute Dzugutov energy and gradient using the provided helper if available,
    else compute manually with vectorized operations."""
    try:
        energy, grad = chem_clusters.dzugutov_energy_gradient(positions)
        return energy, grad
    except Exception:
        pass
    
    # Manual vectorized computation
    A, B, a, b, c, d, m = 5.82, 1.28, 1.87, 1.94, 1.10, 0.27, 16
    n = len(positions)
    energy = 0.0
    grad = np.zeros_like(positions)
    
    # Compute all pairwise distances
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    for i in range(n):
        for j in range(i + 1, n):
            r = dists[i, j]
            if r < 1e-8:
                continue
            
            r_vec = diff[i, j]
            
            if r < a:
                V = (A * (r ** (-m) - B)) * np.exp(c / (r - a))
                dVdr = A * (-m * r ** (-m - 1) - c / (r - a) ** 2 * (r ** (-m) - B)) * np.exp(c / (r - a))
            elif r < b:
                V = B * np.exp(d / (r - b))
                dVdr = B * np.exp(d / (r - b)) * (-d / (r - b) ** 2)
            else:
                continue
            
            energy += V
            force_factor = dVdr / r
            grad[i] -= force_factor * r_vec
            grad[j] += force_factor * r_vec
    
    return energy, grad


def _make_icosahedral_shell(center, radius, num_atoms, rng):
    """Place atoms on an icosahedral shell using proper icosahedral symmetry."""
    phi = (1 + np.sqrt(5)) / 2  # golden ratio
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    ico_verts = ico_verts / np.linalg.norm(ico_verts, axis=1, keepdims=True)
    
    if num_atoms <= 12:
        verts = ico_verts[:num_atoms]
    else:
        # Use icosahedral shell: place extra atoms at edge midpoints or face centers
        # For a proper polytetrahedral structure, use concentric shells
        # First fill the 12 vertices, then add atoms at symmetric positions
        verts_list = [ico_verts[i] for i in range(12)]
        
        # Add atoms at face centers (20 faces of icosahedron)
        # Each face center is the normalized sum of 3 vertices
        # Get unique faces from icosahedron
        faces = [
            (0, 1, 8), (0, 1, 11), (0, 2, 8), (0, 2, 9),
            (0, 3, 9), (0, 3, 11), (1, 4, 8), (1, 4, 10),
            (1, 5, 10), (1, 5, 11), (2, 4, 8), (2, 4, 9),
            (2, 5, 9), (2, 5, 10), (3, 6, 9), (3, 6, 11),
            (3, 7, 11), (3, 7, 10), (6, 4, 8), (6, 7, 10)
        ]
        
        face_centers = []
        for f in faces:
            fc = (ico_verts[f[0]] + ico_verts[f[1]] + ico_verts[f[2]]) / 3.0
            fc = fc / np.linalg.norm(fc)
            face_centers.append(fc)
        
        # Add edge midpoints if needed
        edges = []
        for f in faces:
            for k in range(3):
                e = (f[k], f[(k+1)%3])
                if e not in edges and (e[1], e[0]) not in edges:
                    edges.append(e)
        
        edge_midpoints = []
        for e in edges:
            em = (ico_verts[e[0]] + ico_verts[e[1]]) / 2.0
            em = em / np.linalg.norm(em)
            edge_midpoints.append(em)
        
        # Build full set of candidate positions: vertices, face centers, edge midpoints
        all_candidates = list(verts_list) + face_centers + edge_midpoints
        
        # Remove near-duplicates
        unique_candidates = []
        for cand in all_candidates:
            if all(np.linalg.norm(cand - uc) > 0.05 for uc in unique_candidates):
                unique_candidates.append(cand)
        
        # Pick the required number
        if num_atoms <= len(unique_candidates):
            verts = np.array(unique_candidates[:num_atoms])
        else:
            # Fallback: distribute on sphere with perturbation
            verts = [ico_verts[i % 12] for i in range(num_atoms)]
            for i in range(12, num_atoms):
                verts[i] = verts[i] + rng.normal(0, 0.15, 3)
                verts[i] = verts[i] / np.linalg.norm(verts[i])
            verts = np.array(verts)
    
    positions = center + radius * verts
    return positions


def _make_random_bcc_like(n, spacing, rng):
    """Generate a compact cluster with FCC-like ordering (better for polytetrahedral)."""
    positions = [np.zeros(3)]
    remaining = n - 1
    
    a = spacing * 0.9
    lattice_pts = []
    for ix in range(-3, 4):
        for iy in range(-3, 4):
            for iz in range(-3, 4):
                p = np.array([ix, iy, iz]) * a
                lattice_pts.append(p)
                # FCC: add face centers
                for fc in [(0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]:
                    pc = np.array([ix + fc[0], iy + fc[1], iz + fc[2]]) * a
                    lattice_pts.append(pc)
    
    lattice_pts = np.array(lattice_pts)
    dists = np.linalg.norm(lattice_pts, axis=1)
    order = np.argsort(dists)
    lattice_pts = lattice_pts[order]
    
    # Remove origin and pick closest
    nonzero_mask = dists > 0.1
    lattice_pts = lattice_pts[nonzero_mask]
    
    for i in range(min(remaining, len(lattice_pts))):
        positions.append(lattice_pts[i])
    
    while len(positions) < n:
        p = rng.uniform(-4 * spacing, 4 * spacing, 3)
        positions.append(p)
    
    return np.array(positions[:n])


def _make_random_compact(n, spacing, rng):
    """Generate a random compact cluster with minimum distance enforcement."""
    positions = []
    radius = spacing * n ** (1.0 / 3.0) * 0.9
    min_dist = 0.6 * spacing
    max_attempts = 20000
    
    for _ in range(max_attempts):
        if len(positions) >= n:
            break
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= min_dist for q in positions):
                positions.append(p)
    
    while len(positions) < n:
        p = rng.uniform(-radius, radius, 3)
        positions.append(p)
    
    return np.array(positions[:n])


def _make_multi_shell_icosahedral(n, spacing, rng):
    """Create a multi-shell icosahedral structure for larger clusters."""
    positions = [np.zeros(3)]  # center
    remaining = n - 1
    
    # Shell 1: 12 atoms at icosahedral vertices
    shell1_count = min(12, remaining)
    r1 = spacing * 0.95
    shell1 = _make_icosahedral_shell(np.zeros(3), r1, shell1_count, rng)
    positions.extend(list(shell1))
    remaining -= shell1_count
    
    # Shell 2: 30 atoms (edge midpoints + face centers) at larger radius
    if remaining > 0:
        shell2_count = min(30, remaining)
        r2 = spacing * 1.75
        # For shell 2, use a different set of icosahedral directions
        phi = (1 + np.sqrt(5)) / 2
        # 30 directions: 12 vertices + 20 face centers (or just use perturbed vertices)
        shell2 = _make_icosahedral_shell(np.zeros(3), r2, shell2_count, rng)
        positions.extend(list(shell2))
        remaining -= shell2_count
    
    # Shell 3: additional atoms
    if remaining > 0:
        r3 = spacing * 2.6
        shell3 = _make_icosahedral_shell(np.zeros(3), r3, remaining, rng)
        positions.extend(list(shell3))
    
    return np.array(positions[:n])


def _optimize_positions(positions, potential, parameter, time_budget, rng):
    """Optimize positions using L-BFGS-B with the Dzugutov potential."""
    n_atoms = len(positions)
    
    def objective_with_grad(x):
        pos = x.reshape(-1, 3)
        try:
            energy, grad = _dzugutov_energy_gradient(pos)
            # Add penalty for atoms too close (vectorized)
            diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff ** 2, axis=2))
            np.fill_diagonal(dists, np.inf)
            
            mask = dists < 0.5
            if np.any(mask):
                penalty = np.sum((0.5 - dists[mask]) ** 2)
                # Gradient of penalty
                grad_pen = np.zeros_like(grad)
                for i in range(n_atoms):
                    for j in range(i + 1, n_atoms):
                        d = dists[i, j]
                        if d < 0.5 and d > 1e-8:
                            force_factor = 2 * (0.5 - d) / d
                            r_vec = diff[i, j]
                            grad_pen[i] -= force_factor * r_vec
                            grad_pen[j] += force_factor * r_vec
                return energy + penalty, (grad + grad_pen).flatten()
            return energy, grad.flatten()
        except Exception:
            return 1e10, np.zeros_like(x)
    
    x0 = positions.flatten()
    best_energy = 1e10
    best_x = x0.copy()
    deadline = time.time() + time_budget
    
    try:
        result = minimize(
            objective_with_grad, x0, method='L-BFGS-B',
            options={'maxiter': 300, 'maxfun': 3000, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        if result.fun < best_energy:
            best_energy = result.fun
            best_x = result.x.copy()
        
        # Try perturbations if time allows
        if time.time() < deadline - 3:
            for trial in range(2):
                if time.time() > deadline - 2:
                    break
                perturbed = positions + rng.normal(0, 0.08, positions.shape)
                x0_p = perturbed.flatten()
                result_p = minimize(
                    objective_with_grad, x0_p, method='L-BFGS-B',
                    options={'maxiter': 200, 'maxfun': 2000, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if result_p.fun < best_energy:
                    best_energy = result_p.fun
                    best_x = result_p.x.copy()
    except Exception:
        pass
    
    return best_x.reshape(-1, 3)


def _try_relax_api(positions, potential, parameter):
    """Try using the provided relax API."""
    try:
        energy, pos = relax(positions, potential, parameter)
        return energy, pos
    except Exception:
        return None, None


def _enforce_min_distance(positions, rng, min_dist=0.5):
    """Separate any atoms closer than min_dist."""
    pos = positions.copy()
    for _ in range(20):
        needs_fix = False
        for i in range(len(pos)):
            for j in range(i + 1, len(pos)):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < min_dist:
                    needs_fix = True
                    diff = pos[i] - pos[j]
                    if np.linalg.norm(diff) < 1e-8:
                        diff = rng.normal(0, 1, 3)
                    diff = diff / np.linalg.norm(diff)
                    push = (min_dist - d) / 2 + 0.01
                    pos[i] += diff * push
                    pos[j] -= diff * push
        if not needs_fix:
            break
    return pos


def construct(instance):
    """Find low-energy structures of Dzugutov clusters using model-and-optimize."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    n = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    spacing = 1.1
    
    if n in _best_energies:
        return _best_energies[n]
    
    instances_to_do = [38, 45, 50, 55, 57, 60]
    idx = instances_to_do.index(n) if n in instances_to_do else 0
    base_time_per_instance = TOTAL_TIME_LIMIT / 6.0
    time_budget = min(_remaining() * 0.85, base_time_per_instance * 1.5)
    time_budget = max(time_budget, 8.0)
    
    rng = np.random.default_rng(42 + n)
    
    # Generate diverse initial structures
    candidates = []
    
    # 1. Multi-shell icosahedral (best for polytetrahedral)
    multi_shell = _make_multi_shell_icosahedral(n, spacing, rng)
    candidates.append(multi_shell)
    
    # 2. Single-shell icosahedral
    if n >= 13:
        ico_pos = _make_icosahedral_shell(np.zeros(3), spacing * 0.95, n, rng)
        candidates.append(ico_pos)
    
    # 3. FCC-like
    fcc_pos = _make_random_bcc_like(n, spacing, rng)
    candidates.append(fcc_pos)
    
    # 4. Random compact (2 variants)
    for _ in range(2):
        rand_pos = _make_random_compact(n, spacing, rng)
        candidates.append(rand_pos)
    
    # 5. Different spacing variants for multi-shell
    for sp in [0.95, 1.2]:
        ms = _make_multi_shell_icosahedral(n, sp, rng)
        candidates.append(ms)
    
    best_energy = np.inf
    best_positions = None
    
    # Optimize each candidate
    for i, cand in enumerate(candidates):
        if _remaining() < 4:
            break
        
        # Enforce minimum distance
        cand = _enforce_min_distance(cand, rng)
        
        # Try L-BFGS optimization
        if _remaining() > 6:
            opt_time = min(_remaining() * 0.35, 25.0)
            opt_pos = _optimize_positions(cand, potential, parameter, opt_time, rng)
            
            try:
                final_energy, final_grad = _dzugutov_energy_gradient(opt_pos)
                if final_energy < best_energy:
                    best_energy = final_energy
                    best_positions = opt_pos.copy()
            except Exception:
                pass
        
        # Also try relax API on the candidate
        if _remaining() > 3:
            rel_energy, rel_pos = _try_relax_api(cand, potential, parameter)
            if rel_energy is not None and rel_energy < best_energy:
                best_energy = rel_energy
                best_positions = rel_pos.copy()
            
            # Try relax on optimized positions if we have them
            if best_positions is not None and best_positions is not cand:
                rel_energy2, rel_pos2 = _try_relax_api(best_positions, potential, parameter)
                if rel_energy2 is not None and rel_energy2 < best_energy:
                    best_energy = rel_energy2
                    best_positions = rel_pos2.copy()
    
    # Final fallback
    if best_positions is None:
        if len(candidates) > 0:
            best_positions = _enforce_min_distance(candidates[0], rng)
            rel_energy, rel_pos = _try_relax_api(best_positions, potential, parameter)
            if rel_pos is not None:
                best_positions = rel_pos
        else:
            best_positions = rng.uniform(-3, 3, (n, 3))
            rel_energy, rel_pos = _try_relax_api(best_positions, potential, parameter)
            if rel_pos is not None:
                best_positions = rel_pos
    
    # Final min-distance enforcement
    final_pos = _enforce_min_distance(best_positions, rng)
    
    _best_energies[n] = final_pos
    return final_pos
