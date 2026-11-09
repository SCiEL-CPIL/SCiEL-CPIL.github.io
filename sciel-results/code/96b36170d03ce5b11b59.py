"""Sutton-Chen cluster global minima via LAMMPS-style growth + basin hopping."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

# Deadline guard: 330s wall-clock limit, use 290s to be safe
_DEADLINE = time.time() + 290


def _check_deadline():
    return time.time() > _DEADLINE


def _sc_energy(pos, n_exp, m_exp, c):
    """Compute Sutton-Chen energy."""
    n = len(pos)
    if n < 2:
        return 0.0
    d = cdist(pos, pos)
    np.fill_diagonal(d, np.inf)
    d = np.maximum(d, 1e-12)
    r_inv_n = np.sum(d ** (-n_exp), axis=1)
    r_inv_m = np.sum(d ** (-m_exp), axis=1)
    e_rep = 0.5 * np.sum(r_inv_n)
    e_att = -c * np.sum(np.sqrt(r_inv_m))
    return e_rep + e_att


def _sc_gradient(pos, n_exp, m_exp, c):
    """Compute Sutton-Chen energy gradient."""
    n = len(pos)
    if n < 2:
        return np.zeros_like(pos)
    d = cdist(pos, pos)
    np.fill_diagonal(d, np.inf)
    d = np.maximum(d, 1e-12)
    
    # r_inv_n[j] = sum_{i != j} r_ij^(-n_exp)
    r_inv_n = np.sum(d ** (-n_exp), axis=1)
    # r_inv_m[j] = sum_{i != j} r_ij^(-m_exp)
    r_inv_m = np.sum(d ** (-m_exp), axis=1)
    
    # Gradient of repulsive term: 0.5 * sum_j [n_exp * r_ij^(-n_exp-1) * (pos_i - pos_j)/r_ij]
    # = 0.5 * n_exp * sum_j r_ij^(-n_exp-2) * (pos_i - pos_j)
    # Actually: d/dr [r^(-n)] = -n * r^(-n-1)
    # So dE_rep/dpos_i = 0.5 * sum_{j!=i} n_exp * r_ij^(-n_exp-1) * (pos_i - pos_j) / r_ij
    # Wait, let me be more careful.
    # E_rep = 0.5 * sum_i sum_{j!=i} r_ij^(-n_exp)
    # dE_rep/dpos_i = 0.5 * sum_{j!=i} (-n_exp) * r_ij^(-n_exp-2) * (pos_i - pos_j)
    # = -0.5 * n_exp * sum_{j!=i} r_ij^(-n_exp-2) * (pos_i - pos_j)
    
    # E_att = -c * sum_i sqrt(sum_{j!=i} r_ij^(-m_exp))
    # dE_att/dpos_i = -c * 1/(2*sqrt(r_inv_m[i])) * sum_{j!=i} (-m_exp) * r_ij^(-m_exp-2) * (pos_i - pos_j)
    # = c * m_exp / (2 * sqrt(r_inv_m[i])) * sum_{j!=i} r_ij^(-m_exp-2) * (pos_i - pos_j)
    
    # Build the force matrix
    # F[i,j] = r_ij^(-n_exp-2) * (pos_i - pos_j) for repulsive
    # G[i,j] = r_ij^(-m_exp-2) * (pos_i - pos_j) for attractive
    
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]  # (n, n, 3)
    d3 = d[:, :, np.newaxis]  # (n, n, 1)
    
    # Avoid division by zero on diagonal
    mask = np.ones((n, n), dtype=bool)
    np.fill_diagonal(mask, False)
    
    r_n_exp_2 = np.zeros_like(d)
    r_m_exp_2 = np.zeros_like(d)
    r_n_exp_2[mask] = d[mask] ** (-n_exp - 2)
    r_m_exp_2[mask] = d[mask] ** (-m_exp - 2)
    
    # Repulsive gradient contribution
    rep_force = -0.5 * n_exp * r_n_exp_2[:, :, np.newaxis] * diff
    rep_grad = np.sum(rep_force, axis=1)
    
    # Attractive gradient contribution
    # Need 1/sqrt(r_inv_m[i]) for each i
    with np.errstate(divide='ignore', invalid='ignore'):
        inv_sqrt_rim = np.where(r_inv_m > 0, 1.0 / np.sqrt(r_inv_m), 0.0)
    
    att_force = c * m_exp * 0.5 * (inv_sqrt_rim[:, np.newaxis, np.newaxis]) * r_m_exp_2[:, :, np.newaxis] * diff
    att_grad = np.sum(att_force, axis=1)
    
    return rep_grad + att_grad


def _generate_fcc_seed(n, spacing=1.4):
    """Generate an FCC-like cluster of n atoms."""
    # FCC lattice vectors
    a = spacing
    points = []
    # Generate FCC points in a cube
    for i in range(-3, 4):
        for j in range(-3, 4):
            for k in range(-3, 4):
                # FCC: (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0) in units of a
                for dx, dy, dz in [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]:
                    p = np.array([i + dx, j + dy, k + dz]) * a
                    if len(points) < n:
                        points.append(p)
    if len(points) < n:
        # Pad with random points
        rng = np.random.default_rng(42)
        while len(points) < n:
            p = rng.uniform(-a * 3, a * 3, 3)
            points.append(p)
    return np.array(points[:n])


def _generate_hcp_seed(n, spacing=1.4):
    """Generate an HCP-like cluster of n atoms."""
    a = spacing
    c = a * np.sqrt(8.0 / 3.0)
    points = []
    # HCP layers
    for layer in range(-3, 4):
        z = layer * c
        for i in range(-3, 4):
            for j in range(-3, 4):
                if (i + j) % 2 == 0:
                    p = np.array([i * a, j * a, z])
                else:
                    p = np.array([(i + 0.5) * a, (j + 0.5) * a, z])
                if len(points) < n:
                    points.append(p)
    if len(points) < n:
        rng = np.random.default_rng(42)
        while len(points) < n:
            p = rng.uniform(-a * 3, a * 3, 3)
            points.append(p)
    return np.array(points[:n])


def _generate_random_compact(n, spacing=1.4, seed=42):
    """Generate a random compact cluster."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    points = []
    attempts = 0
    while len(points) < n and attempts < n * 50:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.8 * spacing for q in points):
                points.append(p)
        attempts += 1
    # Pad if needed
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    return np.array(points[:n])


def _grow_cluster_lammmps(n, potential, kind, n_exp, m_exp, c, spacing=1.4):
    """
    LAMMPS-style cluster growth: start from small cluster, add atoms one by one.
    """
    # Start with a small seed
    start_n = min(5, n)
    if start_n == 1:
        pos = np.array([[0.0, 0.0, 0.0]])
    elif start_n == 2:
        pos = np.array([[0.0, 0.0, 0.0], [spacing, 0.0, 0.0]])
    else:
        # Small FCC seed
        pos = _generate_fcc_seed(start_n, spacing)
        pos = relax(pos, potential, kind)[1]
    
    # Add atoms one by one
    for target_n in range(start_n + 1, n + 1):
        if _check_deadline():
            break
        
        # Try multiple starting positions for the new atom
        best_energy = np.inf
        best_pos = None
        
        # Candidate positions: on a sphere around the cluster
        centroid = np.mean(pos, axis=0)
        radius = np.max(np.linalg.norm(pos - centroid, axis=1))
        
        n_candidates = 8
        rng = np.random.default_rng(target_n)
        
        for trial in range(n_candidates):
            # Random direction
            direction = rng.normal(size=3)
            direction /= np.linalg.norm(direction)
            
            # Place new atom at various distances
            for dist in [radius * 0.8, radius * 1.0, radius * 1.2, spacing * 1.5]:
                new_pos = centroid + direction * dist
                candidate = np.vstack([pos, new_pos.reshape(1, 3)])
                
                # Check minimum distance
                if np.min(cdist(candidate, candidate)[np.triu_indices_from(cdist(candidate, candidate), k=1)]) < 0.35:
                    continue
                
                # Relax
                try:
                    energy, relaxed = relax(candidate, potential, kind)
                    if energy < best_energy:
                        best_energy = energy
                        best_pos = relaxed
                except:
                    pass
        
        if best_pos is not None:
            pos = best_pos
        else:
            # Fallback: add atom at random position
            new_pos = centroid + rng.normal(scale=spacing, size=3)
            pos = np.vstack([pos, new_pos.reshape(1, 3)])
            pos = relax(pos, potential, kind)[1]
    
    return pos


def _basin_hopping(pos, potential, kind, n_exp, m_exp, c, n_iter=20, step_size=0.5):
    """Simple basin hopping: perturb, relax, accept if lower."""
    current_pos = pos.copy()
    current_energy, _ = sutton_chen_energy_gradient(current_pos, kind)
    best_pos = current_pos.copy()
    best_energy = current_energy
    
    rng = np.random.default_rng(12345)
    
    for i in range(n_iter):
        if _check_deadline():
            break
        
        # Perturb
        perturbed = current_pos + rng.normal(scale=step_size, size=current_pos.shape)
        
        # Check minimum distance
        d = cdist(perturbed, perturbed)
        np.fill_diagonal(d, np.inf)
        if np.min(d) < 0.35:
            continue
        
        # Relax
        try:
            energy, relaxed = relax(perturbed, potential, kind)
        except:
            continue
        
        if energy < current_energy:
            current_pos = relaxed
            current_energy = energy
            if energy < best_energy:
                best_pos = relaxed.copy()
                best_energy = energy
        else:
            # Metropolis acceptance
            if rng.random() < np.exp(-(energy - current_energy) / 0.1):
                current_pos = relaxed
                current_energy = energy
    
    return best_pos, best_energy


def construct(instance):
    """
    Find global minimum of Sutton-Chen cluster using LAMMPS growth + basin hopping.
    """
    n = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", instance.get("rho", "sutton_chen"))
    n_exp = instance.get("n_exp", 12)
    m_exp = instance.get("m_exp", 6)
    c = instance.get("c", 144.41)
    
    # Map kind to parameters
    if kind in ["12-6", "12_6"]:
        n_exp, m_exp, c = 12, 6, 144.41
    elif kind in ["9-6", "9_6"]:
        n_exp, m_exp, c = 9, 6, 39.432
    elif kind in ["10-8", "10_8"]:
        n_exp, m_exp, c = 10, 8, 34.408
    
    best_energy = np.inf
    best_pos = None
    
    # Strategy 1: LAMMPS growth
    try:
        pos = _grow_cluster_lammmps(n, potential, kind, n_exp, m_exp, c)
        energy, relaxed = relax(pos, potential, kind)
        if energy < best_energy:
            best_energy = energy
            best_pos = relaxed
    except:
        pass
    
    if _check_deadline():
        return best_pos if best_pos is not None else _generate_fcc_seed(n)
    
    # Strategy 2: Basin hopping from LAMMPS result
    if best_pos is not None:
        try:
            bh_pos, bh_energy = _basin_hopping(best_pos, potential, kind, n_exp, m_exp, c, n_iter=30)
            if bh_energy < best_energy:
                best_energy = bh_energy
                best_pos = bh_pos
        except:
            pass
    
    if _check_deadline():
        return best_pos
    
    # Strategy 3: Basin hopping from FCC seed
    try:
        fcc_pos = _generate_fcc_seed(n)
        fcc_energy, fcc_relaxed = relax(fcc_pos, potential, kind)
        if fcc_energy < best_energy:
            best_energy = fcc_energy
            best_pos = fcc_relaxed
        
        bh_pos, bh_energy = _basin_hopping(fcc_relaxed, potential, kind, n_exp, m_exp, c, n_iter=30)
        if bh_energy < best_energy:
            best_energy = bh_energy
            best_pos = bh_pos
    except:
        pass
    
    if _check_deadline():
        return best_pos
    
    # Strategy 4: Basin hopping from HCP seed
    try:
        hcp_pos = _generate_hcp_seed(n)
        hcp_energy, hcp_relaxed = relax(hcp_pos, potential, kind)
        if hcp_energy < best_energy:
            best_energy = hcp_energy
            best_pos = hcp_relaxed
        
        bh_pos, bh_energy = _basin_hopping(hcp_relaxed, potential, kind, n_exp, m_exp, c, n_iter=30)
        if bh_energy < best_energy:
            best_energy = bh_energy
            best_pos = bh_pos
    except:
        pass
    
    if _check_deadline():
        return best_pos
    
    # Strategy 5: Multiple random compact seeds with basin hopping
    for seed in [1, 2, 3]:
        if _check_deadline():
            break
        try:
            rand_pos = _generate_random_compact(n, seed=seed)
            rand_energy, rand_relaxed = relax(rand_pos, potential, kind)
            if rand_energy < best_energy:
                best_energy = rand_energy
                best_pos = rand_relaxed
            
            bh_pos, bh_energy = _basin_hopping(rand_relaxed, potential, kind, n_exp, m_exp, c, n_iter=20)
            if bh_energy < best_energy:
                best_energy = bh_energy
                best_pos = bh_pos
        except:
            pass
    
    # Final relaxation
    if best_pos is not None:
        try:
            final_energy, final_pos = relax(best_pos, potential, kind)
            return final_pos
        except:
            return best_pos
    
    # Fallback
    return _generate_fcc_seed(n)
# EVOLVE-BLOCK-END
