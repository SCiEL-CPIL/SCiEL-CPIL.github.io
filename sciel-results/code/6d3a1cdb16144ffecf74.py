"""Sutton-Chen cluster global-minima search: deterministic LAMMPS-style incremental growth."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_DEADLINE = time.time() + 275.0


def _remaining():
    return _DEADLINE - time.time()


def _valid(pos):
    """Check if positions are valid: finite, no two atoms closer than 0.35."""
    if pos is None or not np.all(np.isfinite(pos)):
        return False
    n = len(pos)
    if n < 2:
        return True
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    return d.min() > 0.35


def _energy_for_new_atom(core_positions, new_pos, kind):
    """Compute the energy contribution of adding new_pos to core_positions."""
    # This is the incremental energy: sum over interactions between new_pos and each core atom
    # E_new = sum_{i in core} [1/2 * r^-n_exp - c * sqrt(1/2 * r^-m_exp)]
    # Actually, the full energy change when adding atom k to a cluster of N atoms is:
    # delta_E = sum_{i=1}^{N} [1/2 * r_ik^-n - c * sqrt(1/2 * r_ik^-m)]
    # because the pair (i,k) contributes to both atom i's sum and atom k's sum,
    # but in the total energy each pair is counted once via the 1/2 factor per atom.
    # Let me re-derive: E = sum_i [1/2 sum_{j!=i} r_ij^-n - c sqrt(sum_{j!=i} r_ij^-m)]
    # When we add atom k, the new terms are:
    # - For atom k: 1/2 sum_{i in core} r_ik^-n - c sqrt(sum_{i in core} r_ik^-m)
    # - For each atom i in core: the term 1/2 * r_ik^-n is added to their sum, and r_ik^-m is added
    #   to their inner sum, so their contribution changes.
    # This is complex. Let's just compute the full energy difference.
    
    # Simpler: compute E(core + new) - E(core)
    # But that's expensive. Instead, use the gradient-based approach:
    # The force on the new atom from the core is what we need for optimization.
    # For now, let's just use a simple heuristic: minimize the sum of r^-n and r^-m terms.
    
    distances = np.linalg.norm(core_positions - new_pos, axis=1)
    # Avoid division by zero
    distances = np.maximum(distances, 0.01)
    
    if kind == '12-6':
        n_exp, m_exp, c = 12, 6, 144.41
    elif kind == '9-6':
        n_exp, m_exp, c = 9, 6, 39.432
    elif kind == '10-8':
        n_exp, m_exp, c = 10, 8, 34.408
    else:
        n_exp, m_exp, c = 12, 6, 144.41
    
    # The incremental energy contribution (approximate)
    repulsive = 0.5 * np.sum(distances ** (-n_exp))
    attractive = -c * np.sqrt(np.sum(distances ** (-m_exp)))
    return repulsive + attractive


def _find_best_position_for_new_atom(core_positions, kind, n_candidates=200):
    """Find the best position to add a new atom to the core."""
    if len(core_positions) == 0:
        return np.array([0.0, 0.0, 0.0])
    
    # Generate candidate positions around the core
    # Strategy: place candidates in a shell around the core's convex hull
    center = core_positions.mean(axis=0)
    spread = np.linalg.norm(core_positions - center, axis=1).max()
    radius = spread + 0.3  # Place new atom just outside the core
    
    # Generate candidates on a sphere and also in a box around the core
    rng = np.random.default_rng(12345)
    candidates = []
    
    # Spherical candidates
    for _ in range(n_candidates // 2):
        theta = rng.uniform(0, 2 * np.pi)
        phi = rng.uniform(0, np.pi)
        r = radius * rng.uniform(0.8, 1.2)
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        candidates.append(center + np.array([x, y, z]))
    
    # Box candidates
    box_size = radius * 1.5
    for _ in range(n_candidates // 2):
        p = center + rng.uniform(-box_size, box_size, 3)
        candidates.append(p)
    
    candidates = np.array(candidates)
    
    # Evaluate each candidate
    best_energy = np.inf
    best_pos = candidates[0]
    
    for cand in candidates:
        # Check minimum distance
        dists = np.linalg.norm(core_positions - cand, axis=1)
        if dists.min() < 0.35:
            continue
        e = _energy_for_new_atom(core_positions, cand, kind)
        if e < best_energy:
            best_energy = e
            best_pos = cand
    
    return best_pos


def _lammps_grow(n_target, kind, seed=42):
    """
    Deterministic LAMMPS-style cluster growth.
    Start with a small seed structure, then iteratively add atoms
    in the lowest-energy position, relaxing the whole cluster after each addition.
    """
    rng = np.random.default_rng(seed)
    
    # Start with a small core (2-4 atoms)
    # Use a small FCC or BCC fragment as the seed
    n_start = min(4, n_target)
    
    if n_start == 1:
        cluster = np.array([[0.0, 0.0, 0.0]])
    elif n_start == 2:
        cluster = np.array([[0.0, 0.0, 0.0], [0.7, 0.0, 0.0]])
    elif n_start == 3:
        # Small triangle
        cluster = np.array([
            [0.0, 0.0, 0.0],
            [0.7, 0.0, 0.0],
            [0.35, 0.6, 0.0]
        ])
    else:
        # Small FCC tetrahedron
        a = 0.7
        cluster = np.array([
            [0.0, 0.0, 0.0],
            [a, 0.0, 0.0],
            [0.0, a, 0.0],
            [0.0, 0.0, a]
        ])
    
    # Relax the initial cluster
    if _remaining() < 5.0:
        return cluster
    
    try:
        energy, cluster = relax(cluster, 'sutton_chen', kind)
        if not _valid(cluster):
            # Fallback to simple positions
            cluster = np.array([[i * 0.5, 0, 0] for i in range(n_start)])
    except:
        pass
    
    # Iteratively add atoms
    for n_curr in range(n_start, n_target):
        if _remaining() < 10.0:
            break
        
        # Find best position for new atom
        new_pos = _find_best_position_for_new_atom(cluster, kind)
        
        # Add the new atom
        cluster = np.vstack([cluster, new_pos])
        
        # Relax the entire cluster
        try:
            energy, cluster = relax(cluster, 'sutton_chen', kind)
            if not _valid(cluster):
                # If relaxation failed, try with a smaller step
                cluster = np.vstack([cluster[:-1], new_pos + rng.normal(0, 0.05, 3)])
                energy, cluster = relax(cluster, 'sutton_chen', kind)
                if not _valid(cluster):
                    # Remove the problematic atom and try a different position
                    cluster = cluster[:-1]
                    new_pos2 = _find_best_position_for_new_atom(cluster, kind)
                    cluster = np.vstack([cluster, new_pos2 + rng.normal(0, 0.03, 3)])
                    energy, cluster = relax(cluster, 'sutton_chen', kind)
                    if not _valid(cluster):
                        # Give up on this atom, keep the previous cluster
                        cluster = cluster[:-1]
                        # Try to add a different atom
                        continue
        except:
            # Relaxation failed, try with perturbation
            try:
                perturbed = cluster + rng.normal(0, 0.05, cluster.shape)
                energy, cluster = relax(perturbed, 'sutton_chen', kind)
                if not _valid(cluster):
                    cluster = cluster[:-1]
            except:
                cluster = cluster[:-1]
    
    return cluster


def _lammps_grow_multi(n_target, kind, n_starts=3):
    """Run LAMMPS growth with different seeds and return the best."""
    best_energy = np.inf
    best_cluster = None
    
    for seed in range(n_starts):
        if _remaining() < 15.0:
            break
        
        try:
            cluster = _lammps_grow(n_target, kind, seed=seed)
            if _valid(cluster):
                energy, _ = sutton_chen_energy_gradient(cluster, kind)
                if energy < best_energy:
                    best_energy = energy
                    best_cluster = cluster
        except:
            pass
    
    return best_cluster


def construct(instance):
    """Deterministic LAMMPS-style incremental growth for Sutton-Chen clusters."""
    size = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind")
    
    # Try LAMMPS growth with multiple seeds
    result = _lammps_grow_multi(size, kind, n_starts=4)
    
    if result is not None and _valid(result):
        return result
    
    # Fallback: simple linear arrangement
    positions = np.zeros((size, 3))
    for i in range(size):
        positions[i] = [i * 0.4, 0, 0]
    
    return positions
# EVOLVE-BLOCK-END
