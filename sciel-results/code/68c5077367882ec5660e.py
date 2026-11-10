"""Dzugutov cluster global minima search using hierarchical assembly and basin hopping."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient
import time
import itertools

# Dzugutov potential parameters
A_POT = 5.82
B_POT = 1.28
a_POT = 1.87
b_POT = 1.94
c_POT = 1.10
d_POT = 0.27
m_POT = 16

# Global deadline management
_DEADLINE = None
_TIME_BUDGET = 320.0  # Leave 10s safety margin before 330s kill


def _check_deadline():
    """Return True if we should stop searching."""
    if _DEADLINE is None:
        return False
    return time.time() >= _DEADLINE


def _dzugutov_energy(positions):
    """Compute Dzugutov potential energy for a cluster."""
    n = len(positions)
    if n < 2:
        return 0.0
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(r, np.inf)
    triu_i, triu_j = np.triu_indices(n, k=1)
    r_ij = r[triu_i, triu_j]
    
    # V(r) = A(r^-m - B)*exp(c/(r-a)) for r < a, else B*exp(d/(r-b)) for r < b, else 0
    # Note: for r >= b, V = 0 (since exp(d/(r-b)) -> 0 as r->inf, but actually the formula says r < b)
    # Let me re-read: V(r) = A(r^-m - B)exp(c/(r - a)) [r < a] + B exp(d/(r - b)) [r < b]
    # This is ambiguous. Let me interpret:
    # For r < a: V = A(r^-m - B)*exp(c/(r-a))
    # For a <= r < b: V = B*exp(d/(r-b))  (since the first term's condition is r<a, and second is r<b)
    # Wait, that doesn't make sense either. Let me think again.
    # Actually the standard Dzugutov potential is:
    # V(r) = A*(r^-m - B)*exp(c/(r-a)) for r < a
    # V(r) = B*exp(d/(r-b)) for a <= r < b  
    # V(r) = 0 for r >= b
    # But the notation [r<a] and [r<b] suggests indicator functions.
    # So V(r) = A(r^-m - B)*exp(c/(r-a)) * I(r<a) + B*exp(d/(r-b)) * I(r<b)
    # For r < a: both terms active
    # For a <= r < b: only second term
    # For r >= b: zero
    
    energy = 0.0
    # Mask for r < a
    mask_a = r_ij < a_POT
    # Mask for r < b  
    mask_b = r_ij < b_POT
    
    if np.any(mask_a):
        r_a = r_ij[mask_a]
        # Avoid division by zero or negative in exp
        # r - a < 0 when r < a, so c/(r-a) is negative, exp is fine
        term1 = A_POT * (r_a ** (-m_POT) - B_POT) * np.exp(c_POT / (r_a - a_POT))
        energy += np.sum(term1)
    
    if np.any(mask_b):
        r_b = r_ij[mask_b]
        # r - b < 0 when r < b, so d/(r-b) is negative
        # But for r >= a and r < b, we only add the second term
        # For r < a, both terms are added
        # So we need to be careful: the second term B*exp(d/(r-b)) is added for ALL r < b
        # The first term is added for r < a
        # So for r < a: energy += term1 + term2
        # For a <= r < b: energy += term2 only
        
        # Let's recompute more carefully
        # Actually, let me just compute both terms for all pairs and apply masks
        pass
    
    # Let me redo this cleanly
    energy = 0.0
    r_a = r_ij[r_ij < a_POT]
    if len(r_a) > 0:
        term1 = A_POT * (r_a ** (-m_POT) - B_POT) * np.exp(c_POT / (r_a - a_POT))
        energy += np.sum(term1)
    
    r_b = r_ij[r_ij < b_POT]
    if len(r_b) > 0:
        term2 = B_POT * np.exp(d_POT / (r_b - b_POT))
        energy += np.sum(term2)
    
    return float(energy)


def _dzugutov_energy_and_grad(positions):
    """Compute energy and gradient using finite differences or analytical if possible."""
    # Use the provided function for gradient
    energy, grad = dzugutov_energy_gradient(positions)
    return energy, grad


def _random_rotation(rng):
    """Generate a random rotation matrix."""
    # Use QR decomposition of a random matrix
    A = rng.normal(size=(3, 3))
    Q, R = np.linalg.qr(A)
    # Ensure proper rotation (det = 1)
    d = np.sign(np.linalg.det(Q))
    Q[:, -1] *= d
    return Q


def _generate_icosahedron(n_atoms, rng, spacing=1.0):
    """Generate an icosahedral cluster with n_atoms atoms.
    
    Icosahedron has 12 vertices. For larger clusters, add shells.
    """
    if n_atoms <= 12:
        # Standard icosahedron
        phi = (1 + np.sqrt(5)) / 2  # golden ratio
        # 12 vertices of icosahedron
        verts = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
        ], dtype=float)
        verts = verts / np.linalg.norm(verts[0]) * spacing  # normalize to unit radius then scale
        if n_atoms < 12:
            # Take first n_atoms vertices
            verts = verts[:n_atoms]
        return verts
    
    # For n > 12, create icosahedron + extra atoms placed randomly nearby
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    verts = verts / np.linalg.norm(verts[0]) * spacing
    
    # Add remaining atoms in a second shell
    extra = n_atoms - 12
    # Place extras at midpoints of edges or random positions on sphere
    # For simplicity, place them at random positions on a sphere of radius ~1.5*spacing
    extra_verts = rng.normal(size=(extra, 3))
    extra_verts = extra_verts / np.linalg.norm(extra_verts, axis=1, keepdims=True) * spacing * 1.5
    verts = np.vstack([verts, extra_verts])
    
    return verts


def _generate_decahedron(n_atoms, rng, spacing=1.0):
    """Generate a decahedral cluster (Mackay decahedron)."""
    # Mackay decahedron: 5-fold symmetry
    # Core: 1 atom at center, 12 at icosahedron vertices
    # For simplicity, use icosahedron-like structure
    return _generate_icosahedron(n_atoms, rng, spacing)


def _assemble_clusters(cluster1, cluster2, rng, min_sep=0.5):
    """Assemble two clusters by finding optimal relative position and orientation."""
    # Try multiple random orientations and positions
    best_energy = np.inf
    best_config = None
    
    n1, n2 = len(cluster1), len(cluster2)
    
    for trial in range(20):
        # Random rotation for cluster2
        R = _random_rotation(rng)
        rotated_c2 = cluster2 @ R.T
        
        # Random translation
        # Estimate separation based on cluster sizes
        sep = (len(cluster1) ** (1/3) + len(cluster2) ** (1/3)) * spacing_estimate * 1.2
        spacing_estimate = 1.1  # typical Dzugutov spacing
        
        # Place cluster2 offset from cluster1
        offset_dir = rng.normal(size=3)
        offset_dir /= np.linalg.norm(offset_dir)
        offset = offset_dir * sep
        
        # Combine
        combined = np.vstack([cluster1, rotated_c2 + offset])
        
        # Quick energy check
        try:
            e = _dzugutov_energy(combined)
            if e < best_energy:
                best_energy = e
                best_config = combined.copy()
        except:
            continue
    
    return best_config, best_energy


def _local_relax(positions, potential='dzugutov', parameter=None, max_iter=1000):
    """Relax positions using the provided relax function."""
    try:
        energy, relaxed = relax(positions, potential, parameter)
        return energy, relaxed
    except Exception as e:
        return _dzugutov_energy(positions), positions


def _basin_hopping_step(positions, rng, potential='dzugutov', parameter=None):
    """One step of basin hopping: perturb, relax, accept/reject."""
    # Perturb with random displacement
    displacement = rng.normal(0, 0.3, size=positions.shape)
    perturbed = positions + displacement
    
    # Relax
    e_pert, relaxed = _local_relax(perturbed, potential, parameter)
    
    return e_pert, relaxed


def construct(instance):
    """Construct low-energy Dzugutov cluster using hierarchical assembly and basin hopping."""
    global _DEADLINE
    _DEADLINE = time.time() + _TIME_BUDGET
    
    n = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    rng = np.random.default_rng(n * 7919)
    
    best_energy = np.inf
    best_positions = None
    
    # Strategy 1: Hierarchical assembly from icosahedral sub-clusters
    # Try different splits of n into sub-cluster sizes
    possible_splits = []
    for i in range(10, n - 9):
        possible_splits.append((i, n - i))
    
    # Sort by balance (prefer balanced splits)
    possible_splits.sort(key=lambda x: abs(x[0] - x[1]))
    
    # Try top few splits
    for split in possible_splits[:8]:
        if _check_deadline():
            break
        
        n1, n2 = split
        
        # Generate sub-clusters
        c1 = _generate_icosahedron(n1, rng, spacing=1.1)
        c2 = _generate_icosahedron(n2, rng, spacing=1.1)
        
        # Assemble
        assembled, e_asm = _assemble_clusters(c1, c2, rng)
        
        if assembled is not None:
            # Relax the assembled cluster
            e_rel, relaxed = _local_relax(assembled, potential, parameter)
            
            if e_rel < best_energy:
                best_energy = e_rel
                best_positions = relaxed.copy()
    
    # Strategy 2: Single icosahedral seed with basin hopping
    ico_seed = _generate_icosahedron(n, rng, spacing=1.1)
    e_ico, ico_relaxed = _local_relax(ico_seed, potential, parameter)
    
    if e_ico < best_energy:
        best_energy = e_ico
        best_positions = ico_relaxed.copy()
    
    # Basin hopping from best position
    current_energy = best_energy
    current_positions = best_positions.copy()
    
    bh_temp = 1.0
    for bh_iter in range(50):
        if _check_deadline():
            break
        
        e_new, new_positions = _basin_hopping_step(current_positions, rng, potential, parameter)
        
        # Metropolis acceptance
        delta_e = e_new - current_energy
        if delta_e < 0 or rng.random() < np.exp(-delta_e / bh_temp):
            current_energy = e_new
            current_positions = new_positions.copy()
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_positions = new_positions.copy()
        
        # Cool
        bh_temp *= 0.95
    
    # Strategy 3: Random compact clusters with relaxation (fallback)
    radius = 1.1 * n ** (1.0 / 3.0) * 0.75
    for _ in range(5):
        if _check_deadline():
            break
        
        points = []
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.8 * 1.1 for q in points):
                points.append(p)
        
        energy, positions = _local_relax(np.array(points), potential, parameter)
        if energy < best_energy:
            best_energy = energy
            best_positions = positions.copy()
    
    # Final check: ensure minimum separation
    if best_positions is not None:
        # Verify no atoms are too close
        for i in range(len(best_positions)):
            for j in range(i + 1, len(best_positions)):
                d = np.linalg.norm(best_positions[i] - best_positions[j])
                if d < 0.5:
                    # Push apart
                    mid = (best_positions[i] + best_positions[j]) / 2
                    diff = best_positions[i] - best_positions[j]
                    diff /= (np.linalg.norm(diff) + 1e-10)
                    push = (0.5 - d) / 2 + 0.01
                    best_positions[i] += diff * push
                    best_positions[j] -= diff * push
    
    if best_positions is None:
        # Fallback: simple random
        best_positions = rng.normal(0, 2, size=(n, 3))
    
    return best_positions
# EVOLVE-BLOCK-END
