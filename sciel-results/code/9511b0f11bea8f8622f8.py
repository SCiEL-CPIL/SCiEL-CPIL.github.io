"""Lennard-Jones 55-atom cluster: Mackay icosahedron construction + L-BFGS relaxation."""
N = 55

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

# Deadline guard: evaluator kills at 150s; we aim to finish well before.
_START_TIME = time.time()
TIME_LIMIT = 130.0  # safety margin


def _lj_energy(pos):
    """Compute total Lennard-Jones energy: 4*sum(r^-12 - r^-6) over all pairs."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=-1)
    # Avoid division by zero (shouldn't happen if atoms are >= 0.5 apart)
    r2 = np.maximum(r2, 1e-12)
    np.fill_diagonal(r2, np.inf)
    r6 = r2 ** 3
    energy = 4.0 * np.sum(1.0 / (r6 ** 2) - 1.0 / r6)
    return energy


def _lj_energy_grad(pos):
    """Compute total Lennard-Jones energy and gradient."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=-1)
    r2 = np.maximum(r2, 1e-12)
    np.fill_diagonal(r2, np.inf)
    r = np.sqrt(r2)
    r6 = r2 ** 3
    r12 = r6 ** 2

    energy = 4.0 * np.sum(1.0 / r12 - 1.0 / r6)

    # Gradient: dE/dr_i = 4 * (-12/r^13 + 6/r^7) * (r_i / r) for each pair
    # Vectorized: for each atom i, sum over j != i of 4*(-12/r_ij^13 + 6/r_ij^7) * (pos_i - pos_j)/r_ij
    # = 4 * (-12/r_ij^13 + 6/r_ij^7) * diff[i,j] / r_ij
    # Note: diff[i,j] = pos[i] - pos[j], so force on i is in direction (pos_i - pos_j)
    coeff = 4.0 * (-12.0 / (r12 * r) + 6.0 / (r6 * r))
    # Set diagonal to 0
    np.fill_diagonal(coeff, 0.0)
    grad = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)

    return energy, grad


def _make_mackay_icosahedron():
    """
    Construct a complete two-shell Mackay icosahedron with 55 atoms.
    Shell structure: 1 (center) + 12 (inner) + 30 (middle) + 12 (outer) = 55
    
    The Mackay icosahedron has icosahedral symmetry.
    Inner shell: 12 vertices of a regular icosahedron at radius r1
    Middle shell: 30 vertices at radius r2
    Outer shell: 12 vertices at radius r3 (same directions as inner, further out)
    
    For LJ, the optimal radii are approximately:
    r1 ≈ 1.12 (nearest-neighbor distance)
    r2 ≈ 1.70 (second shell)
    r3 ≈ 2.05 (outer shell)
    """
    # Golden ratio
    phi = (1.0 + np.sqrt(5.0)) / 2.0

    # 12 vertices of a regular icosahedron (unit vectors)
    icos_dirs = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
    ], dtype=float)
    icos_dirs /= np.linalg.norm(icos_dirs, axis=1, keepdims=True)

    # Radii for the three shells (tuned for LJ)
    r1 = 1.12  # inner shell
    r2 = 1.70  # middle shell  
    r3 = 2.05  # outer shell

    # Shell 0: center (1 atom)
    shell0 = np.array([[0.0, 0.0, 0.0]])

    # Shell 1: 12 atoms at icosahedron vertices, radius r1
    shell1 = icos_dirs * r1

    # Shell 2: 30 atoms - these are at the midpoints of edges of the icosahedron
    # For a regular icosahedron, there are 30 edges.
    # The 30 directions for the middle shell in a Mackay icosahedron are the
    # normalized midpoints of the icosahedron edges.
    # Each vertex connects to 5 others, giving 12*5/2 = 30 edges.
    
    # Find all pairs of icosahedron vertices that form edges
    # (distance between unit vectors for adjacent vertices of icosahedron is 2/sqrt(phi^2+1) ≈ 1.0515)
    # Actually, let's just compute midpoints of all pairs and keep unique ones
    edge_mids = []
    for i in range(12):
        for j in range(i + 1, 12):
            d = icos_dirs[i] + icos_dirs[j]
            norm_d = np.linalg.norm(d)
            # Edge midpoint direction (normalized)
            if norm_d > 1e-10:
                edge_mids.append(d / norm_d)
    
    # Deduplicate: edge midpoints should come in pairs (i,j) and (j,i) give same direction
    # But we only computed i<j, so each edge appears once. We should have 30.
    edge_mids = np.array(edge_mids)
    # Verify we have 30
    assert len(edge_mids) == 30, f"Expected 30 edge midpoints, got {len(edge_mids)}"
    
    shell2 = edge_mids * r2

    # Shell 3: 12 atoms at icosahedron vertices, radius r3 (same directions as shell1)
    shell3 = icos_dirs * r3

    # Combine all shells
    positions = np.vstack([shell0, shell1, shell2, shell3])
    assert len(positions) == 55, f"Expected 55 atoms, got {len(positions)}"

    return positions


def _relax(pos, max_iter=200):
    """Relax positions using L-BFGS with gradient."""
    def fun(x):
        e, g = _lj_energy_grad(x.reshape(-1, 3))
        return e, g

    try:
        result = minimize(
            fun, pos.flatten(),
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        return result.x.reshape(-1, 3), result.fun
    except Exception:
        return pos, _lj_energy(pos)


def _make_perturbed_mackay(pos, scale, seed):
    """Create a perturbed version of the Mackay icosahedron."""
    rng = np.random.default_rng(seed)
    return pos + rng.standard_normal(pos.shape) * scale


def _make_cubic_lattice():
    """Simple cubic lattice fragment as fallback."""
    side = int(np.ceil(N ** (1 / 3)))
    points = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
    points.sort(key=lambda p: (p[0] - (side - 1) / 2) ** 2 + (p[1] - (side - 1) / 2) ** 2 + (p[2] - (side - 1) / 2) ** 2)
    return np.array(points[:N], dtype=float) * 1.12


def construct():
    """
    Construct 55-atom LJ cluster near the global minimum.
    Strategy: Start from Mackay icosahedron, relax with L-BFGS.
    Try multiple perturbations and pick the best.
    """
    global _START_TIME
    _START_TIME = time.time()

    # Base Mackay icosahedron
    mackay = _make_mackay_icosahedron()

    best_pos = mackay.copy()
    best_energy = _lj_energy(mackay)

    # First: relax the ideal Mackay structure
    elapsed = time.time() - _START_TIME
    if elapsed < TIME_LIMIT - 5:
        relaxed, e = _relax(mackay, max_iter=300)
        if e < best_energy:
            best_energy = e
            best_pos = relaxed

    # Try several perturbations to escape potential local minima
    perturb_scales = [0.01, 0.05, 0.1, 0.15, 0.2]
    seeds = [42, 123, 456, 789, 101112]

    for scale in perturb_scales:
        for seed in seeds:
            elapsed = time.time() - _START_TIME
            if elapsed > TIME_LIMIT - 10:
                break

            perturbed = _make_perturbed_mackay(mackay, scale, seed)
            # Ensure minimum distance constraint (0.5 apart)
            # Check and fix if needed
            diff = perturbed[:, np.newaxis, :] - perturbed[np.newaxis, :, :]
            r2 = np.sum(diff ** 2, axis=-1)
            np.fill_diagonal(r2, np.inf)
            min_r2 = np.min(r2)
            if min_r2 < 0.25:  # 0.5^2
                # Nudge apart
                perturbed += np.random.default_rng(seed + 1000).standard_normal(perturbed.shape) * 0.05

            relaxed, e = _relax(perturbed, max_iter=100)
            if e < best_energy:
                best_energy = e
                best_pos = relaxed

    # Final: one more relaxation on the best found
    elapsed = time.time() - _START_TIME
    if elapsed < TIME_LIMIT - 5:
        final_pos, final_e = _relax(best_pos, max_iter=200)
        if final_e < best_energy:
            best_energy = final_e
            best_pos = final_pos

    # Verify minimum distance constraint
    diff = best_pos[:, np.newaxis, :] - best_pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=-1)
    np.fill_diagonal(r2, np.inf)
    if np.min(r2) < 0.25:
        # Fallback to cubic lattice if constraint violated
        best_pos = _make_cubic_lattice()

    return best_pos
# EVOLVE-BLOCK-END
