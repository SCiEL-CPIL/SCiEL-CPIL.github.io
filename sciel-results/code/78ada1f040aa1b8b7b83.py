"""Improved program: construct the 55-atom Mackay icosahedron, pre-optimize shell radii,
then apply L-BFGS relaxation with a deadline guard.
Also attempts a symmetry-breaking liquid-quench path and keeps the best result."""
import numpy as np
import time
from scipy.optimize import minimize

N = 55

# Deadline guard: evaluator kills at 150s wall-clock; use 110s as safety margin
DEADLINE_SECONDS = 110.0

def _lj_energy(pos):
    """Compute total Lennard-Jones energy for N atoms (vectorized)."""
    # pos shape (n, 3)
    # Compute all pairwise displacements via broadcasting
    rij = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]  # (n, n, 3)
    r2 = np.sum(rij**2, axis=2)  # (n, n)
    np.fill_diagonal(r2, 1.0)  # avoid division by zero on diagonal
    r6 = r2**(-3)
    r12 = r6**2
    # Diagonal entries: r2=1 => r6=1, r12=1 => contribution 4*(1-1)=0 per diagonal element.
    # Each pair (i,j) with i!=j is counted twice (once as (i,j), once as (j,i)),
    # but the LJ potential sum over all pairs means we want each unordered pair once.
    # The standard LJ total energy is sum over i<j of 4*(r^-12 - r^-6).
    # Our double sum gives 2 * sum_{i<j} 4*(r^-12 - r^-6), so we need to divide by 2.
    energy = 2.0 * np.sum(r12 - r6)
    return energy

def _lj_energy_grad(pos):
    """Compute gradient of Lennard-Jones energy (vectorized).
    
    E = 2 * sum_{i,j} (r_ij^-12 - r_ij^-6) where the sum is over all ordered pairs
    with diagonal excluded. This equals sum_{i<j} 4*(r^-12 - r^-6).
    
    dE/dx_i = 2 * sum_{j != i} d/dx_i [r_ij^-12 - r_ij^-6]
    r_ij^2 = |x_i - x_j|^2
    d(r^-6)/d(x_i) = -6 * r^-7 * (x_i - x_j)/r = -6 * r^-8 * (x_i - x_j)
    d(r^-12)/d(x_i) = -12 * r^-13 * (x_i - x_j)/r = -12 * r^-14 * (x_i - x_j)
    
    So dE/dx_i = 2 * sum_{j!=i} [-12 * r^-14 + 6 * r^-8] * (x_i - x_j)
              = 2 * sum_{j!=i} [(-12 * r2^-7 + 6 * r2^-4)] * (x_i - x_j)
    """
    # pos shape (n, 3)
    rij = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]  # (n, n, 3)
    r2 = np.sum(rij**2, axis=2)  # (n, n)
    np.fill_diagonal(r2, 1.0)
    # dE/dr2 for each pair in the double-sum formulation:
    # E = 2 * sum (r2^-6 - r2^-12) ... wait, let me re-derive.
    # E = 2 * sum_{i!=j} (r_ij^-12 - r_ij^-6)
    # Let f(r2) = r2^-6 - r2^-12 ... no.
    # r^-12 = (r2)^(1/2 * -12) = r2^-6
    # r^-6 = (r2)^(1/2 * -6) = r2^-3
    # So E = 2 * sum_{i!=j} (r2_ij^-6 - r2_ij^-3)
    # dE/dx_i = 2 * sum_{j!=i} d/dx_i (r2^-6 - r2^-3)
    # d(r2^-6)/d(r2) = -6 * r2^-7
    # d(r2^-3)/d(r2) = -3 * r2^-4
    # d(r2)/dx_i = 2*(x_i - x_j)
    # dE/dx_i = 2 * sum_{j!=i} (-6*r2^-7 + 3*r2^-4) * 2*(x_i - x_j)
    #         = 4 * sum_{j!=i} (-6*r2^-7 + 3*r2^-4) * (x_i - x_j)
    #         = 4 * sum_{j!=i} (3*r2^-4 - 6*r2^-7) * (x_i - x_j)
    coeff = 4.0 * (3.0 * r2**(-4) - 6.0 * r2**(-7))  # (n, n)
    np.fill_diagonal(coeff, 0.0)
    # grad_i = sum_j coeff[i,j] * rij[i,j]
    grad = np.sum(coeff[:, :, np.newaxis] * rij, axis=1)  # (n, 3)
    return grad

def _get_mackay_directions():
    """Precompute the 55 unit direction vectors for the Mackay icosahedron geometry.
    Returns (verts_unit, edge_midpoints_unit) where:
      verts_unit: (12, 3) unit vectors for icosahedron vertices
      edge_midpoints_unit: (30, 3) unit vectors for edge midpoints
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    
    # 12 vertices of regular icosahedron
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    
    # Normalize to unit radius
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts_unit = verts / norms
    
    # Find edges: pairs of vertices at minimum distance
    # Vectorized distance computation
    rij = verts_unit[:, np.newaxis, :] - verts_unit[np.newaxis, :, :]
    dists = np.sqrt(np.sum(rij**2, axis=2))
    
    # Minimum edge length (exclude diagonal)
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    edge_mask = dists < min_dist * 1.001
    
    # Get edge midpoints (30 edges)
    edge_midpoints = []
    for i in range(12):
        for j in range(i+1, 12):
            if edge_mask[i, j]:
                mid = (verts_unit[i] + verts_unit[j]) / 2.0
                mid_unit = mid / np.linalg.norm(mid)
                edge_midpoints.append(mid_unit)
    
    edge_midpoints = np.array(edge_midpoints)  # shape (30, 3)
    return verts_unit, edge_midpoints

# Precompute directions once
_VERTS_UNIT, _EDGE_MIDS_UNIT = _get_mackay_directions()

def _mackay_icosahedron_55(r_inner, r_edge_mid, r_vertex):
    """
    Construct 55-atom Mackay icosahedron with three independent radii.
    - 1 atom at center
    - 12 atoms at inner shell radius r_inner (vertices of regular icosahedron)
    - 30 atoms at outer shell radius r_edge_mid (edge-midpoint directions)
    - 12 atoms at outer shell radius r_vertex (vertex directions)
    """
    points = [np.zeros(3)]  # center atom
    
    # Inner shell: 12 vertices at radius r_inner
    for v in _VERTS_UNIT:
        points.append(r_inner * v)
    
    # Outer shell: 30 edge-midpoint directions at radius r_edge_mid
    for v in _EDGE_MIDS_UNIT:
        points.append(r_edge_mid * v)
    
    # Outer shell: 12 vertex directions at radius r_vertex
    for v in _VERTS_UNIT:
        points.append(r_vertex * v)
    
    return np.array(points, dtype=float)

def _optimize_radii():
    """Optimize the three shell radii to minimize LJ energy."""
    def energy_from_radii(params):
        r_inner, r_edge_mid, r_vertex = params
        if r_inner < 0.3 or r_edge_mid < 0.5 or r_vertex < 0.5:
            return 1e10
        if r_edge_mid < r_inner * 0.9 or r_vertex < r_inner * 0.9:
            return 1e10
        pos = _mackay_icosahedron_55(r_inner, r_edge_mid, r_vertex)
        return _lj_energy(pos)
    
    # Initial guess based on known Mackay geometry
    r0 = [1.054, 1.755, 2.000]
    result = minimize(energy_from_radii, r0, method='Nelder-Mead', 
                      options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12})
    return result.x, result.fun

def _liquid_quench_init(rng):
    """Generate a disordered liquid-like initial configuration of 55 atoms.
    Atoms are placed randomly within a sphere of radius ~3.0, with minimum
    separation enforced by rejection sampling."""
    pos = np.zeros((N, 3))
    pos[0] = [0.0, 0.0, 0.0]  # center atom
    min_dist = 0.51
    max_radius = 3.2
    max_attempts = 20000
    placed = 1
    attempt = 0
    while placed < N and attempt < max_attempts:
        # Random point in sphere
        r = max_radius * (rng.random() ** (1.0 / 3.0))
        theta = rng.random() * 2.0 * np.pi
        phi = np.arccos(2.0 * rng.random() - 1.0)
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        candidate = np.array([x, y, z])
        # Check minimum separation from all placed atoms
        diffs = pos[:placed] - candidate
        dists = np.sqrt(np.sum(diffs**2, axis=1))
        if np.min(dists) >= min_dist:
            pos[placed] = candidate
            placed += 1
        attempt += 1
    # If we couldn't place all atoms (unlikely), fill remaining with small random offsets
    if placed < N:
        for i in range(placed, N):
            pos[i] = pos[placed - 1] + 0.52 * rng.random(3) - 0.26
    return pos


def _quench_schedule(pos0, energy_func, grad_func, time_budget):
    """Rapid deterministic cooling: multiple L-BFGS passes with decreasing
    step-size bounds and increasing tolerance strictness."""
    current = pos0.flatten().copy()
    best_energy = energy_func(current)
    
    # Schedule: (max_step, gtol, ftol, maxiter_fraction)
    schedule = [
        (1.0, 1e-4, 1e-8, 0.4),   # fast collapse
        (0.5, 1e-6, 1e-10, 0.3),  # medium refinement
        (0.2, 1e-8, 1e-12, 0.2),  # fine refinement
        (0.05, 1e-10, 1e-14, 0.1),# final polish
    ]
    
    for max_step, gtol, ftol, iter_frac in schedule:
        elapsed = time.time() - _quench_schedule._start_time
        remaining = time_budget - elapsed
        if remaining < 2.0:
            break
        maxiter = max(10, int(iter_frac * 3000))
        try:
            result = minimize(
                energy_func,
                current,
                jac=grad_func,
                method='L-BFGS-B',
                options={'maxiter': maxiter, 'gtol': gtol, 'ftol': ftol,
                         'maxls': 30}
            )
            if result.fun < best_energy:
                best_energy = result.fun
                current = result.x.copy()
        except Exception:
            break
    
    return current.reshape(-1, 3), best_energy


def construct():
    start_time = time.time()
    _quench_schedule._start_time = start_time
    
    # Step 1: Optimize shell radii
    try:
        radii, radius_energy = _optimize_radii()
    except Exception:
        radii = [1.054, 1.755, 2.000]
        radius_energy = _lj_energy(_mackay_icosahedron_55(1.054, 1.755, 2.000))
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE_SECONDS - 15:
        # Not enough time for relaxation; return the radius-optimized structure
        pos = _mackay_icosahedron_55(radii[0], radii[1], radii[2])
        return pos
    
    # Step 2: Apply L-BFGS relaxation to Mackay structure
    pos0 = _mackay_icosahedron_55(radii[0], radii[1], radii[2])
    
    def energy_func(x):
        return _lj_energy(x.reshape(-1, 3))
    
    def grad_func(x):
        return _lj_energy_grad(x.reshape(-1, 3)).flatten()
    
    try:
        result = minimize(
            energy_func, 
            pos0.flatten(), 
            jac=grad_func,
            method='L-BFGS-B',
            options={'maxiter': 5000, 'gtol': 1e-10, 'ftol': 1e-14}
        )
        pos_relaxed = result.x.reshape(-1, 3)
        final_energy = result.fun
    except Exception:
        pos_relaxed = pos0
        final_energy = radius_energy
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > DEADLINE_SECONDS - 10:
        return pos_relaxed
    
    # Step 3: High-precision polish pass with tighter tolerances
    best_pos = pos_relaxed
    best_energy = final_energy
    
    remaining = DEADLINE_SECONDS - elapsed - 5
    if remaining > 5:
        try:
            res2 = minimize(
                energy_func,
                pos_relaxed.flatten(),
                jac=grad_func,
                method='L-BFGS-B',
                options={'maxiter': 5000, 'gtol': 1e-12, 'ftol': 1e-16}
            )
            e2 = res2.fun
            if e2 < best_energy:
                best_energy = e2
                best_pos = res2.x.reshape(-1, 3)
        except Exception:
            pass
    
    # Step 4: Liquid quench path (symmetry-breaking)
    elapsed = time.time() - start_time
    remaining = DEADLINE_SECONDS - elapsed - 3
    if remaining > 8:
        try:
            rng = np.random.default_rng(42)
            liq_init = _liquid_quench_init(rng)
            # Verify minimum separation
            diffs = liq_init[:, np.newaxis, :] - liq_init[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diffs**2, axis=2))
            np.fill_diagonal(dists, np.inf)
            if np.min(dists) >= 0.5:
                liq_pos, liq_energy = _quench_schedule(liq_init, energy_func, grad_func, remaining)
                if liq_energy < best_energy:
                    best_energy = liq_energy
                    best_pos = liq_pos
        except Exception:
            pass
    
    return best_pos
