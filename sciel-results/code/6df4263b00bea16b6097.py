"""Sutton-Chen transition-metal cluster global minima - shrink-wrap annealing strategy."""
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient


def _soft_core_energy_gradient(positions, kind, alpha):
    """
    Compute energy and gradient of a soft-core modified Sutton-Chen potential.
    The soft-core modification replaces r_ij with sqrt(r_ij^2 + alpha^2) in the
    repulsive term to avoid infinite forces at close contact, allowing smooth
    relaxation from overlapping configurations.
    """
    n_exp, m_exp, c = _get_params(kind)
    n_atoms = len(positions)
    energy = 0.0
    gradient = np.zeros_like(positions)
    
    for i in range(n_atoms):
        pair_energy = 0.0
        pair_grad = np.zeros(3)
        for j in range(i + 1, n_atoms):
            rij = positions[i] - positions[j]
            r2 = np.dot(rij, rij) + alpha * alpha  # soft-core
            r = np.sqrt(r2)
            
            # Repulsive term: 1/2 * r^(-n_exp)
            rep = 0.5 * r2 ** (-n_exp / 2.0)
            # d(rep)/d(rij) = 0.5 * (-n_exp) * r2^(-n_exp/2 - 1) * rij
            drep = -0.5 * n_exp * r2 ** (-n_exp / 2.0 - 1.0) * rij
            
            # Attractive term: -c * sqrt(sum r^(-m_exp))
            # For pair contribution to the sum: r^(-m_exp)
            att_inner = r2 ** (-m_exp / 2.0)
            # d(att_inner)/d(rij) = (-m_exp/2) * r2^(-m_exp/2 - 1) * rij
            datt_inner = (-m_exp / 2.0) * r2 ** (-m_exp / 2.0 - 1.0) * rij
            
            # Accumulate for pair (i,j)
            pair_energy += rep
            pair_grad += drep
            pair_att_inner += att_inner
            pair_datt_inner += datt_inner
        
        # Now compute the full attractive contribution for atom i
        # E_att_i = -c * sqrt(S_i) where S_i = sum_{j!=i} r_ij^(-m_exp)
        # But we computed pairwise, so we need to be careful.
        # Actually, let's restructure: compute all pair distances first.
    
    # Let me rewrite more carefully
    energy = 0.0
    gradient = np.zeros_like(positions)
    
    # Precompute all pair distances
    S = np.zeros(n_atoms)  # S_i = sum_{j!=i} r_ij^(-m_exp)
    pair_reps = np.zeros((n_atoms, n_atoms))
    
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            rij = positions[i] - positions[j]
            r2 = np.dot(rij, rij) + alpha * alpha
            r2_inv_m2 = r2 ** (-m_exp / 2.0)
            S[i] += r2_inv_m2
            S[j] += r2_inv_m2
            # Repulsive energy contribution for pair (i,j): 0.5 * r^(-n_exp)
            pair_reps[i, j] = 0.5 * r2 ** (-n_exp / 2.0)
    
    # Total energy
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            energy += pair_reps[i, j]
        energy -= c * np.sqrt(S[i]) if S[i] > 0 else 0.0
    
    # Gradient
    for i in range(n_atoms):
        grad_i = np.zeros(3)
        sqrt_S_i = np.sqrt(S[i]) if S[i] > 0 else 1.0
        for j in range(i + 1, n_atoms):
            rij = positions[i] - positions[j]
            r2 = np.dot(rij, rij) + alpha * alpha
            
            # Repulsive gradient
            drep = -0.5 * n_exp * r2 ** (-n_exp / 2.0 - 1.0) * rij
            
            # Attractive gradient: d/dx_i [-c * sqrt(S_i)] = -c * (1/(2*sqrt(S_i))) * dS_i/dx_i
            # dS_i/dx_i = sum_{k!=i} d(r_ik^(-m_exp))/dx_i = sum_{k!=i} (-m_exp/2) * r_ik^(-m_exp-2) * (x_i - x_k)
            # But we only have pair (i,j) here, so we need the full sum.
            # This is getting complex. Let me use a different approach.
        
        # Compute dS_i/dx_i = sum_{j!=i} (-m_exp/2) * r_ij^(-m_exp-2) * (x_i - x_j)
        dS_i_dx_i = np.zeros(3)
        for j in range(n_atoms):
            if i == j:
                continue
            rij = positions[i] - positions[j]
            r2 = np.dot(rij, rij) + alpha * alpha
            dS_i_dx_i += (-m_exp / 2.0) * r2 ** (-m_exp / 2.0 - 1.0) * rij
        
        # Gradient of attractive term for atom i
        if S[i] > 0:
            grad_i += -c * (1.0 / (2.0 * np.sqrt(S[i]))) * dS_i_dx_i
        
        # Gradient of repulsive terms for atom i
        for j in range(n_atoms):
            if i == j:
                continue
            if j > i:
                rij = positions[i] - positions[j]
                r2 = np.dot(rij, rij) + alpha * alpha
                drep = -0.5 * n_exp * r2 ** (-n_exp / 2.0 - 1.0) * rij
                grad_i += drep
            else:
                # j < i, we already added this in the loop above? No, let's be consistent.
                # Actually, let's just sum over all j != i for repulsive too.
                rij = positions[i] - positions[j]
                r2 = np.dot(rij, rij) + alpha * alpha
                drep = -0.5 * n_exp * r2 ** (-n_exp / 2.0 - 1.0) * rij
                grad_i += drep
        
        gradient[i] = grad_i
    
    return energy, gradient


def _get_params(kind):
    """Get Sutton-Chen parameters for the given kind."""
    params = {
        '12-6': (12, 6, 144.41),
        '9-6': (9, 6, 39.432),
        '10-8': (10, 8, 34.408),
    }
    return params[kind]


def _soft_core_objective(x, kind, alpha):
    """Objective function for scipy minimize with soft-core potential."""
    positions = x.reshape(-1, 3)
    energy, grad = _soft_core_energy_gradient(positions, kind, alpha)
    return energy, grad.flatten()


def _shrink_wrap_anneal(n, kind, rng, deadline, time_per_instance):
    """
    Shrink-wrap annealing: start with tightly packed overlapping sphere,
    gradually reduce soft-core alpha to let atoms relax to proper distances.
    """
    n_exp, m_exp, c = _get_params(kind)
    
    # Initial radius: tightly packed, overlapping
    # For n atoms in a sphere, if packed at equilibrium distance ~1.0,
    # radius ~ n^(1/3). For overlapping, use smaller radius.
    base_radius = n ** (1.0 / 3.0) * 0.5  # tight packing
    
    # Generate initial positions in a small sphere
    positions = rng.uniform(-base_radius, base_radius, (n, 3))
    
    best_energy = np.inf
    best_positions = positions.copy()
    
    # Annealing schedule: start with large alpha, reduce to 0
    n_steps = 10
    alphas = np.linspace(base_radius * 0.5, 0.01, n_steps)
    
    current_positions = positions.copy()
    
    for step, alpha in enumerate(alphas):
        if time.time() > deadline:
            break
        
        # Define objective with current alpha
        def obj(x, a=alpha):
            pos = x.reshape(-1, 3)
            e, g = _soft_core_energy_gradient(pos, kind, a)
            return e, g.flatten()
        
        # Relax using L-BFGS-B
        try:
            result = minimize(
                obj, current_positions.flatten(),
                jac=True, method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if result.success or result.fun < best_energy:
                current_positions = result.x.reshape(-1, 3)
                # Check for validity
                if np.all(np.isfinite(current_positions)):
                    # Check minimum distance
                    min_dist = np.inf
                    for i in range(n):
                        for j in range(i + 1, n):
                            d = np.linalg.norm(current_positions[i] - current_positions[j])
                            min_dist = min(min_dist, d)
                    if min_dist > 0.35:
                        best_energy = result.fun
                        best_positions = current_positions.copy()
        except Exception:
            continue
    
    return best_positions, best_energy


def _fcc_positions(n):
    """Generate positions for a truncated FCC cluster with n atoms."""
    # FCC lattice: atoms at (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0) + integer shifts
    # Nearest neighbor distance = sqrt(0.5) * a, so for r_nn ~ 1.0, a = sqrt(2)
    a = np.sqrt(2.0)
    
    positions = []
    # Generate candidate FCC points within a sphere
    max_r = (n / 4.0) ** (1.0 / 3.0) * a + 1.0
    n_max = int(np.ceil(max_r / a))
    
    for ix in range(-n_max, n_max + 1):
        for iy in range(-n_max, n_max + 1):
            for iz in range(-n_max, n_max + 1):
                # FCC basis
                for bx, by, bz in [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]:
                    x = (ix + bx) * a
                    y = (iy + by) * a
                    z = (iz + bz) * a
                    if np.sqrt(x*x + y*y + z*z) <= max_r:
                        positions.append([x, y, z])
    
    positions = np.array(positions)
    
    # If we have more than n, take the n closest to origin
    if len(positions) > n:
        dists = np.linalg.norm(positions, axis=1)
        idx = np.argsort(dists)[:n]
        positions = positions[idx]
    elif len(positions) < n:
        # Add random points to fill
        while len(positions) < n:
            p = np.array([0.0, 0.0, 0.0]) + np.random.uniform(-0.5, 0.5, 3)
            positions = np.vstack([positions, p.reshape(1, -1)])
    
    return positions[:n]


def _hcp_positions(n):
    """Generate positions for a truncated HCP cluster with n atoms."""
    # HCP: two interpenetrating FCC-like lattices
    a = np.sqrt(2.0)  # nearest neighbor ~ 1.0
    c = a * np.sqrt(8.0 / 3.0)  # ideal HCP c/a ratio
    
    positions = []
    max_r = (n / 4.0) ** (1.0 / 3.0) * a + 1.0
    n_max = int(np.ceil(max_r / a))
    
    for ix in range(-n_max, n_max + 1):
        for iy in range(-n_max, n_max + 1):
            for iz in range(-n_max, n_max + 1):
                # A-layer
                x = ix * a
                y = iy * a
                z = iz * c
                if np.sqrt(x*x + y*y + z*z) <= max_r:
                    positions.append([x, y, z])
                # B-layer (shifted)
                x = (ix + 1.0/3.0) * a
                y = (iy + np.sqrt(3.0)/3.0) * a
                z = (iz + 1.0/3.0) * c
                if np.sqrt(x*x + y*y + z*z) <= max_r:
                    positions.append([x, y, z])
                # C-layer (shifted)
                x = (ix + 2.0/3.0) * a
                y = (iy - np.sqrt(3.0)/3.0) * a
                z = (iz + 2.0/3.0) * c
                if np.sqrt(x*x + y*y + z*z) <= max_r:
                    positions.append([x, y, z])
    
    positions = np.array(positions)
    
    if len(positions) > n:
        dists = np.linalg.norm(positions, axis=1)
        idx = np.argsort(dists)[:n]
        positions = positions[idx]
    elif len(positions) < n:
        while len(positions) < n:
            p = np.array([0.0, 0.0, 0.0]) + np.random.uniform(-0.5, 0.5, 3)
            positions = np.vstack([positions, p.reshape(1, -1)])
    
    return positions[:n]


def _icosahedral_positions(n):
    """Generate positions for an icosahedral Mackay cluster with n atoms."""
    # Mackay icosahedral clusters: 1, 12, 42, 120, 230, ...
    # Shell radii: 1, sqrt(3), sqrt(5), ...
    # For general n, use nested icosahedral shells
    
    positions = []
    shell_radii = [0.0, 1.0, np.sqrt(3.0), np.sqrt(5.0), 3.0, np.sqrt(9.0)]
    shell_counts = [1, 12, 42, 120, 230, 420]
    
    # Icosahedral vertices (for shell 1)
    phi = (1 + np.sqrt(5)) / 2
    ico_vertices = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
    ])
    ico_vertices /= np.linalg.norm(ico_vertices[0])  # normalize to unit sphere
    
    count = 0
    for shell_idx, (radius, max_count) in enumerate(zip(shell_radii, shell_counts)):
        if count >= n:
            break
        if shell_idx == 0:
            positions.append([0.0, 0.0, 0.0])
            count += 1
        else:
            # Distribute points on icosahedral vertices and edges
            # For simplicity, use the 12 vertices and add more if needed
            n_needed = min(max_count, n - count)
            if n_needed <= 12:
                for v in ico_vertices[:n_needed]:
                    positions.append(radius * v)
                    count += 1
            else:
                # Add vertices
                for v in ico_vertices:
                    if count >= n:
                        break
                    positions.append(radius * v)
                    count += 1
                # Add edge midpoints if needed
                if count < n:
                    for i in range(12):
                        for j in range(i+1, 12):
                            if count >= n:
                                break
                            mid = (ico_vertices[i] + ico_vertices[j]) / 2.0
                            mid /= np.linalg.norm(mid)
                            positions.append(radius * mid)
                            count += 1
                        if count >= n:
                            break
    
    # If still not enough, add random points
    while len(positions) < n:
        p = rng_global.uniform(-1, 1, 3)
        p /= np.linalg.norm(p) if np.linalg.norm(p) > 0 else 1.0
        positions.append((n ** (1/3)) * 0.5 * p)
    
    positions = np.array(positions[:n])
    return positions


rng_global = np.random.default_rng(42)


def construct(instance):
    """Shrink-wrap annealing strategy for Sutton-Chen cluster global minima."""
    global rng_global
    n = instance["n"]
    kind = instance["kind"]
    potential = instance["potential"]
    
    # Time budget: 330s total for all 6 instances, so ~55s per instance
    # But we need a safety margin. Use 50s per instance.
    start_time = time.time()
    deadline = start_time + 50.0  # 50s per instance, 6*50=300s
    
    rng = np.random.default_rng(n * 1000 + hash(kind) % 10000)
    
    best_positions = None
    best_energy = np.inf
    
    # Strategy 1: Shrink-wrap annealing from random tight sphere
    if time.time() < deadline - 10:
        pos, e = _shrink_wrap_anneal(n, kind, rng, deadline - 15, 15)
        if np.all(np.isfinite(pos)) and e < best_energy:
            # Validate
            min_dist = np.inf
            for i in range(n):
                for j in range(i + 1, n):
                    d = np.linalg.norm(pos[i] - pos[j])
                    min_dist = min(min_dist, d)
            if min_dist > 0.35:
                best_positions = pos
                best_energy = e
    
    # Strategy 2: FCC-based starting point
    if time.time() < deadline - 10:
        try:
            fcc_pos = _fcc_positions(n)
            # Relax with standard Sutton-Chen
            energy, positions = relax(fcc_pos, potential, kind)
            if np.all(np.isfinite(positions)) and energy < best_energy:
                min_dist = np.inf
                for i in range(n):
                    for j in range(i + 1, n):
                        d = np.linalg.norm(positions[i] - positions[j])
                        min_dist = min(min_dist, d)
                if min_dist > 0.35:
                    best_positions = positions
                    best_energy = energy
        except Exception:
            pass
    
    # Strategy 3: HCP-based starting point
    if time.time() < deadline - 10:
        try:
            hcp_pos = _hcp_positions(n)
            energy, positions = relax(hcp_pos, potential, kind)
            if np.all(np.isfinite(positions)) and energy < best_energy:
                min_dist = np.inf
                for i in range(n):
                    for j in range(i + 1, n):
                        d = np.linalg.norm(positions[i] - positions[j])
                        min_dist = min(min_dist, d)
                if min_dist > 0.35:
                    best_positions = positions
                    best_energy = energy
        except Exception:
            pass
    
    # Strategy 4: Icosahedral starting point
    if time.time() < deadline - 10:
        try:
            ico_pos = _icosahedral_positions(n)
            energy, positions = relax(ico_pos, potential, kind)
            if np.all(np.isfinite(positions)) and energy < best_energy:
                min_dist = np.inf
                for i in range(n):
                    for j in range(i + 1, n):
                        d = np.linalg.norm(positions[i] - positions[j])
                        min_dist = min(min_dist, d)
                if min_dist > 0.35:
                    best_positions = positions
                    best_energy = energy
        except Exception:
            pass
    
    # Strategy 5: Multiple random restarts with standard relax
    n_restarts = 4
    for r in range(n_restarts):
        if time.time() > deadline - 5:
            break
        try:
            radius = n ** (1.0 / 3.0) * 0.75
            points = []
            while len(points) < n:
                p = rng.uniform(-radius, radius, 3)
                if np.linalg.norm(p) <= radius and all(
                    np.linalg.norm(p - q) >= 0.7 for q in points
                ):
                    points.append(p)
            energy, positions = relax(np.array(points), potential, kind)
            if np.all(np.isfinite(positions)) and energy < best_energy:
                min_dist = np.inf
                for i in range(n):
                    for j in range(i + 1, n):
                        d = np.linalg.norm(positions[i] - positions[j])
                        min_dist = min(min_dist, d)
                if min_dist > 0.35:
                    best_positions = positions
                    best_energy = energy
        except Exception:
            continue
    
    # Final polish: if we have a good candidate, do one more high-precision relax
    if best_positions is not None and time.time() < deadline - 3:
        try:
            energy, positions = relax(best_positions, potential, kind)
            if np.all(np.isfinite(positions)):
                min_dist = np.inf
                for i in range(n):
                    for j in range(i + 1, n):
                        d = np.linalg.norm(positions[i] - positions[j])
                        min_dist = min(min_dist, d)
                if min_dist > 0.35:
                    best_positions = positions
                    best_energy = energy
        except Exception:
            pass
    
    # Fallback: if all else failed, return a valid configuration
    if best_positions is None:
        # Simple spherical shell arrangement
        positions = np.zeros((n, 3))
        if n == 1:
            positions[0] = [0, 0, 0]
        else:
            # Place atoms on a sphere
            radius = n ** (1.0 / 3.0) * 0.5
            for i in range(n):
                # Fibonacci sphere distribution
                phi = np.arccos(1 - 2 * (i + 0.5) / n)
                theta = np.pi * (1 + 5 ** 0.5) * i
                positions[i] = radius * np.array([
                    np.sin(phi) * np.cos(theta),
                    np.sin(phi) * np.sin(theta),
                    np.cos(phi)
                ])
        best_positions = positions
    
    return best_positions
