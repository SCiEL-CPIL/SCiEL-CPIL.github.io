"""Morse cluster global minima search with structured initializers and basin hopping."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform

TOTAL_TIME_BUDGET = 290.0  # Leave 10s margin for safety
_start_time = time.time()


def _morse_energy(positions, rho):
    """Compute Morse potential energy for all pairs."""
    n = len(positions)
    if n < 2:
        return 0.0
    d = pdist(positions)
    # Only compute for pairs where distance > 0 to avoid issues
    mask = d > 1e-10
    if not np.any(mask):
        return 0.0
    r = d[mask]
    e = np.exp(rho * (1.0 - r))
    return np.sum(e * (e - 2.0))


def _morse_gradient(positions, rho):
    """Compute gradient of Morse potential energy (vectorized)."""
    n = len(positions)
    if n < 2:
        return np.zeros_like(positions)
    
    # Compute all pairwise differences
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r = np.linalg.norm(diff, axis=-1)
    
    # Avoid division by zero
    r_safe = np.where(r < 1e-10, 1e-10, r)
    
    # Morse term e = exp(rho * (1 - r))
    e = np.exp(rho * (1.0 - r_safe))
    
    # dE/dr = -rho * e * (2*e - 2)
    # Note: The force on i due to j is -dE/dr * (rij / r)
    # Gradient on i is sum over j of -dE/dr * (r_ij / r)
    # dE/dr * (1/r) * r_ij
    # Let's compute the scalar factor for each pair
    # de_dr = -rho * e * (2e - 2)
    de_dr = -rho * e * (2.0 * e - 2.0)
    
    # We need to sum de_dr * r_ij / r for all j != i
    # r_ij = positions[i] - positions[j]
    # The term is (de_dr / r) * r_ij
    # de_dr / r is a matrix of shape (n, n)
    # We set diagonal to 0
    factor = de_dr / r_safe
    np.fill_diagonal(factor, 0.0)
    
    # Gradient on i is sum_j factor[i,j] * (positions[i] - positions[j])
    # = sum_j factor[i,j] * positions[i] - sum_j factor[i,j] * positions[j]
    # = positions[i] * sum_j factor[i,j] - sum_j (factor[i,j] * positions[j])
    
    sum_factor = np.sum(factor, axis=1)
    
    # sum_j (factor[i,j] * positions[j]) is matrix multiplication factor @ positions
    sum_pos = factor @ positions
    
    grad = sum_factor[:, np.newaxis] * positions - sum_pos
    
    return grad


def _local_minimize(positions, rho, max_iter=200, time_limit=2.0):
    """Local minimization using L-BFGS-B."""
    n = len(positions)
    if n < 2:
        return positions, _morse_energy(positions, rho)
    
    start_t = time.time()
    
    def fun(x):
        pos = x.reshape(-1, 3)
        return _morse_energy(pos, rho)
    
    def jac(x):
        pos = x.reshape(-1, 3)
        return _morse_gradient(pos, rho).ravel()
    
    x0 = positions.ravel()
    # Use tighter convergence criteria and limit function evaluations
    result = minimize(fun, x0, method='L-BFGS-B', jac=jac,
                      options={'maxiter': max_iter, 'maxfun': 200, 'gtol': 1e-5})
    
    pos = result.x.reshape(-1, 3)
    energy = result.fun
    
    return pos, energy


def _generate_fcc_shells(n, spacing=1.0):
    """Generate n atoms in FCC-like shell structure (concentric shells)."""
    if n == 1:
        return np.array([[0.0, 0.0, 0.0]])
    
    # FCC shells are defined by integer coordinates (i,j,k) where i+j+k is even
    # or similar. Here we generate points in a cubic grid and filter by "shell" distance.
    # A more robust way: generate all FCC points within a radius, sort by distance, take n.
    
    max_coord = int(np.ceil(n ** (1.0/3.0))) + 5
    points = []
    
    # FCC lattice points: (i, j, k) where i,j,k are integers and i+j+k is even
    # Scaled by spacing/2? No, standard FCC spacing is a/sqrt(2) if a is cubic edge.
    # Here we want nearest neighbor distance = 1.
    # FCC nearest neighbor distance is a/sqrt(2). So a = sqrt(2).
    # Points are at (i*a, j*a, k*a) with i+j+k even.
    a = np.sqrt(2.0) * spacing
    
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                if (i + j + k) % 2 == 0:
                    points.append([i * a, j * a, k * a])
    
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    
    # Sort and take n
    idx = np.argsort(dists)[:n]
    points = points[idx]
    
    # Center
    points = points - np.mean(points, axis=0)
    
    return points


def _generate_hcp_shells(n, spacing=1.0):
    """Generate n atoms in HCP-like structure (concentric shells)."""
    if n == 1:
        return np.array([[0.0, 0.0, 0.0]])
    
    a = spacing
    c = a * np.sqrt(8.0 / 3.0)
    
    points = []
    max_r = int(np.ceil(n ** (1.0/3.0))) + 5
    
    for layer in range(-max_r, max_r + 1):
        z = layer * c
        for i in range(-max_r, max_r + 1):
            for j in range(-max_r, max_r + 1):
                x = (i + 0.5 * (j % 2)) * a
                y = j * a * np.sqrt(3.0) / 2.0
                if layer % 2 != 0:
                    x += a / 2.0
                    y += a * np.sqrt(3.0) / 6.0
                points.append([x, y, z])
    
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)[:n]
    points = points[idx]
    
    points = points - np.mean(points, axis=0)
    
    return points


def _generate_icosahedral(n, spacing=1.0):
    """Generate n atoms in Mackay icosahedral structure."""
    if n == 1:
        return np.array([[0.0, 0.0, 0.0]])
        
    # Mackay icosahedra are built from concentric shells of icosahedral symmetry.
    # Shell 1: 12 vertices (icosahedron)
    # Shell 2: 42 vertices (icosahedron + octahedron mix? No, specific Mackay shell)
    # Actually, Mackay shells are:
    # k=1: 12
    # k=2: 42 (total 55)
    # k=3: 92 (total 147)
    # k=4: 156 (total 303)
    
    # We will generate points on icosahedral and octahedral vertices scaled by shell radius.
    # For better packing, we use the "Mackay" rule:
    # Shell m has points at distance m * spacing.
    # The vertices of a regular icosahedron and octahedron are used.
    
    phi = (1 + np.sqrt(5)) / 2.0
    # Icosahedral vertices (normalized to radius 1)
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    ico_verts = ico_verts / np.linalg.norm(ico_verts[0])
    
    # Octahedral vertices
    oct_verts = np.array([
        [1, 0, 0], [-1, 0, 0],
        [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1]
    ])
    
    points = [np.zeros(3)] # Center
    
    # Generate shells
    for m in range(1, 10):
        r = m * spacing
        # Add icosahedral vertices
        for v in ico_verts:
            points.append(v * r)
        # Add octahedral vertices (for m > 1, these fill gaps)
        if m > 1:
            for v in oct_verts:
                points.append(v * r)
        
        if len(points) >= n:
            break
    
    points = np.array(points[:n])
    points = points - np.mean(points, axis=0)
    
    return points


def _generate_random_compact(n, rng, spacing=1.0):
    """Generate a random compact cluster."""
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    points = []
    for _ in range(n):
        for _attempt in range(50):
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                if all(np.linalg.norm(p - q) >= 0.5 * spacing for q in points):
                    points.append(p)
                    break
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    return np.array(points)


def _basin_hopping(positions, rho, n_iter=20, time_limit=5.0, step_size=0.3, rng=None):
    """Improved basin hopping algorithm."""
    if rng is None:
        rng = np.random.default_rng()
    
    best_pos = positions.copy()
    best_energy = _morse_energy(positions, rho)
    
    start_t = time.time()
    current_pos = positions.copy()
    current_energy = best_energy
    
    for i in range(n_iter):
        if time.time() - start_t > time_limit:
            break
        
        # Perturb with adaptive step size
        # Reduce step size if we've been successful
        if current_energy < best_energy + 0.1:
            current_step = step_size * 0.8
        else:
            current_step = step_size * 1.2
            
        perturbed = current_pos + rng.normal(0, current_step, current_pos.shape)
        
        # Local minimize
        new_pos, new_energy = _local_minimize(perturbed, rho, max_iter=100, time_limit=1.5)
        
        # Metropolis acceptance
        if new_energy < current_energy:
            current_pos = new_pos
            current_energy = new_energy
            if new_energy < best_energy:
                best_pos = new_pos.copy()
                best_energy = new_energy
        else:
            delta_e = new_energy - current_energy
            # Temperature-like parameter
            T = 0.1
            if delta_e < 1.0:
                accept_prob = np.exp(-delta_e / T)
                if rng.random() < accept_prob:
                    current_pos = new_pos
                    current_energy = new_energy
    
    return best_pos, best_energy


def construct(instance):
    """Find low-energy Morse cluster configurations."""
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind", 10))
    
    start_t = time.time()
    time_budget = 48.0  # ~48s per instance for 6 instances (288s total)
    
    rng = np.random.default_rng(n * 1000 + int(rho))
    
    # Generate diverse initial structures
    initializers = []
    
    # FCC shells
    try:
        fcc = _generate_fcc_shells(n, spacing=1.0)
        if len(fcc) == n:
            initializers.append(fcc)
    except:
        pass
    
    # HCP shells
    try:
        hcp = _generate_hcp_shells(n, spacing=1.0)
        if len(hcp) == n:
            initializers.append(hcp)
    except:
        pass
    
    # Icosahedral
    try:
        ico = _generate_icosahedral(n, spacing=1.0)
        if len(ico) == n:
            initializers.append(ico)
    except:
        pass
    
    # Random compact structures
    for _ in range(2):
        try:
            rand = _generate_random_compact(n, rng, spacing=1.0)
            initializers.append(rand)
        except:
            pass
    
    if not initializers:
        # Fallback: simple cubic
        side = int(np.ceil(n ** (1.0/3.0)))
        pts = []
        for i in range(side):
            for j in range(side):
                for k in range(side):
                    if len(pts) < n:
                        pts.append([i - side/2.0, j - side/2.0, k - side/2.0])
        while len(pts) < n:
            pts.append(rng.uniform(-side, side, 3))
        initializers.append(np.array(pts))
    
    # Local minimize each initializer
    best_pos = None
    best_energy = np.inf
    
    # Time allocation: 2s per initializer for local min
    for init in initializers:
        if time.time() - start_t > time_budget - 10: # Leave 10s for BH
            break
        
        try:
            pos, energy = _local_minimize(init, rho, max_iter=200, time_limit=2.0)
            if energy < best_energy:
                best_energy = energy
                best_pos = pos
        except:
            pass
    
    if best_pos is None:
        # Fallback
        best_pos = initializers[0]
        best_energy = _morse_energy(best_pos, rho)
    
    # Basin hopping from best local minimum
    if time.time() - start_t < time_budget - 5:
        try:
            remaining_time = time_budget - (time.time() - start_t) - 2
            bh_pos, bh_energy = _basin_hopping(
                best_pos, rho, n_iter=30, time_limit=remaining_time,
                step_size=0.3, rng=rng
            )
            if bh_energy < best_energy:
                best_pos = bh_pos
                best_energy = bh_energy
        except:
            pass
    
    # Final local polish
    if time.time() - start_t < time_budget - 1:
        try:
            final_pos, final_energy = _local_minimize(best_pos, rho, max_iter=500, time_limit=2.0)
            if final_energy < best_energy:
                best_pos = final_pos
                best_energy = final_energy
        except:
            pass
    
    return best_pos
# EVOLVE-BLOCK-END
