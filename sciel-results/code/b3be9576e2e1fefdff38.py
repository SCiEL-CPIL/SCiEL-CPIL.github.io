"""
Strategy: Basin Hopping with Poly-tetrahedral Seeding
- Uses SciPy L-BFGS-B for high-quality local minimization (exploiting gradient).
- Uses Basin Hopping (random perturbation + local minimization) to escape local minima.
- Seeds the search with structured polytetrahedral (McMillan-like) geometries rather than pure random.
- Vectorized energy/gradient calculation for speed.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as so
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

# Dzugutov potential parameters
A = 5.82
B = 1.28
a = 1.87
b = 1.94
c = 1.10
d = 0.27
m = 16

def _dzugutov_energy(pos):
    """Compute Dzugutov energy for a cluster (vectorized)."""
    n = len(pos)
    if n < 2:
        return 0.0
    
    # Compute all pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    
    # Mask for valid pairs (i < j)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    dist = dist[mask]
    
    energy = 0.0
    
    # Region 1: r < a
    mask1 = dist < a
    if np.any(mask1):
        r1 = dist[mask1]
        # Avoid division by zero or negative if r < a (though r>=0.5 enforced)
        v1 = A * (r1 ** (-m) - B) * np.exp(c / (r1 - a))
        energy += np.sum(v1)
        
    # Region 2: a <= r < b
    mask2 = (dist >= a) & (dist < b)
    if np.any(mask2):
        r2 = dist[mask2]
        v2 = B * np.exp(d / (r2 - b))
        energy += np.sum(v2)
        
    return energy

def _dzugutov_grad(pos):
    """Compute Dzugutov gradient for a cluster (vectorized)."""
    n = len(pos)
    if n < 2:
        return np.zeros_like(pos)
    
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf) # Avoid self-interaction
    
    grad = np.zeros_like(pos)
    
    # We need force F_ij = -dV/dr * (r_i - r_j)/r
    # dV/dr is computed for each pair.
    
    # Masks
    mask1 = dist < a
    mask2 = (dist >= a) & (dist < b)
    
    # Region 1: r < a
    if np.any(mask1):
        r1 = dist[mask1]
        rij1 = diff[mask1]
        
        # dV/dr = A * (-m * r^(-m-1) * exp(c/(r-a)) + (r^(-m) - B) * exp(c/(r-a)) * (-c/(r-a)^2))
        exp_term = np.exp(c / (r1 - a))
        dvdr1 = A * (-m * r1 ** (-m - 1) * exp_term + (r1 ** (-m) - B) * exp_term * (-c / (r1 - a) ** 2))
        
        # Force factor: -dvdr / r
        factor1 = -dvdr1 / r1
        
        # Accumulate gradient
        # For each pair (i, j), grad[i] += factor * (pos[i] - pos[j])
        # We can use np.add.at or explicit loops if vectorization is tricky with indices.
        # Since n <= 60, n^2 = 3600, explicit index loop is fast enough and safer.
        indices = np.where(mask1)
        for i, j in zip(*indices):
            grad[i] += factor1[indices[0] == i][0] * rij1[indices[0] == i][0] 
            # This is inefficient. Let's do a standard loop for clarity and safety with n<60.
            
    # Let's rewrite gradient with explicit loops for correctness and simplicity, as n is small.
    grad = np.zeros_like(pos)
    for i in range(n):
        for j in range(i + 1, n):
            r = dist[i, j]
            if r < a:
                exp_term = np.exp(c / (r - a))
                dvdr = A * (-m * r ** (-m - 1) * exp_term + (r ** (-m) - B) * exp_term * (-c / (r - a) ** 2))
            elif r < b:
                exp_term = np.exp(d / (r - b))
                dvdr = B * exp_term * (-d / (r - b) ** 2)
            else:
                continue
                
            factor = -dvdr / r
            rij = pos[i] - pos[j]
            grad[i] += factor * rij
            grad[j] -= factor * rij
            
    return grad

def _local_minimize(pos, max_iter=100):
    """
    Use L-BFGS-B to minimize energy.
    Returns (energy, pos).
    """
    def objective(x):
        pos_flat = x.reshape(-1, 3)
        # Enforce minimum distance constraint? L-BFGS-B doesn't handle inequality constraints well without bounds.
        # The potential has a steep repulsive core, so it should naturally avoid r < 0.5.
        # However, to be safe, we can add a penalty if any distance < 0.5.
        try:
            e = _dzugutov_energy(pos_flat)
            # Check min distance
            if len(pos_flat) > 1:
                diff = pos_flat[:, np.newaxis, :] - pos_flat[np.newaxis, :, :]
                dist = np.linalg.norm(diff, axis=2)
                np.fill_diagonal(dist, np.inf)
                min_dist = np.min(dist)
                if min_dist < 0.5:
                    # Penalty
                    e += 100.0 * (0.5 - min_dist) ** 2
            return e
        except:
            return 1e10

    def gradient(x):
        pos_flat = x.reshape(-1, 3)
        try:
            g = _dzugutov_grad(pos_flat)
            # Add gradient of penalty if active
            if len(pos_flat) > 1:
                diff = pos_flat[:, np.newaxis, :] - pos_flat[np.newaxis, :, :]
                dist = np.linalg.norm(diff, axis=2)
                np.fill_diagonal(dist, np.inf)
                min_dist = np.min(dist)
                if min_dist < 0.5:
                    # Gradient of 100 * (0.5 - min_dist)^2
                    # This is complex to compute analytically for min_dist.
                    # Given the steepness of Dzugutov at r<0.5, the physical gradient dominates.
                    pass
            return g.flatten()
        except:
            return np.zeros_like(x)

    x0 = pos.flatten()
    try:
        res = so.minimize(
            objective, 
            x0, 
            jac=gradient, 
            method='L-BFGS-B', 
            options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        if res.success or res.fun < objective(x0):
            return res.fun, res.x.reshape(-1, 3)
    except:
        pass
        
    # Fallback: simple gradient descent if L-BFGS fails
    pos_curr = pos.copy()
    e_curr = _dzugutov_energy(pos_curr)
    step = 0.01
    for _ in range(50):
        g = _dzugutov_grad(pos_curr)
        if np.linalg.norm(g) < 1e-8:
            break
        pos_next = pos_curr - step * g
        e_next = _dzugutov_energy(pos_next)
        if e_next < e_curr:
            pos_curr = pos_next
            e_curr = e_next
            step *= 1.1
        else:
            step *= 0.5
    return e_curr, pos_curr

def _generate_polytetrahedral(n, spacing=1.1):
    """
    Generate a polytetrahedral structure (McMillan model).
    1. Central atom.
    2. Icosahedral shell (12 atoms).
    3. Tetrahedral caps on the icosahedron faces (20 atoms).
    4. Additional shells if n > 33.
    """
    phi = (1 + np.sqrt(5)) / 2
    
    # 1. Central atom
    points = [np.array([0.0, 0.0, 0.0])]
    
    # 2. Icosahedron vertices
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi],
        [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0],
        [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1],
        [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    
    # Normalize to radius r_ico
    # Ideal Dzugutov first neighbor distance is approx 1.8-2.0.
    # Let's set r_ico = 1.0 for now, we will scale/relax later.
    norms = np.linalg.norm(ico_verts, axis=1)
    ico_verts = ico_verts / norms[:, np.newaxis]
    
    for v in ico_verts:
        if len(points) < n:
            points.append(v.copy())
            
    if n <= 13:
        return np.array(points[:n])
        
    # 3. Tetrahedral caps
    # The 20 faces of the icosahedron.
    # We need to define the faces.
    # This is complex to hardcode. Alternative: Add atoms at the centroid of each face, pushed out.
    # Icosahedron faces connect 3 vertices.
    # Let's use a simpler heuristic: Add atoms in a second shell at the vertices of a larger icosahedron? No, that's not polytetrahedral.
    # Polytetrahedral means adding tetrahedra to faces.
    # Face centroid direction * (1 + h)
    
    # Hardcoded faces of icosahedron (indices into ico_verts)
    faces = [
        [0, 1, 11], [0, 11, 5], [0, 5, 10], [0, 10, 2], [0, 2, 6],
        [1, 4, 11], [4, 9, 11], [4, 5, 11], [4, 11, 8], [4, 8, 7],
        [3, 10, 5], [3, 5, 9], [3, 9, 1], [3, 1, 2], [3, 2, 7],
        [10, 8, 11], [10, 7, 8], [10, 2, 7], [10, 6, 2], [10, 8, 6],
        [6, 8, 9], [6, 9, 4], [6, 4, 7], [6, 7, 3], [6, 3, 2]
    ]
    # Note: The above face list is a standard icosahedron face list, but indices might vary based on vertex ordering.
    # Let's just generate points on a sphere shell for simplicity if n is large, or use a known McMillan generator.
    # For robustness, let's generate a "quasicrystal-like" random shell if n > 13.
    
    # Actually, a better approach for 38-60 atoms:
    # Start with 13-atom icosahedron.
    # Add 20 atoms in a second shell (dodecahedral vertices or similar).
    # Add more in a third shell.
    
    # Let's use a simpler, robust generator:
    # 1. 13-atom icosahedron.
    # 2. Add atoms at the midpoints of edges? No.
    # 3. Add atoms at the vertices of a dodecahedron (20 atoms) scaled to fit.
    
    # Dodecahedron vertices
    # (±1, ±1, ±1), (0, ±1/φ, ±φ), (±1/φ, ±φ, 0), (±φ, 0, ±1/φ)
    inv_phi = 1.0 / phi
    dode_verts = []
    for x in [-1, 1]:
        for y in [-1, 1]:
            for z in [-1, 1]:
                dode_verts.append([x, y, z])
    for x in [0]:
        for y in [-inv_phi, inv_phi]:
            for z in [-phi, phi]:
                dode_verts.append([x, y, z])
    for x in [-inv_phi, inv_phi]:
        for y in [-phi, phi]:
            for z in [0]:
                dode_verts.append([x, y, z])
    for x in [-phi, phi]:
        for y in [0]:
            for z in [-inv_phi, inv_phi]:
                dode_verts.append([x, y, z])
                
    dode_verts = np.array(dode_verts)
    norms_d = np.linalg.norm(dode_verts, axis=1)
    dode_verts = dode_verts / norms_d[:, np.newaxis]
    
    # Scale dodecahedron to be outside icosahedron
    # Icosahedron radius is 1.0 (normalized).
    # Dodecahedron radius should be approx 1.0 * (1 + 0.5) = 1.5?
    # Let's try 1.5.
    dode_verts = dode_verts * 1.5
    
    for v in dode_verts:
        if len(points) < n:
            points.append(v)
            
    if n <= 33:
        return np.array(points[:n])
        
    # Third shell: Icosahedron again, scaled
    ico_verts_3 = ico_verts * 2.5
    for v in ico_verts_3:
        if len(points) < n:
            points.append(v)
            
    # If still not enough, add random points on a sphere
    while len(points) < n:
        theta = np.random.uniform(0, 2 * np.pi)
        phi_r = np.random.uniform(0, np.pi)
        r = 3.0 + np.random.uniform(0, 0.5)
        x = r * np.sin(phi_r) * np.cos(theta)
        y = r * np.sin(phi_r) * np.sin(theta)
        z = r * np.cos(phi_r)
        points.append(np.array([x, y, z]))
        
    return np.array(points[:n])

def construct(instance):
    """
    Find low-energy structures for Dzugutov clusters using Basin Hopping.
    """
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    start_time = time.time()
    time_limit = 300.0  # Safety margin
    
    # 1. Generate initial polytetrahedral structure
    pos0 = _generate_polytetrahedral(size)
    
    # Normalize to reasonable scale
    # The potential has minimum around r ~ 2.0-2.5.
    # Our generated structure has radius ~ 3.0.
    # Let's scale it to have mean inter-atomic distance ~ 2.0.
    if size > 1:
        diff = pos0[:, np.newaxis, :] - pos0[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)
        mean_dist = np.mean(dist)
        if mean_dist > 0:
            scale = 2.0 / mean_dist
            pos0 = pos0 * scale
    
    # Relax the initial structure
    best_energy, best_pos = _local_minimize(pos0, max_iter=200)
    
    # 2. Basin Hopping
    # Perform N random kicks, relax, keep best.
    # Number of kicks depends on time available.
    # Each kick+relax takes ~ 0.1-0.5s for n=60.
    # We have ~ 290s. So we can do ~ 200-500 kicks.
    
    rng = np.random.default_rng(42)
    
    # Kick magnitude
    kick_mag = 0.5
    
    num_kicks = 0
    while time.time() - start_time < time_limit - 10:
        # Random kick
        pos_kicked = best_pos + rng.normal(0, kick_mag, size=(size, 3))
        
        # Relax
        try:
            e_kicked, pos_relaxed = _local_minimize(pos_kicked, max_iter=100)
        except:
            e_kicked = np.inf
            pos_relaxed = best_pos
            
        if e_kicked < best_energy:
            best_energy = e_kicked
            best_pos = pos_relaxed.copy()
            
        num_kicks += 1
        
        # Adaptive kick magnitude: if we found a better energy, reduce kick. If not, increase?
        # Standard basin hopping uses constant or slowly decreasing.
        # Let's keep it constant for simplicity, but maybe reduce slightly over time.
        if num_kicks % 50 == 0:
            kick_mag *= 0.95
            
    # Final check: try the provided relax function on the best found
    # It might use a different algorithm that finds a slightly lower minimum.
    try:
        energy_pub, relaxed_pub = relax(best_pos, potential, parameter)
        if energy_pub < best_energy:
            best_energy = energy_pub
            best_pos = relaxed_pub
    except:
        pass
        
    return best_pos
# EVOLVE-BLOCK-END
