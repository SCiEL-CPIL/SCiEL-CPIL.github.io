"""Dzugutov cluster global minima search using basin hopping with diverse polytetrahedral starts and L-BFGS polish."""
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax
from scipy.optimize import minimize

# Dzugutov potential parameters
A = 5.82
B = 1.28
a = 1.87
b = 1.94
c_true = 1.10
d_true = 0.27
m = 16.0

def _find_r_min():
    """Find the minimum energy distance r_min for the Dzugutov potential."""
    r = np.linspace(0.5, 2.0, 2000)
    energies = []
    for ri in r:
        if ri < a:
            e = A * (ri ** (-m) - B) * np.exp(c_true / (ri - a))
        elif ri < b:
            e = B * np.exp(d_true / (ri - b))
        else:
            e = 0.0
        energies.append(e)
    energies = np.array(energies)
    idx = np.argmin(energies)
    return r[idx]

R_MIN = _find_r_min()

def _icosahedron_vertices(radius=1.0):
    """Generate 12 vertices of a regular icosahedron centered at origin."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi],
        [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0],
        [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1],
        [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1)
    verts = verts / norms[:, np.newaxis] * radius
    return verts

def _build_mackay_icosahedron(n, rng, r_min=R_MIN):
    """
    Build a Mackay icosahedral structure.
    For n=13: center + 12 vertices
    For n=55: 1 + 12 + 42 (Mackay icosahedron)
    For other n: start with appropriate Mackay shell and add/remove atoms.
    """
    points = [np.zeros(3)]
    
    # Shell 1: 12 icosahedron vertices
    ico = _icosahedron_vertices(r_min)
    points.extend(list(ico))
    
    if n <= 13:
        if n < 13:
            idx = rng.choice(13, n, replace=False)
            return np.array([points[i] for i in idx])
        return np.array(points)
    
    # Shell 2: Mackay icosahedron has 42 atoms on second shell
    phi = (1 + np.sqrt(5)) / 2
    r2 = r_min * np.sqrt(1 + 1/phi**2)
    
    needed = min(n - 13, 42)
    added = 0
    min_dist = 0.5
    
    # Generate candidate points on the second shell using icosahedral symmetry
    ico_verts = _icosahedron_vertices(1.0)
    n_ico = len(ico_verts)
    
    # Find all triangles (faces) of the icosahedron
    faces = []
    for i in range(n_ico):
        for j in range(i+1, n_ico):
            for k in range(j+1, n_ico):
                d_ij = np.linalg.norm(ico_verts[i] - ico_verts[j])
                d_jk = np.linalg.norm(ico_verts[j] - ico_verts[k])
                d_ik = np.linalg.norm(ico_verts[i] - ico_verts[k])
                if abs(d_ij - d_jk) < 1e-6 and abs(d_jk - d_ik) < 1e-6:
                    if d_ij < 2.5:
                        faces.append((i, j, k))
    
    # Add face centers
    for face in faces:
        if added >= needed:
            break
        center = (ico_verts[face[0]] + ico_verts[face[1]] + ico_verts[face[2]]) / 3
        p = center / np.linalg.norm(center) * r2
        valid = True
        for q in points:
            if np.linalg.norm(p - q) < min_dist:
                valid = False
                break
        if valid:
            points.append(p)
            added += 1
    
    # Add edge midpoints if still needed
    if added < needed:
        edges = set()
        for face in faces:
            for i in range(3):
                for j in range(i+1, 3):
                    edge = tuple(sorted((face[i], face[j])))
                    edges.add(edge)
        
        for edge in edges:
            if added >= needed:
                break
            mid = (ico_verts[edge[0]] + ico_verts[edge[1]]) / 2
            p = mid / np.linalg.norm(mid) * r2
            valid = True
            for q in points:
                if np.linalg.norm(p - q) < min_dist:
                    valid = False
                    break
            if valid:
                points.append(p)
                added += 1
    
    # If we need more than 55, add random points on outer shells
    while len(points) < n:
        r_outer = r2 + 0.3 * rng.random()
        u = rng.random()
        v = rng.random()
        theta = 2 * np.pi * u
        phi_ang = np.arccos(2 * v - 1)
        p = np.array([
            r_outer * np.sin(phi_ang) * np.cos(theta),
            r_outer * np.sin(phi_ang) * np.sin(theta),
            r_outer * np.cos(phi_ang)
        ])
        points.append(p)
    
    if len(points) > n:
        keep = list(range(13))
        for i in range(13, len(points)):
            if len(keep) < n:
                keep.append(i)
        points = [points[i] for i in keep]
    
    return np.array(points)

def _build_random_compact(n, rng, r_min=R_MIN):
    """Build a random compact cluster."""
    radius = r_min * (n / 13) ** (1/3) * 1.2
    points = []
    attempts = 0
    min_dist = 0.5
    
    while len(points) < n and attempts < 10000:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            valid = True
            for q in points:
                if np.linalg.norm(p - q) < min_dist:
                    valid = False
                    break
            if valid:
                points.append(p)
        attempts += 1
        
    while len(points) < n:
        base = points[len(points) % max(1, len(points))] if points else np.zeros(3)
        p = base + 0.2 * rng.standard_normal(3)
        points.append(p)
        
    return np.array(points)

def _dzugutov_energy(positions):
    """Compute exact Dzugutov potential energy."""
    n = len(positions)
    if n < 2:
        return 0.0
    
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    
    iu = np.triu_indices(n, k=1)
    r2_pair = r2[iu]
    
    energy = 0.0
    valid = r2_pair > 1e-12
    if not np.any(valid):
        return 0.0
    
    r = np.sqrt(r2_pair[valid])
    
    mask1 = r < a
    if np.any(mask1):
        r1 = r[mask1]
        energy += np.sum(A * (r1 ** (-m) - B) * np.exp(c_true / (r1 - a)))
    
    mask2 = (r >= a) & (r < b)
    if np.any(mask2):
        r2_ = r[mask2]
        energy += np.sum(B * np.exp(d_true / (r2_ - b)))
    
    return float(energy)

def _dzugutov_energy_and_grad(x):
    """Compute Dzugutov energy and gradient for L-BFGS optimization."""
    pos = x.reshape(-1, 3)
    n = len(pos)
    
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    iu = np.triu_indices(n, k=1)
    r2_pair = r2[iu]
    
    energy = 0.0
    grad = np.zeros_like(pos)
    
    valid = r2_pair > 1e-12
    if not np.any(valid):
        return energy, grad.flatten()
    
    r = np.sqrt(r2_pair[valid])
    diff_ = diff[iu]
    
    # For r < a: V = A*(r^-m - B)*exp(c/(r-a))
    mask1 = r < a
    if np.any(mask1):
        r1 = r[mask1]
        d1 = diff_[mask1]
        
        # V(r) = A*(r^-m - B)*exp(c/(r-a))
        exp_term = np.exp(c_true / (r1 - a))
        poly_term = r1 ** (-m) - B
        v1 = A * poly_term * exp_term
        energy += np.sum(v1)
        
        # dV/dr = A * (-m*r^(-m-1))*exp(c/(r-a)) + A*(r^-m - B)*exp(c/(r-a))*(-c/(r-a)^2)
        dv1_dr = A * (-m * r1 ** (-m - 1)) * exp_term + A * poly_term * exp_term * (-c_true / (r1 - a) ** 2)
        
        # Force on atom i: -dV/dr * (ri - rj)/r
        # grad_i += dv1_dr * d1 / r1
        # grad_j -= dv1_dr * d1 / r1
        force_factor = dv1_dr / r1
        for k in range(len(r1)):
            i, j = iu[0][k], iu[1][k]
            grad[i] += force_factor[k] * d1[k]
            grad[j] -= force_factor[k] * d1[k]
    
    # For a <= r < b: V = B*exp(d/(r-b))
    mask2 = (r >= a) & (r < b)
    if np.any(mask2):
        r2_ = r[mask2]
        d2 = diff_[mask2]
        
        exp_term = np.exp(d_true / (r2_ - b))
        v2 = B * exp_term
        energy += np.sum(v2)
        
        # dV/dr = B*exp(d/(r-b))*(-d/(r-b)^2)
        dv2_dr = B * exp_term * (-d_true / (r2_ - b) ** 2)
        
        force_factor = dv2_dr / r2_
        for k in range(len(r2_)):
            # Need to map back to original indices
            # mask2 is a boolean mask on the valid array
            valid_indices = np.where(valid)[0]
            mask2_indices = np.where(mask2)[0]
            for kk in range(len(mask2_indices)):
                global_idx = valid_indices[mask2_indices[kk]]
                i, j = iu[0][global_idx], iu[1][global_idx]
                grad[i] += force_factor[kk] * d2[kk]
                grad[j] -= force_factor[kk] * d2[kk]
    
    return energy, grad.flatten()

def _polish_with_l_bfgs(positions, max_iter=100):
    """Polish a structure using L-BFGS-B with the Dzugutov energy and gradient."""
    try:
        x0 = positions.flatten()
        result = minimize(_dzugutov_energy_and_grad, x0, method='L-BFGS-B', 
                         jac=True, options={'maxiter': max_iter, 'gtol': 1e-8})
        if result.success:
            return result.x.reshape(-1, 3), result.fun
    except Exception:
        pass
    return positions, _dzugutov_energy(positions)

def construct(instance):
    """Find low-energy Dzugutov clusters using basin hopping with diverse starts."""
    start_time = time.time()
    time_limit = 300.0
    safety_margin = 15.0
    n = instance["n"]
    potential = instance["potential"]
    param = None
    
    rng = np.random.default_rng(n + 42)
    
    best_energy = float('inf')
    best_positions = None
    
    def check_time():
        return (time.time() - start_time) < (time_limit - safety_margin)
    
    def try_relax(pos, label=""):
        nonlocal best_energy, best_positions
        try:
            energy, relaxed_pos = relax(pos, potential, param)
            if energy < best_energy:
                best_energy = energy
                best_positions = relaxed_pos.copy()
                return True
        except Exception:
            pass
        return False
    
    def try_polish(pos, label=""):
        """Try L-BFGS polish on a position."""
        nonlocal best_energy, best_positions
        try:
            polished, energy = _polish_with_l_bfgs(pos, max_iter=50)
            if energy < best_energy:
                best_energy = energy
                best_positions = polished.copy()
                return True
        except Exception:
            pass
        return False
    
    # Phase 1: Generate diverse initial guesses
    initial_guesses = []
    
    # Mackay icosahedral guesses
    for i in range(5):
        if not check_time(): break
        pos = _build_mackay_icosahedron(n, rng)
        pos_noisy = pos + 0.05 * rng.standard_normal((n, 3))
        initial_guesses.append(pos_noisy)
    
    # Random compact guesses
    for i in range(5):
        if not check_time(): break
        pos = _build_random_compact(n, rng)
        initial_guesses.append(pos)
    
    # Relax all initial guesses
    for i, pos_guess in enumerate(initial_guesses):
        if not check_time(): break
        try_relax(pos_guess, f"init_{i}")
    
    # Phase 2: Basin hopping
    if best_positions is not None:
        kick_sizes = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        for kick_size in kick_sizes:
            if not check_time(): break
            
            # Multiple kicks at this size
            n_kicks = max(1, int(20 * (time_limit - safety_margin - (time.time() - start_time)) / time_limit))
            for i in range(n_kicks):
                if not check_time(): break
                
                pos_kicked = best_positions + kick_size * rng.standard_normal((n, 3))
                try_relax(pos_kicked, f"kick_{kick_size}_{i}")
    
    # Phase 3: More diverse starts if time permits
    extra_starts = 10
    for i in range(extra_starts):
        if not check_time(): break
        
        if i % 3 == 0:
            pos = _build_mackay_icosahedron(n, rng)
        elif i % 3 == 1:
            pos = _build_random_compact(n, rng)
        else:
            # Deformed Mackay
            pos = _build_mackay_icosahedron(n, rng)
            pos = pos * (1.0 + 0.1 * rng.standard_normal(3))
        
        pos_noisy = pos + 0.03 * rng.standard_normal((n, 3))
        try_relax(pos_noisy, f"extra_{i}")
        
        # Quick basin hopping from this new start
        if check_time():
            for kick in [0.1, 0.2, 0.3]:
                if not check_time(): break
                pos_kicked = pos_noisy + kick * rng.standard_normal((n, 3))
                try_relax(pos_kicked, f"extra_kick_{i}_{kick}")
    
    # Phase 4: Diverse noise relaxations from best structure
    if best_positions is not None:
        for noise_level in [0.02, 0.05, 0.1, 0.15]:
            if not check_time(): break
            for i in range(3):
                if not check_time(): break
                pos_noisy = best_positions + noise_level * rng.standard_normal((n, 3))
                try_relax(pos_noisy, f"noise_{noise_level}_{i}")
    
    # Phase 5: L-BFGS polish on best structure
    if best_positions is not None and check_time():
        polished, energy = _polish_with_l_bfgs(best_positions, max_iter=100)
        if energy < best_energy:
            best_energy = energy
            best_positions = polished
    
    # Phase 6: Very fine-tuning kicks for local refinement
    if best_positions is not None:
        for kick in [0.01, 0.02, 0.03, 0.05]:
            if not check_time(): break
            for i in range(2):
                if not check_time(): break
                pos_kicked = best_positions + kick * rng.standard_normal((n, 3))
                try_relax(pos_kicked, f"fine_kick_{kick}_{i}")
                # Also try polish
                if check_time():
                    try_polish(pos_kicked, f"fine_polish_{kick}_{i}")
    
    # Phase 7: Final L-BFGS polish with tighter tolerance
    if best_positions is not None and check_time():
        polished, energy = _polish_with_l_bfgs(best_positions, max_iter=200)
        if energy < best_energy:
            best_energy = energy
            best_positions = polished
    
    # Final validation: ensure minimum distance constraint
    if best_positions is not None:
        # Check minimum distance
        diff = best_positions[:, np.newaxis, :] - best_positions[np.newaxis, :, :]
        r2 = np.sum(diff ** 2, axis=2)
        iu = np.triu_indices(n, k=1)
        min_dist = np.sqrt(np.min(r2[iu]))
        
        if min_dist < 0.5:
            # Try to fix by relaxing again
            try:
                energy, relaxed_pos = relax(best_positions, potential, param)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = relaxed_pos
            except Exception:
                pass
    
    # Fallback
    if best_positions is None:
        best_positions = _build_random_compact(n, rng)
        
    return best_positions
