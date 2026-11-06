import numpy as np
import time
import math
from scipy.optimize import minimize

def _morse_energy(pos, rho):
    """Compute Morse energy for a configuration."""
    n = len(pos)
    if n < 2:
        return 0.0
    # Vectorized distance calculation
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    upper = np.triu_indices(n, 1)
    r = dists[upper]
    val = rho * (1.0 - r)
    exp_val = np.exp(val)
    e = np.sum(exp_val * (exp_val - 2.0))
    return float(e)

def _morse_gradient(pos, rho):
    """Compute gradient of Morse energy (fully vectorized)."""
    n = len(pos)
    grad = np.zeros_like(pos)
    if n < 2:
        return grad
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    
    exp_val = np.exp(rho * (1.0 - dists))
    # d/dr [exp(rho(1-r))(exp(rho(1-r))-2)] = -rho * exp(rho(1-r)) * (2*exp(rho(1-r)) - 2)
    # Wait, let's derive carefully.
    # E = sum exp(u)(exp(u)-2) where u = rho(1-r)
    # dE/du = exp(u)(exp(u)-2) + exp(u)*exp(u) = exp(u)(2exp(u)-2)
    # du/dr = -rho
    # dE/dr = -rho * exp(u) * (2exp(u)-2) = -2*rho*exp(u)*(exp(u)-1)
    # Force on i from j: F_ij = -dE/dr_ij * (r_i - r_j)/r_ij
    # dE/dx_i = sum_j dE/dr_ij * (x_i - x_j)/r_ij
    # dE/dr_ij is the scalar derivative above.
    pair_force_mag = -2.0 * rho * exp_val * (exp_val - 1.0)
    np.fill_diagonal(pair_force_mag, 0.0)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        W = pair_force_mag / dists
    W = np.where(dists > 1e-10, W, 0.0)
    # grad_i = sum_j W_ij * (pos_i - pos_j) = pos_i * sum_j W_ij - sum_j W_ij * pos_j
    Ws = W.sum(axis=1)
    grad = Ws[:, None] * pos - W @ pos
    return grad

def _min_dist_check(pos, min_d=0.5):
    """Check minimum pairwise distance."""
    n = len(pos)
    if n < 2:
        return True
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    return np.min(dists) >= min_d - 1e-9

def _ensure_min_dist(pos, min_d=0.51, max_iter=500):
    """Separate atoms that are too close using a simple repulsion scheme."""
    pos = pos.copy()
    n = len(pos)
    for _ in range(max_iter):
        if not _min_dist_check(pos, min_d):
            diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff ** 2, axis=-1))
            np.fill_diagonal(dists, np.inf)
            idx_i, idx_j = np.unravel_index(np.argmin(dists), dists.shape)
            if idx_i > idx_j:
                idx_i, idx_j = idx_j, idx_i
            rij = pos[idx_i] - pos[idx_j]
            r = np.linalg.norm(rij)
            if r < 1e-10:
                rij = np.array([1.0, 0, 0])
                r = 1.0
            push = (min_d - r + 0.01) / 2.0
            pos[idx_i] += push * rij / r
            pos[idx_j] -= push * rij / r
        else:
            break
    return pos

def _fcc_points(n, spacing=1.0):
    """Generate FCC lattice points."""
    pts = []
    a = spacing
    max_r = int(np.ceil(n ** (1/3))) + 4
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            for k in range(-max_r, max_r + 1):
                for dx, dy, dz in [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]:
                    p = np.array([i + dx, j + dy, k + dz]) * a
                    pts.append(p)
    pts = np.array(pts)
    # Center the cluster
    center = np.mean(pts, axis=0)
    pts = pts - center
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)[:n]
    return pts[idx].copy()

def _hcp_points(n, spacing=1.0):
    """Generate HCP lattice points."""
    pts = []
    a = spacing
    c_over_a = np.sqrt(8.0 / 3.0)
    max_r = int(np.ceil(n ** (1/3))) + 4
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            for k in range(-max_r, max_r + 1):
                p1 = np.array([i, j, k * c_over_a]) * a
                p2 = np.array([i + 1/3.0, j + 2/3.0, (k + 0.5) * c_over_a]) * a
                pts.append(p1)
                pts.append(p2)
    pts = np.array(pts)
    center = np.mean(pts, axis=0)
    pts = pts - center
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)[:n]
    return pts[idx].copy()

def _bcc_points(n, spacing=1.0):
    """Generate BCC lattice points."""
    pts = []
    a = spacing
    max_r = int(np.ceil(n ** (1/3))) + 4
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            for k in range(-max_r, max_r + 1):
                p1 = np.array([i, j, k]) * a
                p2 = np.array([i + 0.5, j + 0.5, k + 0.5]) * a
                pts.append(p1)
                pts.append(p2)
    pts = np.array(pts)
    center = np.mean(pts, axis=0)
    pts = pts - center
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)[:n]
    return pts[idx].copy()

def _mckay_icosahedron(n, spacing=1.0):
    """Generate Mackay icosahedron cluster points."""
    phi = (1 + np.sqrt(5)) / 2
    # 12 vertices of icosahedron
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    pts = [np.zeros(3)]
    
    # Shell 1: 12 atoms at vertices
    for v in verts:
        pts.append(v.copy())
        
    # Shell 2: 30 atoms at face centers (scaled)
    # Faces of icosahedron
    faces = [
        [0, 4, 8], [0, 9, 4], [0, 1, 5], [0, 5, 10],
        [0, 10, 9], [0, 8, 7], [0, 7, 11], [0, 11, 9],
        [1, 6, 8], [1, 8, 4], [1, 2, 3], [1, 3, 5],
        [2, 6, 7], [2, 7, 3], [2, 3, 11], [2, 11, 6],
        [3, 7, 11], [3, 11, 5], [3, 5, 2], [4, 9, 8],
        [5, 9, 10], [5, 10, 6], [6, 10, 11], [6, 11, 7],
        [7, 11, 2], [7, 2, 6], [8, 4, 9], [9, 4, 5],
        [10, 5, 11], [11, 5, 6]
    ]
    # Actually, standard Mackay has 30 atoms on the second shell.
    # The distance to face center is different.
    # Let's just generate points on the icosahedral surface and pick the densest?
    # No, Mackay is specific.
    # Shell 2 radius is sqrt(5)/phi? No.
    # Let's use the fact that Mackay N=147 is (1, 12, 30, 60, 44) -> 147.
    # N=13 (1,12), N=55 (1,12,30,12?), N=147.
    # For general N, we take the first N points from the concentric shells.
    
    # Generate points for shells 1, 2, 3, 4
    # Shell 1: 12 vertices (r=1)
    # Shell 2: 30 face centers (r=sqrt(5)/phi? No, r = sqrt(1 + 1/phi^2)?)
    # Let's calculate face centers.
    face_centers = []
    for f in faces:
        c = (verts[f[0]] + verts[f[1]] + verts[f[2]]) / 3.0
        c = c / np.linalg.norm(c)
        face_centers.append(c)
    # There are 20 faces. But Mackay shell 2 has 30 atoms.
    # Ah, the 30 atoms are at the midpoints of edges? Or face centers?
    # Mackay 55: 1 + 12 + 30 + 12.
    # The 30 atoms are at the centers of the 20 faces? No, 20 faces.
    # Actually, the 30 atoms are located at the centers of the 20 faces? 
    # Wait, 1+12+30+12 = 55.
    # The 30 atoms are on the second shell.
    # Let's just use a simpler heuristic: generate many points on icosahedral shells and pick closest to origin.
    
    # Let's generate points on shells with radii 1, 2, 3, 4 (scaled)
    # And pick the first N.
    # But we need the *correct* Mackay positions.
    # For now, let's use the vertices and face centers and edge midpoints.
    
    edge_mids = []
    edges = set()
    for f in faces:
        edges.add(tuple(sorted([f[0], f[1]])))
        edges.add(tuple(sorted([f[1], f[2]])))
        edges.add(tuple(sorted([f[2], f[0]])))
    for e in edges:
        m = (verts[e[0]] + verts[e[1]]) / 2.0
        m = m / np.linalg.norm(m)
        edge_mids.append(m)
        
    # Combine shells
    shell1 = verts
    shell2 = np.array(face_centers) # 20 points
    shell3 = np.array(edge_mids) # 30 points
    # Wait, Mackay 55 is 1+12+30+12.
    # The 30 points are likely the edge midpoints? No, edge midpoints are 30.
    # The 12 points are likely vertices of a smaller icosahedron?
    
    # Let's just use the union of these and sort by distance from center.
    all_pts = [np.zeros(3)]
    all_pts.extend(shell1)
    all_pts.extend(shell3) # 30 edge mids
    all_pts.extend(shell2) # 20 face centers
    # We need 55 for the full 2nd shell. 1+12+30+12=55.
    # The remaining 12 are probably vertices of a dual icosahedron?
    
    # For simplicity and robustness, let's just generate a dense set of icosahedral points
    # and relax. The relaxation will fix the positions if they are close.
    
    pts_list = [np.zeros(3)]
    pts_list.extend(shell1)
    pts_list.extend(shell3)
    pts_list.extend(shell2)
    # Add scaled vertices for larger N
    for scale in [1.5, 2.0, 2.5]:
        pts_list.extend([v * scale for v in shell1])
        pts_list.extend([v * scale for v in shell3])
        
    pts = np.array(pts_list[:n])
    if len(pts) < n:
        # Fill with random points near the surface
        while len(pts) < n:
            p = rng.uniform(-1, 1, 3)
            p = p / np.linalg.norm(p) * 2.0
            pts = np.vstack([pts, [p]])
            
    # Scale to unit distance
    if len(pts) > 1:
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        min_d = np.min(dists)
        if min_d > 1e-10:
            pts = pts * (spacing / min_d)
            
    return pts

def _relax_lbfgs(pos, rho, maxiter=2000):
    """Relax using L-BFGS-B."""
    n = len(pos)
    x0 = pos.flatten()

    def objective(x):
        p = x.reshape(n, 3)
        return _morse_energy(p, rho)

    def gradient(x):
        p = x.reshape(n, 3)
        return _morse_gradient(p, rho).flatten()

    try:
        result = minimize(objective, x0, jac=gradient, method='L-BFGS-B',
                          options={'maxiter': maxiter, 'gtol': 1e-8, 'ftol': 1e-10, 'maxls': 30})
        if result.fun < _morse_energy(pos, rho):
            return result.x.reshape(n, 3), result.fun
    except Exception:
        pass

    return pos, _morse_energy(pos, rho)

def _basin_hopping(pos, rho, rng, max_iter=50, kick_scale=0.3, time_budget=None):
    """
    Basin hopping: kick + local minimize + accept/reject.
    Accepts uphill moves with probability exp(-dE/T) to escape local minima.
    """
    n = len(pos)
    current_pos = pos.copy()
    current_energy = _morse_energy(current_pos, rho)
    best_pos = current_pos.copy()
    best_energy = current_energy
    
    # Temperature schedule: start high, cool down
    T0 = 2.0 * rho  # Initial temperature scaled by rho
    T_final = 0.1
    alpha = (T_final / T0) ** (1.0 / max(max_iter, 1))
    
    for it in range(max_iter):
        if time_budget is not None and time.time() > time_budget:
            break
        
        # Kick: random perturbation
        T = T0 * (alpha ** it)
        kick = rng.normal(0, kick_scale, (n, 3))
        kicked_pos = current_pos + kick
        kicked_pos = _ensure_min_dist(kicked_pos)
        
        if not _min_dist_check(kicked_pos):
            continue
        
        # Local minimize
        try:
            new_pos, new_energy = _relax_lbfgs(kicked_pos, rho, maxiter=300)
        except Exception:
            continue
        
        if not _min_dist_check(new_pos):
            continue
        
        # Accept/reject
        dE = new_energy - current_energy
        if dE < 0 or (T > 0 and rng.random() < math.exp(-dE / T)):
            current_pos = new_pos
            current_energy = new_energy
        
        # Track best
        if current_energy < best_energy:
            best_energy = current_energy
            best_pos = current_pos.copy()
    
    return best_pos, best_energy

def construct(instance):
    """Find low-energy Morse cluster configuration using basin hopping."""
    size = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind", 10))
    
    start_time = time.time()
    deadline = start_time + 45.0  # 300s total / 6 instances = 50s. Use 45s to be safe.
    
    rng = np.random.default_rng(size * 13 + int(rho) * 7 + 42)
    
    best_energy = np.inf
    best_pos = None
    
    # Generate multiple diverse starting structures
    starts = []
    
    if rho <= 6:
        # Long/Medium range: FCC and HCP are strong candidates
        starts.append(_fcc_points(size))
        starts.append(_hcp_points(size))
        # Perturbed FCC
        starts.append(_fcc_points(size) + rng.normal(0, 0.15, (size, 3)))
        # BCC
        starts.append(_bcc_points(size))
    else:
        # Short range: Icosahedral (Mackay) is the strong candidate
        starts.append(_mckay_icosahedron(size))
        # Perturbed Icosahedral
        starts.append(_mckay_icosahedron(size) + rng.normal(0, 0.15, (size, 3)))
        # FCC as fallback
        starts.append(_fcc_points(size))
        # HCP as fallback
        starts.append(_hcp_points(size))
        # BCC
        starts.append(_bcc_points(size))
    
    # Run basin hopping from each start
    for s_idx, pos in enumerate(starts):
        if time.time() > deadline - 5.0:
            break
        
        pos = _ensure_min_dist(pos)
        if not _min_dist_check(pos):
            continue
        
        # First do a quick local relax to get a good starting point
        try:
            relaxed_pos, energy = _relax_lbfgs(pos, rho, maxiter=500)
            if _min_dist_check(relaxed_pos):
                if energy < best_energy:
                    best_energy = energy
                    best_pos = relaxed_pos.copy()
        except Exception:
            continue
        
        # Then basin hopping from this relaxed structure
        if time.time() > deadline - 10.0:
            break
        
        try:
            bh_pos, bh_energy = _basin_hopping(
                relaxed_pos, rho, rng,
                max_iter=30,
                kick_scale=0.25,
                time_budget=deadline - 3.0
            )
            if _min_dist_check(bh_pos) and bh_energy < best_energy:
                best_energy = bh_energy
                best_pos = bh_pos.copy()
        except Exception:
            continue
    
    # If we have time, do more basin hopping iterations from the best solution
    if best_pos is not None and time.time() < deadline - 15.0:
        for _ in range(3):
            if time.time() > deadline - 5.0:
                break
            try:
                bh_pos, bh_energy = _basin_hopping(
                    best_pos, rho, rng,
                    max_iter=20,
                    kick_scale=0.35,
                    time_budget=deadline - 3.0
                )
                if _min_dist_check(bh_pos) and bh_energy < best_energy:
                    best_energy = bh_energy
                    best_pos = bh_pos.copy()
            except Exception:
                continue

    # Final check
    if best_pos is None:
        # Fallback: Random
        radius = 1.0 * (size ** (1/3)) * 0.8
        pts = []
        while len(pts) < size:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                pts.append(p)
        best_pos = _ensure_min_dist(np.array(pts))
        if not _min_dist_check(best_pos):
             for i in range(size):
                 for j in range(i+1, size):
                     if np.linalg.norm(best_pos[i] - best_pos[j]) < 0.5:
                         best_pos[j] += 0.1 * rng.normal(0, 1, 3)
    
    return best_pos
