"""
Morse cluster global minima finder using structural seeds and L-BFGS optimization.
"""
import numpy as np
import time
import sys
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

# Global time management
_GLOBAL_START_TIME = None
_GLOBAL_BUDGET = 280.0  # Leave 20s safety margin for 300s limit

def _get_global_remaining_time():
    global _GLOBAL_START_TIME
    if _GLOBAL_START_TIME is None:
        _GLOBAL_START_TIME = time.time()
    elapsed = time.time() - _GLOBAL_START_TIME
    return max(0.0, _GLOBAL_BUDGET - elapsed)

def _morse_energy(pos, rho):
    """Compute Morse energy for a configuration."""
    n = len(pos)
    if n < 2:
        return 0.0
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    upper = np.triu_indices(n, 1)
    r = dists[upper]
    val = rho * (1.0 - r)
    exp_val = np.exp(val)
    e = np.sum(exp_val * (exp_val - 2.0))
    return float(e)

def _min_dist_check(pos, min_d=0.5):
    """Check minimum pairwise distance."""
    n = len(pos)
    if n < 2:
        return True
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    return np.min(dists) >= min_d - 1e-9

def _fix_min_dist(pos, min_d=0.5, rng=None):
    """Ensure minimum distance by pushing atoms apart."""
    if rng is None:
        rng = np.random.default_rng()
    n = len(pos)
    for _ in range(100):
        if _min_dist_check(pos, min_d):
            break
        # Find closest pair and push them apart
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        min_idx = np.unravel_index(np.argmin(dists), dists.shape)
        i, j = min_idx
        d = dists[i, j]
        if d < min_d:
            dir_vec = pos[i] - pos[j]
            norm = np.linalg.norm(dir_vec)
            if norm < 1e-6:
                dir_vec = rng.normal(0, 1, 3)
            dir_vec = dir_vec / np.linalg.norm(dir_vec)
            push = (min_d - d) / 2.0 + 0.01
            pos[i] += dir_vec * push
            pos[j] -= dir_vec * push
    return pos

def _generate_fcc(n, spacing=1.0):
    """Generate an FCC-like cluster of n atoms."""
    pts = []
    a = spacing
    max_r = int(np.ceil(n ** (1/3))) + 3
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            for k in range(-max_r, max_r + 1):
                for dx, dy, dz in [(0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)]:
                    p = np.array([i + dx, j + dy, k + dz]) * a
                    pts.append(p)
    pts = np.array(pts)
    # Center the cluster
    centroid = pts.mean(axis=0)
    pts = pts - centroid
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)[:n]
    return pts[idx]

def _generate_hcp(n, spacing=1.0):
    """Generate an HCP-like cluster of n atoms."""
    pts = []
    a = spacing
    c = a * np.sqrt(8.0 / 3.0)
    max_r = int(np.ceil(n ** (1/3))) + 3
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            for k in range(-max_r, max_r + 1):
                p1 = np.array([i, j, k]) * a
                p2 = np.array([i + 1/3, j + 2/3, k + 0.5]) * a
                p1[2] *= c / a
                p2[2] *= c / a
                pts.append(p1)
                pts.append(p2)
    pts = np.array(pts)
    centroid = pts.mean(axis=0)
    pts = pts - centroid
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)[:n]
    return pts[idx]

def _generate_mackay_icosahedron(n, spacing=1.0):
    """Generate a Mackay icosahedral cluster."""
    # 13-atom icosahedron
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True) * spacing
    
    pts = [np.zeros(3)]
    if n > 1:
        pts.extend([v.copy() for v in verts])
        
    if n > 13:
        # Add second shell (30 atoms)
        # The second shell of a Mackay icosahedron consists of 30 atoms.
        # They are located at the midpoints of the edges of the first shell? No.
        # They form 5 square faces and 5 triangular faces?
        # Actually, the 55-atom cluster is 1 + 12 + 30 + 12? No.
        # 13 (1st), 55 (2nd), 147 (3rd).
        # 1st shell: 12 vertices.
        # 2nd shell: 30 atoms. These are the centers of the 30 edges? No, the 13-atom icosahedron has 30 edges.
        # The 55-atom cluster has 1 center, 12 vertices, 30 edge centers, and 12 face centers?
        # 1 + 12 + 30 + 12 = 55.
        # So for n=38, we need to take 38 atoms from a 55-atom cluster.
        
        edges = []
        faces = []
        # Find edges (distance between verts < 1.05 * spacing)
        for i in range(12):
            for j in range(i+1, 12):
                d = np.linalg.norm(verts[i] - verts[j])
                if d < 1.5 * spacing: # Approximate edge length
                    edges.append((i, j))
        # Find faces (triplets forming a triangle)
        for i in range(12):
            for j in range(i+1, 12):
                for k in range(j+1, 12):
                    d1 = np.linalg.norm(verts[i] - verts[j])
                    d2 = np.linalg.norm(verts[j] - verts[k])
                    d3 = np.linalg.norm(verts[k] - verts[i])
                    if d1 < 1.5*spacing and d2 < 1.5*spacing and d3 < 1.5*spacing:
                        faces.append((i, j, k))
        
        edge_centers = []
        for i, j in edges:
            c = (verts[i] + verts[j]) / 2.0
            c = c / np.linalg.norm(c) * spacing * np.linalg.norm((verts[i] + verts[j]) / 2.0) / np.linalg.norm(verts[i]) # Adjust radius
            # Actually, just normalize and scale to spacing
            c = c / np.linalg.norm(c) * spacing
            edge_centers.append(c)
            
        face_centers = []
        for i, j, k in faces:
            c = (verts[i] + verts[j] + verts[k]) / 3.0
            c = c / np.linalg.norm(c) * spacing * 0.87 # Approximate
            face_centers.append(c)
            
        pts.extend(edge_centers)
        pts.extend(face_centers)
        
    pts = np.array(pts)
    centroid = pts.mean(axis=0)
    pts = pts - centroid
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)[:n]
    return pts[idx]

def _generate_decahedron(n, spacing=1.0):
    """Generate a decahedral cluster."""
    # 13-atom decahedron is similar to icosahedron but distorted.
    # For simplicity, use a perturbed icosahedron.
    ico = _generate_mackay_icosahedron(n, spacing)
    rng = np.random.default_rng(42)
    ico = ico + rng.normal(0, 0.1, ico.shape)
    return ico

def _generate_random_dense(n, rng, spacing=1.0):
    """Generate a random dense cluster."""
    pts = []
    radius = spacing * n ** (1/3) * 1.5
    min_d = 0.6 * spacing
    attempts = 0
    max_attempts = 10000
    while len(pts) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= min_d for q in pts):
                pts.append(p)
        attempts += 1
    while len(pts) < n:
        p = rng.uniform(-radius, radius, 3)
        pts.append(p)
    return np.array(pts)

def construct(instance):
    """Find low-energy Morse cluster configuration."""
    size = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind"))
    
    remaining_time = _get_global_remaining_time()
    if remaining_time <= 10.0:
        # Emergency fallback
        pos = _generate_fcc(size, spacing=1.0)
        if not _min_dist_check(pos):
            pos = _fix_min_dist(pos, 0.5, np.random.default_rng())
        return pos

    # Allocate time for this instance
    # 6 instances. Assume ~45s each.
    instance_budget = min(40.0, remaining_time * 0.8)
    
    start_time = time.time()
    deadline = start_time + instance_budget
    
    rng = np.random.default_rng(size * 7 + 42)
    
    best_energy = np.inf
    best_pos = None
    
    # Generate diverse initial configurations
    initial_configs = []
    
    # 1. FCC
    try:
        fcc = _generate_fcc(size, spacing=1.0)
        initial_configs.append(fcc)
    except Exception:
        pass
    
    # 2. HCP
    try:
        hcp = _generate_hcp(size, spacing=1.0)
        initial_configs.append(hcp)
    except Exception:
        pass
    
    # 3. Icosahedral
    try:
        ico = _generate_mackay_icosahedron(size, spacing=1.0)
        initial_configs.append(ico)
    except Exception:
        pass
        
    # 4. Decahedral
    try:
        deca = _generate_decahedron(size, spacing=1.0)
        initial_configs.append(deca)
    except Exception:
        pass

    # 5. Random dense
    for i in range(3):
        try:
            rd = _generate_random_dense(size, rng, spacing=1.0)
            initial_configs.append(rd)
        except Exception:
            pass

    # 6. Perturbed FCC
    try:
        fcc_base = _generate_fcc(max(size, 20), spacing=1.0)
        pert = fcc_base + rng.normal(0, 0.2, fcc_base.shape)
        if len(pert) > size:
            centroid = pert.mean(axis=0)
            dists = np.linalg.norm(pert - centroid, axis=1)
            idx = np.argsort(dists)[:size]
            pert = pert[idx]
        initial_configs.append(pert)
    except Exception:
        pass

    # Relax each configuration
    for pos in initial_configs:
        if time.time() > deadline - 5:
            break
        try:
            pos = _fix_min_dist(pos, 0.5, rng)
            # Use L-BFGS-B
            res = minimize(
                fun=lambda x: morse_energy_gradient(x.reshape(-1, 3), rho)[0],
                x0=pos.flatten(),
                jac=lambda x: morse_energy_gradient(x.reshape(-1, 3), rho)[1].flatten(),
                method='L-BFGS-B',
                options={'maxiter': 100, 'maxls': 20}
            )
            if res.success:
                relaxed_pos = res.x.reshape(-1, 3)
                if _min_dist_check(relaxed_pos):
                    energy = _morse_energy(relaxed_pos, rho)
                    if energy < best_energy:
                        best_energy = energy
                        best_pos = relaxed_pos.copy()
        except Exception:
            continue

    # If we have time, do more random restarts
    if best_pos is not None and time.time() < deadline - 10:
        time_per_restart = 2.0 if size < 50 else 4.0
        n_extra = int((deadline - time.time()) / time_per_restart)
        n_extra = max(0, min(15, n_extra))
        
        for i in range(n_extra):
            if time.time() > deadline - 5:
                break
            try:
                rp = _generate_random_dense(size, rng, spacing=1.0)
                rp = _fix_min_dist(rp, 0.5, rng)
                res = minimize(
                    fun=lambda x: morse_energy_gradient(x.reshape(-1, 3), rho)[0],
                    x0=rp.flatten(),
                    jac=lambda x: morse_energy_gradient(x.reshape(-1, 3), rho)[1].flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'maxls': 20}
                )
                if res.success:
                    relaxed_pos = res.x.reshape(-1, 3)
                    if _min_dist_check(relaxed_pos):
                        energy = _morse_energy(relaxed_pos, rho)
                        if energy < best_energy:
                            best_energy = energy
                            best_pos = relaxed_pos.copy()
            except Exception:
                continue

    # Final fallback
    if best_pos is None:
        best_pos = _generate_fcc(size, spacing=1.0)
        best_pos = _fix_min_dist(best_pos, 0.5, rng)

    return best_pos
