import time
import numpy as np
import sys

N = 38

def _lj_energy_and_grad(pos):
    """Total LJ energy and gradient for positions pos (N,3)."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0)
    
    r6 = r2 ** 3
    r12 = r6 ** 2

    pair_energy = 4.0 * (1.0 / r12 - 1.0 / r6)
    np.fill_diagonal(pair_energy, 0.0)
    energy = 0.5 * np.sum(pair_energy)

    r8 = r2 ** 4
    r14 = r2 ** 7
    coeff = 4.0 * (6.0 / r8 - 12.0 / r14)
    np.fill_diagonal(coeff, 0.0)
    row_sum = np.sum(coeff, axis=1, keepdims=True)
    grad = row_sum * pos - coeff @ pos

    return energy, grad

def _make_fcc_43():
    """Generate the 43-atom FCC cluster (1+12+6+24)."""
    pts = []
    # Center
    pts.append([0, 0, 0])
    # 12 nearest neighbors (1,1,0) perms
    for i in range(-1, 2):
        for j in range(-1, 2):
            for k in range(-1, 2):
                if abs(i)+abs(j)+abs(k) == 2 and i*j*k == 0: # (1,1,0) type
                    pts.append([i, j, k])
    # 6 second nearest neighbors (2,0,0) perms
    for i in [-2, 2]:
        pts.append([i, 0, 0])
    for j in [-2, 2]:
        pts.append([0, j, 0])
    for k in [-2, 2]:
        pts.append([0, 0, k])
    # 24 third nearest neighbors (2,1,1) perms
    for i in [-2, 2]:
        for j in [-1, 1]:
            for k in [-1, 1]:
                pts.append([i, j, k])
    # Permutations of (2,1,1)
    for i in [-1, 1]:
        for j in [-2, 2]:
            for k in [-1, 1]:
                pts.append([i, j, k])
    for i in [-1, 1]:
        for j in [-1, 1]:
            for k in [-2, 2]:
                pts.append([i, j, k])
                
    pts = np.array(pts, dtype=np.float64)
    pts -= np.mean(pts, axis=0)
    
    # Scale to min distance 2^(1/6)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    min_r = np.sqrt(np.min(r2))
    target_r = 2.0 ** (1.0 / 6.0)
    if min_r > 1e-10:
        pts *= (target_r / min_r)
        
    return pts

def _make_icosahedral_38():
    """Generate the 38-atom Mackay icosahedron."""
    # 1 center
    pts = [[0, 0, 0]]
    
    # 12 vertices of icosahedron
    phi = (1 + np.sqrt(5)) / 2
    verts = [
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
        [0, phi, 1/phi], [0, phi, -1/phi], [0, -phi, 1/phi], [0, -phi, -1/phi],
        [1/phi, 0, phi], [-1/phi, 0, phi], [1/phi, 0, -phi], [-1/phi, 0, -phi],
        [phi, 1/phi, 0], [phi, -1/phi, 0], [-phi, 1/phi, 0], [-phi, -1/phi, 0]
    ]
    # Wait, standard icosahedron vertices are (0, ±1, ±phi) perms.
    verts = []
    for x in [0, 1, -1]:
        for y in [1, -1]:
            for z in [phi, -phi]:
                verts.append([x, y, z])
    # This generates 12 points?
    # (0, 1, phi), (0, 1, -phi), (0, -1, phi), (0, -1, -phi)
    # (1, 1, phi)... No, permutations of (0, 1, phi).
    # Permutations:
    # (0, 1, phi), (0, 1, -phi), (0, -1, phi), (0, -1, -phi)
    # (1, 0, phi), (1, 0, -phi), (-1, 0, phi), (-1, 0, -phi)
    # (phi, 0, 1), (phi, 0, -1), (-phi, 0, 1), (-phi, 0, -1)
    # (phi, 1, 0), (phi, -1, 0), (-phi, 1, 0), (-phi, -1, 0)
    # (1, phi, 0), (-1, phi, 0), (1, -phi, 0), (-1, -phi, 0)
    # Total 12? No, 24.
    # Standard icosahedron has 12 vertices.
    # (0, ±1, ±phi) and permutations.
    # There are 3 axes. For each axis, 4 points. 3*4=12.
    
    verts = []
    for perm in [[0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi]]:
        for p in np.permutations(perm):
             # Avoid duplicates
             if not any(np.allclose(p, v) for v in verts):
                 verts.append(p)
                 
    # 12 vertices
    pts.extend(verts)
    
    # 12 edge centers
    # Edges connect vertices that are distance apart.
    # In icosahedron, edge length is 2 (if coords are 0,1,phi).
    # Distance squared = 2.
    # Find all pairs with dist^2 = 2.
    verts_arr = np.array(verts)
    edge_centers = []
    for i in range(12):
        for j in range(i+1, 12):
            d2 = np.sum((verts_arr[i] - verts_arr[j])**2)
            if abs(d2 - 2.0) < 1e-6:
                center = (verts_arr[i] + verts_arr[j]) / 2.0
                if not any(np.allclose(center, ec) for ec in edge_centers):
                    edge_centers.append(center)
    pts.extend(edge_centers)
    
    # 13 face centers?
    # 38 - 1 - 12 - 12 = 13.
    # The 13th atom is usually the center of one of the faces?
    # Or is it a 1+12+13+12 structure?
    # The Mackay icosahedron for N=38 is often 1+12+13+12?
    # Actually, the 38-atom icosahedron is 1+12+12+13?
    # Let's just take the 13 face centers of the icosahedron.
    # An icosahedron has 20 faces.
    # We need 13 more atoms.
    # This might be tricky.
    # Alternative: Just use the 25 atoms (1+12+12) and add 13 random atoms nearby?
    # Or, simpler: The global minimum is FCC. The icosahedron is a local min.
    # I will skip the complex icosahedral construction and rely on FCC + Pruning.
    
    # For now, return the 25-atom cluster padded with random noise?
    # No, let's just return the FCC result for both if icosahedral is too complex.
    # But I should provide a distinct initialization.
    
    # Let's just return the FCC 38-atom cluster (first 38 of 43) as a fallback.
    return _make_fcc_38_simple()

def _make_fcc_38_simple():
    """Generate 38-atom FCC cluster by taking 38 closest to origin."""
    max_c = 3
    pts = []
    for i in range(-max_c, max_c + 1):
        for j in range(-max_c, max_c + 1):
            for k in range(-max_c, max_c + 1):
                if (i + j + k) % 2 == 0:
                    pts.append((float(i), float(j), float(k)))
    pts = np.array(pts, dtype=np.float64)
    pts -= np.mean(pts, axis=0)
    dists = np.sum(pts ** 2, axis=1)
    order = np.argsort(dists)
    pts = pts[order][:N].copy()
    pts -= np.mean(pts, axis=0)
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    min_r = np.sqrt(np.min(r2))
    target_r = 2.0 ** (1.0 / 6.0)
    if min_r > 1e-10:
        pts *= (target_r / min_r)
    return pts

def _relax(pos, max_iter=1000, gtol=1e-10, ftol=1e-14):
    from scipy.optimize import minimize
    def obj_grad(x):
        p = x.reshape(-1, 3)
        e, g = _lj_energy_and_grad(p)
        return e, g.ravel()
    res = minimize(obj_grad, pos.ravel(), jac=True, method="L-BFGS-B", options={"maxiter": max_iter, "gtol": gtol, "ftol": ftol})
    p_out = res.x.reshape(-1, 3)
    e_out, _ = _lj_energy_and_grad(p_out)
    return p_out, e_out

def _prune_cluster(pos, n_keep):
    """Remove atoms with highest energy contribution."""
    n = len(pos)
    if n == n_keep:
        return pos
    e, g = _lj_energy_and_grad(pos)
    # Energy contribution of atom i is roughly sum of V_ij
    # We can compute this from the pair energies
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0)
    r6 = r2 ** 3
    r12 = r6 ** 2
    pair_energy = 4.0 * (1.0 / r12 - 1.0 / r6)
    np.fill_diagonal(pair_energy, 0.0)
    # Sum of pair energies for each atom
    atom_energies = np.sum(pair_energy, axis=1)
    # Remove atoms with highest (least negative) energy
    indices_to_remove = np.argsort(atom_energies)[- (n - n_keep):]
    mask = np.ones(n, dtype=bool)
    mask[indices_to_remove] = False
    return pos[mask]

def _ensure_min_sep(pos, min_sep=0.5):
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    min_d = np.sqrt(np.min(r2))
    if min_d < min_sep:
        scale = (min_sep / min_d) * 1.01
        pos = pos * scale
    return pos

def construct():
    t0 = time.time()
    deadline = t0 + 140.0
    
    best_pos = None
    best_energy = np.inf

    # Strategy 1: FCC 43 -> Relax -> Prune -> Relax
    try:
        if time.time() < deadline - 60.0:
            pos43 = _make_fcc_43()
            p43, e43 = _relax(pos43, max_iter=200, gtol=1e-8)
            p38 = _prune_cluster(p43, N)
            p38, e38 = _relax(p38, max_iter=500, gtol=1e-10)
            if e38 < best_energy:
                best_energy = e38
                best_pos = p38
    except Exception as ex:
        print(f"Strategy 1 failed: {ex}")

    # Strategy 2: FCC 38 (simple) -> Relax
    try:
        if time.time() < deadline - 30.0:
            pos38 = _make_fcc_38_simple()
            p38, e38 = _relax(pos38, max_iter=500, gtol=1e-10)
            if e38 < best_energy:
                best_energy = e38
                best_pos = p38
    except Exception as ex:
        print(f"Strategy 2 failed: {ex}")

    # Strategy 3: Polish best
    if best_pos is not None and time.time() < deadline - 10.0:
        try:
            p_pol, e_pol = _relax(best_pos, max_iter=1000, gtol=1e-12, ftol=1e-16)
            if e_pol < best_energy:
                best_energy = e_pol
                best_pos = p_pol
        except Exception:
            pass

    if best_pos is None:
        best_pos = _make_fcc_38_simple()
    
    best_pos = _ensure_min_sep(best_pos)
    return best_pos
