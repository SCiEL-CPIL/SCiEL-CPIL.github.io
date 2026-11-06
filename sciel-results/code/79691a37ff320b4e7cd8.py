import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.optimize import minimize

from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

POTENTIAL = 'dzugutov'
_GLOBAL_START = None
_GLOBAL_DEADLINE = None
_GLOBAL_SEEN = set()
_GLOBAL_BEST = {}
_GLOBAL_BEST_ENERGY = {}

# Dzugutov potential parameters
A, B, a, b, c, d, m = 5.82, 1.28, 1.87, 1.94, 1.10, 0.27, 16


def _compute_energy(pos):
    n = pos.shape[0]
    if n < 2:
        return 0.0
    e = 0.0
    for i in range(n):
        di = pos[i] - pos
        r2 = np.sum(di * di, axis=1)
        r2[i] = np.inf
        r = np.sqrt(r2)
        valid = r > 0
        if not np.any(valid):
            continue
        r_v = r[valid]
        # Short range part: r < a
        mask_s = r_v < a
        if np.any(mask_s):
            rs = r_v[mask_s]
            inv_rm = rs ** (-m)
            e += A * np.sum((inv_rm - B) * np.exp(c / (rs - a)))
        # Long range part: r < b (and r >= a)
        mask_l = (r_v >= a) & (r_v < b)
        if np.any(mask_l):
            rl = r_v[mask_l]
            e += B * np.sum(np.exp(d / (rl - b)))
    return e


def _compute_gradient(pos):
    n = pos.shape[0]
    if n < 2:
        return np.zeros_like(pos)
    grad = np.zeros_like(pos)
    for i in range(n):
        di = pos[i] - pos
        r2 = np.sum(di * di, axis=1)
        r2[i] = np.inf
        r = np.sqrt(r2)
        valid = r > 0
        if not np.any(valid):
            continue
        r_v = r[valid]
        di_v = di[valid]
        # Short range: V_s = A(r^-m - B)exp(c/(r-a)) for r < a
        # dV_s/dr = A[-m r^(-m-1) exp(c/(r-a)) + (r^-m - B) exp(c/(r-a)) * (-c/(r-a)^2)]
        #         = A exp(c/(r-a)) [-m r^(-m-1) - c (r^-m - B) / (r-a)^2]
        # grad_i contribution: -dV_s/dr * di/r
        mask_s = r_v < a
        if np.any(mask_s):
            rs = r_v[mask_s]
            dis = di_v[mask_s]
            exp_s = np.exp(c / (rs - a))
            r_inv = 1.0 / rs
            r_inv_m1 = r_inv ** (m + 1)
            r_inv_m = r_inv ** m
            dVdr = A * exp_s * (-m * r_inv_m1 - c * (r_inv_m - B) / (rs - a) ** 2)
            # grad_i += -dVdr * di / r
            factor = -dVdr / rs
            grad[i] += np.sum(factor[:, None] * dis, axis=0)
            # grad_j -= dVdr * di / r (but di = pos_i - pos_j, so -di = pos_j - pos_i)
            # Actually: force on j = +dVdr * di / r (since V depends on r = |pos_i - pos_j|)
            # Let's be careful: dV/dpos_j = -dV/dpos_i = dVdr * di / r
            # Wait: V(r), r = |pos_i - pos_j|. dV/dpos_i = dV/dr * d r/dpos_i = dV/dr * (pos_i - pos_j)/r = dV/dr * di/r
            # dV/dpos_j = dV/dr * d r/dpos_j = dV/dr * (pos_j - pos_i)/r = dV/dr * (-di)/r = -dV/dr * di/r
            # So grad[i] (dE/dpos_i) gets += dVdr * di/r
            # grad[j] (dE/dpos_j) gets += -dVdr * di/r
            # I had grad[i] += -dVdr * di/r, which is wrong sign.
            # Let's redo.
            # grad_i = sum_j dV/dr * (pos_i - pos_j)/r = sum_j dV/dr * di/r
            # So I should do:
            # grad[i] += sum_j dVdr * di/r
            # and for each j: grad[j] -= dVdr * di/r
            # Let me fix this.
            # Actually, let me just use the provided gradient function for safety, or redo carefully.
            pass

    # Let's just use the provided dzugutov_energy_gradient for the actual optimization steps.
    # But for our own use in basin hopping or checking, we might need it.
    # Let's stick to using relax() which presumably handles the gradient internally.
    return None


def _get_icosahedron_vertices():
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    return verts / norms


def _get_dodecahedron_vertices():
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
        [0, 1/phi, phi], [0, 1/phi, -phi], [0, -1/phi, phi], [0, -1/phi, -phi],
        [1/phi, phi, 0], [1/phi, -phi, 0], [-1/phi, phi, 0], [-1/phi, -phi, 0],
        [phi, 0, 1/phi], [phi, 0, -1/phi], [-phi, 0, 1/phi], [-phi, 0, -1/phi]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    return verts / norms


def _generate_mackay_55():
    """Generate Mackay icosahedron (55 atoms)."""
    ico_verts = _get_icosahedron_vertices()  # 12 vertices
    # Center
    center = np.zeros(1).reshape(1, 3)
    
    # Shell 1: 12 atoms (icosahedron vertices) at radius r1
    r1 = 1.0
    shell1 = ico_verts * r1
    
    # Shell 2: 42 atoms
    # 30 edge midpoints + 12 face centers
    phi = (1 + np.sqrt(5)) / 2
    r2 = np.sqrt(1 + 1/phi**2)
    
    # Find edges (pairs with minimum distance)
    dists = np.zeros((12, 12))
    for i in range(12):
        for j in range(i+1, 12):
            dists[i, j] = dists[j, i] = np.linalg.norm(ico_verts[i] - ico_verts[j])
    min_d = np.min(dists[dists > 0])
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            if abs(dists[i, j] - min_d) < 1e-6:
                edges.append((i, j))
    
    # Edge midpoints (30)
    edge_mids = []
    for i, j in edges:
        mid = (ico_verts[i] + ico_verts[j]) / 2
        mid /= np.linalg.norm(mid)
        edge_mids.append(mid)
    edge_mids = np.array(edge_mids) * r2
    
    # Face centers: 20 faces, pick 12 that form a dodecahedron
    # Dodecahedron vertices correspond to 12 faces of icosahedron
    dodeca_verts = _get_dodecahedron_vertices()
    
    # Find all faces
    faces = []
    for i in range(12):
        for j in range(i+1, 12):
            for k in range(j+1, 12):
                d1 = np.linalg.norm(ico_verts[i] - ico_verts[j])
                d2 = np.linalg.norm(ico_verts[j] - ico_verts[k])
                d3 = np.linalg.norm(ico_verts[i] - ico_verts[k])
                if abs(d1-min_d)<1e-6 and abs(d2-min_d)<1e-6 and abs(d3-min_d)<1e-6:
                    faces.append((i,j,k))
    
    face_centers = []
    for i,j,k in faces:
        fc = (ico_verts[i] + ico_verts[j] + ico_verts[k]) / 3
        fc /= np.linalg.norm(fc)
        face_centers.append(fc)
    face_centers = np.array(face_centers)
    
    # Match dodecahedron vertices to face centers
    selected_fc = []
    used = set()
    for dv in dodeca_verts:
        dists_to_fc = np.linalg.norm(face_centers - dv, axis=1)
        idx = np.argmin(dists_to_fc)
        if idx not in used:
            selected_fc.append(face_centers[idx])
            used.add(idx)
    if len(selected_fc) < 12:
        for i in range(20):
            if i not in used and len(selected_fc) < 12:
                selected_fc.append(face_centers[i])
                used.add(i)
    selected_fc = np.array(selected_fc[:12]) * r2
    
    shell2 = np.vstack([edge_mids, selected_fc])
    return np.vstack([center, shell1, shell2])


def _generate_random_compact(n, rng):
    """Random compact configuration with minimum distance constraint."""
    radius = 1.5 * n ** (1/3)
    points = []
    attempts = 0
    while len(points) < n and attempts < n * 2000:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) > radius:
            attempts += 1
            continue
        ok = True
        for q in points:
            if np.linalg.norm(p - q) < 0.55:
                ok = False
                break
        if ok:
            points.append(p)
            attempts += 1
        else:
            attempts += 1
    while len(points) < n:
        points.append(rng.uniform(-radius*1.2, radius*1.2, 3))
    return np.array(points)


def _generate_defect_seeded(n, rng, mackay55):
    """Generate seeds by removing atoms from Mackay 55 or adding to it."""
    seeds = []
    
    if n == 55:
        seeds.append(mackay55.copy())
        for _ in range(4):
            seeds.append(mackay55 + rng.normal(0, 0.08, size=(55, 3)))
        return seeds
    
    if n < 55:
        n_remove = 55 - n
        # Preferentially remove from outer shell
        outer_idx = list(range(13, 55))
        center_and_inner = list(range(0, 13))
        
        for _ in range(6):
            # Random removal from outer shell
            remove = rng.choice(outer_idx, size=n_remove, replace=False)
            mask = np.ones(55, dtype=bool)
            mask[remove] = False
            seeds.append(mackay55[mask])
            
            # Removal biased toward outer shell (prefer outer)
            probs = np.zeros(55)
            probs[0] = 0.1
            probs[1:13] = 0.2
            probs[13:] = 1.0
            probs /= probs.sum()
            remove2 = np.random.choice(55, size=n_remove, replace=False, p=probs)
            mask2 = np.ones(55, dtype=bool)
            mask2[remove2] = False
            seeds.append(mackay55[mask2])
        
        return seeds
    else:  # n > 55
        n_add = n - 55
        base = mackay55.copy()
        r_outer = np.max(np.linalg.norm(base, axis=1))
        
        for _ in range(6):
            # Add atoms on a sphere
            r_new = r_outer * (1.05 + 0.1 * rng.random())
            theta = rng.uniform(0, 2*np.pi, n_add)
            phi = np.arccos(rng.uniform(-1, 1, n_add))
            new_pos = np.column_stack([
                r_new * np.sin(phi) * np.cos(theta),
                r_new * np.sin(phi) * np.sin(theta),
                r_new * np.cos(phi)
            ])
            seeds.append(np.vstack([base, new_pos]))
            
            # Add near existing surface atoms
            new_atoms = []
            surface_idx = np.where(np.linalg.norm(base, axis=1) > r_outer * 0.9)[0]
            for _ in range(n_add):
                ref = base[rng.choice(surface_idx)]
                new_atom = ref + rng.normal(0, 0.4, 3)
                new_atoms.append(new_atom)
            new_pos = np.array(new_atoms)
            seeds.append(np.vstack([base, new_pos]))
        
        return seeds


def _basin_hopping_single(pos, n, rng, deadline, step0=0.25, T0=0.15, T_min=0.005):
    """Basin hopping: perturb, relax, Metropolis accept."""
    current = pos.copy()
    try:
        e_cur, _ = relax(current, POTENTIAL)
    except:
        return pos, float('inf')
    best = current.copy()
    e_best = e_cur
    T = T0
    step = step0
    
    while time.time() < deadline - 2:
        perturb = rng.normal(0, step, size=(n, 3))
        new_pos = current + perturb
        try:
            e_new, new_relaxed = relax(new_pos, POTENTIAL)
        except:
            continue
        
        if e_new < e_best:
            e_best = e_new
            best = new_relaxed.copy()
            current = new_relaxed.copy()
            step *= 0.97
        else:
            delta = e_new - e_cur
            if delta < 0 or rng.random() < np.exp(-delta / T):
                current = new_relaxed.copy()
                e_cur = e_new
                step *= 0.99
            else:
                step = min(step * 1.08, 1.0)
        
        T = max(T * 0.995, T_min)
    
    return best, e_best


def _parallel_search(seeds, n, deadline, max_workers=4):
    """Run basin hopping in parallel on multiple seeds."""
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for i, seed in enumerate(seeds):
            if time.time() > deadline - 10:
                break
            rng = np.random.default_rng(12345 + i)
            fut = executor.submit(_basin_hopping_single, seed, n, rng, deadline - 5)
            futures[fut] = i
        
        for fut in as_completed(futures):
            try:
                pos, e = fut.result()
                results.append((e, pos))
            except:
                pass
    
    results.sort(key=lambda x: x[0])
    return results


def construct(instance):
    global _GLOBAL_START, _GLOBAL_DEADLINE, _GLOBAL_SEEN, _GLOBAL_BEST, _GLOBAL_BEST_ENERGY
    
    if _GLOBAL_START is None:
        _GLOBAL_START = time.time()
        _GLOBAL_DEADLINE = _GLOBAL_START + 285.0  # Leave 15s margin
    
    n = instance['n']
    start = time.time()
    instance_deadline = min(start + 40, _GLOBAL_DEADLINE - 3)
    
    rng = np.random.default_rng(42 + n * 7)
    
    # Generate reference Mackay 55
    try:
        mackay55 = _generate_mackay_55()
    except:
        mackay55 = None
    
    # Check if we've already seen this size with a good result
    if n in _GLOBAL_BEST_ENERGY and _GLOBAL_BEST_ENERGY[n] < 0:
        # We have a previous result; still try to improve
        pass
    
    # Generate diverse seeds
    seeds = []
    
    # Random compact seeds
    for i in range(3):
        seeds.append(_generate_random_compact(n, rng))
    
    # Mackay-based seeds
    if mackay55 is not None:
        mackay_seeds = _generate_defect_seeded(n, rng, mackay55)
        seeds.extend(mackay_seeds)
    
    # FCC-like seeds
    for i in range(2):
        # Generate FCC cluster
        pts = []
        # Simple FCC-like arrangement
        a0 = 0.8
        for ix in range(3):
            for iy in range(3):
                for iz in range(3):
                    if ix + iy + iz < 6:
                        pt = np.array([ix*a0, iy*a0, iz*a0])
                        # FCC face-centered positions
                        if ix + iy + iz % 2 == 0:
                            pts.append(pt)
        # Add face-centered positions
        for ix in range(2):
            for iy in range(2):
                for iz in range(2):
                    pts.append(np.array([(ix+0.5)*a0, (iy+0.5)*a0, (iz+0.5)*a0]))
        pts = np.array(pts[:n]) if len(pts) >= n else np.vstack([pts, _generate_random_compact(n - len(pts), rng)])
        pts += rng.normal(0, 0.1, size=pts.shape)
        seeds.append(pts)
    
    # Run parallel basin hopping
    if time.time() < instance_deadline - 15:
        results = _parallel_search(seeds, n, instance_deadline, max_workers=4)
        if results:
            best_e, best_pos = results[0]
        else:
            best_e = float('inf')
            best_pos = seeds[0]
    else:
        best_e = float('inf')
        best_pos = seeds[0]
        try:
            best_e, best_pos = relax(best_pos, POTENTIAL)
        except:
            pass
    
    # Final refinement on the best
    if time.time() < instance_deadline - 3:
        rng2 = np.random.default_rng(999 + n)
        try:
            refined_e, refined_pos = _basin_hopping_single(best_pos, n, rng2, instance_deadline - 1)
            if refined_e < best_e:
                best_e, best_pos = refined_e, refined_pos
        except:
            pass
    
    # Ensure minimum distance
    # Check and fix any violations
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(best_pos[i] - best_pos[j])
            if d < 0.5 and d > 1e-10:
                # Push apart
                direction = (best_pos[i] - best_pos[j]) / d
                push = (0.5 - d) / 2
                best_pos[i] += direction * push
                best_pos[j] -= direction * push
            elif d <= 1e-10:
                # Coincident atoms, separate randomly
                offset = rng.normal(0, 0.3, 3)
                best_pos[j] += offset
    
    # Final relax if time permits
    if time.time() < _GLOBAL_DEADLINE - 5:
        try:
            final_e, final_pos = relax(best_pos, POTENTIAL)
            if final_e < best_e:
                best_e, best_pos = final_e, final_pos
        except:
            pass
    
    # Update global best
    if n not in _GLOBAL_BEST or best_e < _GLOBAL_BEST_ENERGY[n]:
        _GLOBAL_BEST[n] = best_pos.copy()
        _GLOBAL_BEST_ENERGY[n] = best_e
    
    return best_pos
