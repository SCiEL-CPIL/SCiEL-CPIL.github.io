"""Dzugutov cluster global minima: parallel basin hopping with softness annealing."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import concurrent.futures

# Dzugutov potential parameters
A = 5.82
B = 1.28
a = 1.87
b = 1.94
c = 1.10
d = 0.27
m = 16.0

# Global time tracker
_start_time = None
TIME_LIMIT = 300.0
SAFETY_MARGIN = 25.0

def _time_left():
    global _start_time
    if _start_time is None:
        return TIME_LIMIT
    return (TIME_LIMIT - SAFETY_MARGIN) - (time.time() - _start_time)

# Pre-computed pair indices cache
_pair_cache = {}

def _get_pair_indices(n):
    if n not in _pair_cache:
        iu = np.triu_indices(n, k=1)
        _pair_cache[n] = iu
    return _pair_cache[n]

def _dzugutov_energy(positions, c_val=None, d_val=None):
    """Compute Dzugutov potential energy with optional annealed softness parameters."""
    if c_val is None:
        c_val = c
    if d_val is None:
        d_val = d
    n = len(positions)
    if n < 2:
        return 0.0
    iu = _get_pair_indices(n)
    diff = positions[iu[0]] - positions[iu[1]]
    r2_pair = np.sum(diff ** 2, axis=1)
    energy = 0.0
    valid = r2_pair > 1e-12
    if not np.any(valid):
        return 0.0
    r = np.sqrt(r2_pair[valid])
    mask1 = r < a
    if np.any(mask1):
        r1 = r[mask1]
        energy += np.sum(A * (r1 ** (-m) - B) * np.exp(c_val / (r1 - a)))
    mask2 = (r >= a) & (r < b)
    if np.any(mask2):
        r2_ = r[mask2]
        energy += np.sum(B * np.exp(d_val / (r2_ - b)))
    return float(energy)

def _dzugutov_gradient(positions, c_val=None, d_val=None):
    """Compute gradient with optional annealed softness parameters."""
    if c_val is None:
        c_val = c
    if d_val is None:
        d_val = d
    n = len(positions)
    grad = np.zeros_like(positions)
    if n < 2:
        return grad
    iu = _get_pair_indices(n)
    i_idx, j_idx = iu[0], iu[1]
    diff = positions[i_idx] - positions[j_idx]
    r2_pair = np.sum(diff ** 2, axis=1)
    valid = r2_pair > 1e-12
    if not np.any(valid):
        return grad
    i_idx_v, j_idx_v = i_idx[valid], j_idx[valid]
    r = np.sqrt(r2_pair[valid])
    rij = diff[valid]
    dV_dr = np.zeros_like(r)
    mask1 = r < a
    if np.any(mask1):
        r1 = r[mask1]
        exp_term = np.exp(c_val / (r1 - a))
        dV_dr[mask1] = A * (-m * r1 ** (-m - 1)) * exp_term + A * (r1 ** (-m) - B) * exp_term * (-c_val / (r1 - a) ** 2)
    mask2 = (r >= a) & (r < b)
    if np.any(mask2):
        r2_ = r[mask2]
        dV_dr[mask2] = B * np.exp(d_val / (r2_ - b)) * (-d_val / (r2_ - b) ** 2)
    factor = -dV_dr / r
    forces = factor[:, np.newaxis] * rij
    np.add.at(grad, i_idx_v, forces)
    np.add.at(grad, j_idx_v, -forces)
    return grad

def _relax(pos, max_iter=200, ftol=1e-10, gtol=1e-8, c_val=None, d_val=None):
    """Relax structure using L-BFGS-B."""
    n = len(pos)
    x0 = pos.flatten()
    def fun(x):
        return _dzugutov_energy(x.reshape(n, 3), c_val, d_val)
    def jac(x):
        return _dzugutov_gradient(x.reshape(n, 3), c_val, d_val).flatten()
    try:
        res = minimize(fun, x0, jac=jac, method='L-BFGS-B',
                       options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol})
        return res.x.reshape(n, 3), float(res.fun)
    except Exception:
        return pos, _dzugutov_energy(pos, c_val, d_val)

def _check_min_dist(pos, min_r=0.5):
    if len(pos) < 2:
        return True
    iu = _get_pair_indices(len(pos))
    diff = pos[iu[0]] - pos[iu[1]]
    r2 = np.sum(diff ** 2, axis=1)
    return np.min(r2) >= min_r * min_r - 1e-9

# --- Geometry generators ---

def _icosahedron_dirs():
    phi = (1 + np.sqrt(5)) / 2
    v = np.array([[0,1,phi],[0,-1,phi],[0,1,-phi],[0,-1,-phi],
                  [1,phi,0],[-1,phi,0],[1,-phi,0],[-1,-phi,0],
                  [phi,0,1],[phi,0,-1],[-phi,0,1],[-phi,0,-1]], dtype=float)
    return v / np.linalg.norm(v, axis=1)[:, None]

def _dodecahedron_dirs():
    phi = (1 + np.sqrt(5)) / 2
    iphi = 1.0 / phi
    v = np.array([
        [1,1,1],[1,1,-1],[1,-1,1],[1,-1,-1],
        [-1,1,1],[-1,1,-1],[-1,-1,1],[-1,-1,-1],
        [0,iphi,phi],[0,iphi,-phi],[0,-iphi,phi],[0,-iphi,-phi],
        [iphi,phi,0],[iphi,-phi,0],[-iphi,phi,0],[-iphi,-phi,0],
        [phi,0,iphi],[phi,0,-iphi],[-phi,0,iphi],[-phi,0,-iphi]
    ], dtype=float)
    return v / np.linalg.norm(v, axis=1)[:, None]

def _icosahedron_edge_dirs():
    v = _icosahedron_dirs()
    min_d = float('inf')
    for i in range(12):
        for j in range(i+1, 12):
            dd = np.linalg.norm(v[i]-v[j])
            if dd < min_d:
                min_d = dd
    dirs = []
    for i in range(12):
        for j in range(i+1, 12):
            if abs(np.linalg.norm(v[i]-v[j]) - min_d) < 1e-9:
                mid = (v[i]+v[j])/2
                dirs.append(mid/np.linalg.norm(mid))
    return np.array(dirs)

def _icosahedron_face_dirs():
    v = _icosahedron_dirs()
    min_d = float('inf')
    for i in range(12):
        for j in range(i+1, 12):
            dd = np.linalg.norm(v[i]-v[j])
            if dd < min_d:
                min_d = dd
    adj = {i: [] for i in range(12)}
    for i in range(12):
        for j in range(i+1, 12):
            if abs(np.linalg.norm(v[i]-v[j]) - min_d) < 1e-9:
                adj[i].append(j)
                adj[j].append(i)
    faces = []
    seen = set()
    for i in range(12):
        for j in adj[i]:
            if j <= i:
                continue
            for k in adj[j]:
                if k <= j or k == i:
                    continue
                if k in adj[i]:
                    key = tuple(sorted([i,j,k]))
                    if key not in seen:
                        seen.add(key)
                        c = (v[i]+v[j]+v[k])/3
                        faces.append(c/np.linalg.norm(c))
    return np.array(faces)

def _generate_initial_structures(n, rng):
    """Generate diverse initial structures for n atoms."""
    candidates = []
    ico_dirs = _icosahedron_dirs()
    dod_dirs = _dodecahedron_dirs()
    edge_dirs = _icosahedron_edge_dirs()
    face_dirs = _icosahedron_face_dirs()
    
    # --- Family 1: Mackay icosahedra ---
    for r1 in [1.30, 1.35, 1.40, 1.45, 1.50, 1.55, 1.60]:
        for r2 in [2.20, 2.30, 2.40, 2.50, 2.60, 2.70]:
            for r3 in [2.20, 2.30, 2.40, 2.50, 2.60, 2.70]:
                pts = [np.zeros(3)]
                pts.extend((ico_dirs * r1).tolist())
                pts.extend((edge_dirs * r2).tolist())
                pts.extend((ico_dirs * r3).tolist())
                pts.extend((face_dirs * r3).tolist())
                pts = np.array(pts)
                if len(pts) >= n:
                    candidates.append(pts[:n])
                else:
                    extra = n - len(pts)
                    max_r = np.max(np.linalg.norm(pts, axis=1))
                    pad = rng.random((extra, 3)) * 2 - 1
                    pad = pad / (np.linalg.norm(pad, axis=1)[:, None] + 1e-12) * (max_r + 0.3)
                    candidates.append(np.vstack([pts, pad]))
    
    # --- Family 2: Dodecahedral shells ---
    for r1 in [1.30, 1.40, 1.50, 1.60]:
        for r2 in [2.20, 2.40, 2.60, 2.80]:
            pts = [np.zeros(3)]
            pts.extend((dod_dirs * r1).tolist())
            pts.extend((dod_dirs * r2).tolist())
            pts = np.array(pts)
            if len(pts) >= n:
                candidates.append(pts[:n])
            else:
                extra = n - len(pts)
                max_r = np.max(np.linalg.norm(pts, axis=1))
                pad = rng.random((extra, 3)) * 2 - 1
                pad = pad / (np.linalg.norm(pad, axis=1)[:, None] + 1e-12) * (max_r + 0.3)
                candidates.append(np.vstack([pts, pad]))
    
    # --- Family 3: Icosahedral shells ---
    for r1 in [1.30, 1.40, 1.50, 1.60]:
        for r2 in [2.20, 2.40, 2.60, 2.80]:
            for r3 in [2.20, 2.40, 2.60, 2.80]:
                pts = [np.zeros(3)]
                pts.extend((ico_dirs * r1).tolist())
                pts.extend((edge_dirs * r2).tolist())
                pts.extend((ico_dirs * r3).tolist())
                pts.extend((face_dirs * r3).tolist())
                pts = np.array(pts)
                if len(pts) >= n:
                    candidates.append(pts[:n])
                else:
                    extra = n - len(pts)
                    max_r = np.max(np.linalg.norm(pts, axis=1))
                    pad = rng.random((extra, 3)) * 2 - 1
                    pad = pad / (np.linalg.norm(pad, axis=1)[:, None] + 1e-12) * (max_r + 0.3)
                    candidates.append(np.vstack([pts, pad]))
    
    # --- Family 4: Random shells with icosahedral symmetry hints ---
    for _ in range(16):
        r_inner = 1.2 + rng.random() * 0.6
        r_outer = 2.0 + rng.random() * 1.0
        pts = [np.zeros(3)]
        inner = ico_dirs * r_inner
        pts.extend(inner.tolist())
        edges = edge_dirs * r_outer
        pts.extend(edges.tolist())
        remaining = n - len(pts)
        if remaining > 0:
            pad = rng.random((remaining, 3)) * 2 - 1
            pad = pad / (np.linalg.norm(pad, axis=1)[:, None] + 1e-12) * r_outer
            pts.extend(pad.tolist())
        pts = np.array(pts)
        candidates.append(pts[:n])
    
    # --- Family 5: Pure random (diversity) ---
    for _ in range(12):
        pts = rng.normal(0, 1.5, (n, 3))
        candidates.append(pts)
    
    # Deduplicate and filter
    unique = []
    for c in candidates:
        if c.shape != (n, 3):
            continue
        if not _check_min_dist(c):
            continue
        unique.append(c)
    
    return unique

def _parallel_relax(candidates, max_iter=150, n_threads=4):
    """Relax multiple candidates in parallel using thread pool."""
    if not candidates:
        return []
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = {executor.submit(_relax, c, max_iter): c for c in candidates}
        for future in concurrent.futures.as_completed(futures):
            try:
                pos, e = future.result()
                if _check_min_dist(pos):
                    results.append((e, pos))
            except Exception:
                continue
    results.sort(key=lambda x: x[0])
    return results

def _basin_hopping_annealed(pos, n_steps, sigma_start, sigma_end, rng, max_iter_per_relax=150):
    """Basin hopping with annealed softness parameters for better exploration."""
    n = len(pos)
    best_pos = pos.copy()
    best_energy = _dzugutov_energy(pos)
    current_pos = pos.copy()
    current_energy = best_energy
    
    for step in range(n_steps):
        if _time_left() < 10:
            break
        
        # Anneal sigma and softness parameters
        frac = step / max(n_steps - 1, 1)
        sigma = sigma_start + (sigma_end - sigma_start) * frac
        # Start with softer potential, end with exact
        c_anneal = c * (1.0 + 0.5 * (1.0 - frac))  # Softer at start
        d_anneal = d * (1.0 + 0.3 * (1.0 - frac))
        
        perturbed = current_pos + rng.normal(0, sigma, (n, 3))
        
        # Relax with annealed potential
        relaxed, e_relaxed = _relax(perturbed, max_iter=max_iter_per_relax, 
                                     c_val=c_anneal, d_val=d_anneal)
        
        # Recompute energy with exact potential
        e_exact = _dzugutov_energy(relaxed)
        
        if not _check_min_dist(relaxed):
            continue
        
        if e_exact < best_energy:
            best_energy = e_exact
            best_pos = relaxed.copy()
        
        current_pos = relaxed
        current_energy = e_exact
    
    return best_pos, best_energy

def _polish(pos, max_iter=1000, ftol=1e-14, gtol=1e-12):
    """High-precision final polish."""
    n = len(pos)
    x0 = pos.flatten()
    def fun(x):
        return _dzugutov_energy(x.reshape(n, 3))
    def jac(x):
        return _dzugutov_gradient(x.reshape(n, 3)).flatten()
    try:
        res = minimize(fun, x0, jac=jac, method='L-BFGS-B',
                       options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol})
        return res.x.reshape(n, 3), float(res.fun)
    except Exception:
        return pos, _dzugutov_energy(pos)

def construct(instance):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    n = instance["n"]
    
    if _time_left() < 5:
        return np.random.randn(n, 3) * 0.5
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initial structures
    candidates = _generate_initial_structures(n, rng)
    
    if not candidates:
        return np.random.randn(n, 3) * 0.5
    
    # Score initial candidates (quick energy eval)
    scored = []
    for c in candidates:
        try:
            e = _dzugutov_energy(c)
            scored.append((e, c))
        except Exception:
            continue
    
    scored.sort(key=lambda x: x[0])
    
    best_energy = float('inf')
    best_pos = None
    
    # Phase 1: Parallel relaxation of top candidates
    n_top = min(len(scored), 20)
    if _time_left() > 20:
        top_candidates = [pos for e, pos in scored[:n_top]]
        relaxed_results = _parallel_relax(top_candidates, max_iter=150, n_threads=4)
        if relaxed_results:
            best_energy = relaxed_results[0][0]
            best_pos = relaxed_results[0][1].copy()
    
    if best_pos is None and scored:
        best_pos = scored[0][1].copy()
        best_energy = _dzugutov_energy(best_pos)
    
    # Phase 2: Basin hopping with annealing from multiple starts
    if best_pos is not None and _time_left() > 40:
        bh_starts = [best_pos]
        # Add a few other good starts
        for e_s, p_s in scored[1:min(8, len(scored))]:
            if _check_min_dist(p_s):
                bh_starts.append(p_s)
        # Add perturbed versions of best
        for _ in range(4):
            bh_starts.append(best_pos + rng.normal(0, 0.2, (n, 3)))
        
        for start in bh_starts:
            if _time_left() < 20:
                break
            try:
                bh_pos, bh_e = _basin_hopping_annealed(start, n_steps=12, 
                                                        sigma_start=0.20, sigma_end=0.10,
                                                        rng=rng, max_iter_per_relax=130)
                if _check_min_dist(bh_pos) and bh_e < best_energy:
                    best_energy = bh_e
                    best_pos = bh_pos
            except Exception:
                continue
    
    # Phase 3: More basin hopping if time remains
    if best_pos is not None and _time_left() > 30:
        try:
            bh_pos, bh_e = _basin_hopping_annealed(best_pos, n_steps=15, 
                                                    sigma_start=0.15, sigma_end=0.08,
                                                    rng=rng, max_iter_per_relax=150)
            if _check_min_dist(bh_pos) and bh_e < best_energy:
                best_energy = bh_e
                best_pos = bh_pos
        except Exception:
            pass
    
    # Phase 4: Final high-precision polish
    if best_pos is not None and _time_left() > 10:
        try:
            polished, e_pol = _polish(best_pos, max_iter=1000, ftol=1e-14, gtol=1e-12)
            if _check_min_dist(polished) and e_pol < best_energy:
                best_energy = e_pol
                best_pos = polished
        except Exception:
            pass
    
    if best_pos is None:
        best_pos = np.random.randn(n, 3) * 0.5
    
    return best_pos

# EVOLVE-BLOCK-END
