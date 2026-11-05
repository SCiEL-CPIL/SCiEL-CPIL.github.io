"""Dzugutov cluster global minima: topological optimization with polytetrahedral constraints."""
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
SAFETY_MARGIN = 20.0

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

def _check_min_dist(pos, min_r=0.5):
    if len(pos) < 2:
        return True
    iu = _get_pair_indices(len(pos))
    diff = pos[iu[0]] - pos[iu[1]]
    r2 = np.sum(diff ** 2, axis=1)
    return np.min(r2) >= min_r * min_r - 1e-9

def _relax(pos, max_iter=300, ftol=1e-12, gtol=1e-10):
    """Relax structure using L-BFGS-B."""
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

# --- Proper Mackay icosahedron construction ---

def _icosahedron_vertices():
    """Return unit vectors to icosahedron vertices."""
    phi = (1 + np.sqrt(5)) / 2
    v = np.array([[0,1,phi],[0,-1,phi],[0,1,-phi],[0,-1,-phi],
                  [1,phi,0],[-1,phi,0],[1,-phi,0],[-1,-phi,0],
                  [phi,0,1],[phi,0,-1],[-phi,0,1],[-phi,0,-1]], dtype=float)
    return v / np.linalg.norm(v, axis=1)[:, None]

def _icosahedron_edge_midpoints():
    """Return unit vectors to icosahedron edge midpoints (30 edges)."""
    v = _icosahedron_vertices()
    # Find all edges (pairs at minimum distance)
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

def _icosahedron_face_centers():
    """Return unit vectors to icosahedron face centers (20 faces)."""
    v = _icosahedron_vertices()
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

def _build_mackay_icosahedron(n, rng):
    """Build a proper Mackay icosahedral cluster with exactly n atoms.
    
    Uses the standard shell structure:
    - Shell 0: 1 atom at center
    - Shell 1: 12 atoms at icosahedron vertices
    - Shell 2: 30 atoms at edge midpoints
    - Shell 3: 12 atoms at icosahedron vertices (outer)
    - Shell 4: 30 atoms at edge midpoints (outer)
    - etc.
    
    For n atoms, we fill shells in order: 1, 12, 30, 12, 30, ...
    """
    ico_v = _icosahedron_vertices()
    ico_e = _icosahedron_edge_midpoints()
    ico_f = _icosahedron_face_centers()
    
    # Optimal radii based on Dzugutov potential (first minimum ~1.4, second ~2.4)
    r1 = 1.40  # first shell
    r2 = 2.35  # second shell  
    r3 = 3.30  # third shell
    
    pts = [np.zeros(3)]  # center
    
    # Fill shell 1 (12 vertices)
    if n >= 13:
        pts.extend((ico_v * r1).tolist())
        n_used = 13
    else:
        # For small n, just use center + some vertices
        n_needed = n - 1
        pts.extend((ico_v[:n_needed] * r1).tolist())
        return np.array(pts)
    
    # Fill shell 2 (30 edge midpoints)
    if n >= 43:
        pts.extend((ico_e * r2).tolist())
        n_used = 43
    elif n > 13:
        # Partial second shell
        n_needed = n - 13
        pts.extend((ico_e[:n_needed] * r2).tolist())
        return np.array(pts)
    
    # Fill shell 3 (12 vertices at larger radius)
    if n >= 55:
        pts.extend((ico_v * r3).tolist())
        n_used = 55
    elif n > 43:
        # Partial third shell
        n_needed = n - 43
        pts.extend((ico_v[:n_needed] * r3).tolist())
        return np.array(pts)
    
    # Fill shell 4 (30 edge midpoints at larger radius)
    if n >= 85:
        pts.extend((ico_e * r3).tolist())
        n_used = 85
    elif n > 55:
        # Partial fourth shell
        n_needed = n - 55
        pts.extend((ico_e[:n_needed] * r3).tolist())
        return np.array(pts)
    
    # If we still need more, add face centers or random
    if len(pts) < n:
        n_needed = n - len(pts)
        # Add face centers first
        face_pts = (ico_f * r3).tolist()
        take = min(n_needed, len(face_pts))
        pts.extend(face_pts[:take])
        n_needed -= take
        if n_needed > 0:
            # Random on outer sphere
            theta = rng.uniform(0, np.pi, n_needed)
            phi = rng.uniform(0, 2*np.pi, n_needed)
            r_shell = r3 + 0.5
            x = r_shell * np.sin(theta) * np.cos(phi)
            y = r_shell * np.sin(theta) * np.sin(phi)
            z = r_shell * np.cos(theta)
            pts.extend(np.column_stack([x, y, z]).tolist())
    
    result = np.array(pts[:n])
    return result

def _build_varied_mackay(n, rng):
    """Build multiple varied Mackay icosahedra with slightly different radii."""
    candidates = []
    ico_v = _icosahedron_vertices()
    ico_e = _icosahedron_edge_midpoints()
    ico_f = _icosahedron_face_centers()
    
    # Vary the radii
    r1_vals = [1.35, 1.40, 1.45, 1.50]
    r2_vals = [2.25, 2.35, 2.45, 2.55]
    r3_vals = [3.20, 3.30, 3.40]
    
    for r1 in r1_vals:
        for r2 in r2_vals[:2]:
            for r3 in r3_vals[:2]:
                pts = [np.zeros(3)]
                if n >= 13:
                    pts.extend((ico_v * r1).tolist())
                    n_used = 13
                else:
                    n_needed = n - 1
                    pts.extend((ico_v[:n_needed] * r1).tolist())
                    candidates.append(np.array(pts))
                    continue
                
                if n >= 43:
                    pts.extend((ico_e * r2).tolist())
                    n_used = 43
                elif n > 13:
                    n_needed = n - 13
                    pts.extend((ico_e[:n_needed] * r2).tolist())
                    candidates.append(np.array(pts))
                    continue
                
                if n >= 55:
                    pts.extend((ico_v * r3).tolist())
                    n_used = 55
                elif n > 43:
                    n_needed = n - 43
                    pts.extend((ico_v[:n_needed] * r3).tolist())
                    candidates.append(np.array(pts))
                    continue
                
                if n >= 85:
                    pts.extend((ico_e * r3).tolist())
                elif n > 55:
                    n_needed = n - 55
                    pts.extend((ico_e[:n_needed] * r3).tolist())
                
                if len(pts) < n:
                    n_needed = n - len(pts)
                    face_pts = (ico_f * r3).tolist()
                    take = min(n_needed, len(face_pts))
                    pts.extend(face_pts[:take])
                    n_needed -= take
                    if n_needed > 0:
                        theta = rng.uniform(0, np.pi, n_needed)
                        phi = rng.uniform(0, 2*np.pi, n_needed)
                        r_shell = r3 + 0.5
                        x = r_shell * np.sin(theta) * np.cos(phi)
                        y = r_shell * np.sin(theta) * np.sin(phi)
                        z = r_shell * np.cos(theta)
                        pts.extend(np.column_stack([x, y, z]).tolist())
                
                result = np.array(pts[:n])
                if _check_min_dist(result):
                    candidates.append(result)
    
    return candidates

def _build_random_compact(n, rng):
    """Build random compact structures as diversity fallback."""
    candidates = []
    for _ in range(5):
        # Random points in a ball
        pts = rng.normal(0, 1.0, (n, 3))
        # Scale to ensure compactness
        norms = np.linalg.norm(pts, axis=1)
        max_n = np.max(norms)
        if max_n > 0:
            pts = pts / max_n * 2.5
        if _check_min_dist(pts):
            candidates.append(pts)
    return candidates

def _topological_barrier(positions, target_bonds, barrier_height=50.0, barrier_width=0.3):
    """Add a barrier that penalizes breaking target bonds or forming non-target bonds.
    
    target_bonds: list of (i, j) pairs that should be bonded
    """
    n = len(positions)
    iu = _get_pair_indices(n)
    diff = positions[iu[0]] - positions[iu[1]]
    r2_pair = np.sum(diff ** 2, axis=1)
    r = np.sqrt(r2_pair)
    
    barrier = 0.0
    # Penalize atoms that should be bonded but aren't
    bond_set = set()
    for i, j in target_bonds:
        bond_set.add((min(i,j), max(i,j)))
    
    # For each target bond, if distance is too large, add penalty
    for i, j in target_bonds:
        idx_i, idx_j = min(i,j), max(i,j)
        # Find the pair index
        # Simple approach: compute distance directly
        d = np.linalg.norm(positions[i] - positions[j])
        if d > 1.8:
            barrier += barrier_height * np.exp(-(d - 1.8) / barrier_width)
    
    return barrier

def _polish(pos, max_iter=2000, ftol=1e-15, gtol=1e-13):
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

def _basin_hopping_single(start_pos, n_steps=15, sigma_start=0.15, sigma_end=0.05,
                          max_iter_per_relax=200, rng=None):
    """Single basin hopping trajectory."""
    if rng is None:
        rng = np.random.default_rng()
    
    best_pos = start_pos.copy()
    best_energy = _dzugutov_energy(start_pos)
    current_pos = start_pos.copy()
    current_energy = best_energy
    
    for step in range(n_steps):
        if _time_left() < 10:
            break
        
        frac = step / max(n_steps - 1, 1)
        sigma = sigma_start + (sigma_end - sigma_start) * frac
        
        perturbed = current_pos + rng.normal(0, sigma, (len(start_pos), 3))
        
        relaxed, e_relaxed = _relax(perturbed, max_iter=max_iter_per_relax)
        
        if not _check_min_dist(relaxed):
            continue
        
        e_exact = _dzugutov_energy(relaxed)
        
        if e_exact < best_energy:
            best_energy = e_exact
            best_pos = relaxed.copy()
        
        current_pos = relaxed
        current_energy = e_exact
    
    return best_pos, best_energy

def _parallel_basin_hopping(starts, n_steps_each=15, sigma_start=0.15, sigma_end=0.05,
                            n_threads=4, max_iter_per_relax=200, rng=None):
    """Parallel basin hopping from multiple starting points."""
    if rng is None:
        rng = np.random.default_rng()
    
    def _bh_wrapper(start_pos, seed_offset):
        local_rng = np.random.default_rng(rng.integers(0, 2**31) + seed_offset)
        return _basin_hopping_single(start_pos, n_steps_each, sigma_start, sigma_end,
                                      max_iter_per_relax, local_rng)
    
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = {executor.submit(_bh_wrapper, s, i): i for i, s in enumerate(starts)}
        for future in concurrent.futures.as_completed(futures):
            try:
                pos, e = future.result()
                if _check_min_dist(pos):
                    results.append((e, pos))
            except Exception:
                continue
    
    results.sort(key=lambda x: x[0])
    return results

def construct(instance):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    n = instance["n"]
    
    if _time_left() < 5:
        return np.random.randn(n, 3) * 0.5
    
    rng = np.random.default_rng(42)
    
    # Generate initial structures: primarily proper Mackay icosahedra
    candidates = _build_varied_mackay(n, rng)
    
    # Add a few random compact structures for diversity
    candidates.extend(_build_random_compact(n, rng))
    
    if not candidates:
        return np.random.randn(n, 3) * 0.5
    
    # Score and relax top candidates
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
    
    # Phase 1: Relax top candidates in parallel
    n_top = min(len(scored), 8)
    if _time_left() > 40:
        top_candidates = [pos for e, pos in scored[:n_top]]
        
        def _relax_wrapper(pos, idx):
            if _time_left() < 5:
                return pos, _dzugutov_energy(pos)
            return _relax(pos, max_iter=300)
        
        relaxed_results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(_relax_wrapper, c, i): i for i, c in enumerate(top_candidates)}
            for future in concurrent.futures.as_completed(futures):
                try:
                    pos, e = future.result()
                    if _check_min_dist(pos):
                        relaxed_results.append((e, pos))
                except Exception:
                    continue
        
        relaxed_results.sort(key=lambda x: x[0])
        if relaxed_results:
            best_energy = relaxed_results[0][0]
            best_pos = relaxed_results[0][1].copy()
    
    if best_pos is None and scored:
        best_pos = scored[0][1].copy()
        best_energy = _dzugutov_energy(best_pos)
    
    # Phase 2: Basin hopping from best and other good starts
    if best_pos is not None and _time_left() > 60:
        bh_starts = [best_pos]
        # Add other good relaxed results
        for e_s, p_s in relaxed_results[1:min(4, len(relaxed_results))]:
            if _check_min_dist(p_s):
                bh_starts.append(p_s)
        # Add perturbed versions
        for _ in range(3):
            perturbed = best_pos + rng.normal(0, 0.20, (n, 3))
            if _check_min_dist(perturbed):
                bh_starts.append(perturbed)
        
        bh_results = _parallel_basin_hopping(bh_starts[:6], n_steps_each=15,
                                              sigma_start=0.15, sigma_end=0.05,
                                              n_threads=4, max_iter_per_relax=200, rng=rng)
        if bh_results:
            if bh_results[0][0] < best_energy:
                best_energy = bh_results[0][0]
                best_pos = bh_results[0][1].copy()
    
    # Phase 3: More basin hopping if time remains
    if best_pos is not None and _time_left() > 40:
        bh_starts2 = [best_pos]
        for _ in range(4):
            perturbed = best_pos + rng.normal(0, 0.15, (n, 3))
            if _check_min_dist(perturbed):
                bh_starts2.append(perturbed)
        
        bh_results2 = _parallel_basin_hopping(bh_starts2[:5], n_steps_each=20,
                                               sigma_start=0.12, sigma_end=0.04,
                                               n_threads=4, max_iter_per_relax=250, rng=rng)
        if bh_results2:
            if bh_results2[0][0] < best_energy:
                best_energy = bh_results2[0][0]
                best_pos = bh_results2[0][1].copy()
    
    # Phase 4: Final high-precision polish
    if best_pos is not None and _time_left() > 15:
        try:
            polished, e_pol = _polish(best_pos, max_iter=2000, ftol=1e-15, gtol=1e-13)
            if _check_min_dist(polished) and e_pol < best_energy:
                best_energy = e_pol
                best_pos = polished
        except Exception:
            pass
    
    # Final validation
    if best_pos is not None:
        if not _check_min_dist(best_pos):
            # Nudge overlapping atoms apart
            for _ in range(100):
                iu = _get_pair_indices(n)
                diff = best_pos[iu[0]] - best_pos[iu[1]]
                r2 = np.sum(diff ** 2, axis=1)
                too_close = r2 < 0.25 - 1e-9
                if not np.any(too_close):
                    break
                for idx in np.where(too_close)[0]:
                    i, j = iu[0][idx], iu[1][idx]
                    d = np.linalg.norm(diff[idx])
                    if d < 1e-12:
                        best_pos[j] += rng.normal(0, 0.1, 3)
                    else:
                        direction = diff[idx] / d
                        best_pos[i] -= direction * 0.05
                        best_pos[j] += direction * 0.05
    
    if best_pos is None:
        best_pos = np.random.randn(n, 3) * 0.5
    
    return best_pos

# EVOLVE-BLOCK-END
