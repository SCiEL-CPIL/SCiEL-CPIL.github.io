"""Sutton-Chen transition-metal cluster global minima: adaptive basin-hopping with structured seeds."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient


def _deadline_ok(t0, margin=35.0, limit=330.0):
    return (time.time() - t0) < (limit - margin)


def _ensure_min_dist(pts, dmin=0.355):
    """Push apart any pair closer than dmin so the result is always valid."""
    pts = np.asarray(pts, dtype=float).copy()
    n = pts.shape[0]
    if n < 2:
        return pts
    for _ in range(500):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                d = pts[i] - pts[j]
                r = np.linalg.norm(d)
                if r < dmin:
                    if r < 1e-9:
                        pts[j] += np.array([dmin, 0.0, 0.0])
                    else:
                        push = (dmin - r) * 0.5
                        pts[i] += push * d / r
                        pts[j] -= push * d / r
                    moved = True
        if not moved:
            break
    return pts


def _fcc_cluster(n, spacing=0.72):
    if n <= 0:
        return np.empty((0, 3))
    basis = np.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]], dtype=float)
    max_shell = int(np.ceil(n ** (1.0 / 3.0))) + 5
    sites = []
    for i in range(-max_shell, max_shell + 1):
        for j in range(-max_shell, max_shell + 1):
            for k in range(-max_shell, max_shell + 1):
                base = spacing * np.array([i, j, k], dtype=float)
                for b in basis:
                    sites.append(base + spacing * b)
    sites = np.array(sites)
    d = np.linalg.norm(sites, axis=1)
    idx = np.argsort(d)[:n]
    return sites[idx]


def _hcp_cluster(n, spacing=0.72):
    if n <= 0:
        return np.empty((0, 3))
    sites = []
    max_shell = int(np.ceil(n ** (1.0 / 3.0))) + 5
    a = spacing
    c = a * np.sqrt(8.0 / 3.0)
    for layer in range(-max_shell, max_shell + 1):
        z = layer * c
        for i in range(-max_shell, max_shell + 1):
            for j in range(-max_shell, max_shell + 1):
                x = a * (i + 0.5 * (j % 2))
                y = a * np.sqrt(3.0) / 2.0 * j
                sites.append([x, y, z])
                sites.append([x + a * 0.5, y + a * np.sqrt(3.0) / 6.0, z])
    sites = np.array(sites)
    d = np.linalg.norm(sites, axis=1)
    idx = np.argsort(d)[:n]
    return sites[idx]


def _icosahedral_cluster(n, spacing=0.72):
    if n <= 0:
        return np.empty((0, 3))
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
    ], dtype=float)
    verts /= np.linalg.norm(verts[0])
    shell_r = [0.0, 1.0, np.sqrt(2.0), np.sqrt(2.0 + 2.0 / phi), np.sqrt(2.0 + 2.0 * phi)]
    cand = [np.array([0.0, 0.0, 0.0])]
    for r in shell_r[1:]:
        for a in verts:
            cand.append(r * a)
        for i in range(len(verts)):
            for j in range(i + 1, len(verts)):
                dv = verts[i] - verts[j]
                if abs(np.linalg.norm(dv) - 2.0) < 1e-6:
                    mid = (verts[i] + verts[j]) / 2.0
                    nm = np.linalg.norm(mid)
                    if nm > 1e-9:
                        cand.append(r * (mid / nm))
    cand = np.array(cand)
    uniq = []
    for p in cand:
        if not any(np.allclose(p, u, atol=1e-6) for u in uniq):
            uniq.append(p)
    cand = np.array(uniq)
    d = np.linalg.norm(cand, axis=1)
    idx = np.argsort(d)[:n]
    return spacing * cand[idx]


def _random_compact(n, spacing, seed):
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * 0.9
    pts = []
    attempts = 0
    while len(pts) < n and attempts < 30000:
        attempts += 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.65 * spacing for q in pts):
            pts.append(p)
    while len(pts) < n:
        p = rng.uniform(-radius, radius, 3)
        pts.append(p)
    return np.array(pts)


def _basin_hopping(pts, potential, parameter, t0, max_iter=40, sigma_start=0.5, sigma_min=0.05):
    """Adaptive basin hopping: perturb, relax, accept if better or with Metropolis."""
    try:
        e_cur, p_cur = relax(pts, potential, parameter)
    except Exception:
        return None, None
    if not np.all(np.isfinite(p_cur)):
        return None, None
    
    best_e = e_cur
    best_p = p_cur.copy()
    sigma = sigma_start
    n = p_cur.shape[0]
    
    for it in range(max_iter):
        if not _deadline_ok(t0):
            break
        
        # Adaptive perturbation: shrink sigma over time
        sigma = max(sigma_min, sigma_start * (0.85 ** it))
        
        # Random perturbation
        rng = np.random.default_rng(int(time.time() * 1000) + it)
        p_new = p_cur + sigma * rng.standard_normal(p_cur.shape)
        
        try:
            e_new, p_new = relax(p_new, potential, parameter)
        except Exception:
            continue
        
        if not np.all(np.isfinite(p_new)):
            continue
        
        # Accept if better, or with Metropolis probability
        if e_new < e_cur:
            p_cur = p_new
            e_cur = e_new
            if e_new < best_e:
                best_e = e_new
                best_p = p_new.copy()
        else:
            # Metropolis acceptance
            delta_e = e_new - e_cur
            # Temperature scales with energy magnitude
            temp = 0.05 * max(1.0, abs(best_e))
            if np.random.default_rng(it * 7919).random() < np.exp(-delta_e / max(temp, 1e-10)):
                p_cur = p_new
                e_cur = e_new
    
    return best_e, best_p


def construct(instance):
    t0 = time.time()
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    spacing = {"morse": 1.0, "sutton_chen": 0.72, "dzugutov": 1.1}.get(potential, 0.72)

    # Build diverse starting configurations
    starts = []
    for gen in (_fcc_cluster, _hcp_cluster, _icosahedral_cluster):
        starts.append(gen(size, spacing))
    
    # Multiple random compact seeds
    for s in (size, size + 137, size + 271, size + 409):
        starts.append(_random_compact(size, spacing, s))
    
    best_energy = None
    best_pos = None
    
    # First pass: quick relax to get a valid fallback
    for start in starts[:3]:
        if not _deadline_ok(t0):
            break
        try:
            e, p = relax(start, potential, parameter)
            if p is not None and np.all(np.isfinite(p)):
                if best_energy is None or e < best_energy:
                    best_energy = e
                    best_pos = p
        except Exception:
            pass
    
    # Ensure we always have a valid result
    if best_pos is None:
        best_pos = _ensure_min_dist(starts[0])
    
    # Second pass: basin hopping from each seed
    for k, start in enumerate(starts):
        if not _deadline_ok(t0):
            break
        
        e, p = _basin_hopping(start, potential, parameter, t0, max_iter=30)
        if p is not None and np.all(np.isfinite(p)):
            if best_energy is None or e < best_energy:
                best_energy = e
                best_pos = p
    
    # Final polish: one more relax from the best
    if _deadline_ok(t0):
        try:
            e_final, p_final = relax(best_pos, potential, parameter)
            if p_final is not None and np.all(np.isfinite(p_final)):
                if best_energy is None or e_final < best_energy:
                    best_energy = e_final
                    best_pos = p_final
        except Exception:
            pass
    
    return _ensure_min_dist(best_pos)
# EVOLVE-BLOCK-END
