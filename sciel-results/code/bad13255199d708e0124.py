"""Morse cluster global minima: deterministic Voronoi-based iterative relaxation
with structured seeds (FCC, HCP, icosahedral) followed by L-BFGS polish.
A hard wall-clock deadline guard returns the best finite result before the kill."""

import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay, cKDTree

_T0 = time.time()
_KILL = 330.0
_MARGIN = 25.0


def _time_left():
    return _KILL - _MARGIN - (time.time() - _T0)


def _morse_energy(positions, rho):
    """Morse energy, well depth 1: E = sum exp(rho(1-r))(exp(rho(1-r))-2)."""
    n = len(positions)
    if n < 2:
        return 0.0
    d = positions[:, None, :] - positions[None, :, :]
    r = np.sqrt(np.einsum("ijk,ijk->ij", d, d))
    np.fill_diagonal(r, np.inf)
    u = np.exp(rho * (1.0 - r))
    return float(np.sum(u * (u - 2.0)))


def _morse_grad(positions, rho):
    """Gradient of Morse energy."""
    n = len(positions)
    g = np.zeros_like(positions)
    for i in range(n):
        for j in range(i + 1, n):
            d = positions[i] - positions[j]
            r = np.linalg.norm(d)
            if r < 1e-12:
                continue
            u = np.exp(rho * (1.0 - r))
            # dE/dr = -2*rho*u*(u-1)
            # dE/dx_i = (dE/dr) * (d/dr)(r) where r = |x_i - x_j|
            # dr/dx_i = d/r
            force = -2.0 * rho * u * (u - 1.0) / r
            g[i] += force * d
            g[j] -= force * d
    return g


def _generate_fcc_cluster(n):
    """Generate an FCC truncated sphere cluster."""
    if n <= 1:
        return np.zeros((1, 3))
    pts = []
    R = 0
    while len(pts) < n + 20:
        for i in range(-R, R + 1):
            for j in range(-R, R + 1):
                for k in range(-R, R + 1):
                    if (i + j + k) % 2 == 0:
                        p = np.array([i, j, k], dtype=float)
                        r = np.linalg.norm(p)
                        if r <= R + 0.1:
                            pts.append(p)
        R += 1
    pts = np.array(pts)
    if len(pts) > 1:
        pts = np.unique(np.round(pts, 6), axis=0)
    pts = pts / np.sqrt(2.0)
    if len(pts) >= n:
        return pts[:n]
    return pts


def _generate_hcp_cluster(n):
    """Generate an HCP truncated sphere cluster."""
    if n <= 1:
        return np.zeros((1, 3))
    pts = []
    a = 1.0  # in-plane lattice constant
    c = np.sqrt(8.0 / 3.0)  # ideal c/a ratio
    R = 0
    while len(pts) < n + 20:
        for i in range(-R, R + 1):
            for j in range(-R, R + 1):
                # Layer A
                p_a = np.array([a * (i + 0.5 * j), a * np.sqrt(3) / 2 * j, 0.0])
                if np.linalg.norm(p_a) <= R + 0.5:
                    pts.append(p_a)
                # Layer B (offset)
                p_b = np.array([a * (i + 0.5 * j + 1.0 / (3.0 * np.sqrt(3))),
                                a * np.sqrt(3) / 2 * (j + 1.0 / 3.0),
                                c / 2.0])
                if np.linalg.norm(p_b) <= R + 0.5:
                    pts.append(p_b)
        R += 0.5
    pts = np.array(pts)
    if len(pts) > 1:
        pts = np.unique(np.round(pts, 6), axis=0)
    if len(pts) >= n:
        return pts[:n]
    return pts


def _generate_icosahedral_cluster(n):
    """Generate an icosahedral cluster using golden-ratio spiral shells."""
    if n <= 1:
        return np.zeros((1, 3))
    # Icosahedral shells: 1, 12, 42, 92, 150, 220, 302, ...
    shell_counts = [1, 12, 42, 92, 150, 220, 302, 392, 490, 596, 710]
    ga = (1.0 + np.sqrt(5.0)) / 2.0
    atoms = [np.zeros(3)]
    for L in range(1, len(shell_counts)):
        cnt = shell_counts[L]
        rad = np.sqrt(L * L + L)
        for i in range(cnt):
            y = 1.0 - (2.0 * i + 1.0) / cnt
            rr = np.sqrt(max(0.0, 1.0 - y * y))
            phi = 2.0 * np.pi * i / ga
            atoms.append(np.array([np.cos(phi) * rr, y, np.sin(phi) * rr]) * rad)
        if len(atoms) >= n:
            break
    arr = np.array(atoms)
    if len(arr) >= n:
        return arr[:n]
    return arr


def _generate_random_compact(n, rng, seed_offset=0):
    """Generate a random compact cluster."""
    radius = n ** (1.0 / 3.0) * 0.9
    points = [np.zeros(3)]
    attempts = 0
    max_attempts = n * 50
    while len(points) < n and attempts < max_attempts:
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
    while len(points) < n:
        points.append(rng.uniform(-radius, radius, 3))
    return np.array(points)


def _voronoi_relax(positions, rho, max_iter=50, tol=1e-8):
    """Voronoi-based iterative relaxation: move each atom toward centroid of its Voronoi cell."""
    n = len(positions)
    if n < 4:
        return positions.copy()
    
    pos = positions.copy()
    for it in range(max_iter):
        if _time_left() < 5.0:
            break
        try:
            tree = cKDTree(pos)
            # Find nearest neighbors for each atom
            nn_idx, nn_dist = tree.query(pos, k=min(12, n))
            
            # Compute weighted centroid shift
            shifts = np.zeros_like(pos)
            for i in range(n):
                # Get neighbors within reasonable distance
                mask = nn_dist[i] < 3.0
                if not np.any(mask):
                    continue
                nbrs = nn_idx[i][mask]
                dists = nn_dist[i][mask]
                
                # Weight by inverse Morse force magnitude
                weights = np.zeros_like(dists)
                for k, (j, d) in enumerate(zip(nbrs, dists)):
                    if d < 1e-12:
                        continue
                    u = np.exp(rho * (1.0 - d))
                    force_mag = abs(2.0 * rho * u * (u - 1.0))
                    weights[k] = 1.0 / (force_mag + 1e-10)
                
                if np.sum(weights) < 1e-10:
                    continue
                
                # Weighted centroid of neighbors
                centroid = np.average(pos[nbrs], axis=0, weights=weights)
                # Move toward centroid with damping
                shift = centroid - pos[i]
                shifts[i] = 0.1 * shift  # damping factor
            
            # Apply shifts
            new_pos = pos + shifts
            move = np.linalg.norm(shifts)
            
            # Check convergence
            if move < tol:
                break
            
            pos = new_pos
        except Exception:
            break
    
    return pos


def _lbfgs_polish(positions, rho, max_iter=100):
    """Polish with L-BFGS-B."""
    n = len(positions)
    x0 = positions.reshape(-1)
    
    def fun(x):
        return _morse_energy(x.reshape(n, 3), rho)
    
    def jac(x):
        return _morse_grad(x.reshape(n, 3), rho).reshape(-1)
    
    try:
        res = minimize(fun, x0, jac=jac, method='L-BFGS-B',
                       options={'maxiter': max_iter, 'gtol': 1e-10, 'ftol': 1e-12})
        if res.fun < fun(x0):
            return res.x.reshape(n, 3), float(res.fun)
    except Exception:
        pass
    
    return positions.copy(), _morse_energy(positions, rho)


def _check_min_dist(positions, min_dist=0.5):
    """Check that all atoms are at least min_dist apart."""
    n = len(positions)
    tree = cKDTree(positions)
    pairs = tree.query_pairs(min_dist)
    if len(pairs) == 0:
        return True
    return False


def _push_apart(positions, min_dist=0.5, max_iter=100):
    """Push apart atoms that are too close."""
    pos = positions.copy()
    for it in range(max_iter):
        if _time_left() < 3.0:
            break
        tree = cKDTree(pos)
        pairs = list(tree.query_pairs(min_dist))
        if len(pairs) == 0:
            break
        for i, j in pairs:
            d = pos[i] - pos[j]
            r = np.linalg.norm(d)
            if r < 1e-12:
                continue
            push = (min_dist - r) / 2.0
            d_hat = d / r
            pos[i] += push * d_hat
            pos[j] -= push * d_hat
    return pos


def construct(instance):
    """Build a low-energy Morse cluster for the given instance."""
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind", 6))
    
    rng = np.random.default_rng(42)
    
    # Generate diverse candidate initial geometries
    candidates = []
    try:
        candidates.append(_generate_fcc_cluster(n))
    except Exception:
        pass
    try:
        candidates.append(_generate_hcp_cluster(n))
    except Exception:
        pass
    try:
        candidates.append(_generate_icosahedral_cluster(n))
    except Exception:
        pass
    for i in range(4):
        try:
            candidates.append(_generate_random_compact(n, rng, seed_offset=i))
        except Exception:
            pass
    
    best_E = np.inf
    best_pos = None
    num_cands = len(candidates)
    per_candidate_time = max(3.0, _time_left() / (num_cands + 1))
    
    for idx, cand in enumerate(candidates):
        if _time_left() < 8.0:
            break
        
        cand = np.asarray(cand, dtype=float)
        if len(cand) < n:
            extra = rng.uniform(-2, 2, (n - len(cand), 3))
            cand = np.vstack([cand, extra])
        elif len(cand) > n:
            cand = cand[:n]
        
        # Ensure minimum distance
        cand = _push_apart(cand, min_dist=0.5, max_iter=50)
        
        # Voronoi relaxation
        try:
            cand = _voronoi_relax(cand, rho, max_iter=30, tol=1e-7)
        except Exception:
            pass
        
        # L-BFGS polish
        try:
            pos, E = _lbfgs_polish(cand, rho, max_iter=80)
            if E < best_E:
                best_E = E
                best_pos = pos
        except Exception:
            try:
                E = _morse_energy(cand, rho)
                if E < best_E:
                    best_E = E
                    best_pos = cand.copy()
            except Exception:
                pass
    
    # Final safety
    if best_pos is None or not np.all(np.isfinite(best_pos)):
        best_pos = _generate_fcc_cluster(n)[:n]
        if len(best_pos) < n:
            best_pos = np.vstack([best_pos,
                                  rng.uniform(-1, 1, (n - len(best_pos), 3))])
        best_pos = best_pos[:n]
        best_E = _morse_energy(best_pos, rho)
    
    # Final check for minimum distance
    best_pos = _push_apart(best_pos, min_dist=0.5, max_iter=30)
    
    return np.asarray(best_pos[:n], dtype=float)
