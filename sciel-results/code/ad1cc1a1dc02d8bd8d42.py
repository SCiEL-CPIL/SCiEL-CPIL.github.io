"""Program for Sutton-Chen TM cluster global minima search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import threading
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_START_TIME = time.time()
_DEADLINE = 325.0  # Safety margin before 330s kill


def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)


def _is_valid(positions, min_dist=0.35):
    """Check if positions are valid (finite, no atoms closer than min_dist)."""
    if not np.all(np.isfinite(positions)):
        return False
    n = len(positions)
    if n < 2:
        return True
    # Vectorized distance check
    # Compute pairwise distances
    # For large n, this is O(n^2). n is max 80, so 6400 distances. Fast.
    dists = np.sqrt(np.sum((positions[:, np.newaxis, :] - positions[np.newaxis, :, :]) ** 2, axis=-1))
    # Mask diagonal
    mask = ~np.eye(n, dtype=bool)
    min_d = np.min(dists[mask])
    if min_d < min_dist:
        return False
    return True


def _make_fcc(n):
    """Generate n atoms in FCC-like arrangement (closest to origin)."""
    max_r = int(np.ceil(n ** (1 / 3))) + 5
    candidates = []
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            for k in range(-max_r, max_r + 1):
                if (i + j + k) % 2 == 0:
                    p = np.array([i, j, k], dtype=float) / np.sqrt(2.0)
                    candidates.append((np.dot(p, p), p))
    candidates.sort(key=lambda x: x[0])
    pts = [p for _, p in candidates[:n]]
    return np.array(pts)


def _make_bcc(n):
    """Generate n atoms in BCC-like arrangement."""
    max_r = int(np.ceil(n ** (1 / 3))) + 5
    candidates = []
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            for k in range(-max_r, max_r + 1):
                p1 = np.array([i, j, k], dtype=float)
                candidates.append((np.dot(p1, p1), p1))
                p2 = p1 + 0.5
                candidates.append((np.dot(p2, p2), p2))
    candidates.sort(key=lambda x: x[0])
    pts = [p for _, p in candidates[:n]]
    return np.array(pts)


def _make_icosahedral(n):
    """Generate Mackay icosahedral cluster for n=13, 55."""
    phi = (1 + np.sqrt(5)) / 2
    # Icosahedron vertices
    v = [
        np.array([0, 1, phi]), np.array([0, 1, -phi]),
        np.array([0, -1, phi]), np.array([0, -1, -phi]),
        np.array([1, phi, 0]), np.array([1, -phi, 0]),
        np.array([-1, phi, 0]), np.array([-1, -phi, 0]),
        np.array([phi, 0, 1]), np.array([phi, 0, -1]),
        np.array([-phi, 0, 1]), np.array([-phi, 0, -1])
    ]
    v = [p / np.linalg.norm(p) for p in v]
    
    pts = [np.zeros(3)]
    if n <= 13:
        pts.extend(v[:min(12, n - 1)])
        return np.array(pts[:n])
    
    if n == 55:
        # Shell 1: 12 atoms at radius 1
        pts.extend(v)
        
        # Shell 2: 42 atoms at radius 2
        # 30 edge midpoints
        edges = []
        for i in range(12):
            for j in range(i + 1, 12):
                d = np.linalg.norm(v[i] - v[j])
                if 1.5 < d < 2.5:  # adjacent vertices (edge length ~1.9)
                    edges.append((i, j))
        
        shell2_pts = []
        for i, j in edges[:30]:
            mid = (v[i] + v[j]) / 2
            mid = mid * 2.0 / np.linalg.norm(mid)
            shell2_pts.append(mid)
            
        # 12 face centers? No, 20 faces.
        # The remaining 12 atoms in Mackay shell 2 are at the centers of the 12 *vertices*? No.
        # Actually, the 42 atoms are: 30 edge midpoints + 12 "face centers" of the *dual*?
        # Let's just add 12 points at the positions of the icosahedron vertices scaled to radius 2? No, that's duplicate.
        # The 42nd set of points are at the centers of the 20 faces? No, 20 != 12.
        # Wait, 12 + 30 = 42. So we need 12 more.
        # The 12 more are at the centers of the 12 *edges* of the *dodecahedron*?
        # Let's just use the 12 vertices of the icosahedron scaled to radius 2? No, that's where the shell 1 atoms are.
        # Actually, the 42 atoms in the second shell are:
        # 30 at the midpoints of the edges of the icosahedron.
        # 12 at the centers of the faces of the *icosahedron*? No, 20 faces.
        # Let's check the literature. Mackay icosahedron shell 2 has 42 atoms.
        # 30 are at the edge midpoints.
        # 12 are at the face centers of the *triangular faces*? No.
        # Ah, the 12 are at the positions of the vertices of a *dodecahedron* inscribed in the sphere?
        # Let's just take the 12 face centers of the icosahedron (there are 20) and pick 12? No.
        # Let's assume the 12 are at the centers of the 12 *pentagonal faces* of the dodecahedron dual?
        # For now, I'll add 12 points at the vertices of the icosahedron scaled to radius 2? No, that's invalid (overlap with shell 1 direction).
        # Let's just add 12 random points on the sphere of radius 2, far from existing points.
        rng = np.random.default_rng(42)
        for _ in range(100):
            if len(shell2_pts) >= 42:
                break
            p = rng.normal(size=3)
            p = p * 2.0 / np.linalg.norm(p)
            # Check distance to existing shell2 points
            if all(np.linalg.norm(p - q) > 0.5 for q in shell2_pts):
                shell2_pts.append(p)
        
        pts.extend(shell2_pts[:42])
        return np.array(pts[:n])
        
    # For n > 55, fallback to random compact
    return _make_random_compact(n, 123)


def _make_decahedral(n):
    """Generate a decahedral cluster (5 FCC tetrahedra)."""
    # This is a complex structure. 
    # Approximation: Generate FCC and twist.
    # For now, use FCC as a proxy if decahedral is too hard to generate exactly.
    # But decahedral is often the global min.
    # Let's try to generate 5 tetrahedra.
    # Tetrahedron 1: (1,1,1), (1,-1,-1), (-1,1,-1), (-1,-1,1)
    # These are 4 points.
    # The 5 tetrahedra share the origin? No, they share a central atom.
    # Let's just use FCC for now and rely on Basin Hopping to find the twist.
    return _make_fcc(n)


def _make_random_compact(n, seed):
    """Generate random positions in a compact sphere."""
    rng = np.random.default_rng(seed)
    r = 0.4 * np.sqrt(n)
    pts = rng.normal(size=(n, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    pts = pts / norms * r * rng.random((n, 1)) ** (1 / 3)
    return pts


def _basin_hop_chain(positions, kind, n_steps, rng):
    """Run a basin hopping chain."""
    e, current = relax(positions, 'sutton_chen', kind)
    if not (np.isfinite(e) and _is_valid(current)):
        return positions, np.inf
    
    best_e = e
    best_pos = current.copy()
    
    for i in range(n_steps):
        if _time_left() < 10.0:
            break
        
        # Perturb
        sigma = 0.1 + 0.1 * np.sin(i * 0.1)  # Vary sigma slightly
        pos = current + rng.normal(0, sigma, (len(current), 3))
        
        if not _is_valid(pos):
            continue
        
        try:
            e_new, relaxed = relax(pos, 'sutton_chen', kind)
            if np.isfinite(e_new) and _is_valid(relaxed):
                # Metropolis accept? No, just keep if better for global min search.
                # But pure hill climbing in BH can get stuck.
                # Standard BH: accept if E_new < E_old, else accept with prob exp(-dE/T).
                # For global min, we want to explore.
                if e_new < best_e:
                    best_e = e_new
                    best_pos = relaxed.copy()
                # Always move to new basin to explore
                current = relaxed
        except Exception:
            continue
            
    return best_pos, best_e


def construct(instance):
    n = instance["n"]
    kind = instance["kind"]
    
    best_energy = np.inf
    best_pos = None
    
    # Phase 1: Generate and relax diverse initial structures
    generators = [
        lambda: _make_fcc(n),
        lambda: _make_bcc(n),
        lambda: _make_icosahedral(n),
        lambda: _make_random_compact(n, 1),
        lambda: _make_random_compact(n, 2),
        lambda: _make_random_compact(n, 3),
    ]
    
    initial_candidates = []
    for gen in generators:
        if _time_left() < 20.0:
            break
        try:
            base = gen()
            if _is_valid(base):
                energy, relaxed = relax(base, 'sutton_chen', kind)
                if np.isfinite(energy) and _is_valid(relaxed):
                    initial_candidates.append((energy, relaxed))
        except Exception:
            continue
    
    if not initial_candidates:
        # Fallback
        best_pos = _make_fcc(n)
        if not _is_valid(best_pos):
            best_pos = _make_random_compact(n, 0)
        return best_pos
    
    # Sort by energy
    initial_candidates.sort(key=lambda x: x[0])
    
    # Phase 2: Basin Hopping from the best few candidates
    # Use threads if possible, but GIL might limit. 
    # Let's run 3 chains sequentially to be safe with GIL, but fast.
    n_chains = 3
    rngs = [np.random.default_rng(100 + i) for i in range(n_chains)]
    
    # We will run BH from the top 3 initial candidates
    for i in range(min(n_chains, len(initial_candidates))):
        if _time_left() < 30.0:
            break
        
        e_init, pos_init = initial_candidates[i]
        if e_init < best_energy:
            best_energy = e_init
            best_pos = pos_init.copy()
        
        # Run BH chain
        bh_pos, bh_energy = _basin_hop_chain(pos_init, kind, n_steps=100, rng=rngs[i])
        
        if bh_energy < best_energy:
            best_energy = bh_energy
            best_pos = bh_pos
    
    # If time remains, run more BH from the global best found so far
    if best_pos is not None and _time_left() > 20.0:
        # Continue BH from the best found
        rng_final = np.random.default_rng(999)
        bh_pos, bh_energy = _basin_hop_chain(best_pos, kind, n_steps=50, rng=rng_final)
        if bh_energy < best_energy:
            best_energy = bh_energy
            best_pos = bh_pos

    return best_pos

# EVOLVE-BLOCK-END
