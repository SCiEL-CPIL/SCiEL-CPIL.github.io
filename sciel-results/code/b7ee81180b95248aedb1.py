"""Portfolio-search program for Sutton-Chen TM cluster global minima."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_START_TIME = time.time()
_DEADLINE = 300.0


def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)


def _is_valid(positions, min_dist=0.35):
    """Check that positions are finite and no two atoms are too close."""
    if not np.all(np.isfinite(positions)):
        return False
    n = len(positions)
    for i in range(n):
        d = np.linalg.norm(positions[i + 1:] - positions[i], axis=1)
        if np.any(d < min_dist):
            return False
    return True


def _make_fcc(n):
    """Generate a compact FCC-like cluster of n atoms."""
    pts = []
    mc = int(np.ceil(n ** (1 / 3))) + 2
    for i in range(-mc, mc + 1):
        for j in range(-mc, mc + 1):
            for k in range(-mc, mc + 1):
                # FCC lattice: positions at (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5) offsets
                for dx, dy, dz in [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]:
                    p = np.array([i + dx, j + dy, k + dz]) * 0.72
                    pts.append(p)
    # Sort by distance from center and take n
    if len(pts) >= n:
        pts = np.array(pts)
        center = pts.mean(axis=0)
        d2 = np.sum((pts - center) ** 2, axis=1)
        idx = np.argsort(d2)[:n]
        pts = pts[idx]
    else:
        pts = np.array(pts)
        # Pad with random points if needed
        while len(pts) < n:
            pts = np.vstack([pts, np.random.uniform(-2, 2, 3)])
    return pts


def _make_bcc(n):
    """Generate a compact BCC-like cluster of n atoms."""
    pts = []
    mc = int(np.ceil(n ** (1 / 3))) + 2
    for i in range(-mc, mc + 1):
        for j in range(-mc, mc + 1):
            for k in range(-mc, mc + 1):
                p = np.array([i, j, k]) * 0.72
                pts.append(p)
                if i == 0 and j == 0 and k == 0:
                    continue
                # BCC center atom
                pc = np.array([i + 0.5, j + 0.5, k + 0.5]) * 0.72
                pts.append(pc)
    if len(pts) >= n:
        pts = np.array(pts)
        center = pts.mean(axis=0)
        d2 = np.sum((pts - center) ** 2, axis=1)
        idx = np.argsort(d2)[:n]
        pts = pts[idx]
    else:
        pts = np.array(pts)
        while len(pts) < n:
            pts = np.vstack([pts, np.random.uniform(-2, 2, 3)])
    return pts


def _make_mackay(n):
    """Generate a Mackay icosahedral cluster seed."""
    # Mackay icosahedra have shells: 1, 12, 42, 120, 220, ...
    # For arbitrary n, build concentric icosahedral shells
    pts = []
    # Inner atom
    pts.append(np.array([0.0, 0.0, 0.0]))
    
    # First shell: 12 vertices of icosahedron
    t = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, t], [0, -1, t], [0, 1, -t], [0, -1, -t],
        [1, t, 0], [-1, t, 0], [1, -t, 0], [-1, -t, 0],
        [t, 0, 1], [-t, 0, 1], [t, 0, -1], [-t, 0, -1]
    ], dtype=float)
    ico_verts = ico_verts / np.linalg.norm(ico_verts[0]) * 0.72
    for v in ico_verts:
        pts.append(v)
    
    # Second shell: 30 edges + 12 vertices positions (Mackay 42-atom)
    if n > 13:
        # Add edge midpoints (30)
        edges = []
        for i in range(12):
            for j in range(i + 1, 12):
                d = np.linalg.norm(ico_verts[i] - ico_verts[j])
                if 0.6 < d < 1.2:  # adjacent vertices
                    edges.append((i, j))
        for i, j in edges:
            mid = (ico_verts[i] + ico_verts[j]) / 2 * 1.3
            pts.append(mid)
        
        # Add face centers (20)
        if n > 43:
            # Approximate face centers
            faces = []
            for i in range(12):
                for j in range(i + 1, 12):
                    for k in range(j + 1, 12):
                        d1 = np.linalg.norm(ico_verts[i] - ico_verts[j])
                        d2 = np.linalg.norm(ico_verts[j] - ico_verts[k])
                        d3 = np.linalg.norm(ico_verts[i] - ico_verts[k])
                        if 0.6 < d1 < 1.2 and 0.6 < d2 < 1.2 and 0.6 < d3 < 1.2:
                            faces.append((i, j, k))
            for i, j, k in faces:
                fc = (ico_verts[i] + ico_verts[j] + ico_verts[k]) / 3 * 1.5
                pts.append(fc)
    
    # Take first n points
    pts = np.array(pts)
    if len(pts) > n:
        center = pts.mean(axis=0)
        d2 = np.sum((pts - center) ** 2, axis=1)
        idx = np.argsort(d2)[:n]
        pts = pts[idx]
    else:
        while len(pts) < n:
            pts = np.vstack([pts, np.random.uniform(-2, 2, 3)])
    return pts


def _basin_hopping(positions, potential, kind, max_iter=50, kick_scale=0.2, rng=None):
    """Simple basin hopping: perturb, relax, accept if lower energy."""
    if rng is None:
        rng = np.random.default_rng()
    
    energy, pos = relax(positions.copy(), potential, kind)
    best_energy = energy
    best_pos = pos.copy()
    
    for _ in range(max_iter):
        if _time_left() < 5.0:
            break
        # Kick
        kick = rng.normal(0, kick_scale, size=pos.shape)
        perturbed = pos + kick
        # Relax
        try:
            new_energy, new_pos = relax(perturbed, potential, kind)
            if not _is_valid(new_pos):
                continue
            if new_energy < best_energy:
                best_energy = new_energy
                best_pos = new_pos.copy()
                kick_scale = max(0.05, kick_scale * 0.95)
            else:
                kick_scale = min(0.5, kick_scale * 1.05)
            pos = new_pos
        except Exception:
            continue
    
    return best_energy, best_pos


def construct(instance):
    """Multi-start portfolio search with basin hopping for Sutton-Chen clusters."""
    size = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", "12-6")
    
    best_energy = np.inf
    best_pos = None
    rng = np.random.default_rng(size * 1000 + 42)
    
    # Generate diverse starting points
    starts = []
    
    # 1. FCC-like
    starts.append(_make_fcc(size))
    # 2. BCC-like
    starts.append(_make_bcc(size))
    # 3. Mackay icosahedral
    starts.append(_make_mackay(size))
    
    # 4-8. Random compact starts
    for _ in range(5):
        radius = 0.72 * size ** (1.0 / 3.0) * 0.8
        points = []
        attempts = 0
        while len(points) < size and attempts < size * 100:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                if all(np.linalg.norm(p - q) >= 0.5 * 0.72 for q in points):
                    points.append(p)
            attempts += 1
        if len(points) < size:
            # Fill remaining randomly
            while len(points) < size:
                points.append(rng.uniform(-radius, radius, 3))
        starts.append(np.array(points))
    
    # Relax each start and keep the best
    for start in starts:
        if _time_left() < 10.0:
            break
        try:
            energy, positions = relax(start, potential, kind)
            if _is_valid(positions) and energy < best_energy:
                best_energy = energy
                best_pos = positions.copy()
        except Exception:
            continue
    
    # Basin hopping from the best result
    if best_pos is not None and _time_left() > 30.0:
        try:
            bh_energy, bh_pos = _basin_hopping(best_pos, potential, kind, 
                                                max_iter=100, kick_scale=0.15, rng=rng)
            if _is_valid(bh_pos) and bh_energy < best_energy:
                best_energy = bh_energy
                best_pos = bh_pos.copy()
        except Exception:
            pass
    
    # If still have time, try basin hopping from other good starts
    if best_pos is not None and _time_left() > 60.0:
        for start in starts[1:4]:  # Try a few more starts
            if _time_left() < 30.0:
                break
            try:
                energy, positions = relax(start, potential, kind)
                if _is_valid(positions):
                    bh_energy, bh_pos = _basin_hopping(positions, potential, kind,
                                                        max_iter=50, kick_scale=0.15, rng=rng)
                    if _is_valid(bh_pos) and bh_energy < best_energy:
                        best_energy = bh_energy
                        best_pos = bh_pos.copy()
            except Exception:
                continue
    
    # Final validation
    if best_pos is None or not _is_valid(best_pos):
        # Fallback: return a simple valid configuration
        best_pos = np.zeros((size, 3))
        for i in range(size):
            best_pos[i] = rng.uniform(-3, 3, 3)
        # Ensure minimum distance
        for i in range(size):
            for j in range(i + 1, size):
                d = np.linalg.norm(best_pos[i] - best_pos[j])
                if d < 0.35:
                    best_pos[j] = best_pos[i] + 0.4 * rng.normal(0, 1, 3)
    
    return best_pos
# EVOLVE-BLOCK-END
