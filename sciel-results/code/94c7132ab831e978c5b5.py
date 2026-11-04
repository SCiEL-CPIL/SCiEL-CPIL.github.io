"""Rewritten solver: dense FCC/HCP init + active-set refinement for 50 spheres in a unit cube."""
import time
import numpy as np

N = 50
DEADLINE = 140.0  # safety margin below the 150s kill

def _radius(pts):
    """Largest common radius for given centres (validity-checked)."""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape != (N, 3):
        return -1.0
    if not np.all(np.isfinite(pts)):
        return -1.0
    # wall clearance
    wall = 0.5 - np.max(np.abs(pts), axis=1)
    if np.any(wall < -1e-12):
        return -1.0
    wall = float(np.min(wall))
    # pairwise min distance / 2
    d = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(d * d, axis=2))
    np.fill_diagonal(dist, np.inf)
    pair = float(np.min(dist)) / 2.0
    return min(wall, pair)

def _fcc_init():
    """Dense FCC lattice, fill a bounding box, keep the N most central points."""
    a = 1.0
    pts = []
    for i in range(6):
        for j in range(6):
            for k in range(6):
                x = a * (i + 0.5 * (j & 1) + 0.5 * (k & 1))
                y = a * (j + 0.5 * (i & 1) + 0.5 * (k & 1))
                z = a * (k + 0.5 * (i & 1) + 0.5 * (j & 1))
                pts.append([x, y, z])
    pts = np.array(pts)
    # centre the cloud at origin
    pts -= pts.mean(axis=0)
    # pick the N most central
    r2 = np.sum(pts * pts, axis=1)
    idx = np.argsort(r2)[:N]
    p = pts[idx]
    # scale to fit comfortably inside cube (leave margin)
    # current extent
    ext = np.max(np.abs(p), axis=0)
    scale = 0.42 / np.max(ext)
    p = p * scale
    # jitter-free but ensure validity: clip
    p = np.clip(p, -0.5, 0.5)
    return p

def _hcp_init():
    """Hexagonal close packed layers, fill, keep N most central."""
    pts = []
    a = 1.0
    for layer in range(6):
        z = a * layer * np.sqrt(2.0 / 3.0)
        for i in range(6):
            for j in range(6):
                x = a * (i + 0.5 * (j & 1))
                y = a * (j * np.sqrt(3.0) / 2.0)
                # alternate layer offset
                if layer % 2 == 1:
                    x += a * 0.5
                pts.append([x, y, z])
    pts = np.array(pts)
    pts -= pts.mean(axis=0)
    r2 = np.sum(pts * pts, axis=1)
    idx = np.argsort(r2)[:N]
    p = pts[idx]
    ext = np.max(np.abs(p), axis=0)
    scale = 0.42 / np.max(ext)
    p = p * scale
    p = np.clip(p, -0.5, 0.5)
    return p

def _refine(p0, iters, step):
    """Active-set max-min refinement. Returns best pts found."""
    p = p0.copy()
    best_r = _radius(p)
    best_p = p.copy()
    for it in range(iters):
        # wall clearance
        wall = 0.5 - np.max(np.abs(p), axis=1)
        # pairwise distances
        d = p[:, None, :] - p[None, :, :]
        dist = np.sqrt(np.sum(d * d, axis=2))
        np.fill_diagonal(dist, np.inf)
        i, j = np.unravel_index(np.argmin(dist), dist.shape)
        min_pair = dist[i, j] / 2.0
        r = min(float(np.min(wall)), min_pair)
        if r > best_r + 1e-15:
            best_r = r
            best_p = p.copy()
        # determine active constraints
        moved = False
        # closest pair
        if min_pair <= r + 1e-12:
            v = p[i] - p[j]
            n = np.linalg.norm(v)
            if n > 1e-14:
                v = v / n
                p[i] += v * step * 0.5
                p[j] -= v * step * 0.5
                moved = True
        # wall violations (points with min wall clearance)
        wmin = np.min(wall)
        if wmin <= r + 1e-12:
            for k in range(N):
                if wall[k] <= wmin + 1e-9:
                    # push inward along the axis of the max |coord|
                    c = p[k]
                    ax = int(np.argmax(np.abs(c)))
                    p[k, ax] -= np.sign(c[ax]) * step
                    moved = True
        # clip into cube
        p = np.clip(p, -0.5, 0.5)
        if not moved:
            break
    # final recompute
    r = _radius(p)
    if r > best_r:
        best_r = r
        best_p = p.copy()
    return best_p, best_r

def construct():
    t0 = time.time()
    inits = [_fcc_init(), _hcp_init()]
    best_p = None
    best_r = -1.0
    # progressive refinement: coarse then fine
    stages = [(400, 0.02), (600, 0.01), (800, 0.005), (1000, 0.002), (1200, 0.001), (1500, 0.0005)]
    for init in inits:
        cur = init.copy()
        cur_r = _radius(cur)
        if cur_r > best_r:
            best_r = cur_r
            best_p = cur.copy()
        for iters, step in stages:
            if time.time() - t0 > DEADLINE:
                break
            cur, cur_r = _refine(cur, iters, step)
            if cur_r > best_r:
                best_r = cur_r
                best_p = cur.copy()
        if time.time() - t0 > DEADLINE:
            break
    # final high-precision polish on the best
    if best_p is not None and time.time() - t0 < DEADLINE:
        best_p, best_r2 = _refine(best_p, 2000, 0.0002)
        if best_r2 > best_r:
            best_r = best_r2
    if best_p is None:
        best_p = _fcc_init()
    return best_p
