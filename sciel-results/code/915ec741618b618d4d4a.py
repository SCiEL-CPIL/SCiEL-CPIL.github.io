"""Improved sphere packing: multi-start with BCC/FCC lattices + gradient-free local search."""
import numpy as np
import time
from scipy.spatial import cKDTree

N = 50
DEADLINE = 135.0  # safety margin before 150s kill
START_TIME = time.time()


def _time_remaining():
    return DEADLINE - (time.time() - START_TIME)


def _score(centers):
    """Compute the radius: min of half-min-pair-distance and min-boundary-distance."""
    if centers.shape[0] != N:
        return 0.0
    # Boundary distance
    bd = np.min(np.abs(centers), axis=1)
    bd = 0.5 - np.max(bd, axis=1)
    min_bd = np.min(bd)
    if min_bd < 0:
        return 0.0
    # Pair distances
    tree = cKDTree(centers)
    pairs = tree.query_pairs(r=0.5)
    if not pairs:
        return min_bd
    min_pd = np.inf
    for i, j in pairs:
        d = np.linalg.norm(centers[i] - centers[j])
        if d < min_pd:
            min_pd = d
        if d < min_bd:
            # Early exit if pair distance already smaller
            pass
    return min(min_bd, min_pd / 2.0)


def _bcc_lattice():
    """Generate a BCC lattice initialization for 50 points in the cube."""
    # BCC: simple cubic + body centers
    # We need ~50 points. Use a 4x4x4 grid = 64, take first 50
    # Better: use a smaller grid and add body centers
    # 3x3x3 = 27 simple + 27 body = 54, take 50
    m = 3
    pts = []
    for i in range(m):
        for j in range(m):
            for k in range(m):
                # Simple cubic point
                x = (i + 0.5) / m - 0.5
                y = (j + 0.5) / m - 0.5
                z = (k + 0.5) / m - 0.5
                pts.append([x, y, z])
                # Body center
                xb = (i + 1.0) / m - 0.5
                yb = (j + 1.0) / m - 0.5
                zb = (k + 1.0) / m - 0.5
                pts.append([xb, yb, zb])
    pts = np.array(pts[:N])
    # Scale to fit in cube with some margin
    # The points are already in [-0.5, 0.5]
    return pts


def _fcc_lattice():
    """Generate an FCC-like lattice initialization for 50 points."""
    # FCC: face-centered cubic
    # Use a 3x3x3 simple cubic grid + face centers
    m = 3
    pts = []
    for i in range(m):
        for j in range(m):
            for k in range(m):
                x = (i + 0.5) / m - 0.5
                y = (j + 0.5) / m - 0.5
                z = (k + 0.5) / m - 0.5
                pts.append([x, y, z])
                # Face centers: (x, y+0.5/m, z), (x, y, z+0.5/m), (x+0.5/m, y, z)
                pts.append([x, (j + 1.0) / m - 0.5, z])
                pts.append([x, y, (k + 1.0) / m - 0.5])
                pts.append([(i + 1.0) / m - 0.5, y, z])
    pts = np.array(pts[:N])
    return pts


def _random_init():
    """Random initialization within the cube."""
    return np.random.uniform(-0.45, 0.45, size=(N, 3))


def _local_search(centers, max_iter=5000, step0=0.05):
    """Simple gradient-free local search: random perturbations of the worst pair."""
    centers = centers.copy()
    best_score = _score(centers)
    step = step0
    
    for it in range(max_iter):
        if _time_remaining() < 5.0:
            break
        # Find the closest pair
        tree = cKDTree(centers)
        pairs = tree.query_pairs(r=0.5)
        if not pairs:
            break
        # Find the minimum distance pair
        min_d = np.inf
        min_pair = None
        for i, j in pairs:
            d = np.linalg.norm(centers[i] - centers[j])
            if d < min_d:
                min_d = d
                min_pair = (i, j)
        
        # Also check boundary
        bd = 0.5 - np.max(np.abs(centers), axis=1)
        min_bd = np.min(bd)
        worst_idx = np.argmin(bd)
        
        if min_d / 2.0 < min_bd:
            # Move apart the closest pair
            i, j = min_pair
            diff = centers[i] - centers[j]
            norm = np.linalg.norm(diff)
            if norm > 1e-10:
                diff /= norm
            else:
                diff = np.random.randn(3)
                diff /= np.linalg.norm(diff)
            centers[i] += 0.5 * step * diff
            centers[j] -= 0.5 * step * diff
        else:
            # Move the worst-boundary point inward
            idx = worst_idx
            # Move toward center
            direction = -centers[idx]
            norm = np.linalg.norm(direction)
            if norm > 1e-10:
                direction /= norm
            else:
                direction = np.random.randn(3)
                direction /= np.linalg.norm(direction)
            centers[idx] += step * direction
        
        # Clamp to cube
        centers = np.clip(centers, -0.5, 0.5)
        
        new_score = _score(centers)
        if new_score > best_score:
            best_score = new_score
            step = min(step * 1.05, 0.2)
        else:
            # Revert
            centers = centers.copy()
            step *= 0.9
            if step < 1e-5:
                break
    
    return centers, best_score


def construct():
    global START_TIME
    START_TIME = time.time()
    
    np.random.seed(42)
    
    # Generate initial candidates from different lattice structures
    inits = []
    inits.append(_bcc_lattice())
    inits.append(_fcc_lattice())
    
    # Add some random inits
    for _ in range(5):
        inits.append(_random_init())
    
    best_centers = None
    best_score = 0.0
    
    for init in inits:
        if _time_remaining() < 10.0:
            break
        
        # Run local search
        centers, score = _local_search(init, max_iter=3000, step0=0.03)
        
        if score > best_score:
            best_score = score
            best_centers = centers.copy()
    
    # Final polish with smaller steps
    if _time_remaining() > 5.0 and best_centers is not None:
        centers, score = _local_search(best_centers, max_iter=2000, step0=0.01)
        if score > best_score:
            best_score = score
            best_centers = centers
    
    if best_centers is None:
        # Fallback: simple grid
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N]
    
    return best_centers
