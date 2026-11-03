"""Stronger algorithm for packing 50 equal spheres in a unit cube."""
import numpy as np
import time
import itertools

N = 50

# EVOLVE-BLOCK-START

def _score(centers):
    """Compute the minimum radius for the given centers."""
    if centers.shape != (N, 3):
        return -1.0
    if not np.all(np.isfinite(centers)):
        return -1.0
    # Check if all centers are inside the cube [-0.5, 0.5]^3
    if np.any(np.abs(centers) > 0.5 + 1e-12):
        return -1.0
    # Compute minimum distance to walls
    wall_dists = np.min(0.5 - np.abs(centers), axis=1)
    min_wall = np.min(wall_dists)
    # Compute minimum pairwise distance
    min_pair = np.inf
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < min_pair:
                min_pair = d
    return min(min_wall, min_pair / 2.0)


def _fcc_lattice(n):
    """Generate FCC lattice points and select n of them."""
    points = []
    # FCC basis vectors
    a = 1.0
    basis = [
        np.array([0, 0, 0]),
        np.array([0, 0.5, 0.5]),
        np.array([0.5, 0, 0.5]),
        np.array([0.5, 0.5, 0]),
    ]
    # Generate lattice points in a larger region
    for i in range(-2, 5):
        for j in range(-2, 5):
            for k in range(-2, 5):
                for b in basis:
                    p = np.array([i * a, j * a, k * a]) + b
                    if -0.6 <= p[0] <= 0.6 and -0.6 <= p[1] <= 0.6 and -0.6 <= p[2] <= 0.6:
                        points.append(p)
    points = np.array(points)
    # Filter to those inside the cube
    mask = np.all(np.abs(points) <= 0.5, axis=1)
    points = points[mask]
    # Sort by distance from center to prefer central points
    dists = np.linalg.norm(points, axis=1)
    order = np.argsort(dists)
    points = points[order]
    return points[:n]


def _bcc_lattice(n):
    """Generate BCC lattice points and select n of them."""
    points = []
    a = 1.0
    basis = [
        np.array([0, 0, 0]),
        np.array([0.5, 0.5, 0.5]),
    ]
    for i in range(-2, 5):
        for j in range(-2, 5):
            for k in range(-2, 5):
                for b in basis:
                    p = np.array([i * a, j * a, k * a]) + b
                    if -0.6 <= p[0] <= 0.6 and -0.6 <= p[1] <= 0.6 and -0.6 <= p[2] <= 0.6:
                        points.append(p)
    points = np.array(points)
    mask = np.all(np.abs(points) <= 0.5, axis=1)
    points = points[mask]
    dists = np.linalg.norm(points, axis=1)
    order = np.argsort(dists)
    points = points[order]
    return points[:n]


def _local_search(centers, max_iters=2000, tol=1e-10):
    """Local search to maximize minimum radius."""
    centers = centers.copy()
    r = _score(centers)
    if r <= 0:
        return centers, r
    
    step = 0.01
    for it in range(max_iters):
        improved = False
        for i in range(N):
            best_r = r
            best_pos = centers[i].copy()
            # Try small moves in random directions
            for _ in range(20):
                direction = np.random.randn(3)
                direction /= np.linalg.norm(direction)
                for s in [step, step/2, step/4]:
                    new_pos = centers[i] + s * direction
                    # Check if new position is inside cube
                    if np.any(np.abs(new_pos) > 0.5):
                        continue
                    old = centers[i]
                    centers[i] = new_pos
                    new_r = _score(centers)
                    if new_r > best_r + 1e-12:
                        best_r = new_r
                        best_pos = new_pos.copy()
                    centers[i] = old
            if best_r > r + 1e-12:
                centers[i] = best_pos
                r = best_r
                improved = True
        if not improved:
            step *= 0.5
            if step < tol:
                break
    return centers, r


def _gradient_ascent(centers, max_iters=5000, lr=0.01):
    """Gradient ascent on the smooth surrogate of the minimum radius."""
    centers = centers.copy()
    r = _score(centers)
    if r <= 0:
        return centers, r
    
    for it in range(max_iters):
        # Compute gradient of the min radius
        # The min radius is min over all constraints
        # For each sphere i, the binding constraints are:
        # - Distance to walls: 0.5 - |c_ik| for each k
        # - Half distance to nearest neighbor: |c_i - c_j|/2
        
        # Compute all pairwise distances and wall distances
        grad = np.zeros_like(centers)
        min_val = np.inf
        min_type = None  # ('wall', i, k) or ('pair', i, j)
        
        for i in range(N):
            for k in range(3):
                d = 0.5 - np.abs(centers[i][k])
                if d < min_val:
                    min_val = d
                    min_type = ('wall', i, k)
        
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(centers[i] - centers[j]) / 2.0
                if d < min_val:
                    min_val = d
                    min_type = ('pair', i, j)
        
        if min_type is None:
            break
        
        # Compute gradient contribution
        if min_type[0] == 'wall':
            _, i, k = min_type
            sign = np.sign(centers[i][k])
            if sign == 0:
                sign = 1.0
            grad[i][k] -= sign  # To increase 0.5 - |c_ik|, decrease |c_ik|
        else:
            _, i, j = min_type
            diff = centers[i] - centers[j]
            norm = np.linalg.norm(diff)
            if norm < 1e-12:
                continue
            diff /= norm
            grad[i] += 0.5 * diff  # To increase |c_i - c_j|/2
            grad[j] -= 0.5 * diff
        
        # Update
        centers += lr * grad
        
        # Project back to cube
        centers = np.clip(centers, -0.5, 0.5)
        
        new_r = _score(centers)
        if new_r < r - 1e-8:
            # Revert if we got worse
            centers -= lr * grad
            centers = np.clip(centers, -0.5, 0.5)
            lr *= 0.5
            if lr < 1e-8:
                break
        else:
            r = new_r
            # Decay learning rate slowly
            lr *= 0.999
    
    return centers, r


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    best_centers = None
    best_r = -1.0
    
    # Try multiple initializations
    initializations = []
    
    # FCC lattice
    fcc = _fcc_lattice(N)
    if len(fcc) >= N:
        initializations.append(fcc)
    
    # BCC lattice
    bcc = _bcc_lattice(N)
    if len(bcc) >= N:
        initializations.append(bcc)
    
    # Random initialization
    for _ in range(5):
        if time.time() > deadline:
            break
        rand = np.random.uniform(-0.45, 0.45, (N, 3))
        initializations.append(rand)
    
    # Grid initialization
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    if len(grid) >= N:
        initializations.append(grid[:N])
    
    for init in initializations:
        if time.time() > deadline:
            break
        if len(init) < N:
            continue
        init = init[:N].copy()
        
        # Local search
        centers, r = _local_search(init, max_iters=1000)
        if r > best_r:
            best_r = r
            best_centers = centers.copy()
        
        # Gradient ascent
        if time.time() < deadline - 20:
            centers, r = _gradient_ascent(centers, max_iters=2000, lr=0.01)
            if r > best_r:
                best_r = r
                best_centers = centers.copy()
    
    if best_centers is None:
        # Fallback: simple grid
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_centers = grid[:N]
    
    # Final validation
    if _score(best_centers) < 0:
        # Ensure validity
        best_centers = np.clip(best_centers, -0.5, 0.5)
    
    return best_centers

# EVOLVE-BLOCK-END
