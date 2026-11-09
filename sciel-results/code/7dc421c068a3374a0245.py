"""
Sphere packing in unit cube: 50 equal spheres.

Strategy: Physics-based force-directed repulsion.
- Each sphere exerts a repulsive force on every other sphere: F = k / d^4 along the separation vector.
- Walls exert a restoring force pushing spheres inward when too close.
- Use adaptive gradient descent with momentum to find a local maximum of the minimum radius.
- Multiple restarts with different initializations.
- Final polish with Nelder-Mead on the actual objective.
"""

import numpy as np
import time
from scipy.optimize import minimize

N = 50
CUBE_HALF = 0.5


def _compute_radius(centers):
    """Compute the largest common radius for given centers."""
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    centers = np.clip(centers, -CUBE_HALF, CUBE_HALF)

    # Wall distances
    wall = CUBE_HALF - np.abs(centers)
    min_wall = np.min(wall, axis=1)
    global_wall = np.min(min_wall)

    # Pairwise distances
    diff = centers[:, None, :] - centers[None, :, :]
    dists_sq = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(dists_sq, np.inf)
    min_pair = np.sqrt(np.min(dists_sq))

    return min(min_pair / 2.0, global_wall)


def _force_and_score(centers):
    """
    Compute net force on each center and the current radius.
    Force = repulsion between spheres + wall pressure.
    Returns (forces, radius).
    """
    centers = np.asarray(centers, dtype=np.float64)
    forces = np.zeros_like(centers)

    # Pairwise repulsion: F_ij = k * (r_target - d)^3 / d^2 * direction  (for d < r_target)
    # Use inverse-power: F = k / d^4 * direction  (always repulsive, stronger when close)
    # Actually, use a smoother potential: U = k / d^2, F = 2k/d^3 * direction
    # Let's use: F = k / d^4 (stronger repulsion at short range)
    
    k = 1.0
    diff = centers[:, None, :] - centers[None, :, :]  # (N, N, 3)
    dists_sq = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(dists_sq, 1e-10)
    dists = np.sqrt(dists_sq)
    
    # Force magnitude: k / d^4, direction: diff / d
    # F_ij = k * diff_ij / d_ij^4  (force on i from j)
    # But we want repulsion: if i is at position p_i, j at p_j,
    # force on i should push i away from j: direction = (p_i - p_j) / d = diff_ij / d
    # So F_on_i = sum_j k * diff_ij / d_ij^4
    # = sum_j k * diff_ij / (dists_sq_ij^2)
    
    # Avoid division by zero on diagonal
    inv_d4 = np.zeros_like(dists_sq)
    mask = dists_sq > 1e-12
    inv_d4[mask] = 1.0 / (dists_sq[mask] ** 2)
    np.fill_diagonal(inv_d4, 0.0)
    
    # forces[i] = sum_j k * diff[i,j] * inv_d4[i,j]
    forces += k * np.einsum('ij,ijk->ik', inv_d4, diff)
    
    # Wall pressure: push inward if close to wall
    # For each coordinate, if center[k] > 0 - margin, push left; if < 0 + margin, push right
    # Use a linear spring: F_wall = -c * (center - clamp(center, -inner, inner))
    # Or simply: if center > CUBE_HALF - r_est, push toward center
    # Better: always apply a gentle centering force + strong wall repulsion
    
    # Wall repulsion: for each axis, if |c| > threshold, push back
    # Use: F_wall = -c_wall * (|c| - thresh)^2 * sign(c) for |c| > thresh
    # Simpler: linear spring toward center with strength increasing near walls
    
    # Let's use: F_wall = -c1 * c - c2 * c^3  (odd function, pushes toward center)
    # Near walls, c2 term dominates
    c1 = 0.1
    c2 = 5.0
    forces -= c1 * centers + c2 * centers ** 3
    
    # Compute radius
    r = _compute_radius(centers)
    return forces, r


def _force_descent(centers, deadline, lr_init=0.001, momentum=0.9):
    """
    Adaptive gradient descent on the force-directed potential.
    We want to maximize the minimum radius, which is equivalent to
    minimizing the maximum overlap. The force field naturally pushes
    spheres apart.
    """
    current = centers.copy()
    vel = np.zeros_like(current)
    current_score = _compute_radius(current)
    best = current.copy()
    best_score = current_score
    
    lr = lr_init
    step = 0
    no_improve = 0
    
    while time.time() < deadline:
        forces, r = _force_and_score(current)
        
        # Adaptive learning rate: reduce if score drops
        if r < current_score * 0.99:
            lr *= 0.7
            no_improve += 1
        else:
            lr *= 1.02
            if r > current_score:
                no_improve = 0
            else:
                no_improve += 1
        
        # Momentum update
        vel = momentum * vel + lr * forces
        current = current + vel
        
        # Project back into cube
        current = np.clip(current, -CUBE_HALF, CUBE_HALF)
        
        new_score = _compute_radius(current)
        if new_score > best_score:
            best_score = new_score
            best = current.copy()
        
        current_score = new_score
        step += 1
        
        # Adaptive: if stuck for too long, perturb
        if no_improve > 50 and step % 100 == 0:
            current += np.random.normal(0, 0.01, current.shape)
            current = np.clip(current, -CUBE_HALF, CUBE_HALF)
            vel *= 0.5
            no_improve = 0
    
    return best, best_score


def _nelder_mead_polish(centers, deadline):
    """Polish with Nelder-Mead on the actual objective."""
    x0 = centers.flatten()
    
    def obj(x):
        c = np.clip(x.reshape(N, 3), -CUBE_HALF, CUBE_HALF)
        return -_compute_radius(c)
    
    try:
        res = minimize(
            obj, x0,
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        new_centers = np.clip(res.x.reshape(N, 3), -CUBE_HALF, CUBE_HALF)
        new_score = _compute_radius(new_centers)
        return new_centers, new_score
    except Exception:
        return centers, _compute_radius(centers)


def _initial_grid():
    """4x4x4 grid, take 50 centers."""
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    pts = []
    for z in ticks:
        for y in ticks:
            for x in ticks:
                pts.append([x, y, z])
    pts = np.array(pts[:N])
    return pts


def _initial_fcc():
    """FCC-like lattice initialization."""
    # FCC has points at (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0) in a unit cell
    # Scale to fit in [-0.5, 0.5]^3
    pts = []
    # Generate FCC points in a 2x2x2 supercell
    for ix in range(2):
        for iy in range(2):
            for iz in range(2):
                base = np.array([ix, iy, iz]) * 0.5
                # FCC positions within cell
                pts.append(base + np.array([0, 0, 0]))
                pts.append(base + np.array([0, 0.25, 0.25]))
                pts.append(base + np.array([0.25, 0, 0.25]))
                pts.append(base + np.array([0.25, 0.25, 0]))
    # Center the whole thing
    pts = np.array(pts)
    pts -= pts.mean(axis=0)
    # Scale to fit in cube
    max_abs = np.max(np.abs(pts))
    if max_abs > 0:
        pts *= 0.45 / max_abs
    # Take first 50
    return pts[:N]


def _initial_random(seed=None):
    if seed is not None:
        np.random.seed(seed)
    return np.random.uniform(-0.45, 0.45, (N, 3))


def _initial_halton(seed=None):
    """Halton sequence initialization."""
    if seed is not None:
        np.random.seed(seed)
    
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r
    
    pts = []
    for i in range(1, N + 1):
        x = halton(i, 2) - 0.5
        y = halton(i, 3) - 0.5
        z = halton(i, 5) - 0.5
        pts.append([x, y, z])
    return np.array(pts)


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    np.random.seed(12345)
    
    best_centers = None
    best_score = -np.inf
    
    # Phase 1: Force-directed optimization from multiple starts
    initializers = [
        lambda: _initial_grid(),
        lambda: _initial_fcc(),
        lambda: _initial_halton(1),
        lambda: _initial_random(1),
        lambda: _initial_random(2),
        lambda: _initial_random(3),
        lambda: _initial_random(4),
        lambda: _initial_random(5),
    ]
    
    # Allocate time per start
    time_per_start = min(12.0, (deadline - start_time) / len(initializers))
    
    for i, init_func in enumerate(initializers):
        if time.time() > deadline - 10.0:
            break
        
        centers_init = init_func()
        init_score = _compute_radius(centers_init)
        if init_score > best_score:
            best_score = init_score
            best_centers = centers_init.copy()
        
        # Force-directed optimization
        stage_deadline = min(deadline - 8.0, time.time() + time_per_start)
        centers_opt, score_opt = _force_descent(centers_init, stage_deadline)
        
        if score_opt > best_score:
            best_score = score_opt
            best_centers = centers_opt.copy()
        
        # Nelder-Mead polish on the force-directed result
        if time.time() < deadline - 5.0:
            nm_deadline = min(deadline - 4.0, time.time() + 3.0)
            centers_nm, score_nm = _nelder_mead_polish(centers_opt, nm_deadline)
            if score_nm > best_score:
                best_score = score_nm
                best_centers = centers_nm.copy()
        
        # Adjust time for remaining starts
        remaining = deadline - 10.0 - time.time()
        remaining_starts = len(initializers) - i - 1
        if remaining_starts > 0:
            time_per_start = max(4.0, remaining / (remaining_starts * 2))
    
    # Phase 2: Final polish on best solution
    if best_centers is not None and time.time() < deadline - 5.0:
        polish_deadline = deadline - 2.0
        centers_polish, score_polish = _nelder_mead_polish(best_centers, polish_deadline)
        if score_polish > best_score:
            best_score = score_polish
            best_centers = centers_polish.copy()
        
        # One more force-directed pass on the polished result
        if time.time() < deadline - 2.0:
            fd_deadline = deadline - 1.0
            centers_fd, score_fd = _force_descent(best_centers, fd_deadline, lr_init=0.0005)
            if score_fd > best_score:
                best_score = score_fd
                best_centers = centers_fd.copy()
    
    # Fallback
    if best_centers is None:
        best_centers = _initial_grid()
        best_score = _compute_radius(best_centers)
    
    # Final validation
    best_centers = np.clip(best_centers, -CUBE_HALF, CUBE_HALF)
    if not np.all(np.isfinite(best_centers)):
        best_centers = _initial_grid()
    
    return best_centers


def _score(centres):
    return _compute_radius(centres)
