"""Circle packing in a unit square: maximize common radius for 50 equal circles.

Uses SLSQP on the exact min-radius objective with box constraints,
combined with force-directed relaxation and Langevin refinement.
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 50
DEADLINE = 118.0  # safety margin before 150s kill


def _score(pos):
    """Compute the maximum valid radius for given positions."""
    pos = np.asarray(pos, dtype=np.float64)
    if pos.shape != (N, 2):
        return -1.0
    if not np.all(np.isfinite(pos)):
        return -1.0
    # Wall constraint
    wall = 0.5 - np.abs(pos).max(axis=1)
    r_wall = wall.min()
    if r_wall < 0:
        return -1.0
    # Pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1))
    np.fill_diagonal(dist, np.inf)
    r_pair = (dist.min() / 2.0)
    return min(r_wall, r_pair)


def _neg_radius(pos_flat):
    """Negative of the radius objective (to be minimized)."""
    pos = pos_flat.reshape(N, 2)
    if not np.all(np.isfinite(pos)):
        return 1e10
    wall = 0.5 - np.abs(pos).max(axis=1)
    r_wall = wall.min()
    if r_wall < 0:
        return 1e10
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1))
    np.fill_diagonal(dist, np.inf)
    r_pair = (dist.min() / 2.0)
    return -min(r_wall, r_pair)


def _slsqp_optimize(c0, t0, deadline, maxiter=200):
    """Run SLSQP to maximize the minimum radius from initial position c0."""
    c0 = np.asarray(c0, dtype=np.float64).copy()
    x0 = c0.flatten()
    
    # Box constraints: each coordinate in [-0.5, 0.5]
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    try:
        result = minimize(
            _neg_radius, x0, method='SLSQP',
            bounds=bounds,
            options={'maxiter': maxiter, 'ftol': 1e-12, 'disp': False}
        )
        if result.success or result.fun < -0.01:
            c = result.x.reshape(N, 2)
            r = _score(c)
            if r > 0:
                return c, r
    except Exception:
        pass
    return c0, _score(c0)


def _relax(c, R, iters, eta0, eta1, t0=None, deadline=None):
    """
    Relaxation step: push overlapping circles apart and keep them inside the square.
    """
    c = c.copy()
    n = len(c)
    for it in range(iters):
        if t0 is not None and deadline is not None:
            if time.time() - t0 > deadline - 3:
                break
        t = it / max(iters, 1)
        eta = eta0 + (eta1 - eta0) * t
        
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        d = np.sqrt((diff ** 2).sum(-1))
        np.fill_diagonal(d, 1e-12)
        mag = np.maximum(2 * R - d, 0.0) / d
        np.fill_diagonal(mag, 0.0)
        f = (mag[:, :, np.newaxis] * diff).sum(1)
        # Walls
        f[:, 0] += -np.maximum(c[:, 0] - (0.5 - R), 0.0) + np.maximum((-0.5 + R) - c[:, 0], 0.0)
        f[:, 1] += -np.maximum(c[:, 1] - (0.5 - R), 0.0) + np.maximum((-0.5 + R) - c[:, 1], 0.0)
        c += eta * f
        c = np.clip(c, -0.5, 0.5)
    return c


def _grow(start, R0, R1, step, iters, eta0, eta1, t0, deadline):
    """
    Grow the radius from R0 to R1 in steps, relaxing at each step.
    Returns (best_centers, best_radius).
    """
    c = start.copy()
    best = (c.copy(), _score(c))
    R = R0
    while R <= R1 and (time.time() - t0) < deadline - 5:
        c = _relax(c, R, iters, eta0, eta1, t0, deadline)
        r = _score(c)
        if r > best[1]:
            best = (c.copy(), r)
        R += step
        if (time.time() - t0) > deadline - 5:
            break
    return best


def _init_grid_7x7():
    """7x7 grid with one extra circle at center."""
    ticks = (np.arange(7) + 0.5) / 7 - 0.5
    g = np.array([(x, y) for y in ticks for x in ticks])
    extra = np.array([[0.0, 0.0]])
    return np.vstack([g, extra])


def _init_jitter8(rng):
    """8x8 grid, pick 50, add jitter."""
    ticks = (np.arange(8) + 0.5) / 8 - 0.5
    g = np.array([(x, y) for y in ticks for x in ticks])
    idx = rng.permutation(len(g))[:N]
    c = g[idx].copy()
    c += rng.uniform(-0.03, 0.03, c.shape)
    return c


def _init_rand(rng):
    """Random uniform positions."""
    return rng.uniform(-0.45, 0.45, (N, 2))


def _init_hex():
    """Hexagonal lattice starting position."""
    positions = []
    spacing = 0.14
    row_spacing = spacing * np.sqrt(3) / 2
    for row in range(8):
        for col in range(7):
            x = (col - 3) * spacing + (0.5 * spacing if row % 2 else 0)
            y = (row - 3.5) * row_spacing
            positions.append([x, y])
    positions = np.array(positions[:N])
    max_abs = np.abs(positions).max()
    if max_abs > 0.45:
        positions = positions * (0.45 / max_abs)
    return positions


def _init_grid_8x8():
    """8x8 grid, pick 50."""
    ticks = (np.arange(8) + 0.5) / 8 - 0.5
    g = np.array([(x, y) for y in ticks for x in ticks])
    return g[:N].copy()


def _langevin_refine(c, r, t0, deadline, steps=5000, T0=0.001, T1=0.0001, lr=0.002):
    """
    Langevin dynamics refinement: add stochastic kicks to escape local minima.
    """
    c = c.copy()
    best_c = c.copy()
    best_r = r
    rng = np.random.default_rng(999)
    
    for step in range(steps):
        if time.time() - t0 > deadline - 5:
            break
        t = step / steps
        T = T0 + (T1 - T0) * t
        lr_t = lr * (1.0 - 0.5 * t)
        
        # Compute forces (same as relaxation)
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        d = np.sqrt((diff ** 2).sum(-1))
        np.fill_diagonal(d, 1e-12)
        mag = np.maximum(2 * r - d, 0.0) / d
        np.fill_diagonal(mag, 0.0)
        f = (mag[:, :, np.newaxis] * diff).sum(1)
        f[:, 0] += -np.maximum(c[:, 0] - (0.5 - r), 0.0) + np.maximum((-0.5 + r) - c[:, 0], 0.0)
        f[:, 1] += -np.maximum(c[:, 1] - (0.5 - r), 0.0) + np.maximum((-0.5 + r) - c[:, 1], 0.0)
        
        # Langevin update: deterministic + stochastic
        noise = T * rng.standard_normal(c.shape)
        c += lr_t * f + noise
        c = np.clip(c, -0.5, 0.5)
        
        r_new = _score(c)
        if r_new > best_r:
            best_r = r_new
            best_c = c.copy()
    
    return best_c, best_r


def _delaunay_refine(c, r, t0, deadline, max_iter=100):
    """
    Refine using Delaunay triangulation to identify contact graph,
    then solve for optimal positions with those contacts active.
    """
    c = c.copy()
    best_c = c.copy()
    best_r = r
    
    for outer in range(max_iter):
        if time.time() - t0 > deadline - 5:
            break
        
        # Compute current score
        r_cur = _score(c)
        if r_cur > best_r:
            best_r = r_cur
            best_c = c.copy()
        
        # Get Delaunay triangulation
        try:
            tri = Delaunay(c)
        except Exception:
            break
        
        # Build contact graph from Delaunay edges
        edges = set()
        for simplex in tri.simplices:
            for i in range(len(simplex)):
                for j in range(i+1, len(simplex)):
                    edges.add((simplex[i], simplex[j]))
        
        # For each edge, check if it's a "contact" (distance close to 2*r)
        # and build a system to push them apart
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        d = np.sqrt((diff ** 2).sum(-1))
        np.fill_diagonal(d, 1e12)
        
        # Force-directed step with emphasis on Delaunay edges
        f = np.zeros_like(c)
        for (i, j) in edges:
            dij = d[i, j]
            if dij < 2 * r_cur * 1.05:  # near contact
                overlap = 2 * r_cur - dij
                if overlap > 0:
                    direction = (c[i] - c[j]) / max(dij, 1e-12)
                    f[i] += 0.5 * overlap * direction
                    f[j] -= 0.5 * overlap * direction
        
        # Wall forces
        f[:, 0] += -np.maximum(c[:, 0] - (0.5 - r_cur), 0.0) + np.maximum((-0.5 + r_cur) - c[:, 0], 0.0)
        f[:, 1] += -np.maximum(c[:, 1] - (0.5 - r_cur), 0.0) + np.maximum((-0.5 + r_cur) - c[:, 1], 0.0)
        
        # Apply small step
        eta = 0.01 * (1.0 - outer / max_iter)
        c += eta * f
        c = np.clip(c, -0.5, 0.5)
    
    return best_c, best_r


def _slsqp_polish(c, t0, deadline, maxiter=300):
    """Final SLSQP polish with tighter tolerance."""
    c = np.asarray(c, dtype=np.float64).copy()
    x0 = c.flatten()
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    try:
        result = minimize(
            _neg_radius, x0, method='SLSQP',
            bounds=bounds,
            options={'maxiter': maxiter, 'ftol': 1e-14, 'disp': False}
        )
        if result.fun < -0.01:
            c_new = result.x.reshape(N, 2)
            r_new = _score(c_new)
            if r_new > _score(c):
                return c_new, r_new
    except Exception:
        pass
    return c, _score(c)


def construct():
    t0 = time.time()
    rng = np.random.default_rng(12345)
    
    # Multiple diverse initializations
    inits = [
        _init_grid_7x7(),
        _init_jitter8(rng),
        _init_hex(),
        _init_grid_8x8(),
        _init_rand(np.random.default_rng(42)),
        _init_rand(np.random.default_rng(77)),
    ]
    
    best = None
    best_r = -1.0
    
    # Phase 1: Grow radius from each starting configuration
    n_inits = len(inits)
    time_per_init = (DEADLINE - (time.time() - t0) - 30) / n_inits
    
    for i, start in enumerate(inits):
        remaining = DEADLINE - (time.time() - t0)
        if remaining < 15:
            break
        
        budget = min(time_per_init, remaining * 0.85 / (n_inits - i))
        budget = max(budget, 3.0)
        
        # Grow from 0.05 to 0.0715
        c, r = _grow(start, 0.05, 0.0715, 0.0004, 1000, 0.02, 0.002, t0, DEADLINE)
        
        # SLSQP optimization from the grown position
        c_sl, r_sl = _slsqp_optimize(c, t0, DEADLINE, maxiter=100)
        if r_sl > r:
            c, r = c_sl, r_sl
        
        if r > best_r:
            best_r = r
            best = c.copy()
        
        # Check time
        if time.time() - t0 > DEADLINE - 30:
            break
    
    if best is None:
        best = _init_grid_7x7()
        best_r = _score(best)
    
    # Phase 2: Delaunay-based refinement
    remaining = DEADLINE - (time.time() - t0)
    if remaining > 15:
        c, r = _delaunay_refine(best, best_r, t0, DEADLINE, max_iter=50)
        if r > best_r:
            best_r = r
            best = c
    
    # Phase 3: SLSQP polish
    remaining = DEADLINE - (time.time() - t0)
    if remaining > 15:
        c, r = _slsqp_polish(best, t0, DEADLINE, maxiter=200)
        if r > best_r:
            best_r = r
            best = c
    
    # Phase 4: Langevin refinement
    remaining = DEADLINE - (time.time() - t0)
    if remaining > 10:
        c, r = _langevin_refine(best, best_r, t0, DEADLINE, steps=5000, T0=0.001, T1=0.00005, lr=0.002)
        if r > best_r:
            best_r = r
            best = c
    
    # Phase 5: Final SLSQP polish
    remaining = DEADLINE - (time.time() - t0)
    if remaining > 10:
        c, r = _slsqp_polish(best, t0, DEADLINE, maxiter=300)
        if r > best_r:
            best_r = r
            best = c
    
    # Phase 6: One more relaxation + SLSQP
    remaining = DEADLINE - (time.time() - t0)
    if remaining > 8:
        r_cur = _score(best)
        c = _relax(best, r_cur, 2000, 0.01, 0.0005, t0, DEADLINE)
        c, r = _slsqp_polish(c, t0, DEADLINE, maxiter=200)
        if r > best_r:
            best_r = r
            best = c
    
    # Final check
    if not np.all(np.isfinite(best)) or best.shape != (N, 2):
        best = _init_grid_7x7()
    
    return best
