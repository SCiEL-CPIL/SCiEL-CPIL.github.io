"""Physics-based circle packing: repulsive dynamics + SLSQP epigraph polish."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
SEED = 12345
DEADLINE = 140.0  # safety margin below 150s wall-clock


def _score(pts):
    """Compute the max common radius for a configuration."""
    p = np.asarray(pts, dtype=float)
    if p.shape != (N, 2) or not np.all(np.isfinite(p)):
        return 0.0
    # pairwise distances
    d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    r_pairs = d.min() / 2.0
    # boundary clearance
    r_bound = (0.5 - np.abs(p).max())
    return float(min(r_pairs, r_bound))


def _repulsion_force(pts, k=50.0, r_target=0.08):
    """Compute repulsive forces pushing circles apart (and from walls)."""
    p = np.asarray(pts, dtype=float)
    f = np.zeros_like(p)
    # pairwise repulsion
    d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=2)
    np.fill_diagonal(d, 1.0)
    # force magnitude: strong when overlapping
    overlap = np.maximum(0.0, r_target - d)
    # unit vectors
    diff = p[:, None, :] - p[None, :, :]
    mag = d[:, :, None]
    unit = diff / np.maximum(mag, 1e-12)
    # force = k * overlap * unit
    force = k * overlap[:, :, None] * unit
    f += force.sum(axis=0)
    # wall repulsion
    for axis in range(2):
        for sign in [1, -1]:
            dist = 0.5 - sign * p[:, axis]  # distance to wall
            overlap_w = np.maximum(0.0, r_target - dist)
            f[:, axis] -= sign * k * overlap_w
    return f


def _physics_init(seed, max_time=4.0):
    """Run repulsive dynamics to find a good initial configuration."""
    rng = np.random.default_rng(seed)
    # start from random positions
    pts = rng.uniform(-0.4, 0.4, size=(N, 2))
    # also try a grid start
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])[:N]
    pts = pts.copy()
    
    dt = 0.002
    damping = 0.95
    best = _score(pts)
    best_pts = pts.copy()
    
    t0 = time.time()
    vel = np.zeros_like(pts)
    while time.time() - t0 < max_time:
        f = _repulsion_force(pts)
        vel = damping * vel + dt * f
        pts = pts + vel
        # clamp to square
        pts[:, 0] = np.clip(pts[:, 0], -0.48, 0.48)
        pts[:, 1] = np.clip(pts[:, 1], -0.48, 0.48)
        s = _score(pts)
        if s > best:
            best = s
            best_pts = pts.copy()
        # adaptive: reduce step if score drops
        if s < best - 0.005:
            dt *= 0.9
            damping = 0.9
    return best_pts


def _slsqp_polish(pts, time_budget=30.0):
    """SLSQP epigraph polish: maximize r subject to constraints."""
    p0 = np.asarray(pts, dtype=float).copy()
    
    def neg_radius(x):
        """Objective: minimize -r (maximize r)."""
        p = x.reshape(N, 2)
        if not np.all(np.isfinite(p)):
            return 1e6
        d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        r_pairs = d.min() / 2.0
        r_bound = 0.5 - np.abs(p).max()
        return -min(r_pairs, r_bound)
    
    def radius_with_eps(x):
        """Compute radius with small epsilon for smoothness."""
        p = x.reshape(N, 2)
        d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        r_pairs = d.min() / 2.0
        r_bound = 0.5 - np.abs(p).max()
        return min(r_pairs, r_bound)
    
    # Constraints: pairwise distance >= 2*r, boundary
    # Use epigraph: introduce r as variable, maximize r
    # x = [pts (2N), r]
    def objective(x):
        p = x[:2*N].reshape(N, 2)
        r = x[2*N]
        if not np.all(np.isfinite(p)) or not np.isfinite(r):
            return 1e6
        # minimize -r
        return -r
    
    def constraint_pairwise(x, i, j):
        p = x[:2*N].reshape(N, 2)
        r = x[2*N]
        d = np.linalg.norm(p[i] - p[j])
        return d - 2.0 * r
    
    def constraint_boundary(x, i, k):
        p = x[:2*N].reshape(N, 2)
        r = x[2*N]
        # |p_ik| + r <= 0.5
        return 0.5 - np.abs(p[i, k]) - r
    
    # Build constraints
    constraints = []
    for i in range(N):
        for j in range(i+1, N):
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i, j=j: constraint_pairwise(x, i, j)})
        for k in range(2):
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i, k=k: constraint_boundary(x, i, k)})
    
    # Bounds on r
    r0 = radius_with_eps(p0)
    x0 = np.concatenate([p0.ravel(), [r0]])
    bounds = [(-0.5, 0.5)] * (2*N) + [(0.0, 0.5)]
    
    t0 = time.time()
    result = minimize(
        objective, x0, method='SLSQP',
        bounds=bounds, constraints=constraints,
        options={'maxiter': 200, 'ftol': 1e-12}
    )
    
    if result.success or result.fun < -r0:
        p_opt = result.x[:2*N].reshape(N, 2)
        r_opt = result.x[2*N]
        # verify
        s = _score(p_opt)
        if s > r0:
            return p_opt
    return p0


def construct():
    t_start = time.time()
    rng = np.random.default_rng(SEED)
    
    best_pts = None
    best_score = 0.0
    
    # Multiple random restarts with physics initialization
    n_restarts = 8
    for i in range(n_restarts):
        if time.time() - t_start > DEADLINE - 60:
            break
        
        # Physics-based initialization
        pts = _physics_init(seed=SEED + i * 1000, max_time=3.0)
        s = _score(pts)
        
        if s > best_score:
            best_score = s
            best_pts = pts.copy()
        
        # SLSQP polish if time allows
        if time.time() - t_start < DEADLINE - 40:
            polished = _slsqp_polish(pts, time_budget=30.0)
            s_pol = _score(polished)
            if s_pol > best_score:
                best_score = s_pol
                best_pts = polished.copy()
    
    # Final micro-polish with more iterations
    if best_pts is not None and time.time() - t_start < DEADLINE - 10:
        # Try a few more SLSQP runs with different starting points
        for i in range(3):
            if time.time() - t_start > DEADLINE - 5:
                break
            # perturb best
            perturbed = best_pts + rng.normal(0, 0.001, size=(N, 2))
            perturbed[:, 0] = np.clip(perturbed[:, 0], -0.49, 0.49)
            perturbed[:, 1] = np.clip(perturbed[:, 1], -0.49, 0.49)
            polished = _slsqp_polish(perturbed, time_budget=20.0)
            s_pol = _score(polished)
            if s_pol > best_score:
                best_score = s_pol
                best_pts = polished.copy()
    
    # Ensure validity
    if best_pts is None:
        # fallback to grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        best_pts = np.array([(x, y) for y in ticks for x in ticks])[:N]
    
    # Final validation and clamping
    best_pts[:, 0] = np.clip(best_pts[:, 0], -0.5 + 1e-10, 0.5 - 1e-10)
    best_pts[:, 1] = np.clip(best_pts[:, 1], -0.5 + 1e-10, 0.5 - 1e-10)
    
    return best_pts
