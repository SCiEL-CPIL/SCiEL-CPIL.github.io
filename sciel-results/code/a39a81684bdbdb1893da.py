"""Packing 50 equal circles in a unit square [-1/2, 1/2]^2.
Uses simulated annealing + Nelder-Mead polish to maximize the minimum distance.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
DEADLINE = 140.0  # safety margin before 150s kill
START_TIME = time.time()


def _elapsed():
    return time.time() - START_TIME


def _remaining():
    return max(0.0, DEADLINE - _elapsed())


def _min_dist(pts):
    """Compute the minimum distance between any pair of points and to the boundary."""
    n = len(pts)
    # Pairwise distances
    dx = np.subtract.outer(pts[:, 0], pts[:, 0])
    dy = np.subtract.outer(pts[:, 1], pts[:, 1])
    d = np.sqrt(dx * dx + dy * dy)
    np.fill_diagonal(d, np.inf)
    min_pair = d.min()
    # Distance to boundary
    dist_bound = np.minimum(0.5 - np.abs(pts), 0.5)  # min over x,y for each point
    min_bound = dist_bound.min()
    return min(min_pair, min_bound)


def _score(pts):
    """The radius is half the minimum distance to another circle or to the wall."""
    return _min_dist(pts) / 2.0


def _clamp(pts):
    """Ensure points stay inside [-0.5, 0.5]^2."""
    return np.clip(pts, -0.5, 0.5)


def _simulated_annealing(pts0, n_iter=200000, t_start=0.05, t_end=0.0001,
                         alpha=0.99995, max_step=0.05):
    """Simulated annealing to find a good packing."""
    pts = pts0.copy()
    best_pts = pts.copy()
    best_score = _score(pts)
    n = len(pts)
    rng = np.random.default_rng(42)

    for it in range(n_iter):
        if _remaining() < 10.0:
            break
        # Propose a move: perturb a random subset of points
        k = rng.integers(1, min(5, n))
        idx = rng.choice(n, size=k, replace=False)
        step_size = max_step * (t_start / max(t_start, t_start * (alpha ** it)))
        step_size = max(step_size, 0.001)
        new_pts = pts.copy()
        new_pts[idx] += rng.normal(0, step_size, size=(k, 2))
        new_pts = _clamp(new_pts)

        new_score = _score(new_pts)
        delta = new_score - _score(pts)
        if delta > 0 or rng.random() < np.exp(delta / max(t_start * (alpha ** it), 1e-10)):
            pts = new_pts
            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts.copy()

    return best_pts, best_score


def _nelder_mead_polish(pts):
    """Polish using Nelder-Mead on the negative min-distance objective."""
    def objective(x):
        p = x.reshape(-1, 2)
        p = _clamp(p)
        return -_min_dist(p)

    x0 = pts.flatten()
    res = minimize(objective, x0, method='Nelder-Mead',
                   options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10,
                            'adaptive': True})
    return res.x.reshape(-1, 2), -res.fun / 2.0


def _grid_init():
    """Create an initial grid-based packing."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N]


def _random_init():
    """Random initialization inside the square."""
    rng = np.random.default_rng(0)
    return rng.uniform(-0.5, 0.5, size=(N, 2))


def construct():
    global START_TIME
    START_TIME = time.time()

    # Start with grid, then improve with SA + NM
    pts = _grid_init()
    best_pts = pts.copy()
    best_score = _score(pts)

    # Multiple SA restarts with different initializations
    inits = [pts.copy(), _random_init(), _random_init()]
    for init in inits:
        if _remaining() < 15.0:
            break
        sa_pts, sa_score = _simulated_annealing(
            init, n_iter=150000, t_start=0.04, t_end=0.00005,
            alpha=0.99996, max_step=0.04
        )
        if sa_score > best_score:
            best_score = sa_score
            best_pts = sa_pts.copy()

    # Polish the best with Nelder-Mead
    if _remaining() > 5.0:
        nm_pts, nm_score = _nelder_mead_polish(best_pts)
        if nm_score > best_score:
            best_score = nm_score
            best_pts = nm_pts.copy()

    # Final check: ensure all points are strictly inside
    best_pts = np.clip(best_pts, -0.4999999, 0.4999999)

    return best_pts
