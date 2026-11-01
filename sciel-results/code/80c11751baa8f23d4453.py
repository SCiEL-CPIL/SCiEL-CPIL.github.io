"""
Simulated annealing + local refinement for packing 50 equal circles in a unit square.
Structural change: replace gradient-free deterministic optimizers with a stochastic
simulated annealing search that directly maximizes the minimum clearance (pairwise
distance / 2 and wall distance), followed by a fast local polish.
"""

import numpy as np
import time
from scipy.spatial import cKDTree

N = 50
HALF = 0.5


def _compute_radius(points):
    """
    Compute the maximum radius such that circles centered at points
    are inside [-0.5, 0.5]^2 and non-overlapping.
    r = min(
        min_{i<j} ||p_i - p_j|| / 2,
        min_i min_k (0.5 - |p_ik|)
    )
    """
    n = len(points)
    if n == 0:
        return 0.0

    # Distance to walls (vectorized)
    wall_dists = np.minimum(HALF - np.abs(points[:, 0]), HALF - np.abs(points[:, 1]))
    min_wall = float(np.min(wall_dists))

    # Minimum pairwise distance via cKDTree
    tree = cKDTree(points)
    # Query nearest neighbors for each point (k=2 to get self + 1 neighbor)
    dists, _ = tree.query(points, k=2)
    min_pair = float(np.min(dists[:, 1]))

    r = min(min_pair / 2.0, min_wall)
    return r if r > 0 else 0.0


def _neg_radius(params):
    """Objective: minimize negative radius (i.e., maximize radius)."""
    points = params.reshape(-1, 2)
    if np.any(np.abs(points) > HALF + 1e-10):
        return 1e10
    if not np.all(np.isfinite(points)):
        return 1e10
    r = _compute_radius(points)
    return -r


def _hex_initial():
    """Hexagonal lattice initial placement, clipped to square."""
    # Use a hexagonal grid with spacing chosen so ~50 points fit
    # Hexagonal packing: rows offset by half spacing
    spacing = 0.14  # approximate
    pts = []
    row = 0
    while len(pts) < N:
        y = -HALF + spacing * (row + 0.5)
        if y > HALF:
            break
        offset = 0 if row % 2 == 0 else spacing * 0.5
        x = -HALF + spacing * 0.5 + offset
        col = 0
        while x < HALF and len(pts) < N:
            pts.append([x, y])
            x += spacing
            col += 1
        row += 1
    pts = np.array(pts[:N])
    # Jitter slightly to break symmetry
    rng = np.random.RandomState(42)
    pts += rng.uniform(-0.005, 0.005, size=pts.shape)
    pts = np.clip(pts, -HALF + 0.01, HALF - 0.01)
    return pts


def _grid_init():
    """Create initial grid placement."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N]


def _simulated_annealing(points, t0=0.01, t_end=1e-6, cooling=0.9995, max_iter=200000, deadline=None):
    """
    Simulated annealing to maximize the minimum clearance.
    We minimize the negative radius.
    """
    rng = np.random.RandomState(12345)
    current = points.copy()
    current_r = _compute_radius(current)
    best = current.copy()
    best_r = current_r

    n = len(current)
    t = t0

    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break

        # Pick a random point to perturb
        idx = rng.randint(n)
        old_pos = current[idx].copy()

        # Perturb with Gaussian noise, scale with temperature
        sigma = 0.005 * (t / t0) ** 0.5
        current[idx] += rng.normal(0, sigma, size=2)

        # Keep inside square
        current[idx] = np.clip(current[idx], -HALF + 1e-12, HALF - 1e-12)

        new_r = _compute_radius(current)
        delta = new_r - current_r  # positive is improvement

        if delta > 0 or rng.random() < np.exp(delta / max(t, 1e-12)):
            current_r = new_r
            if new_r > best_r:
                best_r = new_r
                best = current.copy()
        else:
            current[idx] = old_pos

        # Cool down
        t *= cooling
        if t < t_end:
            t = t_end

    return best, best_r


def _local_polish(points, deadline=None):
    """
    Fast local refinement: iteratively push the most constrained points apart.
    This is a simple gradient-free local search.
    """
    rng = np.random.RandomState(777)
    current = points.copy()
    current_r = _compute_radius(current)
    best = current.copy()
    best_r = current_r

    n = len(current)
    step_size = 0.002
    max_iter = 50000

    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break

        # Find the point with the smallest clearance
        wall_dists = np.minimum(HALF - np.abs(current[:, 0]), HALF - np.abs(current[:, 1]))
        tree = cKDTree(current)
        dists, _ = tree.query(current, k=2)
        pair_dists = dists[:, 1]
        clearances = np.minimum(wall_dists, pair_dists / 2.0)
        worst_idx = int(np.argmin(clearances))

        # Try to move the worst point in a random direction
        direction = rng.normal(0, 1, size=2)
        direction /= (np.linalg.norm(direction) + 1e-15)

        # Try several step sizes
        improved = False
        for ss in [step_size, step_size * 0.5, step_size * 2.0]:
            trial = current.copy()
            trial[worst_idx] = current[worst_idx] + direction * ss
            trial[worst_idx] = np.clip(trial[worst_idx], -HALF + 1e-12, HALF - 1e-12)
            trial_r = _compute_radius(trial)
            if trial_r > current_r:
                current = trial
                current_r = trial_r
                if current_r > best_r:
                    best_r = current_r
                    best = current.copy()
                improved = True
                break

        if not improved:
            # Try moving in opposite direction
            for ss in [step_size, step_size * 0.5]:
                trial = current.copy()
                trial[worst_idx] = current[worst_idx] - direction * ss
                trial[worst_idx] = np.clip(trial[worst_idx], -HALF + 1e-12, HALF - 1e-12)
                trial_r = _compute_radius(trial)
                if trial_r > current_r:
                    current = trial
                    current_r = trial_r
                    if current_r > best_r:
                        best_r = current_r
                        best = current.copy()
                    break

        # Occasionally jitter all points slightly
        if it % 5000 == 0 and it > 0:
            current += rng.normal(0, 0.001, size=current.shape)
            current = np.clip(current, -HALF + 1e-12, HALF - 1e-12)
            current_r = _compute_radius(current)
            if current_r > best_r:
                best_r = current_r
                best = current.copy()

    return best, best_r


def construct():
    deadline = time.time() + 135.0  # Safety margin before 150s kill

    # Multiple initializations
    inits = [
        _grid_init(),
        _hex_initial(),
    ]

    # Add a random initialization
    rng = np.random.RandomState(999)
    random_init = rng.uniform(-HALF + 0.05, HALF - 0.05, size=(N, 2))
    inits.append(random_init)

    best_points = None
    best_r = 0.0

    for init in inits:
        if time.time() > deadline:
            break

        # Phase 1: Simulated annealing
        sa_points, sa_r = _simulated_annealing(
            init,
            t0=0.008,
            t_end=1e-7,
            cooling=0.9997,
            max_iter=100000,
            deadline=deadline
        )
        if sa_r > best_r:
            best_r = sa_r
            best_points = sa_points

        # Phase 2: Local polish
        if time.time() > deadline:
            break
        polish_points, polish_r = _local_polish(
            sa_points,
            deadline=deadline
        )
        if polish_r > best_r:
            best_r = polish_r
            best_points = polish_points

    # Fallback
    if best_points is None:
        best_points = _grid_init()
        best_r = _compute_radius(best_points)

    # Final validation
    best_points = np.clip(best_points, -HALF + 1e-12, HALF - 1e-12)
    if not np.all(np.isfinite(best_points)) or best_points.shape != (N, 2):
        best_points = _grid_init()

    return best_points


# EVOLVE-BLOCK-START
# (The above construct function is the entry point)
# EVOLVE-BLOCK-END
