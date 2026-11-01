"""Optimize 14 points in 3D to minimize (max_pairwise_dist / min_pairwise_dist)^2."""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 14
D = 3
_DEADLINE = 115.0  # seconds, safety margin before 120s kill


def _pairwise_distances(pts):
    """Compute all pairwise squared distances for n points."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    sq = np.sum(diff * diff, axis=2)
    # Zero out diagonal
    iu = np.triu_indices(N, 1)
    return sq[iu]


def _objective_flat(x):
    """Objective: (max_d / min_d)^2 where d are pairwise distances.
    x is flat array of shape (N*D,)."""
    pts = x.reshape(N, D)
    sq = _pairwise_distances(pts)
    min_sq = np.min(sq)
    max_sq = np.max(sq)
    if min_sq < 1e-14:
        return 1e10
    return max_sq / min_sq


def _soft_objective(x, beta=50.0):
    """Smooth surrogate using log-sum-exp for max and soft-min for min."""
    pts = x.reshape(N, D)
    sq = _pairwise_distances(pts)
    # Soft max via log-sum-exp
    m = np.max(sq)
    logsumexp = m + np.log(np.sum(np.exp(sq - m)))
    # Soft min: -logsumexp(-sq)
    m2 = np.max(-sq)
    logsumexp_neg = m2 + np.log(np.sum(np.exp(-sq - m2)))
    soft_min = -logsumexp_neg
    if soft_min < 1e-14:
        return 1e10
    return logsumexp / soft_min


def _tetrahedron_base(scale=1.0):
    """Vertices of a regular tetrahedron centered at origin."""
    v = np.array([
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1]
    ], dtype=float) * scale
    return v


def _make_stella_octangula(rot_angle, trans):
    """Two interlocking regular tetrahedra (stella octangula).
    rot_angle: rotation angle of second tetra around z-axis
    trans: translation vector (3,)
    Returns 8 points."""
    t1 = _tetrahedron_base(1.0)
    t2 = _tetrahedron_base(1.0)
    # Rotate t2 around z-axis
    c, s = np.cos(rot_angle), np.sin(rot_angle)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    t2 = t2 @ R.T + trans
    return np.vstack([t1, t2])


def _make_initial_stella():
    """Create initial 14-point configuration: stella octangula (8) + 6 extra points.
    Extra points placed at midpoints of cube edges or similar symmetric positions."""
    pts = _make_stella_octangula(np.pi / 3, np.zeros(3))
    # Add 6 more points: place them at positions that maintain good symmetry
    # Use cube face centers or similar
    # Actually, let's just use a grid-like extension for now
    extra = np.array([
        [0, 0, 2.0],
        [0, 0, -2.0],
        [2.0, 0, 0],
        [-2.0, 0, 0],
        [0, 2.0, 0],
        [0, -2.0, 0]
    ])
    return np.vstack([pts, extra])


def _make_random_initial(rng):
    """Generate a random initial configuration."""
    return rng.random((N, D)) * 4.0 - 2.0


def construct():
    start_time = time.time()
    
    # Best result tracking
    best_pts = None
    best_score = np.inf
    
    def _update_best(pts, score):
        nonlocal best_pts, best_score
        if score < best_score:
            best_score = score
            best_pts = pts.copy()
    
    # Initial configurations to try
    init_configs = []
    
    # 1. Stella octangula based
    init_configs.append(_make_initial_stella())
    
    # 2. Random starts
    rng = np.random.default_rng(42)
    for _ in range(3):
        init_configs.append(_make_random_initial(rng))
    
    # 3. Simple grid
    side = int(np.ceil(N ** (1 / D)))
    import itertools
    grid = list(itertools.product(range(side), repeat=D))[:N]
    init_configs.append(np.array(grid, dtype=float))
    
    # Evaluate initial configs
    for cfg in init_configs:
        score = _objective_flat(cfg.flatten())
        _update_best(cfg, score)
    
    # Check deadline
    if time.time() - start_time > _DEADLINE - 30:
        return best_pts
    
    # Phase 1: Differential evolution on a subset of good starts
    # Use the best initial config as center for DE bounds
    center = best_pts.flatten()
    bounds = [(c - 3.0, c + 3.0) for c in center]
    
    try:
        result = differential_evolution(
            _objective_flat,
            bounds=bounds,
            seed=42,
            maxiter=30,
            popsize=8,
            tol=1e-6,
            polish=True,
            init='latinhypercube',
            callback=lambda x, convergence: None
        )
        if result.success and result.fun < best_score:
            _update_best(result.x.reshape(N, D), result.fun)
    except Exception:
        pass
    
    # Check deadline
    if time.time() - start_time > _DEADLINE - 20:
        return best_pts
    
    # Phase 2: Gradient-based refinement (L-BFGS-B) from multiple starts
    starts = [best_pts.flatten()]
    # Add perturbations
    for _ in range(4):
        perturbed = best_pts.flatten() + rng.normal(0, 0.1, size=N * D)
        starts.append(perturbed)
    
    for s in starts:
        if time.time() - start_time > _DEADLINE - 10:
            break
        try:
            res = minimize(
                _objective_flat,
                s,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res.fun < best_score:
                _update_best(res.x.reshape(N, D), res.fun)
        except Exception:
            pass
    
    # Check deadline
    if time.time() - start_time > _DEADLINE - 5:
        return best_pts
    
    # Phase 3: High-precision polish with tight tolerance
    if best_pts is not None:
        try:
            res = minimize(
                _objective_flat,
                best_pts.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-10}
            )
            if res.fun < best_score:
                _update_best(res.x.reshape(N, D), res.fun)
        except Exception:
            pass
    
    # Phase 4: Try soft objective refinement if time allows
    if time.time() - start_time < _DEADLINE - 5 and best_pts is not None:
        try:
            res = minimize(
                lambda x: _soft_objective(x, beta=100.0),
                best_pts.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            # Evaluate with true objective
            true_score = _objective_flat(res.x)
            if true_score < best_score:
                _update_best(res.x.reshape(N, D), true_score)
        except Exception:
            pass
    
    if best_pts is None:
        # Fallback: return grid
        import itertools
        side = int(np.ceil(N ** (1 / D)))
        grid = list(itertools.product(range(side), repeat=D))[:N]
        return np.array(grid, dtype=float)
    
    return best_pts
