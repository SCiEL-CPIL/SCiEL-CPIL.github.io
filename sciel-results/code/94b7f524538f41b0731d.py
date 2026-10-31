"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16
D = 2

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize


def _score_from_points(points):
    """Compute (max pairwise dist / min pairwise dist)^2. Lower is better."""
    n = len(points)
    if n < 2:
        return np.inf
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    iu = np.triu_indices(n, k=1)
    pd = dists[iu]
    if len(pd) == 0:
        return np.inf
    dmin = pd.min()
    dmax = pd.max()
    if dmin <= 1e-12:
        return np.inf
    return (dmax / dmin) ** 2


def _sinusoidal_deform(params):
    """
    Build 16 points from a 4x4 base grid with sinusoidal deformation.
    params (9 total):
      a1, a2, a3: amplitudes for x-direction deformation modes
      b1, b2, b3: amplitudes for y-direction deformation modes
      phi1, phi2: phase shifts
      rot: global rotation angle
    """
    a1, a2, a3, b1, b2, b3, phi1, phi2, rot = params

    # Base 4x4 grid centered at origin, normalized to [-1, 1]
    xs = np.array([-1.5, -0.5, 0.5, 1.5], dtype=float) / 1.5
    ys = np.array([-1.5, -0.5, 0.5, 1.5], dtype=float) / 1.5
    GX, GY = np.meshgrid(xs, ys, indexing='ij')
    x = GX.ravel()
    y = GY.ravel()

    # Sinusoidal deformation modes
    # x-deformation: combination of sin/cos in x and y
    dx = (a1 * np.sin(np.pi * x + phi1) +
          a2 * np.sin(2 * np.pi * x) +
          a3 * np.cos(np.pi * y + phi2))

    # y-deformation: combination of sin/cos in y and x
    dy = (b1 * np.sin(np.pi * y + phi2) +
          b2 * np.sin(2 * np.pi * y) +
          b3 * np.cos(np.pi * x + phi1))

    x_new = x + dx
    y_new = y + dy

    # Apply rotation
    c, s = np.cos(rot), np.sin(rot)
    xr = c * x_new - s * y_new
    yr = s * x_new + c * y_new

    return np.column_stack([xr, yr])


def _objective(params):
    """Objective function for DE: minimize the ratio squared."""
    points = _sinusoidal_deform(params)
    d = _score_from_points(points)
    if not np.isfinite(d):
        return 1e10
    return d


def construct():
    # Define bounds for DE (9 parameters)
    bounds = [
        (-0.3, 0.3),   # a1: x-deform mode 1 amplitude
        (-0.3, 0.3),   # a2: x-deform mode 2 amplitude
        (-0.3, 0.3),   # a3: x-deform mode 3 amplitude
        (-0.3, 0.3),   # b1: y-deform mode 1 amplitude
        (-0.3, 0.3),   # b2: y-deform mode 2 amplitude
        (-0.3, 0.3),   # b3: y-deform mode 3 amplitude
        (-np.pi/2, np.pi/2),  # phi1: phase shift 1
        (-np.pi/2, np.pi/2),  # phi2: phase shift 2
        (-np.pi/4, np.pi/4),  # rot: rotation
    ]

    # Run differential evolution with multiple restarts for robustness
    best_params = None
    best_score = np.inf

    for seed in [42, 123, 777]:
        result = differential_evolution(
            _objective,
            bounds,
            maxiter=200,
            popsize=25,
            tol=1e-8,
            seed=seed,
            polish=False,
            init='sobol',
            mutation=(0.4, 1.0),
            recombination=0.8,
            updating='deferred',
        )

        if result.fun < best_score:
            best_score = result.fun
            best_params = result.x

    # Local refinement using L-BFGS-B
    lb = [b[0] for b in bounds]
    ub = [b[1] for b in bounds]

    try:
        local_res = minimize(
            _objective,
            best_params,
            method='L-BFGS-B',
            bounds=list(zip(lb, ub)),
            options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        if local_res.fun < best_score:
            best_score = local_res.fun
            best_params = local_res.x
    except Exception:
        pass

    points = _sinusoidal_deform(best_params)

    # Verify distinctness
    score = _score_from_points(points)
    if not np.isfinite(score):
        # Fallback: regular grid
        side = int(np.ceil(N ** (1 / D)))
        import itertools
        grid = list(itertools.product(range(side), repeat=D))[:N]
        return np.array(grid, dtype=float)

    return points
# EVOLVE-BLOCK-END
