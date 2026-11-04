"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

N = 11

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
TRI_AREA = 0.5 * np.sqrt(3) / 2  # area of unit equilateral triangle

# Precompute indices for all triples
_IDX_TRIPLES = np.array([(i, j, k) for i in range(N) for j in range(i + 1, N) for k in range(j + 1, N)])


def _bary_to_cart(u, v):
    """Convert 2D barycentric coords (u, v) to Cartesian (x, y). w = 1-u-v."""
    w = 1.0 - u - v
    x = v * 1.0 + w * 0.5
    y = w * np.sqrt(3) / 2
    return np.column_stack((x, y))


def _all_dets_from_bary(bary):
    """Compute all 165 determinants (2*area ratio) from 2D barycentric coords (N,2)."""
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v
    P = np.column_stack((u, v, w))

    i_idx = _IDX_TRIPLES[:, 0]
    j_idx = _IDX_TRIPLES[:, 1]
    k_idx = _IDX_TRIPLES[:, 2]

    Pi = P[i_idx]
    Pj = P[j_idx]
    Pk = P[k_idx]

    det = (Pi[:, 0] * (Pj[:, 1] * Pk[:, 2] - Pj[:, 2] * Pk[:, 1])
           - Pi[:, 1] * (Pj[:, 0] * Pk[:, 2] - Pj[:, 2] * Pk[:, 0])
           + Pi[:, 2] * (Pj[:, 0] * Pk[:, 1] - Pj[:, 1] * Pk[:, 0]))
    return det


def _min_area_ratio_from_bary(bary):
    """Returns the minimum |det| (proportional to min area ratio)."""
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v

    if np.any(u < -1e-9) or np.any(v < -1e-9) or np.any(w < -1e-9):
        return -1.0

    dets = _all_dets_from_bary(bary)
    return np.min(np.abs(dets))


def _objective_smooth(x, beta, penalty_lambda=100.0):
    """
    Smooth objective for L-BFGS-B with epigraph formulation.
    x: (2N,) array of u, v coords.
    """
    bary = x.reshape(N, 2)
    u = bary[:, 0]
    v = bary[:, 1]
    w = 1.0 - u - v

    # Penalty for invalid barycentric coords
    penalty = 0.0
    excess = np.maximum(0, u + v - 1.0)
    penalty += penalty_lambda * np.sum(excess ** 2)
    neg_u = np.maximum(0, -u)
    penalty += penalty_lambda * np.sum(neg_u ** 2)
    neg_v = np.maximum(0, -v)
    penalty += penalty_lambda * np.sum(neg_v ** 2)
    neg_w = np.maximum(0, -w)
    penalty += penalty_lambda * np.sum(neg_w ** 2)

    dets = _all_dets_from_bary(bary)
    areas = np.abs(dets)

    if len(areas) == 0:
        return penalty

    m = np.min(areas)
    shifted = areas - m
    # Log-sum-exp trick for numerical stability
    log_sum_exp = np.log(np.sum(np.exp(-beta * shifted)))

    smooth_min = m - (1.0 / beta) * log_sum_exp

    return -smooth_min + penalty


def _objective_hard(x):
    """
    Hard objective for SLSQP: minimize -min_area.
    x: (2N,) array.
    """
    bary = x.reshape(N, 2)
    u = np.clip(bary[:, 0], 0, None)
    v = np.clip(bary[:, 1], 0, None)
    s = u + v
    mask = s > 1.0
    if np.any(mask):
        u[mask] /= s[mask]
        v[mask] /= s[mask]

    bary_clamped = np.column_stack((u, v))
    dets = _all_dets_from_bary(bary_clamped)
    min_area = np.min(np.abs(dets))
    return -min_area


def _project_to_bary_simplex(bary):
    """Project 2D barycentric coords to valid simplex."""
    u = np.maximum(0, bary[:, 0])
    v = np.maximum(0, bary[:, 1])
    s = u + v
    mask = s > 1.0
    if np.any(mask):
        u[mask] /= s[mask]
        v[mask] /= s[mask]
    return np.column_stack((u, v))


def _structured_init_fps():
    """Farthest Point Sampling on a grid of barycentric coordinates."""
    n_grid = 20
    candidates = []
    for i in range(n_grid + 1):
        for j in range(n_grid + 1 - i):
            u = i / n_grid
            v = j / n_grid
            candidates.append([u, v])
    candidates = np.array(candidates)

    selected = [np.array([1 / 3, 1 / 3])]

    for _ in range(N - 1):
        dists = []
        for c in candidates:
            min_d = np.min(np.linalg.norm(c - np.array(selected), axis=1))
            dists.append(min_d)
        dists = np.array(dists)
        max_idx = np.argmax(dists)
        selected.append(candidates[max_idx])

    return np.array(selected)


def _generate_diverse_inits():
    """Generate multiple diverse initial configurations in 2D barycentric coords."""
    inits = []

    # 1. FPS-based initialization
    inits.append(_structured_init_fps())

    # 2. Grid-based initialization
    grid_bary = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
        [0.5, 0.0], [0.0, 0.5], [0.5, 0.5],
        [1 / 3, 1 / 3],
        [0.25, 0.0], [0.75, 0.0], [0.25, 0.25], [0.5, 0.25]
    ])
    inits.append(_project_to_bary_simplex(grid_bary))

    # 3. Hexagonal lattice pattern
    bary_hex = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
        [0.5, 0.0], [0.5, 0.5], [0.0, 0.5],
        [1 / 3, 1 / 3],
        [0.25, 0.0], [0.75, 0.0], [0.25, 0.25], [0.5, 0.75]
    ])
    inits.append(_project_to_bary_simplex(bary_hex))

    # 4. Concentric hexagons
    bary_conc = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
        [0.333, 0.0], [0.667, 0.0], [0.0, 0.333],
        [0.0, 0.667], [0.333, 0.333], [0.667, 0.333],
        [0.333, 0.667], [0.5, 0.5]
    ])
    inits.append(_project_to_bary_simplex(bary_conc))

    # 5. Random initializations
    for seed in [1, 7, 13, 21, 100, 200, 300]:
        rng = np.random.RandomState(seed)
        bary = np.zeros((N, 2))
        for i in range(N):
            u = rng.random()
            v = rng.random()
            while u + v > 1.0:
                u = rng.random()
                v = rng.random()
            bary[i] = [u, v]
        inits.append(bary)

    return inits


def _local_optimize_smooth(x0, beta=100.0, maxiter=200):
    """Optimize using L-BFGS-B with smooth objective."""
    bounds = [(0.0, 1.0)] * (2 * N)
    try:
        res = minimize(
            lambda x: _objective_smooth(x, beta),
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-9}
        )
        return res.x
    except Exception:
        return x0


def _local_optimize_slsqp(x0, maxiter=300):
    """Optimize using SLSQP with hard objective."""
    bounds = [(0.0, 1.0)] * (2 * N)
    try:
        res = minimize(
            _objective_hard,
            x0,
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': maxiter, 'ftol': 1e-10}
        )
        return res.x
    except Exception:
        return x0


def _local_optimize_nm(x0, maxiter=300):
    """Optimize using Nelder-Mead."""
    try:
        res = minimize(
            _objective_hard,
            x0,
            method='Nelder-Mead',
            options={'maxiter': maxiter, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
        )
        return res.x
    except Exception:
        return x0


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    best_x = None
    best_score = -float('inf')

    inits = _generate_diverse_inits()

    def time_left():
        return deadline - time.time()

    # Multi-stage optimization: smooth annealing -> hard polish
    for idx, bary_init in enumerate(inits):
        if time_left() < 20.0:
            break

        x = bary_init.flatten()

        # Stage 1: Smooth optimization with beta annealing
        for beta in [50.0, 100.0, 200.0]:
            if time_left() < 15.0:
                break
            x = _local_optimize_smooth(x, beta=beta, maxiter=100)
            x = _project_to_bary_simplex(x.reshape(N, 2)).flatten()

        # Stage 2: Hard SLSQP polish
        if time_left() < 10.0:
            break
        x = _local_optimize_slsqp(x, maxiter=200)
        x = _project_to_bary_simplex(x.reshape(N, 2)).flatten()

        # Stage 3: Nelder-Mead for final refinement
        if time_left() > 8.0:
            x = _local_optimize_nm(x, maxiter=200)
            x = _project_to_bary_simplex(x.reshape(N, 2)).flatten()

        # Evaluate
        bary = x.reshape(N, 2)
        score = _min_area_ratio_from_bary(bary)
        if score > best_score:
            best_score = score
            best_x = x.copy()

    # If time remains, try more random starts with quick optimization
    if time_left() > 25.0:
        for seed in [500, 600, 700, 800, 900, 1000]:
            if time_left() < 15.0:
                break
            rng = np.random.RandomState(seed)
            bary_rand = np.zeros((N, 2))
            for i in range(N):
                u = rng.random()
                v = rng.random()
                while u + v > 1.0:
                    u = rng.random()
                    v = rng.random()
                bary_rand[i] = [u, v]

            x = bary_rand.flatten()

            # Quick smooth optimization
            for beta in [100.0, 200.0]:
                if time_left() < 10.0:
                    break
                x = _local_optimize_smooth(x, beta=beta, maxiter=50)
                x = _project_to_bary_simplex(x.reshape(N, 2)).flatten()

            # Quick SLSQP
            if time_left() < 5.0:
                break
            x = _local_optimize_slsqp(x, maxiter=100)
            x = _project_to_bary_simplex(x.reshape(N, 2)).flatten()

            bary = x.reshape(N, 2)
            score = _min_area_ratio_from_bary(bary)
            if score > best_score:
                best_score = score
                best_x = x.copy()

    # Fallback
    if best_x is None:
        best_bary = np.array([
            [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
            [1 / 3, 1 / 3], [0.5, 0.0], [0.0, 0.5],
            [0.5, 0.5], [0.25, 0.0], [0.75, 0.0],
            [0.25, 0.25], [0.5, 0.25]
        ])
        best_x = best_bary.flatten()

    # Convert to Cartesian
    bary = _project_to_bary_simplex(best_x.reshape(N, 2))
    points = _bary_to_cart(bary[:, 0], bary[:, 1])

    return points
# EVOLVE-BLOCK-END
