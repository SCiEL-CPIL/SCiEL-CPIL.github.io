import numpy as np
import time
import math
from scipy.optimize import minimize
from scipy.spatial import cKDTree

N = 50

# ---------- Scoring helpers ----------

def _exact_score(centers):
    """Exact minimum radius score."""
    if len(centers) < 2:
        return 0.5
    coords = np.abs(centers)
    wall_dists = 0.5 - coords
    min_wall = np.min(wall_dists)

    tree = cKDTree(centers)
    dists, _ = tree.query(centers, k=2)
    min_pair = np.min(dists[:, 1])

    return min(min_pair / 2.0, min_wall)


def _soft_min_score(centers, beta=50.0):
    """Differentiable soft-minimum score for gradient-based optimization.

    Uses log-sum-exp for numerical stability.
    """
    coords = np.abs(centers)
    wall_dists = 0.5 - coords
    min_wall = np.min(wall_dists)

    tree = cKDTree(centers)
    dists, _ = tree.query(centers, k=2)
    min_pair = np.min(dists[:, 1])

    a = min_pair / 2.0
    b = min_wall
    # Soft min: -1/beta * log(exp(-beta*a) + exp(-beta*b))
    m = min(a, b)
    ea = math.exp(-beta * (a - m))
    eb = math.exp(-beta * (b - m))
    return m - math.log(ea + eb) / beta


def _objective_flat(x, beta=50.0):
    """Objective for L-BFGS-B: minimize negative soft-min score."""
    centers = x.reshape(N, 3)
    centers = np.clip(centers, -0.5, 0.5)
    return -_soft_min_score(centers, beta=beta)


# ---------- Initialization ----------

def _initial_grid_444():
    """4x4x4 grid, first 50 points."""
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()


def _jittered_grid_init(seed=0):
    """Jittered 4x4x4 grid for diverse restarts."""
    rng = np.random.RandomState(seed)
    m = 4
    cell_size = 1.0 / m
    grid_pts = []
    for i in range(m):
        for j in range(m):
            for k in range(m):
                x = -0.5 + (i + 0.5) * cell_size
                y = -0.5 + (j + 0.5) * cell_size
                z = -0.5 + (k + 0.5) * cell_size
                grid_pts.append([x, y, z])
    grid_pts = np.array(grid_pts)
    idx = rng.choice(len(grid_pts), N, replace=False)
    init_pts = grid_pts[idx].copy()
    jitter = rng.uniform(-0.08, 0.08, init_pts.shape)
    init_pts += jitter
    return np.clip(init_pts, -0.5 + 1e-6, 0.5 - 1e-6)


# ---------- Repulsive dynamics pre-processing ----------

def _repulsive_dynamics(centers, time_limit=5.0, steps=800):
    """Repulsive dynamics to expand the packing from a grid-like start."""
    current = centers.copy()
    start_time = time.time()

    k_repel = 0.005
    k_wall = 0.005
    dt = 0.008

    for step in range(steps):
        if time.time() - start_time > time_limit:
            break

        forces = np.zeros_like(current)

        # Pairwise repulsion (O(N^2) is fine for N=50)
        for i in range(N):
            for j in range(i + 1, N):
                diff = current[i] - current[j]
                d = np.linalg.norm(diff)
                if d < 1e-6:
                    diff = np.random.uniform(-0.01, 0.01, 3)
                    d = np.linalg.norm(diff)
                f_mag = k_repel / (d * d)
                f_vec = (diff / d) * f_mag
                forces[i] += f_vec
                forces[j] -= f_vec

        # Wall repulsion
        for i in range(N):
            for axis in range(3):
                d_pos = 0.5 - current[i, axis]
                if d_pos < 0.15:
                    f_mag = k_wall / (d_pos * d_pos + 1e-6)
                    forces[i, axis] -= f_mag
                d_neg = 0.5 + current[i, axis]
                if d_neg < 0.15:
                    f_mag = k_wall / (d_neg * d_neg + 1e-6)
                    forces[i, axis] += f_mag

        current += forces * dt
        current = np.clip(current, -0.5, 0.5)

    return current


# ---------- Kick local search ----------

def _kick_local_search(centers, time_limit=5.0, max_iter=8000):
    """Local search that resolves overlaps by kicking spheres apart."""
    current = centers.copy()
    best = current.copy()
    best_score = _exact_score(best)
    start_time = time.time()

    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break

        tree = cKDTree(current)
        dists, idxs = tree.query(current, k=2)
        min_d = np.min(dists[:, 1])
        min_idx_i = np.argmin(dists[:, 1])
        min_idx_j = idxs[min_idx_i, 1]

        wall_dists = 0.5 - np.abs(current)
        min_wall = np.min(wall_dists)
        current_radius = min(min_d / 2.0, min_wall)

        if current_radius > best_score:
            best = current.copy()
            best_score = current_radius

        # Kick overlapping pair apart
        if min_d < 2.0 * best_score * 1.02:
            diff = current[min_idx_i] - current[min_idx_j]
            d = np.linalg.norm(diff)
            if d > 1e-6:
                push = (diff / d) * (2.0 * best_score * 1.02 - min_d) * 0.5
                current[min_idx_i] += push
                current[min_idx_j] -= push

        # Wall collision handling
        for i in range(N):
            for axis in range(3):
                if current[i, axis] > 0.5 - best_score * 1.02:
                    current[i, axis] = 0.5 - best_score * 1.02
                elif current[i, axis] < -0.5 + best_score * 1.02:
                    current[i, axis] = -0.5 + best_score * 1.02

        # Small random perturbation every 200 iters
        if it % 200 == 0:
            idx = np.random.randint(N)
            current[idx] += np.random.uniform(-0.004, 0.004, 3)
            current = np.clip(current, -0.5, 0.5)

    return best, best_score


# ---------- Multi-phase L-BFGS with beta annealing ----------

def _l_bfgs_refine(centers, time_limit=15.0):
    """Refine using L-BFGS-B with beta annealing: coarse → fine → polish."""
    start_time = time.time()
    x0 = centers.flatten()
    bounds = [(-0.5, 0.5)] * (N * 3)

    # Phase 1: coarse, low beta (smooth landscape, escape local minima)
    if time.time() - start_time < time_limit * 0.35:
        res1 = minimize(
            _objective_flat, x0, args=(10.0,),
            method='L-BFGS-B', bounds=bounds,
            options={'maxiter': 150, 'ftol': 1e-10, 'gtol': 1e-7}
        )
        x0 = res1.x

    # Phase 2: medium beta
    if time.time() - start_time < time_limit * 0.65:
        res2 = minimize(
            _objective_flat, x0, args=(40.0,),
            method='L-BFGS-B', bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-9}
        )
        x0 = res2.x

    # Phase 3: high beta polish
    if time.time() - start_time < time_limit * 0.95:
        res3 = minimize(
            _objective_flat, x0, args=(120.0,),
            method='L-BFGS-B', bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-11}
        )
        x0 = res3.x

    return np.clip(x0.reshape(N, 3), -0.5, 0.5)


# ---------- Main construct ----------

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill

    best_centers = _initial_grid_444()
    best_score = _exact_score(best_centers)

    # Phase A: Repulsive dynamics from grid (cheap, ~5s)
    if time.time() < deadline - 60.0:
        dyn_centers = _repulsive_dynamics(best_centers, time_limit=5.0)
        dyn_score = _exact_score(dyn_centers)
        if dyn_score > best_score:
            best_score = dyn_score
            best_centers = dyn_centers

    # Phase B: Multi-restart L-BFGS with beta annealing + kick local search
    num_restarts = 8
    for restart in range(num_restarts):
        if time.time() > deadline - 20.0:
            break

        # Diverse initialization
        if restart == 0:
            init = best_centers.copy()
        else:
            init = _jittered_grid_init(seed=restart * 137 + 42)

        # L-BFGS refinement with annealing
        refined = _l_bfgs_refine(init, time_limit=12.0)
        refined_score = _exact_score(refined)

        if refined_score > best_score:
            best_score = refined_score
            best_centers = refined

        # Kick local search polish (cheap, ~4s)
        if time.time() < deadline - 15.0:
            ls_centers, ls_score = _kick_local_search(refined, time_limit=4.0)
            if ls_score > best_score:
                best_score = ls_score
                best_centers = ls_centers

    # Phase C: Final high-precision polish on the best configuration
    if time.time() < deadline - 5.0:
        final = _l_bfgs_refine(best_centers, time_limit=8.0)
        final_score = _exact_score(final)
        if final_score > best_score:
            best_score = final_score
            best_centers = final

    # Safety: ensure validity
    best_centers = np.clip(best_centers, -0.5 + 1e-9, 0.5 - 1e-9)
    if not np.all(np.isfinite(best_centers)):
        best_centers = _initial_grid_444()

    return best_centers
