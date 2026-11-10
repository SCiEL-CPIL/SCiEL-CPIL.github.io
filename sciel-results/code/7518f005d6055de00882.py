"""Improved packing of 50 equal spheres in a unit cube using vectorized force-directed relaxation."""
import numpy as np
import time
import math
from scipy.spatial import cKDTree

N = 50
HALF = 0.5
EPS = 1e-12

# EVOLVE-BLOCK-START

def _min_pair_dist(centers):
    """Return minimum pairwise distance using KDTree."""
    n = len(centers)
    if n < 2:
        return float('inf')
    tree = cKDTree(centers)
    dists, idxs = tree.query(centers, k=2)
    return np.min(dists[:, 1])

def _min_wall_dist(centers):
    """Return minimum distance from any center to any wall."""
    coords = np.abs(centers)
    max_coords = np.max(coords, axis=1)
    dists = HALF - max_coords
    return np.min(dists)

def _score(centers):
    """Return the radius: min(min_pair_dist/2, min_wall_dist)."""
    mpd = _min_pair_dist(centers)
    mwd = _min_wall_dist(centers)
    return min(mpd / 2.0, mwd)

def _is_valid(centers):
    if centers.shape != (N, 3):
        return False
    if not np.all(np.isfinite(centers)):
        return False
    if np.any(np.abs(centers) > HALF + 1e-12):
        return False
    return True

# ---------- Initializations ----------

def _init_grid():
    m = int(np.ceil(N ** (1.0 / 3.0)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()

def _init_fcc():
    """FCC lattice initialization."""
    a = 1.0 / (4 ** (1.0/3.0))  # spacing so 4x4x4 fits
    pts = []
    for i in range(4):
        for j in range(4):
            for k in range(4):
                x = (i + 0.5) * a - 0.5
                y = (j + 0.5) * a - 0.5
                z = (k + 0.5) * a - 0.5
                pts.append([x, y, z])
                if len(pts) >= N:
                    break
            if len(pts) >= N:
                break
        if len(pts) >= N:
            break
    arr = np.array(pts[:N])
    return arr

def _init_bcc():
    """BCC lattice initialization."""
    a = 1.0 / (3.5 ** (1.0/3.0))
    pts = []
    for i in range(4):
        for j in range(4):
            for k in range(4):
                x = (i + 0.5) * a - 0.5
                y = (j + 0.5) * a - 0.5
                z = (k + 0.5) * a - 0.5
                pts.append([x, y, z])
    # Add BCC center points
    for i in range(3):
        for j in range(3):
            for k in range(3):
                x = (i + 1.0) * a - 0.5
                y = (j + 1.0) * a - 0.5
                z = (k + 1.0) * a - 0.5
                pts.append([x, y, z])
    rng = np.random.RandomState(0)
    rng.shuffle(pts)
    return np.array(pts[:N])

def _init_random(seed=42):
    rng = np.random.RandomState(seed)
    return rng.uniform(-0.45, 0.45, size=(N, 3))

def _init_jittered_grid(seed=42):
    base = _init_grid()
    rng = np.random.RandomState(seed)
    jitter = rng.uniform(-0.03, 0.03, size=(N, 3))
    return np.clip(base + jitter, -0.49, 0.49)

# ---------- Force-directed relaxation (vectorized) ----------

def _force_relax(centers, time_limit, seed=42, max_iter=50000):
    """
    Vectorized force-directed relaxation to maximize minimum distance.
    Each point is repelled by all others (inverse-square law) and attracted
    toward the cube center when near walls. Uses adaptive step size with
    simulated-annealing-style random kicks to escape local minima.
    """
    rng = np.random.RandomState(seed)
    current = centers.copy()
    best = current.copy()
    best_score = _score(best)
    current_score = best_score

    start_time = time.time()
    step_size = 0.005
    no_improve = 0
    max_no_improve = 200
    kick_count = 0

    for it in range(max_iter):
        if time.time() - start_time > time_limit:
            break

        # Compute all pairwise distances (N x N)
        diff = current[:, np.newaxis, :] - current[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, 1.0)  # avoid division by zero

        # Repulsion force: each point is pushed away from others
        # F_i = sum_j (diff_ij / dist_ij^3)  (inverse-square repulsion)
        inv_dist3 = np.zeros_like(dists)
        mask = dists > 1e-10
        inv_dist3[mask] = 1.0 / (dists[mask] ** 3)
        np.fill_diagonal(inv_dist3, 0.0)

        forces = np.sum(diff * inv_dist3[:, :, np.newaxis], axis=1)

        # Wall forces: push points away from walls
        wall_forces = np.zeros_like(current)
        for d in range(3):
            # Push away from +wall
            near_plus = current[:, d] > 0
            wall_forces[near_plus, d] -= (current[near_plus, d] - 0.0) * 0.5
            # Push away from -wall
            near_minus = current[:, d] < 0
            wall_forces[near_minus, d] -= (current[near_minus, d] + 0.0) * 0.5
            # Stronger push near walls
            close_plus = current[:, d] > 0.3
            wall_forces[close_plus, d] -= (current[close_plus, d] - 0.3) * 2.0
            close_minus = current[:, d] < -0.3
            wall_forces[close_minus, d] -= (current[close_minus, d] + 0.3) * 2.0

        total_forces = forces + wall_forces

        # Normalize forces to prevent explosion
        force_norms = np.linalg.norm(total_forces, axis=1, keepdims=True)
        force_norms = np.maximum(force_norms, 1e-10)
        normalized = total_forces / force_norms

        # Adaptive step: scale by current score deficit
        target_r = 0.136  # target radius
        deficit = max(0.0, target_r - current_score)
        adapt = 1.0 + deficit * 10.0
        effective_step = step_size * adapt

        # Update positions
        new_positions = current + normalized * effective_step
        new_positions = np.clip(new_positions, -HALF + 1e-6, HALF - 1e-6)

        # Check if this is an improvement
        new_score = _score(new_positions)

        if new_score > current_score:
            current = new_positions
            current_score = new_score
            no_improve = 0
            if new_score > best_score:
                best = current.copy()
                best_score = new_score
        else:
            no_improve += 1
            if no_improve > max_no_improve:
                # Random kick to escape local minimum
                kick_count += 1
                kick_strength = 0.01 * (1.0 + 0.1 * kick_count)
                idxs = rng.choice(N, size=max(1, N // 10), replace=False)
                current[idxs] += rng.normal(0, kick_strength, size=(len(idxs), 3))
                current = np.clip(current, -HALF + 1e-6, HALF - 1e-6)
                no_improve = 0
                step_size = 0.008
                continue

        # Decay step size
        step_size *= 0.9995
        if step_size < 1e-5:
            step_size = 0.002

    return best, best_score

# ---------- Simulated annealing (vectorized scoring) ----------

def _optimize_with_sa(centers, time_limit, seed=42):
    """Simulated annealing with fast KDTree scoring."""
    rng = np.random.RandomState(seed)
    current = centers.copy()
    best = current.copy()
    best_score = _score(current)
    current_score = best_score

    start_time = time.time()
    start_temp = 0.03
    end_temp = 0.0005

    while time.time() - start_time < time_limit:
        progress = min(1.0, (time.time() - start_time) / time_limit)
        temp = start_temp * (end_temp / start_temp) ** progress

        idx = rng.randint(N)
        delta = rng.uniform(-0.015, 0.015, size=3)
        candidate = current.copy()
        candidate[idx] += delta

        if np.any(np.abs(candidate[idx]) > HALF - 1e-6):
            continue

        new_score = _score(candidate)
        delta_score = new_score - current_score

        if delta_score > 0 or rng.random() < math.exp(delta_score / temp):
            current = candidate
            current_score = new_score
            if new_score > best_score:
                best = candidate.copy()
                best_score = new_score

    return best, best_score

# ---------- Polish pass ----------

def _polish(centers, time_limit, seed=42):
    """Fine-tuning pass with very small steps."""
    rng = np.random.RandomState(seed)
    current = centers.copy()
    best = current.copy()
    best_score = _score(current)

    start_time = time.time()
    step_size = 0.001
    no_improve = 0
    max_no_improve = 300

    while time.time() - start_time < time_limit:
        mpd = _min_pair_dist(current)
        mwd = _min_wall_dist(current)
        current_score = min(mpd / 2.0, mwd)

        if current_score > best_score:
            best_score = current_score
            best = current.copy()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve > max_no_improve:
                idx = rng.randint(N)
                current[idx] += rng.normal(0, 0.001, size=3)
                current = np.clip(current, -HALF + 1e-6, HALF - 1e-6)
                no_improve = 0
                continue

        # Move the most constrained point
        if mpd / 2.0 < mwd:
            tree = cKDTree(current)
            dists, idxs = tree.query(current, k=2)
            nn_dists = dists[:, 1]
            min_nn = np.min(nn_dists)
            threshold = min_nn + 1e-4
            candidates = np.where(nn_dists < threshold)[0]
            if len(candidates) > 0:
                idx = candidates[rng.randint(len(candidates))]
                nn_idx = idxs[idx, 1]
                dist = nn_dists[idx]
                if dist > 1e-9:
                    direction = (current[idx] - current[nn_idx]) / dist
                    current[idx] += direction * step_size
                else:
                    current[idx] += rng.normal(0, step_size, 3)
            else:
                idx = rng.randint(N)
                nn_idx = idxs[idx, 1]
                dist = nn_dists[idx]
                if dist > 1e-9:
                    direction = (current[idx] - current[nn_idx]) / dist
                    current[idx] += direction * step_size
                else:
                    current[idx] += rng.normal(0, step_size, 3)
        else:
            wall_dists = HALF - np.max(np.abs(current), axis=1)
            min_wall_val = np.min(wall_dists)
            threshold = min_wall_val + 1e-4
            candidates = np.where(wall_dists < threshold)[0]
            if len(candidates) > 0:
                idx = candidates[rng.randint(len(candidates))]
            else:
                idx = rng.randint(N)
            norm = np.linalg.norm(current[idx])
            if norm > 1e-9:
                direction = -current[idx] / norm
            else:
                direction = rng.normal(0, 1, 3)
                direction /= np.linalg.norm(direction)
            current[idx] += direction * step_size

        current = np.clip(current, -HALF + 1e-6, HALF - 1e-6)
        step_size *= 0.9998
        if step_size < 1e-6:
            step_size = 0.0005

    return best, best_score

# ---------- Main construct ----------

def construct():
    start_time = time.time()
    deadline = 140.0  # 10s safety margin before 150s limit

    # Generate diverse initializations
    initializations = [
        _init_grid(),
        _init_fcc(),
        _init_bcc(),
        _init_random(seed=42),
        _init_random(seed=123),
        _init_jittered_grid(seed=7),
        _init_jittered_grid(seed=99),
    ]

    best_centers = None
    best_score = -1.0

    # Phase 1: Force-directed relaxation from each initialization
    for i, init in enumerate(initializations):
        if time.time() > deadline - 20.0:
            break
        remaining = deadline - 20.0 - (time.time() - start_time)
        if remaining < 2.0:
            break
        relax_time = min(12.0, remaining / max(1, len(initializations) - i))
        fc, fs = _force_relax(init, time_limit=relax_time, seed=i * 17 + 3)
        if fs > best_score:
            best_score = fs
            best_centers = fc

    # Phase 2: Simulated annealing refinement
    if time.time() < deadline - 10.0 and best_centers is not None:
        sa_time = min(15.0, deadline - 10.0 - (time.time() - start_time))
        if sa_time > 1.0:
            sa_c, sa_s = _optimize_with_sa(best_centers, time_limit=sa_time, seed=200)
            if sa_s > best_score:
                best_score = sa_s
                best_centers = sa_c

    # Phase 3: Final polish
    if time.time() < deadline - 2.0 and best_centers is not None:
        polish_time = min(10.0, deadline - 2.0 - (time.time() - start_time))
        if polish_time > 0.5:
            pol_c, pol_s = _polish(best_centers, time_limit=polish_time, seed=999)
            if pol_s > best_score:
                best_score = pol_s
                best_centers = pol_c

    # Final validation
    if best_centers is None or not _is_valid(best_centers):
        best_centers = _init_grid()

    return best_centers

# EVOLVE-BLOCK-END
