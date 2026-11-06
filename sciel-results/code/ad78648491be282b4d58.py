import numpy as np
import time
from scipy.spatial.distance import cdist

N = 100

def _calc_score(pts):
    """Calculate the actual score (radius) for a set of points."""
    if pts.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    if np.any(pts < -0.5) or np.any(pts > 0.5):
        return 0.0
    dists = cdist(pts, pts)
    np.fill_diagonal(dists, 1e10)
    min_pair_dist = np.min(dists)
    min_wall_dist = np.min(0.5 - np.abs(pts))
    return min(min_pair_dist / 2.0, min_wall_dist)

def _soft_min_ascent_step(pts, beta, step_size, momentum):
    """
    One gradient ascent step on the soft-min objective using log-sum-exp.
    Returns updated points, velocity (momentum buffer), and the soft-min value.
    """
    # Pairwise distances / 2
    dists = cdist(pts, pts) / 2.0
    # Mask diagonal
    mask = ~np.eye(N, dtype=bool)
    dists_masked = np.where(mask, dists, 1.0)

    # Wall distances
    wall_dists = np.concatenate([
        (pts + 0.5).flatten(),
        (0.5 - pts).flatten()
    ])

    # Combine all "bottleneck" distances
    all_dists = np.concatenate([dists_masked.flatten(), wall_dists])

    # Soft min: (1/beta) * log(sum(exp(-beta * d_i)))
    d_min = np.min(all_dists)
    exp_terms = np.exp(-beta * (all_dists - d_min))
    log_sum = np.log(np.sum(exp_terms))
    soft_min_val = d_min + log_sum / beta

    # Weights (softmax over negative distances)
    weights = exp_terms / np.sum(exp_terms)

    # Build gradient w.r.t. pts
    grad = np.zeros_like(pts)

    # Pair contributions
    pair_weights = weights[:N * N].reshape(N, N)
    # dists_masked[i,j] = |pts[i]-pts[j]|/2
    # d(dists_masked[i,j])/d(pts[i]) = (pts[i]-pts[j]) / (2 * |pts[i]-pts[j]|) = (pts[i]-pts[j]) / (4 * dists_masked[i,j])
    # But we want to MAXIMIZE soft_min, so gradient ascent: grad += weight * d(soft_min)/d(pts)
    # d(soft_min)/d(pts[i]) = sum_j weight_ij * d(dists_masked_ij)/d(pts[i])
    # For i != j: d(|pi-pj|/2)/dpi = (pi-pj)/(|pi-pj|) = (pi-pj)/(2*dists_masked_ij)
    # So: contribution to grad[i] from pair (i,j) = weight_ij * (pi-pj)/(2*dists_masked_ij)

    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (N,N,3)
    # factor[i,j] = pair_weights[i,j] / (2 * dists_masked[i,j]) for i != j
    with np.errstate(divide='ignore', invalid='ignore'):
        factor = np.where(mask, pair_weights / (2.0 * dists_masked), 0.0)
    # grad[i] += sum_j factor[i,j] * diff[i,j]
    weighted_diff = diff * factor[:, :, np.newaxis]
    grad += np.sum(weighted_diff, axis=1)

    # Wall contributions
    wall_weights = weights[N * N:]
    # First half: d = x + 0.5, d(soft_min)/dx = wall_weight * 1
    grad += wall_weights[:N * 3].reshape(N, 3)
    # Second half: d = 0.5 - x, d(soft_min)/dx = wall_weight * (-1)
    grad -= wall_weights[N * 3:].reshape(N, 3)

    return grad, soft_min_val

def _momentum_ascent(pts0, beta_schedule, step_schedule, momentum_schedule, max_time):
    """
    Run momentum-based gradient ascent on soft-min objective.
    """
    start_time = time.time()
    pts = pts0.copy()
    velocity = np.zeros_like(pts)
    best_pts = pts.copy()
    best_score = _calc_score(pts)
    best_soft = -np.inf

    for step in range(len(beta_schedule)):
        if time.time() > start_time + max_time:
            break

        beta = beta_schedule[step]
        step_size = step_schedule[step]
        mu = momentum_schedule[step]

        # Batch of sub-steps for this phase
        n_substeps = max(1, int((max_time - (time.time() - start_time)) * 50 / len(beta_schedule)))
        n_substeps = min(n_substeps, 200)

        for _ in range(n_substeps):
            if time.time() > start_time + max_time:
                break

            grad, soft_val = _soft_min_ascent_step(pts, beta, step_size, mu)

            # Momentum update
            velocity = mu * velocity + (1 - mu) * grad
            pts = pts + step_size * velocity

            # Clip to cube
            pts = np.clip(pts, -0.4999, 0.4999)

            # Track best
            if soft_val > best_soft:
                best_soft = soft_val
                # Only compute expensive score periodically
                if _calc_score(pts) > best_score:
                    best_score = _calc_score(pts)
                    best_pts = pts.copy()

    return best_pts, best_score

def _initial_grid():
    """Generate a reasonable initial grid: 5x4x5 simple cubic."""
    x_ticks = np.linspace(-0.4, 0.4, 5)
    y_ticks = np.linspace(-0.375, 0.375, 4)
    z_ticks = np.linspace(-0.4, 0.4, 5)
    pts = []
    for x in x_ticks:
        for y in y_ticks:
            for z in z_ticks:
                pts.append([x, y, z])
                if len(pts) >= N:
                    break
            if len(pts) >= N:
                break
        if len(pts) >= N:
            break
    return np.array(pts[:N])

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    np.random.seed(42)

    # Initial configuration
    pts = _initial_grid()
    best_pts = pts.copy()
    best_score = _calc_score(pts)

    # Phase 1: Coarse soft-min ascent (low beta, large steps)
    if time.time() < deadline - 60:
        beta1 = np.full(30, 20.0)
        step1 = np.full(30, 0.005)
        mom1 = np.full(30, 0.9)
        t1 = min(30.0, deadline - time.time() - 60)
        if t1 > 5:
            p, s = _momentum_ascent(pts, beta1, step1, mom1, t1)
            if s > best_score:
                best_score = s
                best_pts = p
            pts = p

    # Phase 2: Medium soft-min ascent (medium beta, medium steps)
    if time.time() < deadline - 30:
        beta2 = np.full(30, 50.0)
        step2 = np.linspace(0.003, 0.001, 30)
        mom2 = np.full(30, 0.85)
        t2 = min(25.0, deadline - time.time() - 30)
        if t2 > 5:
            p, s = _momentum_ascent(pts, beta2, step2, mom2, t2)
            if s > best_score:
                best_score = s
                best_pts = p
            pts = p

    # Phase 3: Fine soft-min ascent (high beta, small steps)
    if time.time() < deadline - 15:
        beta3 = np.linspace(100.0, 300.0, 30)
        step3 = np.linspace(0.001, 0.0003, 30)
        mom3 = np.full(30, 0.8)
        t3 = min(15.0, deadline - time.time() - 15)
        if t3 > 3:
            p, s = _momentum_ascent(pts, beta3, step3, mom3, t3)
            if s > best_score:
                best_score = s
                best_pts = p
            pts = p

    # Phase 4: High-precision polish (very high beta, tiny steps)
    if time.time() < deadline - 5:
        beta4 = np.full(20, 1000.0)
        step4 = np.full(20, 0.0001)
        mom4 = np.full(20, 0.7)
        t4 = min(8.0, deadline - time.time() - 5)
        if t4 > 2:
            p, s = _momentum_ascent(pts, beta4, step4, mom4, t4)
            if s > best_score:
                best_score = s
                best_pts = p
            pts = p

    # Final validation and return
    final_pts = np.clip(best_pts, -0.5, 0.5)
    final_score = _calc_score(final_pts)
    if final_score <= 0:
        # Fallback: return the initial grid
        return np.clip(pts, -0.5, 0.5)

    return final_pts
