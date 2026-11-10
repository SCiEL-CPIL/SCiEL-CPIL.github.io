"""Pack 50 equal spheres in a unit cube [-0.5,0.5]^3 to maximize common radius."""
import numpy as np
import time

N = 50
TIME_LIMIT = 110.0  # safety margin before 120s kill


def _score(centers):
    """Compute the maximum common radius for given centers."""
    # Distance between all pairs
    n = len(centers)
    # Use broadcasting for pair distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dist, np.inf)
    min_pair_dist = np.min(dist)

    # Distance to walls
    wall_dists = []
    for k in range(3):
        wall_dists.append(0.5 - np.abs(centers[:, k]))
    wall_dists = np.array(wall_dists)
    min_wall = np.min(wall_dists)

    r = min(min_pair_dist / 2.0, min_wall)
    return r


def _soft_min_penalty(centers, r_target, beta=100.0):
    """
    Smooth surrogate for the constraint: all pairs must be >= 2*r_target apart,
    and all centers must be within [−r_target, 0.5−r_target] on each axis.
    Returns a scalar penalty (0 if all constraints satisfied).
    """
    n = len(centers)
    # Pair distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dist, np.inf)

    # Soft-min of (dist - 2*r_target) over all pairs
    # Use log-sum-exp trick: soft_min(x) = -logsumexp(-x) / beta ... 
    # Actually we want a smooth max of violations: violation_i = max(0, 2*r_target - dist_i)
    # Let's use: penalty = sum of soft violations
    # For each pair, violation = max(0, 2*r_target - dist)
    # Smooth approx: soft_violation = (1/beta) * log(1 + exp(beta*(2*r_target - dist)))
    # But this is expensive. Let's just use the raw violations for gradient.

    # For gradient descent, let's compute the actual constraint violations
    pair_viol = np.maximum(0, 2 * r_target - dist)
    # Wall violations
    wall_viol = np.zeros_like(centers)
    for k in range(3):
        wall_viol[:, k] = np.maximum(0, centers[:, k] + r_target) + np.maximum(0, r_target - centers[:, k])

    total_penalty = np.sum(pair_viol) + np.sum(wall_viol)
    return total_penalty


def _gradient_step(centers, r_target, lr):
    """
    One projected gradient descent step to minimize constraint violations.
    Gradient of pair violation: for pair (i,j), if dist < 2*r_target,
    push i and j apart along their connecting vector.
    Gradient of wall violation: push center inward.
    """
    n = len(centers)
    grad = np.zeros_like(centers)

    # Pair forces
    for i in range(n):
        for j in range(i + 1, n):
            d = centers[i] - centers[j]
            dist = np.linalg.norm(d)
            if dist < 2 * r_target and dist > 1e-12:
                # Force to push apart
                f = (2 * r_target - dist) / dist
                grad[i] += f * d
                grad[j] -= f * d
            elif dist <= 1e-12:
                # Coincident: random tiny push
                grad[i] += 1e-6 * np.random.randn(3)
                grad[j] -= 1e-6 * np.random.randn(3)

    # Wall forces
    for k in range(3):
        # If center[k] < -0.5 + r_target, push right
        mask_left = centers[:, k] < -0.5 + r_target
        grad[mask_left, k] += (r_target - 0.5 - centers[mask_left, k])
        # If center[k] > 0.5 - r_target, push left
        mask_right = centers[:, k] > 0.5 - r_target
        grad[mask_right, k] -= (centers[mask_right, k] - 0.5 + r_target)

    centers_new = centers + lr * grad

    # Project back into the feasible region for radius r_target
    for k in range(3):
        centers_new[:, k] = np.clip(centers_new[:, k], -0.5 + r_target, 0.5 - r_target)

    return centers_new


def _bisection_search(initial_centers, r_lo, r_hi, time_budget):
    """
    Bisection search on r. For each candidate r, run gradient descent
    to try to find a feasible configuration. Return best r found.
    """
    start_time = time.time()
    best_r = r_lo
    best_centers = initial_centers.copy()

    # First, try to improve from initial
    centers = initial_centers.copy()

    while time.time() - start_time < time_budget:
        r_mid = (r_lo + r_hi) / 2.0

        # Try to find feasible config at r_mid
        trial = best_centers.copy()
        success = False

        # Multiple restarts with gradient descent
        for restart in range(20):
            if time.time() - start_time > time_budget:
                break

            if restart == 0:
                cur = best_centers.copy()
            else:
                cur = best_centers + 0.01 * np.random.randn(N, 3)
                for k in range(3):
                    cur[:, k] = np.clip(cur[:, k], -0.5 + r_mid, 0.5 - r_mid)

            # Gradient descent
            lr = 0.001
            for step in range(2000):
                if time.time() - start_time > time_budget:
                    break
                cur = _gradient_step(cur, r_mid, lr)
                # Decay learning rate
                if step % 200 == 199:
                    lr *= 0.9

            # Check feasibility
            r_achieved = _score(cur)
            if r_achieved >= r_mid - 1e-10:
                success = True
                best_r = max(best_r, r_achieved)
                best_centers = cur.copy()
                break
            else:
                # Keep the best trial
                if r_achieved > _score(best_centers):
                    best_centers = cur.copy()

        if success:
            r_lo = r_mid
        else:
            r_hi = r_mid

        # If interval is small enough, stop
        if r_hi - r_lo < 1e-8:
            break

    return best_r, best_centers


def _local_refinement(centers, time_budget):
    """
    Given a good initial config, do a local search to maximize the radius.
    Uses a combination of gradient ascent on r and local moves.
    """
    start_time = time.time()
    best_r = _score(centers)
    best_centers = centers.copy()

    # Iterative local improvement
    for iteration in range(1000):
        if time.time() - start_time > time_budget:
            break

        cur_r = _score(best_centers)

        # Try to increase r slightly and re-optimize positions
        r_try = cur_r * 1.001 + 1e-6

        # Run a few gradient steps to accommodate larger r
        trial = best_centers.copy()
        for step in range(500):
            if time.time() - start_time > time_budget:
                break
            trial = _gradient_step(trial, r_try, 0.0005)

        new_r = _score(trial)
        if new_r > best_r + 1e-12:
            best_r = new_r
            best_centers = trial.copy()
        else:
            # Try smaller step
            r_try2 = cur_r * 1.0001 + 1e-7
            trial2 = best_centers.copy()
            for step in range(300):
                if time.time() - start_time > time_budget:
                    break
                trial2 = _gradient_step(trial2, r_try2, 0.0003)
            new_r2 = _score(trial2)
            if new_r2 > best_r + 1e-12:
                best_r = new_r2
                best_centers = trial2.copy()
            else:
                # Try random perturbations
                improved = False
                for _ in range(5):
                    if time.time() - start_time > time_budget:
                        break
                    perturbed = best_centers + 0.005 * np.random.randn(N, 3)
                    for k in range(3):
                        perturbed[:, k] = np.clip(perturbed[:, k], -0.5, 0.5)
                    # Gradient descent to fix
                    for step in range(200):
                        if time.time() - start_time > time_budget:
                            break
                        perturbed = _gradient_step(perturbed, _score(perturbed), 0.001)
                    r_pert = _score(perturbed)
                    if r_pert > best_r + 1e-12:
                        best_r = r_pert
                        best_centers = perturbed.copy()
                        improved = True
                        break
                if not improved:
                    # No improvement, try even smaller
                    break

    return best_r, best_centers


def construct():
    start_time = time.time()
    np.random.seed(42)

    # Initial placement: body-centered cubic-like or random
    # Start with a grid-like arrangement
    m = int(np.ceil(N ** (1.0 / 3.0)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    initial = grid[:N].copy()

    # Add small random perturbation to break symmetry
    initial += 0.01 * np.random.randn(N, 3)
    for k in range(3):
        initial[:, k] = np.clip(initial[:, k], -0.45, 0.45)

    # Phase 1: Bisection search to find a good radius
    r_lo = 0.1
    r_hi = 0.14
    time_budget_bisect = min(40.0, TIME_LIMIT - (time.time() - start_time) - 20.0)
    best_r, best_centers = _bisection_search(initial, r_lo, r_hi, max(10.0, time_budget_bisect))

    # Phase 2: Local refinement
    time_budget_refine = min(40.0, TIME_LIMIT - (time.time() - start_time) - 5.0)
    if time_budget_refine > 5.0:
        refined_r, refined_centers = _local_refinement(best_centers, time_budget_refine)
        if refined_r > best_r:
            best_r = refined_r
            best_centers = refined_centers

    # Final validity check
    r_final = _score(best_centers)
    if r_final <= 0 or not np.all(np.isfinite(best_centers)):
        # Fallback to grid
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N]

    return best_centers
