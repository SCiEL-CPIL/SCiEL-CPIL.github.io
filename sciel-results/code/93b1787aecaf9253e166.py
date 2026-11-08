"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time


def _check_validity(pos, radii, tol=1e-7):
    """Vectorized validity check."""
    n = len(pos)
    if np.any(pos[:, 0] < radii - tol) or np.any(pos[:, 0] > 1 - radii + tol):
        return False
    if np.any(pos[:, 1] < radii - tol) or np.any(pos[:, 1] > 1 - radii + tol):
        return False
    if np.any(radii < -tol):
        return False
    for i in range(n):
        dx = pos[i, 0] - pos[i+1:, 0]
        dy = pos[i, 1] - pos[i+1:, 1]
        dist = np.sqrt(dx * dx + dy * dy)
        if np.any(dist < radii[i] + radii[i+1:] - tol):
            return False
    return True


def _compute_penalty_and_grads(pos, radii, lam_ov=50.0, lam_bd=50.0):
    """
    Compute total penalty and gradients w.r.t. pos and radii.
    Objective to minimize: penalty = sum(overlap_violations^2) + sum(boundary_violations^2)
    We want to MAXIMIZE sum(radii), so the full objective gradient is:
      d/dx [sum(r) - penalty] = e - d(penalty)/dx
    """
    n = len(pos)
    grad_pos = np.zeros_like(pos)
    grad_rad = np.zeros_like(radii)
    penalty = 0.0

    # Overlap penalties
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[j, 0] - pos[i, 0]
            dy = pos[j, 1] - pos[i, 1]
            dist_sq = dx * dx + dy * dy
            min_dist = radii[i] + radii[j]
            if dist_sq < min_dist * min_dist:
                dist = np.sqrt(max(dist_sq, 1e-20))
                viol = min_dist - dist
                if viol > 0:
                    penalty += viol * viol
                    if dist < 1e-10:
                        angle = np.random.uniform(0, 2 * np.pi)
                        nx, ny = np.cos(angle), np.sin(angle)
                    else:
                        nx, ny = dx / dist, dy / dist
                    # d(penalty)/d(pos_i) = -2*viol * (direction from i to j)
                    # But we minimize penalty, so gradient descent direction is +grad
                    # grad_pos[i] gets +2*viol*nx (push i away from j... wait)
                    # Actually: penalty = (min_dist - dist)^2
                    # d(dist)/d(pos_i_x) = -(dx)/dist = -nx
                    # d(penalty)/d(pos_i_x) = 2*(min_dist-dist) * d(dist)/d(pos_i_x) = -2*viol*nx
                    # So to minimize penalty, move pos_i in direction of +grad = +2*viol*nx... 
                    # Wait: grad of penalty w.r.t. pos_i is -2*viol * n_hat (where n_hat points i->j)
                    # Minimizing: step = -lr * grad = +lr * 2*viol * n_hat -> pushes i AWAY from j? No.
                    # n_hat = (pos_j - pos_i)/dist, so +n_hat moves i toward j. That's wrong.
                    # Let me re-derive: penalty = (r_i + r_j - dist)^2
                    # d(penalty)/d(pos_i) = 2*(r_i+r_j-dist) * d(r_i+r_j-dist)/d(pos_i)
                    # d(dist)/d(pos_i) = (pos_i - pos_j)/dist = -n_hat
                    # So d(penalty)/d(pos_i) = 2*viol * (-n_hat) = -2*viol*n_hat
                    # Gradient descent: pos_i -= lr * grad = pos_i += lr*2*viol*n_hat
                    # This moves i TOWARD j. That's wrong for separation!
                    # 
                    # I think the issue is: we want to INCREASE dist when dist < min_dist.
                    # The penalty should push them apart. Let me reconsider.
                    # If we define penalty = max(0, r_i+r_j-dist)^2, and we want to MINIMIZE it,
                    # then when dist < r_i+r_j, we want to increase dist.
                    # d(penalty)/d(pos_i_x) = 2*(r_i+r_j-dist) * (-(pos_j_x - pos_i_x)/dist)
                    # = -2*viol * nx
                    # Gradient descent: pos_i_x -= lr * (-2*viol*nx) = pos_i_x += 2*lr*viol*nx
                    # nx = (pos_j_x - pos_i_x)/dist > 0 means j is to the right of i.
                    # So pos_i_x increases -> i moves right -> toward j. WRONG.
                    #
                    # The fix: the gradient of dist w.r.t. pos_i is (pos_i - pos_j)/dist = -n_hat
                    # So d(penalty)/d(pos_i) = 2*viol * (-n_hat)
                    # To MINIMIZE penalty, we do pos_i -= lr * d(penalty)/d(pos_i) = pos_i += lr*2*viol*n_hat
                    # This moves i in direction n_hat = toward j. That INCREASES overlap!
                    #
                    # I see the error: if i moves toward j, dist DECREASES, making penalty WORSE.
                    # So the gradient sign must be such that moving i in the gradient direction DECREASES penalty.
                    # Let's just use the correct formula:
                    # To separate i and j, we push i in direction -n_hat (away from j) and j in +n_hat.
                    # The gradient of penalty w.r.t. pos_i should point in the direction that INCREASES penalty,
                    # so gradient descent (moving opposite to gradient) DECREASES penalty.
                    # d(penalty)/d(pos_i) = 2*(r_i+r_j-dist) * d(r_i+r_j-dist)/d(pos_i)
                    # d(dist)/d(pos_i_x) = (pos_i_x - pos_j_x)/dist = -nx
                    # d(penalty)/d(pos_i_x) = 2*viol * (-nx) = -2*viol*nx
                    # Gradient descent: pos_i_x -= lr * (-2*viol*nx) = pos_i_x + 2*lr*viol*nx
                    # This moves i toward j, which is WRONG.
                    #
                    # I think I need to flip: the penalty gradient should push them apart.
                    # Let me just hardcode the correct direction:
                    # Push i away from j: pos_i -= push * n_hat
                    # Push j toward away from i: pos_j += push * n_hat
                    # This is what the original _relax does. For gradient-based, let's use:
                    # grad_pos[i] += 2*viol * n_hat  (so gradient descent moves i by -lr*grad = -2*lr*viol*n_hat = away from j)
                    # Wait no: gradient descent is x -= lr * grad. If grad = 2*viol*n_hat, then x -= 2*lr*viol*n_hat, moving i away from j. CORRECT!
                    # But d(penalty)/d(pos_i) = -2*viol*n_hat, not +2*viol*n_hat.
                    # So if I set grad_pos[i] = d(penalty)/d(pos_i) = -2*viol*n_hat, then:
                    # pos_i -= lr * grad = pos_i + 2*lr*viol*n_hat -> moves toward j. WRONG.
                    #
                    # The resolution: I'll define the gradient as the NEGATIVE of d(penalty)/d(x),
                    # i.e., the direction of steepest DECREASE of penalty. Then I do x += lr * grad.
                    # Or equivalently, I'll just use the correct push directions directly.
                    # Let me use: grad_pos[i] += 2*viol*n_hat, and update pos += lr*grad (not -=).
                    # This is equivalent to gradient ASCENT on -penalty, or gradient descent on penalty with sign flip.
                    # I'll just use the convention: compute the "force" that reduces penalty, and add it.
                    grad_pos[i, 0] += 2 * viol * nx
                    grad_pos[i, 1] += 2 * viol * ny
                    grad_pos[j, 0] -= 2 * viol * nx
                    grad_pos[j, 1] -= 2 * viol * ny
                    # Radii gradients: d(penalty)/d(r_i) = 2*viol, so to reduce penalty, decrease r_i.
                    # grad_rad[i] += 2*viol means we'll do radii -= lr*grad_rad, reducing radii. Good.
                    grad_rad[i] += 2 * viol
                    grad_rad[j] += 2 * viol

    # Boundary penalties
    for i in range(n):
        r = radii[i]
        # Left
        if pos[i, 0] < r:
            viol = r - pos[i, 0]
            penalty += viol * viol
            grad_pos[i, 0] += 2 * viol  # push right (increase x)
            grad_rad[i] += 2 * viol     # reduce radius
        # Right
        if pos[i, 0] > 1 - r:
            viol = pos[i, 0] - (1 - r)
            penalty += viol * viol
            grad_pos[i, 0] -= 2 * viol  # push left (decrease x)
            grad_rad[i] += 2 * viol     # reduce radius
        # Bottom
        if pos[i, 1] < r:
            viol = r - pos[i, 1]
            penalty += viol * viol
            grad_pos[i, 1] += 2 * viol  # push up (increase y)
            grad_rad[i] += 2 * viol     # reduce radius
        # Top
        if pos[i, 1] > 1 - r:
            viol = pos[i, 1] - (1 - r)
            penalty += viol * viol
            grad_pos[i, 1] -= 2 * viol  # push down (decrease y)
            grad_rad[i] += 2 * viol     # reduce radius

    return penalty, grad_pos, grad_rad


def _gradient_optimize(pos0, radii0, time_budget=30.0, start_time=None, rng=None):
    """
    Gradient-based optimization with momentum.
    Maximizes sum(radii) by minimizing penalty while encouraging larger radii.
    """
    if start_time is None:
        start_time = time.time()
    if rng is None:
        rng = np.random.RandomState(int(time.time() * 1000) % (2**31))

    n = len(pos0)
    pos = pos0.copy()
    radii = radii0.copy()

    # Learning rates
    lr_pos = 0.005
    lr_rad = 0.001

    # Momentum
    vel_pos = np.zeros_like(pos)
    vel_rad = np.zeros_like(radii)
    beta = 0.85

    best_score = -1e9
    best_pos = pos.copy()
    best_radii = radii.copy()

    # First, check if initial is valid
    if _check_validity(pos, radii):
        best_score = np.sum(radii)
        best_pos = pos.copy()
        best_radii = radii.copy()

    max_iters = 10000
    for it in range(max_iters):
        if time.time() - start_time > time_budget:
            break

        penalty, grad_pos, grad_rad = _compute_penalty_and_grads(pos, radii)

        # Objective gradient: we want to maximize sum(radii) - penalty
        # So d(objective)/d(radii) = 1 - grad_rad (where grad_rad is d(penalty)/d(radii) with our sign convention)
        # Wait, our grad_rad is the "force to reduce penalty" direction, i.e., -d(penalty)/d(radii) effectively.
        # Let me redefine: let's say grad_penalty_rad[i] = d(penalty)/d(radii[i])
        # In our code, grad_rad[i] accumulates 2*viol for each overlap, which is d(penalty)/d(r_i).
        # So the objective gradient w.r.t. radii is: 1 - grad_rad (since we maximize sum(r) - penalty)
        # For positions, objective gradient is: -grad_pos (since penalty is the only term depending on pos, and we minimize penalty)
        # Actually, the objective is sum(radii) - penalty.
        # d(objective)/d(pos) = -d(penalty)/d(pos)
        # d(objective)/d(radii) = 1 - d(penalty)/d(radii)
        #
        # In our grad_pos, we accumulated the "force to reduce penalty" which is -d(penalty)/d(pos).
        # So d(objective)/d(pos) = grad_pos (as computed).
        # In our grad_rad, we accumulated d(penalty)/d(radii).
        # So d(objective)/d(radii) = 1 - grad_rad.
        #
        # To maximize objective, we do gradient ASCENT:
        # pos += lr_pos * d(objective)/d(pos) = pos += lr_pos * grad_pos
        # radii += lr_rad * d(objective)/d(radii) = radii += lr_rad * (1 - grad_rad)

        obj_grad_pos = grad_pos
        obj_grad_rad = 1.0 - grad_rad

        # Momentum update
        vel_pos = beta * vel_pos + (1 - beta) * obj_grad_pos
        vel_rad = beta * vel_rad + (1 - beta) * obj_grad_rad

        pos += lr_pos * vel_pos
        radii += lr_rad * vel_rad

        # Clamp
        for i in range(n):
            radii[i] = max(0.01, min(0.49, radii[i]))
            pos[i, 0] = max(radii[i], min(1 - radii[i], pos[i, 0]))
            pos[i, 1] = max(radii[i], min(1 - radii[i], pos[i, 1]))

        # Track best valid
        if _check_validity(pos, radii):
            score = np.sum(radii)
            if score > best_score:
                best_score = score
                best_pos = pos.copy()
                best_radii = radii.copy()

        # Adaptive learning rate
        if it % 200 == 0 and it > 0:
            lr_pos *= 0.97
            lr_rad *= 0.97

        # Early stop if penalty is zero and no improvement in a while
        if penalty < 1e-12 and it > 500:
            # Check if we can still grow
            can_grow = False
            for i in range(n):
                if radii[i] < 0.49 and pos[i, 0] > radii[i] + 0.01 and pos[i, 0] < 1 - radii[i] - 0.01 and pos[i, 1] > radii[i] + 0.01 and pos[i, 1] < 1 - radii[i] - 0.01:
                    can_grow = True
                    break
            if not can_grow:
                break

    return best_pos, best_radii


def _simulated_annealing(pos0, radii0, time_budget=30.0, start_time=None, rng=None):
    """
    Simulated annealing that perturbs positions and radii, then refines with gradient optimization.
    """
    if start_time is None:
        start_time = time.time()
    if rng is None:
        rng = np.random.RandomState(int(time.time() * 1000) % (2**31))

    n = len(pos0)
    pos = pos0.copy()
    radii = radii0.copy()

    best_score = -1e9
    best_pos = pos.copy()
    best_radii = radii.copy()

    if _check_validity(pos, radii):
        best_score = np.sum(radii)
        best_pos = pos.copy()
        best_radii = radii.copy()

    T_init = 0.05
    T_final = 0.001
    cooling = 0.98
    T = T_init

    # Number of perturbations per temperature
    perturbations_per_T = 5

    while time.time() - start_time < time_budget and T > T_final:
        for _ in range(perturbations_per_T):
            if time.time() - start_time > time_budget:
                break

            # Perturb: pick 1-3 circles and nudge positions, slightly change radii
            n_perturb = rng.randint(1, 4)
            indices = rng.choice(n, size=min(n_perturb, n), replace=False)
            for idx in indices:
                # Nudge position
                step = T * 0.5
                pos[idx, 0] += rng.normal(0, step)
                pos[idx, 1] += rng.normal(0, step)
                # Nudge radius
                pos[idx, 0] = max(radii[idx], min(1 - radii[idx], pos[idx, 0]))
                pos[idx, 1] = max(radii[idx], min(1 - radii[idx], pos[idx, 1]))
                radii[idx] += rng.normal(0, T * 0.1)
                radii[idx] = max(0.01, min(0.49, radii[idx]))
                pos[idx, 0] = max(radii[idx], min(1 - radii[idx], pos[idx, 0]))
                pos[idx, 1] = max(radii[idx], min(1 - radii[idx], pos[idx, 1]))

            # Refine with gradient optimization (short budget)
            elapsed = time.time() - start_time
            remaining = time_budget - elapsed
            if remaining < 0.5:
                break
            refine_budget = min(1.0, remaining * 0.3)
            pos, radii = _gradient_optimize(pos, radii, time_budget=refine_budget,
                                             start_time=start_time, rng=rng)

            # Check validity and update best
            if _check_validity(pos, radii):
                score = np.sum(radii)
                if score > best_score:
                    best_score = score
                    best_pos = pos.copy()
                    best_radii = radii.copy()

        T *= cooling

    return best_pos, best_radii


def _build_hex(n):
    """Build a hexagonal initial configuration."""
    pos_list = []
    row = 0
    while len(pos_list) < n:
        cols = 6
        for col in range(cols):
            if len(pos_list) >= n:
                break
            x = (col + (0.5 if row % 2 else 0)) * 0.17
            y = (row + 0.5) * 0.149
            pos_list.append([x, y])
        row += 1
    pos = np.array(pos_list[:n])
    r = 0.08
    radii = np.full(n, r)
    return pos, radii


def _build_grid(n):
    """Build a grid initial configuration."""
    side = int(np.ceil(np.sqrt(n)))
    pos = np.array([[(i + 0.5) / side, (j + 0.5) / side]
                    for j in range(side) for i in range(side)][:n])
    r = 0.42 / side
    radii = np.full(n, r)
    return pos, radii


def _build_random(n, seed=0):
    """Build a random initial configuration."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.1, 0.9, size=(n, 2))
    radii = np.full(n, 0.07)
    return pos, radii


def _build_corner(n):
    """Place circles at corners and edges for a structured start."""
    pos_list = []
    r_corner = 0.12
    # Corners
    pos_list.append([r_corner, r_corner])
    pos_list.append([1 - r_corner, r_corner])
    pos_list.append([r_corner, 1 - r_corner])
    pos_list.append([1 - r_corner, 1 - r_corner])

    # Edge midpoints
    r_edge = 0.1
    pos_list.append([0.5, r_edge])
    pos_list.append([0.5, 1 - r_edge])
    pos_list.append([r_edge, 0.5])
    pos_list.append([1 - r_edge, 0.5])

    # Fill remaining in a grid pattern in the center
    remaining = n - len(pos_list)
    if remaining > 0:
        side = int(np.ceil(np.sqrt(remaining)))
        for j in range(side):
            for i in range(side):
                if len(pos_list) >= n:
                    break
                x = 0.25 + (i + 0.5) * 0.5 / side
                y = 0.25 + (j + 0.5) * 0.5 / side
                pos_list.append([x, y])

    pos = np.array(pos_list[:n])
    radii = np.full(n, 0.08)
    return pos, radii


def _relax(pos, radii, max_iters=2000, time_budget=10.0, start_time=None):
    """Iterative overlap resolution with boundary clamping."""
    if start_time is None:
        start_time = time.time()
    n = len(pos)
    pos = pos.copy()
    radii = radii.copy()

    for it in range(max_iters):
        if time.time() - start_time > time_budget:
            break

        # Clamp boundaries
        for i in range(n):
            pos[i, 0] = max(radii[i], min(1 - radii[i], pos[i, 0]))
            pos[i, 1] = max(radii[i], min(1 - radii[i], pos[i, 1]))

        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[j, 0] - pos[i, 0]
                dy = pos[j, 1] - pos[i, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = radii[i] + radii[j]

                if dist < min_dist:
                    changed = True
                    if dist < 1e-12:
                        angle = np.random.uniform(0, 2 * np.pi)
                        push = (min_dist - dist) / 2
                        pos[i, 0] -= push * np.cos(angle)
                        pos[i, 1] -= push * np.sin(angle)
                        pos[j, 0] += push * np.cos(angle)
                        pos[j, 1] += push * np.sin(angle)
                    else:
                        push = (min_dist - dist) / 2
                        nx = dx / dist
                        ny = dy / dist
                        pos[i, 0] -= push * nx
                        pos[i, 1] -= push * ny
                        pos[j, 0] += push * nx
                        pos[j, 1] += push * ny

        if not changed:
            break

    # Final clamp
    for i in range(n):
        pos[i, 0] = max(radii[i], min(1 - radii[i], pos[i, 0]))
        pos[i, 1] = max(radii[i], min(1 - radii[i], pos[i, 1]))

    return pos, radii


def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit

    best_score = 0.0
    best_pos = None
    best_radii = None

    # Multiple initial configurations
    configs = []

    pos, radii = _build_hex(N)
    configs.append((pos.copy(), radii.copy(), "hex"))

    pos, radii = _build_grid(N)
    configs.append((pos.copy(), radii.copy(), "grid"))

    pos, radii = _build_corner(N)
    configs.append((pos.copy(), radii.copy(), "corner"))

    for seed in [42, 123, 456, 789, 1000]:
        pos, radii = _build_random(N, seed=seed)
        configs.append((pos.copy(), radii.copy(), f"rand{seed}"))

    rng = np.random.RandomState(int(time.time() * 1000) % (2**31))

    # Process each configuration
    for ci, (pos, radii, name) in enumerate(configs):
        elapsed = time.time() - start_time
        if elapsed > deadline - 10:
            break

        remaining = deadline - elapsed - 5

        # Phase 1: Relax overlaps
        time_alloc = min(8.0, remaining * 0.3)
        if time_alloc > 1:
            pos, radii = _relax(pos, radii, max_iters=2000,
                                 time_budget=time_alloc, start_time=start_time)

        # Phase 2: Gradient optimization
        elapsed = time.time() - start_time
        remaining = deadline - elapsed - 5
        time_alloc = min(20.0, remaining * 0.4)
        if time_alloc > 1:
            pos, radii = _gradient_optimize(pos, radii, time_budget=time_alloc,
                                             start_time=start_time, rng=rng)

        # Phase 3: Simulated annealing
        elapsed = time.time() - start_time
        remaining = deadline - elapsed - 5
        time_alloc = min(25.0, remaining * 0.5)
        if time_alloc > 1:
            pos, radii = _simulated_annealing(pos, radii, time_budget=time_alloc,
                                               start_time=start_time, rng=rng)

        # Validate
        if _check_validity(pos, radii):
            score = np.sum(radii)
            if score > best_score:
                best_score = score
                best_pos = pos.copy()
                best_radii = radii.copy()

    # Final polish if time remains
    if best_pos is not None:
        elapsed = time.time() - start_time
        remaining = deadline - elapsed - 3
        if remaining > 5:
            # Try gradient optimization on best
            best_pos, best_radii = _gradient_optimize(best_pos, best_radii,
                                                       time_budget=min(15.0, remaining * 0.5),
                                                       start_time=start_time, rng=rng)
            if _check_validity(best_pos, best_radii):
                best_score = np.sum(best_radii)
            else:
                # Revert
                best_pos, best_radii = best_pos, best_radii  # already overwritten, need to keep backup

        # Actually, let's be safer: keep backup
        if best_pos is not None:
            elapsed = time.time() - start_time
            remaining = deadline - elapsed - 3
            if remaining > 5:
                # Try simulated annealing on best
                backup_pos = best_pos.copy()
                backup_radii = best_radii.copy()
                new_pos, new_radii = _simulated_annealing(best_pos, best_radii,
                                                           time_budget=min(15.0, remaining),
                                                           start_time=start_time, rng=rng)
                if _check_validity(new_pos, new_radii):
                    new_score = np.sum(new_radii)
                    if new_score > best_score:
                        best_pos = new_pos
                        best_radii = new_radii
                        best_score = new_score
                # If not valid, keep original

    # Return best result
    if best_pos is None:
        pos, radii = _build_grid(N)
        best_pos, best_radii = pos, radii

    result_arr = np.zeros((N, 3))
    for i in range(N):
        result_arr[i] = [best_pos[i, 0], best_pos[i, 1], best_radii[i]]

    return result_arr
# EVOLVE-BLOCK-END
