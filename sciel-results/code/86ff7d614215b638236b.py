"""Circle packing in unit square - 26 circles maximizing sum of radii.
Combines compact structural parameterization with Differential Evolution
and multi-phase local optimization (L-BFGS-B + SLSQP polish).
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

def _is_valid(pos, radii, tol=1e-9):
    """Check if circles are valid (inside square, disjoint interiors)."""
    n = len(radii)
    for i in range(n):
        if pos[i, 0] - radii[i] < -tol or pos[i, 0] + radii[i] > 1.0 + tol:
            return False
        if pos[i, 1] - radii[i] < -tol or pos[i, 1] + radii[i] > 1.0 + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < radii[i] + radii[j] - tol:
                return False
    return True


def _objective(x, deadline=None):
    """Objective: minimize negative sum of radii + smooth penalty."""
    if deadline is not None and time.time() > deadline:
        return 1e10
    n = N
    pos = x[:2 * n].reshape(n, 2)
    radii = x[2 * n:]

    # Boundary violations (vectorized)
    left = radii - pos[:, 0]
    right = pos[:, 0] - (1.0 - radii)
    bot = radii - pos[:, 1]
    top = pos[:, 1] - (1.0 - radii)

    # Overlap violations (vectorized via distance matrix)
    d = pos[:, 0][:, None] - pos[:, 0][None, :]
    e = pos[:, 1][:, None] - pos[:, 1][None, :]
    dist = np.sqrt(d * d + e * e)
    min_dist = radii[:, None] + radii[None, :]
    viol = min_dist - dist
    np.fill_diagonal(viol, 0.0)

    # Stack all violation magnitudes
    all_v = np.concatenate([left, right, bot, top, viol.ravel()])

    # Quadratic penalty on positive violations
    pos_viol = np.maximum(all_v, 0.0)
    quad_penalty = np.sum(pos_viol ** 2)

    # Log-sum-exp soft-max on the worst violation
    beta = 60.0
    m = np.max(all_v)
    if m > 0:
        shifted = beta * (all_v - m)
        sm = (np.log(np.sum(np.exp(shifted))) + beta * m) / beta
    else:
        sm = 0.0

    # Combined penalty
    penalty = 50.0 * quad_penalty + 20.0 * sm

    return -np.sum(radii) + penalty


def _post_process(pos, radii, deadline=None):
    """Iteratively shrink circles to ensure validity."""
    pos = pos.copy()
    radii = radii.copy()

    for _ in range(500):
        if deadline is not None and time.time() > deadline:
            break
        valid = True
        for i in range(N):
            max_r = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
            if radii[i] > max_r:
                radii[i] = max_r * 0.999
                valid = False
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    if radii[i] <= radii[j]:
                        needed = (min_dist - dist) / (2.0 * radii[i] + 1e-12)
                        radii[i] *= max(0.5, 1.0 - needed * 0.999)
                    else:
                        needed = (min_dist - dist) / (2.0 * radii[j] + 1e-12)
                        radii[j] *= max(0.5, 1.0 - needed * 0.999)
                    valid = False
        if valid:
            break

    for i in range(N):
        max_r = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
        radii[i] = max(1e-6, min(radii[i], max_r * 0.999))

    return pos, radii


def _lbfgs_polish(pos, radii, time_budget=15.0):
    """L-BFGS-B polishing."""
    pos = pos.copy()
    radii = radii.copy()
    pos, radii = _post_process(pos, radii)

    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = float(np.sum(radii))

    def objective(x):
        return _objective(x)

    x0 = np.concatenate([pos.flatten(), radii])

    bounds = []
    for i in range(N):
        bounds.append((0.001, 0.999))
        bounds.append((0.001, 0.999))
    for i in range(N):
        bounds.append((0.001, 0.35))

    start = time.time()
    try:
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
        if time.time() - start < time_budget:
            x_opt = result.x
            p_opt = x_opt[:2*N].reshape(N, 2)
            r_opt = x_opt[2*N:]
            p_opt, r_opt = _post_process(p_opt, r_opt)
            if _is_valid(p_opt, r_opt):
                s = float(np.sum(r_opt))
                if s > best_score:
                    best_score = s
                    best_pos = p_opt.copy()
                    best_radii = r_opt.copy()
    except Exception:
        pass

    return best_pos, best_radii, best_score


def _slsqp_polish(pos, radii, time_budget=10.0):
    """SLSQP polishing."""
    pos = pos.copy()
    radii = radii.copy()
    pos, radii = _post_process(pos, radii)

    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = float(np.sum(radii))

    def objective(x):
        return _objective(x)

    x0 = np.concatenate([pos.flatten(), radii])

    bounds = []
    for i in range(N):
        bounds.append((0.001, 0.999))
        bounds.append((0.001, 0.999))
    for i in range(N):
        bounds.append((0.001, 0.35))

    start = time.time()
    try:
        result = minimize(objective, x0, method='SLSQP', bounds=bounds,
                          options={'maxiter': 300, 'ftol': 1e-12})
        if time.time() - start < time_budget:
            x_opt = result.x
            p_opt = x_opt[:2*N].reshape(N, 2)
            r_opt = x_opt[2*N:]
            p_opt, r_opt = _post_process(p_opt, r_opt)
            if _is_valid(p_opt, r_opt):
                s = float(np.sum(r_opt))
                if s > best_score:
                    best_score = s
                    best_pos = p_opt.copy()
                    best_radii = r_opt.copy()
    except Exception:
        pass

    return best_pos, best_radii, best_score


def _anneal(pos, radii, time_budget=20.0, seed=42):
    """Simulated annealing with force-directed moves."""
    rng = np.random.RandomState(seed)
    pos = pos.copy()
    radii = radii.copy()
    pos, radii = _post_process(pos, radii)

    best_pos = pos.copy()
    best_radii = radii.copy()
    best_score = float(np.sum(radii))

    T = 0.01
    T_min = 1e-7
    cooling = 0.995
    max_iter = 50000

    start = time.time()

    for it in range(max_iter):
        if time.time() - start > time_budget:
            break

        move_type = rng.random()
        if move_type < 0.5:
            i = rng.randint(N)
            dx = rng.normal(0, 0.02)
            dy = rng.normal(0, 0.02)
            pos[i, 0] = np.clip(pos[i, 0] + dx, 0.005, 0.995)
            pos[i, 1] = np.clip(pos[i, 1] + dy, 0.005, 0.995)
        elif move_type < 0.8:
            i = rng.randint(N)
            dr = rng.normal(0, 0.005)
            radii[i] = np.clip(radii[i] + dr, 0.005, 0.35)
        else:
            i, j = rng.choice(N, size=2, replace=False)
            pos[i, 0] = np.clip(pos[i, 0] + rng.normal(0, 0.02), 0.005, 0.995)
            pos[i, 1] = np.clip(pos[i, 1] + rng.normal(0, 0.02), 0.005, 0.995)
            pos[j, 0] = np.clip(pos[j, 0] + rng.normal(0, 0.02), 0.005, 0.995)
            pos[j, 1] = np.clip(pos[j, 1] + rng.normal(0, 0.02), 0.005, 0.995)

        pos, radii = _post_process(pos, radii)

        x = np.concatenate([pos.flatten(), radii])
        new_obj = _objective(x)
        new_score = -new_obj  # This is sum(radii) - penalty

        x_best = np.concatenate([best_pos.flatten(), best_radii])
        old_obj = _objective(x_best)
        old_score = -old_obj

        delta = new_score - old_score
        if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-10)):
            if _is_valid(pos, radii):
                s = float(np.sum(radii))
                if s > best_score:
                    best_score = s
                    best_pos = pos.copy()
                    best_radii = radii.copy()

        T *= cooling
        if T < T_min:
            T = T_min

    return best_pos, best_radii, best_score


def _make_grid(seed=42):
    """5x5 grid with one extra circle."""
    rng = np.random.RandomState(seed)
    side = 5
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    for i in range(5):
        for j in range(5):
            idx = i * 5 + j
            if idx < N:
                pos[idx] = [(i + 0.5) / side, (j + 0.5) / side]
                radii[idx] = 0.5 / side - 0.003
    pos[25] = [0.5, 0.02]
    radii[25] = 0.015
    return pos, radii


def _make_hexagonal(seed=42):
    """Hexagonal packing pattern."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    r = 0.095
    idx = 0
    for row in range(6):
        y = 0.1 + row * r * np.sqrt(3)
        if y > 0.9:
            break
        offset = r if row % 2 == 0 else 2 * r
        for col in range(5):
            x = offset + col * 2 * r
            if x > 0.9:
                break
            if idx < N:
                pos[idx] = [x, y]
                radii[idx] = r
                idx += 1
    while idx < N:
        pos[idx] = [0.5, 0.5]
        radii[idx] = 0.05
        idx += 1
    return pos, radii


def _make_random(seed=42):
    """Random initial configuration."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.1, 0.9, size=(N, 2))
    radii = rng.uniform(0.03, 0.08, size=N)
    return pos, radii


def _make_center_ring(seed=42):
    """Center + inner ring + outer ring."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    pos[0] = [0.5, 0.5]
    radii[0] = 0.15
    r_inner = 0.12
    dist_inner = 0.3
    for i in range(8):
        angle = 2 * np.pi * i / 8
        pos[i + 1] = [0.5 + dist_inner * np.cos(angle), 0.5 + dist_inner * np.sin(angle)]
        radii[i + 1] = r_inner
    r_outer = 0.08
    dist_outer = 0.42
    for i in range(17):
        angle = 2 * np.pi * i / 17
        pos[i + 9] = [0.5 + dist_outer * np.cos(angle), 0.5 + dist_outer * np.sin(angle)]
        radii[i + 9] = r_outer
    return pos, radii


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill

    best_pos = None
    best_radii = None
    best_score = -np.inf

    strategies = [
        ('grid', 42), ('grid', 123), ('grid', 7),
        ('hexagonal', 42), ('hexagonal', 99),
        ('random', 42), ('random', 123), ('random', 7), ('random', 55),
        ('center_ring', 42), ('center_ring', 123),
    ]

    for strat_name, seed in strategies:
        if time.time() > deadline - 30:
            break

        if strat_name == 'grid':
            pos, radii = _make_grid(seed)
        elif strat_name == 'hexagonal':
            pos, radii = _make_hexagonal(seed)
        elif strat_name == 'random':
            pos, radii = _make_random(seed)
        elif strat_name == 'center_ring':
            pos, radii = _make_center_ring(seed)
        else:
            continue

        # Annealing
        remaining = deadline - time.time()
        if remaining < 15:
            break
        anneal_budget = min(18.0, remaining * 0.4)
        pos_a, radii_a, score_a = _anneal(pos, radii, time_budget=anneal_budget, seed=seed)

        if score_a > best_score:
            best_score = score_a
            best_pos = pos_a.copy()
            best_radii = radii_a.copy()

        # L-BFGS-B polish
        remaining = deadline - time.time()
        if remaining < 15:
            continue
        polish_budget = min(12.0, remaining * 0.35)
        pos_p, radii_p, score_p = _lbfgs_polish(pos_a, radii_a, time_budget=polish_budget)

        if score_p > best_score:
            best_score = score_p
            best_pos = pos_p.copy()
            best_radii = radii_p.copy()

        # SLSQP polish
        remaining = deadline - time.time()
        if remaining < 12:
            continue
        slsqp_budget = min(8.0, remaining * 0.25)
        pos_s, radii_s, score_s = _slsqp_polish(pos_p, radii_p, time_budget=slsqp_budget)

        if score_s > best_score:
            best_score = score_s
            best_pos = pos_s.copy()
            best_radii = radii_s.copy()

    # Final polish on best result if time allows
    if best_pos is not None:
        remaining = deadline - time.time()
        if remaining > 20:
            pos_f, radii_f, score_f = _lbfgs_polish(best_pos, best_radii, time_budget=min(15.0, remaining - 8))
            if score_f > best_score:
                best_score = score_f
                best_pos = pos_f.copy()
                best_radii = radii_f.copy()

        remaining = deadline - time.time()
        if remaining > 12:
            pos_s2, radii_s2, score_s2 = _slsqp_polish(best_pos, best_radii, time_budget=min(8.0, remaining - 5))
            if score_s2 > best_score:
                best_score = score_s2
                best_pos = pos_s2.copy()
                best_radii = radii_s2.copy()

    # Fallback: simple grid
    if best_pos is None:
        side = 5
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 0.005) for i in range(side) for j in range(side)]
        result = np.array(cells[:N])
        return result

    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
