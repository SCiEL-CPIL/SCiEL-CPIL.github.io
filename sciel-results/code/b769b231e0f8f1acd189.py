"""Circle packing in unit square - 26 circles maximizing sum of radii.
Uses a contact-graph SLSQP approach: build a contact graph from the current
configuration, then solve a constrained SLSQP problem using only the active
constraints (contacts + boundaries). This is much faster than using all
pairwise constraints, allowing more restarts and a better final polish.

Key improvements over the previous design:
  * Contact-graph SLSQP: only enforce constraints for circles that are
    (or nearly) touching, reducing the constraint count from ~325 to ~50-80
  * More aggressive multi-start: random restarts + structural seeds
  * Final coordinate-descent radius maximization to squeeze out remaining slack
  * Tighter time management to fit within 150s
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


def _objective(x, deadline):
    """Objective: minimize negative sum of radii + smooth penalty."""
    if time.time() > deadline:
        return 1e10
    n = N
    pos = x[:2 * n].reshape(n, 2)
    radii = x[2 * n:]

    # Boundary violations
    left = radii - pos[:, 0]
    right = pos[:, 0] - (1.0 - radii)
    bot = radii - pos[:, 1]
    top = pos[:, 1] - (1.0 - radii)

    # Overlap violations
    d = pos[:, 0][:, None] - pos[:, 0][None, :]
    e = pos[:, 1][:, None] - pos[:, 1][None, :]
    dist = np.sqrt(d * d + e * e)
    min_dist = radii[:, None] + radii[None, :]
    viol = min_dist - dist
    np.fill_diagonal(viol, 0.0)

    all_v = np.concatenate([left, right, bot, top, viol.ravel()])
    pos_viol = np.maximum(all_v, 0.0)
    penalty = 50.0 * np.sum(pos_viol ** 2)

    return -np.sum(radii) + penalty


def _build_contact_graph(pos, radii, margin=0.01):
    """Build a contact graph: pairs of circles that are within (r_i + r_j + margin)."""
    contacts = set()
    for i in range(N):
        for j in range(i + 1, N):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < radii[i] + radii[j] + margin:
                contacts.add((i, j))
    return contacts


def _contact_slsqp(pos, radii, deadline, max_iter=200, ftol=1e-12):
    """SLSQP with only contact-graph constraints. Much faster than full constraint set."""
    pos = pos.copy()
    radii = radii.copy()
    x0 = np.concatenate([pos.flatten(), radii])

    # Bounds
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
    for i in range(N):
        bounds.append((1e-6, 0.35))

    # Build contact graph
    contacts = _build_contact_graph(pos, radii, margin=0.02)

    # Inequality constraints: g(x) >= 0
    cons = []
    # Boundary constraints (all circles)
    for i in range(N):
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2 * i] - x[2 * N + i]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: (1.0 - x[2 * i]) - x[2 * N + i]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2 * i + 1] - x[2 * N + i]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: (1.0 - x[2 * i + 1]) - x[2 * N + i]})

    # Non-overlap: only for contact pairs
    for (i, j) in contacts:
        cons.append({'type': 'ineq',
                     'fun': lambda x, i=i, j=j:
                     np.sqrt((x[2 * i] - x[2 * j]) ** 2 +
                             (x[2 * i + 1] - x[2 * j + 1]) ** 2)
                     - x[2 * N + i] - x[2 * N + j]})

    def neg_sum_rad(x):
        if time.time() > deadline:
            return 1e10
        return -np.sum(x[2 * N:])

    try:
        result = minimize(neg_sum_rad, x0, method='SLSQP', bounds=bounds,
                          constraints=cons,
                          options={'maxiter': max_iter, 'ftol': ftol})
        if result.x is not None:
            pos_opt = result.x[:2 * N].reshape(N, 2)
            radii_opt = result.x[2 * N:]
            return pos_opt, radii_opt
    except Exception:
        pass

    return pos, radii


def _full_slsqp(pos, radii, deadline, max_iter=300, ftol=1e-12):
    """SLSQP with all pairwise constraints (slower but more robust)."""
    pos = pos.copy()
    radii = radii.copy()
    x0 = np.concatenate([pos.flatten(), radii])

    bounds = [(0.0, 1.0)] * (2 * N) + [(1e-6, 0.35)] * N

    cons = []
    for i in range(N):
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2 * i] - x[2 * N + i]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: (1.0 - x[2 * i]) - x[2 * N + i]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2 * i + 1] - x[2 * N + i]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: (1.0 - x[2 * i + 1]) - x[2 * N + i]})
    for i in range(N):
        for j in range(i + 1, N):
            cons.append({'type': 'ineq',
                         'fun': lambda x, i=i, j=j:
                         np.sqrt((x[2 * i] - x[2 * j]) ** 2 +
                                 (x[2 * i + 1] - x[2 * j + 1]) ** 2)
                         - x[2 * N + i] - x[2 * N + j]})

    def neg_sum_rad(x):
        if time.time() > deadline:
            return 1e10
        return -np.sum(x[2 * N:])

    try:
        result = minimize(neg_sum_rad, x0, method='SLSQP', bounds=bounds,
                          constraints=cons,
                          options={'maxiter': max_iter, 'ftol': ftol})
        if result.x is not None:
            pos_opt = result.x[:2 * N].reshape(N, 2)
            radii_opt = result.x[2 * N:]
            return pos_opt, radii_opt
    except Exception:
        pass

    return pos, radii


def _post_process(pos, radii, deadline):
    """Iteratively shrink circles to ensure validity."""
    pos = pos.copy()
    radii = radii.copy()

    for _ in range(200):
        if time.time() > deadline:
            break
        valid = True
        for i in range(N):
            max_r = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
            if radii[i] > max_r:
                radii[i] = max_r * 0.9999
                valid = False
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    overlap = min_dist - dist
                    shrink_i = min(overlap * 0.5, radii[i] * 0.1)
                    shrink_j = min(overlap * 0.5, radii[j] * 0.1)
                    if shrink_i + shrink_j < overlap:
                        total_avail = shrink_i + shrink_j + 1e-12
                        scale = overlap / total_avail
                        shrink_i = min(shrink_i * scale, radii[i] * 0.3)
                        shrink_j = min(shrink_j * scale, radii[j] * 0.3)
                        if shrink_i + shrink_j < overlap:
                            if radii[i] <= radii[j]:
                                radii[i] -= (overlap - shrink_j)
                            else:
                                radii[j] -= (overlap - shrink_i)
                            valid = False
                            continue
                    radii[i] -= shrink_i
                    radii[j] -= shrink_j
                    valid = False
        if valid:
            break

    for i in range(N):
        max_r = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
        radii[i] = max(1e-6, min(radii[i], max_r))

    return pos, radii


def _grow_radii(pos, radii, deadline, max_steps=200):
    """Iteratively grow radii to fill available space."""
    pos = pos.copy()
    radii = radii.copy()
    for _ in range(max_steps):
        if time.time() > deadline:
            break
        grew = False
        for i in range(N):
            max_r = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
            for j in range(N):
                if i == j:
                    continue
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                max_r = min(max_r, dist - radii[j])
            if max_r > radii[i] * 1.001:
                new_r = min(max_r * 0.9999, radii[i] * 1.005)
                if new_r > radii[i]:
                    radii[i] = new_r
                    grew = True
        if not grew:
            break
    return pos, radii


def _coordinate_descent_radius(pos, radii, deadline, max_iter=1000):
    """Coordinate descent: for each circle, try to increase its radius
    while moving its center optimally. This is a 2D subproblem per circle."""
    pos = pos.copy()
    radii = radii.copy()

    for _ in range(max_iter):
        if time.time() > deadline:
            break
        improved = False
        for i in range(N):
            # For circle i, find the best (x, y, r) given all other circles fixed
            # Maximize r subject to:
            #   r <= x, r <= 1-x, r <= y, r <= 1-y
            #   dist(i, j) >= r + r_j for all j != i
            # This is a small QP-like problem; solve via grid search + local opt

            best_r = radii[i]
            best_x, best_y = pos[i, 0], pos[i, 1]

            # Coarse grid search around current position
            for dx in np.linspace(-0.02, 0.02, 5):
                for dy in np.linspace(-0.02, 0.02, 5):
                    x_try = np.clip(pos[i, 0] + dx, 0.01, 0.99)
                    y_try = np.clip(pos[i, 1] + dy, 0.01, 0.99)
                    # Max radius at this position
                    r_max = min(x_try, 1.0 - x_try, y_try, 1.0 - y_try)
                    for j in range(N):
                        if i == j:
                            continue
                        d = np.sqrt((x_try - pos[j, 0]) ** 2 + (y_try - pos[j, 1]) ** 2)
                        r_max = min(r_max, d - radii[j])
                    if r_max > best_r:
                        best_r = r_max
                        best_x, best_y = x_try, y_try

            if best_r > radii[i] * 1.001:
                radii[i] = best_r * 0.9999
                pos[i, 0] = best_x
                pos[i, 1] = best_y
                improved = True

        if not improved:
            break

    return pos, radii


def _make_grid(n, seed=42):
    side = 5
    pos = np.zeros((n, 2))
    radii = np.zeros(n)
    for i in range(side):
        for j in range(side):
            idx = i * side + j
            if idx < n:
                pos[idx] = [(i + 0.5) / side, (j + 0.5) / side]
                radii[idx] = 0.5 / side - 0.005
    return pos, radii


def _make_hexagonal(n, seed=42):
    pos = np.zeros((n, 2))
    radii = np.zeros(n)
    r = 0.1
    rows = 5
    for row in range(rows):
        y = 0.1 + row * r * np.sqrt(3)
        if row % 2 == 0:
            xs = np.arange(5) * 2 * r + r
        else:
            xs = np.arange(5) * 2 * r + 2 * r
        for col, x in enumerate(xs):
            idx = row * 5 + col
            if idx < n:
                pos[idx] = [x, y]
                radii[idx] = r
    return pos, radii


def _make_center_ring(n, seed=42):
    pos = np.zeros((n, 2))
    radii = np.zeros(n)
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


def _make_random(n, seed=42):
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.1, 0.9, size=(n, 2))
    radii = rng.uniform(0.03, 0.1, size=n)
    return pos, radii


def _make_structural(seed_params):
    """Create a structured initial configuration from parameters."""
    corner_r, center_r, cx, cy = seed_params[:4]
    r1_off, r1_r = seed_params[4:6]
    r2_off, r2_r = seed_params[6:8]

    pos = np.zeros((N, 2))
    radii = np.zeros(N)

    pos[0] = [corner_r, corner_r]
    pos[1] = [1.0 - corner_r, corner_r]
    pos[2] = [corner_r, 1.0 - corner_r]
    pos[3] = [1.0 - corner_r, 1.0 - corner_r]
    radii[0:4] = corner_r

    pos[4] = [cx, cy]
    radii[4] = center_r

    for i in range(8):
        ang = 2 * np.pi * i / 8.0
        pos[5 + i] = [cx + r1_off * np.cos(ang), cy + r1_off * np.sin(ang)]
        radii[5 + i] = r1_r

    for i in range(13):
        ang = 2 * np.pi * i / 13.0 + 0.1
        pos[13 + i] = [cx + r2_off * np.cos(ang), cy + r2_off * np.sin(ang)]
        radii[13 + i] = r2_r

    return pos, radii


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin

    best_pos = None
    best_radii = None
    best_score = -np.inf

    def update_best(pos, radii):
        nonlocal best_pos, best_radii, best_score
        if time.time() > deadline:
            return
        pos_pp, radii_pp = _post_process(pos, radii, deadline)
        if _is_valid(pos_pp, radii_pp):
            score = np.sum(radii_pp)
            if score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()

    # Generate diverse initial configurations
    initial_configs = []

    # Grid
    pos, radii = _make_grid(N)
    initial_configs.append((pos, radii))

    # Hexagonal
    pos, radii = _make_hexagonal(N)
    initial_configs.append((pos, radii))

    # Center + ring
    pos, radii = _make_center_ring(N)
    initial_configs.append((pos, radii))

    # Random starts
    for seed in [1, 2, 3, 4, 5]:
        pos, radii = _make_random(N, seed)
        initial_configs.append((pos, radii))

    # Structural starts
    struct_params_list = [
        [0.13, 0.16, 0.5, 0.5, 0.30, 0.11, 0.42, 0.07],
        [0.12, 0.18, 0.5, 0.5, 0.33, 0.10, 0.45, 0.06],
        [0.15, 0.14, 0.5, 0.5, 0.28, 0.12, 0.40, 0.08],
        [0.14, 0.15, 0.5, 0.5, 0.31, 0.105, 0.43, 0.065],
    ]
    for sp in struct_params_list:
        pos, radii = _make_structural(sp)
        initial_configs.append((pos, radii))

    # Phase 1: Contact-graph SLSQP from each initial config
    for pos0, radii0 in initial_configs:
        if time.time() > deadline - 30:
            break
        # First do a quick L-BFGS-B to get a valid starting point
        x0 = np.concatenate([pos0.flatten(), radii0])
        bounds = []
        for i in range(N):
            bounds.append((0.01, 0.99))
            bounds.append((0.01, 0.99))
        for i in range(N):
            bounds.append((0.01, 0.32))
        try:
            result = minimize(lambda x: _objective(x, deadline), x0, method='L-BFGS-B',
                              bounds=bounds, options={'maxiter': 500, 'ftol': 1e-12})
            pos_opt = result.x[:2 * N].reshape(N, 2)
            radii_opt = result.x[2 * N:]
            pos_pp, radii_pp = _post_process(pos_opt, radii_opt, deadline)
            if _is_valid(pos_pp, radii_pp):
                # Now do contact-graph SLSQP
                if time.time() < deadline - 25:
                    pos_c, radii_c = _contact_slsqp(pos_pp, radii_pp, deadline, max_iter=150)
                    pos_c, radii_c = _post_process(pos_c, radii_c, deadline)
                    if _is_valid(pos_c, radii_c):
                        update_best(pos_c, radii_c)
                    else:
                        update_best(pos_pp, radii_pp)
                else:
                    update_best(pos_pp, radii_pp)
        except Exception:
            update_best(pos0, radii0)

    # Phase 2: Full SLSQP from best result
    if best_pos is not None and time.time() < deadline - 20:
        try:
            pos_f, radii_f = _full_slsqp(best_pos, best_radii, deadline, max_iter=200)
            pos_f, radii_f = _post_process(pos_f, radii_f, deadline)
            if _is_valid(pos_f, radii_f):
                update_best(pos_f, radii_f)
        except Exception:
            pass

    # Phase 3: Contact-graph SLSQP again (refined contact graph)
    if best_pos is not None and time.time() < deadline - 15:
        try:
            pos_c2, radii_c2 = _contact_slsqp(best_pos, best_radii, deadline, max_iter=200, ftol=1e-14)
            pos_c2, radii_c2 = _post_process(pos_c2, radii_c2, deadline)
            if _is_valid(pos_c2, radii_c2):
                update_best(pos_c2, radii_c2)
        except Exception:
            pass

    # Phase 4: Coordinate descent radius maximization
    if best_pos is not None and time.time() < deadline - 10:
        try:
            pos_cd, radii_cd = _coordinate_descent_radius(best_pos, best_radii, deadline, max_iter=500)
            pos_cd, radii_cd = _post_process(pos_cd, radii_cd, deadline)
            if _is_valid(pos_cd, radii_cd):
                update_best(pos_cd, radii_cd)
        except Exception:
            pass

    # Phase 5: Final contact-graph SLSQP polish
    if best_pos is not None and time.time() < deadline - 5:
        try:
            pos_p, radii_p = _contact_slsqp(best_pos, best_radii, deadline, max_iter=300, ftol=1e-15)
            pos_p, radii_p = _post_process(pos_p, radii_p, deadline)
            if _is_valid(pos_p, radii_p):
                update_best(pos_p, radii_p)
        except Exception:
            pass

    # Phase 6: Final radius growth
    if best_pos is not None and time.time() < deadline - 1:
        try:
            pos_g, radii_g = _grow_radii(best_pos, best_radii, deadline, max_steps=300)
            if _is_valid(pos_g, radii_g):
                s = np.sum(radii_g)
                if s > best_score:
                    best_score = s
                    best_pos = pos_g.copy()
                    best_radii = radii_g.copy()
        except Exception:
            pass

    # Fallback
    if best_pos is None:
        pos, radii = _make_grid(N)
        best_pos, best_radii = _post_process(pos, radii, deadline)

    return np.column_stack([best_pos, best_radii])

# EVOLVE-BLOCK-END
