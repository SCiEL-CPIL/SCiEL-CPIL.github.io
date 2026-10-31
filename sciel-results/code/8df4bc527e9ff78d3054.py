"""Circle packing in unit square - 26 circles maximizing sum of radii.
Combines multi-phase local optimization with a compact structural
parameterization (corner + center + rings) that reduces the global-search
dimensionality so Differential Evolution can actually explore it.

Key improvements over parent:
  * Smooth combined penalty (quadratic + log-sum-exp soft-max) for better gradients
  * Hard-constrained SLSQP polish pass (strict non-overlap and boundary enforcement)
  * Re-tuned time budget: trim redundant direct 78-dim track, give more time to
    DE structural exploration and the final high-precision polish
  * Smarter post-processing that shrinks the smaller circle preferentially
"""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

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
    """Objective: minimize negative sum of radii + smooth combined penalty.
    
    Uses quadratic penalty on positive violations (smooth, captures total magnitude)
    plus a log-sum-exp soft-max on the worst violation (directional, focuses on
    binding constraints). This gives both global smoothness and local directionality.
    """
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

    # Overlap violations (vectorized)
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
        # Numerically stable log-sum-exp
        sm = (np.log(np.sum(np.exp(shifted))) + beta * m) / beta
    else:
        sm = 0.0

    penalty = 50.0 * quad_penalty + 20.0 * sm
    return -np.sum(radii) + penalty


# ---------------- Compact structural parameterization ----------------
# p = [corner_r, center_r, cx, cy,
#      ring1_off, ring1_r, ring1_r2,
#      ring2_off, ring2_r, ring2_r2]
# ring1: 8 circles, ring2: 13 circles, plus 4 corners + 1 center = 26.

def _expand_struct(p):
    """Map a 10-dim structural vector to (pos, radii) of 26 circles."""
    corner_r = np.clip(p[0], 0.05, 0.30)
    center_r = np.clip(p[1], 0.05, 0.35)
    cx = np.clip(p[2], 0.2, 0.8)
    cy = np.clip(p[3], 0.2, 0.8)
    r1_off = np.clip(p[4], 0.15, 0.6)
    r1_r = np.clip(p[5], 0.03, 0.20)
    r1_r2 = np.clip(p[6], 0.03, 0.20)
    r2_off = np.clip(p[7], 0.25, 0.7)
    r2_r = np.clip(p[8], 0.02, 0.16)
    r2_r2 = np.clip(p[9], 0.02, 0.16)

    pos = np.zeros((N, 2))
    radii = np.zeros(N)

    # 4 corners
    pos[0] = [corner_r, corner_r]
    pos[1] = [1.0 - corner_r, corner_r]
    pos[2] = [corner_r, 1.0 - corner_r]
    pos[3] = [1.0 - corner_r, 1.0 - corner_r]
    radii[0:4] = corner_r

    # center
    pos[4] = [cx, cy]
    radii[4] = center_r

    # ring 1: 8 circles (indices 5..12)
    for i in range(8):
        ang = 2 * np.pi * i / 8.0
        pos[5 + i] = [cx + r1_off * np.cos(ang), cy + r1_off * np.sin(ang)]
        radii[5 + i] = r1_r if i % 2 == 0 else r1_r2

    # ring 2: 13 circles (indices 13..25)
    for i in range(13):
        ang = 2 * np.pi * i / 13.0 + 0.1
        pos[13 + i] = [cx + r2_off * np.cos(ang), cy + r2_off * np.sin(ang)]
        radii[13 + i] = r2_r if i % 2 == 0 else r2_r2

    return pos, radii


def _struct_objective(p, deadline):
    pos, radii = _expand_struct(p)
    x = np.concatenate([pos.flatten(), radii])
    return _objective(x, deadline)


# ---------------- Structural seeds ----------------

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


# ---------------- Post-processing & polish ----------------

def _post_process(pos, radii, deadline):
    """Iteratively shrink circles to ensure validity, preferring to shrink smaller circles."""
    pos = pos.copy()
    radii = radii.copy()

    for _ in range(500):
        if time.time() > deadline:
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
                    # Shrink the smaller circle more to preserve larger ones
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
        radii[i] = max(1e-6, min(radii[i], max_r))

    return pos, radii


def _slsqp_polish(pos, radii, deadline):
    """High-precision SLSQP polish with hard constraints and analytic Jacobians."""
    pos = pos.copy()
    radii = radii.copy()
    x0 = np.concatenate([pos.flatten(), radii])

    # Bounds: positions must allow the circle to fit, radii must be positive
    bounds = []
    for i in range(N):
        bounds.append((0.001, 0.999))
        bounds.append((0.001, 0.999))
    for i in range(N):
        bounds.append((1e-4, 0.35))

    # Boundary constraints with analytic Jacobians
    # x_i - r_i >= 0
    def bnd_l_val(x, i): return x[2*i] - x[2*N + i]
    def bnd_l_jac(x, i):
        j = np.zeros(3*N)
        j[2*i] = 1.0
        j[2*N + i] = -1.0
        return j
    # 1 - x_i - r_i >= 0
    def bnd_r_val(x, i): return 1.0 - x[2*i] - x[2*N + i]
    def bnd_r_jac(x, i):
        j = np.zeros(3*N)
        j[2*i] = -1.0
        j[2*N + i] = -1.0
        return j
    # y_i - r_i >= 0
    def bnd_b_val(x, i): return x[2*i+1] - x[2*N + i]
    def bnd_b_jac(x, i):
        j = np.zeros(3*N)
        j[2*i+1] = 1.0
        j[2*N + i] = -1.0
        return j
    # 1 - y_i - r_i >= 0
    def bnd_t_val(x, i): return 1.0 - x[2*i+1] - x[2*N + i]
    def bnd_t_jac(x, i):
        j = np.zeros(3*N)
        j[2*i+1] = -1.0
        j[2*N + i] = -1.0
        return j

    cons = []
    for i in range(N):
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: bnd_l_val(x, i),
                     'jac': lambda x, i=i: bnd_l_jac(x, i)})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: bnd_r_val(x, i),
                     'jac': lambda x, i=i: bnd_r_jac(x, i)})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: bnd_b_val(x, i),
                     'jac': lambda x, i=i: bnd_b_jac(x, i)})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: bnd_t_val(x, i),
                     'jac': lambda x, i=i: bnd_t_jac(x, i)})

    # Non-overlap constraints with analytic Jacobians
    for i in range(N):
        for j in range(i + 1, N):
            def ov_val(x, i=i, j=j):
                dx = x[2*i] - x[2*j]
                dy = x[2*i+1] - x[2*j+1]
                d = np.sqrt(dx*dx + dy*dy)
                return d - x[2*N + i] - x[2*N + j]
            def ov_jac(x, i=i, j=j):
                dx = x[2*i] - x[2*j]
                dy = x[2*i+1] - x[2*j+1]
                d = np.sqrt(dx*dx + dy*dy)
                if d < 1e-15:
                    d = 1e-15
                jv = np.zeros(3*N)
                jv[2*i] = dx / d
                jv[2*i+1] = dy / d
                jv[2*j] = -dx / d
                jv[2*j+1] = -dy / d
                jv[2*N + i] = -1.0
                jv[2*N + j] = -1.0
                return jv
            cons.append({'type': 'ineq', 'fun': ov_val, 'jac': ov_jac})

    def obj(x):
        return -np.sum(x[2*N:])
    def obj_jac(x):
        j = np.zeros(3*N)
        j[2*N:] = -1.0
        return j

    try:
        result = minimize(obj, x0, method='SLSQP', jac=obj_jac,
                          bounds=bounds, constraints=cons,
                          options={'maxiter': 1000, 'ftol': 1e-15})
        pos_opt = result.x[:2 * N].reshape(N, 2)
        radii_opt = result.x[2 * N:]
        pos_pp, radii_pp = _post_process(pos_opt, radii_opt, deadline)
        if _is_valid(pos_pp, radii_pp):
            return pos_pp, radii_pp
    except Exception:
        pass
    return pos, radii


def _radius_growth(pos, radii, deadline):
    """Exploit slack: grow each circle as much as possible given current positions.
    Uses a conservative iterative scheme that respects all pairwise constraints."""
    pos = pos.copy()
    radii = radii.copy()
    n = N
    for _ in range(300):
        if time.time() > deadline:
            break
        changed = False
        # For each circle, compute max radius given current positions and other radii
        for i in range(n):
            # Boundary limit
            max_r = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
            # Pairwise limits
            for j in range(n):
                if i == j:
                    continue
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                # dist >= r_i + r_j  =>  r_i <= dist - r_j
                max_r = min(max_r, dist - radii[j])
            max_r = max(max_r, 1e-6)
            if max_r > radii[i] * (1.0 + 1e-10):
                radii[i] = max_r * 0.99995
                changed = True
        if not changed:
            break
    # Final boundary clamp
    for i in range(n):
        max_r = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
        radii[i] = max(1e-6, min(radii[i], max_r))
    return pos, radii


def _coord_descent_polish(pos, radii, deadline):
    """Per-circle coordinate descent: for each circle, binary-search the maximum
    radius that keeps all constraints satisfied, given fixed positions of others.
    Then optionally nudge position toward the binding constraint to allow further growth.
    This is a smooth local refinement that exploits contact constraints tightly."""
    pos = pos.copy()
    radii = radii.copy()
    n = N
    max_rounds = 200
    for _ in range(max_rounds):
        if time.time() > deadline - 1:
            break
        changed = False
        for i in range(n):
            # Binary search max radius for circle i given fixed positions
            lo = radii[i] * 0.5
            hi = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1]) * 1.01
            # Ensure hi is at least current radius
            hi = max(hi, radii[i])
            # Check if current radius is valid (sanity)
            def _r_valid(r, i, pos, radii):
                if r < 1e-9:
                    return False
                # Boundary
                if pos[i, 0] - r < -1e-12 or pos[i, 0] + r > 1.0 + 1e-12:
                    return False
                if pos[i, 1] - r < -1e-12 or pos[i, 1] + r > 1.0 + 1e-12:
                    return False
                # Pairwise
                for j in range(n):
                    if i == j:
                        continue
                    dx = pos[i, 0] - pos[j, 0]
                    dy = pos[i, 1] - pos[j, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    if dist < r + radii[j] - 1e-12:
                        return False
                return True
            if not _r_valid(radii[i], i, pos, radii):
                # Shrink to valid
                for _bs in range(30):
                    mid = (lo + hi) / 2
                    if _r_valid(mid, i, pos, radii):
                        lo = mid
                    else:
                        hi = mid
                radii[i] = max(lo, 1e-6)
                changed = True
                continue
            # Binary search for max valid radius
            for _bs in range(40):
                mid = (lo + hi) / 2
                if _r_valid(mid, i, pos, radii):
                    lo = mid
                else:
                    hi = mid
            new_r = max(lo * 0.99999, 1e-6)
            if new_r > radii[i] * (1.0 + 1e-11):
                radii[i] = new_r
                changed = True
        if not changed:
            break
    # Final clamp
    for i in range(n):
        max_r = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
        radii[i] = max(1e-6, min(radii[i], max_r))
    return pos, radii


def _high_beta_smooth_pass(pos, radii, deadline):
    """Run L-BFGS-B with a very high-beta smooth penalty to squeeze radius
    from constraint boundaries. Uses log-sum-exp with beta=300 for near-exact
    max-violation focusing."""
    pos = pos.copy()
    radii = radii.copy()
    x0 = np.concatenate([pos.flatten(), radii])

    def _obj_high_beta(x):
        if time.time() > deadline:
            return 1e10
        n = N
        p = x[:2 * n].reshape(n, 2)
        r = x[2 * n:]
        left = r - p[:, 0]
        right = p[:, 0] - (1.0 - r)
        bot = r - p[:, 1]
        top = p[:, 1] - (1.0 - r)
        d = p[:, 0][:, None] - p[:, 0][None, :]
        e = p[:, 1][:, None] - p[:, 1][None, :]
        dist = np.sqrt(d * d + e * e)
        min_dist = r[:, None] + r[None, :]
        viol = min_dist - dist
        np.fill_diagonal(viol, 0.0)
        all_v = np.concatenate([left, right, bot, top, viol.ravel()])
        pos_viol = np.maximum(all_v, 0.0)
        quad_penalty = np.sum(pos_viol ** 2)
        beta = 300.0
        m = np.max(all_v)
        if m > 0:
            shifted = beta * (all_v - m)
            sm = (np.log(np.sum(np.exp(shifted))) + beta * m) / beta
        else:
            sm = 0.0
        penalty = 100.0 * quad_penalty + 50.0 * sm
        return -np.sum(r) + penalty

    bounds = []
    for i in range(N):
        bounds.append((0.001, 0.999))
        bounds.append((0.001, 0.999))
    for i in range(N):
        bounds.append((1e-5, 0.35))

    try:
        result = minimize(_obj_high_beta, x0, method='L-BFGS-B',
                          bounds=bounds, options={'maxiter': 800, 'ftol': 1e-15, 'gtol': 1e-13})
        pos_opt = result.x[:2 * N].reshape(N, 2)
        radii_opt = result.x[2 * N:]
        pos_pp, radii_pp = _post_process(pos_opt, radii_opt, deadline)
        if _is_valid(pos_pp, radii_pp):
            return pos_pp, radii_pp
    except Exception:
        pass
    return pos, radii


def _polish(pos, radii, deadline):
    """Multi-stage polish: L-BFGS-B first, then SLSQP for hard constraints."""
    pos = pos.copy()
    radii = radii.copy()
    x0 = np.concatenate([pos.flatten(), radii])
    bounds = []
    for i in range(N):
        bounds.append((max(0.01, pos[i, 0] - radii[i] * 0.5), min(0.99, pos[i, 0] + radii[i] * 0.5)))
        bounds.append((max(0.01, pos[i, 1] - radii[i] * 0.5), min(0.99, pos[i, 1] + radii[i] * 0.5)))
    for i in range(N):
        bounds.append((1e-4, 0.35))

    # Stage 1: L-BFGS-B with penalty objective
    try:
        result = minimize(lambda x: _objective(x, deadline), x0, method='L-BFGS-B',
                          bounds=bounds, options={'maxiter': 600, 'ftol': 1e-14, 'gtol': 1e-12})
        pos_opt = result.x[:2 * N].reshape(N, 2)
        radii_opt = result.x[2 * N:]
        pos_pp, radii_pp = _post_process(pos_opt, radii_opt, deadline)
        if _is_valid(pos_pp, radii_pp):
            pos, radii = pos_pp, radii_pp
    except Exception:
        pass

    # Stage 2: SLSQP with hard constraints (if time allows)
    if time.time() < deadline - 8:
        pos_s, radii_s = _slsqp_polish(pos, radii, deadline)
        if _is_valid(pos_s, radii_s):
            s_new = np.sum(radii_s)
            s_old = np.sum(radii)
            if s_new >= s_old - 1e-10:
                pos, radii = pos_s, radii_s

    # Stage 3: radius growth pass (exploit slack in contact constraints)
    if time.time() < deadline - 2:
        pos_g, radii_g = _radius_growth(pos, radii, deadline)
        if _is_valid(pos_g, radii_g):
            s_g = np.sum(radii_g)
            s_cur = np.sum(radii)
            if s_g >= s_cur - 1e-10:
                pos, radii = pos_g, radii_g

    # Stage 4: Coordinate descent polish (tight binary search per circle)
    if time.time() < deadline - 4:
        pos_cd, radii_cd = _coord_descent_polish(pos, radii, deadline)
        if _is_valid(pos_cd, radii_cd):
            s_cd = np.sum(radii_cd)
            s_cur_cd = np.sum(radii)
            if s_cd >= s_cur_cd - 1e-10:
                pos, radii = pos_cd, radii_cd

    # Stage 5: High-beta smooth penalty pass to squeeze from boundaries
    if time.time() < deadline - 5:
        pos_hb, radii_hb = _high_beta_smooth_pass(pos, radii, deadline)
        if _is_valid(pos_hb, radii_hb):
            s_hb = np.sum(radii_hb)
            s_cur_hb = np.sum(radii)
            if s_hb >= s_cur_hb - 1e-10:
                pos, radii = pos_hb, radii_hb

    # Stage 6: Another SLSQP pass on the refined configuration (if time allows)
    if time.time() < deadline - 3:
        pos_s2, radii_s2 = _slsqp_polish(pos, radii, deadline)
        if _is_valid(pos_s2, radii_s2):
            s_new2 = np.sum(radii_s2)
            s_cur2 = np.sum(radii)
            if s_new2 >= s_cur2 - 1e-10:
                pos, radii = pos_s2, radii_s2

    # Stage 7: Final coordinate descent + radius growth squeeze
    if time.time() < deadline - 1:
        pos_f1, radii_f1 = _coord_descent_polish(pos, radii, deadline)
        if _is_valid(pos_f1, radii_f1):
            s_f1 = np.sum(radii_f1)
            s_cur_f1 = np.sum(radii)
            if s_f1 >= s_cur_f1 - 1e-10:
                pos, radii = pos_f1, radii_f1
        pos_f2, radii_f2 = _radius_growth(pos, radii, deadline)
        if _is_valid(pos_f2, radii_f2):
            s_f2 = np.sum(radii_f2)
            s_cur_f2 = np.sum(radii)
            if s_f2 >= s_cur_f2 - 1e-10:
                pos, radii = pos_f2, radii_f2

    return pos, radii


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin under 150s

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

    # ---------- Track A: compact structural parameterization ----------
    struct_bounds = [
        (0.06, 0.28),   # corner_r
        (0.06, 0.32),   # center_r
        (0.3, 0.7),     # cx
        (0.3, 0.7),     # cy
        (0.18, 0.45),   # ring1 offset
        (0.04, 0.16),   # ring1 r
        (0.04, 0.16),   # ring1 r2
        (0.30, 0.55),   # ring2 offset
        (0.03, 0.12),   # ring2 r
        (0.03, 0.12),   # ring2 r2
    ]

    if time.time() < deadline - 55:
        try:
            de_res = differential_evolution(_struct_objective, struct_bounds,
                                            args=(deadline,),
                                            maxiter=150, popsize=22, seed=7,
                                            tol=1e-8, polish=False,
                                            mutation=(0.4, 1.0), recombination=0.7,
                                            updating='deferred')
            if de_res.x is not None:
                pos_s, radii_s = _expand_struct(de_res.x)
                update_best(pos_s, radii_s)
                # Polish the DE structural result in full space
                if time.time() < deadline - 20:
                    pos_p, radii_p = _polish(pos_s, radii_s, deadline)
                    if _is_valid(pos_p, radii_p):
                        s = np.sum(radii_p)
                        if s > best_score:
                            best_score = s
                            best_pos = pos_p.copy()
                            best_radii = radii_p.copy()
        except Exception:
            pass

    # L-BFGS-B on structural vector from a few good seeds
    struct_seeds = [
        [0.13, 0.16, 0.5, 0.5, 0.30, 0.11, 0.11, 0.42, 0.07, 0.07],
        [0.12, 0.18, 0.5, 0.5, 0.33, 0.10, 0.10, 0.45, 0.06, 0.06],
        [0.15, 0.14, 0.5, 0.5, 0.28, 0.12, 0.12, 0.40, 0.08, 0.08],
    ]
    for s0 in struct_seeds:
        if time.time() > deadline - 25:
            break
        try:
            res = minimize(_struct_objective, np.array(s0), args=(deadline,),
                           method='L-BFGS-B',
                           bounds=struct_bounds,
                           options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
            pos_s, radii_s = _expand_struct(res.x)
            update_best(pos_s, radii_s)
        except Exception:
            pass

    # ---------- Track B: direct optimization from diverse seeds ----------
    configs = []
    configs.append(_make_grid(N))
    configs.append(_make_hexagonal(N))
    configs.append(_make_center_ring(N))
    pos_s0, radii_s0 = _expand_struct(np.array(struct_seeds[0]))
    configs.append((pos_s0, radii_s0))

    for pos, radii in configs:
        if time.time() > deadline - 15:
            break
        x0 = np.concatenate([pos.flatten(), radii])
        bounds = []
        for i in range(N):
            bounds.append((0.01, 0.99))
            bounds.append((0.01, 0.99))
        for i in range(N):
            bounds.append((0.01, 0.32))
        try:
            result = minimize(lambda x: _objective(x, deadline), x0, method='L-BFGS-B',
                              bounds=bounds, options={'maxiter': 400, 'ftol': 1e-11})
            if result.success or result.fun < _objective(x0, deadline):
                pos_opt = result.x[:2 * N].reshape(N, 2)
                radii_opt = result.x[2 * N:]
                update_best(pos_opt, radii_opt)
        except Exception:
            pass

    # ---------- Final polish of overall best ----------
    if best_pos is not None and time.time() < deadline - 8:
        pos_p, radii_p = _polish(best_pos, best_radii, deadline)
        if _is_valid(pos_p, radii_p):
            s = np.sum(radii_p)
            if s > best_score:
                best_score = s
                best_pos = pos_p.copy()
                best_radii = radii_p.copy()

    # Final radius growth squeeze
    if best_pos is not None and time.time() < deadline - 4:
        pos_g, radii_g = _radius_growth(best_pos, best_radii, deadline)
        if _is_valid(pos_g, radii_g):
            s_g = np.sum(radii_g)
            if s_g > best_score:
                best_score = s_g
                best_pos = pos_g.copy()
                best_radii = radii_g.copy()

    # Final coordinate descent polish
    if best_pos is not None and time.time() < deadline - 3:
        pos_cd, radii_cd = _coord_descent_polish(best_pos, best_radii, deadline)
        if _is_valid(pos_cd, radii_cd):
            s_cd = np.sum(radii_cd)
            if s_cd > best_score:
                best_score = s_cd
                best_pos = pos_cd.copy()
                best_radii = radii_cd.copy()

    # Final high-beta smooth pass
    if best_pos is not None and time.time() < deadline - 4:
        pos_hb, radii_hb = _high_beta_smooth_pass(best_pos, best_radii, deadline)
        if _is_valid(pos_hb, radii_hb):
            s_hb = np.sum(radii_hb)
            if s_hb > best_score:
                best_score = s_hb
                best_pos = pos_hb.copy()
                best_radii = radii_hb.copy()

    # Final SLSQP polish if time allows
    if best_pos is not None and time.time() < deadline - 2:
        pos_f, radii_f = _slsqp_polish(best_pos, best_radii, deadline)
        if _is_valid(pos_f, radii_f):
            s_f = np.sum(radii_f)
            if s_f > best_score:
                best_score = s_f
                best_pos = pos_f.copy()
                best_radii = radii_f.copy()

    # Final coordinate descent squeeze
    if best_pos is not None and time.time() < deadline - 0.5:
        pos_cf, radii_cf = _coord_descent_polish(best_pos, best_radii, deadline)
        if _is_valid(pos_cf, radii_cf):
            s_cf = np.sum(radii_cf)
            if s_cf > best_score:
                best_score = s_cf
                best_pos = pos_cf.copy()
                best_radii = radii_cf.copy()

    # Fallback
    if best_pos is None:
        pos, radii = _make_grid(N)
        best_pos, best_radii = _post_process(pos, radii, deadline)

    return np.column_stack([best_pos, best_radii])

# EVOLVE-BLOCK-END
