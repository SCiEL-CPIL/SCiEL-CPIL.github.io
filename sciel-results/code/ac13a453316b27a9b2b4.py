"""Circle packing in perimeter-4 rectangle: SLSQP + pressure-field growth + contact-graph refine + LSE epigraph."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21


def _vectorized_objective(z, penalty_weight=1000.0):
    """Vectorized objective with smooth penalties for speed."""
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]
    w = z[-1]
    h = 2.0 - w

    if w < 0.2 or h < 0.2 or np.any(r < 0):
        return 1e6

    obj = -np.sum(r)

    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]

    iu = np.triu_indices(N, k=1)
    d_sq_pairs = d_sq[iu]
    r_sum_pairs = r_sum[iu]

    d_pairs = np.sqrt(np.maximum(d_sq_pairs, 0.0))
    overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
    overlap_penalty = penalty_weight * np.sum(overlap * overlap)

    obj += overlap_penalty

    left_viol = np.maximum(r - x, 0.0)
    right_viol = np.maximum(x + r - w, 0.0)
    bottom_viol = np.maximum(r - y, 0.0)
    top_viol = np.maximum(y + r - h, 0.0)

    boundary_penalty = penalty_weight * (
        np.sum(left_viol * left_viol) +
        np.sum(right_viol * right_viol) +
        np.sum(bottom_viol * bottom_viol) +
        np.sum(top_viol * top_viol)
    )

    obj += boundary_penalty

    if w < 0.5:
        obj += penalty_weight * (0.5 - w) ** 2
    elif w > 1.5:
        obj += penalty_weight * (w - 1.5) ** 2

    return obj


def _is_valid(cx, cy, r, W, H, tol=1e-8):
    """Check if packing is valid."""
    if W <= 0.05 or H <= 0.05:
        return False
    if np.any(r <= 0):
        return False
    if np.any(cx - r < -tol) or np.any(cy - r < -tol):
        return False
    if np.any(cx + r > W + tol) or np.any(cy + r > H + tol):
        return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            if dx * dx + dy * dy < (r[i] + r[j]) ** 2 - tol:
                return False
    return True


def _force_relax_unified(z, max_iter=500):
    """Force-directed relaxation in unified parameter space with adaptive step."""
    z = z.copy().astype(float)
    n = N
    dt = 0.005
    min_dt = 0.0001
    convergence_tol = 1e-8

    for it in range(max_iter):
        x = z[0::3][:n]
        y = z[1::3][:n]
        r = z[2::3][:n]
        w = z[-1]
        h = 2.0 - w

        if w < 0.1 or h < 0.1:
            break

        fx = np.zeros(n)
        fy = np.zeros(n)

        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        fx -= left_viol * 100
        fx += right_viol * 100
        fy -= bottom_viol * 100
        fy += top_viol * 100

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]

        iu = np.triu_indices(n, k=1)
        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]

        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
        overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)
        max_overlap = float(np.max(overlap)) if overlap.size > 0 else 0.0

        if max_overlap > convergence_tol:
            f_pairs = overlap * 15.0
            dx_pairs = dx[iu]
            dy_pairs = dy[iu]
            ux = dx_pairs / d_pairs
            uy = dy_pairs / d_pairs

            for idx in range(len(iu[0])):
                i_idx = iu[0][idx]
                j_idx = iu[1][idx]
                fx[i_idx] += ux[idx] * f_pairs[idx]
                fy[i_idx] += uy[idx] * f_pairs[idx]
                fx[j_idx] -= ux[idx] * f_pairs[idx]
                fy[j_idx] -= uy[idx] * f_pairs[idx]

        if max_overlap < convergence_tol:
            break

        force_mag = float(np.max(np.sqrt(fx * fx + fy * fy)))
        if force_mag > 1.0:
            dt = max(min_dt, dt * 0.5)
        elif force_mag < 0.1:
            dt = min(0.01, dt * 1.1)

        z[0::3][:n] += dt * fx
        z[1::3][:n] += dt * fy

        for i in range(n):
            z[3 * i] = np.clip(z[3 * i], r[i], w - r[i])
            z[3 * i + 1] = np.clip(z[3 * i + 1], r[i], h - r[i])

    return z


def _soft_min_clearance(x, y, r, w, h, i, soft_len=0.03):
    """Smooth soft-min of clearance distances for circle i via log-sum-exp."""
    wall_dists = np.array([
        x[i] - r[i],
        w - x[i] - r[i],
        y[i] - r[i],
        h - y[i] - r[i]
    ])

    neighbor_dists = np.empty(N - 1)
    k = 0
    for j in range(N):
        if i == j:
            continue
        dx = x[i] - x[j]
        dy = y[i] - y[j]
        d = np.sqrt(dx * dx + dy * dy)
        neighbor_dists[k] = d - r[i] - r[j]
        k += 1

    all_dists = np.concatenate([wall_dists, neighbor_dists])

    neg_dists = -all_dists
    max_neg = np.max(neg_dists)
    if max_neg > 50:
        max_neg = 50
    exp_neg = np.exp((neg_dists - max_neg) / soft_len)
    soft_max_neg = max_neg + soft_len * np.log(np.sum(exp_neg))
    return -soft_max_neg


def _pressure_field_growth(z, time_budget, n_rounds=20):
    """
    Sequential radial growth using a smooth pressure field.
    Grows the most-constrained (lowest-clearance) circles first,
    interleaved with force relaxation to create space.
    """
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    soft_len = 0.03
    growth_rate = 0.003

    for rnd in range(n_rounds):
        if time.time() > time_budget:
            break

        pressures = np.zeros(n)
        for i in range(n):
            clearance = _soft_min_clearance(x, y, r, w, h, i, soft_len)
            pressures[i] = -clearance  # higher pressure = more constrained

        order = np.argsort(-pressures)  # most constrained first

        for i in order:
            if time.time() > time_budget:
                break

            new_r = r[i] * (1.0 + growth_rate)
            r_test = r.copy()
            r_test[i] = new_r
            if _is_valid(x, y, r_test, w, h):
                r[i] = new_r
            else:
                lo = r[i]
                hi = new_r
                for _ in range(10):
                    mid = (lo + hi) / 2.0
                    r_test = r.copy()
                    r_test[i] = mid
                    if _is_valid(x, y, r_test, w, h):
                        lo = mid
                    else:
                        hi = mid
                r[i] = lo

        # Force relaxation to create space after growth (keep radii, update positions)
        z_temp = np.empty(3 * N + 1)
        z_temp[0::3][:n] = x
        z_temp[1::3][:n] = y
        z_temp[2::3][:n] = r
        z_temp[-1] = w

        z_relaxed = _force_relax_unified(z_temp, max_iter=80)
        x = z_relaxed[0::3][:n].copy()
        y = z_relaxed[1::3][:n].copy()
        for i in range(n):
            x[i] = np.clip(x[i], r[i], w - r[i])
            y[i] = np.clip(y[i], r[i], h - r[i])

    if not _is_valid(x, y, r, w, h):
        # Repair by shrinking slightly
        for factor in [0.999, 0.998, 0.995, 0.99]:
            if _is_valid(x, y, r * factor, w, h):
                r = r * factor
                break
        if not _is_valid(x, y, r, w, h):
            return None

    z_new = np.empty(3 * N + 1)
    z_new[0::3][:n] = x
    z_new[1::3][:n] = y
    z_new[2::3][:n] = r
    z_new[-1] = w
    return z_new


def _contact_graph_refine(z, time_budget):
    """
    Contact-graph-aware local search: find mutually-contacted clusters and
    solve each cluster jointly with SLSQP to maximize sum of radii.
    """
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    # Build contact graph (edges where clearance is small)
    contact_tol = 0.005
    adj = {i: set() for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx * dx + dy * dy)
            clearance = d - r[i] - r[j]
            if clearance < contact_tol:
                adj[i].add(j)
                adj[j].add(i)

    # Find connected components
    visited = set()
    clusters = []
    for start in range(n):
        if start in visited:
            continue
        comp = []
        stack = [start]
        visited.add(start)
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nb in adj[cur]:
                if nb not in visited:
                    visited.add(nb)
                    stack.append(nb)
        if len(comp) >= 2:
            clusters.append(comp)

    # Solve each cluster jointly
    for comp in clusters:
        if time.time() > time_budget:
            break
        if len(comp) < 2:
            continue
        idx = np.array(comp)
        m = len(idx)

        # Local objective: maximize sum of radii in cluster, with penalties
        def cluster_obj(v):
            xi = v[0::3][:m]
            yi = v[1::3][:m]
            ri = v[2::3][:m]
            if np.any(ri < 0):
                return 1e6
            obj = -np.sum(ri)
            # Intra-cluster overlap
            dxl = xi[:, None] - xi[None, :]
            dyl = yi[:, None] - yi[None, :]
            dl = np.sqrt(np.maximum(dxl * dxl + dyl * dyl, 1e-12))
            rl = ri[:, None] + ri[None, :]
            iu = np.triu_indices(m, k=1)
            overlap = np.maximum(rl[iu] - dl[iu], 0.0)
            obj += 500.0 * np.sum(overlap * overlap)
            # Boundary
            for a, ii in enumerate(idx):
                pass
            # Use global positions for boundary via mapping
            return obj

        # Build initial vector
        v0 = np.empty(3 * m)
        v0[0::3] = x[idx]
        v0[1::3] = y[idx]
        v0[2::3] = r[idx]

        # Full objective with global context (penalize overlap with all circles + boundary)
        def obj_full(v):
            xi = v[0::3][:m]
            yi = v[1::3][:m]
            ri = v[2::3][:m]
            if np.any(ri < 0):
                return 1e6
            obj = -np.sum(ri)
            # Overlap with all circles (including outside cluster)
            for a, ii in enumerate(idx):
                for b in range(n):
                    if b == ii:
                        continue
                    dx = xi[a] - x[b]
                    dy = yi[a] - y[b]
                    d = np.sqrt(dx * dx + dy * dy)
                    ov = max(ri[a] + r[b] - d, 0.0)
                    obj += 500.0 * ov * ov
            # Boundary
            for a, ii in enumerate(idx):
                lv = max(ri[a] - xi[a], 0.0)
                rv = max(xi[a] + ri[a] - w, 0.0)
                bv = max(ri[a] - yi[a], 0.0)
                tv = max(yi[a] + ri[a] - h, 0.0)
                obj += 500.0 * (lv * lv + rv * rv + bv * bv + tv * tv)
            return obj

        bounds = [(0, 2), (0, 2), (0.001, 1.0)] * m
        try:
            res = minimize(obj_full, v0, method='SLSQP', bounds=bounds,
                           options={'maxiter': 300, 'ftol': 1e-14, 'maxls': 100})
            vo = res.x
            xi = vo[0::3][:m]
            yi = vo[1::3][:m]
            ri = vo[2::3][:m]
            # Write back and validate
            x_new = x.copy()
            y_new = y.copy()
            r_new = r.copy()
            x_new[idx] = xi
            y_new[idx] = yi
            r_new[idx] = ri
            if _is_valid(x_new, y_new, r_new, w, h):
                x, y, r = x_new, y_new, r_new
        except Exception:
            continue

    if not _is_valid(x, y, r, w, h):
        for factor in [0.999, 0.998, 0.995]:
            if _is_valid(x, y, r * factor, w, h):
                r = r * factor
                break
        if not _is_valid(x, y, r, w, h):
            return None

    z_new = np.empty(3 * N + 1)
    z_new[0::3][:n] = x
    z_new[1::3][:n] = y
    z_new[2::3][:n] = r
    z_new[-1] = w
    return z_new


def _polish_result(z, time_budget):
    """Polish a valid packing with force-directed position adjustment and radius growth."""
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    dt = 0.002
    max_iter = int(min(5000, time_budget * 200))

    for it in range(max_iter):
        if time.time() > time_budget:
            break

        fx = np.zeros(n)
        fy = np.zeros(n)

        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        fx -= left_viol * 40
        fx += right_viol * 40
        fy -= bottom_viol * 40
        fy += top_viol * 40

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]

        iu = np.triu_indices(n, k=1)
        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]

        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
        overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)

        if np.max(overlap) > 1e-12:
            f_pairs = overlap * 20.0
            dx_pairs = dx[iu]
            dy_pairs = dy[iu]
            ux = dx_pairs / d_pairs
            uy = dy_pairs / d_pairs

            for idx in range(len(iu[0])):
                i_idx = iu[0][idx]
                j_idx = iu[1][idx]
                fx[i_idx] += ux[idx] * f_pairs[idx]
                fy[i_idx] += uy[idx] * f_pairs[idx]
                fx[j_idx] -= ux[idx] * f_pairs[idx]
                fy[j_idx] -= uy[idx] * f_pairs[idx]

        max_force = max(float(np.max(np.abs(fx))), float(np.max(np.abs(fy))), 1e-8)
        if max_force > 0.1:
            dt = 0.0005
        elif max_force > 0.01:
            dt = 0.001
        else:
            dt = 0.002

        x_new = x + dt * fx
        y_new = y + dt * fy
        x_new = np.clip(x_new, r, w - r)
        y_new = np.clip(y_new, r, h - r)

        if _is_valid(x_new, y_new, r, w, h):
            x = x_new
            y = y_new

    # Uniform growth with binary search
    for _ in range(100):
        if time.time() > time_budget:
            break
        lo, hi = 1.0, 1.05
        for _ in range(15):
            mid = (lo + hi) / 2.0
            if _is_valid(x, y, r * mid, w, h):
                lo = mid
            else:
                hi = mid
        if lo > 1.0 + 1e-9:
            r = r * lo
        else:
            break

    z_new = np.empty(3 * N + 1)
    z_new[0::3][:n] = x
    z_new[1::3][:n] = y
    z_new[2::3][:n] = r
    z_new[-1] = w

    if _is_valid(x, y, r, w, h):
        return z_new
    return None


def _expansion_relaxation_polish(z, time_budget):
    """Iteratively expand circles and shift positions to create space."""
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    dirs = [
        (1, 0), (-1, 0), (0, 1), (0, -1),
        (0.7071, 0.7071), (-0.7071, 0.7071), (0.7071, -0.7071), (-0.7071, -0.7071)
    ]

    def try_expand_circle(i):
        r_wall = min(x[i], w - x[i], y[i], h - y[i])
        r_max = r_wall
        for j in range(n):
            if i == j:
                continue
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx * dx + dy * dy)
            r_allowed = d - r[j]
            if r_allowed < r_max:
                r_max = r_allowed
        r_max = max(0.0, r_max)

        if r_max > r[i] + 1e-6:
            lo = r[i]
            hi = r_max
            for _ in range(25):
                mid = (lo + hi) / 2.0
                r_test = r.copy()
                r_test[i] = mid
                if _is_valid(x, y, r_test, w, h):
                    lo = mid
                else:
                    hi = mid
            if lo > r[i] + 1e-7:
                r[i] = lo
                return True
        return False

    def try_move_circle(i):
        best_dir = None
        best_score = -float('inf')

        for dx_d, dy_d in dirs:
            new_x = x[i] + dx_d * 0.01
            new_y = y[i] + dy_d * 0.01
            if new_x < r[i] or new_x > w - r[i] or new_y < r[i] or new_y > h - r[i]:
                continue

            min_gap = float('inf')
            for j in range(n):
                if i == j:
                    continue
                dx = new_x - x[j]
                dy = new_y - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                gap = d - (r[i] + r[j])
                if gap < min_gap:
                    min_gap = gap

            wall_dist = min(new_x - r[i], w - new_x - r[i], new_y - r[i], h - new_y - r[i])
            score = min(min_gap, wall_dist)

            if score > best_score:
                best_score = score
                best_dir = (dx_d, dy_d)

        if best_dir is None:
            return False

        dx_d, dy_d = best_dir

        for step in [0.05, 0.03, 0.02, 0.01, 0.005, 0.003, 0.001]:
            new_x = x[i] + dx_d * step
            new_y = y[i] + dy_d * step
            if _is_valid_move(x, y, r, w, h, i, new_x, new_y):
                x[i] = new_x
                y[i] = new_y
                return True

        return False

    max_rounds = 40
    for rnd in range(max_rounds):
        if time.time() > time_budget:
            break

        improved = False
        for i in range(n):
            if time.time() > time_budget:
                break

            if try_expand_circle(i):
                improved = True

            r_wall = min(x[i], w - x[i], y[i], h - y[i])
            r_max = r_wall
            for j in range(n):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                r_allowed = d - r[j]
                if r_allowed < r_max:
                    r_max = r_allowed
            r_max = max(0.0, r_max)

            if r_max - r[i] < 0.01:
                if try_move_circle(i):
                    improved = True

        if not improved:
            break

    for rnd in range(20):
        if time.time() > time_budget:
            break
        improved = False
        for i in range(n):
            if time.time() > time_budget:
                break
            if try_expand_circle(i):
                improved = True
        if not improved:
            break

    for _ in range(100):
        if time.time() > time_budget:
            break
        lo, hi = 1.0, 1.05
        for _ in range(15):
            mid = (lo + hi) / 2.0
            if _is_valid(x, y, r * mid, w, h):
                lo = mid
            else:
                hi = mid
        if lo > 1.0 + 1e-9:
            r = r * lo
        else:
            break

    if _is_valid(x, y, r, w, h):
        z_new = np.empty(3 * N + 1)
        z_new[0::3][:n] = x
        z_new[1::3][:n] = y
        z_new[2::3][:n] = r
        z_new[-1] = w
        return z_new
    return None


def _is_valid_move(x, y, r, w, h, i, new_x, new_y):
    if new_x < r[i] or new_x > w - r[i]:
        return False
    if new_y < r[i] or new_y > h - r[i]:
        return False
    for j in range(N):
        if i == j:
            continue
        dx = new_x - x[j]
        dy = new_y - y[j]
        d2 = dx * dx + dy * dy
        r_sum = r[i] + r[j]
        if d2 < r_sum * r_sum - 1e-9:
            return False
    return True


def _constraint_refine(z, time_budget, tight=False):
    """High-precision SLSQP refinement using smooth penalty."""
    z = z.copy().astype(float)
    n = N

    penalty = 1000.0 if tight else 500.0
    ftol = 1e-18 if tight else 1e-15
    maxls = 200 if tight else 150

    def obj_refine(zv):
        x = zv[0::3][:n]
        y = zv[1::3][:n]
        r = zv[2::3][:n]
        w = zv[-1]
        h = 2.0 - w

        if w < 0.2 or h < 0.2 or np.any(r < 0):
            return 1e6

        obj = -np.sum(r)

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]

        iu = np.triu_indices(n, k=1)
        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]

        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
        overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)

        obj += penalty * np.sum(overlap * overlap)

        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)

        obj += penalty * (
            np.sum(left_viol * left_viol) +
            np.sum(right_viol * right_viol) +
            np.sum(bottom_viol * bottom_viol) +
            np.sum(top_viol * top_viol)
        )

        return obj

    bounds = []
    for i in range(n):
        bounds.extend([(0, 2), (0, 2), (0.001, 1.0)])
    bounds.append((0.3, 1.7))

    maxiter = int(min(2000 if tight else 1000, time_budget * 200))
    try:
        res = minimize(obj_refine, z, method='SLSQP',
                       bounds=bounds,
                       options={'maxiter': maxiter, 'ftol': ftol, 'maxls': maxls})

        z_opt = res.x
        x = z_opt[0::3][:n]
        y = z_opt[1::3][:n]
        r = z_opt[2::3][:n]
        w = z_opt[-1]
        h = 2.0 - w

        if not _is_valid(x, y, r, w, h):
            for factor in [0.999, 0.998, 0.997, 0.995, 0.99]:
                r_trial = r * factor
                if _is_valid(x, y, r_trial, w, h):
                    r = r_trial
                    break

        if _is_valid(x, y, r, w, h):
            z_opt[2::3][:n] = r
            return z_opt
        return None
    except Exception:
        return None


def _lse_epigraph_refine(z, time_budget, soft_len=0.01):
    """
    Log-sum-exp epigraph formulation: maximize sum(r_i) - beta * LSE of violations.
    This provides a smooth, differentiable surrogate for the non-convex constraint
    min_i (d_ij - r_i - r_j) >= 0, helping escape local minima that trap penalty methods.
    """
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    iu = np.triu_indices(n, k=1)
    n_pairs = len(iu[0])

    def lse_obj(zv):
        xi = zv[0::3][:n]
        yi = zv[1::3][:n]
        ri = zv[2::3][:n]
        wi = zv[-1]
        hi = 2.0 - wi

        if wi < 0.2 or hi < 0.2 or np.any(ri < 0):
            return 1e6

        # Pairwise violations: r_i + r_j - d_ij (positive = overlap)
        dx = xi[:, None] - xi[None, :]
        dy = yi[:, None] - yi[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = ri[:, None] + ri[None, :]
        d_pairs = np.sqrt(np.maximum(d_sq[iu], 1e-12))
        r_sum_pairs = r_sum[iu]
        pair_viol = r_sum_pairs - d_pairs  # positive when overlapping

        # Boundary violations
        bviol = np.concatenate([
            ri - xi,          # left
            xi + ri - wi,     # right
            ri - yi,          # bottom
            yi + ri - hi      # top
        ])

        # LSE of all violations (smooth max surrogate)
        all_viol = np.concatenate([pair_viol, bviol])
        # Clip for numerical stability
        all_viol = np.clip(all_viol, -50.0, 50.0)
        max_v = np.max(all_viol)
        lse_val = max_v + soft_len * np.log(np.sum(np.exp((all_viol - max_v) / soft_len)))

        # Objective: maximize sum(r) - penalty * LSE
        return -np.sum(ri) + 100.0 * lse_val

    bounds = []
    for i in range(n):
        bounds.extend([(0.001, 1.5), (0.001, 1.5), (0.001, 0.5)])
    bounds.append((0.3, 1.7))

    maxiter = int(min(800, time_budget * 150))
    try:
        res = minimize(lse_obj, z, method='SLSQP', bounds=bounds,
                       options={'maxiter': maxiter, 'ftol': 1e-14, 'maxls': 120})
        z_opt = res.x
        x_opt = z_opt[0::3][:n]
        y_opt = z_opt[1::3][:n]
        r_opt = z_opt[2::3][:n]
        w_opt = z_opt[-1]
        h_opt = 2.0 - w_opt

        if _is_valid(x_opt, y_opt, r_opt, w_opt, h_opt):
            return z_opt
        else:
            for factor in [0.999, 0.998, 0.997, 0.995]:
                if _is_valid(x_opt, y_opt, r_opt * factor, w_opt, h_opt):
                    z_opt[2::3][:n] = r_opt * factor
                    return z_opt
            return None
    except Exception:
        return None


def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    best_value = 0.0
    best_result = None

    penalty_weight = 1000.0

    def objective(z):
        return _vectorized_objective(z, penalty_weight)

    def check_validity(z, tol=1e-9):
        x = z[0::3][:N]
        y = z[1::3][:N]
        r = z[2::3][:N]
        w = z[-1]
        h = 2.0 - w

        if w <= 0.05 or h <= 0.05:
            return False, 0.0

        for i in range(N):
            if r[i] <= 0:
                return False, 0.0
            if x[i] < r[i] - tol or x[i] > w - r[i] + tol:
                return False, 0.0
            if y[i] < r[i] - tol or y[i] > h - r[i] + tol:
                return False, 0.0

        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d2 = dx * dx + dy * dy
                r_sum = r[i] + r[j]
                if d2 < r_sum ** 2 - tol:
                    return False, 0.0

        val = float(np.sum(r))
        return True, val

    def get_initial_guesses():
        guesses = []

        # Grid-based initializations with optimal aspect ratios for perimeter-4
        # For perimeter 4: w + h = 2. For N circles in a grid, optimal aspect
        # depends on grid shape. Try multiple grid shapes.
        grid_configs = [
            (5, 5), (4, 6), (6, 4), (3, 7), (7, 3), (5, 5), (4, 5), (5, 4)
        ]

        for rows, cols in grid_configs:
            n_cells = rows * cols
            if n_cells < N:
                continue
            # Optimal w for this grid: w/h = cols/rows, and w+h=2
            ratio = cols / rows
            w0 = 2.0 * ratio / (1.0 + ratio)
            h0 = 2.0 - w0
            if w0 < 0.3 or h0 < 0.3:
                continue

            z = np.zeros(3 * N + 1)
            for i in range(N):
                row, col = divmod(i, cols)
                if row >= rows:
                    break
                cx = (col + 0.5) / cols * w0
                cy = (row + 0.5) / rows * h0
                r0 = min(w0 / cols, h0 / rows) * 0.48
                z[3 * i] = cx
                z[3 * i + 1] = cy
                z[3 * i + 2] = r0
            z[-1] = w0
            guesses.append(z)

        # Also try square-ish with slight perturbations
        rng = np.random.default_rng(42)
        for trial in range(12):
            w0 = rng.uniform(0.6, 1.4)
            h0 = 2.0 - w0
            z = np.zeros(3 * N + 1)
            side = int(np.ceil(np.sqrt(N)))
            for i in range(N):
                row, col = divmod(i, side)
                cx = (col + 0.5) / side * w0 + rng.normal(0, 0.01)
                cy = (row + 0.5) / side * h0 + rng.normal(0, 0.01)
                r0 = min(w0 / side, h0 / side) * 0.42
                cx = np.clip(cx, r0, w0 - r0)
                cy = np.clip(cy, r0, h0 - r0)
                z[3 * i] = cx
                z[3 * i + 1] = cy
                z[3 * i + 2] = r0
            z[-1] = w0
            guesses.append(z)

        return guesses

    guesses = get_initial_guesses()

    for i, z0 in enumerate(guesses):
        if time.time() > deadline - 35:
            break

        z0_relaxed = _force_relax_unified(z0, max_iter=400)

        remaining = deadline - (time.time() - start_time)
        maxiter = int(min(500, max(120, remaining * 40)))
        try:
            res = minimize(objective, z0_relaxed, method='SLSQP',
                           bounds=[(0, 2)] * (3 * N) + [(0.3, 1.7)],
                           options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 80})

            z_opt = res.x
            valid, val = check_validity(z_opt)

            if valid and val > best_value:
                best_value = val
                best_result = z_opt.copy()
            elif not valid:
                z_repair = z_opt.copy()
                r_idx = np.arange(2, len(z_repair) - 1, 3)
                z_repair[r_idx] *= 0.95
                valid, val = check_validity(z_repair)
                if valid and val > best_value:
                    best_value = val
                    best_result = z_repair.copy()

        except Exception:
            continue

    # --- Phase 2: Polish on best result ---
    if best_result is not None and time.time() < deadline - 30:
        # 1. Force-directed position polish
        polished = _polish_result(best_result, deadline - (time.time() - start_time) - 24)
        if polished is not None:
            valid, val = check_validity(polished)
            if valid and val > best_value:
                best_value = val
                best_result = polished

        # 2. Pressure-field growth (smooth surrogate for min-clearance)
        if best_result is not None and time.time() < deadline - 24:
            grown = _pressure_field_growth(best_result, deadline - (time.time() - start_time) - 16)
            if grown is not None:
                valid, val = check_validity(grown)
                if valid and val > best_value:
                    best_value = val
                    best_result = grown

        # 3. Expansion/Relocation Polish
        if best_result is not None and time.time() < deadline - 16:
            expanded = _expansion_relaxation_polish(best_result, deadline - (time.time() - start_time) - 10)
            if expanded is not None:
                valid, val = check_validity(expanded)
                if valid and val > best_value:
                    best_value = val
                    best_result = expanded

        # 4. Contact-graph-aware local search
        if best_result is not None and time.time() < deadline - 10:
            cg = _contact_graph_refine(best_result, deadline - (time.time() - start_time) - 6)
            if cg is not None:
                valid, val = check_validity(cg)
                if valid and val > best_value:
                    best_value = val
                    best_result = cg

        # 5. LSE epigraph refinement (smooth surrogate for non-convex constraints)
        if best_result is not None and time.time() < deadline - 8:
            lse_ref = _lse_epigraph_refine(best_result, deadline - (time.time() - start_time) - 6, soft_len=0.01)
            if lse_ref is not None:
                valid, val = check_validity(lse_ref)
                if valid and val > best_value:
                    best_value = val
                    best_result = lse_ref

        # 5b. Second LSE pass with tighter soft_len
        if best_result is not None and time.time() < deadline - 6:
            lse_ref2 = _lse_epigraph_refine(best_result, deadline - (time.time() - start_time) - 4, soft_len=0.005)
            if lse_ref2 is not None:
                valid, val = check_validity(lse_ref2)
                if valid and val > best_value:
                    best_value = val
                    best_result = lse_ref2

        # 6. Constraint-based SLSQP refinement
        if best_result is not None and time.time() < deadline - 4:
            refined = _constraint_refine(best_result, deadline - (time.time() - start_time) - 2)
            if refined is not None:
                valid, val = check_validity(refined)
                if valid and val > best_value:
                    best_value = val
                    best_result = refined

        # 7. Final high-precision SLSQP
        if best_result is not None and time.time() < deadline - 1.5:
            final_refined = _constraint_refine(best_result, deadline - (time.time() - start_time) - 0.5, tight=True)
            if final_refined is not None:
                valid, val = check_validity(final_refined)
                if valid and val > best_value:
                    best_value = val
                    best_result = final_refined

    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    x = best_result[0::3][:N]
    y = best_result[1::3][:N]
    r = best_result[2::3][:N]
    w = best_result[-1]
    h = 2.0 - w

    for _ in range(20):
        r_trial = r * (1 - 1e-8)
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    return np.column_stack((x, y, r))
