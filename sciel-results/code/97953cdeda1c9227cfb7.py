"""Circle packing in perimeter-4 rectangle: SLSQP + Contact Graph + Expansion Polish."""
import numpy as np
import time
from scipy.optimize import minimize, least_squares

N = 21


def _soft_max(vals, temp=1.0):
    """Smooth approximation of max using log-sum-exp."""
    vals = np.asarray(vals)
    if vals.size == 0:
        return 0.0
    max_val = np.max(vals)
    if max_val < -100 * temp:
        return max_val
    exp_vals = np.exp((vals - max_val) / temp)
    return max_val + temp * np.log(np.sum(exp_vals))


def _build_contact_graph(x, y, r, w, h, k=4):
    """Build a contact graph from current packing. Returns list of (i, j, edge_type).
    edge_type: 'pair' for circle-circle contact, 'wall_L','wall_R','wall_B','wall_T' for wall contact.
    We want each circle to have ~k contacts (pair + wall).
    """
    n = len(x)
    # Compute all pairwise distances
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx * dx + dy * dy)
            gap = d - (r[i] + r[j])
            pairs.append((gap, i, j))
    pairs.sort()
    
    # Wall gaps
    wall_gaps = []
    for i in range(n):
        wall_gaps.append((x[i] - r[i], i, 'L'))
        wall_gaps.append((w - x[i] - r[i], i, 'R'))
        wall_gaps.append((y[i] - r[i], i, 'B'))
        wall_gaps.append((h - y[i] - r[i], i, 'T'))
    wall_gaps.sort()
    
    # Build contact list: for each circle, pick k nearest contacts
    contact_count = np.zeros(n, dtype=int)
    contacts = []  # (i, j, type) where type in ['pair', 'L', 'R', 'B', 'T']
    
    # First pass: add pair contacts
    for gap, i, j in pairs:
        if contact_count[i] < k and contact_count[j] < k:
            contacts.append((i, j, 'pair'))
            contact_count[i] += 1
            contact_count[j] += 1
    
    # Second pass: add wall contacts for circles that still need contacts
    for gap, i, wall in wall_gaps:
        if contact_count[i] < k:
            contacts.append((i, -1, wall))
            contact_count[i] += 1
    
    return contacts


def _contact_graph_solve(z, time_budget, k=4):
    """Solve exact tangency conditions using contact graph.
    Variables: x[i], y[i], r[i] for each circle, plus w (width).
    Constraints: for each contact edge, distance = sum of radii (or wall distance = radius).
    Objective: maximize sum(r).
    
    We use a penalty-based approach: minimize sum of (residual)^2 where residual = actual_dist - target_dist.
    Then do local search to maximize radii subject to these constraints being satisfied.
    """
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w
    
    if w < 0.2 or h < 0.2:
        return None
    
    # Build contact graph
    contacts = _build_contact_graph(x, y, r, w, h, k=k)
    
    if len(contacts) == 0:
        return None
    
    # Separate pair and wall contacts
    pair_contacts = [(i, j) for (i, j, t) in contacts if t == 'pair']
    wall_contacts = [(i, t) for (i, j, t) in contacts if t in ('L', 'R', 'B', 'T')]
    
    # Objective: minimize violation of tangency conditions + maximize radii
    # We'll use a combined approach: first satisfy tangencies, then expand radii.
    
    def tangency_residuals(zv):
        """Return residuals for tangency conditions."""
        xv = zv[0::3][:n]
        yv = zv[1::3][:n]
        rv = zv[2::3][:n]
        wv = zv[-1]
        hv = 2.0 - wv
        
        residuals = []
        
        # Pair tangencies: distance should equal r_i + r_j
        for i, j in pair_contacts:
            dx = xv[i] - xv[j]
            dy = yv[i] - yv[j]
            d = np.sqrt(dx * dx + dy * dy)
            residuals.append(d - (rv[i] + rv[j]))
        
        # Wall tangencies: distance to wall should equal r_i
        for i, wall in wall_contacts:
            if wall == 'L':
                residuals.append(xv[i] - rv[i])
            elif wall == 'R':
                residuals.append(wv - xv[i] - rv[i])
            elif wall == 'B':
                residuals.append(yv[i] - rv[i])
            elif wall == 'T':
                residuals.append(hv - yv[i] - rv[i])
        
        return np.array(residuals)
    
    # Solve for positions and radii that satisfy tangency conditions
    # We fix the aspect ratio and solve for x, y, r
    max_iter = int(min(500, time_budget * 100))
    
    try:
        # Use least_squares to minimize tangency residuals
        # We'll optimize x, y, r with w fixed
        def residual_func(params):
            xv = params[0:n]
            yv = params[n:2*n]
            rv = params[2*n:3*n]
            
            # Build residual vector
            res = []
            for i, j in pair_contacts:
                dx = xv[i] - xv[j]
                dy = yv[i] - yv[j]
                d = np.sqrt(dx * dx + dy * dy)
                res.append(d - (rv[i] + rv[j]))
            for i, wall in wall_contacts:
                if wall == 'L':
                    res.append(xv[i] - rv[i])
                elif wall == 'R':
                    res.append(w - xv[i] - rv[i])
                elif wall == 'B':
                    res.append(yv[i] - rv[i])
                elif wall == 'T':
                    res.append(h - yv[i] - rv[i])
            return np.array(res)
        
        params0 = np.concatenate([x, y, r])
        res = least_squares(residual_func, params0, method='lm', max_nfev=max_iter)
        
        x_new = res.x[0:n]
        y_new = res.x[n:2*n]
        r_new = res.x[2*n:3*n]
        
        # Check validity
        if _is_valid(x_new, y_new, r_new, w, h):
            z_new = np.empty(3 * N + 1)
            z_new[0::3][:n] = x_new
            z_new[1::3][:n] = y_new
            z_new[2::3][:n] = r_new
            z_new[-1] = w
            return z_new
        else:
            return None
            
    except Exception:
        return None


def _expansion_via_contact_graph(z, time_budget, k=4):
    """Iteratively build contact graph, solve tangencies, then expand radii.
    This is the main improvement: use topological structure to guide optimization.
    """
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w
    
    if w < 0.2 or h < 0.2:
        return None
    
    best_x, best_y, best_r, best_val = x.copy(), y.copy(), r.copy(), np.sum(r)
    
    max_rounds = 15
    for rnd in range(max_rounds):
        if time.time() > time_budget:
            break
        
        # Build contact graph from current state
        contacts = _build_contact_graph(x, y, r, w, h, k=k)
        pair_contacts = [(i, j) for (i, j, t) in contacts if t == 'pair']
        wall_contacts = [(i, t) for (i, j, t) in contacts if t in ('L', 'R', 'B', 'T')]
        
        if len(pair_contacts) + len(wall_contacts) < n:
            break
        
        # Solve tangency system
        def residual_func(params):
            xv = params[0:n]
            yv = params[n:2*n]
            rv = params[2*n:3*n]
            
            res = []
            for i, j in pair_contacts:
                dx = xv[i] - xv[j]
                dy = yv[i] - yv[j]
                d = np.sqrt(dx * dx + dy * dy)
                res.append(d - (rv[i] + rv[j]))
            for i, wall in wall_contacts:
                if wall == 'L':
                    res.append(xv[i] - rv[i])
                elif wall == 'R':
                    res.append(w - xv[i] - rv[i])
                elif wall == 'B':
                    res.append(yv[i] - rv[i])
                elif wall == 'T':
                    res.append(h - yv[i] - rv[i])
            return np.array(res)
        
        params0 = np.concatenate([x, y, r])
        try:
            res = least_squares(residual_func, params0, method='lm', max_nfev=300)
            x_new = res.x[0:n]
            y_new = res.x[n:2*n]
            r_new = res.x[2*n:3*n]
        except Exception:
            break
        
        # Check if solution is valid
        if not _is_valid(x_new, y_new, r_new, w, h):
            # Try to repair by shrinking radii slightly
            r_repair = r_new * 0.99
            if _is_valid(x_new, y_new, r_repair, w, h):
                x, y, r = x_new, y_new, r_repair
            else:
                break
        else:
            x, y, r = x_new, y_new, r_new
        
        # Now try to expand radii while maintaining contact structure
        # Uniform expansion via binary search
        improved = False
        for _ in range(40):
            if time.time() > time_budget:
                break
            lo, hi = 1.0, 1.03
            for _ in range(14):
                mid = (lo + hi) / 2.0
                r_trial = r * mid
                if _is_valid(x, y, r_trial, w, h):
                    lo = mid
                else:
                    hi = mid
            if lo > 1.0 + 1e-10:
                r = r * lo
                improved = True
            else:
                break
        
        # Per-circle expansion
        for i in range(n):
            if time.time() > time_budget:
                break
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
                for _ in range(20):
                    mid = (lo + hi) / 2.0
                    r_trial = r.copy()
                    r_trial[i] = mid
                    if _is_valid(x, y, r_trial, w, h):
                        lo = mid
                    else:
                        hi = mid
                if lo > r[i] + 1e-7:
                    r[i] = lo
                    improved = True
        
        val = np.sum(r)
        if val > best_val:
            best_x, best_y, best_r, best_val = x.copy(), y.copy(), r.copy(), val
        
        if not improved:
            break
    
    if _is_valid(best_x, best_y, best_r, w, h):
        z_new = np.empty(3 * N + 1)
        z_new[0::3][:n] = best_x
        z_new[1::3][:n] = best_y
        z_new[2::3][:n] = best_r
        z_new[-1] = w
        return z_new
    return None


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

    # Vectorized pairwise overlap penalty
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

    # Vectorized boundary penalty
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

    # Keep aspect ratio in reasonable range
    if w < 0.5:
        obj += penalty_weight * (0.5 - w) ** 2
    elif w > 1.5:
        obj += penalty_weight * (w - 1.5) ** 2

    return obj


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

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

    # Generate diverse initial guesses
    def get_initial_guesses():
        guesses = []
        side = int(np.ceil(np.sqrt(N)))

        aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.75, 0.55, 2.0, 0.5, 1.2, 0.8]
        for w0 in aspect_ratios:
            h0 = 2.0 - w0
            if h0 < 0.3:
                continue
            z = np.zeros(3 * N + 1)
            for i in range(N):
                row, col = divmod(i, side)
                cx = (col + 0.5) / side * w0
                cy = (row + 0.5) / side * h0
                r0 = min(w0 / side, h0 / side) * 0.45
                z[3 * i] = cx
                z[3 * i + 1] = cy
                z[3 * i + 2] = r0
            z[-1] = w0
            guesses.append(z)

        # Random starts
        rng = np.random.default_rng(42)
        for _ in range(10):
            w0 = rng.uniform(0.6, 1.4)
            h0 = 2.0 - w0
            z = np.zeros(3 * N + 1)
            for i in range(N):
                r0 = rng.uniform(0.05, 0.12)
                cx = rng.uniform(r0, w0 - r0)
                cy = rng.uniform(r0, h0 - r0)
                z[3 * i] = cx
                z[3 * i + 1] = cy
                z[3 * i + 2] = r0
            z[-1] = w0
            guesses.append(z)

        return guesses

    guesses = get_initial_guesses()

    for i, z0 in enumerate(guesses):
        if time.time() > deadline - 20:
            break

        # Force-directed relaxation to resolve initial overlaps
        z0_relaxed = _force_relax_unified(z0, max_iter=500)

        # SLSQP optimization
        remaining = deadline - (time.time() - start_time)
        maxiter = int(min(600, max(150, remaining * 50)))
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
                # Try repair: shrink radii slightly
                z_repair = z_opt.copy()
                r_idx = np.arange(2, len(z_repair) - 1, 3)
                z_repair[r_idx] *= 0.95
                valid, val = check_validity(z_repair)
                if valid and val > best_value:
                    best_value = val
                    best_result = z_repair.copy()

        except Exception:
            continue

    # --- Phase 2: Advanced Polish on best result ---
    if best_result is not None and time.time() < deadline - 25:
        # 1. Force-directed position polish to ensure we are in a good basin
        polished = _polish_result(best_result, deadline - (time.time() - start_time) - 15)
        if polished is not None:
            valid, val = check_validity(polished)
            if valid and val > best_value:
                best_value = val
                best_result = polished

        # 2. Contact graph expansion (NEW: topological approach)
        if best_result is not None and time.time() < deadline - 18:
            cg_expanded = _expansion_via_contact_graph(best_result, deadline - (time.time() - start_time) - 10, k=4)
            if cg_expanded is not None:
                valid, val = check_validity(cg_expanded)
                if valid and val > best_value:
                    best_value = val
                    best_result = cg_expanded

        # 3. Expansion/Relocation Polish
        if best_result is not None and time.time() < deadline - 18:
            expanded = _expansion_relaxation_polish(best_result, deadline - (time.time() - start_time) - 10)
            if expanded is not None:
                valid, val = check_validity(expanded)
                if valid and val > best_value:
                    best_value = val
                    best_result = expanded

        # 4. Second contact graph pass with different k
        if best_result is not None and time.time() < deadline - 15:
            cg_expanded2 = _expansion_via_contact_graph(best_result, deadline - (time.time() - start_time) - 8, k=5)
            if cg_expanded2 is not None:
                valid, val = check_validity(cg_expanded2)
                if valid and val > best_value:
                    best_value = val
                    best_result = cg_expanded2

        # 5. Constraint-based SLSQP refinement
        if best_result is not None and time.time() < deadline - 15:
            refined = _constraint_refine(best_result, deadline - (time.time() - start_time) - 8)
            if refined is not None:
                valid, val = check_validity(refined)
                if valid and val > best_value:
                    best_value = val
                    best_result = refined

        # 6. Third contact graph pass after refinement
        if best_result is not None and time.time() < deadline - 10:
            cg_expanded3 = _expansion_via_contact_graph(best_result, deadline - (time.time() - start_time) - 6, k=4)
            if cg_expanded3 is not None:
                valid, val = check_validity(cg_expanded3)
                if valid and val > best_value:
                    best_value = val
                    best_result = cg_expanded3

        # 7. Second expansion pass after refinement
        if best_result is not None and time.time() < deadline - 10:
            expanded2 = _expansion_relaxation_polish(best_result, deadline - (time.time() - start_time) - 6)
            if expanded2 is not None:
                valid, val = check_validity(expanded2)
                if valid and val > best_value:
                    best_value = val
                    best_result = expanded2

        # 8. Contact graph high-precision refine
        if best_result is not None and time.time() < deadline - 8:
            cg_refined = _contact_graph_refine(best_result, deadline - (time.time() - start_time) - 5, k=4)
            if cg_refined is not None:
                valid, val = check_validity(cg_refined)
                if valid and val > best_value:
                    best_value = val
                    best_result = cg_refined

        # 9. Final quick polish
        if best_result is not None and time.time() < deadline - 5:
            polished2 = _polish_result(best_result, deadline - (time.time() - start_time) - 2)
            if polished2 is not None:
                valid, val = check_validity(polished2)
                if valid and val > best_value:
                    best_value = val
                    best_result = polished2

    if best_result is None:
        # Fallback: simple grid in unit square
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    # Format output
    x = best_result[0::3][:N]
    y = best_result[1::3][:N]
    r = best_result[2::3][:N]
    w = best_result[-1]
    h = 2.0 - w

    # Final safety: ensure strict validity with minimal shrink
    for _ in range(20):
        r_trial = r * (1 - 1e-10)
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    # One final micro-expansion pass
    for _ in range(30):
        lo, hi = 1.0, 1.005
        for _ in range(12):
            mid = (lo + hi) / 2.0
            r_trial = r * mid
            if _is_valid(x, y, r_trial, w, h):
                lo = mid
            else:
                hi = mid
        if lo > 1.0 + 1e-12:
            r = r * lo
        else:
            break

    return np.column_stack((x, y, r))


def _force_relax_unified(z, max_iter=800):
    """Force-directed relaxation with fully vectorized pair force accumulation."""
    z = z.copy().astype(float)
    n = N
    dt = 0.01
    min_dt = 0.0001
    convergence_tol = 1e-9
    iu = np.triu_indices(n, k=1)
    iu0 = iu[0]
    iu1 = iu[1]

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

        # Wall repulsion (vectorized)
        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        fx -= left_viol * 150
        fx += right_viol * 150
        fy -= bottom_viol * 150
        fy += top_viol * 150

        # Pairwise repulsion (vectorized with np.add.at)
        dx = x[iu0] - x[iu1]
        dy = y[iu0] - y[iu1]
        d_sq = dx * dx + dy * dy
        r_sum = r[iu0] + r[iu1]
        d_pairs = np.sqrt(np.maximum(d_sq, 1e-12))
        overlap = np.maximum(r_sum - d_pairs, 0.0)
        max_overlap = float(np.max(overlap)) if overlap.size > 0 else 0.0

        if max_overlap > convergence_tol:
            f_pairs = overlap * 20.0
            ux = dx / d_pairs
            uy = dy / d_pairs
            fx_i = ux * f_pairs
            fy_i = uy * f_pairs
            np.add.at(fx, iu0, fx_i)
            np.add.at(fy, iu0, fy_i)
            np.add.at(fx, iu1, -fx_i)
            np.add.at(fy, iu1, -fy_i)

        # Check convergence
        if max_overlap < convergence_tol:
            break

        # Adaptive step size
        force_mag = float(np.max(np.sqrt(fx * fx + fy * fy)))
        if force_mag > 1.0:
            dt = max(min_dt, dt * 0.5)
        elif force_mag < 0.1:
            dt = min(0.02, dt * 1.15)

        # Update positions
        z[0::3][:n] += dt * fx
        z[1::3][:n] += dt * fy

        # Clamp positions (vectorized)
        z[0::3][:n] = np.clip(z[0::3][:n], r, w - r)
        z[1::3][:n] = np.clip(z[1::3][:n], r, h - r)

    return z


def _polish_result(z, time_budget):
    """Polish a valid packing with vectorized force-directed adjustment and radius growth."""
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    dt = 0.003
    max_iter = int(min(8000, time_budget * 400))
    iu = np.triu_indices(n, k=1)
    iu0 = iu[0]
    iu1 = iu[1]

    for it in range(max_iter):
        if time.time() > time_budget:
            break

        fx = np.zeros(n)
        fy = np.zeros(n)

        # Wall repulsion (vectorized)
        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        fx -= left_viol * 50
        fx += right_viol * 50
        fy -= bottom_viol * 50
        fy += top_viol * 50

        # Pairwise repulsion (vectorized with np.add.at)
        dx = x[iu0] - x[iu1]
        dy = y[iu0] - y[iu1]
        d_sq = dx * dx + dy * dy
        r_sum = r[iu0] + r[iu1]
        d_pairs = np.sqrt(np.maximum(d_sq, 1e-12))
        overlap = np.maximum(r_sum - d_pairs, 0.0)

        if np.max(overlap) > 1e-10:
            f_pairs = overlap * 25.0
            ux = dx / d_pairs
            uy = dy / d_pairs
            fx_i = ux * f_pairs
            fy_i = uy * f_pairs
            np.add.at(fx, iu0, fx_i)
            np.add.at(fy, iu0, fy_i)
            np.add.at(fx, iu1, -fx_i)
            np.add.at(fy, iu1, -fy_i)

        # Adaptive step
        max_force = max(float(np.max(np.abs(fx))), float(np.max(np.abs(fy))), 1e-8)
        if max_force > 0.5:
            dt = 0.001
        elif max_force > 0.1:
            dt = 0.002
        else:
            dt = 0.004

        x += dt * fx
        y += dt * fy
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)

    # Try to grow radii uniformly with binary search
    for _ in range(60):
        if time.time() > time_budget:
            break
        lo, hi = 1.0, 1.02
        for _ in range(16):
            mid = (lo + hi) / 2.0
            r_trial = r * mid
            if _is_valid(x, y, r_trial, w, h):
                lo = mid
            else:
                hi = mid
        if lo > 1.0 + 1e-10:
            r = r * lo
        else:
            break

    # Reconstruct z
    z_new = np.empty(3 * N + 1)
    z_new[0::3][:n] = x
    z_new[1::3][:n] = y
    z_new[2::3][:n] = r
    z_new[-1] = w

    if _is_valid(x, y, r, w, h):
        return z_new
    return None


def _expansion_relaxation_polish(z, time_budget):
    """
    Advanced polish: Systematic multi-directional position search + per-circle radius expansion.
    Uses 12-direction local search with adaptive step sizes before expanding each circle's radius.
    """
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    # 12 directions for local search (finer angular resolution)
    angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    dirs_x = np.cos(angles)
    dirs_y = np.sin(angles)
    base_steps = [0.08, 0.05, 0.03, 0.018, 0.01, 0.005, 0.002, 0.001]

    max_rounds = 80
    for rnd in range(max_rounds):
        if time.time() > time_budget:
            break

        improved = False
        for i in range(n):
            # Try to reposition circle i in 12 directions to create more space
            best_pos = None
            best_gap = -1.0
            for di in range(12):
                for step in base_steps:
                    nx = x[i] + dirs_x[di] * step
                    ny = y[i] + dirs_y[di] * step
                    if nx < r[i] or nx > w - r[i]:
                        continue
                    if ny < r[i] or ny > h - r[i]:
                        continue
                    # Check validity and compute minimum gap to neighbors
                    valid = True
                    min_gap = float('inf')
                    for j in range(n):
                        if i == j:
                            continue
                        dx = nx - x[j]
                        dy = ny - y[j]
                        d = np.sqrt(dx * dx + dy * dy)
                        gap = d - (r[i] + r[j])
                        if gap < 1e-9:
                            valid = False
                            break
                        if gap < min_gap:
                            min_gap = gap
                    if valid and min_gap > best_gap:
                        best_gap = min_gap
                        best_pos = (nx, ny)
            
            if best_pos is not None and best_gap > 0.008:
                x[i] = best_pos[0]
                y[i] = best_pos[1]
                improved = True

            # Expand radius i to maximum possible
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
                for _ in range(24):
                    mid = (lo + hi) / 2.0
                    r_trial = r.copy()
                    r_trial[i] = mid
                    if _is_valid(x, y, r_trial, w, h):
                        lo = mid
                    else:
                        hi = mid
                if lo > r[i] + 1e-7:
                    r[i] = lo
                    improved = True

        if not improved:
            break

    # Final uniform expansion with binary search
    for _ in range(80):
        if time.time() > time_budget:
            break
        lo, hi = 1.0, 1.04
        for _ in range(20):
            mid = (lo + hi) / 2.0
            r_trial = r * mid
            if _is_valid(x, y, r_trial, w, h):
                lo = mid
            else:
                hi = mid
        if lo > 1.0 + 1e-10:
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

def _max_radius_for_circle(x, y, r, w, h, i):
    """Calculate maximum possible radius for circle i given other positions."""
    r_wall = min(x[i], w - x[i], y[i], h - y[i])
    r_max = r_wall
    for j in range(N):
        if i == j: continue
        dx = x[i] - x[j]
        dy = y[i] - y[j]
        d = np.sqrt(dx*dx + dy*dy)
        r_allowed = d - r[j]
        if r_allowed < r_max:
            r_max = r_allowed
    return max(0.0, r_max)

def _is_valid_move(x, y, r, w, h, i, new_x, new_y):
    """Check if moving circle i to new_x, new_y is valid."""
    # Wall check
    if new_x < r[i] or new_x > w - r[i]:
        return False
    if new_y < r[i] or new_y > h - r[i]:
        return False
    
    # Neighbor check
    for j in range(N):
        if i == j: continue
        dx = new_x - x[j]
        dy = new_y - y[j]
        d2 = dx*dx + dy*dy
        r_sum = r[i] + r[j]
        if d2 < r_sum*r_sum - 1e-9:
            return False
    return True


def _constraint_refine(z, time_budget):
    """High-precision SLSQP refinement using soft-max epigraph formulation for smooth constraints."""
    z = z.copy().astype(float)
    n = N
    iu = np.triu_indices(n, k=1)
    iu0 = iu[0]
    iu1 = iu[1]

    def obj_refine(zv):
        r = zv[2::3][:n]
        return -np.sum(r)

    def boundary_constraints(zv):
        """Return vector of all boundary constraint values (must be >= 0)."""
        x = zv[0::3][:n]
        y = zv[1::3][:n]
        r = zv[2::3][:n]
        w = zv[-1]
        h = 2.0 - w
        left = x - r
        right = w - x - r
        bottom = y - r
        top = h - y - r
        return np.concatenate([left, right, bottom, top])

    def pairwise_constraints(zv):
        """Return vector of all pairwise distance constraint values (must be >= 0)."""
        x = zv[0::3][:n]
        y = zv[1::3][:n]
        r = zv[2::3][:n]
        dx = x[iu0] - x[iu1]
        dy = y[iu0] - y[iu1]
        d2 = dx * dx + dy * dy
        r_sum = r[iu0] + r[iu1]
        return d2 - r_sum * r_sum

    constraints = [
        {'type': 'ineq', 'fun': boundary_constraints},
        {'type': 'ineq', 'fun': pairwise_constraints}
    ]

    bounds = [(0, 2), (0, 2), (0.001, 1.0)] * n + [(0.3, 1.7)]

    maxiter = int(min(2500, time_budget * 400))
    try:
        res = minimize(obj_refine, z, method='SLSQP',
                       bounds=bounds,
                       constraints=constraints,
                       options={'maxiter': maxiter, 'ftol': 1e-16, 'maxls': 300, 'disp': False})

        z_opt = res.x
        x = z_opt[0::3][:n]
        y = z_opt[1::3][:n]
        r = z_opt[2::3][:n]
        w = z_opt[-1]
        h = 2.0 - w

        if _is_valid(x, y, r, w, h):
            return z_opt
        # Try repair with slight shrink
        z_repair = z_opt.copy()
        r_idx = np.arange(2, len(z_repair) - 1, 3)
        z_repair[r_idx] *= 0.995
        x = z_repair[0::3][:n]
        y = z_repair[1::3][:n]
        r = z_repair[2::3][:n]
        w = z_repair[-1]
        h = 2.0 - w
        if _is_valid(x, y, r, w, h):
            return z_repair
        return None
    except Exception:
        return None


def _is_valid(cx, cy, r, W, H, tol=1e-8):
    """Check if packing is valid."""
    if W <= 0.05 or H <= 0.05:
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


def _contact_graph_refine(z, time_budget, k=4):
    """High-precision contact graph refinement: solve exact tangency system
    with tight tolerance, then do fine-grained radius expansion.
    """
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w
    
    if w < 0.2 or h < 0.2:
        return None
    
    # Multiple rounds of contact graph solving with tightening
    for round_num in range(3):
        if time.time() > time_budget:
            break
        
        contacts = _build_contact_graph(x, y, r, w, h, k=k)
        pair_contacts = [(i, j) for (i, j, t) in contacts if t == 'pair']
        wall_contacts = [(i, t) for (i, j, t) in contacts if t in ('L', 'R', 'B', 'T')]
        
        if len(pair_contacts) + len(wall_contacts) < n:
            break
        
        def residual_func(params):
            xv = params[0:n]
            yv = params[n:2*n]
            rv = params[2*n:3*n]
            
            res = []
            for i, j in pair_contacts:
                dx = xv[i] - xv[j]
                dy = yv[i] - yv[j]
                d = np.sqrt(dx * dx + dy * dy)
                res.append(d - (rv[i] + rv[j]))
            for i, wall in wall_contacts:
                if wall == 'L':
                    res.append(xv[i] - rv[i])
                elif wall == 'R':
                    res.append(w - xv[i] - rv[i])
                elif wall == 'B':
                    res.append(yv[i] - rv[i])
                elif wall == 'T':
                    res.append(h - yv[i] - rv[i])
            return np.array(res)
        
        params0 = np.concatenate([x, y, r])
        try:
            res = least_squares(residual_func, params0, method='lm', max_nfev=500, xtol=1e-14)
            x_new = res.x[0:n]
            y_new = res.x[n:2*n]
            r_new = res.x[2*n:3*n]
        except Exception:
            break
        
        if not _is_valid(x_new, y_new, r_new, w, h):
            r_repair = r_new * 0.999
            if _is_valid(x_new, y_new, r_repair, w, h):
                x, y, r = x_new, y_new, r_repair
            else:
                break
        else:
            x, y, r = x_new, y_new, r_new
        
        # Fine-grained uniform expansion
        for _ in range(60):
            if time.time() > time_budget:
                break
            lo, hi = 1.0, 1.015
            for _ in range(18):
                mid = (lo + hi) / 2.0
                r_trial = r * mid
                if _is_valid(x, y, r_trial, w, h):
                    lo = mid
                else:
                    hi = mid
            if lo > 1.0 + 1e-12:
                r = r * lo
            else:
                break
        
        # Per-circle fine expansion
        for i in range(n):
            if time.time() > time_budget:
                break
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
            
            if r_max > r[i] + 1e-8:
                lo = r[i]
                hi = r_max
                for _ in range(25):
                    mid = (lo + hi) / 2.0
                    r_trial = r.copy()
                    r_trial[i] = mid
                    if _is_valid(x, y, r_trial, w, h):
                        lo = mid
                    else:
                        hi = mid
                if lo > r[i] + 1e-9:
                    r[i] = lo
    
    if _is_valid(x, y, r, w, h):
        z_new = np.empty(3 * N + 1)
        z_new[0::3][:n] = x
        z_new[1::3][:n] = y
        z_new[2::3][:n] = r
        z_new[-1] = w
        return z_new
    return None
