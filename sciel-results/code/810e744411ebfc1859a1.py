"""Circle packing in perimeter-4 rectangle: SLSQP + Advanced Expansion Polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21
_TIME_LIMIT = 145.0  # Safety margin before 150s kill


def _vectorized_objective(z, penalty_weight=1000.0, beta=80.0):
    """Vectorized objective with smooth penalties for speed."""
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]
    w = z[-1]
    h = 2.0 - w

    if w < 0.2 or h < 0.2 or np.any(r < 0):
        return 1e6

    obj = -np.sum(r)

    # Vectorized pairwise overlap penalty using smooth approximation
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]

    iu = np.triu_indices(N, k=1)
    d_sq_pairs = d_sq[iu]
    r_sum_pairs = r_sum[iu]

    d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
    # Use smooth soft-min for overlap: log-sum-exp trick for max(a,0)
    overlap_raw = r_sum_pairs - d_pairs
    # Smooth approximation of max(overlap, 0) using softplus-like function
    overlap = np.logaddexp(0.0, beta * overlap_raw) / beta - np.log(2.0) / beta
    # Clip to avoid numerical issues
    overlap = np.maximum(overlap, 0.0)
    overlap_penalty = penalty_weight * np.sum(overlap * overlap)

    obj += overlap_penalty

    # Vectorized boundary penalty (smooth)
    left_viol_raw = r - x
    right_viol_raw = x + r - w
    bottom_viol_raw = r - y
    top_viol_raw = y + r - h

    beta_b = 60.0
    left_viol = np.logaddexp(0.0, beta_b * left_viol_raw) / beta_b - np.log(2.0) / beta_b
    right_viol = np.logaddexp(0.0, beta_b * right_viol_raw) / beta_b - np.log(2.0) / beta_b
    bottom_viol = np.logaddexp(0.0, beta_b * bottom_viol_raw) / beta_b - np.log(2.0) / beta_b
    top_viol = np.logaddexp(0.0, beta_b * top_viol_raw) / beta_b - np.log(2.0) / beta_b

    boundary_penalty = penalty_weight * (
        np.sum(left_viol * left_viol) +
        np.sum(right_viol * right_viol) +
        np.sum(bottom_viol * bottom_viol) +
        np.sum(top_viol * top_viol)
    )

    obj += boundary_penalty

    # Keep aspect ratio in reasonable range (smooth)
    if w < 0.5:
        obj += penalty_weight * (0.5 - w) ** 2
    elif w > 1.5:
        obj += penalty_weight * (w - 1.5) ** 2

    return obj


def _is_valid(x, y, r, w, h, tol=1e-9):
    """Check validity of circles in a w x h rectangle."""
    if w <= 0.05 or h <= 0.05:
        return False
    for i in range(len(r)):
        if r[i] <= 0:
            return False
        if x[i] < r[i] - tol or x[i] > w - r[i] + tol:
            return False
        if y[i] < r[i] - tol or y[i] > h - r[i] + tol:
            return False
    for i in range(len(r)):
        for j in range(i + 1, len(r)):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d2 = dx * dx + dy * dy
            r_sum = r[i] + r[j]
            if d2 < r_sum * r_sum - tol:
                return False
    return True


def _force_relax_unified(z, max_iter=500, max_time=30.0, start_time=None):
    """Force-directed relaxation in unified parameter space with adaptive step (vectorized)."""
    if start_time is None:
        start_time = time.time()
    
    z = z.copy().astype(float)
    n = N
    dt = 0.005
    min_dt = 0.0001
    convergence_tol = 1e-10

    for it in range(max_iter):
        if time.time() - start_time > max_time:
            break
        
        x = z[0::3][:n]
        y = z[1::3][:n]
        r = z[2::3][:n]
        w = z[-1]
        h = 2.0 - w

        if w < 0.1 or h < 0.1:
            break

        fx = np.zeros(n)
        fy = np.zeros(n)
        fr = np.zeros(n)
        fw = 0.0

        # Pairwise forces
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx*dx + dy*dy) + 1e-10
                min_d = r[i] + r[j]
                if d < min_d:
                    force = (min_d - d) / d
                    fx[i] += force * dx
                    fy[i] += force * dy
                    fx[j] -= force * dx
                    fy[j] -= force * dy
                    # Slightly shrink radii when overlapping
                    fr[i] -= force * 0.1
                    fr[j] -= force * 0.1

        # Boundary forces
        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)

        fx -= 10.0 * left_viol
        fx += 10.0 * right_viol
        fy -= 10.0 * bottom_viol
        fy += 10.0 * top_viol

        # Update
        z[0::3][:n] += dt * fx
        z[1::3][:n] += dt * fy
        z[2::3][:n] += dt * fr

        # Clamp
        z[0::3][:n] = np.clip(z[0::3][:n], 0, w)
        z[1::3][:n] = np.clip(z[1::3][:n], 0, h)
        z[2::3][:n] = np.maximum(z[2::3][:n], 0.01)

        # Adaptive step
        max_force = max(np.max(np.abs(fx)), np.max(np.abs(fy)), np.max(np.abs(fr)))
        if max_force < convergence_tol:
            break
        elif max_force > 10:
            dt = max(min_dt, dt * 0.5)
        else:
            dt = min(0.01, dt * 1.05)

    return z


def _expansion_relaxation_polish(z, max_time=30.0, start_time=None):
    """
    Iteratively try to expand radii and adjust positions to accommodate.
    """
    if start_time is None:
        start_time = time.time()
    
    z = z.copy()
    n = N
    step = 0.002
    
    for it in range(200):
        if time.time() - start_time > max_time:
            break
        
        x = z[0::3][:n]
        y = z[1::3][:n]
        r = z[2::3][:n]
        w = z[-1]
        h = 2.0 - w
        
        improved = False
        for i in range(n):
            # Try to expand radius i
            new_r = r.copy()
            new_r[i] += step
            
            if new_r[i] > w/2 or new_r[i] > h/2:
                continue
                
            # Try small position adjustments to fit
            z_test = z.copy()
            z_test[2*i + 2] = new_r[i]
            
            # Quick check: does it still fit?
            if z_test[2*i + 2] <= z_test[0::3][i] and \
               z_test[0::3][i] + z_test[2*i + 2] <= z_test[-1] and \
               z_test[2*i + 2] <= z_test[1::3][i] and \
               z_test[1::3][i] + z_test[2*i + 2] <= 2.0 - z_test[-1]:
                
                # Check overlaps roughly
                valid = True
                for j in range(n):
                    if i == j:
                        continue
                    dx = z_test[0::3][i] - z_test[0::3][j]
                    dy = z_test[1::3][i] - z_test[1::3][j]
                    d = np.sqrt(dx*dx + dy*dy)
                    if d < new_r[i] + r[j] - 1e-9:
                        valid = False
                        break
                
                if valid:
                    z = z_test
                    improved = True
                    break
        
        if not improved:
            step *= 0.7
            if step < 1e-7:
                break
    
    return z


def _constraint_refine(z, max_time=20.0, start_time=None):
    """
    Final high-precision polish: tighten tolerances and run SLSQP with tight ftol.
    """
    if start_time is None:
        start_time = time.time()
    
    n = N
    z0 = z.copy()
    
    def objective_tight(z_):
        return _vectorized_objective(z_, penalty_weight=5000.0, beta=150.0)
    
    bounds = [(0, 2.0)] * (3 * n) + [(0.3, 1.7)]
    
    try:
        res = minimize(objective_tight, z0, method='SLSQP',
                       bounds=bounds,
                       options={'maxiter': 500, 'ftol': 1e-15, 'maxls': 200})
        return res.x
    except Exception:
        return z0


def _cellular_decomposition(w, h, n_circles):
    """
    Generate a grid-based cellular decomposition that guarantees disjoint interiors.
    Returns initial positions and radii for n_circles circles.
    Uses variable cell sizes to better utilize space.
    """
    # Choose grid dimensions close to sqrt(n)
    # Try several aspect ratios for the grid
    best_cells = None
    best_score = -1
    
    for cols in range(1, n_circles + 1):
        rows = int(np.ceil(n_circles / cols))
        if rows * cols < n_circles:
            continue
        
        # Cell dimensions
        cw = w / cols
        ch = h / rows
        
        # Inradius of each cell
        r_cell = min(cw, ch) / 2.0 - 1e-6
        
        # Score: total area of circles / rectangle area (packing density proxy)
        score = np.pi * r_cell * r_cell * n_circles / (w * h)
        
        if score > best_score:
            best_score = score
            best_cells = (cols, rows, cw, ch, r_cell)
    
    if best_cells is None:
        cols = int(np.ceil(np.sqrt(n_circles)))
        rows = int(np.ceil(n_circles / cols))
        cw = w / cols
        ch = h / rows
        r_cell = min(cw, ch) / 2.0 - 1e-6
        best_cells = (cols, rows, cw, ch, r_cell)
    
    cols, rows, cw, ch, r_cell = best_cells
    
    x = np.zeros(n_circles)
    y = np.zeros(n_circles)
    r = np.zeros(n_circles)
    
    for i in range(n_circles):
        row, col = divmod(i, cols)
        # Center of cell
        x[i] = (col + 0.5) * cw
        y[i] = (row + 0.5) * ch
        r[i] = r_cell
    
    return x, y, r


def construct():
    start_time = time.time()
    deadline = start_time + _TIME_LIMIT  # Safety margin before 150s kill

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

    # Generate diverse initial guesses using cellular decomposition
    def get_initial_guesses():
        guesses = []
        
        # Try different aspect ratios with cellular decomposition
        # Include more aspect ratios for better coverage
        aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.75, 0.55, 2.0, 0.5, 1.2, 0.8, 1.1, 0.9, 1.4, 0.7, 1.6, 0.65]
        for w0 in aspect_ratios:
            h0 = 2.0 - w0
            if h0 < 0.3:
                continue
            
            x, y, r = _cellular_decomposition(w0, h0, N)
            z = np.zeros(3 * N + 1)
            z[0::3][:N] = x
            z[1::3][:N] = y
            z[2::3][:N] = r
            z[-1] = w0
            guesses.append(z)
        
        # Random starts as backup with better distribution
        rng = np.random.default_rng(42)
        for _ in range(8):
            w0 = rng.uniform(0.6, 1.4)
            h0 = 2.0 - w0
            z = np.zeros(3 * N + 1)
            for i in range(N):
                r0 = rng.uniform(0.04, 0.15)
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
        if time.time() > deadline - 25:
            break

        # Force-directed relaxation to resolve initial overlaps
        z0_relaxed = _force_relax_unified(z0, max_iter=500)

        # SLSQP optimization with adaptive penalty weight
        remaining = deadline - (time.time() - start_time)
        maxiter = int(min(600, max(150, remaining * 50)))
        
        # Use higher beta for later iterations (sharper constraints)
        current_beta = 80.0 + 20.0 * (i / max(1, len(guesses)))
        
        def objective_with_beta(z_, beta=current_beta):
            return _vectorized_objective(z_, penalty_weight=1000.0, beta=beta)
        
        try:
            res = minimize(objective_with_beta, z0_relaxed, method='SLSQP',
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
    if best_result is not None and time.time() < deadline - 30:
        # 1. Force-directed position polish to ensure we are in a good basin
        polished = _force_relax_unified(best_result, max_iter=300, max_time=15.0)
        valid, val = check_validity(polished)
        if valid and val > best_value:
            best_value = val
            best_result = polished

        # 2. Expansion/Relocation Polish
        if best_result is not None and time.time() < deadline - 18:
            expanded = _expansion_relaxation_polish(best_result, max_time=12.0)
            valid, val = check_validity(expanded)
            if valid and val > best_value:
                best_value = val
                best_result = expanded

        # 3. Constraint-based SLSQP refinement with high precision
        if best_result is not None and time.time() < deadline - 10:
            refined = _constraint_refine(best_result, max_time=8.0)
            valid, val = check_validity(refined)
            if valid and val > best_value:
                best_value = val
                best_result = refined

        # 4. Second round of force-directed polish after refinement
        if best_result is not None and time.time() < deadline - 5:
            polished2 = _force_relax_unified(best_result, max_iter=200, max_time=4.0)
            valid, val = check_validity(polished2)
            if valid and val > best_value:
                best_value = val
                best_result = polished2

        # 5. Final high-precision SLSQP with very tight tolerances
        if best_result is not None and time.time() < deadline - 2:
            def final_objective(z_):
                return _vectorized_objective(z_, penalty_weight=10000.0, beta=200.0)
            
            try:
                res_final = minimize(final_objective, best_result, method='SLSQP',
                                     bounds=[(0, 2)] * (3 * N) + [(0.3, 1.7)],
                                     options={'maxiter': 200, 'ftol': 1e-16, 'maxls': 300})
                z_final = res_final.x
                valid, val = check_validity(z_final)
                if valid and val > best_value:
                    best_value = val
                    best_result = z_final.copy()
            except Exception:
                pass

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
    for _ in range(10):
        r_trial = r * (1 - 1e-9)
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    return np.column_stack((x, y, r))
