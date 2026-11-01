"""Circle packing in perimeter-4 rectangle: SLSQP + Advanced Expansion Polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

def _smooth_max(v, eps=1e-6):
    """Smooth approximation of max(0, v)."""
    return 0.5 * (v + np.sqrt(v * v + eps))

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

    # Vectorized pairwise overlap penalty (smooth)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d_sq = dx * dx + dy * dy
    r_sum = r[:, None] + r[None, :]

    iu = np.triu_indices(N, k=1)
    d_sq_pairs = d_sq[iu]
    r_sum_pairs = r_sum[iu]

    d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
    overlap = _smooth_max(r_sum_pairs - d_pairs)
    overlap_penalty = penalty_weight * np.sum(overlap * overlap)

    obj += overlap_penalty

    # Vectorized boundary penalty (smooth)
    left_viol = _smooth_max(r - x)
    right_viol = _smooth_max(x + r - w)
    bottom_viol = _smooth_max(r - y)
    top_viol = _smooth_max(y + r - h)

    boundary_penalty = penalty_weight * (
        np.sum(left_viol * left_viol) +
        np.sum(right_viol * right_viol) +
        np.sum(bottom_viol * bottom_viol) +
        np.sum(top_viol * top_viol)
    )

    obj += boundary_penalty

    # Keep aspect ratio in reasonable range (smooth penalty)
    w_low_viol = _smooth_max(0.5 - w)
    w_high_viol = _smooth_max(w - 1.5)
    obj += penalty_weight * (w_low_viol * w_low_viol + w_high_viol * w_high_viol)

    return obj


def _is_valid(x, y, r, w, h, tol=1e-9):
    """Check if the configuration is strictly valid."""
    if w <= 0.05 or h <= 0.05:
        return False
    for i in range(N):
        if r[i] <= 0:
            return False
        if x[i] < r[i] - tol or x[i] > w - r[i] + tol:
            return False
        if y[i] < r[i] - tol or y[i] > h - r[i] + tol:
            return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d2 = dx * dx + dy * dy
            r_sum = r[i] + r[j]
            if d2 < r_sum ** 2 - tol:
                return False
    return True


def _check_validity(z, tol=1e-9):
    """Check validity of unified parameter vector."""
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]
    w = z[-1]
    h = 2.0 - w
    if _is_valid(x, y, r, w, h, tol):
        return True, float(np.sum(r))
    return False, 0.0


def _force_relax_unified(z, max_iter=500, deadline=None):
    """Force-directed relaxation in unified parameter space with adaptive step (vectorized)."""
    z = z.copy().astype(float)
    n = N
    dt = 0.005
    min_dt = 0.0001
    convergence_tol = 1e-8
    prev_obj = _vectorized_objective(z)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
            
        x = z[0::3][:n]
        y = z[1::3][:n]
        r = z[2::3][:n]
        w = z[-1]
        h = 2.0 - w
        
        if w <= 0.1 or h <= 0.1:
            break

        # Compute forces
        fx = np.zeros(n)
        fy = np.zeros(n)
        fr = np.zeros(n)
        fw = 0.0

        # Pairwise forces
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d_sq = dx * dx + dy * dy
                d = np.sqrt(d_sq) if d_sq > 1e-12 else 1e-12
                r_sum = r[i] + r[j]
                
                if d < r_sum:
                    # Overlap: push apart and shrink
                    overlap = r_sum - d
                    force = overlap * 10.0
                    if d > 1e-9:
                        fx[i] += force * dx / d
                        fy[i] += force * dy / d
                        fx[j] -= force * dx / d
                        fy[j] -= force * dy / d
                    fr[i] -= force * 0.1
                    fr[j] -= force * 0.1
                else:
                    # Small repulsion to prevent sticking
                    repulse = 0.01 / (d + 0.1)
                    if d > 1e-9:
                        fx[i] += repulse * dx / d
                        fy[i] += repulse * dy / d
                        fx[j] -= repulse * dx / d
                        fy[j] -= repulse * dy / d

        # Boundary forces
        for i in range(n):
            if x[i] < r[i]:
                fx[i] += (r[i] - x[i]) * 10.0
                fr[i] -= (r[i] - x[i]) * 0.5
            elif x[i] > w - r[i]:
                fx[i] -= (x[i] - (w - r[i])) * 10.0
                fr[i] -= (x[i] - (w - r[i])) * 0.5
            if y[i] < r[i]:
                fy[i] += (r[i] - y[i]) * 10.0
                fr[i] -= (r[i] - y[i]) * 0.5
            elif y[i] > h - r[i]:
                fy[i] -= (y[i] - (h - r[i])) * 10.0
                fr[i] -= (y[i] - (h - r[i])) * 0.5
            
            # Grow radius if possible
            fr[i] += 0.01

        # Update
        z[0::3][:n] += dt * fx
        z[1::3][:n] += dt * fy
        z[2::3][:n] += dt * fr
        
        # Clip
        z[0::3][:n] = np.clip(z[0::3][:n], 0.01, w - 0.01)
        z[1::3][:n] = np.clip(z[1::3][:n], 0.01, h - 0.01)
        z[2::3][:n] = np.clip(z[2::3][:n], 0.005, min(w, h) / 2 - 0.01)
        
        obj = _vectorized_objective(z)
        if abs(prev_obj - obj) < convergence_tol:
            break
        prev_obj = obj
        dt *= 0.99
        if dt < min_dt:
            dt = min_dt

    return z


def _polish_result(z_best, time_budget):
    """Polish a result using SLSQP with tight tolerance."""
    start_time = time.time()
    deadline = start_time + time_budget
    
    x = z_best[0::3][:N]
    y = z_best[1::3][:N]
    r = z_best[2::3][:N]
    w = z_best[-1]
    h = 2.0 - w
    
    # Tighter bounds
    bounds = []
    for i in range(N):
        bounds.append((r[i] * 0.9, w - r[i] * 0.9)) # x
        bounds.append((r[i] * 0.9, h - r[i] * 0.9)) # y
        bounds.append((0.01, min(w, h) / 2)) # r
    bounds.append((0.3, 1.7)) # w
    
    def obj(z):
        return _vectorized_objective(z, penalty_weight=10000.0)
    
    try:
        res = minimize(obj, z_best, method='SLSQP', bounds=bounds,
                       options={'maxiter': 1000, 'ftol': 1e-14, 'maxls': 100})
        if time.time() > deadline:
            return None
        return res.x
    except Exception:
        return None

def _mini_polish(z_best, time_budget):
    """Quick polish step for use inside expansion loops."""
    start_time = time.time()
    deadline = start_time + time_budget
    def obj(z):
        return _vectorized_objective(z, penalty_weight=5000.0)
    try:
        res = minimize(obj, z_best, method='SLSQP',
                       bounds=[(0, 2)] * (3 * N) + [(0.3, 1.7)],
                       options={'maxiter': 200, 'ftol': 1e-12, 'maxls': 50})
        if time.time() > deadline:
            return None
        return res.x
    except Exception:
        return None


def _expansion_relaxation_polish(z_best, time_budget):
    """Try to expand radii slightly and relax positions."""
    start_time = time.time()
    deadline = start_time + time_budget
    
    z = z_best.copy()
    x = z[0::3][:N]
    y = z[1::3][:N]
    r = z[2::3][:N]
    w = z[-1]
    h = 2.0 - w
    
    # Attempt small expansion
    expansion_steps = 20
    for step in range(expansion_steps):
        if time.time() > deadline:
            break
        
        # Grow radii by 0.5%
        r_new = r * 1.005
        z[2::3][:N] = r_new
        
        # Quick force relax
        z = _force_relax_unified(z, max_iter=50, deadline=deadline)
        
        # Quick SLSQP polish
        z_polished = _mini_polish(z, deadline - (time.time() - start_time) - 5)
        if z_polished is not None:
            z = z_polished
            
        # Check if valid
        valid, val = _check_validity(z)
        if valid:
            r = z[2::3][:N]
        else:
            # Roll back
            z[2::3][:N] = r
            break
            
    return z


def _constraint_refine(z_best, time_budget):
    """Refine using SLSQP with constraints instead of penalties (if possible, else fallback)."""
    # For simplicity and robustness, we use a higher penalty weight SLSQP here
    # which effectively acts like constraint handling for the optimizer.
    start_time = time.time()
    deadline = start_time + time_budget
    
    def obj(z):
        return _vectorized_objective(z, penalty_weight=50000.0)
        
    try:
        res = minimize(obj, z_best, method='SLSQP',
                       bounds=[(0, 2)] * (3 * N) + [(0.3, 1.7)],
                       options={'maxiter': 500, 'ftol': 1e-15, 'maxls': 200})
        if time.time() > deadline:
            return None
        return res.x
    except Exception:
        return None


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    best_value = 0.0
    best_result = None

    penalty_weight = 1000.0

    def objective(z):
        return _vectorized_objective(z, penalty_weight)

    def check_validity(z, tol=1e-9):
        return _check_validity(z, tol)

    # Generate diverse initial guesses
    def get_initial_guesses():
        guesses = []
        side = int(np.ceil(np.sqrt(N)))

        # More diverse aspect ratios
        aspect_ratios = [1.0, 0.85, 1.18, 0.75, 1.33, 0.67, 1.5, 0.6, 1.75, 0.55, 2.0, 0.5, 1.2, 0.8, 0.9, 1.1]
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
        for _ in range(15):
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
        z0_relaxed = _force_relax_unified(z0, max_iter=500, deadline=deadline - 20)

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

        # 2. Expansion/Relocation Polish (Enhanced with mini-polish)
        if best_result is not None and time.time() < deadline - 15:
            expanded = _expansion_relaxation_polish(best_result, deadline - (time.time() - start_time) - 8)
            if expanded is not None:
                valid, val = check_validity(expanded)
                if valid and val > best_value:
                    best_value = val
                    best_result = expanded

        # 3. Constraint-based SLSQP refinement
        if best_result is not None and time.time() < deadline - 12:
            refined = _constraint_refine(best_result, deadline - (time.time() - start_time) - 6)
            if refined is not None:
                valid, val = check_validity(refined)
                if valid and val > best_value:
                    best_value = val
                    best_result = refined

        # 4. Final quick polish
        if best_result is not None and time.time() < deadline - 4:
            polished2 = _polish_result(best_result, deadline - (time.time() - start_time))
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
    for _ in range(10):
        r_trial = r * (1 - 1e-9)
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    return np.column_stack((x, y, r))
