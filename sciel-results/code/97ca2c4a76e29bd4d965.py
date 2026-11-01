"""Circle packing in perimeter-4 rectangle: SLSQP + Enhanced Force-Directed Polish."""
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

    # --- Phase 2: Enhanced Polish on best result ---
    if best_result is not None and time.time() < deadline - 25:
        # 1. Force-directed position polish to ensure we are in a good basin
        polished = _polish_result(best_result, deadline - (time.time() - start_time) - 15)
        if polished is not None:
            valid, val = check_validity(polished)
            if valid and val > best_value:
                best_value = val
                best_result = polished

        # 2. Expansion/Relocation Polish with enhanced local search
        if best_result is not None and time.time() < deadline - 20:
            expanded = _expansion_relaxation_polish_enhanced(best_result, deadline - (time.time() - start_time) - 10)
            if expanded is not None:
                valid, val = check_validity(expanded)
                if valid and val > best_value:
                    best_value = val
                    best_result = expanded

        # 3. Constraint-based SLSQP refinement
        if best_result is not None and time.time() < deadline - 20:
            refined = _constraint_refine(best_result, deadline - (time.time() - start_time) - 10)
            if refined is not None:
                valid, val = check_validity(refined)
                if valid and val > best_value:
                    best_value = val
                    best_result = refined

        # 4. Epigraph refinement (smooth surrogate)
        if best_result is not None and time.time() < deadline - 15:
            epi = _epigraph_refine(best_result, deadline - (time.time() - start_time) - 8)
            if epi is not None:
                valid, val = check_validity(epi)
                if valid and val > best_value:
                    best_value = val
                    best_result = epi

        # 5. Coordinate ascent refinement
        if best_result is not None and time.time() < deadline - 10:
            ca = _coordinate_ascent_refine(best_result, deadline - (time.time() - start_time) - 5)
            if ca is not None:
                valid, val = check_validity(ca)
                if valid and val > best_value:
                    best_value = val
                    best_result = ca

        # 6. Adaptive radius polish (fine binary search)
        if best_result is not None and time.time() < deadline - 6:
            ar = _adaptive_radius_polish(best_result, deadline - (time.time() - start_time) - 3)
            if ar is not None:
                valid, val = check_validity(ar)
                if valid and val > best_value:
                    best_value = val
                    best_result = ar

        # 7. Final quick polish
        if best_result is not None and time.time() < deadline - 3:
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
    for _ in range(20):
        r_trial = r * (1 - 1e-10)
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    # Final per-circle expansion pass with very tight tolerance
    # This is the key improvement: squeeze out remaining gains by expanding each circle as much as possible
    for i in range(N):
        if time.time() > deadline - 1:
            break
        r_wall = min(x[i], w - x[i], y[i], h - y[i])
        r_max = r_wall
        for j in range(N):
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
            # Increase binary search steps for precision
            for _ in range(30):
                mid = (lo + hi) / 2.0
                if _is_valid_move(x, y, np.where(np.arange(N) == i, mid, r), w, h, i, x[i], y[i]):
                    lo = mid
                else:
                    hi = mid
            new_r = lo
            if new_r > r[i] + 1e-9:
                r[i] = new_r

    return np.column_stack((x, y, r))


def _force_relax_unified(z, max_iter=500):
    """Force-directed relaxation in unified parameter space with adaptive step (vectorized)."""
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

        # Wall repulsion (vectorized)
        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        fx -= left_viol * 100
        fx += right_viol * 100
        fy -= bottom_viol * 100
        fy += top_viol * 100

        # Pairwise repulsion (vectorized)
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

        # Check convergence
        if max_overlap < convergence_tol:
            break

        # Adaptive step size
        force_mag = float(np.max(np.sqrt(fx * fx + fy * fy)))
        if force_mag > 1.0:
            dt = max(min_dt, dt * 0.5)
        elif force_mag < 0.1:
            dt = min(0.01, dt * 1.1)

        # Update positions
        z[0::3][:n] += dt * fx
        z[1::3][:n] += dt * fy

        # Clamp positions
        for i in range(n):
            z[3 * i] = np.clip(z[3 * i], r[i], w - r[i])
            z[3 * i + 1] = np.clip(z[3 * i + 1], r[i], h - r[i])

    return z


def _polish_result(z, time_budget):
    """Polish a valid packing with force-directed position adjustment and radius growth (vectorized)."""
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    dt = 0.001
    max_iter = int(min(3000, time_budget * 150))

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
        fx -= left_viol * 35
        fx += right_viol * 35
        fy -= bottom_viol * 35
        fy += top_viol * 35

        # Pairwise repulsion (vectorized)
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]

        iu = np.triu_indices(n, k=1)
        d_sq_pairs = d_sq[iu]
        r_sum_pairs = r_sum[iu]

        d_pairs = np.sqrt(np.maximum(d_sq_pairs, 1e-12))
        overlap = np.maximum(r_sum_pairs - d_pairs, 0.0)

        if np.max(overlap) > 1e-10:
            f_pairs = overlap * 18.0
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

        # Adaptive step
        max_force = max(float(np.max(np.abs(fx))), float(np.max(np.abs(fy))), 1e-8)
        if max_force > 0.1:
            dt = 0.0005
        else:
            dt = 0.001

        x += dt * fx
        y += dt * fy
        x = np.clip(x, r, w - r)
        y = np.clip(y, r, h - r)

    # Try to grow radii uniformly
    grow_factor = 1.0003
    for _ in range(300):
        if time.time() > time_budget:
            break
        r_trial = r * grow_factor
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
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


def _expansion_relaxation_polish_enhanced(z, time_budget):
    """
    Enhanced polish: Iteratively expand circles with improved position optimization.
    Uses multi-direction local search with adaptive step sizes for better exploration.
    """
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    # Enhanced directions: 12 directions for better coverage
    directions = [
        (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0),
        (0.7071, 0.7071), (-0.7071, 0.7071), (0.7071, -0.7071), (-0.7071, -0.7071),
        (0.866, 0.5), (-0.866, 0.5), (0.866, -0.5), (-0.866, -0.5)
    ]

    max_rounds = 50
    for rnd in range(max_rounds):
        if time.time() > time_budget:
            break
        
        improved = False
        for i in range(n):
            if time.time() > time_budget:
                break
            
            # 1. Try to expand radius i to max possible using binary search
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
                # Binary search for the true max valid radius
                lo = r[i]
                hi = r_max
                for _ in range(30):
                    mid = (lo + hi) / 2.0
                    if _is_valid_move(x, y, np.where(np.arange(n) == i, mid, r), w, h, i, x[i], y[i]):
                        lo = mid
                    else:
                        hi = mid
                new_r = lo
                if new_r > r[i] + 1e-7:
                    r[i] = new_r
                    improved = True
            
            # 2. If expansion is limited, try to shift circle i to create space
            # Find the most constraining direction
            min_gap = float('inf')
            best_dx, best_dy = 0.0, 0.0
            
            # Check walls
            d_left = x[i] - r[i]
            d_right = w - x[i] - r[i]
            d_bottom = y[i] - r[i]
            d_top = h - y[i] - r[i]
            
            if d_left < min_gap:
                min_gap = d_left
                best_dx, best_dy = 1.0, 0.0
            if d_right < min_gap:
                min_gap = d_right
                best_dx, best_dy = -1.0, 0.0
            if d_bottom < min_gap:
                min_gap = d_bottom
                best_dx, best_dy = 0.0, 1.0
            if d_top < min_gap:
                min_gap = d_top
                best_dx, best_dy = 0.0, -1.0
            
            # Check other circles
            for j in range(n):
                if i == j:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                gap = d - (r[i] + r[j])
                if gap < min_gap:
                    min_gap = gap
                    if d > 1e-6:
                        best_dx = dx / d
                        best_dy = dy / d
            
            # 3. Enhanced 12-direction local search to find a better position
            if min_gap < 0.05:
                # Try moving in 12 directions with multiple step sizes
                found_better = False
                for dx_dir, dy_dir in directions:
                    if found_better:
                        break
                    for step in [0.05, 0.03, 0.015, 0.008, 0.004, 0.002, 0.001]:
                        new_x = x[i] + dx_dir * step
                        new_y = y[i] + dy_dir * step
                        if _is_valid_move(x, y, r, w, h, i, new_x, new_y):
                            # Check if this move allows larger radius
                            old_r_max = _max_radius_for_circle(x, y, r, w, h, i)
                            x[i], y[i] = new_x, new_y
                            new_r_max = _max_radius_for_circle(x, y, r, w, h, i)
                            if new_r_max > old_r_max + 1e-6:
                                improved = True
                                found_better = True
                                break
                            else:
                                # Revert if no benefit
                                x[i], y[i] = x[i] - dx_dir * step, y[i] - dy_dir * step

                # 4. After repositioning, try to expand again
                if found_better:
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
                        for _ in range(30):
                            mid = (lo + hi) / 2.0
                            if _is_valid_move(x, y, np.where(np.arange(n) == i, mid, r), w, h, i, x[i], y[i]):
                                lo = mid
                            else:
                                hi = mid
                        new_r = lo
                        if new_r > r[i] + 1e-7:
                            r[i] = new_r
                            improved = True

        if not improved:
            break

    # Second pass: more aggressive expansion with position optimization
    if time.time() < time_budget - 5:
        for rnd in range(30):
            if time.time() > time_budget:
                break
            
            improved = False
            for i in range(n):
                if time.time() > time_budget:
                    break
                
                # Try to expand radius i
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
                
                if r_max > r[i] + 1e-5:
                    # Binary search for the true max valid radius
                    lo = r[i]
                    hi = r_max
                    for _ in range(35):
                        mid = (lo + hi) / 2.0
                        if _is_valid_move(x, y, np.where(np.arange(n) == i, mid, r), w, h, i, x[i], y[i]):
                            lo = mid
                        else:
                            hi = mid
                    new_r = lo
                    if new_r > r[i] + 1e-6:
                        r[i] = new_r
                        improved = True

            if not improved:
                break

    # Final uniform expansion with binary search
    grow_factor = 1.0005
    for _ in range(300):
        if time.time() > time_budget:
            break
        r_trial = r * grow_factor
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
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
    """High-precision SLSQP refinement using explicit constraints."""
    z = z.copy().astype(float)
    n = N

    def obj_refine(zv):
        r = zv[2::3][:n]
        return -np.sum(r)

    constraints = []
    bounds = []

    for i in range(n):
        bounds.extend([(0, 2), (0, 2), (0.001, 1.0)])
    bounds.append((0.3, 1.7))

    def make_boundary_constraint(idx, axis, side):
        if axis == 0:
            if side == 0:
                return lambda p: p[3 * idx] - p[3 * idx + 2]
            else:
                return lambda p: p[-1] - p[3 * idx] - p[3 * idx + 2]
        else:
            if side == 0:
                return lambda p: p[3 * idx + 1] - p[3 * idx + 2]
            else:
                return lambda p: (2.0 - p[-1]) - p[3 * idx + 1] - p[3 * idx + 2]

    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': make_boundary_constraint(i, 0, 0)})
        constraints.append({'type': 'ineq', 'fun': make_boundary_constraint(i, 0, 1)})
        constraints.append({'type': 'ineq', 'fun': make_boundary_constraint(i, 1, 0)})
        constraints.append({'type': 'ineq', 'fun': make_boundary_constraint(i, 1, 1)})

    for i in range(n):
        for j in range(i + 1, n):
            def make_overlap_constraint(idx1, idx2):
                return lambda p: (
                    (p[3 * idx1] - p[3 * idx2]) ** 2 +
                    (p[3 * idx1 + 1] - p[3 * idx2 + 1]) ** 2 -
                    (p[3 * idx1 + 2] + p[3 * idx2 + 2]) ** 2
                )
            constraints.append({'type': 'ineq', 'fun': make_overlap_constraint(i, j)})

    maxiter = int(min(800, time_budget * 150))
    try:
        res = minimize(obj_refine, z, method='SLSQP',
                       bounds=bounds,
                       constraints=constraints,
                       options={'maxiter': maxiter, 'ftol': 1e-14, 'maxls': 100})

        z_opt = res.x
        x = z_opt[0::3][:n]
        y = z_opt[1::3][:n]
        r = z_opt[2::3][:n]
        w = z_opt[-1]
        h = 2.0 - w

        if _is_valid(x, y, r, w, h):
            return z_opt
        return None
    except Exception:
        return None


def _lse(values, beta=100.0):
    """Log-sum-exp for smooth max approximation."""
    values = np.asarray(values)
    m = np.max(values)
    if m < -1e6:
        return m
    return m + np.log(np.sum(np.exp(beta * (values - m)))) / beta


def _epigraph_refine(z, time_budget):
    """
    Epigraph formulation: introduce auxiliary variable t for each circle's max radius.
    Objective: maximize sum(t_i) subject to t_i <= min over all constraints of circle i.
    Uses smooth log-sum-exp surrogate for the min constraint.
    """
    z = z.copy().astype(float)
    n = N
    x0 = z[0::3][:n].copy()
    y0 = z[1::3][:n].copy()
    r0 = z[2::3][:n].copy()
    w0 = z[-1]
    h0 = 2.0 - w0

    # Variables: x[n], y[n], t[n], w (total 3n+1)
    # t[i] <= min of all radius constraints for circle i
    # We use a smooth penalty: sum(t) - beta * sum(lse(-gaps_i))
    # where gaps_i are the constraint values (positive = feasible)

    def epiph_obj(p):
        x = p[0:n]
        y = p[n:2*n]
        t = p[2*n:3*n]
        w = p[3*n]
        h = 2.0 - w

        if w < 0.2 or h < 0.2:
            return 1e6

        obj = -np.sum(t)

        # For each circle i, compute all gap constraints:
        # Wall gaps: x[i]-t[i], w-x[i]-t[i], y[i]-t[i], h-y[i]-t[i]
        # Pair gaps: dist(i,j) - t[i] - t[j] for all j != i
        # We want t[i] <= min(all gaps for circle i)
        # Smooth: penalize t[i] - lse(-gaps_i) using log-sum-exp

        for i in range(n):
            gaps = np.array([
                x[i] - t[i],
                w - x[i] - t[i],
                y[i] - t[i],
                h - y[i] - t[i]
            ])
            # Add pair gaps
            for j in range(n):
                if j == i:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                gaps = np.append(gaps, d - t[i] - t[j])

            # Smooth max of (t[i] - gap) over all gaps
            violations = t[i] - gaps
            # lse of violations: smooth max
            max_viol = _lse(violations, beta=80.0)
            obj += max(0.0, max_viol) ** 2 * 50.0

        # Keep w in range
        if w < 0.4:
            obj += 200.0 * (0.4 - w) ** 2
        elif w > 1.6:
            obj += 200.0 * (w - 1.6) ** 2

        return obj

    # Initial guess: t[i] = r0[i]
    p0 = np.empty(3 * n + 1)
    p0[0:n] = x0
    p0[n:2*n] = y0
    p0[2*n:3*n] = r0
    p0[3*n] = w0

    bounds = [(0, 2)] * (2 * n) + [(0.001, 0.6)] * n + [(0.3, 1.7)]

    maxiter = int(min(600, time_budget * 80))
    try:
        res = minimize(epiph_obj, p0, method='SLSQP',
                       bounds=bounds,
                       options={'maxiter': maxiter, 'ftol': 1e-13, 'maxls': 100})

        p_opt = res.x
        x = p_opt[0:n]
        y = p_opt[n:2*n]
        t = p_opt[2*n:3*n]
        w = p_opt[3*n]
        h = 2.0 - w

        # Use min(t, actual max radius) to ensure validity
        r = np.zeros(n)
        for i in range(n):
            r_wall = min(x[i], w - x[i], y[i], h - y[i])
            r_max = r_wall
            for j in range(n):
                if j == i:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                r_allowed = d - t[j]
                if r_allowed < r_max:
                    r_max = r_allowed
            r[i] = min(t[i], r_max)

        if _is_valid(x, y, r, w, h):
            z_new = np.empty(3 * N + 1)
            z_new[0::3][:n] = x
            z_new[1::3][:n] = y
            z_new[2::3][:n] = r
            z_new[-1] = w
            return z_new
        return None
    except Exception:
        return None


def _coordinate_ascent_refine(z, time_budget):
    """
    Coordinate ascent: for each circle, optimize its (x, y) position
    to maximize its radius given all other circles fixed.
    Uses golden-section search in 2D.
    """
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    def max_radius_at(xi, yi, i):
        """Max radius for circle i at position (xi, yi) given others fixed."""
        r_wall = min(xi, w - xi, yi, h - yi)
        if r_wall <= 0:
            return 0.0
        r_max = r_wall
        for j in range(n):
            if j == i:
                continue
            dx = xi - x[j]
            dy = yi - y[j]
            d = np.sqrt(dx * dx + dy * dy)
            r_allowed = d - r[j]
            if r_allowed < r_max:
                r_max = r_allowed
        return max(0.0, r_max)

    improved = True
    max_rounds = 20
    for rnd in range(max_rounds):
        if time.time() > time_budget:
            break
        improved = False
        for i in range(n):
            if time.time() > time_budget:
                break

            # Current best radius at current position
            r_cur = max_radius_at(x[i], y[i], i)

            # Try to find better position via local search
            # Search in a small neighborhood
            best_r = r_cur
            best_x, best_y = x[i], y[i]

            # Multi-scale local search
            for scale in [0.1, 0.05, 0.02, 0.01, 0.005, 0.002]:
                # 8 directions + center
                dirs = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
                         (0.7071, 0.7071), (-0.7071, 0.7071),
                         (0.7071, -0.7071), (-0.7071, -0.7071)]
                for dx_d, dy_d in dirs:
                    nx = x[i] + dx_d * scale
                    ny = y[i] + dy_d * scale
                    # Clamp to valid range
                    nx = np.clip(nx, 0, w)
                    ny = np.clip(ny, 0, h)
                    r_try = max_radius_at(nx, ny, i)
                    if r_try > best_r + 1e-8:
                        best_r = r_try
                        best_x, best_y = nx, ny

            if best_r > r_cur + 1e-7:
                x[i], y[i] = best_x, best_y
                r[i] = min(r[i], best_r)  # Keep radius valid
                # Actually set to the max possible
                r[i] = best_r
                improved = True

        if not improved:
            break

    # Final: set each radius to its max
    for i in range(n):
        r[i] = max_radius_at(x[i], y[i], i)

    if _is_valid(x, y, r, w, h):
        z_new = np.empty(3 * N + 1)
        z_new[0::3][:n] = x
        z_new[1::3][:n] = y
        z_new[2::3][:n] = r
        z_new[-1] = w
        return z_new
    return None


def _adaptive_radius_polish(z, time_budget):
    """
    Per-circle adaptive radius polish: for each circle, use binary search
    with very fine tolerance to find the exact maximum radius.
    Interleaves with small position adjustments.
    """
    z = z.copy().astype(float)
    n = N
    x = z[0::3][:n].copy()
    y = z[1::3][:n].copy()
    r = z[2::3][:n].copy()
    w = z[-1]
    h = 2.0 - w

    max_rounds = 40
    for rnd in range(max_rounds):
        if time.time() > time_budget:
            break
        improved = False

        for i in range(n):
            if time.time() > time_budget:
                break

            # Compute max radius at current position
            r_wall = min(x[i], w - x[i], y[i], h - y[i])
            r_max = r_wall
            for j in range(n):
                if j == i:
                    continue
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d = np.sqrt(dx * dx + dy * dy)
                r_allowed = d - r[j]
                if r_allowed < r_max:
                    r_max = r_allowed
            r_max = max(0.0, r_max)

            if r_max > r[i] + 1e-10:
                # Fine binary search (40 iterations for ~1e-12 precision)
                lo = r[i]
                hi = r_max
                for _ in range(40):
                    mid = (lo + hi) / 2.0
                    r_test = r.copy()
                    r_test[i] = mid
                    if _is_valid(x, y, r_test, w, h):
                        lo = mid
                    else:
                        hi = mid
                new_r = lo
                if new_r > r[i] + 1e-10:
                    r[i] = new_r
                    improved = True
            elif r_max < r[i] - 1e-10:
                # Radius too large, shrink
                r[i] = r_max
                improved = True

        # Small position nudges to create more space
        if rnd % 3 == 2 and time.time() < time_budget - 5:
            for i in range(n):
                if time.time() > time_budget:
                    break
                # Try tiny random nudges
                for _ in range(5):
                    dx = np.random.uniform(-0.005, 0.005)
                    dy = np.random.uniform(-0.005, 0.005)
                    nx = x[i] + dx
                    ny = y[i] + dy
                    if 0 <= nx <= w and 0 <= ny <= h:
                        # Check if this allows larger radius
                        r_wall = min(nx, w - nx, ny, h - ny)
                        r_max_try = r_wall
                        for j in range(n):
                            if j == i:
                                continue
                            ddx = nx - x[j]
                            ddy = ny - y[j]
                            d = np.sqrt(ddx * ddx + ddy * ddy)
                            r_allowed = d - r[j]
                            if r_allowed < r_max_try:
                                r_max_try = r_allowed
                        r_max_try = max(0.0, r_max_try)
                        if r_max_try > r[i] + 1e-7:
                            x[i], y[i] = nx, ny
                            r[i] = r_max_try
                            improved = True
                            break

        if not improved:
            break

    if _is_valid(x, y, r, w, h):
        z_new = np.empty(3 * N + 1)
        z_new[0::3][:n] = x
        z_new[1::3][:n] = y
        z_new[2::3][:n] = r
        z_new[-1] = w
        return z_new
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
