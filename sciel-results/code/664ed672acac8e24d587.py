import numpy as np
from scipy.optimize import minimize
import time

N = 32

def construct():
    start_time = time.time()
    deadline = start_time + 135

    def softplus(x, scale=1.0):
        """Numerically stable softplus for smooth max(0, x)."""
        x_scaled = x / scale
        return scale * np.log1p(np.exp(-np.abs(x_scaled))) + np.maximum(0, x)

    def physics_optimize(params, max_iter=200, alpha=100.0, beta=50.0):
        n = N
        xy = params[:2*n].reshape(n, 2).copy()
        r = params[2*n:].copy()
        pair_i, pair_j = np.triu_indices(n, k=1)
        
        for it in range(max_iter):
            fx = np.zeros(n)
            fy = np.zeros(n)
            
            # Vectorized pair forces with smooth contact detection
            dx = xy[pair_j, 0] - xy[pair_i, 0]
            dy = xy[pair_j, 1] - xy[pair_i, 1]
            d2 = dx*dx + dy*dy
            d = np.sqrt(np.maximum(d2, 1e-12))
            min_d = r[pair_i] + r[pair_j]
            
            # Soft contact: smooth transition around contact distance
            contact_margin = min_d - d  # positive when overlapping
            # Use softplus for smooth force activation
            soft_contact = softplus(contact_margin, scale=0.008)
            
            # Force magnitude: strong repulsion when overlapping, smooth decay
            force_mag = alpha * soft_contact / (d + 0.005)
            # Add weak attraction for near-contact pairs to encourage tight packing
            near_contact = softplus(-contact_margin, scale=0.015) * 0.1
            force_mag -= near_contact * alpha * 0.08
            force_mag = np.clip(force_mag, -alpha*10, alpha*10)
            
            fx[pair_i] -= force_mag * dx / d
            fy[pair_i] -= force_mag * dy / d
            fx[pair_j] += force_mag * dx / d
            fy[pair_j] += force_mag * dy / d
            
            # Vectorized boundary forces with smooth margins
            dist_left = xy[:, 0] - r
            dist_right = 1 - xy[:, 0] - r
            dist_bottom = xy[:, 1] - r
            dist_top = 1 - xy[:, 1] - r
            
            # Smooth boundary repulsion
            margin = 0.03
            fx += beta * softplus(margin - dist_left, scale=0.008)
            fx -= beta * softplus(margin - dist_right, scale=0.008)
            fy += beta * softplus(margin - dist_bottom, scale=0.008)
            fy -= beta * softplus(margin - dist_top, scale=0.008)
            
            # Centering force
            fx += 0.05 * (0.5 - xy[:, 0])
            fy += 0.05 * (0.5 - xy[:, 1])
            
            lr = 0.008 / (1 + it * 0.015)
            xy[:, 0] += lr * fx
            xy[:, 1] += lr * fy
            
            xy[:, 0] = np.clip(xy[:, 0], r, 1 - r)
            xy[:, 1] = np.clip(xy[:, 1], r, 1 - r)
            
            if np.max(np.abs(fx)) < 1e-9 and np.max(np.abs(fy)) < 1e-9:
                break
        
        return np.concatenate([xy.flatten(), r])

    best_result = None
    best_score = -np.inf

    def penalty_function(params):
        n = N
        xy = params[:2*n].reshape(n, 2)
        r = params[2*n:]

        d_left = xy[:, 0] - r
        d_right = 1 - xy[:, 0] - r
        d_bottom = xy[:, 1] - r
        d_top = 1 - xy[:, 1] - r
        boundary_viol = np.sum(np.maximum(0, -d_left)**2) + np.sum(np.maximum(0, -d_right)**2) + \
                        np.sum(np.maximum(0, -d_bottom)**2) + np.sum(np.maximum(0, -d_top)**2)

        dx = xy[:, 0][:, None] - xy[:, 0][None, :]
        dy = xy[:, 1][:, None] - xy[:, 1][None, :]
        dist = np.sqrt(dx**2 + dy**2)
        min_dist = r[:, None] + r[None, :]
        iu = np.triu_indices(n, k=1)
        overlap = np.maximum(0, min_dist[iu] - dist[iu])
        overlap_viol = np.sum(overlap**2)

        return -np.sum(r) + 100 * boundary_viol + 100 * overlap_viol

    def smooth_penalty_function(params):
        n = N
        xy = params[:2*n].reshape(n, 2)
        r = params[2*n:]

        d_left = xy[:, 0] - r
        d_right = 1 - xy[:, 0] - r
        d_bottom = xy[:, 1] - r
        d_top = 1 - xy[:, 1] - r

        # Softplus-based smooth max for better gradient flow
        def smooth_max_square(val, scale=0.005):
            # val > 0 means violation
            sm = softplus(val, scale=scale)
            return np.sum(sm**2)

        boundary_viol = smooth_max_square(r - xy[:, 0]) + \
                        smooth_max_square(r - (1 - xy[:, 0])) + \
                        smooth_max_square(r - xy[:, 1]) + \
                        smooth_max_square(r - (1 - xy[:, 1]))

        dx = xy[:, 0][:, None] - xy[:, 0][None, :]
        dy = xy[:, 1][:, None] - xy[:, 1][None, :]
        dist = np.sqrt(dx**2 + dy**2)
        min_dist = r[:, None] + r[None, :]
        iu = np.triu_indices(n, k=1)
        
        diff = min_dist[iu] - dist[iu]
        overlap_viol = smooth_max_square(diff, scale=0.01)

        return -np.sum(r) + 100 * boundary_viol + 100 * overlap_viol

    def soft_contact_objective(params):
        """
        Objective using differentiable soft contact graph with adaptive
        activation threshold based on nearest-neighbor distances.
        """
        n = N
        xy = params[:2*n].reshape(n, 2)
        r = params[2*n:]

        # Boundary constraints with softplus
        d_left = xy[:, 0] - r
        d_right = 1 - xy[:, 0] - r
        d_bottom = xy[:, 1] - r
        d_top = 1 - xy[:, 1] - r
        boundary_viol = np.sum(softplus(-d_left, scale=0.004)**2) + \
                        np.sum(softplus(-d_right, scale=0.004)**2) + \
                        np.sum(softplus(-d_bottom, scale=0.004)**2) + \
                        np.sum(softplus(-d_top, scale=0.004)**2)

        # Differentiable soft contact graph
        dx = xy[:, 0][:, None] - xy[:, 0][None, :]
        dy = xy[:, 1][:, None] - xy[:, 1][None, :]
        dist = np.sqrt(dx**2 + dy**2 + 1e-12)
        min_dist = r[:, None] + r[None, :]
        iu = np.triu_indices(n, k=1)
        
        # Contact margin: positive when overlapping
        contact_margin = min_dist[iu] - dist[iu]
        
        # Adaptive activation: use log-sum-exp for smooth max over top-k contacts
        # This creates a differentiable "contact graph" where only the most
        # constrained pairs get strong gradient signal
        k = 8  # number of nearest neighbors to consider
        sorted_margins = np.sort(contact_margin)[::-1]
        top_margins = sorted_margins[:k]
        
        # Log-sum-exp for smooth max (with temperature)
        temp = 0.008
        lse = temp * np.log(np.sum(np.exp(top_margins / temp)))
        
        # Soft activation with adaptive threshold
        active_contact = softplus(contact_margin, scale=0.002)
        overlap_viol = np.sum(active_contact**2)
        
        # Additional penalty on the smooth max to ensure top constraints are satisfied
        lse_penalty = softplus(lse, scale=0.004) ** 2
        
        # Tight packing encouragement for near-contact pairs
        near_but_not_contact = softplus(-contact_margin, scale=0.008) * 0.08
        tightness_bonus = np.sum(near_but_not_contact) * 0.008

        return -np.sum(r) + 100 * boundary_viol + 100 * (overlap_viol + lse_penalty) - tightness_bonus

    def is_valid(result, tol=1e-9):
        n = N
        xy = result[:2*n].reshape(n, 2)
        r = result[2*n:]

        if np.any(r < -tol):
            return False
        if np.any(xy[:, 0] - r < -tol) or np.any(1 - xy[:, 0] - r < -tol):
            return False
        if np.any(xy[:, 1] - r < -tol) or np.any(1 - xy[:, 1] - r < -tol):
            return False

        dx = xy[:, 0][:, None] - xy[:, 0][None, :]
        dy = xy[:, 1][:, None] - xy[:, 1][None, :]
        dist = np.sqrt(dx**2 + dy**2)
        min_dist = r[:, None] + r[None, :]
        iu = np.triu_indices(n, k=1)
        if np.any(dist[iu] < min_dist[iu] - tol):
            return False

        return True

    def get_score(result):
        n = N
        r = result[2*n:]
        return np.sum(r)

    def push_apart(params, n_iters=100):
        n = N
        xy = params[:2*n].reshape(n, 2).copy()
        r = params[2*n:].copy()

        for _ in range(n_iters):
            moved = False

            for i in range(n):
                for j in range(i+1, n):
                    dx = xy[j, 0] - xy[i, 0]
                    dy = xy[j, 1] - xy[i, 1]
                    dist = np.sqrt(dx**2 + dy**2)
                    min_dist = r[i] + r[j]

                    if dist < min_dist and dist > 1e-12:
                        overlap = min_dist - dist
                        wi = r[i] / (r[i] + r[j]) if r[i] + r[j] > 1e-15 else 0.5
                        wj = 1 - wi

                        push_x = (dx / dist) * overlap
                        push_y = (dy / dist) * overlap

                        xy[i, 0] -= push_x * wi
                        xy[i, 1] -= push_y * wi
                        xy[j, 0] += push_x * wj
                        xy[j, 1] += push_y * wj
                        moved = True
                    elif dist <= 1e-12:
                        angle = np.random.uniform(0, 2*np.pi)
                        xy[i, 0] -= 0.01 * np.cos(angle)
                        xy[i, 1] -= 0.01 * np.sin(angle)
                        xy[j, 0] += 0.01 * np.cos(angle)
                        xy[j, 1] += 0.01 * np.sin(angle)
                        moved = True

            for i in range(n):
                if xy[i, 0] < r[i]:
                    xy[i, 0] = r[i]
                    moved = True
                if xy[i, 0] > 1 - r[i]:
                    xy[i, 0] = 1 - r[i]
                    moved = True
                if xy[i, 1] < r[i]:
                    xy[i, 1] = r[i]
                    moved = True
                if xy[i, 1] > 1 - r[i]:
                    xy[i, 1] = 1 - r[i]
                    moved = True

            if not moved:
                break

        return np.concatenate([xy.flatten(), r])

    def epigraph_objective(params):
        """
        Epigraph formulation: introduce auxiliary variable t >= max contact margin.
        This decouples the max constraint from the objective for better conditioning.
        """
        n = N
        # params: [xy (2n), r (n), t (1)]
        xy = params[:2*n].reshape(n, 2)
        r = params[2*n:3*n]
        t = params[3*n]

        # Boundary constraints
        d_left = xy[:, 0] - r
        d_right = 1 - xy[:, 0] - r
        d_bottom = xy[:, 1] - r
        d_top = 1 - xy[:, 1] - r
        boundary_viol = np.sum(softplus(-d_left, scale=0.004)**2) + \
                        np.sum(softplus(-d_right, scale=0.004)**2) + \
                        np.sum(softplus(-d_bottom, scale=0.004)**2) + \
                        np.sum(softplus(-d_top, scale=0.004)**2)

        # Contact constraints via epigraph: penalize if any pair margin > t
        dx = xy[:, 0][:, None] - xy[:, 0][None, :]
        dy = xy[:, 1][:, None] - xy[:, 1][None, :]
        dist = np.sqrt(dx**2 + dy**2 + 1e-12)
        min_dist = r[:, None] + r[None, :]
        iu = np.triu_indices(n, k=1)
        
        contact_margin = min_dist[iu] - dist[iu]
        
        # Smooth max via log-sum-exp, penalize if exceeds t
        temp = 0.004
        lse = temp * np.log(np.sum(np.exp(contact_margin / temp)))
        epigraph_viol = softplus(lse - t, scale=0.002) ** 2
        
        # Also penalize t being too large (encourages tight packing)
        t_penalty = softplus(t, scale=0.008) ** 2 * 0.15
        
        # Direct overlap penalty for safety
        active_contact = softplus(contact_margin, scale=0.002)
        overlap_viol = np.sum(active_contact**2)

        return -np.sum(r) + 100 * boundary_viol + 100 * epigraph_viol + \
               10 * t_penalty + 50 * overlap_viol

    def refine_solution(params, max_iter=500, tol=1e-14, use_smooth=False):
        try:
            if use_smooth:
                obj = smooth_penalty_function
            else:
                # Use soft contact objective for better gradient flow
                obj = soft_contact_objective
            result = minimize(obj, params, method='L-BFGS-B',
                             options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-13})
            return result.x
        except Exception:
            return params

    def grow_radii(params, max_attempts=50):
        """Try to grow radii while maintaining feasibility."""
        n = N
        xy = params[:2*n].reshape(n, 2).copy()
        r = params[2*n:].copy()
        
        for attempt in range(max_attempts):
            # Try growing all radii by a small amount
            dr = 0.001 / (1 + attempt * 0.5)
            r_new = r + dr
            
            # Check if still feasible
            feasible = True
            for i in range(n):
                if r_new[i] < 1e-6 or r_new[i] > 0.5:
                    feasible = False
                    break
                if xy[i, 0] - r_new[i] < -1e-10 or 1 - xy[i, 0] - r_new[i] < -1e-10:
                    feasible = False
                    break
                if xy[i, 1] - r_new[i] < -1e-10 or 1 - xy[i, 1] - r_new[i] < -1e-10:
                    feasible = False
                    break
            
            if not feasible:
                break
            
            # Check pairwise distances
            for i in range(n):
                for j in range(i+1, n):
                    dx = xy[j, 0] - xy[i, 0]
                    dy = xy[j, 1] - xy[i, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < r_new[i] + r_new[j] - 1e-10:
                        feasible = False
                        break
                if not feasible:
                    break
            
            if not feasible:
                break
            
            r = r_new
        
        return np.concatenate([xy.flatten(), r])

    def refine_epigraph(params, max_iter=500, tol=1e-14):
        """Refine using epigraph formulation with auxiliary variable t."""
        try:
            n = N
            # Append initial t estimate: max contact margin
            xy = params[:2*n].reshape(n, 2)
            r = params[2*n:]
            dx = xy[:, 0][:, None] - xy[:, 0][None, :]
            dy = xy[:, 1][:, None] - xy[:, 1][None, :]
            dist = np.sqrt(dx**2 + dy**2 + 1e-12)
            min_dist = r[:, None] + r[None, :]
            iu = np.triu_indices(n, k=1)
            t_init = np.max(min_dist[iu] - dist[iu]) + 0.01
            t_init = max(t_init, 0.0)
            
            full_params = np.concatenate([params, [t_init]])
            result = minimize(epigraph_objective, full_params, method='L-BFGS-B',
                             options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-12})
            # Return only the original parameters (drop t)
            return result.x[:2*n+n]
        except Exception:
            return params

    def physics_refine(params, max_iter=200):
        phys_result = physics_optimize(params, max_iter=max_iter, alpha=100.0, beta=50.0)
        
        if is_valid(phys_result):
            # Try soft contact objective first for better local optimization
            polished_soft = refine_solution(phys_result, max_iter=300, tol=1e-13, use_smooth=False)
            if is_valid(polished_soft):
                # Then polish with epigraph formulation for better conditioning
                polished_epi = refine_epigraph(polished_soft, max_iter=200, tol=1e-14)
                if is_valid(polished_epi):
                    # Try to grow radii
                    grown = grow_radii(polished_epi, max_attempts=30)
                    if is_valid(grown) and get_score(grown) > get_score(polished_epi):
                        polished_epi = grown
                    # Final polish with smooth penalty
                    polished = refine_solution(polished_epi, max_iter=200, tol=1e-14, use_smooth=True)
                    if is_valid(polished):
                        # Try growing again after final polish
                        grown2 = grow_radii(polished, max_attempts=20)
                        if is_valid(grown2) and get_score(grown2) > get_score(polished):
                            return grown2
                        return polished
                    return polished_epi
                # Then polish with smooth penalty
                polished = refine_solution(polished_soft, max_iter=200, tol=1e-14, use_smooth=True)
                if is_valid(polished):
                    grown = grow_radii(polished, max_attempts=20)
                    if is_valid(grown) and get_score(grown) > get_score(polished):
                        return grown
                    return polished
                return polished_soft
            else:
                polished = refine_solution(phys_result, max_iter=300, tol=1e-13, use_smooth=True)
                if is_valid(polished):
                    grown = grow_radii(polished, max_attempts=20)
                    if is_valid(grown) and get_score(grown) > get_score(polished):
                        return grown
                    return polished
                return phys_result
        else:
            repaired = push_apart(phys_result, n_iters=80)
            if is_valid(repaired):
                polished = refine_solution(repaired, max_iter=200, tol=1e-13, use_smooth=True)
                if is_valid(polished):
                    return polished
            return params

    def get_initial_config(restart, rng):
        n = N
        side = int(np.ceil(np.sqrt(N)))

        if restart % 10 == 0:
            xy = np.zeros((n, 2))
            r = np.zeros(n)
            for i in range(n):
                row = i // side
                col = i % side
                xy[i] = [(col + 0.5)/side, (row + 0.5)/side]
                r[i] = 0.5/side - 0.001
        elif restart % 10 == 1:
            xy = np.zeros((n, 2))
            r = np.zeros(n)
            for i in range(n):
                row = i // side
                col = i % side
                xy[i] = [(col + 0.5)/side, (row + 0.5)/side]
                r[i] = 0.5/side - 0.001
            xy += rng.uniform(-0.02, 0.02, size=(n, 2))
        elif restart % 10 == 2:
            xy = np.zeros((n, 2))
            r = np.zeros(n)
            for i in range(n):
                row = i // 6
                col = i % 6
                x = (col + 0.5)/6 + (0.25 if row % 2 == 1 else 0)
                y = (row + 0.5)/6
                xy[i] = [x, y]
                r[i] = 0.08
        elif restart % 10 == 3:
            xy = rng.uniform(0.08, 0.92, size=(n, 2))
            r = rng.uniform(0.05, 0.1, size=n)
        elif restart % 10 == 4:
            xy = np.zeros((n, 2))
            r = np.zeros(n)
            for i in range(n):
                row = i // 8
                col = i % 8
                xy[i] = [(col + 0.5)/8, (row + 0.5)/4]
                r[i] = 0.5/8 - 0.001
            xy += rng.uniform(-0.01, 0.01, size=(n, 2))
        elif restart % 10 == 5:
            xy = np.zeros((n, 2))
            r = np.zeros(n)
            for i in range(n):
                row = i // 4
                col = i % 4
                xy[i] = [(col + 0.5)/4, (row + 0.5)/8]
                r[i] = 0.5/4 - 0.001
            xy += rng.uniform(-0.01, 0.01, size=(n, 2))
        elif restart % 10 == 6:
            xy = np.zeros((n, 2))
            r = np.zeros(n)
            for i in range(n):
                row = i // 6
                col = i % 6
                x = (col + 0.5)/6 + (0.2 if row % 2 == 0 else 0)
                y = (row + 0.5)/6
                xy[i] = [x, y]
                r[i] = 0.075
        elif restart % 10 == 7:
            xy = np.zeros((n, 2))
            r = np.zeros(n)
            centers = [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)]
            for i in range(n):
                group = i // 8
                cx, cy = centers[group]
                sub_i = i % 8
                sx = (sub_i % 3) / 3.0 + 0.5/3
                sy = (sub_i // 3) / 3.0 + 0.5/3
                xy[i] = [cx + (sx - 0.5)*0.4, cy + (sy - 0.5)*0.4]
                r[i] = 0.06
        elif restart % 10 == 8:
            xy = rng.uniform(0.08, 0.92, size=(n, 2))
            r = rng.uniform(0.06, 0.09, size=n)
        else:
            xy = rng.uniform(0.05, 0.95, size=(n, 2))
            r = rng.uniform(0.03, 0.06, size=n)

        params = np.concatenate([xy.flatten(), r])
        return params

    def voronoi_based_packing(rng):
        n = N
        side = 6
        xy = np.zeros((n, 2))
        r = np.zeros(n)
        
        for i in range(n):
            row = i // side
            col = i % side
            xy[i] = [(col + 0.5)/side, (row + 0.5)/side]
            r[i] = 0.08
        
        xy += rng.uniform(-0.03, 0.03, size=(n, 2))
        r += rng.uniform(-0.01, 0.01, size=n)
        r = np.clip(r, 0.03, 0.15)
        
        params = np.concatenate([xy.flatten(), r])
        params = push_apart(params, n_iters=80)
        
        refined = refine_solution(params, max_iter=400, tol=1e-14, use_smooth=True)
        
        if is_valid(refined):
            return refined
        else:
            repaired = push_apart(refined, n_iters=100)
            if is_valid(repaired):
                return repaired
            else:
                return params if is_valid(params) else None

    rng = np.random.RandomState(12345)

    n_restarts = 100
    for restart in range(n_restarts):
        if time.time() > deadline - 10:
            break

        if restart % 5 == 0:
            params = voronoi_based_packing(rng)
            if params is None:
                continue
        else:
            params = get_initial_config(restart, rng)
            params = push_apart(params, n_iters=50)

        if is_valid(params):
            score = get_score(params)
            if score > best_score:
                best_score = score
                best_result = params.copy()

        use_smooth = (restart % 2 == 0)
        refined = refine_solution(params, max_iter=400, tol=1e-13, use_smooth=use_smooth)
        
        if restart % 3 == 0 and time.time() < deadline - 5:
            phys_refined = physics_refine(params, max_iter=150)
            if is_valid(phys_refined):
                phys_score = get_score(phys_refined)
                if phys_score > best_score:
                    best_score = phys_score
                    best_result = phys_refined.copy()

        if is_valid(refined):
            score = get_score(refined)
            if score > best_score:
                best_score = score
                best_result = refined.copy()
        else:
            repaired = push_apart(refined, n_iters=80)
            if is_valid(repaired):
                score = get_score(repaired)
                if score > best_score:
                    best_score = score
                    best_result = repaired.copy()
            else:
                xy = refined[:2*N].reshape(N, 2).copy()
                r = refined[2*N:].copy()
                for _ in range(200):
                    if is_valid(np.concatenate([xy.flatten(), r])):
                        break
                    r *= 0.95
                final_params = np.concatenate([xy.flatten(), r])
                if is_valid(final_params):
                    score = get_score(final_params)
                    if score > best_score:
                        best_score = score
                        best_result = final_params.copy()

    if best_result is not None and time.time() < deadline - 5:
        # Multi-stage high-precision polish with soft contact graph
        # Stage 1: Soft contact objective for better gradient flow
        polished = refine_solution(best_result, max_iter=800, tol=1e-15, use_smooth=False)
        if is_valid(polished):
            score = get_score(polished)
            if score > best_score:
                best_score = score
                best_result = polished.copy()

        # Stage 2: Epigraph formulation for better conditioning of max constraint
        polished_epi = refine_epigraph(best_result, max_iter=800, tol=1e-15)
        if is_valid(polished_epi):
            score = get_score(polished_epi)
            if score > best_score:
                best_score = score
                best_result = polished_epi.copy()

        # Stage 3: Smooth penalty for final tightening
        polished_smooth = refine_solution(best_result, max_iter=800, tol=1e-15, use_smooth=True)
        if is_valid(polished_smooth):
            score = get_score(polished_smooth)
            if score > best_score:
                best_score = score
                best_result = polished_smooth.copy()

        # Stage 4: Try to grow radii
        grown = grow_radii(best_result, max_attempts=50)
        if is_valid(grown):
            score = get_score(grown)
            if score > best_score:
                best_score = score
                best_result = grown.copy()

        if time.time() < deadline - 15:
            # Physics-based refinement with soft contacts
            phys_polished = physics_refine(best_result, max_iter=300)
            if is_valid(phys_polished):
                phys_score = get_score(phys_polished)
                if phys_score > best_score:
                    best_score = phys_score
                    best_result = phys_polished.copy()
                # Follow up with epigraph on the physics result
                epi_polish = refine_epigraph(phys_polished, max_iter=500, tol=1e-16)
                if is_valid(epi_polish):
                    ep_score = get_score(epi_polish)
                    if ep_score > best_score:
                        best_score = ep_score
                        best_result = epi_polish.copy()
                # Follow up with L-BFGS-B on the best so far
                final_polish = refine_solution(best_result, max_iter=500, tol=1e-16, use_smooth=False)
                if is_valid(final_polish):
                    fp_score = get_score(final_polish)
                    if fp_score > best_score:
                        best_score = fp_score
                        best_result = final_polish.copy()
                # Final smooth polish
                final_smooth = refine_solution(best_result, max_iter=400, tol=1e-16, use_smooth=True)
                if is_valid(final_smooth):
                    fs_score = get_score(final_smooth)
                    if fs_score > best_score:
                        best_score = fs_score
                        best_result = final_smooth.copy()

        # Local perturbation search with tighter perturbations
        for polish_iter in range(20):
            if time.time() > deadline - 3:
                break
            perturbed = best_result.copy()
            xy_p = perturbed[:2*N].reshape(N, 2).copy()
            r_p = perturbed[2*N:].copy()
            # Smaller perturbations for later iterations
            pert_scale = 0.0015 * (1.0 - polish_iter / 25.0)
            pert_scale = max(pert_scale, 0.0002)
            xy_p += rng.uniform(-pert_scale, pert_scale, size=(N, 2))
            perturbed = np.concatenate([xy_p.flatten(), r_p])
            perturbed = push_apart(perturbed, n_iters=40)
            
            # Alternate between soft contact, epigraph, and physics
            if polish_iter % 3 == 0:
                polished2 = refine_solution(perturbed, max_iter=500, tol=1e-16, use_smooth=False)
            elif polish_iter % 3 == 1:
                polished2 = refine_epigraph(perturbed, max_iter=400, tol=1e-16)
            else:
                polished2 = physics_refine(perturbed, max_iter=200)
            
            if is_valid(polished2):
                # Try to grow radii
                grown2 = grow_radii(polished2, max_attempts=20)
                if is_valid(grown2):
                    polished2 = grown2
                score = get_score(polished2)
                if score > best_score:
                    best_score = score
                    best_result = polished2.copy()

    if best_result is not None:
        n = N
        xy = best_result[:2*n].reshape(n, 2).copy()
        r = best_result[2*n:].copy()
        r = np.clip(r, 0, 0.5)
        xy[:, 0] = np.clip(xy[:, 0], r, 1-r)
        xy[:, 1] = np.clip(xy[:, 1], r, 1-r)
        final_params = np.concatenate([xy.flatten(), r])
        if is_valid(final_params):
            # Final attempt to grow radii
            grown_final = grow_radii(final_params, max_attempts=30)
            if is_valid(grown_final) and get_score(grown_final) > get_score(final_params):
                xy_f = grown_final[:2*n].reshape(n, 2).copy()
                r_f = grown_final[2*n:].copy()
                return np.column_stack([xy_f, r_f])
            return np.column_stack([xy, r])
        # Fallback: try to repair with smaller radii
        for scale in [0.95, 0.9, 0.85, 0.8]:
            r2 = r * scale
            xy2 = xy.copy()
            xy2[:, 0] = np.clip(xy2[:, 0], r2, 1-r2)
            xy2[:, 1] = np.clip(xy2[:, 1], r2, 1-r2)
            fp2 = np.concatenate([xy2.flatten(), r2])
            if is_valid(fp2):
                return np.column_stack([xy2, r2])

    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    return np.array(cells[:N])
