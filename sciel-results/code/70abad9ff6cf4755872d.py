import numpy as np
from scipy.optimize import minimize
import time

N = 32

def construct():
    start_time = time.time()
    deadline = start_time + 135  # 135s out of 150s, safety margin

    best_result = None
    best_score = -np.inf

    def penalty_function(params):
        n = N
        xy = params[:2*n].reshape(n, 2)
        r = params[2*n:]

        # Boundary penalty
        d_left = xy[:, 0] - r
        d_right = 1 - xy[:, 0] - r
        d_bottom = xy[:, 1] - r
        d_top = 1 - xy[:, 1] - r
        boundary_viol = np.sum(np.maximum(0, -d_left)**2) + np.sum(np.maximum(0, -d_right)**2) + \
                        np.sum(np.maximum(0, -d_bottom)**2) + np.sum(np.maximum(0, -d_top)**2)

        # Overlap penalty
        dx = xy[:, 0][:, None] - xy[:, 0][None, :]
        dy = xy[:, 1][:, None] - xy[:, 1][None, :]
        dist = np.sqrt(dx**2 + dy**2)
        min_dist = r[:, None] + r[None, :]
        iu = np.triu_indices(n, k=1)
        overlap = np.maximum(0, min_dist[iu] - dist[iu])
        overlap_viol = np.sum(overlap**2)

        return -np.sum(r) + 100 * boundary_viol + 100 * overlap_viol

    def smooth_penalty_function(params):
        """
        Smooth surrogate using log-sum-exp for the penalty terms.
        This avoids the kink at zero violation and may help the optimizer
        find better local minima.
        """
        n = N
        xy = params[:2*n].reshape(n, 2)
        r = params[2*n:]

        # Boundary violations (smooth)
        d_left = xy[:, 0] - r
        d_right = 1 - xy[:, 0] - r
        d_bottom = xy[:, 1] - r
        d_top = 1 - xy[:, 1] - r

        # Use soft penalty: -log(1 + exp(-k * d)) approximates max(0, -d) smoothly
        # But for minimization, we want to penalize violations.
        # Let's use a smooth approximation of max(0, -d)^2
        # Actually, let's stick with the quadratic but use a smooth version of the max
        # For boundary: penalty = sum(softplus(-d)^2) where softplus(x) = log(1+exp(x))
        # But that's expensive. Let's try a different approach.
        
        # Simple smooth quadratic: for each violation v = max(0, -d), use v^2
        # The kink is in the max. We can smooth it with:
        # smooth_max(0, x) = 0.5 * (x + sqrt(x^2 + eps))
        # Then square it.
        
        eps = 1e-8
        
        def smooth_max_square(a, b):
            """Smooth approximation of max(0, a-b)^2"""
            diff = a - b
            sm = 0.5 * (diff + np.sqrt(diff**2 + eps))
            return np.sum(sm**2)

        boundary_viol = smooth_max_square(r, xy[:, 0]) + \
                        smooth_max_square(r, 1 - xy[:, 0]) + \
                        smooth_max_square(r, xy[:, 1]) + \
                        smooth_max_square(r, 1 - xy[:, 1])

        # Overlap penalty (smooth)
        dx = xy[:, 0][:, None] - xy[:, 0][None, :]
        dy = xy[:, 1][:, None] - xy[:, 1][None, :]
        dist = np.sqrt(dx**2 + dy**2)
        min_dist = r[:, None] + r[None, :]
        iu = np.triu_indices(n, k=1)
        
        # Smooth max(0, min_dist - dist)^2
        diff = min_dist[iu] - dist[iu]
        sm = 0.5 * (diff + np.sqrt(diff**2 + eps))
        overlap_viol = np.sum(sm**2)

        return -np.sum(r) + 100 * boundary_viol + 100 * overlap_viol

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

    def refine_solution(params, max_iter=500, tol=1e-14, use_smooth=False):
        try:
            obj = smooth_penalty_function if use_smooth else penalty_function
            result = minimize(obj, params, method='L-BFGS-B',
                             options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-12})
            return result.x
        except Exception:
            return params

    def get_initial_config(restart, rng):
        n = N
        side = int(np.ceil(np.sqrt(N)))

        if restart % 10 == 0:
            # Grid packing
            xy = np.zeros((n, 2))
            r = np.zeros(n)
            for i in range(n):
                row = i // side
                col = i % side
                xy[i] = [(col + 0.5)/side, (row + 0.5)/side]
                r[i] = 0.5/side - 0.001
        elif restart % 10 == 1:
            # Perturbed grid
            xy = np.zeros((n, 2))
            r = np.zeros(n)
            for i in range(n):
                row = i // side
                col = i % side
                xy[i] = [(col + 0.5)/side, (row + 0.5)/side]
                r[i] = 0.5/side - 0.001
            xy += rng.uniform(-0.02, 0.02, size=(n, 2))
        elif restart % 10 == 2:
            # Hexagonal packing
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
            # Random uniform
            xy = rng.uniform(0.08, 0.92, size=(n, 2))
            r = rng.uniform(0.05, 0.1, size=n)
        elif restart % 10 == 4:
            # 8x4 grid
            xy = np.zeros((n, 2))
            r = np.zeros(n)
            for i in range(n):
                row = i // 8
                col = i % 8
                xy[i] = [(col + 0.5)/8, (row + 0.5)/4]
                r[i] = 0.5/8 - 0.001
            xy += rng.uniform(-0.01, 0.01, size=(n, 2))
        elif restart % 10 == 5:
            # 4x8 grid
            xy = np.zeros((n, 2))
            r = np.zeros(n)
            for i in range(n):
                row = i // 4
                col = i % 4
                xy[i] = [(col + 0.5)/4, (row + 0.5)/8]
                r[i] = 0.5/4 - 0.001
            xy += rng.uniform(-0.01, 0.01, size=(n, 2))
        elif restart % 10 == 6:
            # Hexagonal variant
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
            # Clustered: 4 groups of 8
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
            # Random medium
            xy = rng.uniform(0.08, 0.92, size=(n, 2))
            r = rng.uniform(0.06, 0.09, size=n)
        else:
            # Random small
            xy = rng.uniform(0.05, 0.95, size=(n, 2))
            r = rng.uniform(0.03, 0.06, size=n)

        params = np.concatenate([xy.flatten(), r])
        return params

    def voronoi_based_packing(rng):
        """
        Use a Voronoi-like approach: start with a grid, then iteratively
        adjust centers and radii based on local Voronoi cell inradii.
        This ensures natural separation.
        """
        n = N
        # Start with a 6x6 grid (36 cells, use 32)
        side = 6
        xy = np.zeros((n, 2))
        r = np.zeros(n)
        
        for i in range(n):
            row = i // side
            col = i % side
            xy[i] = [(col + 0.5)/side, (row + 0.5)/side]
            r[i] = 0.08
        
        # Add some randomness
        xy += rng.uniform(-0.03, 0.03, size=(n, 2))
        r += rng.uniform(-0.01, 0.01, size=n)
        r = np.clip(r, 0.03, 0.15)
        
        params = np.concatenate([xy.flatten(), r])
        params = push_apart(params, n_iters=80)
        
        # Refine with smooth penalty
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

    # Phase 1: Many restarts with diverse initializations
    n_restarts = 80
    for restart in range(n_restarts):
        if time.time() > deadline:
            break

        # Use Voronoi-based approach for some restarts
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

        # Refine with L-BFGS-B, alternating between sharp and smooth penalty
        use_smooth = (restart % 2 == 0)
        refined = refine_solution(params, max_iter=300, tol=1e-13, use_smooth=use_smooth)

        if is_valid(refined):
            score = get_score(refined)
            if score > best_score:
                best_score = score
                best_result = refined.copy()
        else:
            # Try to repair
            repaired = push_apart(refined, n_iters=80)
            if is_valid(repaired):
                score = get_score(repaired)
                if score > best_score:
                    best_score = score
                    best_result = repaired.copy()
            else:
                # Shrink radii until valid
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

    # Phase 2: High-precision polish on best solution
    if best_result is not None and time.time() < deadline - 5:
        # Polish with sharp penalty
        polished = refine_solution(best_result, max_iter=800, tol=1e-15, use_smooth=False)
        if is_valid(polished):
            score = get_score(polished)
            if score > best_score:
                best_score = score
                best_result = polished.copy()

        # Polish with smooth penalty
        polished_smooth = refine_solution(best_result, max_iter=800, tol=1e-15, use_smooth=True)
        if is_valid(polished_smooth):
            score = get_score(polished_smooth)
            if score > best_score:
                best_score = score
                best_result = polished_smooth.copy()

        # Try perturbations and re-refine
        for polish_iter in range(10):
            if time.time() > deadline:
                break
            perturbed = best_result.copy()
            xy_p = perturbed[:2*N].reshape(N, 2).copy()
            r_p = perturbed[2*N:].copy()
            xy_p += rng.uniform(-0.003, 0.003, size=(N, 2))
            perturbed = np.concatenate([xy_p.flatten(), r_p])
            perturbed = push_apart(perturbed, n_iters=40)
            use_smooth = (polish_iter % 2 == 0)
            polished2 = refine_solution(perturbed, max_iter=400, tol=1e-15, use_smooth=use_smooth)
            if is_valid(polished2):
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
            return np.column_stack([xy, r])

    # Fallback: grid
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    return np.array(cells[:N])
