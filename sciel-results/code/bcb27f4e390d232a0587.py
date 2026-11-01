import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay
import time

N = 32

def construct():
    start_time = time.time()
    deadline = start_time + 138  # safety margin before 150s kill

    def elapsed():
        return time.time() - start_time

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

    def penalty_function(params, weight=100.0):
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

        return -np.sum(r) + weight * boundary_viol + weight * overlap_viol

    def smooth_penalty(params, weight=100.0, eps=1e-10):
        n = N
        xy = params[:2*n].reshape(n, 2)
        r = params[2*n:]

        def smax2(a, b):
            d = a - b
            sm = 0.5 * (d + np.sqrt(d**2 + eps))
            return np.sum(sm**2)

        bv = smax2(r, xy[:, 0]) + smax2(r, 1 - xy[:, 0]) + smax2(r, xy[:, 1]) + smax2(r, 1 - xy[:, 1])

        dx = xy[:, 0][:, None] - xy[:, 0][None, :]
        dy = xy[:, 1][:, None] - xy[:, 1][None, :]
        dist = np.sqrt(dx**2 + dy**2)
        min_dist = r[:, None] + r[None, :]
        iu = np.triu_indices(n, k=1)
        d = min_dist[iu] - dist[iu]
        sm = 0.5 * (d + np.sqrt(d**2 + eps))
        ov = np.sum(sm**2)

        return -np.sum(r) + weight * bv + weight * ov

    def push_apart(params, n_iters=200):
        n = N
        xy = params[:2*n].reshape(n, 2).copy()
        r = params[2*n:].copy()

        for _ in range(n_iters):
            moved = False
            for i in range(n):
                for j in range(i+1, n):
                    dx = xy[j, 0] - xy[i, 0]
                    dy = xy[j, 1] - xy[i, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    min_d = r[i] + r[j]
                    if dist < min_d and dist > 1e-12:
                        overlap = min_d - dist
                        wi = r[i] / (r[i] + r[j]) if r[i] + r[j] > 1e-15 else 0.5
                        wj = 1 - wi
                        px = (dx / dist) * overlap
                        py = (dy / dist) * overlap
                        xy[i, 0] -= px * wi
                        xy[i, 1] -= py * wi
                        xy[j, 0] += px * wj
                        xy[j, 1] += py * wj
                        moved = True
                    elif dist <= 1e-12:
                        ang = np.random.uniform(0, 2*np.pi)
                        xy[i, 0] -= 0.01 * np.cos(ang)
                        xy[i, 1] -= 0.01 * np.sin(ang)
                        xy[j, 0] += 0.01 * np.cos(ang)
                        xy[j, 1] += 0.01 * np.sin(ang)
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

    def refine(params, max_iter=500, tol=1e-14, smooth=False, weight=100.0):
        try:
            obj = smooth_penalty if smooth else penalty_function
            if smooth:
                res = minimize(lambda p: obj(p, weight=weight), params, method='L-BFGS-B',
                               options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-12})
            else:
                res = minimize(lambda p: obj(p, weight=weight), params, method='L-BFGS-B',
                               options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-12})
            return res.x
        except Exception:
            return params

    def voronoi_inradii(xy, n):
        """Compute inradii of Voronoi cells for points xy inside unit square."""
        # Add boundary points to constrain Voronoi
        boundary_pts = []
        m = 20
        for k in range(m):
            t = (k + 0.5) / m
            boundary_pts.append([t, 0])
            boundary_pts.append([t, 1])
            boundary_pts.append([0, t])
            boundary_pts.append([1, t])
        all_pts = np.vstack([xy, np.array(boundary_pts)])
        
        try:
            tri = Delaunay(all_pts)
        except Exception:
            return np.full(n, 0.05)
        
        inradii = np.zeros(n)
        for i in range(n):
            # Find triangles containing point i
            # Use a simpler approach: find nearest neighbors and compute minimum distance
            dists = np.sqrt((xy[:, 0] - xy[i, 0])**2 + (xy[:, 1] - xy[i, 1])**2)
            dists[i] = np.inf
            min_nn = np.min(dists)
            
            # Distance to boundary
            d_bound = min(xy[i, 0], 1 - xy[i, 0], xy[i, 1], 1 - xy[i, 1])
            
            inradii[i] = 0.5 * min(min_nn, 2 * d_bound)
        
        return inradii

    def voronoi_optimize(xy, r, max_iter=50, rng=None):
        """Iteratively adjust radii based on Voronoi inradii and optimize positions."""
        n = N
        xy = xy.copy()
        r = r.copy()
        
        for it in range(max_iter):
            if elapsed() > deadline - 5:
                break
            inr = voronoi_inradii(xy, n)
            # Set radii to a fraction of inradii
            r_new = np.clip(inr * 0.95, 0.01, 0.5)
            # If radii decrease too much, keep old
            r = np.maximum(r_new, r * 0.9)
            
            params = np.concatenate([xy.flatten(), r])
            params = push_apart(params, n_iters=30)
            xy = params[:2*n].reshape(n, 2)
            r = params[2*n:]
        
        return np.concatenate([xy.flatten(), r])

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

        return np.concatenate([xy.flatten(), r])

    rng = np.random.RandomState(12345)

    best_result = None
    best_score = -np.inf

    def update_best(params):
        nonlocal best_result, best_score
        if is_valid(params):
            score = get_score(params)
            if score > best_score:
                best_score = score
                best_result = params.copy()

    # Phase 1: Diverse restarts
    n_restarts = 60
    for restart in range(n_restarts):
        if elapsed() > deadline - 20:
            break

        params = get_initial_config(restart, rng)
        params = push_apart(params, n_iters=60)
        update_best(params)

        # Voronoi-based refinement for some restarts
        if restart % 3 == 0 and elapsed() < deadline - 15:
            xy = params[:2*N].reshape(N, 2)
            r = params[2*N:]
            v_params = voronoi_optimize(xy, r, max_iter=30, rng=rng)
            update_best(v_params)
            params = v_params

        # L-BFGS-B refinement
        if elapsed() < deadline - 10:
            smooth = (restart % 2 == 0)
            refined = refine(params, max_iter=300, tol=1e-13, smooth=smooth)
            update_best(refined)
            if not is_valid(refined):
                repaired = push_apart(refined, n_iters=60)
                update_best(repaired)
                params = repaired
            else:
                params = refined

    # Phase 2: Intensive polish on best
    if best_result is not None and elapsed() < deadline - 15:
        # Multiple polish passes with different settings
        for polish in range(12):
            if elapsed() > deadline - 5:
                break
            
            # Alternate between sharp and smooth
            smooth = (polish % 2 == 0)
            # Increase weight for tighter constraints
            w = 100.0 + polish * 50.0
            
            polished = refine(best_result, max_iter=600, tol=1e-15, smooth=smooth, weight=w)
            if is_valid(polished):
                score = get_score(polished)
                if score > best_score:
                    best_score = score
                    best_result = polished.copy()

            # Perturb and re-refine
            if elapsed() < deadline - 5:
                perturbed = best_result.copy()
                xy_p = perturbed[:2*N].reshape(N, 2).copy()
                r_p = perturbed[2*N:].copy()
                scale = 0.005 / (1 + polish * 0.3)
                xy_p += rng.uniform(-scale, scale, size=(N, 2))
                perturbed = np.concatenate([xy_p.flatten(), r_p])
                perturbed = push_apart(perturbed, n_iters=40)
                polished2 = refine(perturbed, max_iter=400, tol=1e-15, smooth=(polish % 2 == 1))
                if is_valid(polished2):
                    score = get_score(polished2)
                    if score > best_score:
                        best_score = score
                        best_result = polished2.copy()

    # Phase 3: Final high-precision polish
    if best_result is not None and elapsed() < deadline - 3:
        final = refine(best_result, max_iter=1000, tol=1e-16, smooth=False, weight=500.0)
        if is_valid(final):
            score = get_score(final)
            if score > best_score:
                best_score = score
                best_result = final.copy()

        final_s = refine(best_result, max_iter=1000, tol=1e-16, smooth=True, weight=500.0)
        if is_valid(final_s):
            score = get_score(final_s)
            if score > best_score:
                best_score = score
                best_result = final_s.copy()

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
