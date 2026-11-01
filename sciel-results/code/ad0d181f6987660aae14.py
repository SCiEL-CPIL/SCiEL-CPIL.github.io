import numpy as np
from scipy.optimize import minimize
import time

N = 32

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit

    pair_i, pair_j = np.triu_indices(N, k=1)
    n_pairs = len(pair_i)

    def elapsed():
        return time.time() - start_time

    def time_left():
        return deadline - time.time()

    def is_valid(xy, r, tol=1e-9):
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
        if np.any(dist[pair_i, pair_j] < min_dist[pair_i, pair_j] - tol):
            return False
        return True

    def score(xy, r):
        return np.sum(r)

    def clip_to_box(xy, r):
        xy[:, 0] = np.clip(xy[:, 0], r, 1 - r)
        xy[:, 1] = np.clip(xy[:, 1], r, 1 - r)

    def push_apart(xy, r, n_iters=200):
        for _ in range(n_iters):
            moved = False
            for i in range(N):
                for j in range(i + 1, N):
                    dx = xy[j, 0] - xy[i, 0]
                    dy = xy[j, 1] - xy[i, 1]
                    dist = np.sqrt(dx**2 + dy**2)
                    min_dist = r[i] + r[j]
                    if dist < min_dist:
                        if dist < 1e-12:
                            angle = np.random.uniform(0, 2 * np.pi)
                            xy[i, 0] -= 0.01 * np.cos(angle)
                            xy[i, 1] -= 0.01 * np.sin(angle)
                            xy[j, 0] += 0.01 * np.cos(angle)
                            xy[j, 1] += 0.01 * np.sin(angle)
                        else:
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
            for i in range(N):
                if xy[i, 0] < r[i]:
                    xy[i, 0] = r[i]; moved = True
                if xy[i, 0] > 1 - r[i]:
                    xy[i, 0] = 1 - r[i]; moved = True
                if xy[i, 1] < r[i]:
                    xy[i, 1] = r[i]; moved = True
                if xy[i, 1] > 1 - r[i]:
                    xy[i, 1] = 1 - r[i]; moved = True
            if not moved:
                break
        clip_to_box(xy, r)
        return xy, r

    def sa_optimize(xy, r, n_steps=5000, T0=0.05, T1=0.001, momentum=0.5):
        """Simulated annealing with momentum for circle packing."""
        xy = xy.copy()
        r = r.copy()
        vx = np.zeros_like(xy)
        vy = np.zeros_like(xy)

        best_xy = xy.copy()
        best_r = r.copy()
        best_score = score(best_xy, best_r)

        for step in range(n_steps):
            T = T0 * (T1 / T0) ** (step / max(n_steps - 1, 1))

            # Compute forces
            fx = np.zeros(N)
            fy = np.zeros(N)

            # Pair forces
            dx = xy[pair_j, 0] - xy[pair_i, 0]
            dy = xy[pair_j, 1] - xy[pair_i, 1]
            d2 = dx * dx + dy * dy
            d = np.sqrt(np.maximum(d2, 1e-14))
            min_d = r[pair_i] + r[pair_j]
            overlap = min_d - d

            # Repulsive force when overlapping
            force = np.where(overlap > 0, 50.0 * overlap, 0.0)
            force = np.clip(force, 0, 50.0)

            fx[pair_i] -= force * dx / d
            fy[pair_i] -= force * dy / d
            fx[pair_j] += force * dx / d
            fy[pair_j] += force * dy / d

            # Boundary forces
            dl = xy[:, 0] - r
            dr = 1 - xy[:, 0] - r
            db = xy[:, 1] - r
            dt = 1 - xy[:, 1] - r
            fx += 20.0 * np.maximum(0, 0.02 - dl)
            fx -= 20.0 * np.maximum(0, 0.02 - dr)
            fy += 20.0 * np.maximum(0, 0.02 - db)
            fy -= 20.0 * np.maximum(0, 0.02 - dt)

            # Centering
            fx += 0.05 * (0.5 - xy[:, 0])
            fy += 0.05 * (0.5 - xy[:, 1])

            # Update with momentum
            vx = momentum * vx + 0.1 * fx
            vy = momentum * vy + 0.1 * fy

            # Random perturbation
            scale = T * 0.5
            xy_new = xy + vx + np.random.normal(0, scale, size=(N, 2))
            r_new = r + np.random.normal(0, scale * 0.5, size=N)
            r_new = np.clip(r_new, 0.01, 0.4)

            # Clip to box
            xy_new[:, 0] = np.clip(xy_new[:, 0], r_new, 1 - r_new)
            xy_new[:, 1] = np.clip(xy_new[:, 1], r_new, 1 - r_new)

            # Check validity and accept
            if is_valid(xy_new, r_new, tol=1e-8):
                s_new = score(xy_new, r_new)
                s_old = score(xy, r)
                if s_new >= s_old or np.random.random() < np.exp((s_new - s_old) / max(T, 1e-10)):
                    xy = xy_new
                    r = r_new
                    if s_new > best_score:
                        best_score = s_new
                        best_xy = xy.copy()
                        best_r = r.copy()
            else:
                # Try just moving positions, keeping radii
                xy_try = xy + vx + np.random.normal(0, scale, size=(N, 2))
                xy_try[:, 0] = np.clip(xy_try[:, 0], r, 1 - r)
                xy_try[:, 1] = np.clip(xy_try[:, 1], r, 1 - r)
                if is_valid(xy_try, r, tol=1e-8):
                    s_try = score(xy_try, r)
                    s_old = score(xy, r)
                    if s_try >= s_old or np.random.random() < np.exp((s_try - s_old) / max(T, 1e-10)):
                        xy = xy_try
                        if s_try > best_score:
                            best_score = s_try
                            best_xy = xy.copy()
                            best_r = r.copy()

            # Check time
            if step % 500 == 0 and elapsed() > deadline - 15:
                break

        return best_xy, best_r

    def lbfgs_polish(xy, r, max_iter=500):
        """Polish with L-BFGS-B using a smooth penalty function."""
        def objective(params):
            xy_p = params[:2*N].reshape(N, 2)
            r_p = params[2*N:]
            r_p = np.clip(r_p, 0.001, 0.4)

            # Boundary violations
            dl = xy_p[:, 0] - r_p
            dr = 1 - xy_p[:, 0] - r_p
            db = xy_p[:, 1] - r_p
            dt = 1 - xy_p[:, 1] - r_p
            bv = np.sum(np.maximum(0, -dl)**2) + np.sum(np.maximum(0, -dr)**2) + \
                 np.sum(np.maximum(0, -db)**2) + np.sum(np.maximum(0, -dt)**2)

            # Overlap violations
            dx = xy_p[:, 0][:, None] - xy_p[:, 0][None, :]
            dy = xy_p[:, 1][:, None] - xy_p[:, 1][None, :]
            dist = np.sqrt(dx**2 + dy**2)
            min_dist = r_p[:, None] + r_p[None, :]
            ov = np.maximum(0, min_dist[pair_i, pair_j] - dist[pair_i, pair_j])
            ov_v = np.sum(ov**2)

            return -np.sum(r_p) + 200 * bv + 200 * ov_v

        params = np.concatenate([xy.flatten(), r])
        try:
            res = minimize(objective, params, method='L-BFGS-B',
                           options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-12})
            xy_res = res.x[:2*N].reshape(N, 2)
            r_res = res.x[2*N:]
            if is_valid(xy_res, r_res, tol=1e-8):
                return xy_res, r_res
        except Exception:
            pass
        return xy, r

    def local_search(xy, r, n_attempts=20, base_scale=0.02):
        """Local search with adaptive perturbation."""
        best_xy = xy.copy()
        best_r = r.copy()
        best_s = score(best_xy, best_r)

        for attempt in range(n_attempts):
            if elapsed() > deadline - 10:
                break
            scale = base_scale * (1.0 - attempt / (n_attempts + 5))
            scale = max(scale, 0.002)

            xy_p = xy + np.random.normal(0, scale, size=(N, 2))
            r_p = r + np.random.normal(0, scale * 0.3, size=N)
            r_p = np.clip(r_p, 0.01, 0.4)
            clip_to_box(xy_p, r_p)

            if not is_valid(xy_p, r_p, tol=1e-8):
                xy_p, r_p = push_apart(xy_p, r_p, n_iters=50)

            if is_valid(xy_p, r_p, tol=1e-8):
                s_p = score(xy_p, r_p)
                if s_p > best_s:
                    best_s = s_p
                    best_xy = xy_p.copy()
                    best_r = r_p.copy()
                    xy = xy_p
                    r = r_p

        return best_xy, best_r

    def get_grid_init(rng, cols, rows, jitter=0.0):
        """Grid initialization with optional jitter."""
        xy = np.zeros((N, 2))
        r = np.zeros(N)
        for i in range(N):
            row = i // cols
            col = i % cols
            x = (col + 0.5) / cols
            y = (row + 0.5) / rows
            xy[i] = [x, y]
            r[i] = min(0.5 / cols, 0.5 / rows) - 0.005
        if jitter > 0:
            xy += rng.uniform(-jitter, jitter, size=(N, 2))
        clip_to_box(xy, r)
        return xy, r

    def get_random_init(rng, r_min=0.05, r_max=0.12):
        """Random initialization."""
        xy = rng.uniform(r_max, 1 - r_max, size=(N, 2))
        r = rng.uniform(r_min, r_max, size=N)
        clip_to_box(xy, r)
        return xy, r

    def get_cluster_init(rng):
        """Cluster-based initialization: 4 clusters of 8."""
        xy = np.zeros((N, 2))
        r = np.zeros(N)
        centers = [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)]
        for i in range(N):
            group = i // 8
            cx, cy = centers[group]
            sub = i % 8
            angle = sub * 2 * np.pi / 8
            rad = 0.08
            xy[i] = [cx + rad * np.cos(angle) + rng.uniform(-0.02, 0.02),
                     cy + rad * np.sin(angle) + rng.uniform(-0.02, 0.02)]
            r[i] = 0.06
        clip_to_box(xy, r)
        return xy, r

    def get_honeycomb_init(rng):
        """Honeycomb-like initialization."""
        xy = np.zeros((N, 2))
        r = np.zeros(N)
        # 6 rows of ~5-6 circles
        for i in range(N):
            row = i // 6
            col = i % 6
            x = (col + 0.5) / 6 + (0.5 / 6 if row % 2 == 1 else 0)
            y = (row + 0.5) / 6
            xy[i] = [x + rng.uniform(-0.01, 0.01), y + rng.uniform(-0.01, 0.01)]
            r[i] = 0.075
        clip_to_box(xy, r)
        return xy, r

    rng = np.random.RandomState(42)

    best_xy = None
    best_r = None
    best_score = -np.inf

    # Phase 1: Multiple diverse restarts with SA
    n_restarts = 30
    for restart in range(n_restarts):
        if time_left() < 20:
            break

        # Choose initialization strategy
        strategy = restart % 5
        if strategy == 0:
            xy, r = get_grid_init(rng, 6, 6, jitter=0.02)
        elif strategy == 1:
            xy, r = get_random_init(rng, 0.05, 0.10)
        elif strategy == 2:
            xy, r = get_cluster_init(rng)
        elif strategy == 3:
            xy, r = get_honeycomb_init(rng)
        else:
            xy, r = get_grid_init(rng, 8, 4, jitter=0.01)

        # Push apart to fix initial overlaps
        xy, r = push_apart(xy, r, n_iters=100)

        if not is_valid(xy, r, tol=1e-8):
            continue

        s_init = score(xy, r)
        if s_init > best_score:
            best_score = s_init
            best_xy = xy.copy()
            best_r = r.copy()

        # SA optimization
        n_sa_steps = 3000 if time_left() > 60 else 1500
        xy_sa, r_sa = sa_optimize(xy, r, n_steps=n_sa_steps, T0=0.04, T1=0.002)

        if is_valid(xy_sa, r_sa, tol=1e-8):
            s_sa = score(xy_sa, r_sa)
            if s_sa > best_score:
                best_score = s_sa
                best_xy = xy_sa.copy()
                best_r = r_sa.copy()

        # L-BFGS polish
        if time_left() > 15:
            xy_pol, r_pol = lbfgs_polish(xy_sa, r_sa, max_iter=300)
            if is_valid(xy_pol, r_pol, tol=1e-8):
                s_pol = score(xy_pol, r_pol)
                if s_pol > best_score:
                    best_score = s_pol
                    best_xy = xy_pol.copy()
                    best_r = r_pol.copy()

    # Phase 2: Intensive local search around best
    if best_xy is not None and time_left() > 15:
        xy_ls, r_ls = local_search(best_xy, best_r, n_attempts=30, base_scale=0.015)
        if is_valid(xy_ls, r_ls, tol=1e-8):
            s_ls = score(xy_ls, r_ls)
            if s_ls > best_score:
                best_score = s_ls
                best_xy = xy_ls.copy()
                best_r = r_ls.copy()

        # SA from best
        if time_left() > 20:
            xy_sa2, r_sa2 = sa_optimize(best_xy, best_r, n_steps=4000, T0=0.02, T1=0.001)
            if is_valid(xy_sa2, r_sa2, tol=1e-8):
                s_sa2 = score(xy_sa2, r_sa2)
                if s_sa2 > best_score:
                    best_score = s_sa2
                    best_xy = xy_sa2.copy()
                    best_r = r_sa2.copy()

        # Final L-BFGS polish
        if time_left() > 10:
            xy_fin, r_fin = lbfgs_polish(best_xy, best_r, max_iter=500)
            if is_valid(xy_fin, r_fin, tol=1e-8):
                s_fin = score(xy_fin, r_fin)
                if s_fin > best_score:
                    best_score = s_fin
                    best_xy = xy_fin.copy()
                    best_r = r_fin.copy()

    # Phase 3: More local search with smaller perturbations
    if best_xy is not None and time_left() > 10:
        xy_ls2, r_ls2 = local_search(best_xy, best_r, n_attempts=40, base_scale=0.008)
        if is_valid(xy_ls2, r_ls2, tol=1e-8):
            s_ls2 = score(xy_ls2, r_ls2)
            if s_ls2 > best_score:
                best_score = s_ls2
                best_xy = xy_ls2.copy()
                best_r = r_ls2.copy()

        # Final polish
        if time_left() > 8:
            xy_fin2, r_fin2 = lbfgs_polish(best_xy, best_r, max_iter=400)
            if is_valid(xy_fin2, r_fin2, tol=1e-8):
                s_fin2 = score(xy_fin2, r_fin2)
                if s_fin2 > best_score:
                    best_score = s_fin2
                    best_xy = xy_fin2.copy()
                    best_r = r_fin2.copy()

    # Final validity check and return
    if best_xy is not None:
        clip_to_box(best_xy, best_r)
        if is_valid(best_xy, best_r, tol=1e-9):
            return np.column_stack([best_xy, best_r])

        # Try to repair
        best_xy_rep, best_r_rep = push_apart(best_xy, best_r, n_iters=200)
        if is_valid(best_xy_rep, best_r_rep, tol=1e-9):
            return np.column_stack([best_xy_rep, best_r_rep])

        # Shrink radii
        for scale in [0.95, 0.9, 0.85, 0.8, 0.7]:
            r_shrunk = best_r * scale
            clip_to_box(best_xy, r_shrunk)
            if is_valid(best_xy, r_shrunk, tol=1e-9):
                return np.column_stack([best_xy, r_shrunk])

    # Fallback: simple grid
    xy = np.zeros((N, 2))
    r = np.zeros(N)
    for i in range(N):
        row = i // 6
        col = i % 6
        xy[i] = [(col + 0.5) / 6, (row + 0.5) / 6]
        r[i] = 0.5 / 6 - 0.005
    return np.column_stack([xy, r])
