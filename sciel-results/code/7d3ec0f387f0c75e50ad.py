import numpy as np
from scipy.optimize import minimize
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s limit

    def calc_score(x):
        """Compute the packing radius for given points."""
        if len(x) != N:
            return 0.0
        if not np.all(np.isfinite(x)):
            return 0.0
        norms_sq = np.sum(x * x, axis=1)
        if np.any(norms_sq > 1.0):
            return 0.0
        norms = np.sqrt(norms_sq)
        b_min = np.min(1.0 - norms)
        # Use squared distances to avoid sqrt
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        d_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(d_sq, np.inf)
        min_pair = np.sqrt(np.min(d_sq)) / 2.0
        return min(min_pair, b_min)

    def hexagonal_init(spacing):
        """Generate points on a hexagonal lattice clipped to unit circle, pick N closest to origin."""
        points = []
        k = int(1.0 / spacing) + 3
        for i in range(-k, k + 1):
            for j in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if x * x + y * y <= 1.0:
                    points.append([x, y])
        points = np.array(points)
        if len(points) < N:
            return None
        dists = np.linalg.norm(points, axis=1)
        indices = np.argsort(dists)[:N]
        return points[indices]

    def ring_init(n_rings, r_start, r_end):
        """Initialize with concentric rings."""
        pts = [[0.0, 0.0]]
        for ring in range(1, n_rings + 1):
            r = r_start + (r_end - r_start) * (ring - 1) / max(n_rings - 1, 1)
            if ring == 1:
                n_pts = 6
            else:
                n_pts = ring * 6
            n_pts = min(n_pts, N - len(pts))
            if n_pts <= 0:
                break
            angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
            for a in angles:
                pts.append([r * np.cos(a), r * np.sin(a)])
        pts = np.array(pts[:N])
        if len(pts) < N:
            return None
        return pts

    # Generate diverse initializations
    candidates = []

    # Hexagonal grids with various spacings
    for s in np.linspace(0.09, 0.26, 25):
        pts = hexagonal_init(s)
        if pts is not None:
            candidates.append(pts)

    # Ring-based inits
    for n_rings in [5, 6, 7, 8, 9]:
        for r_start in [0.05, 0.08, 0.1, 0.12, 0.15]:
            pts = ring_init(n_rings, r_start, 0.92)
            if pts is not None:
                candidates.append(pts)

    # Random inits
    for _ in range(15):
        pts = np.random.rand(N, 2) * 2 - 1
        norms_sq = np.sum(pts * pts, axis=1)
        norms = np.sqrt(norms_sq)
        pts = pts / norms[:, None] * 0.95
        candidates.append(pts)

    # Evaluate all candidates and pick the best few
    scored_candidates = []
    for c in candidates:
        r = calc_score(c)
        scored_candidates.append((r, c))
    scored_candidates.sort(key=lambda x: -x[0])

    # Take top candidates for optimization
    top_candidates = [c for r, c in scored_candidates[:6]]
    if not top_candidates:
        top_candidates = [np.random.rand(N, 2) * 0.8]

    best_x = top_candidates[0].copy()
    best_score = calc_score(best_x)

    # Optimization using soft-min formulation with L-BFGS-B
    def make_obj_func(beta_val):
        """Create objective function with given beta for soft-min."""
        def obj_func(flat_x):
            x = flat_x.reshape(N, 2)
            diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
            d_sq = np.sum(diff * diff, axis=2)
            np.fill_diagonal(d_sq, np.inf)
            d = np.sqrt(d_sq)
            norms_sq = np.sum(x * x, axis=1)
            norms = np.sqrt(norms_sq)
            dist_boundary = 1.0 - norms

            iu = np.triu_indices(N, k=1)
            d_pairs = d[iu]
            d_bnds = dist_boundary
            all_d = np.concatenate([d_pairs, d_bnds])

            exponents = -beta_val * all_d
            max_exp = np.max(exponents)

            if not np.isfinite(max_exp):
                return 100.0, np.zeros_like(flat_x)

            sum_exp = np.sum(np.exp(exponents - max_exp))
            log_sum = max_exp + np.log(sum_exp)
            obj = log_sum / beta_val

            weights = np.exp(exponents - max_exp) / sum_exp
            grad = np.zeros_like(x)

            n_pairs = len(d_pairs)
            w_pairs = weights[:n_pairs]
            w_bnds = weights[n_pairs:]

            i_idx, j_idx = iu
            d_safe = np.where(d_pairs > 1e-10, d_pairs, 1e-10)
            vecs = x[i_idx] - x[j_idx]
            norm_vecs = vecs / d_safe[:, np.newaxis]

            np.add.at(grad, i_idx, -w_pairs[:, np.newaxis] * norm_vecs)
            np.add.at(grad, j_idx, w_pairs[:, np.newaxis] * norm_vecs)

            norms_safe = np.where(norms > 1e-10, norms, 1e-10)
            norm_x = x / norms_safe[:, np.newaxis]

            np.add.at(grad, np.arange(N), w_bnds[:, np.newaxis] * norm_x)

            return obj, grad.flatten()
        return obj_func

    # Annealing schedule: start with low beta (smooth), increase for sharper min
    betas = [8, 20, 40, 80, 150, 250, 400, 600]

    for cand_idx, current_x in enumerate(top_candidates):
        if time.time() > deadline - 15:
            break

        current_x = current_x.copy()
        local_best_x = current_x.copy()
        local_best_score = calc_score(current_x)

        for beta in betas:
            if time.time() > deadline - 15:
                break

            obj_func = make_obj_func(beta)
            try:
                res = minimize(obj_func, current_x.flatten(), method='L-BFGS-B', jac=True,
                               options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10})
                if res.x is not None:
                    new_x = res.x.reshape(N, 2)
                    new_score = calc_score(new_x)
                    if new_score > local_best_score:
                        local_best_score = new_score
                        local_best_x = new_x
                    current_x = new_x
            except Exception:
                pass

        if local_best_score > best_score:
            best_score = local_best_score
            best_x = local_best_x

    # Final polish with Nelder-Mead on the best solution
    if time.time() < deadline - 20:
        def nm_obj(flat_x):
            x = flat_x.reshape(N, 2)
            diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
            d_sq = np.sum(diff * diff, axis=2)
            np.fill_diagonal(d_sq, np.inf)
            min_pair = np.sqrt(np.min(d_sq)) / 2.0
            norms_sq = np.sum(x * x, axis=1)
            min_bound = np.min(1.0 - np.sqrt(norms_sq))
            return -min(min_pair, min_bound)

        try:
            res_nm = minimize(nm_obj, best_x.flatten(), method='Nelder-Mead',
                              options={'maxiter': 300, 'xatol': 1e-12, 'fatol': 1e-12})
            if res_nm.x is not None:
                final_x = res_nm.x.reshape(N, 2)
                final_score = calc_score(final_x)
                if final_score > best_score:
                    best_x = final_x
                    best_score = final_score
        except Exception:
            pass

    # One more L-BFGS pass with high beta if time permits
    if time.time() < deadline - 8:
        try:
            obj_func = make_obj_func(800)
            res = minimize(obj_func, best_x.flatten(), method='L-BFGS-B', jac=True,
                           options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10})
            if res.x is not None:
                final_x = res.x.reshape(N, 2)
                final_score = calc_score(final_x)
                if final_score > best_score:
                    best_x = final_x
                    best_score = final_score
        except Exception:
            pass

    # Final very high beta polish
    if time.time() < deadline - 3:
        try:
            obj_func = make_obj_func(1500)
            res = minimize(obj_func, best_x.flatten(), method='L-BFGS-B', jac=True,
                           options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10})
            if res.x is not None:
                final_x = res.x.reshape(N, 2)
                final_score = calc_score(final_x)
                if final_score > best_score:
                    best_x = final_x
                    best_score = final_score
        except Exception:
            pass

    return best_x
