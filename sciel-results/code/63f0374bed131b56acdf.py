import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 50
DEADLINE = 132.0  # wall-clock safety margin before the 150s kill


# ---------------------------------------------------------------------------
# Scoring / smooth objective
# ---------------------------------------------------------------------------
def _pairwise_distances(pts):
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=-1))


def _score(pts):
    if pts is None or len(pts) != N:
        return -1.0
    pts = np.asarray(pts, dtype=float)
    if not np.all(np.isfinite(pts)):
        return -1.0
    d = _pairwise_distances(pts)
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d) / 2.0
    min_boundary = np.min(1.0 - np.sqrt(np.sum(pts ** 2, axis=1)))
    val = min(min_pair, min_boundary)
    if val <= 0:
        return -1.0
    return val


def _softmin_logsumexp(values, beta):
    """Numerically-stable soft-min via log-sum-exp."""
    if len(values) == 0:
        return 0.0
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return 0.0
    neg = -values * beta
    neg_max = np.max(neg)
    if not np.isfinite(neg_max):
        return 0.0
    exp_sum = np.sum(np.exp(neg - neg_max))
    if exp_sum <= 0.0:
        return 0.0
    return -(neg_max + np.log(exp_sum)) / beta


def _smooth_objective(x, beta):
    pts = x.reshape(N, 2)
    if not np.all(np.isfinite(pts)):
        return 1e10
    d = _pairwise_distances(pts)
    np.fill_diagonal(d, np.inf)
    boundary = 1.0 - np.sqrt(np.sum(pts ** 2, axis=1))
    pair_half = d / 2.0
    all_vals = np.concatenate([pair_half.ravel(), boundary])
    all_vals = all_vals[np.isfinite(all_vals)]
    if len(all_vals) == 0:
        return 1e10
    return -_softmin_logsumexp(all_vals, beta)


# ---------------------------------------------------------------------------
# Ring helpers
# ---------------------------------------------------------------------------
def _ring_to_points(radii, angles_list):
    pts = [np.array([0.0, 0.0])]
    for i in range(1, len(radii)):
        r = radii[i]
        a = angles_list[i]
        pts.append(np.column_stack([r * np.cos(a), r * np.sin(a)]))
    return np.vstack(pts)


def _flatten_ring(radii, angles_list):
    vec = list(radii)
    for i in range(1, len(angles_list)):
        vec.extend(list(angles_list[i]))
    return np.array(vec)


def _unflatten_ring(vec, ring_sizes):
    radii = vec[:len(ring_sizes)].copy()
    angles_list = [np.array([0.0])]
    idx = len(ring_sizes)
    for i in range(1, len(ring_sizes)):
        n = ring_sizes[i]
        angles_list.append(vec[idx:idx + n].copy())
        idx += n
    return radii, angles_list


# ---------------------------------------------------------------------------
# Direct constraint-push polish
# ---------------------------------------------------------------------------
def _constraint_push(pts, r_target, deadline, max_iters=400):
    pts = pts.copy()
    for _ in range(max_iters):
        if time.time() > deadline - 3:
            break
        d = _pairwise_distances(pts)
        np.fill_diagonal(d, np.inf)
        pair_margins = d / 2.0
        boundary_margins = 1.0 - np.sqrt(np.sum(pts ** 2, axis=1))

        if np.min(pair_margins) < np.min(boundary_margins) - 1e-14:
            i, j = np.unravel_index(np.argmin(pair_margins), pair_margins.shape)
            if i == j:
                break
            di = pts[i] - pts[j]
            dist = np.linalg.norm(di)
            if dist < 1e-15:
                break
            di /= dist
            push = (2.0 * r_target - dist) / 2.0
            if push > 1e-15:
                pts[i] += push * di
                pts[j] -= push * di
        else:
            i = int(np.argmin(boundary_margins))
            pi = pts[i]
            dist = np.linalg.norm(pi)
            if dist < 1e-15:
                pts[i] += 1e-4 * np.array([1.0, 0.0])
            else:
                di = pi / dist
                push = (1.0 - r_target - dist)
                if push > 1e-15:
                    pts[i] += push * di

        d = _pairwise_distances(pts)
        np.fill_diagonal(d, np.inf)
        r = min(np.min(d) / 2.0, np.min(1.0 - np.sqrt(np.sum(pts ** 2, axis=1))))
        if r >= r_target - 1e-12:
            break
    return pts


# ---------------------------------------------------------------------------
# Simulated Annealing for full points
# ---------------------------------------------------------------------------
def _simulated_annealing(pts, deadline, max_iters=5000):
    current = pts.copy()
    current_score = _score(current)
    best = current.copy()
    best_score = current_score
    
    # Temperature schedule
    t_start = 0.05
    t_end = 0.001
    cooling = (t_end / t_start) ** (1.0 / max_iters)
    
    rng = np.random.default_rng(42)
    
    for i in range(max_iters):
        if time.time() > deadline - 10:
            break
            
        T = t_start * (cooling ** i)
        
        # Random move: perturb a random circle
        idx = rng.integers(0, N)
        noise = rng.normal(0, T, size=2)
        new_pts = current.copy()
        new_pts[idx] += noise
        
        # Keep inside boundary roughly to avoid huge penalties
        rad = np.linalg.norm(new_pts[idx])
        if rad > 1.0:
            new_pts[idx] *= (1.0 / rad)
            
        new_score = _score(new_pts)
        
        if new_score > current_score:
            current = new_pts
            current_score = new_score
            if current_score > best_score:
                best = current.copy()
                best_score = current_score
        else:
            # Metropolis criterion
            delta = current_score - new_score
            if delta < 0:
                current = new_pts
                current_score = new_score
            else:
                prob = np.exp(-delta / T)
                if rng.random() < prob:
                    current = new_pts
                    current_score = new_score
                    
    return best, best_score

# ---------------------------------------------------------------------------
# Multi-stage annealing on full points
# ---------------------------------------------------------------------------
def _anneal_points(pts, deadline):
    current = pts.copy()
    current_score = _score(current)
    betas = [5.0, 15.0, 40.0, 100.0, 250.0, 500.0, 1000.0, 2000.0, 5000.0]
    for beta in betas:
        if time.time() > deadline - 4:
            break
        try:
            res = minimize(_smooth_objective, current.ravel(), args=(beta,),
                           method='L-BFGS-B',
                           options={'maxiter': 800, 'ftol': 1e-15, 'gtol': 1e-13,
                                    'maxfun': 20000})
            new_pts = res.x.reshape(N, 2)
            if not np.all(np.isfinite(new_pts)):
                continue
            s = _score(new_pts)
            if s > current_score:
                current = new_pts
                current_score = s
        except Exception:
            pass
    return current, current_score

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE

    ring_sizes = [1, 6, 12, 18, 13]
    init_radii = np.array([0.0, 0.252, 0.504, 0.756, 0.875])
    angles_list = []
    for i, n in enumerate(ring_sizes):
        if i == 0:
            angles_list.append(np.array([0.0]))
        else:
            offset = np.pi / n if i % 2 == 0 else 0.0
            angles_list.append(np.linspace(0, 2 * np.pi, n, endpoint=False) + offset)

    best_points = None
    best_score = -1.0
    ring_x = None

    def update_best(pts):
        nonlocal best_points, best_score
        if pts is None:
            return
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_points = np.asarray(pts, dtype=float).copy()
        return s

    def make_ring_bounds():
        b = [(0.0, 0.95)] * len(ring_sizes)
        for i in range(1, len(ring_sizes)):
            b.extend([(0.0, 2.0 * np.pi)] * ring_sizes[i])
        return b

    # ---- Stage 1: Differential evolution over ring parameters ----
    if time.time() < deadline - 55:
        bounds = make_ring_bounds()

        def ring_obj(vec):
            if time.time() > deadline - 55:
                return 1e10
            radii, al = _unflatten_ring(vec, ring_sizes)
            radii = np.clip(radii, 0.0, 0.95)
            pts = _ring_to_points(radii, al)
            return -_softmin_logsumexp(
                np.concatenate([(_pairwise_distances(pts) / 2.0).ravel(),
                                1.0 - np.sqrt(np.sum(pts ** 2, axis=1))]), 30.0)

        try:
            res_de = differential_evolution(
                ring_obj, bounds, maxiter=120, popsize=20, tol=1e-9,
                mutation=(0.3, 1.0), recombination=0.8, seed=42, polish=False)
            if time.time() < deadline - 45:
                ring_x = res_de.x
                radii_opt, al_opt = _unflatten_ring(ring_x, ring_sizes)
                radii_opt = np.clip(radii_opt, 0.0, 0.95)
                update_best(_ring_to_points(radii_opt, al_opt))
        except Exception:
            pass

    # ---- Stage 2: L-BFGS-B refinement on ring parameters ----
    if time.time() < deadline - 35 and best_points is not None:
        bounds = make_ring_bounds()
        x_start = ring_x if ring_x is not None else _flatten_ring(init_radii, angles_list)

        def ring_obj2(vec):
            if time.time() > deadline - 35:
                return 1e10
            radii, al = _unflatten_ring(vec, ring_sizes)
            radii = np.clip(radii, 0.0, 0.95)
            pts = _ring_to_points(radii, al)
            return -_softmin_logsumexp(
                np.concatenate([(_pairwise_distances(pts) / 2.0).ravel(),
                                1.0 - np.sqrt(np.sum(pts ** 2, axis=1))]), 60.0)

        try:
            res_lb = minimize(ring_obj2, x_start, method='L-BFGS-B', bounds=bounds,
                              options={'maxiter': 1500, 'ftol': 1e-15, 'gtol': 1e-13,
                                       'maxfun': 30000})
            if time.time() < deadline - 30:
                radii_opt, al_opt = _unflatten_ring(res_lb.x, ring_sizes)
                radii_opt = np.clip(radii_opt, 0.0, 0.95)
                update_best(_ring_to_points(radii_opt, al_opt))
        except Exception:
            pass

    # ---- Stage 3: Binary search for max radius via soft violation ----
    if time.time() < deadline - 25 and best_points is not None:
        C_best = best_points.copy()

        def soft_viol(C, r):
            diff = C[:, np.newaxis, :] - C[np.newaxis, :, :]
            dist = np.sqrt(np.sum(diff ** 2, axis=2))
            np.fill_diagonal(dist, np.inf)
            pair_viol = np.maximum(0.0, 2.0 * r - dist)
            rad = np.linalg.norm(C, axis=1)
            bound_viol = np.maximum(0.0, rad - (1.0 - r))
            return float(np.sum(pair_viol ** 2) + np.sum(bound_viol ** 2))

        r_low = max(0.1, best_score)
        r_high = 0.135
        for _ in range(10):
            if time.time() > deadline - 18:
                break
            r_mid = (r_low + r_high) / 2.0
            limit = 1.0 - r_mid
            bnds = [(-limit, limit)] * (2 * N)

            def obj(x, r=r_mid):
                return soft_viol(x.reshape(N, 2), r)

            try:
                res = minimize(obj, C_best.flatten(), method='L-BFGS-B', bounds=bnds,
                               options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12,
                                        'maxfun': 10000})
                if time.time() > deadline - 15:
                    break
                C_opt = res.x.reshape(N, 2)
                if soft_viol(C_opt, r_mid) < 1e-8:
                    r_low = r_mid
                    C_best = C_opt
                else:
                    r_high = r_mid
            except Exception:
                r_high = r_mid
        update_best(C_best)

    # ---- Stage 4: Full-point progressive beta annealing ----
    if time.time() < deadline - 12 and best_points is not None:
        try:
            ann_pts, ann_score = _anneal_points(best_points, deadline)
            update_best(ann_pts)
        except Exception:
            pass

    # ---- Stage 5: Simulated Annealing to escape local optima ----
    if time.time() < deadline - 5 and best_points is not None:
        try:
            sa_pts, sa_score = _simulated_annealing(best_points, deadline, max_iters=2000)
            update_best(sa_pts)
        except Exception:
            pass

    # ---- Stage 6: Direct constraint-push polish toward best_score ----
    if time.time() < deadline - 2 and best_points is not None:
        try:
            pushed = _constraint_push(best_points, best_score, deadline, max_iters=400)
            update_best(pushed)
        except Exception:
            pass

    # ---- Fallback: hexagonal lattice ----
    if best_points is None or best_score < 0.05:
        spacing = 0.24
        points = []
        for j in range(-10, 11):
            for i in range(-10, 11):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if np.sqrt(x * x + y * y) <= 1.0 - spacing / 2.0:
                    points.append([x, y])
        points = np.array(points)
        if len(points) > N:
            d = np.linalg.norm(points, axis=1)
            points = points[np.argsort(d, kind='stable')[:N]]
        update_best(points)

    return best_points
