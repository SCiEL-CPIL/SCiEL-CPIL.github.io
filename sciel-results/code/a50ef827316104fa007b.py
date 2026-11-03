"""Combined approach: ring-structured init + soft-min log-sum-exp + binary search polish."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 50


def _pairwise_distances(pts):
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=-1))


def _score(pts):
    """Compute the packing score."""
    if pts.shape != (N, 2) or not np.all(np.isfinite(pts)):
        return -np.inf
    d = _pairwise_distances(pts)
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d) / 2.0
    min_boundary = np.min(1.0 - np.linalg.norm(pts, axis=1))
    return min(min_pair, min_boundary)


def _softmin_logsumexp(values, beta):
    """Soft-min via log-sum-exp: approximates min(values) smoothly."""
    neg = -values * beta
    neg_max = np.max(neg)
    if not np.isfinite(neg_max):
        return 0.0
    exp_sum = np.sum(np.exp(neg - neg_max))
    if exp_sum <= 0 or not np.isfinite(exp_sum):
        return 0.0
    return -(neg_max + np.log(exp_sum)) / beta


def _softmin_objective(x, beta=50.0):
    """Minimize -softmin of all constraint margins (pairwise/2 and boundary)."""
    pts = x.reshape(N, 2)
    d = _pairwise_distances(pts)
    np.fill_diagonal(d, np.inf)
    boundary = 1.0 - np.linalg.norm(pts, axis=1)
    pair_half = d / 2.0
    all_vals = np.concatenate([pair_half.ravel(), boundary])
    all_vals = all_vals[np.isfinite(all_vals)]
    if len(all_vals) == 0:
        return 0.0
    soft_min = _softmin_logsumexp(all_vals, beta)
    return -soft_min


def _softmin_grad(x, beta=50.0):
    """Gradient of -softmin objective."""
    pts = x.reshape(N, 2)
    d = _pairwise_distances(pts)
    np.fill_diagonal(d, np.inf)
    boundary = 1.0 - np.linalg.norm(pts, axis=1)
    pair_half = d / 2.0

    all_vals = np.concatenate([pair_half.ravel(), boundary])
    finite_mask = np.isfinite(all_vals)
    if not np.any(finite_mask):
        return np.zeros_like(x)

    vals = all_vals[finite_mask]
    neg = -vals * beta
    neg_max = np.max(neg)
    exps = np.exp(neg - neg_max)
    exps /= np.sum(exps)
    # d(-softmin)/d(all_vals) = -exps (since softmin = -softmax(-vals))
    # Actually: softmin = -(neg_max + log(sum(exp(neg-neg_max))))/beta
    # d(softmin)/d(v_k) = -exps_k (for the finite ones)
    # d(-softmin)/d(v_k) = exps_k
    # So gradient w.r.t. all_vals = exps_k for finite, 0 for non-finite

    grad_all = np.zeros_like(all_vals)
    grad_all[finite_mask] = exps

    # Now back-propagate to pts
    grad = np.zeros_like(pts)

    # Pairwise part: all_vals[:N*(N-1)] corresponds to pair_half.ravel()
    # pair_half[i,j] = d[i,j]/2 for i!=j
    # d(pair_half[i,j])/d(pts[i]) = -diff[i,j]/(2*dist[i,j]) * ... 
    # Actually pair_half[i,j] = ||pts[i]-pts[j]||/2
    # d(pair_half[i,j])/d(pts[i]) = (pts[i]-pts[j])/(2*dist[i,j])
    # d(pair_half[i,j])/d(pts[j]) = (pts[j]-pts[i])/(2*dist[i,j])

    pair_grad = grad_all[:N * N].reshape(N, N)
    # For each pair (i,j), contribution to grad[i] and grad[j]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.where(d > 0, d, 1.0)
    inv_dist = 1.0 / dist
    # d(pair_half[i,j])/d(pts[i]) = diff[i,j] / (2*dist[i,j])
    # pair_half[i,j] appears at index i*N+j in the flattened array
    # grad[i] += sum_j pair_grad[i,j] * diff[i,j]/(2*dist[i,j])
    # grad[j] -= sum_i pair_grad[i,j] * diff[i,j]/(2*dist[i,j])  [since pair_half[i,j] = pair_half[j,i]]

    # Vectorized:
    W = pair_grad * inv_dist * 0.5
    # grad[i] = sum_j W[i,j] * diff[i,j]
    term1 = (W * diff).sum(axis=1)
    # grad[j] = -sum_i W[i,j] * diff[i,j] = sum_i W[j,i] * diff[j,i]  (by symmetry of W and diff)
    # But W is not necessarily symmetric. Let's be careful.
    # pair_half[i,j] = dist[i,j]/2, d/d(pts[i]) = diff[i,j]/(2*dist[i,j])
    # So grad[i] += sum_{j!=i} pair_grad[i,j] * diff[i,j]/(2*dist[i,j])
    # grad[j] += sum_{i!=j} pair_grad[i,j] * d(pair_half[i,j])/d(pts[j])
    #           = sum_{i!=j} pair_grad[i,j] * (pts[j]-pts[i])/(2*dist[i,j])
    #           = -sum_{i!=j} pair_grad[i,j] * diff[i,j]/(2*dist[i,j])
    # So grad[j] -= sum_{i!=j} pair_grad[i,j] * diff[i,j]/(2*dist[i,j])
    # Which means: grad[j] = -sum_i W[i,j] * diff[i,j] + (diagonal is 0)
    # In matrix form: grad = W @ diff - (W.T @ diff) ... let me think again.
    # grad[i] = sum_j W[i,j] * diff[i,j]
    # grad[j] = -sum_i W[i,j] * diff[i,j]  (from the pair (i,j) where i<j, but we sum over all)
    # Actually for a given j, the contribution from pair (i,j) to grad[j] is -W[i,j]*diff[i,j]
    # So grad[j] = -sum_i W[i,j] * diff[i,j]
    # In matrix notation: grad = W @ diff - (W.T @ diff)  ... no.
    # Let me just do it explicitly with broadcasting.
    grad_pair_i = (W * diff).sum(axis=1)  # sum over j for each i
    grad_pair_j = -(W.T * diff.T).sum(axis=0)  # sum over i for each j, negated
    # Wait, that's not right either. Let me be very explicit.
    # For point i: grad[i] = sum_{j != i} W[i,j] * diff[i,j]
    # For point j: from pair (i,j): contribution is -W[i,j] * diff[i,j]
    # So grad[j] = -sum_{i != j} W[i,j] * diff[i,j]
    # This is: grad = rowsum(W * diff) - colsum(W * diff) ... 
    # rowsum(W*diff)[i] = sum_j W[i,j]*diff[i,j]  ✓ for grad[i]
    # colsum(W*diff)[j] = sum_i W[i,j]*diff[i,j]  and we negate for grad[j]
    # So grad = rowsum - colsum
    # But rowsum and colsum are both 1D arrays of length N.
    # grad[i] = rowsum[i] - colsum[i]? No, that's not right.
    # grad[i] = rowsum[i]  (from the i-row)
    # grad[j] = -colsum[j]  (from the j-column, negated)
    # But these are for the SAME index. Let me re-derive.
    # Total grad[k] = sum over all pairs (i,j) involving k:
    #   if k=i: +W[i,j]*diff[i,j]
    #   if k=j: -W[i,j]*diff[i,j]
    # = sum_j W[k,j]*diff[k,j] - sum_i W[i,k]*diff[i,k]
    # = rowsum(W*diff)[k] - colsum(W*diff)[k]
    # Yes! So grad = rowsum - colsum.
    row_sum = (W * diff).sum(axis=1)
    col_sum = (W * diff).sum(axis=0)
    grad += row_sum - col_sum

    # Boundary part: all_vals[N*N:] corresponds to boundary
    # boundary[i] = 1 - ||pts[i]||
    # d(boundary[i])/d(pts[i]) = -pts[i]/||pts[i]||
    bound_grad = grad_all[N * N:]
    dist_origin = np.linalg.norm(pts, axis=1)
    inv_dist_origin = np.where(dist_origin > 1e-12, 1.0 / dist_origin, 0.0)
    grad += (bound_grad * inv_dist_origin)[:, np.newaxis] * (-pts)

    return grad.flatten()


def _get_ring_points(params, ring_counts):
    k = len(ring_counts)
    P = np.zeros((N, 2))
    radii = params[:k]
    offsets = params[k:2 * k]
    count = 0
    for i, n_pts in enumerate(ring_counts):
        if n_pts == 1:
            P[count] = [0.0, 0.0]
            count += 1
        else:
            R = radii[i]
            offset = offsets[i]
            angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False) + offset
            P[count:count + n_pts, 0] = R * np.cos(angles)
            P[count:count + n_pts, 1] = R * np.sin(angles)
            count += n_pts
    return P


def _get_hex_initial_points():
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        margin = spacing / 2.0
        points, radius = points[radius <= 1.0 - margin], radius[radius <= 1.0 - margin]
        if len(points) >= N:
            indices = np.argsort(radius, kind="stable")
            return points[indices[:N]]
        spacing *= 0.98


def _optimize_softmin(P_init, beta=50.0, time_limit=5.0, max_iter=500):
    """Local refinement using L-BFGS-B on the soft-min objective."""
    start = time.time()

    def obj(pf):
        return _softmin_objective(pf, beta)

    def grd(pf):
        return _softmin_grad(pf, beta)

    def _callback(state):
        return time.time() - start > time_limit

    try:
        res = minimize(obj, P_init.flatten(), jac=grd, method='L-BFGS-B',
                       options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-12,
                                'maxls': 30, 'callback': _callback})
        P_opt = res.x.reshape(N, 2)
        if not np.all(np.isfinite(P_opt)):
            P_opt = P_init
        return P_opt
    except Exception:
        return P_init


def _global_search_ring(ring_counts, time_limit=15.0, seed=42):
    """Differential evolution over ring radii + phases for a fixed topology."""
    k = len(ring_counts)
    bounds = [(0.0, 1.0)] * k + [(0.0, 2 * np.pi)] * k

    def neg_score(params):
        P = _get_ring_points(params, ring_counts)
        return -_score(P)

    start = time.time()

    def _callback(xk, convergence):
        return time.time() - start > time_limit

    try:
        res = differential_evolution(neg_score, bounds, maxiter=40, popsize=14,
                                     tol=1e-5, seed=seed, polish=True,
                                     init='latinhypercube', callback=_callback)
        best_params = res.x
        return _get_ring_points(best_params, ring_counts)
    except Exception:
        return None


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill

    np.random.seed(42)

    # Baseline: hexagonal lattice
    P_best = _get_hex_initial_points()
    score_best = _score(P_best)

    ring_structures = [
        [1, 6, 12, 18, 13],
        [1, 6, 12, 14, 17],
        [1, 6, 12, 16, 15],
        [1, 6, 12, 15, 16],
        [1, 6, 12, 17, 14],
    ]

    # Phase 1: global search for ring topology parameters
    for idx, ring_counts in enumerate(ring_structures):
        if time.time() > deadline - 45.0:
            break
        P_cand = _global_search_ring(ring_counts, time_limit=12.0, seed=42 + idx)
        if P_cand is None:
            continue
        s = _score(P_cand)
        if s > score_best:
            P_best = P_cand
            score_best = s

    # Phase 2: binary search on r with soft-min polish
    r_low = score_best
    r_high = 0.135
    for _ in range(9):
        if time.time() > deadline - 15.0:
            break
        r_mid = (r_low + r_high) / 2.0
        best_P = None
        best_obj = np.inf
        for restart in range(3):
            if time.time() > deadline - 15.0:
                break
            if restart == 0:
                P_init = P_best.copy()
            else:
                P_init = P_best + 0.003 * (restart + 1) * np.random.randn(N, 2)
            P_opt = _optimize_softmin(P_init, beta=50.0, time_limit=2.5)
            obj_val = _softmin_objective(P_opt.flatten(), beta=50.0)
            if obj_val < best_obj:
                best_obj = obj_val
                best_P = P_opt
        if best_P is None:
            break
        s = _score(best_P)
        if s >= r_mid - 1e-9:
            r_low = r_mid
            if s > score_best:
                P_best = best_P
                score_best = s
        else:
            r_high = r_mid

    # Phase 3: final high-precision polish with increasing beta
    if time.time() < deadline - 5.0:
        for beta in [100.0, 200.0, 500.0]:
            if time.time() > deadline - 3.0:
                break
            for restart in range(2):
                if time.time() > deadline - 2.0:
                    break
                if restart == 0:
                    P_init = P_best.copy()
                else:
                    P_init = P_best + 0.0005 * np.random.randn(N, 2)
                P_opt = _optimize_softmin(P_init, beta=beta, time_limit=2.0, max_iter=800)
                s = _score(P_opt)
                if s > score_best:
                    P_best = P_opt
                    score_best = s

    # Ensure all points strictly inside the unit circle
    norms = np.linalg.norm(P_best, axis=1)
    mask = norms > 1.0 - 1e-10
    if np.any(mask):
        P_best[mask] *= (1.0 - 1e-10) / norms[mask]

    return P_best
