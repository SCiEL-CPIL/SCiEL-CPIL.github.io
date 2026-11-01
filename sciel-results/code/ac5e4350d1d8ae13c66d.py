import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 14
D = 3

def _pairwise_distances(pts):
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=-1))
    return dist

def _true_score(pts):
    dist = _pairwise_distances(pts)
    mask = ~np.eye(N, dtype=bool)
    d = dist[mask]
    mn = np.min(d)
    mx = np.max(d)
    if mn < 1e-9:
        return np.inf
    return (mx / mn) ** 2

def _soft_log_ratio(dist, tau):
    mask = ~np.eye(N, dtype=bool)
    d = dist[mask]
    max_d = np.max(d)
    soft_max = max_d + tau * np.log(np.sum(np.exp((d - max_d) / tau)))
    min_d = np.min(d)
    soft_min = min_d - tau * np.log(np.sum(np.exp((min_d - d) / tau)))
    return soft_max - soft_min

def _objective_coords(x, tau):
    pts = x.reshape(N, D)
    pts = pts - np.mean(pts, axis=0)
    norm_sq = np.sum(pts**2)
    if norm_sq < 1e-10:
        return 1e10
    pts = pts * np.sqrt(N / norm_sq)
    dist = _pairwise_distances(pts)
    mask = ~np.eye(N, dtype=bool)
    d = dist[mask]
    min_d = np.min(d)
    if min_d < 1e-5:
        return 1e10
    return _soft_log_ratio(dist, tau)

def _generate_icosahedron():
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    return verts

def _generate_fib_sphere(n):
    i = np.arange(n)
    phi = np.arccos(1 - 2 * (i + 0.5) / n)
    theta = np.pi * (1 + np.sqrt(5)) * i
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.column_stack((x, y, z))

def _generate_initial_guesses():
    guesses = []
    ico = _generate_icosahedron()
    for h in [0.3, 0.5, 0.7, 1.0]:
        extra = np.array([[0, 0, h], [0, 0, -h]])
        guesses.append(np.vstack([ico, extra]))
    for h in [0.5, 1.0, 1.5]:
        extra = np.array([[0, 0, 0], [0, 0, h]])
        guesses.append(np.vstack([ico, extra]))
    guesses.append(_generate_fib_sphere(N))
    rng = np.random.default_rng(42)
    for _ in range(8):
        pts = ico.copy()
        extra = np.array([[0, 0, 0.5], [0, 0, -0.5]])
        pts = np.vstack([pts, extra])
        pts = pts + rng.normal(0, 0.1, size=pts.shape)
        guesses.append(pts)
    for _ in range(5):
        pts = rng.normal(0, 1, size=(N, D))
        guesses.append(pts)
    for top_h, bot_h, r_top, r_bot in [(0.7, -0.7, 0.7, 0.7), (0.5, -0.5, 0.9, 0.9), (0.8, -0.8, 0.5, 0.5)]:
        angles_top = 2 * np.pi * np.arange(6) / 6
        angles_bot = 2 * np.pi * np.arange(6) / 6 + np.pi / 6
        top_pts = np.column_stack([r_top * np.cos(angles_top), r_top * np.sin(angles_top), np.full(6, top_h)])
        bot_pts = np.column_stack([r_bot * np.cos(angles_bot), r_bot * np.sin(angles_bot), np.full(6, bot_h)])
        poles = np.array([[0, 0, 1.2], [0, 0, -1.2]])
        guesses.append(np.vstack([top_pts, bot_pts, poles]))
    return guesses

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    best_score = np.inf
    best_pts = None
    guesses = _generate_initial_guesses()
    tau_stages = [0.3, 0.1, 0.05, 0.02]
    candidates = []
    for guess in guesses:
        if time.time() > deadline - 20:
            break
        x0 = guess.flatten()
        try:
            res = minimize(_objective_coords, x0, args=(tau_stages[0],),
                           method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-6})
            if res.fun < 1e5:
                candidates.append(res.x)
        except Exception:
            continue
    if candidates:
        cand_scores = []
        for cx in candidates:
            pts = cx.reshape(N, D)
            pts = pts - np.mean(pts, axis=0)
            norm_sq = np.sum(pts**2)
            if norm_sq > 1e-10:
                pts = pts * np.sqrt(N / norm_sq)
            score = _true_score(pts)
            cand_scores.append((score, cx))
        cand_scores.sort(key=lambda x: x[0])
        top_candidates = [c[1] for c in cand_scores[:6]]
    else:
        top_candidates = [g.flatten() for g in guesses[:3]]
    for cx in top_candidates:
        if time.time() > deadline:
            break
        x_curr = cx.copy()
        for tau in tau_stages[1:]:
            if time.time() > deadline:
                break
            try:
                res = minimize(_objective_coords, x_curr, args=(tau,),
                               method='L-BFGS-B', options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8})
                if res.fun < _objective_coords(x_curr, tau) + 1e-6:
                    x_curr = res.x
            except Exception:
                continue
        pts = x_curr.reshape(N, D)
        pts = pts - np.mean(pts, axis=0)
        norm_sq = np.sum(pts**2)
        if norm_sq > 1e-10:
            pts = pts * np.sqrt(N / norm_sq)
        score = _true_score(pts)
        if score < best_score:
            best_score = score
            best_pts = pts
    if time.time() < deadline - 15:
        rng = np.random.default_rng(123)
        for _ in range(10):
            if time.time() > deadline - 5:
                break
            x0 = rng.normal(0, 1, size=(N, D)).flatten()
            try:
                res1 = minimize(_objective_coords, x0, args=(0.2,),
                                method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-9})
                if res1.fun > 1e5:
                    continue
                res2 = minimize(_objective_coords, res1.x, args=(0.03,),
                                method='L-BFGS-B', options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8})
                pts = res2.x.reshape(N, D)
                pts = pts - np.mean(pts, axis=0)
                norm_sq = np.sum(pts**2)
                if norm_sq > 1e-10:
                    pts = pts * np.sqrt(N / norm_sq)
                score = _true_score(pts)
                if score < best_score:
                    best_score = score
                    best_pts = pts
            except Exception:
                continue
    if best_pts is None:
        best_pts = _generate_fib_sphere(N)
    return best_pts
