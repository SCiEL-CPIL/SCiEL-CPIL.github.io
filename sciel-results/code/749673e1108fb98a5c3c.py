import numpy as np
import time
from scipy.optimize import minimize

N = 14
D = 3
NP = N * (N - 1) // 2  # 91 pairwise distances

def get_pairwise_sq_dists(points):
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    iu = np.triu_indices(N, k=1)
    return sq_dists[iu]

def raw_score(points):
    sq_dists = get_pairwise_sq_dists(points)
    min_d = np.min(sq_dists)
    if min_d < 1e-12:
        return np.inf
    return (np.max(sq_dists) / min_d)

def normalize_points(points):
    points = points - np.mean(points, axis=0)
    mean_norm_sq = np.mean(np.sum(points**2, axis=1))
    if mean_norm_sq > 1e-12:
        points = points / np.sqrt(mean_norm_sq)
    return points

# ---------- helpers for gradients ----------
def _pair_indices():
    iu = np.triu_indices(N, k=1)
    return iu[0], iu[1]

I_A, I_B = _pair_indices()

def _pair_grad_points(x, d):
    """Gradient of pairwise squared distances d (len NP) w.r.t. x (len N*D)."""
    p = x.reshape(N, D)
    diff = p[I_A] - p[I_B]
    g = np.zeros_like(p)
    np.add.at(g, I_A, 2.0 * diff)
    np.add.at(g, I_B, -2.0 * diff)
    return g

# ---------- objective: maximize min d^2 (max d^2 <= 1) ----------
def objective_epigraph(x, tau=2000.0):
    p = x.reshape(N, D)
    d = get_pairwise_sq_dists(p)
    d = np.maximum(d, 1e-15)
    m = np.min(d)
    # soft min
    e = np.exp(-tau * (d - m))
    s = np.sum(e)
    soft_min = m - np.log(s) / tau
    # penalty for d > 1
    pen = 0.0
    if np.any(d > 1.0):
        pen = 1e3 * np.sum(np.maximum(d - 1.0, 0.0) ** 2)
    val = -soft_min + pen
    # gradient
    g = np.zeros_like(p)
    w = e / s
    g = -np.sum(w[:, None] * _pair_grad_points(x, d), axis=0)
    if np.any(d > 1.0):
        over = np.maximum(d - 1.0, 0.0)
        g += 2.0e3 * np.sum(over[:, None] * _pair_grad_points(x, d), axis=0)
    return val, g.ravel()

# ---------- objective: direct ratio max d^2 / min d^2 ----------
def objective_ratio(x, tau=2000.0):
    p = x.reshape(N, D)
    d = get_pairwise_sq_dists(p)
    d = np.maximum(d, 1e-15)
    mx = np.max(d)
    mn = np.min(d)
    # soft max
    em = np.exp(tau * (d - mx))
    smax = mx + np.log(np.sum(em)) / tau
    # soft min
    en = np.exp(-tau * (d - mn))
    smin = mn - np.log(np.sum(en)) / tau
    val = np.log(smax) - np.log(smin)
    gw_max = em / np.sum(em)
    gw_min = en / np.sum(en)
    g = np.sum(gw_max[:, None] * _pair_grad_points(x, d), axis=0) / smax
    g -= np.sum(gw_min[:, None] * _pair_grad_points(x, d), axis=0) / smin
    return val, g.ravel()

def _make_f(obj):
    def f(x):
        v, g = obj(x)
        return v, g
    return f

# ---------- initial candidates ----------
def _make_seeds():
    seeds = []
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    ico = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)

    # seed 1: icosahedron + poles
    pts = np.vstack([ico, np.array([[0, 0, 1.0]]), np.array([[0, 0, -1.0]])])
    seeds.append(normalize_points(pts).flatten())

    # seed 2: icosahedron + 2 points opposite on a great circle
    a = np.array([1.0, 0.0, 0.0]); a /= np.linalg.norm(a)
    pts = np.vstack([ico, a.reshape(1, D), (-a).reshape(1, D)])
    seeds.append(normalize_points(pts).flatten())

    # seed 3: icosahedron + 2 generic points
    rng = np.random.default_rng(7)
    for _ in range(2):
        v = rng.normal(size=D); v /= np.linalg.norm(v)
        w = rng.normal(size=D); w /= np.linalg.norm(w)
        pts = np.vstack([ico, v.reshape(1, D), w.reshape(1, D)])
        seeds.append(normalize_points(pts).flatten())

    # seed 4: icosahedron + 2 points near poles (perturbed)
    for ang in (0.0, np.pi / 2):
        top = np.array([0, 0, 1.0])
        bot = np.array([0, 0, -1.0])
        pts = np.vstack([ico, top.reshape(1, D), bot.reshape(1, D)])
        pts = normalize_points(pts)
        seeds.append(pts.flatten())

    # seed 5: pure icosahedron subset + 2 (use 12 ico, drop 2, add 2)
    for k in range(2):
        sub = ico[:12]
        v = rng.normal(size=D); v /= np.linalg.norm(v)
        w = rng.normal(size=D); w /= np.linalg.norm(w)
        pts = np.vstack([sub, v.reshape(1, D), w.reshape(1, D)])
        seeds.append(normalize_points(pts).flatten())

    # seed 6: 14 points on sphere via fibonacci (deterministic)
    golden = (1.0 + np.sqrt(5.0)) / 2.0
    fib = np.zeros((N, D))
    for i in range(N):
        z = 1.0 - 2.0 * (i + 0.5) / N
        r = np.sqrt(max(0.0, 1.0 - z * z))
        th = 2.0 * np.pi * i / golden
        fib[i] = [r * np.cos(th), r * np.sin(th), z]
    fib = fib / np.linalg.norm(fib, axis=1, keepdims=True)
    seeds.append(normalize_points(fib).flatten())

    # seed 7: random gaussian
    rng2 = np.random.default_rng(123)
    for _ in range(4):
        pts = rng2.normal(size=(N, D))
        seeds.append(normalize_points(pts).flatten())

    # seed 8: random on sphere
    for _ in range(4):
        pts = rng2.normal(size=(N, D))
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        seeds.append(normalize_points(pts).flatten())

    return seeds

def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s kill

    best_score = np.inf
    best_points = None

    def update_best(pts):
        nonlocal best_score, best_points
        if pts is None:
            return
        pts = np.asarray(pts, dtype=float)
        if pts.shape != (N, D):
            return
        pts = normalize_points(pts)
        sc = raw_score(pts)
        if sc < best_score:
            best_score = sc
            best_points = pts.copy()

    seeds = _make_seeds()
    update_best(seeds[0].reshape(N, D))

    # tau schedule: start smooth, end sharp
    taus = [50.0, 200.0, 1000.0, 5000.0, 20000.0]

    rng = np.random.default_rng(2024)

    # alternating use of epigraph and ratio objectives
    def run_opt(x0):
        nonlocal best_points
        cur = x0
        for tau in taus:
            if time.time() - start_time > deadline - 5:
                return
            obj = objective_epigraph if (tau % 2 == 0) else objective_ratio
            f = _make_f(obj)
            try:
                res = minimize(f, cur, jac=True, method='L-BFGS-B',
                               args=(tau,),
                               options={'maxiter': 150, 'ftol': 1e-14, 'gtol': 1e-10})
                if res.x is not None:
                    cur = res.x
                    update_best(cur.reshape(N, D))
            except Exception:
                pass
        return cur

    # Phase 1: optimize each seed
    for s in seeds:
        if time.time() - start_time > deadline - 15:
            break
        run_opt(s.copy())

    # Phase 2: more random restarts if time permits
    extra = 0
    while time.time() - start_time < deadline - 20:
        if rng.random() < 0.5:
            pts = rng.normal(size=(N, D))
            pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        else:
            pts = rng.normal(size=(N, D))
        x0 = normalize_points(pts).flatten()
        run_opt(x0)
        extra += 1
        if extra > 60:
            break

    # Phase 3: high-precision polish on best
    if best_points is not None and time.time() - start_time < deadline - 5:
        for tau in (20000.0, 100000.0):
            if time.time() - start_time > deadline - 3:
                break
            f = _make_f(objective_epigraph)
            try:
                res = minimize(f, best_points.flatten(), jac=True, method='L-BFGS-B',
                               args=(tau,),
                               options={'maxiter': 400, 'ftol': 1e-16, 'gtol': 1e-12})
                if res.x is not None:
                    update_best(res.x.reshape(N, D))
            except Exception:
                pass

    if best_points is None:
        best_points = normalize_points(seeds[0].reshape(N, D))

    return best_points

if __name__ == "__main__":
    pts = construct()
    score = raw_score(pts)
    print(f"Score: {score}")
