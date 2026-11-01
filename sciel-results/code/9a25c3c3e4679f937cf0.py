import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
import itertools
from functools import partial

N = 14
D = 3
NUM_PAIRS = N * (N - 1) // 2

def get_pairwise_sq_dists(points):
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    iu = np.triu_indices(N, k=1)
    return sq_dists[iu]

def raw_score(points):
    sq_dists = get_pairwise_sq_dists(points)
    if np.min(sq_dists) < 1e-15:
        return np.inf
    return np.max(sq_dists) / np.min(sq_dists)

def _soft_minmax(sq_dists, tau):
    """Return (soft_max, soft_min) via log-sum-exp (numerically stable)."""
    sq_dists = np.maximum(sq_dists, 1e-18)
    m_max = np.max(sq_dists)
    m_min = np.min(sq_dists)
    # soft max
    a = tau * (sq_dists - m_max)
    a = np.clip(a, -50, 50)
    soft_max = m_max + np.log(np.sum(np.exp(a))) / tau
    # soft min
    b = -tau * (sq_dists - m_min)
    b = np.clip(b, -50, 50)
    soft_min = m_min - np.log(np.sum(np.exp(b))) / tau
    soft_min = max(soft_min, 1e-18)
    return soft_max, soft_min

def objective_log_ratio(x, tau=20.0):
    """Smooth log-ratio objective (minimize log(max/min) = log ratio)."""
    points = x.reshape(N, D)
    sq_dists = get_pairwise_sq_dists(points)
    soft_max, soft_min = _soft_minmax(sq_dists, tau)
    return np.log(soft_max) - np.log(soft_min)

def objective_epigraph(x, tau=20.0, mu=5.0):
    """
    Epigraph: x = [points (N*D), t]. Minimize t + mu*penalty
    s.t. soft_max <= t * soft_min  (i.e. ratio <= t)
    """
    points = x[:N*D].reshape(N, D)
    t = x[N*D]
    sq_dists = get_pairwise_sq_dists(points)
    soft_max, soft_min = _soft_minmax(sq_dists, tau)
    # constraint: soft_max - t*soft_min <= 0
    viol = soft_max - t * soft_min
    penalty = np.maximum(0.0, viol) ** 2
    return t + mu * penalty

def _norm_scale(points):
    pts = points - np.mean(points, axis=0)
    norm = np.sqrt(np.mean(np.sum(pts**2, axis=1)))
    if norm < 1e-15:
        return pts
    return pts / norm

# ---------- Structured initializations ----------
def cuboctahedron_init():
    """Cuboctahedron (6) + octahedron (6) + 2 extra. All coords from {0,±1,±1}.
    Cuboctahedron vertices: permutations of (0,±1,±1) -> 8? Actually 6? 
    Let's use: the 6 vertices of octahedron (±1,0,0),(0,±1,0),(0,0,±1)
    and cuboctahedron 6 vertices? Use icosahedron-like.
    """
    # Octahedron 6
    octa = np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]], float)
    # Cuboctahedron 6 (subset of 12): use (±1,±1,0),(±1,0,±1),(0,±1,±1) pick 6
    cubo = np.array([[1,1,0],[1,-1,0],[-1,1,0],[-1,-1,0],[1,0,1],[1,0,-1]], float)
    pts = np.vstack([octa, cubo])  # 12 points
    # 2 extra: place near center-ish but distinct, e.g. small offsets
    extra = np.array([[0.3,0.3,0.3],[-0.3,-0.3,-0.3]], float)
    pts = np.vstack([pts, extra])
    return _norm_scale(pts)

def icosahedron_init():
    """Icosahedron 12 vertices + 2 extra."""
    phi = (1 + np.sqrt(5)) / 2
    verts = []
    for s1 in [1,-1]:
        for s2 in [1,-1]:
            for s3 in [1,-1]:
                pass
    # standard 12 icosahedron vertices
    ico = np.array([
        [0, 1, phi],[0, 1, -phi],[0, -1, phi],[0, -1, -phi],
        [1, phi, 0],[1, -phi, 0],[-1, phi, 0],[-1, -phi, 0],
        [phi, 0, 1],[phi, 0, -1],[-phi, 0, 1],[-phi, 0, -1]
    ], float)
    # 2 extra: place along a symmetry axis direction, small
    extra = np.array([[0,0,2.2],[0,0,-2.2]], float)
    pts = np.vstack([ico, extra])
    return _norm_scale(pts)

def fibonacci_sphere(n_points):
    points = []
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(n_points):
        y = 1.0 - (i / (n_points - 1)) * 2.0
        radius = np.sqrt(max(0.0, 1.0 - y**2))
        theta = golden_angle * i
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        points.append([x, y, z])
    return np.array(points)

def grid_init():
    """3x3x3 grid minus corners to get 14, then scale."""
    pts = []
    for x in range(3):
        for y in range(3):
            for z in range(3):
                pts.append([x-1, y-1, z-1])
    pts = np.array(pts, float)
    # remove 5 corners to get 14 (total 27, remove 13)
    # keep 14: take first 14
    pts = pts[:14]
    return _norm_scale(pts)

def random_sphere(seed):
    r = np.random.default_rng(seed)
    pts = r.normal(size=(N, D))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / np.maximum(norms, 1e-12)
    return pts

def random_ball(seed, scale=1.5):
    r = np.random.default_rng(seed)
    pts = r.normal(size=(N, D)) * scale
    return _norm_scale(pts)

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill
    def remaining():
        return DEADLINE - (time.time() - start_time)

    best_score = np.inf
    best_points = None

    def consider(pts):
        nonlocal best_score, best_points
        if pts is None:
            return
        pts = np.asarray(pts, float)
        if pts.shape != (N, D) or not np.all(np.isfinite(pts)):
            return
        s = raw_score(pts)
        if np.isfinite(s) and s < best_score:
            best_score = s
            best_points = pts.copy()

    # Build diverse initializations
    inits = []
    inits.append(cuboctahedron_init())
    inits.append(icosahedron_init())
    inits.append(fibonacci_sphere(N))
    inits.append(grid_init())
    for s in range(8):
        inits.append(random_sphere(s))
    for s in range(8):
        inits.append(random_ball(100 + s))
    # perturbed structured
    for base in inits[:4]:
        for s in range(3):
            r = np.random.default_rng(500 + s)
            inits.append(_norm_scale(base + r.normal(size=(N, D)) * 0.15))

    # Optimization: for each init, run annealed smooth L-BFGS-B then Nelder-Mead polish
    tau_schedule = [8.0, 30.0, 120.0, 400.0]
    maxiter_schedule = [150, 200, 250, 300]

    for i, init_x in enumerate(inits):
        if remaining() < 6.0:
            break
        x_curr = init_x.flatten().copy()
        try:
            # Annealed smooth log-ratio with L-BFGS-B
            for idx, tau in enumerate(tau_schedule):
                if remaining() < 4.0:
                    break
                obj = partial(objective_log_ratio, tau=tau)
                res = minimize(
                    obj, x_curr, method='L-BFGS-B',
                    options={'maxiter': maxiter_schedule[idx],
                             'ftol': 1e-14, 'gtol': 1e-11, 'maxls': 50}
                )
                if res.x is not None and np.isfinite(res.fun):
                    x_curr = res.x
            # Epigraph refinement to directly target ratio t
            if remaining() > 5.0:
                x_epi = np.concatenate([x_curr, [1.2]])
                obj_epi = partial(objective_epigraph, tau=200.0, mu=10.0)
                res_epi = minimize(
                    obj_epi, x_epi, method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-11}
                )
                if res_epi.x is not None and np.isfinite(res_epi.fun):
                    x_curr = res_epi.x[:N*D]
            # Final Nelder-Mead polish on log-ratio with high tau
            if remaining() > 3.0:
                obj_nm = partial(objective_log_ratio, tau=800.0)
                res_nm = minimize(
                    obj_nm, x_curr, method='Nelder-Mead',
                    options={'maxiter': 4000, 'xatol': 1e-10, 'fatol': 1e-12,
                             'adaptive': True}
                )
                if res_nm.x is not None and np.isfinite(res_nm.fun):
                    x_curr = res_nm.x
            pts = x_curr.reshape(N, D)
            consider(pts)
        except Exception:
            pass

    # If nothing found, fallback to best raw init
    if best_points is None:
        for init_x in inits:
            consider(init_x)
        if best_points is None:
            best_points = fibonacci_sphere(N)

    return best_points

if __name__ == "__main__":
    pts = construct()
    sq = get_pairwise_sq_dists(pts)
    print(f"Score: {np.max(sq)/np.min(sq)}")
