"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations

H = np.sqrt(3) / 2
S3 = np.sqrt(3)

# Precompute all C(11,3) = 165 triples
_TRIPLES = np.array([(i, j, k) for i in range(N) for j in range(i + 1, N) for k in range(j + 1, N)])
_N_TRIPLES = _TRIPLES.shape[0]

# Triangle vertices in Cartesian
_V0 = np.array([0.0, 0.0])
_V1 = np.array([1.0, 0.0])
_V2 = np.array([0.5, H])


def _bary_to_cart(bary):
    """Convert barycentric (u,v) with w=1-u-v to Cartesian."""
    u = bary[:, 0]
    v = bary[:, 1]
    x = u * _V1[0] + v * _V2[0]
    y = u * _V1[1] + v * _V2[1]
    return np.column_stack([x, y])


def _cart_to_bary(points):
    """Convert Cartesian to barycentric (u,v) with w=1-u-v."""
    x = points[:, 0]
    y = points[:, 1]
    v = y / H  # since y = H * v (w component)
    u = x - 0.5 * v
    u = np.clip(u, 0, 1)
    v = np.clip(v, 0, 1)
    w = 1.0 - u - v
    # Re-normalize if outside
    mask = w < 0
    if np.any(mask):
        s = u[mask] + v[mask]
        if s > 0:
            u[mask] /= s
            v[mask] /= s
        else:
            u[mask] = 1/3
            v[mask] = 1/3
    return np.column_stack([u, v])


def _min_area_from_bary(bary):
    """Compute min triangle area from barycentric coords (u,v) where w=1-u-v."""
    pts = _bary_to_cart(bary)
    p1 = pts[_TRIPLES[:, 0]]
    p2 = pts[_TRIPLES[:, 1]]
    p3 = pts[_TRIPLES[:, 2]]
    d1 = p2 - p1
    d2 = p3 - p1
    cross = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    areas = 0.5 * np.abs(cross)
    return np.min(areas), areas


def _project_to_simplex(uv):
    """Project (u,v) to valid barycentric: u>=0, v>=0, u+v<=1."""
    u = uv[:, 0].copy()
    v = uv[:, 1].copy()
    # Clip to non-negative
    u = np.maximum(u, 0)
    v = np.maximum(v, 0)
    # Clip to u+v <= 1
    s = u + v
    mask = s > 1
    if np.any(mask):
        u[mask] /= s[mask]
        v[mask] /= s[mask]
    return np.column_stack([u, v])


def _smooth_min_objective(uv_flat, beta=50.0):
    """
    Smooth surrogate: -softmin(log(areas)) + penalty for invalid barycentric.
    Lower is better.
    """
    uv = uv_flat.reshape(N, 2)
    uv = _project_to_simplex(uv)
    pts = _bary_to_cart(uv)
    p1 = pts[_TRIPLES[:, 0]]
    p2 = pts[_TRIPLES[:, 1]]
    p3 = pts[_TRIPLES[:, 2]]
    d1 = p2 - p1
    d2 = p3 - p1
    cross = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    areas = 0.5 * np.abs(cross)
    # Avoid log(0)
    areas_safe = np.maximum(areas, 1e-15)
    log_areas = np.log(areas_safe)
    # Softmin: -logsumexp(log_areas) / beta approximates min
    m = np.min(log_areas)
    softmin = m - np.log(np.sum(np.exp(log_areas - m))) / beta
    # We want to MAXIMIZE min area, so minimize -softmin
    return -softmin


def _structured_init_fps():
    """Generate structured initial barycentric configs using farthest-point sampling idea."""
    inits = []
    
    # Symmetric near corners + center
    b1 = np.array([
        [0.03, 0.03],
        [0.97, 0.03],
        [0.03, 0.97],
        [0.47, 0.03],
        [0.03, 0.47],
        [0.50, 0.47],
        [0.33, 0.33],
        [0.25, 0.55],
        [0.55, 0.25],
        [0.25, 0.25],
        [0.45, 0.30],
    ])
    inits.append(b1.copy())
    
    # Grid-like
    b2 = np.array([
        [0.05, 0.05],
        [0.30, 0.05],
        [0.55, 0.05],
        [0.80, 0.05],
        [0.15, 0.30],
        [0.40, 0.30],
        [0.65, 0.30],
        [0.25, 0.50],
        [0.50, 0.50],
        [0.35, 0.40],
        [0.60, 0.40],
    ])
    inits.append(b2.copy())
    
    # Ring + center
    b3 = np.array([
        [0.33, 0.33],
        [0.10, 0.10],
        [0.80, 0.10],
        [0.10, 0.80],
        [0.50, 0.05],
        [0.05, 0.50],
        [0.55, 0.50],
        [0.20, 0.55],
        [0.55, 0.20],
        [0.25, 0.25],
        [0.45, 0.35],
    ])
    inits.append(b3.copy())
    
    return inits


def _simulated_annealing(uv_init, start_time, deadline, beta_start=20.0, beta_end=200.0, n_steps=3000, seed=42):
    """Simulated annealing on barycentric (u,v) coordinates."""
    rng = np.random.RandomState(seed)
    uv = uv_init.copy()
    uv = _project_to_simplex(uv)
    
    best_uv = uv.copy()
    best_val = _smooth_min_objective(uv.flatten(), beta_start)
    best_raw = _min_area_from_bary(uv)[0]
    
    for step in range(n_steps):
        if time.time() > deadline - 5:
            break
        
        # Annealing schedule
        t = step / max(n_steps - 1, 1)
        beta = beta_start * (beta_end / beta_start) ** t
        step_size = 0.05 * (1.0 - 0.7 * t)
        
        # Propose move: perturb a random point
        idx = rng.randint(N)
        du = rng.normal(0, step_size)
        dv = rng.normal(0, step_size)
        uv_new = uv.copy()
        uv_new[idx, 0] += du
        uv_new[idx, 1] += dv
        uv_new = _project_to_simplex(uv_new)
        
        val_new = _smooth_min_objective(uv_new.flatten(), beta)
        val_cur = _smooth_min_objective(uv.flatten(), beta)
        
        delta = val_new - val_cur
        if delta < 0 or rng.random() < np.exp(-delta / max(0.01, 0.1 * (1 - t))):
            uv = uv_new
        
        # Track best by raw min area
        raw = _min_area_from_bary(uv)[0]
        if raw > best_raw:
            best_raw = raw
            best_uv = uv.copy()
    
    return best_uv, best_raw


def _local_optimize_smooth(uv_init, beta=80.0, start_time=None, deadline=None):
    """Local optimization using smooth surrogate with SLSQP or Nelder-Mead."""
    if start_time is None:
        start_time = time.time()
    if deadline is None:
        deadline = start_time + 20
    
    uv = uv_init.copy().flatten()
    
    def obj(x):
        return _smooth_min_objective(x, beta)
    
    def grad(x):
        # Numerical gradient
        eps = 1e-7
        g = np.zeros_like(x)
        for i in range(len(x)):
            xp = x.copy()
            xp[i] += eps
            xm = x.copy()
            xm[i] -= eps
            g[i] = (obj(xp) - obj(xm)) / (2 * eps)
        return g
    
    res = minimize(
        obj,
        uv,
        jac=grad,
        method='L-BFGS-B',
        options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-8}
    )
    
    uv_opt = _project_to_simplex(res.x.reshape(N, 2))
    raw = _min_area_from_bary(uv_opt)[0]
    return uv_opt, raw


def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    best_uv = None
    best_raw = -np.inf
    
    # Phase 1: Simulated annealing from multiple structured starts
    inits = _structured_init_fps()
    
    # Add random inits
    rng = np.random.RandomState(123)
    for _ in range(3):
        random_uv = rng.dirichlet([1, 1, 1], size=N)[:, :2]
        inits.append(random_uv)
    
    for i, init in enumerate(inits):
        if time.time() > deadline - 30:
            break
        try:
            uv_sa, raw_sa = _simulated_annealing(
                init, start_time, deadline,
                beta_start=20, beta_end=150,
                n_steps=2000, seed=42 + i
            )
            if raw_sa > best_raw:
                best_raw = raw_sa
                best_uv = uv_sa.copy()
        except Exception:
            continue
    
    # Phase 2: Local refinement with smooth surrogate
    if best_uv is not None:
        if time.time() < deadline - 15:
            try:
                uv_opt, raw_opt = _local_optimize_smooth(best_uv, beta=80, start_time=start_time, deadline=deadline)
                if raw_opt > best_raw:
                    best_raw = raw_opt
                    best_uv = uv_opt.copy()
            except Exception:
                pass
    
    # Phase 3: Additional SA rounds from best if time permits
    if best_uv is not None and time.time() < deadline - 10:
        try:
            uv_sa2, raw_sa2 = _simulated_annealing(
                best_uv, start_time, deadline,
                beta_start=10, beta_end=100,
                n_steps=1500, seed=999
            )
            if raw_sa2 > best_raw:
                best_raw = raw_sa2
                best_uv = uv_sa2.copy()
        except Exception:
            pass
    
    # Phase 4: Final local polish
    if best_uv is not None and time.time() < deadline - 5:
        try:
            uv_final, raw_final = _local_optimize_smooth(best_uv, beta=120, start_time=start_time, deadline=deadline)
            if raw_final > best_raw:
                best_raw = raw_final
                best_uv = uv_final.copy()
        except Exception:
            pass
    
    # Fallback
    if best_uv is None or not np.all(np.isfinite(best_uv)):
        fallback = np.array([
            [0.03, 0.03],
            [0.97, 0.03],
            [0.03, 0.97],
            [0.47, 0.03],
            [0.03, 0.47],
            [0.50, 0.47],
            [0.33, 0.33],
            [0.25, 0.55],
            [0.55, 0.25],
            [0.25, 0.25],
            [0.45, 0.30],
        ])
        best_uv = fallback
    
    best_uv = _project_to_simplex(best_uv)
    return _bary_to_cart(best_uv)
# EVOLVE-BLOCK-END
