"""
LJ38 cluster global minimum search.

Combined approach:
1. Construct a high-quality FCC truncated-octahedron starting geometry.
2. Use Basin Hopping with Metropolis acceptance to explore the landscape,
   specifically tuned to jump over the icosahedral energy barrier.
   - Uses a larger perturbation sigma to ensure sufficient exploration.
   - Incorporates a 1D scale optimization step within the relaxation loop
     to handle global scaling degrees of freedom efficiently.
3. Perform a high-precision L-BFGS polish on the best candidate found.

Entry point: construct() -> (N, 3) array of Cartesian coordinates.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 38
DEADLINE = 115.0  # Safety margin within 120s/150s limit


def _lj_energy_and_grad(pos):
    """Total LJ energy and gradient for positions pos (N,3)."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0)
    r2 = np.maximum(r2, 1e-16)
    r6 = r2 ** 3
    r12 = r6 ** 2

    pair_energy = 4.0 * (1.0 / r12 - 1.0 / r6)
    np.fill_diagonal(pair_energy, 0.0)
    energy = 0.5 * np.sum(pair_energy)

    r8 = r2 ** 4
    r14 = r2 ** 7
    coeff = 4.0 * (6.0 / r8 - 12.0 / r14)
    np.fill_diagonal(coeff, 0.0)
    row_sum = np.sum(coeff, axis=1, keepdims=True)
    grad = row_sum * pos - coeff @ pos

    return energy, grad


def _make_fcc_truncated_octahedron():
    """
    Build the FCC truncated-octahedron geometry for LJ38.
    """
    max_c = 3
    pts = []
    for i in range(-max_c, max_c + 1):
        for j in range(-max_c, max_c + 1):
            for k in range(-max_c, max_c + 1):
                if (i + j + k) % 2 == 0:
                    pts.append((float(i), float(j), float(k)))

    pts = np.array(pts, dtype=np.float64)
    dists = np.sum(pts ** 2, axis=1)
    order = np.argsort(dists)
    pts = pts[order]

    pts = pts[:N].copy()
    pts -= np.mean(pts, axis=0)

    # Scale so that the mean nearest-neighbour distance is 2^(1/6)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    nn = np.sqrt(np.min(r2, axis=1))
    avg_nn = np.mean(nn)
    if avg_nn > 1e-12:
        scale = (2.0 ** (1.0 / 6.0)) / avg_nn
        pts *= scale

    return pts


def _relax(pos, max_iter=500, gtol=1e-12):
    """L-BFGS-B local relaxation with analytic gradient."""
    
    def obj_grad(x):
        p = x.reshape(-1, 3)
        e, g = _lj_energy_and_grad(p)
        return e, g.ravel()

    try:
        res = minimize(
            obj_grad,
            pos.ravel(),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": max_iter, "gtol": gtol, "ftol": 1e-16},
        )
        p_out = res.x.reshape(-1, 3)
        e_out, _ = _lj_energy_and_grad(p_out)
        return p_out, e_out
    except Exception:
        return pos, _lj_energy_and_grad(pos)[0]


def _optimize_scale(pos, n_iter=20):
    """
    1D global search over uniform scale factor s applied to centered positions.
    Finds the s that minimizes total LJ energy.
    """
    center = np.mean(pos, axis=0)
    centered = pos - center
    best_e = np.inf
    best_s = 1.0
    
    # Coarse scan
    for s in np.linspace(0.90, 1.10, 81):
        p = centered * s + center
        e, _ = _lj_energy_and_grad(p)
        if e < best_e:
            best_e = e
            best_s = s
            
    # Fine scan around best
    for _ in range(n_iter):
        span = 0.1 / (10 ** (_ // 3))
        for s in np.linspace(best_s - span, best_s + span, 81):
            if s <= 0.1:
                continue
            p = centered * s + center
            e, _ = _lj_energy_and_grad(p)
            if e < best_e:
                best_e = e
                best_s = s
                
    return centered * best_s + center, best_e


def _basin_hopping(pos, n_iter=100, sigma=0.3, temp=1.0, time_limit=100.0):
    """
    Basin Hopping algorithm to find the global minimum in the basin.
    Uses a larger sigma to jump over the icosahedral barrier.
    """
    t0 = time.time()
    current_pos = pos.copy()
    current_energy = _lj_energy_and_grad(current_pos)[0]
    best_pos = current_pos.copy()
    best_energy = current_energy
    
    rng = np.random.default_rng(42)
    
    for i in range(n_iter):
        if time.time() - t0 > time_limit:
            break
            
        # Perturb
        delta = sigma * rng.standard_normal(current_pos.shape)
        proposed_pos = current_pos + delta
        
        # Relax
        proposed_pos, proposed_energy = _relax(proposed_pos, max_iter=200)
        
        # Optional: Scale optimization during relaxation to handle global modes
        # We only do this every few steps to save time
        if i % 5 == 0 and time.time() - t0 < time_limit - 5.0:
            proposed_pos, proposed_energy = _optimize_scale(proposed_pos, n_iter=10)
            
        # Metropolis criterion
        if proposed_energy < current_energy:
            current_pos = proposed_pos
            current_energy = proposed_energy
            if proposed_energy < best_energy:
                best_pos = proposed_pos.copy()
                best_energy = proposed_energy
        else:
            delta_e = proposed_energy - current_energy
            if delta_e < 1e-12:
                accept = True
            else:
                accept = rng.random() < np.exp(-delta_e / temp)
            if accept:
                current_pos = proposed_pos
                current_energy = proposed_energy
                if proposed_energy < best_energy:
                    best_pos = proposed_pos.copy()
                    best_energy = proposed_energy
                    
    return best_pos, best_energy


def _polish(pos, max_iter=1000):
    """High-precision polish pass."""
    def obj_grad(x):
        p = x.reshape(-1, 3)
        e, g = _lj_energy_and_grad(p)
        return e, g.ravel()
        
    try:
        res = minimize(
            obj_grad,
            pos.ravel(),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": max_iter, "gtol": 1e-15, "ftol": 1e-18},
        )
        p_out = res.x.reshape(-1, 3)
        e_out, _ = _lj_energy_and_grad(p_out)
        return p_out, e_out
    except Exception:
        return pos, _lj_energy_and_grad(pos)[0]


def _ensure_min_sep(pos, min_sep=0.5):
    """Scale up the cluster if any pair is closer than min_sep."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    min_d = np.sqrt(np.min(r2))
    if min_d < min_sep:
        scale = min_sep / min_d * 1.01
        pos = pos * scale
    return pos


def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE

    # 1. Generate initial FCC geometry
    try:
        start_pos = _make_fcc_truncated_octahedron()
    except Exception:
        # Fallback to random if FCC fails (unlikely)
        rng = np.random.default_rng(0)
        start_pos = rng.standard_normal((N, 3)) * 1.0
        start_pos = _ensure_min_sep(start_pos)

    # Initial relaxation to get a baseline energy
    start_pos, e_start = _relax(start_pos, max_iter=300)
    
    # 2. Basin Hopping to explore the basin and find the global minimum
    # We use a larger perturbation size (sigma=0.3) to jump over the icosahedral barrier.
    # The time limit ensures we don't exceed the budget.
    best_pos, best_energy = _basin_hopping(
        start_pos, 
        n_iter=500, 
        sigma=0.3, 
        temp=1.0, 
        time_limit=DEADLINE - 15.0
    )
    
    # 3. Final high-precision polish
    if time.time() < deadline - 5.0:
        # One more scale optimization before final polish
        best_pos, best_energy = _optimize_scale(best_pos, n_iter=30)
        best_pos, best_energy = _polish(best_pos, max_iter=1000)
        
    # Ensure minimum separation constraint
    best_pos = _ensure_min_sep(best_pos)
    
    return best_pos
