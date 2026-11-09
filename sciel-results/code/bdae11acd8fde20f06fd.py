"""Smooth inverse-power-sum packing: hex seed + physics-based annealing + L-BFGS-B polish.

Structural idea: minimize a smooth sum of inverse powers of pairwise and wall
distances.  The core optimizer is a physics-based annealing loop (velocity-Verlet
with cooling) that models circles as charged particles repelling each other and
being pushed from walls.  Multiple annealing cycles with reheating escape local
minima.  A final L-BFGS-B polish sharpens the minimum distance.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 200
HALF = 0.5
SEED = 1234567


def _radius_of(c):
    """True packing radius for centers c (N,2)."""
    c = np.asarray(c)
    wall = np.min(HALF - np.abs(c))
    d = c[:, None, :] - c[None, :, :]
    dist = np.sqrt(np.sum(d * d, axis=2))
    np.fill_diagonal(dist, np.inf)
    pair = np.min(dist) / 2.0
    return float(min(wall, pair))


def _hexagon_seed(n):
    """Staggered hexagonal rows inside [-1/2,1/2]^2, return n centers.

    Selects the n points closest to the centre so the packing is symmetric and
    boundary-aware (drops the outermost, least-well-supported points).
    """
    for attempt in range(200):
        r0 = 0.036
        h = 2 * r0
        v = r0 * np.sqrt(3)
        x0 = -HALF + r0
        y0 = -HALF + r0
        n_rows = int((1 - 2 * r0) / v) + 1
        pts = []
        for i in range(n_rows):
            y = y0 + i * v
            x_start = x0 + (h / 2 if i % 2 else 0.0)
            if x_start > HALF - r0:
                continue
            n_cols = int((HALF - r0 - x_start) / h) + 1
            for j in range(n_cols):
                x = x_start + j * h
                pts.append([x, y])
        pts = np.array(pts)
        if len(pts) >= n:
            if len(pts) == n:
                return pts
            # keep the n closest to centre for a symmetric, well-supported set
            dctr = np.linalg.norm(pts, axis=1)
            order = np.argsort(dctr)
            return pts[order[:n]]
        r0 *= 0.995
    m = int(np.ceil(np.sqrt(n)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:n]


def _soft_min_energy_grad(x, alpha, eps):
    """Log-sum-exp based soft-min of pairwise and wall distances.

    Minimizes -log(sum exp(-alpha*d)) which approximates -min(d) as alpha->inf.
    Returns (energy, grad) where x is flat (2N,).
    """
    c = x.reshape(N, 2)

    # --- pairwise ---
    diff = c[:, None, :] - c[None, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    dist = np.sqrt(dist_sq)
    dclip = np.maximum(dist, eps)

    # -alpha * d
    a_d = -alpha * dclip
    # log-sum-exp for numerical stability
    a_d_max = np.max(a_d, axis=1, keepdims=True)
    log_sum_exp = a_d_max + np.log(np.sum(np.exp(a_d - a_d_max), axis=1, keepdims=True))
    # energy = -sum(log_sum_exp)
    energy_pairs = -np.sum(log_sum_exp)

    # grad: d(-log(sum_j exp(-alpha*d_ij)))/d(c_i) 
    # = sum_j [alpha * exp(-alpha*d_ij) / sum_k exp(-alpha*d_ik)] * (c_i - c_j)/d_ij
    weights = np.exp(a_d - a_d_max) / np.sum(np.exp(a_d - a_d_max), axis=1, keepdims=True)
    np.fill_diagonal(weights, 0.0)
    inv_d = np.zeros_like(dclip)
    mask = dclip > eps
    inv_d[mask] = 1.0 / dclip[mask]
    np.fill_diagonal(inv_d, 0.0)
    # weight * (c_i - c_j) / d_ij
    weighted_diff = weights[:, :, None] * diff * inv_d[:, :, None]
    grads = alpha * np.sum(weighted_diff, axis=1)

    # --- walls ---
    d_left = HALF + c[:, 0]
    d_right = HALF - c[:, 0]
    d_bottom = HALF + c[:, 1]
    d_top = HALF - c[:, 1]
    walls = np.stack([d_left, d_right, d_bottom, d_top], axis=1)  # (N, 4)
    wclip = np.maximum(walls, eps)
    a_w = -alpha * wclip
    a_w_max = np.max(a_w, axis=1, keepdims=True)
    log_sum_exp_w = a_w_max + np.log(np.sum(np.exp(a_w - a_w_max), axis=1, keepdims=True))
    energy_walls = -np.sum(log_sum_exp_w)

    # grad for walls
    w_weights = np.exp(a_w - a_w_max) / np.sum(np.exp(a_w - a_w_max), axis=1, keepdims=True)
    # d(wall_k)/d(coord) for each wall
    # left: +1, right: -1, bottom: +1, top: -1
    wall_grads = np.array([[1.0, -1.0, 1.0, -1.0]])  # (1, 4)
    grads_walls = alpha * np.sum(w_weights * wall_grads, axis=1)  # (N,)
    # For x-coordinate: left(+1), right(-1); for y: bottom(+1), top(-1)
    gx_w = alpha * (w_weights[:, 0] - w_weights[:, 1])
    gy_w = alpha * (w_weights[:, 2] - w_weights[:, 3])
    grads[:, 0] += gx_w
    grads[:, 1] += gy_w

    return energy_pairs + energy_walls, grads.ravel()


def _physics_anneal(c0, t_start, deadline, rng, n_cycles=4):
    """Physics-based annealing with gradient-based refinement steps.
    
    Alternates between physics-based moves and gradient steps on soft-min
    objective for faster convergence.
    """
    c = c0.copy()
    v = np.zeros_like(c)
    best_c = c.copy()
    best_r = _radius_of(c)
    
    # Parameters
    k_repel = 80.0
    k_wall = 200.0
    dt = 0.003
    damping = 0.9
    T0 = 0.015
    T_min = 0.0005
    cooling_rate = 0.985
    alpha = 50.0  # soft-min sharpness
    lr = 0.001   # gradient step size
    
    for cycle in range(n_cycles):
        T = T0 if cycle == 0 else max(T_min, T * 0.4)
        
        while time.time() - t_start < deadline - 10.0:
            # Compute forces
            forces = np.zeros_like(c)
            
            diff = c[:, None, :] - c[None, :, :]
            dist_sq = np.sum(diff * diff, axis=2)
            np.fill_diagonal(dist_sq, np.inf)
            dist = np.sqrt(dist_sq)
            
            cutoff = 0.12
            mask = (dist < cutoff) & (dist > 1e-8)
            inv_d3 = np.zeros_like(dist)
            inv_d3[mask] = 1.0 / (dist[mask] ** 3)
            np.fill_diagonal(inv_d3, 0.0)
            
            forces += k_repel * np.sum(inv_d3[:, :, None] * diff, axis=1)
            
            d_left = HALF + c[:, 0]
            d_right = HALF - c[:, 0]
            d_bottom = HALF + c[:, 1]
            d_top = HALF - c[:, 1]
            
            wall_dist = 0.04
            forces[:, 0] += k_wall * np.where(d_left < wall_dist, 1.0 / np.maximum(d_left, 1e-6), 0.0)
            forces[:, 0] -= k_wall * np.where(d_right < wall_dist, 1.0 / np.maximum(d_right, 1e-6), 0.0)
            forces[:, 1] += k_wall * np.where(d_bottom < wall_dist, 1.0 / np.maximum(d_bottom, 1e-6), 0.0)
            forces[:, 1] -= k_wall * np.where(d_top < wall_dist, 1.0 / np.maximum(d_top, 1e-6), 0.0)
            
            noise = rng.normal(0.0, np.sqrt(T), size=c.shape)
            
            v = v * damping + forces * dt + noise
            c = c + v * dt
            c = np.clip(c, -HALF + 1e-9, HALF - 1e-9)
            
            T *= cooling_rate
            
            # Every 20 steps, do a gradient refinement
            if int((T0 / T) * 100) % 20 == 0 and T < T0 * 0.5:
                x_flat = c.ravel()
                _, grad = _soft_min_energy_grad(x_flat, alpha, 1e-7)
                c_new = c - lr * grad.reshape(N, 2)
                c_new = np.clip(c_new, -HALF + 1e-9, HALF - 1e-9)
                r_new = _radius_of(c_new)
                if r_new > best_r:
                    best_r = r_new
                    best_c = c_new.copy()
                    c = c_new
                    v = np.zeros_like(c)
            
            r = _radius_of(c)
            if r > best_r:
                best_r = r
                best_c = c.copy()
            
            if T < T_min:
                break
            
            if time.time() - t_start > deadline - 10.0:
                break
        
        if cycle < n_cycles - 1:
            c = best_c + rng.normal(0.0, 0.008, size=best_c.shape)
            c = np.clip(c, -HALF + 1e-6, HALF - 1e-6)
            v = np.zeros_like(c)
    
    return best_c, best_r


def _optimize_energy(c0, alpha, eps, rem):
    """Run L-BFGS-B on soft-min objective; return (centers, radius)."""
    x0 = c0.ravel().copy()
    bounds = [(-HALF + 1e-9, HALF - 1e-9)] * (2 * N)

    def obj(x):
        e, _ = _soft_min_energy_grad(x, alpha, eps)
        return e

    def grad(x):
        _, g = _soft_min_energy_grad(x, alpha, eps)
        return g

    best_x = x0
    try:
        res = minimize(obj, x0, jac=grad, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 6000, 'ftol': 1e-15, 'gtol': 1e-11,
                                'maxls': 100})
        if res.x is not None and np.all(np.isfinite(res.x)):
            if res.fun < obj(x0) or res.success:
                best_x = res.x
    except Exception:
        pass

    best_c = np.clip(best_x.reshape(N, 2), -HALF + 1e-9, HALF - 1e-9)
    r = _radius_of(best_c)
    return best_c, r


def construct():
    t_start = time.time()
    deadline = 115.0  # safety margin vs 150s wall
    rng = np.random.default_rng(SEED)

    c0 = _hexagon_seed(N)
    c0 = np.clip(c0, -HALF + 1e-6, HALF - 1e-6)
    best_c = c0.copy()
    best_r = _radius_of(best_c)

    eps = 1e-5

    # ---- Phase A: Physics-based annealing (main optimization) ----
    # Multiple cycles with reheating to escape local minima
    t_anneal_end = deadline - 30.0  # leave time for polish
    c_anneal, r_anneal = _physics_anneal(best_c, t_start, t_anneal_end, rng, n_cycles=3)
    if r_anneal > best_r:
        best_r = r_anneal
        best_c = c_anneal

    # ---- Phase B: L-BFGS-B polish (moderate sharpness) ----
    rem = deadline - (time.time() - t_start)
    if rem > 20.0:
        c_polish, r_polish = _optimize_energy(best_c, alpha=30.0, eps=1e-6, rem=rem - 5.0)
        if r_polish > best_r:
            best_r = r_polish
            best_c = c_polish

    # ---- Phase C: Fine polish (higher sharpness) ----
    rem = deadline - (time.time() - t_start)
    if rem > 15.0:
        c_polish, r_polish = _optimize_energy(best_c, alpha=80.0, eps=1e-7, rem=rem - 5.0)
        if r_polish > best_r:
            best_r = r_polish
            best_c = c_polish

    # ---- Phase D: Ultra-fine final polish ----
    rem = deadline - (time.time() - t_start)
    if rem > 10.0:
        c_polish, r_polish = _optimize_energy(best_c, alpha=200.0, eps=1e-8, rem=rem - 5.0)
        if r_polish > best_r:
            best_r = r_polish
            best_c = c_polish

    # final safety
    best_c = np.clip(best_c, -HALF + 1e-9, HALF - 1e-9)
    if not np.all(np.isfinite(best_c)):
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        best_c = np.array([(x, y) for y in ticks for x in ticks])[:N]
    return best_c


if __name__ == "__main__":
    c = construct()
    print("shape", c.shape, "radius", _radius_of(c))
