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


def _energy_and_grad(x, p, eps):
    """Smooth sum of inverse powers of pairwise and wall distances.

    Returns (energy, grad) where x is a flat (2N,) vector of centers.
    """
    c = x.reshape(N, 2)

    # --- pairwise ---
    diff = c[:, None, :] - c[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    dclip = np.maximum(dist, eps)
    energy_pairs = np.sum(np.power(dclip, -p))

    # gradient of sum_{j!=i} d_ij^-p wrt c_i = -p * sum_j d_ij^(-p-2) * (c_i - c_j)
    weights = -p * np.power(dclip, -p - 2)
    np.fill_diagonal(weights, 0.0)
    sum_w = np.sum(weights, axis=1)
    weighted_c = weights @ c
    grads = c * sum_w[:, None] - weighted_c

    # --- walls ---
    d_left = HALF + c[:, 0]
    d_right = HALF - c[:, 0]
    d_bottom = HALF + c[:, 1]
    d_top = HALF - c[:, 1]
    w = np.minimum.reduce([d_left, d_right, d_bottom, d_top])
    wclip = np.maximum(w, eps)
    energy_walls = np.sum(np.power(wclip, -p))

    # gradient of w_i^-p wrt coords: -p w_i^(-p-1) * d(w_i)/d(coord)
    coeff = -p * np.power(wclip, -p - 1)
    tol = 1e-9
    gx = np.zeros(N)
    gy = np.zeros(N)
    gx += (np.abs(d_left - w) <= tol) * 1.0
    gx += (np.abs(d_right - w) <= tol) * (-1.0)
    gy += (np.abs(d_bottom - w) <= tol) * 1.0
    gy += (np.abs(d_top - w) <= tol) * (-1.0)
    grads[:, 0] += coeff * gx
    grads[:, 1] += coeff * gy

    return energy_pairs + energy_walls, grads.ravel()


def _physics_anneal(c0, t_start, deadline, rng, n_cycles=3):
    """Physics-based annealing: circles repel each other and walls.
    
    Uses velocity-Verlet integration with cooling schedule.
    Multiple cycles with reheating to escape local minima.
    """
    c = c0.copy()
    v = np.zeros_like(c)
    best_c = c.copy()
    best_r = _radius_of(c)
    
    # Parameters
    k_repel = 50.0       # repulsion strength
    k_wall = 100.0       # wall repulsion strength
    dt = 0.005           # time step
    damping = 0.95       # velocity damping
    T0 = 0.02            # initial temperature
    T_min = 0.001        # minimum temperature
    cooling_rate = 0.98  # geometric cooling per step
    
    for cycle in range(n_cycles):
        T = T0 if cycle == 0 else max(T_min, T * 0.5)
        
        # Run annealing until deadline
        while time.time() - t_start < deadline - 10.0:
            # Compute forces
            forces = np.zeros_like(c)
            
            # Pairwise repulsion: F_ij = k_repel * (c_i - c_j) / d_ij^3 for d_ij < 2*r_target
            diff = c[:, None, :] - c[None, :, :]
            dist = np.sqrt(np.sum(diff * diff, axis=2))
            np.fill_diagonal(dist, np.inf)
            
            # Only interact with neighbors within some cutoff for efficiency
            cutoff = 0.15
            mask = (dist < cutoff) & (dist > 1e-8)
            # Force = k_repel * (c_i - c_j) / d_ij^3
            inv_d3 = np.zeros_like(dist)
            inv_d3[mask] = 1.0 / (dist[mask] ** 3)
            np.fill_diagonal(inv_d3, 0.0)
            
            # Sum forces: F_i = sum_j k_repel * (c_i - c_j) / d_ij^3
            forces += k_repel * np.sum(inv_d3[:, :, None] * diff, axis=1)
            
            # Wall forces: push away from walls
            d_left = HALF + c[:, 0]
            d_right = HALF - c[:, 0]
            d_bottom = HALF + c[:, 1]
            d_top = HALF - c[:, 1]
            
            # Only apply if close to wall
            wall_dist = 0.05
            forces[:, 0] += k_wall * np.where(d_left < wall_dist, 1.0 / np.maximum(d_left, 1e-6), 0.0)
            forces[:, 0] -= k_wall * np.where(d_right < wall_dist, 1.0 / np.maximum(d_right, 1e-6), 0.0)
            forces[:, 1] += k_wall * np.where(d_bottom < wall_dist, 1.0 / np.maximum(d_bottom, 1e-6), 0.0)
            forces[:, 1] -= k_wall * np.where(d_top < wall_dist, 1.0 / np.maximum(d_top, 1e-6), 0.0)
            
            # Add thermal noise
            noise = rng.normal(0.0, np.sqrt(T), size=c.shape)
            
            # Velocity Verlet update
            v = v * damping + forces * dt + noise
            c = c + v * dt
            
            # Keep within bounds
            c = np.clip(c, -HALF + 1e-9, HALF - 1e-9)
            
            # Cool
            T *= cooling_rate
            
            # Check for improvement
            r = _radius_of(c)
            if r > best_r:
                best_r = r
                best_c = c.copy()
            
            # Stop if converged (temperature too low)
            if T < T_min:
                break
            
            # Check time
            if time.time() - t_start > deadline - 10.0:
                break
        
        # Reheat for next cycle
        if cycle < n_cycles - 1:
            # Small perturbation to escape local minima
            c = best_c + rng.normal(0.0, 0.01, size=best_c.shape)
            c = np.clip(c, -HALF + 1e-6, HALF - 1e-6)
            v = np.zeros_like(c)
    
    return best_c, best_r


def _optimize_energy(c0, p, eps, rem):
    """Run L-BFGS-B on the inverse-power objective; return (centers, radius)."""
    x0 = c0.ravel().copy()
    bounds = [(-HALF + 1e-9, HALF - 1e-9)] * (2 * N)

    def obj(x):
        e, _ = _energy_and_grad(x, p, eps)
        return e

    def grad(x):
        _, g = _energy_and_grad(x, p, eps)
        return g

    best_x = x0
    try:
        res = minimize(obj, x0, jac=grad, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 4000, 'ftol': 1e-14, 'gtol': 1e-10,
                                'maxls': 50})
        if res.x is not None and np.all(np.isfinite(res.x)):
            # accept if it improves the objective (or just succeeded)
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

    # ---- Phase B: L-BFGS-B polish (coarse) ----
    rem = deadline - (time.time() - t_start)
    if rem > 20.0:
        c_polish, r_polish = _optimize_energy(best_c, p=4.0, eps=eps, rem=rem - 5.0)
        if r_polish > best_r:
            best_r = r_polish
            best_c = c_polish

    # ---- Phase C: Fine polish (large power sharpens the min-distance) ----
    rem = deadline - (time.time() - t_start)
    if rem > 15.0:
        c_polish, r_polish = _optimize_energy(best_c, p=12.0, eps=1e-6, rem=rem - 5.0)
        if r_polish > best_r:
            best_r = r_polish
            best_c = c_polish

    # ---- Phase D: Ultra-fine final polish if time remains ----
    rem = deadline - (time.time() - t_start)
    if rem > 10.0:
        c_polish, r_polish = _optimize_energy(best_c, p=20.0, eps=1e-7, rem=rem - 5.0)
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
