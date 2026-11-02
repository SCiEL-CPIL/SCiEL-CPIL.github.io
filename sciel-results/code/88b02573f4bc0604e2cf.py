"""Circle packing in unit square: maximize sum of radii for 26 disjoint circles.
Strategy: Simulated Annealing with soft penalty objective, followed by SLSQP polish.
This explores the non-convex landscape globally by allowing temporary violations,
escaping local minima that trap gradient-based methods.
"""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 26
EPS = 1e-9
DEADLINE = 145.0  # safety margin below 150s kill


def _max_radius(pos, r):
    """Maximal radius for each circle given current positions and radii."""
    n = len(r)
    mr = np.empty(n)
    for i in range(n):
        dmin = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
        for j in range(n):
            if i != j:
                dmin = min(dmin, np.hypot(pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1]) - r[j])
        mr[i] = max(0.0, dmin)
    return mr


def _valid_check(arr, tol=1e-7):
    """Check all circles are in-square and non-overlapping."""
    for i in range(len(arr)):
        x, y, r = arr[i]
        if r < 0 or x < r + tol or x > 1 - r - tol or y < r + tol or y > 1 - r - tol:
            return False
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            d = np.hypot(arr[i, 0] - arr[j, 0], arr[i, 1] - arr[j, 1])
            if d < arr[i, 2] + arr[j, 2] - tol:
                return False
    return True


def _delaunay_edges(pos):
    """Return set of (i,j) pairs that are Delaunay neighbors."""
    try:
        tri = Delaunay(pos)
        edges = set()
        for simplex in tri.simplices:
            for a in range(3):
                for b in range(a + 1, 3):
                    i, j = simplex[a], simplex[b]
                    edges.add((min(i, j), max(i, j)))
        return edges
    except Exception:
        return [(i, j) for i in range(N) for j in range(i + 1, N)]


def _build_constraints(pos):
    """Build constraint functions for SLSQP based on Delaunay edges + walls."""
    edges = _delaunay_edges(pos)
    cons = []
    for i in range(N):
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*N + i] - x[2*i]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[2*i] - x[2*N + i]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*N + i] - x[2*i+1]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[2*i+1] - x[2*N + i]})
    for (i, j) in edges:
        cons.append({'type': 'ineq', 'fun': lambda x, i=i, j=j:
                     np.hypot(x[2*i] - x[2*j], x[2*i+1] - x[2*j+1]) - x[2*N+i] - x[2*N+j]})
    return cons, edges


def _slsqp_polish(pos, rad, maxiter=200):
    """Polish a near-valid configuration using SLSQP with hard constraints."""
    pos0 = pos.copy()
    rad0 = np.maximum(rad, 0.0)
    # Ensure initial feasibility by shrinking radii slightly if needed
    x0 = np.concatenate([pos0.reshape(-1), rad0])
    
    # Quick validity check and repair
    arr = np.column_stack([pos0, rad0])
    if not _valid_check(arr):
        rad0 = np.maximum(rad0 * 0.99, 0.0)
        x0 = np.concatenate([pos0.reshape(-1), rad0])
        arr = np.column_stack([pos0, rad0])
        if not _valid_check(arr):
            # More aggressive repair
            rad0 = np.maximum(rad0 * 0.95, 0.0)
            x0 = np.concatenate([pos0.reshape(-1), rad0])
            arr = np.column_stack([pos0, rad0])
            if not _valid_check(arr):
                return pos0, rad0

    cons, edges = _build_constraints(pos0)

    def obj(x):
        return -np.sum(x[2*N:])

    def jac(x):
        g = np.zeros_like(x)
        g[2*N:] = -1.0
        return g

    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
    for i in range(N):
        bounds.append((0.0, 0.5))

    try:
        res = minimize(obj, x0, jac=jac, method='SLSQP', bounds=bounds,
                       constraints=cons, options={'maxiter': maxiter, 'ftol': 1e-10})
        if res.success or np.sum(res.x[2*N:]) > np.sum(rad0):
            pos_new = res.x[:2*N].reshape(N, 2)
            rad_new = np.maximum(res.x[2*N:], 0.0)
            # Final validity check
            arr_new = np.column_stack([pos_new, rad_new])
            if _valid_check(arr_new):
                return pos_new, rad_new
    except Exception:
        pass
    return pos0, rad0


def _soft_penalty(pos, rad, penalty_weight):
    """Compute soft penalty objective: -sum(r) + penalty * violations^2."""
    # Wall violations
    wall_viol = 0.0
    for i in range(N):
        wall_viol += max(0, rad[i] - pos[i, 0]) ** 2
        wall_viol += max(0, pos[i, 0] + rad[i] - 1.0) ** 2
        wall_viol += max(0, rad[i] - pos[i, 1]) ** 2
        wall_viol += max(0, pos[i, 1] + rad[i] - 1.0) ** 2
    
    # Overlap violations
    overlap_viol = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            d = np.hypot(pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1])
            viol = max(0, rad[i] + rad[j] - d)
            overlap_viol += viol ** 2
    
    total_viol = wall_viol + overlap_viol
    return -np.sum(rad) + penalty_weight * total_viol


def _anneal(pos, rad, t_start, time_budget):
    """Simulated Annealing to find a good configuration."""
    rng = np.random.default_rng(42)
    best_pos = pos.copy()
    best_rad = rad.copy()
    best_score = -np.sum(rad)
    
    T = 0.05  # Initial temperature
    T_min = 1e-5
    alpha = 0.995  # Cooling rate
    penalty = 100.0  # Penalty weight
    
    current_score = _soft_penalty(pos, rad, penalty)
    
    t0 = time.time()
    steps = 0
    while time.time() - t0 < time_budget:
        # Generate a perturbation
        pos_new = pos.copy()
        rad_new = rad.copy()
        
        # Perturb positions of a random subset
        n_perturb = rng.integers(2, 6)
        idx = rng.choice(N, size=n_perturb, replace=False)
        pos_new[idx] += 0.03 * (rng.random((n_perturb, 2)) * 2 - 1)
        pos_new = np.clip(pos_new, 0.0, 1.0)
        
        # Perturb radii slightly
        rad_new[idx] += 0.01 * (rng.random(n_perturb) * 2 - 1)
        rad_new = np.clip(rad_new, 0.0, 0.5)
        
        # Optionally, try to grow radii
        if rng.random() < 0.3:
            rad_new = _max_radius(pos_new, rad_new)
        
        score_new = _soft_penalty(pos_new, rad_new, penalty)
        
        # Metropolis acceptance
        if score_new < current_score:
            accept = True
        else:
            delta = score_new - current_score
            if T > 0:
                accept = rng.random() < np.exp(-delta / T)
            else:
                accept = False
        
        if accept:
            pos = pos_new
            rad = rad_new
            current_score = score_new
            
            # Track best valid or near-valid solution
            # Use a stricter penalty for tracking "quality"
            if _valid_check(np.column_stack([pos, rad])):
                s = float(np.sum(rad))
                if s > best_score:
                    best_score = s
                    best_pos = pos.copy()
                    best_rad = rad.copy()
            elif score_new < best_score - 1e-6:
                # Even if not perfectly valid, track if it's much better
                pass
        
        # Cool down
        T *= alpha
        steps += 1
        
        # Restart if stuck
        if steps % 500 == 0 and T > T_min:
            # Slight reheating to escape local minima
            T = max(T * 1.5, T_min)
    
    return best_pos, best_rad


def construct():
    t_start = time.time()
    rng = np.random.default_rng(42)
    best = None
    best_score = -1.0

    def consider(pos, rad):
        nonlocal best, best_score
        arr = np.column_stack([pos, rad])
        if not _valid_check(arr):
            # Try to fix: reduce radii slightly
            rad_fixed = np.maximum(rad * 0.999, 0)
            arr_fixed = np.column_stack([pos, rad_fixed])
            if not _valid_check(arr_fixed):
                rad_fixed = np.maximum(rad * 0.99, 0)
                arr_fixed = np.column_stack([pos, rad_fixed])
                if not _valid_check(arr_fixed):
                    return
            rad = rad_fixed
            arr = arr_fixed
        s = float(arr[:, 2].sum())
        if s > best_score:
            best_score = s
            best = arr

    # ---- Phase A: Generate diverse initializations ----
    starts = []

    # Grid initializations
    for side in range(4, 8):
        if time.time() > DEADLINE - 40:
            break
        cells = []
        for i in range(side):
            for j in range(side):
                cells.append(((i + 0.5) / side, (j + 0.5) / side))
        cells = cells[:N]
        pos = np.array(cells, dtype=float)
        rad = np.full(N, 0.5 / side - 1e-4)
        rad = _max_radius(pos, rad)
        starts.append((pos.copy(), rad.copy()))

        # Perturbed grids
        for k in range(2):
            if time.time() > DEADLINE - 40:
                break
            p0 = pos + 0.05 * (rng.random((N, 2)) * 2 - 1)
            p0 = np.clip(p0, 0.05, 0.95)
            r0 = np.full(N, 0.5 / side)
            r0 = _max_radius(p0, r0)
            starts.append((p0, r0))

    # Hexagonal packing initialization
    if time.time() <= DEADLINE - 40:
        hex_pos = []
        r_hex = 0.12
        dx = 2 * r_hex
        dy = r_hex * np.sqrt(3)
        for row in range(8):
            for col in range(8):
                x = col * dx + (row % 2) * dx / 2 + 0.15
                y = row * dy + 0.15
                if 0.05 < x < 0.95 and 0.05 < y < 0.95:
                    hex_pos.append((x, y))
        hex_pos = hex_pos[:N]
        if len(hex_pos) >= N:
            hp = np.array(hex_pos, dtype=float)
            hr = _max_radius(hp, np.full(N, 0.1))
            starts.append((hp, hr))

    # Random initializations
    for k in range(6):
        if time.time() > DEADLINE - 40:
            break
        p0 = rng.random((N, 2)) * 0.8 + 0.1
        r0 = np.full(N, 0.05)
        r0 = _max_radius(p0, r0)
        starts.append((p0, r0))

    # ---- Phase B: Anneal from each start ----
    for idx, (pos0, rad0) in enumerate(starts):
        if time.time() > DEADLINE - 30:
            break
        budget = min(10.0, max(3.0, (DEADLINE - 30 - time.time()) / max(1, len(starts) - idx)))
        pos_a, rad_a = _anneal(pos0, rad0, t_start, time_budget=budget)
        consider(pos_a, rad_a)

    # ---- Phase C: Polish the best solution found ----
    if best is not None:
        pos_b = best[:, :2].copy()
        rad_b = best[:, 2].copy()
        pos_p, rad_p = _slsqp_polish(pos_b, rad_b)
        consider(pos_p, rad_p)

    # ---- Phase D: Local search perturbation on best ----
    if best is not None and time.time() < DEADLINE - 15:
        for k in range(5):
            if time.time() > DEADLINE - 5:
                break
            pos_p = best[:, :2].copy()
            rad_p = best[:, 2].copy()
            n_perturb = rng.integers(3, 8)
            idx_perturb = rng.choice(N, size=n_perturb, replace=False)
            pos_p[idx_perturb] += 0.02 * (rng.random((n_perturb, 2)) * 2 - 1)
            pos_p = np.clip(pos_p, 0.05, 0.95)
            rad_p = _max_radius(pos_p, rad_p)
            pos_f, rad_f = _slsqp_polish(pos_p, rad_p)
            consider(pos_f, rad_f)

    if best is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-4)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    return best


if __name__ == '__main__':
    out = construct()
    print("sum of radii:", out[:, 2].sum())
    print(out)
