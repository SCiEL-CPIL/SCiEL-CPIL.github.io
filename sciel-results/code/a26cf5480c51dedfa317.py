"""Circle packing in perimeter-4 rectangle: DE global + SLSQP constrained + penalty polish.

Strategy:
  1. For a set of aspect ratios (w + h = 2), generate a grid seed.
  2. Use differential evolution to explore (w + centers) to find promising basins.
  3. Run SLSQP directly on (radii, centers) with explicit wall + non-overlap
     constraints to push radii up (radii are explicit variables).
  4. Final penalty-based gradient polish to squeeze remaining slack.
  5. Return best feasible packing (centers, radii) of shape (N, 3).
"""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 21
_START = time.time()
DEADLINE = _START + 132.0  # safety margin before 150 s kill


def _time_ok(margin=0.0):
    return time.time() < DEADLINE - margin


def _feasible_penalty(x, w, h, mu=1000.0):
    """Return (penalty, objective=-sum(r)) for x=[r, cx, cy]."""
    r = x[:N]
    cx = x[N:2 * N]
    cy = x[2 * N:3 * N]
    if w <= 0.1 or h <= 0.1:
        return 1e6, 0.0

    # Wall violations
    d_min = np.minimum(np.minimum(cx, cy), np.minimum(w - cx, h - cy))
    viol_wall = np.maximum(0.0, r - d_min)
    pen = mu * np.sum(viol_wall ** 2)

    # Pairwise overlap violations
    for i in range(N):
        for j in range(i + 1, N):
            d2 = (cx[i] - cx[j]) ** 2 + (cy[i] - cy[j]) ** 2
            s = r[i] + r[j]
            if d2 < s * s:
                pen += mu * (s * s - d2)
    return pen, -np.sum(r)


def _max_radii(centers, w, h):
    """Maximum feasible radius per circle given centers and box."""
    n = len(centers)
    radii = np.zeros(n)
    for i in range(n):
        xi, yi = centers[i]
        r = min(xi, w - xi, yi, h - yi)
        if r <= 0:
            radii[i] = 0.0
            continue
        for j in range(n):
            if i == j:
                continue
            d = np.hypot(xi - centers[j][0], yi - centers[j][1])
            r = min(r, d / 2.0)
            if r <= 0:
                break
        radii[i] = max(0.0, r)
    return radii


def _check_validity(centers, w, h, radii, tol=1e-7):
    for i in range(N):
        xi, yi = centers[i]
        ri = radii[i]
        if ri < 1e-9:
            return False
        if xi < -tol or xi > w + tol or yi < -tol or yi > h + tol:
            return False
        if ri > min(xi, w - xi, yi, h - yi) + tol:
            return False
    for i in range(N):
        for j in range(i + 1, N):
            d = np.hypot(centers[i][0] - centers[j][0], centers[i][1] - centers[j][1])
            if d < radii[i] + radii[j] - tol:
                return False
    return True


def _grid_seed(w, h):
    """Grid seed x=[r, cx, cy] for SLSQP. Returns best feasible grid."""
    best_seed = None
    best_score = -1e9
    for cols in range(1, N + 1):
        rows = (N + cols - 1) // cols
        if cols * rows < N:
            continue
        cw = w / cols
        ch = h / rows
        r = 0.5 * min(cw, ch) * 0.999
        x = np.zeros(3 * N)
        idx = 0
        for j in range(rows):
            for i in range(cols):
                if idx >= N:
                    break
                x[idx] = r
                x[N + idx] = (i + 0.5) * cw
                x[2 * N + idx] = (j + 0.5) * ch
                idx += 1
        if _check_validity(x[N:2 * N], w, h, x[:N], tol=1e-6):
            s = float(np.sum(x[:N]))
            if s > best_score:
                best_score = s
                best_seed = x.copy()
    if best_seed is None:
        # fallback: small circles in a row
        w2, h2 = w, h
        x = np.zeros(3 * N)
        x[:N] = 1e-3
        xs = np.linspace(1e-3, w2 - 1e-3, N)
        x[N:2 * N] = xs
        x[2 * N:3 * N] = h2 / 2.0
        best_seed = x
    return best_seed


def _slsqp_optimize(x0, w, h, time_budget=30.0):
    """SLSQP on (r, cx, cy) with wall + non-overlap constraints."""
    h_val = h
    cons = []
    for i in range(N):
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[i] - 1e-5})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[N + i] - x[i]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: (w - x[N + i]) - x[i]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2 * N + i] - x[i]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: (h_val - x[2 * N + i]) - x[i]})
    for i in range(N):
        for j in range(i + 1, N):
            cons.append({
                'type': 'ineq',
                'fun': lambda x, i=i, j=j: (
                    (x[N + i] - x[N + j]) ** 2 +
                    (x[2 * N + i] - x[2 * N + j]) ** 2 -
                    (x[i] + x[j]) ** 2
                ),
            })

    bounds = []
    for i in range(N):
        bounds.append((1e-5, min(w, h) / 2.0))
    for i in range(N):
        bounds.append((0.0, w))
    for i in range(N):
        bounds.append((0.0, h))

    start = time.time()
    max_iter = 400 if time_budget > 25 else 200
    try:
        res = minimize(
            lambda x: -np.sum(x[:N]),
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': max_iter, 'ftol': 1e-12},
        )
    except Exception:
        return x0
    x = res.x
    x[:N] = np.maximum(x[:N], 1e-5)
    return x


def _penalty_polish(x, w, h, max_iter=150, mu=500.0):
    """Gradient-based polish using penalty objective."""
    x = x.copy()
    for it in range(max_iter):
        if not _time_ok(5.0):
            break
        pen, obj = _feasible_penalty(x, w, h, mu=mu)
        if pen < 1e-8:
            break
        # Simple Nelder-Mead style or coordinate nudge on radii to reduce penalty
        improved = False
        # Try increasing radii slightly where there is slack
        for i in range(N):
            d_min = min(x[N + i], h - x[2 * N + i], w - x[N + i], h - x[2 * N + i])
            # pairwise slack
            for j in range(N):
                if i == j:
                    continue
                d = np.hypot(x[N + i] - x[N + j], x[2 * N + i] - x[2 * N + j])
                d_min = min(d_min, (d - x[j]) if d > x[j] else 0.0)
            max_r = max(1e-5, d_min * 0.9999)
            if max_r > x[i] + 1e-7:
                trial = x.copy()
                trial[i] = min(max_r, x[i] * 1.01 + 1e-4)
                pen2, _ = _feasible_penalty(trial, w, h, mu=mu)
                if pen2 <= pen + 1e-12 and trial[i] > x[i]:
                    x = trial
                    improved = True
        if not improved:
            break
    return x


def _de_objective(x):
    w = x[0]
    h = 2.0 - w
    if w < 0.3 or h < 0.3:
        return 1e6
    centers = x[1:].reshape(N, 2)
    centers = np.clip(centers, 0.0, np.array([w, h]))
    radii = _max_radii(centers, w, h)
    return -np.sum(radii)


def _differential_evolution(objective, bounds, pop_size=25, max_iter=150,
                            mutation=(0.5, 1.0), crossover_prob=0.7,
                            time_limit=40.0):
    n_dims = len(bounds)
    start = time.time()
    pop = np.zeros((pop_size, n_dims))
    for i in range(pop_size):
        for j in range(n_dims):
            pop[i, j] = random.uniform(bounds[j][0], bounds[j][1])
    fitness = np.array([objective(ind) for ind in pop])
    best_idx = np.argmin(fitness)
    best_x = pop[best_idx].copy()
    best_f = fitness[best_idx]
    for gen in range(max_iter):
        if time.time() - start > time_limit:
            break
        new_pop = []
        new_fitness = []
        for i in range(pop_size):
            r1, r2, r3 = random.sample(range(pop_size), 3)
            F = random.uniform(*mutation)
            mutant = pop[r1] + F * (pop[r2] - pop[r3])
            for j in range(n_dims):
                mutant[j] = np.clip(mutant[j], bounds[j][0], bounds[j][1])
            trial = pop[i].copy()
            for j in range(n_dims):
                if random.random() < crossover_prob:
                    trial[j] = mutant[j]
            f_trial = objective(trial)
            if f_trial <= fitness[i]:
                new_pop.append(trial)
                new_fitness.append(f_trial)
            else:
                new_pop.append(pop[i])
                new_fitness.append(fitness[i])
        pop = np.array(new_pop)
        fitness = np.array(new_fitness)
        gi = np.argmin(fitness)
        if fitness[gi] < best_f:
            best_f = fitness[gi]
            best_x = pop[gi].copy()
    return best_x, best_f


def construct():
    best_val = -1e9
    best_arr = None

    aspect_ratios = [1.0, 1.2, 0.85, 1.3, 0.75, 1.1, 0.9, 1.4, 0.7, 1.25, 0.8]

    for w0 in aspect_ratios:
        if not _time_ok(30.0):
            break
        if w0 <= 0.2 or w0 >= 1.8:
            continue
        h0 = 2.0 - w0

        # Build DE bounds
        bounds = [(max(0.3, w0 - 0.25), min(1.7, w0 + 0.25))]
        for _ in range(N * 2):
            bounds.append((0.01, 1.99))

        # Grid seed for DE initial population centering
        seed_x = _grid_seed(w0, h0)
        centers0 = seed_x[N:2 * N].copy()
        radii0 = seed_x[:N].copy()

        x0 = np.concatenate([[w0], centers0.flatten()])
        x0 = x0 + 0.04 * np.random.randn(len(x0))
        x0[0] = np.clip(x0[0], 0.3, 1.7)

        time_left = DEADLINE - time.time() - 35.0
        if time_left < 8:
            # Not enough time for DE; go straight to SLSQP on grid seed
            slsqp_x = _slsqp_optimize(seed_x, w0, h0, time_budget=20.0)
            slsqp_x = _penalty_polish(slsqp_x, w0, h0, max_iter=80)
            if _check_validity(slsqp_x[N:2 * N], w0, h0, slsqp_x[:N]):
                val = float(np.sum(slsqp_x[:N]))
                if val > best_val:
                    best_val = val
                    best_arr = np.column_stack([slsqp_x[N:2 * N], slsqp_x[2 * N:3 * N], slsqp_x[:N]])
            continue

        pop_size = 25 if time_left > 50 else 15
        max_iter = 120 if time_left > 50 else 60
        try:
            best_x, best_f = _differential_evolution(
                _de_objective, bounds,
                pop_size=pop_size,
                max_iter=max_iter,
                mutation=(0.5, 1.0),
                crossover_prob=0.7,
                time_limit=max(5.0, time_left - 25.0),
            )
        except Exception:
            best_x = x0

        w = best_x[0]
        h = 2.0 - w
        centers = best_x[1:].reshape(N, 2)
        centers = np.clip(centers, 0.0, np.array([w, h]))
        radii = _max_radii(centers, w, h)

        # Build SLSQP seed from DE result
        slsqp_seed = np.concatenate([radii, centers[:, 0], centers[:, 1]])
        slsqp_x = _slsqp_optimize(slsqp_seed, w, h, time_budget=25.0)
        slsqp_x = _penalty_polish(slsqp_x, w, h, max_iter=100)

        if _check_validity(slsqp_x[N:2 * N], w, h, slsqp_x[:N]):
            val = float(np.sum(slsqp_x[:N]))
            if val > best_val:
                best_val = val
                best_arr = np.column_stack([slsqp_x[N:2 * N], slsqp_x[2 * N:3 * N], slsqp_x[:N]])

    # Fallback
    if best_arr is None:
        w = h = 1.0
        seed_x = _grid_seed(w, h)
        slsqp_x = _slsqp_optimize(seed_x, w, h, time_budget=20.0)
        if not _check_validity(slsqp_x[N:2 * N], w, h, slsqp_x[:N]):
            # trivial valid
            xs = np.linspace(1e-3, w - 1e-3, N)
            ys = np.full(N, h / 2.0)
            r = np.full(N, 1e-3)
            best_arr = np.column_stack([xs, ys, r])
        else:
            best_arr = np.column_stack([slsqp_x[N:2 * N], slsqp_x[2 * N:3 * N], slsqp_x[:N]])

    out = np.array(best_arr[:N])
    return out
