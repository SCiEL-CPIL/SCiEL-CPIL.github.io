# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize


def squares_in_square10() -> tuple:
    """
    Erdos "squares in a square" problem. Place n squares with pairwise disjoint
    interiors inside the unit square [0,1]^2 so as to MAXIMIZE the sum of their
    side lengths. Here n = 10.

    Strategy:
    1. Use differential evolution with a penalty-based objective to find a
       good feasible configuration (global search over the non-convex landscape).
    2. Refine the best candidate with SLSQP using explicit non-overlap and
       boundary constraints for local optimality.
    3. Return the best feasible solution found.

    Returns:
        centers: np.ndarray of shape (10, 2) with the (x, y) center of each square.
        sides:   np.ndarray of shape (10,) with the side length of each square.
    """
    n = 10
    tol = 1e-7
    deadline = time.time() + 570  # safety margin before 630s kill

    def violation(x):
        """Return total constraint violation (0 if feasible)."""
        cx, cy, s = x[:n], x[n:2*n], x[2*n:3*n]
        v = 0.0
        # Boundary: each square inside [-tol, 1+tol]^2
        for i in range(n):
            v += max(0.0, (-tol) - (cx[i] - s[i]/2))
            v += max(0.0, (cx[i] + s[i]/2) - (1.0 + tol))
            v += max(0.0, (-tol) - (cy[i] - s[i]/2))
            v += max(0.0, (cy[i] + s[i]/2) - (1.0 + tol))
        # Non-overlap: for each pair, |dx| >= t or |dy| >= t where t=(si+sj)/2-tol
        for i in range(n):
            for j in range(i + 1, n):
                t = (s[i] + s[j]) / 2.0 - tol
                dx = abs(cx[i] - cx[j])
                dy = abs(cy[i] - cy[j])
                # Violation if both dx < t and dy < t
                if dx < t and dy < t:
                    v += (t - dx) + (t - dy)
        return v

    def objective_sum(x):
        """Sum of side lengths."""
        s = x[2*n:3*n]
        return np.sum(s)

    def penalty_objective(x):
        """Negative sum of sides plus large penalty for violations."""
        s = x[2*n:3*n]
        obj = -np.sum(s)
        v = violation(x)
        return obj + 1000.0 * v

    def is_feasible(x):
        return violation(x) < 1e-6

    # Grid baseline as initial guess
    k = 4  # ceil(sqrt(10)) = 4
    side = 1.0 / 4.0
    idx = np.arange(n)
    cols = idx % k
    rows = idx // k
    cx0 = (cols + 0.5) * side
    cy0 = (rows + 0.5) * side
    s0 = np.full(n, side)
    x0 = np.concatenate([cx0, cy0, s0])

    best_x = x0.copy()
    best_obj = objective_sum(x0)

    # --- Phase 1: Differential Evolution for global search ---
    bounds = [(0.0, 1.0)] * n + [(0.0, 1.0)] * n + [(1e-8, 1.0)] * n

    def de_callback(xk, convergence):
        nonlocal best_x, best_obj
        if time.time() > deadline - 120:  # leave time for refinement
            return True
        if is_feasible(xk):
            obj = objective_sum(xk)
            if obj > best_obj:
                best_obj = obj
                best_x = xk.copy()
        return False

    try:
        de_res = differential_evolution(
            penalty_objective, bounds,
            maxiter=200, tol=1e-8, popsize=20,
            mutation=(0.3, 0.9), recombination=0.9,
            seed=42, callback=de_callback,
            polish=False  # we do our own polishing
        )
        x_de = de_res.x
        if is_feasible(x_de):
            obj_de = objective_sum(x_de)
            if obj_de > best_obj:
                best_obj = obj_de
                best_x = x_de.copy()
    except Exception:
        pass

    # --- Phase 2: SLSQP refinement from multiple starting points ---
    rng = np.random.default_rng(42)
    n_starts = 30

    # Collect candidate starting points
    candidates = [best_x.copy()]
    # Add the grid baseline
    candidates.append(x0.copy())
    # Add some random feasible-ish starts
    for _ in range(n_starts):
        cx_init = rng.uniform(0.05, 0.95, n)
        cy_init = rng.uniform(0.05, 0.95, n)
        s_init = rng.uniform(0.05, 0.4, n)
        for i in range(n):
            s_init[i] = min(s_init[i], 0.5)
            cx_init[i] = np.clip(cx_init[i], s_init[i]/2 - tol, 1 - s_init[i]/2 + tol)
            cy_init[i] = np.clip(cy_init[i], s_init[i]/2 - tol, 1 - s_init[i]/2 + tol)
        candidates.append(np.concatenate([cx_init, cy_init, s_init]))

    # Also add the DE result if feasible
    if is_feasible(de_res.x):
        candidates.append(de_res.x.copy())

    for x_init in candidates:
        if time.time() > deadline - 60:
            break

        try:
            res = minimize(
                penalty_objective, x_init, method='SLSQP', bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-14}
            )
            x_res = res.x
            # Enforce minimum side
            x_res[2*n:3*n] = np.maximum(x_res[2*n:3*n], 1e-8)

            if is_feasible(x_res):
                obj_res = objective_sum(x_res)
                if obj_res > best_obj:
                    best_obj = obj_res
                    best_x = x_res.copy()
        except Exception:
            continue

    # --- Phase 3: Final local polish on the best solution ---
    if time.time() < deadline - 30:
        try:
            # Try a few different SLSQP settings on the best solution
            for ftol in [1e-15, 1e-14, 1e-13]:
                res = minimize(
                    penalty_objective, best_x, method='SLSQP', bounds=bounds,
                    options={'maxiter': 500, 'ftol': ftol}
                )
                x_res = res.x
                x_res[2*n:3*n] = np.maximum(x_res[2*n:3*n], 1e-8)
                if is_feasible(x_res):
                    obj_res = objective_sum(x_res)
                    if obj_res > best_obj:
                        best_obj = obj_res
                        best_x = x_res.copy()
        except Exception:
            pass

    # Extract result
    cx = best_x[:n]
    cy = best_x[n:2*n]
    s = best_x[2*n:3*n]

    # Final cleanup: ensure all sides > 0 and finite, and clip to valid range
    s = np.maximum(s, 1e-8)
    # Clip centers to ensure boundary feasibility
    for i in range(n):
        cx[i] = np.clip(cx[i], s[i]/2 - tol, 1.0 - s[i]/2 + tol)
        cy[i] = np.clip(cy[i], s[i]/2 - tol, 1.0 - s[i]/2 + tol)

    centers = np.stack([cx, cy], axis=1).astype(np.float64)
    sides = s.astype(np.float64)

    return centers, sides


# EVOLVE-BLOCK-END
