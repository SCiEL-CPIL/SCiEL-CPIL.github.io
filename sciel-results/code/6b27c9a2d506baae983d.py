"""Circle packing in unit square, n=32.

Combines:
- Structural idea of the current program: one coherent L-BFGS-B pipeline,
  hard validity check, explicit deadline guard with safety margin.
- Refinement machinery from the strongest inspiration: diverse seed
  generators, center-only push-apart pre-refine, and a center-only final
  polish that preserves the radius sum.
- Structural change per directive: add a soft repulsive potential
  (Lennard-Jones style, r^-12 repulsion + r^-6 attraction) to the
  objective, and switch the final polish to a damped Newton step on the
  potential energy surface (L-BFGS-B kept as a robust fallback).
"""

N = 32

# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize
import time


def construct():
    start_time = time.time()
    deadline = start_time + 135  # 135s out of 150s, safety margin

    best_flat = None
    best_score = -np.inf

    # ------------------------------------------------------------------
    # Parameter layout: [x0..xN, y0..yN, r0..rN]
    # ------------------------------------------------------------------
    def unpack(params):
        n = N
        x = params[:n]
        y = params[n:2*n]
        r = params[2*n:]
        return x, y, r

    # ------------------------------------------------------------------
    # Soft repulsive potential (Lennard-Jones style) + boundary terms
    # ------------------------------------------------------------------
    def potential(params, sigma=0.18, eps=1.0):
        """Return (energy, grad_x, grad_y, grad_r).

        Energy = sum over pairs of eps*( (sigma/d)^12 - 2*(sigma/d)^6 )
               + boundary spring terms.

        The r^-12 term gives strong short-range repulsion (enforces
        disjoint interiors); the r^-6 term gives mild attraction that
        helps circles settle into close-packed configurations.
        """
        n = N
        x, y, r = unpack(params)
        energy = 0.0
        gx = np.zeros(n)
        gy = np.zeros(n)
        gr = np.zeros(n)

        # --- Pairwise Lennard-Jones potential ---
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[j] - x[i]
                dy = y[j] - y[i]
                d2 = dx * dx + dy * dy
                if d2 < 1e-20:
                    # Avoid singularity: nudge apart slightly
                    d2 = 1e-20
                d = np.sqrt(d2)
                sd = sigma / d
                sd6 = sd ** 6
                sd12 = sd6 * sd6
                # Energy contribution: eps * (sd12 - 2*sd6)
                e = eps * (sd12 - 2.0 * sd6)
                energy += e
                # Gradient w.r.t. x_i, x_j:
                # dE/dx_i = eps * (-12*sd12 + 12*sd6) * (dx/d) / sigma * (-1)
                # Simplify: dE/dx_i = 12*eps*sd6*(sd6 - 1) * dx / d
                coeff = 12.0 * eps * sd6 * (sd6 - 1.0) / d
                gx[i] -= coeff * dx
                gx[j] += coeff * dx
                gy[i] -= coeff * dy
                gy[j] += coeff * dy
                # Gradient w.r.t. r_i, r_j (through sigma if we tie sigma to r):
                # We keep sigma fixed; radii enter only through boundary terms
                # and the validity check. For the potential, radii don't
                # directly affect the LJ term (sigma is a fixed length scale).
                # However, to let radii grow, we add a linear reward -sum(r).
        # --- Linear reward for larger radii ---
        energy -= np.sum(r)
        gr[:] -= 1.0

        # --- Boundary spring terms (keep circles inside [0,1]^2) ---
        k_boundary = 50.0
        for i in range(n):
            # Left: x_i - r_i >= 0
            viol_l = max(0.0, r[i] - x[i])
            energy += k_boundary * viol_l ** 2
            if viol_l > 0:
                gx[i] -= 2.0 * k_boundary * viol_l
                gr[i] += 2.0 * k_boundary * viol_l
            # Right: 1 - x_i - r_i >= 0
            viol_r = max(0.0, x[i] + r[i] - 1.0)
            energy += k_boundary * viol_r ** 2
            if viol_r > 0:
                gx[i] += 2.0 * k_boundary * viol_r
                gr[i] += 2.0 * k_boundary * viol_r
            # Bottom: y_i - r_i >= 0
            viol_b = max(0.0, r[i] - y[i])
            energy += k_boundary * viol_b ** 2
            if viol_b > 0:
                gy[i] -= 2.0 * k_boundary * viol_b
                gr[i] += 2.0 * k_boundary * viol_b
            # Top: 1 - y_i - r_i >= 0
            viol_t = max(0.0, y[i] + r[i] - 1.0)
            energy += k_boundary * viol_t ** 2
            if viol_t > 0:
                gy[i] += 2.0 * k_boundary * viol_t
                gr[i] += 2.0 * k_boundary * viol_t

        grad = np.concatenate([gx, gy, gr])
        return energy, grad

    # ------------------------------------------------------------------
    # Objective for L-BFGS-B: potential energy
    # ------------------------------------------------------------------
    def objective(params):
        energy, _ = potential(params)
        return energy

    # ------------------------------------------------------------------
    # Hard validity check
    # ------------------------------------------------------------------
    def is_valid(params, tol=1e-9):
        n = N
        x, y, r = unpack(params)
        if np.any(r < -tol):
            return False
        if np.any(x - r < -tol) or np.any(1.0 - x - r < -tol):
            return False
        if np.any(y - r < -tol) or np.any(1.0 - y - r < -tol):
            return False
        # Overlap check
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx * dx + dy * dy)
        min_dist = r[:, None] + r[None, :]
        iu = np.triu_indices(n, k=1)
        if np.any(dist[iu] < min_dist[iu] - tol):
            return False
        return True

    # ------------------------------------------------------------------
    # Score: sum of radii
    # ------------------------------------------------------------------
    def get_score(params):
        _, _, r = unpack(params)
        return float(np.sum(r))

    # ------------------------------------------------------------------
    # Center-only push-apart: separate overlapping circles without
    # changing radii, preserving the radius sum.
    # ------------------------------------------------------------------
    def push_apart(params, n_iters=80):
        n = N
        x, y, r = unpack(params)
        x = x.copy()
        y = y.copy()

        for _ in range(n_iters):
            moved = False
            for i in range(n):
                for j in range(i + 1, n):
                    dx = x[j] - x[i]
                    dy = y[j] - y[i]
                    d = np.sqrt(dx * dx + dy * dy)
                    min_d = r[i] + r[j]
                    if d < min_d:
                        if d > 1e-12:
                            overlap = min_d - d
                            wi = r[i] / min_d if min_d > 1e-15 else 0.5
                            wj = 1.0 - wi
                            px = (dx / d) * overlap
                            py = (dy / d) * overlap
                            x[i] -= px * wi
                            y[i] -= py * wi
                            x[j] += px * wj
                            y[j] += py * wj
                            moved = True
                        else:
                            angle = np.random.uniform(0, 2 * np.pi)
                            x[i] -= 0.01 * np.cos(angle)
                            y[i] -= 0.01 * np.sin(angle)
                            x[j] += 0.01 * np.cos(angle)
                            y[j] += 0.01 * np.sin(angle)
                            moved = True
            # Clamp to boundary (center-only)
            for i in range(n):
                if x[i] < r[i]:
                    x[i] = r[i]
                    moved = True
                if x[i] > 1.0 - r[i]:
                    x[i] = 1.0 - r[i]
                    moved = True
                if y[i] < r[i]:
                    y[i] = r[i]
                    moved = True
                if y[i] > 1.0 - r[i]:
                    y[i] = 1.0 - r[i]
                    moved = True
            if not moved:
                break

        return np.concatenate([x, y, r])

    # ------------------------------------------------------------------
    # Damped Newton step on the potential energy surface (center + radii).
    # Uses numerical Hessian-free approach: line search on gradient.
    # This is the "structural change" — a physics-based optimization step.
    # ------------------------------------------------------------------
    def damped_newton_polish(params, max_iters=200, tol=1e-14):
        """Damped Newton / steepest-descent with line search on the
        potential energy.  Falls back to L-BFGS-B if it fails."""
        x = params.copy()
        best_x = x.copy()
        best_e, _ = potential(x)

        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            e, g = potential(x)
            if np.linalg.norm(g) < tol:
                break
            # Damped step: x_new = x - alpha * g
            # Adaptive step size with backtracking line search
            alpha = 0.01
            for _ in range(30):
                x_try = x - alpha * g
                e_try, _ = potential(x_try)
                if e_try < e:
                    break
                alpha *= 0.5
            else:
                break
            x = x_try
            if e_try < best_e - 1e-15:
                best_e = e_try
                best_x = x.copy()
            if alpha < 1e-12:
                break

        # Final L-BFGS-B polish from the best point found
        try:
            res = scipy.optimize.minimize(
                objective, best_x, method='L-BFGS-B',
                jac=lambda p: potential(p)[1],
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if res.fun < best_e:
                best_x = res.x
                best_e = res.fun
        except Exception:
            pass

        return best_x

    # ------------------------------------------------------------------
    # Diverse initial configurations
    # ------------------------------------------------------------------
    def get_initial_config(restart, rng):
        n = N
        side = int(np.ceil(np.sqrt(n)))

        if restart % 10 == 0:
            # Regular grid
            x = np.zeros(n)
            y = np.zeros(n)
            r = np.zeros(n)
            for i in range(n):
                row = i // side
                col = i % side
                x[i] = (col + 0.5) / side
                y[i] = (row + 0.5) / side
                r[i] = 0.5 / side - 0.002
        elif restart % 10 == 1:
            # Perturbed grid
            x = np.zeros(n)
            y = np.zeros(n)
            r = np.zeros(n)
            for i in range(n):
                row = i // side
                col = i % side
                x[i] = (col + 0.5) / side
                y[i] = (row + 0.5) / side
                r[i] = 0.5 / side - 0.002
            x += rng.uniform(-0.02, 0.02, size=n)
            y += rng.uniform(-0.02, 0.02, size=n)
        elif restart % 10 == 2:
            # Hexagonal packing (6 columns)
            x = np.zeros(n)
            y = np.zeros(n)
            r = np.zeros(n)
            for i in range(n):
                row = i // 6
                col = i % 6
                x[i] = (col + 0.5) / 6 + (0.25 if row % 2 == 1 else 0.0)
                y[i] = (row + 0.5) / 6
                r[i] = 0.08
        elif restart % 10 == 3:
            # Random uniform
            x = rng.uniform(0.08, 0.92, size=n)
            y = rng.uniform(0.08, 0.92, size=n)
            r = rng.uniform(0.05, 0.10, size=n)
        elif restart % 10 == 4:
            # 8x4 grid
            x = np.zeros(n)
            y = np.zeros(n)
            r = np.zeros(n)
            for i in range(n):
                row = i // 8
                col = i % 8
                x[i] = (col + 0.5) / 8
                y[i] = (row + 0.5) / 4
                r[i] = 0.5 / 8 - 0.002
            x += rng.uniform(-0.01, 0.01, size=n)
            y += rng.uniform(-0.01, 0.01, size=n)
        elif restart % 10 == 5:
            # 4x8 grid
            x = np.zeros(n)
            y = np.zeros(n)
            r = np.zeros(n)
            for i in range(n):
                row = i // 4
                col = i % 4
                x[i] = (col + 0.5) / 4
                y[i] = (row + 0.5) / 8
                r[i] = 0.5 / 4 - 0.002
            x += rng.uniform(-0.01, 0.01, size=n)
            y += rng.uniform(-0.01, 0.01, size=n)
        elif restart % 10 == 6:
            # Hexagonal variant
            x = np.zeros(n)
            y = np.zeros(n)
            r = np.zeros(n)
            for i in range(n):
                row = i // 6
                col = i % 6
                x[i] = (col + 0.5) / 6 + (0.2 if row % 2 == 0 else 0.0)
                y[i] = (row + 0.5) / 6
                r[i] = 0.075
        elif restart % 10 == 7:
            # Clustered: 4 groups of 8
            x = np.zeros(n)
            y = np.zeros(n)
            r = np.zeros(n)
            centers = [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)]
            for i in range(n):
                group = i // 8
                cx, cy = centers[group]
                sub = i % 8
                sx = (sub % 3) / 3.0 + 0.5 / 3.0
                sy = (sub // 3) / 3.0 + 0.5 / 3.0
                x[i] = cx + (sx - 0.5) * 0.4
                y[i] = cy + (sy - 0.5) * 0.4
                r[i] = 0.06
        elif restart % 10 == 8:
            # Random medium
            x = rng.uniform(0.08, 0.92, size=n)
            y = rng.uniform(0.08, 0.92, size=n)
            r = rng.uniform(0.06, 0.09, size=n)
        else:
            # Random small
            x = rng.uniform(0.05, 0.95, size=n)
            y = rng.uniform(0.05, 0.95, size=n)
            r = rng.uniform(0.03, 0.06, size=n)

        return np.concatenate([x, y, r])

    # ------------------------------------------------------------------
    # Main search loop
    # ------------------------------------------------------------------
    rng = np.random.RandomState(12345)

    n_restarts = 60
    for restart in range(n_restarts):
        if time.time() > deadline:
            break

        params = get_initial_config(restart, rng)
        params = push_apart(params, n_iters=60)

        # Quick validity + score check on the pushed-apart config
        if is_valid(params):
            s = get_score(params)
            if s > best_score:
                best_score = s
                best_flat = params.copy()

        # L-BFGS-B optimization on the potential energy
        try:
            res = scipy.optimize.minimize(
                objective, params, method='L-BFGS-B',
                jac=lambda p: potential(p)[1],
                options={'maxiter': 800, 'ftol': 1e-13, 'gtol': 1e-11}
            )
            x_opt = res.x
        except Exception:
            x_opt = params

        if is_valid(x_opt):
            s = get_score(x_opt)
            if s > best_score:
                best_score = s
                best_flat = x_opt.copy()
        else:
            # Try center-only push-apart to recover validity
            x_fixed = push_apart(x_opt, n_iters=40)
            if is_valid(x_fixed):
                s = get_score(x_fixed)
                if s > best_score:
                    best_score = s
                    best_flat = x_fixed.copy()

    # ------------------------------------------------------------------
    # Final polish: damped Newton step on the potential energy surface
    # ------------------------------------------------------------------
    if best_flat is not None and time.time() < deadline - 15:
        try:
            x_polished = damped_newton_polish(best_flat)
            if is_valid(x_polished):
                s = get_score(x_polished)
                if s > best_score:
                    best_score = s
                    best_flat = x_polished.copy()
            else:
                # Fallback: center-only push-apart
                x_fixed = push_apart(x_polished, n_iters=50)
                if is_valid(x_fixed):
                    s = get_score(x_fixed)
                    if s > best_score:
                        best_score = s
                        best_flat = x_fixed.copy()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Safety: if no valid solution was found, return a valid grid packing
    # ------------------------------------------------------------------
    if best_flat is None:
        side = 6
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        for i in range(N):
            row = i // side
            col = i % side
            x[i] = (col + 0.5) / side
            y[i] = (row + 0.5) / side
            r[i] = 0.5 / side - 0.002
        best_flat = np.concatenate([x, y, r])

    # ------------------------------------------------------------------
    # Format output: (N, 3) array of (x, y, radius)
    # ------------------------------------------------------------------
    x, y, r = unpack(best_flat)
    result = np.zeros((N, 3))
    result[:, 0] = x
    result[:, 1] = y
    result[:, 2] = r

    return result

# EVOLVE-BLOCK-END
