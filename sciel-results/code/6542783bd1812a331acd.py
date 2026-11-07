"""Root finder for 3 fixed 2D systems using Broyden's method with multi-start and safeguards."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.
    Uses Broyden's method with finite-difference Jacobian initialization,
    line search safeguards, multi-start from angular/radial perturbations,
    and a deadline guard.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit

    x0 = np.array(x0, dtype=float).reshape(2)

    def time_up():
        return time.time() >= deadline

    def is_good(x, Fx):
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(Fx)):
            return False
        return np.linalg.norm(Fx) <= 1e-8

    def broyden_start(x_init, Fx_init, max_iter=80):
        """
        Broyden's method (good-Broyden / rank-1 update) starting from x_init, Fx_init.
        Returns (x, Fx, n_calls) where n_calls is number of F evaluations made
        in this call (excluding the initial Fx_init which was already counted).
        """
        x = x_init.copy()
        Fx = Fx_init.copy()
        n_calls = 0
        n = 2

        # Compute initial Jacobian via finite differences (2 extra calls)
        J = np.zeros((n, n))
        h = 1e-6
        for j in range(n):
            e = np.zeros(n)
            e[j] = h
            Fx_plus = F(x + e)
            n_calls += 1
            J[:, j] = (Fx_plus - Fx) / h

        # Check if already good
        if is_good(x, Fx):
            return x, Fx, n_calls

        for it in range(max_iter):
            if time_up():
                return x, Fx, n_calls

            # Check convergence
            if is_good(x, Fx):
                return x, Fx, n_calls

            # Solve J * dx = -Fx
            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                # If singular, try a damped step
                dx = -Fx * 0.01

            # Line search: try full step first, then damp
            step_accepted = False
            for lam in [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125]:
                x_new = x + lam * dx
                Fx_new = F(x_new)
                n_calls += 1
                if not np.all(np.isfinite(Fx_new)):
                    continue
                # Armijo-like: accept if ||Fx_new|| < ||Fx||
                if np.linalg.norm(Fx_new) < np.linalg.norm(Fx):
                    # Update Broyden
                    s = x_new - x
                    y = Fx_new - Fx
                    x = x_new
                    Fx = Fx_new
                    # Rank-1 update: J = J + (y - J*s) * s^T / (s^T * s)
                    Js = J @ s
                    if np.dot(s, s) > 1e-14:
                        J += np.outer(y - Js, s) / np.dot(s, s)
                    step_accepted = True
                    break
                if time_up():
                    return x, Fx, n_calls

            if not step_accepted:
                # Try a smaller random perturbation
                break

        return x, Fx, n_calls

    # Multi-start strategy: try x0 first, then angular/radial perturbations
    best_x = x0.copy()
    best_Fx = F(x0)
    best_norm = np.linalg.norm(best_Fx) if np.all(np.isfinite(best_Fx)) else np.inf
    total_calls = 1  # count the initial F(x0)

    if is_good(best_x, best_Fx):
        return best_x

    # Generate candidate starting points
    candidates = []
    # Radii to try
    radii = [0.1, 0.5, 1.0, 2.0, 5.0]
    # Angles: 8 directions
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)

    for r in radii:
        for a in angles:
            offset = r * np.array([np.cos(a), np.sin(a)])
            candidates.append(x0 + offset)

    # Also add some larger random-ish perturbations (deterministic)
    for r in [10.0, 20.0, 50.0]:
        for a in angles[:4]:
            offset = r * np.array([np.cos(a), np.sin(a)])
            candidates.append(x0 + offset)

    # Sort candidates by distance to x0 (closer first) to find local roots quickly
    candidates.sort(key=lambda c: np.linalg.norm(c - x0))

    for x_cand in candidates:
        if time_up():
            break

        Fx_cand = F(x_cand)
        total_calls += 1

        if not np.all(np.isfinite(Fx_cand)):
            continue

        norm_cand = np.linalg.norm(Fx_cand)
        if norm_cand < best_norm:
            best_x = x_cand.copy()
            best_Fx = Fx_cand.copy()
            best_norm = norm_cand

        if is_good(best_x, best_Fx):
            return best_x

        # Run Broyden from this candidate
        x_res, Fx_res, n_calls = broyden_start(x_cand, Fx_cand)
        total_calls += n_calls

        if np.all(np.isfinite(Fx_res)):
            norm_res = np.linalg.norm(Fx_res)
            if norm_res < best_norm:
                best_x = x_res.copy()
                best_Fx = Fx_res.copy()
                best_norm = norm_res

            if is_good(best_x, best_Fx):
                return best_x

        if time_up():
            break

        # If we've used too many calls, stop trying new starts
        if total_calls > 450:
            break

    # Final: if best is not good enough, try one more Broyden from best_x
    if best_norm > 1e-8 and not time_up() and total_calls < 480:
        x_res, Fx_res, n_calls = broyden_start(best_x, best_Fx, max_iter=40)
        total_calls += n_calls
        if np.all(np.isfinite(Fx_res)) and np.linalg.norm(Fx_res) < best_norm:
            best_x = x_res.copy()
            best_Fx = Fx_res.copy()

    return best_x
# EVOLVE-BLOCK-END
