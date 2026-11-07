"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """1+1-ES with per-dimension adaptive sigma, then FD gradient polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 310.0  # safety margin before 330s kill

        rng = np.random.default_rng(42)

        # State
        best_x = np.zeros(dim)
        best_fx = np.inf
        evals = 0

        def check_deadline():
            return time.time() >= deadline

        def eval_point(x):
            nonlocal best_x, best_fx, evals
            fx = func(x)
            evals += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            return fx

        # Phase 1: Global search with 1+1-ES, per-dimension sigma
        # Weierstrass is multi-modal with many local minima.
        # Use a larger initial sigma to explore, then adapt.

        x = np.zeros(dim)
        fx = eval_point(x)

        # Per-dimension step sizes
        sigmas = np.full(dim, 3.0)
        # Learning rates for sigma adaptation
        c_sigma = 1.0 / dim  # small learning rate
        # Success tracking
        successes = 0
        total = 0
        window = 10

        # Phase 1 budget: use ~60% of evaluations for global search
        phase1_budget = int(budget * 0.6)

        for i in range(phase1_budget):
            if evals >= budget - 1 or check_deadline():
                break

            # Generate candidate
            z = rng.standard_normal(dim)
            candidate = x + z * sigmas
            candidate = np.clip(candidate, -5, 5)

            f_candidate = eval_point(candidate)

            # Update sigma based on success
            total += 1
            if f_candidate < fx:
                x = candidate
                fx = f_candidate
                successes += 1

            if total >= window:
                rate = successes / total
                # Adapt sigmas: increase if success rate > 0.5, decrease if < 0.5
                # Use a simple multiplicative update
                for d in range(dim):
                    if rate > 0.55:
                        sigmas[d] *= 1.05
                    elif rate < 0.45:
                        sigmas[d] *= 0.95
                    sigmas[d] = np.clip(sigmas[d], 1e-4, 10.0)
                successes = 0
                total = 0

            # Periodically try a random restart to escape local minima
            if i % 50 == 49 and i < phase1_budget - 50:
                # Try a random point; if it's better, jump there
                rx = rng.uniform(-4, 4, size=dim)
                fr = eval_point(rx)
                if fr < fx:
                    x = rx
                    fx = fr
                    # Reset sigmas to moderate
                    sigmas = np.full(dim, 2.0)
                    successes = 0
                    total = 0

        # Phase 2: Local polish with finite-difference gradient descent
        # Weierstrass is smooth, so FD gradients are effective.
        x = best_x.copy()
        fx = best_fx
        step_size = 0.1
        fd_eps = 1e-4

        phase2_budget = budget - evals - 1
        if phase2_budget > 10 and not check_deadline():
            for it in range(min(200, phase2_budget)):
                if evals >= budget - 1 or check_deadline():
                    break

                # Compute FD gradient
                grad = np.zeros(dim)
                for d in range(dim):
                    if evals >= budget - 1 or check_deadline():
                        break
                    xp = x.copy()
                    xp[d] += fd_eps
                    xp = np.clip(xp, -5, 5)
                    xp[d] = min(xp[d], 5.0)
                    fp = eval_point(xp)

                    xm = x.copy()
                    xm[d] -= fd_eps
                    xm = np.clip(xm, -5, 5)
                    xm[d] = max(xm[d], -5.0)
                    fm = eval_point(xm)

                    grad[d] = (fp - fm) / (2 * fd_eps)

                # Check if we ran out of budget during gradient computation
                if evals >= budget - 1:
                    break

                # Gradient norm
                gnorm = np.linalg.norm(grad)
                if gnorm < 1e-10:
                    break

                # Take a step in the negative gradient direction
                direction = -grad / gnorm
                x_new = x + step_size * direction
                x_new = np.clip(x_new, -5, 5)

                f_new = eval_point(x_new)

                if f_new < fx:
                    x = x_new
                    fx = f_new
                    # Increase step size slightly
                    step_size = min(step_size * 1.1, 1.0)
                else:
                    # Decrease step size
                    step_size *= 0.5
                    if step_size < 1e-8:
                        break

        # Phase 3: Final coordinate-wise fine-tuning
        if evals < budget - 1 and not check_deadline():
            x = best_x.copy()
            fx = best_fx
            coord_step = 0.01
            for it in range(min(100, budget - evals - 1)):
                if evals >= budget - 1 or check_deadline():
                    break
                improved = False
                for d in range(dim):
                    if evals >= budget - 1 or check_deadline():
                        break
                    # Try positive
                    xp = x.copy()
                    xp[d] = min(5.0, x[d] + coord_step)
                    if xp[d] != x[d]:
                        fp = eval_point(xp)
                        if fp < fx:
                            x = xp
                            fx = fp
                            improved = True
                            continue
                    # Try negative
                    xm = x.copy()
                    xm[d] = max(-5.0, x[d] - coord_step)
                    if xm[d] != x[d]:
                        fm = eval_point(xm)
                        if fm < fx:
                            x = xm
                            fx = fm
                            improved = True

                if not improved:
                    coord_step *= 0.5
                    if coord_step < 1e-8:
                        break

        return best_x, best_fx
# EVOLVE-BLOCK-END
