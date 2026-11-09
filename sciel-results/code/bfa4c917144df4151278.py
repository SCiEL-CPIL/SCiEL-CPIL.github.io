"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Coordinate Descent with global step decay, periodic random perturbations,
    and a final Nelder-Mead polish phase for the Lunacek bi-Rastrigin f24."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 330s kill
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # --- Phase 1: Multi-start coordinate descent ---
        # We use the first 10 calls to evaluate several random starting points
        # and pick the best one to begin the main descent.
        n_starts = min(10, budget // 5)
        starts = [rng.uniform(-5, 5, dim) for _ in range(n_starts)]
        fvals = [func(s) for s in starts]
        best_idx = np.argmin(fvals)
        x = starts[best_idx].copy()
        fx = fvals[best_idx]
        calls_used = n_starts

        # Global step size for coordinate descent
        step = 2.0
        # Decay rate for step size
        step_decay = 0.95
        # Periodic perturbation interval
        perturb_every = 20
        # Track iterations since last improvement
        iters_since_improve = 0
        max_stagnation = 15

        # --- Phase 2: Coordinate descent main loop ---
        axis = 0
        while calls_used < budget:
            # Check time limit
            if time.time() > deadline - 30:
                break

            # Try moving along current axis in both directions
            improved = False
            for direction in [1.0, -1.0]:
                y = x.copy()
                y[axis] += direction * step
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_used += 1
                if calls_used >= budget:
                    break
                if fy < fx:
                    x, fx = y, fy
                    improved = True
                    iters_since_improve = 0
                    # Move the other axis next
                    break

            if calls_used >= budget:
                break

            if not improved:
                # No improvement along this axis; try next axis
                iters_since_improve += 1

            # Move to next axis cyclically
            axis = (axis + 1) % dim

            # Periodic random perturbation to escape local minima
            if iters_since_improve >= max_stagnation:
                # Large random jump
                x = x + rng.uniform(-3, 3, dim)
                x = np.clip(x, -5, 5)
                fx = func(x)
                calls_used += 1
                iters_since_improve = 0
                # Reset step size
                step = 2.0
            elif iters_since_improve % perturb_every == 0 and iters_since_improve > 0:
                # Smaller perturbation
                x = x + rng.uniform(-0.5, 0.5, dim)
                x = np.clip(x, -5, 5)
                fx = func(x)
                calls_used += 1
                iters_since_improve = max(0, iters_since_improve - 5)

            # Decay step size
            step *= step_decay
            if step < 1e-4:
                step = 1e-4

        # --- Phase 3: Nelder-Mead polish ---
        # Use remaining budget for a local polish with Nelder-Mead
        remaining = budget - calls_used
        if remaining >= dim + 2:
            # Build initial simplex around current best
            simplex = [x.copy()]
            for i in range(dim):
                v = x.copy()
                v[i] += 0.1
                simplex.append(np.clip(v, -5, 5))
            # Evaluate simplex
            f_simplex = [func(s) for s in simplex]
            calls_used += dim + 1

            # Nelder-Mead iterations
            alpha = 1.0   # reflection
            gamma = 2.0   # expansion
            rho = 0.5     # contraction
            sigma = 0.5   # shrink
            max_nm_iters = min(remaining - (dim + 1), 100)

            for _ in range(max_nm_iters):
                if time.time() > deadline - 10:
                    break
                if calls_used >= budget:
                    break

                # Sort simplex by function value
                order = np.argsort(f_simplex)
                simplex = [simplex[i] for i in order]
                f_simplex = [f_simplex[i] for i in order]

                # Centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = func(xr)
                calls_used += 1

                if fr < f_simplex[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fe = func(xe)
                    calls_used += 1
                    if fe < fr:
                        simplex[-1] = xe
                        f_simplex[-1] = fe
                    else:
                        simplex[-1] = xr
                        f_simplex[-1] = fr
                elif fr < f_simplex[-2]:
                    # Accept reflection
                    simplex[-1] = xr
                    f_simplex[-1] = fr
                else:
                    # Contraction
                    if fr < f_simplex[-1]:
                        # Outside contraction
                        xc = centroid + rho * (xr - centroid)
                    else:
                        # Inside contraction
                        xc = centroid - rho * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5, 5)
                    fc = func(xc)
                    calls_used += 1
                    if fc < min(fr, f_simplex[-1]):
                        simplex[-1] = xc
                        f_simplex[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            f_simplex[i] = func(simplex[i])
                            calls_used += 1
                            if calls_used >= budget:
                                break

                # Update best
                best_i = np.argmin(f_simplex)
                if f_simplex[best_i] < fx:
                    x, fx = simplex[best_i].copy(), f_simplex[best_i]

        return x, fx
# EVOLVE-BLOCK-END
