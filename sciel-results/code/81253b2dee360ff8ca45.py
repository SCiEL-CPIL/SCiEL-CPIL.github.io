import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.deadline = time.time() + 295  # 5s safety margin before 300s limit

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Initialize x0 - start near center to avoid boundary issues
        x0 = np.zeros(dim)
        f0 = func(x0)
        best_x = x0.copy()
        best_f = f0
        calls = 1

        # Nelder-Mead Parameters
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5

        # Initialize Simplex with adaptive step size
        simplex = np.zeros((dim + 1, dim))
        simplex[0] = x0
        # Use smaller initial step for Bent Cigar's stiff dimension
        init_step = 0.5
        for i in range(dim):
            simplex[i + 1] = x0.copy()
            simplex[i + 1][i] += init_step
            simplex[i + 1] = np.clip(simplex[i + 1], -5, 5)

        fvals = np.zeros(dim + 1)
        fvals[0] = f0
        for i in range(1, dim + 1):
            if calls >= budget:
                break
            fvals[i] = func(simplex[i])
            calls += 1
            if fvals[i] < best_f:
                best_f = fvals[i]
                best_x = simplex[i].copy()

        # Main Loop
        max_iterations = 300
        no_improve_count = 0
        prev_best_f = best_f
        for _ in range(max_iterations):
            if calls >= budget or time.time() > self.deadline:
                break

            # Sort simplex by function value
            order = np.argsort(fvals)
            simplex = simplex[order]
            fvals = fvals[order]

            # Check convergence
            if np.max(fvals) - np.min(fvals) < 1e-10:
                break

            # Centroid of all points except the worst
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, -5, 5)
            if calls >= budget: break
            fr = func(xr)
            calls += 1
            improved = False
            if fr < best_f:
                best_f = fr
                best_x = xr.copy()
                improved = True

            if fr < fvals[0]:
                # Expansion
                xe = centroid + gamma * (centroid - simplex[-1])
                xe = np.clip(xe, -5, 5)
                if calls >= budget: break
                fe = func(xe)
                calls += 1
                if fe < best_f:
                    best_f = fe
                    best_x = xe.copy()
                    improved = True
                if fe < fr:
                    simplex[-1] = xe
                    fvals[-1] = fe
                else:
                    simplex[-1] = xr
                    fvals[-1] = fr
            elif fr < fvals[-2]:
                # Reflection accepted
                simplex[-1] = xr
                fvals[-1] = fr
            else:
                # Contraction
                # Outside contraction
                xc = centroid + rho * (xr - centroid)
                xc = np.clip(xc, -5, 5)
                if calls >= budget: break
                fc = func(xc)
                calls += 1
                if fc < best_f:
                    best_f = fc
                    best_x = xc.copy()
                    improved = True
                if fc < fr:
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    # Inside contraction
                    xc_in = centroid - rho * (centroid - simplex[-1])
                    xc_in = np.clip(xc_in, -5, 5)
                    if calls >= budget: break
                    fc_in = func(xc_in)
                    calls += 1
                    if fc_in < best_f:
                        best_f = fc_in
                        best_x = xc_in.copy()
                        improved = True
                    if fc_in < fvals[-1]:
                        simplex[-1] = xc_in
                        fvals[-1] = fc_in
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            if calls >= budget: break
                            fvals[i] = func(simplex[i])
                            calls += 1
                            if fvals[i] < best_f:
                                best_f = fvals[i]
                                best_x = simplex[i].copy()
                                improved = True
                        if calls >= budget: break

            # Track no-improvement streaks for adaptive behavior
            if improved:
                no_improve_count = 0
            else:
                no_improve_count += 1

            # If stuck, try a small random perturbation to escape local minima
            if no_improve_count >= 10 and calls + dim < budget:
                for i in range(dim):
                    if time.time() > self.deadline or calls >= budget:
                        break
                    perturbation = rng.normal(0, 0.1, dim)
                    x_new = best_x + perturbation
                    x_new = np.clip(x_new, -5, 5)
                    f_new = func(x_new)
                    calls += 1
                    if f_new < best_f:
                        best_f = f_new
                        best_x = x_new.copy()
                        no_improve_count = 0
                        # Reset simplex around new best
                        simplex[0] = best_x.copy()
                        fvals[0] = best_f
                        for j in range(1, dim + 1):
                            simplex[j] = best_x.copy()
                            simplex[j][j-1] += 0.1
                            simplex[j] = np.clip(simplex[j], -5, 5)
                            if calls >= budget:
                                break
                            fvals[j] = func(simplex[j])
                            calls += 1
                            if fvals[j] < best_f:
                                best_f = fvals[j]
                                best_x = simplex[j].copy()
                        break

        # Phase 2: Adaptive Coordinate Descent for fine-tuning
        # This exploits the anisotropy of Bent Cigar by shrinking steps in narrow dimensions faster
        if calls < budget and time.time() < self.deadline:
            # Estimate step sizes from the simplex
            # Use the range of the simplex in each dimension as a rough estimate
            step_sizes = np.zeros(dim)
            for i in range(dim):
                spread = np.max(simplex[:, i]) - np.min(simplex[:, i])
                step_sizes[i] = max(spread / 2, 0.01)

            x = best_x.copy()
            fx = best_f

            # Adaptive coordinate descent
            successes = np.zeros(dim)
            total = np.zeros(dim)
            success_threshold = 0.2
            min_step = 1e-6
            max_step = 5.0

            while calls < budget and time.time() < self.deadline:
                progress = False
                for i in range(dim):
                    if calls >= budget or time.time() > self.deadline:
                        break

                    # Try positive step
                    x_new = x.copy()
                    x_new[i] += step_sizes[i]
                    x_new = np.clip(x_new, -5, 5)
                    f_new = func(x_new)
                    calls += 1

                    if f_new < fx:
                        x = x_new
                        fx = f_new
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                        successes[i] += 1
                        progress = True
                    total[i] += 1

                    # Try negative step
                    x_new = x.copy()
                    x_new[i] -= step_sizes[i]
                    x_new = np.clip(x_new, -5, 5)
                    f_new = func(x_new)
                    calls += 1

                    if f_new < fx:
                        x = x_new
                        fx = f_new
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                        successes[i] += 1
                        progress = True
                    total[i] += 1

                    # Adapt step size based on success rate
                    if total[i] >= 5:
                        rate = successes[i] / total[i]
                        if rate > success_threshold:
                            step_sizes[i] *= 1.3
                        elif rate < success_threshold:
                            step_sizes[i] *= 0.7
                        successes[i] = 0
                        total[i] = 0

                    # Clamp step size
                    step_sizes[i] = np.clip(step_sizes[i], min_step, max_step)

                # If no progress in a full cycle, reduce all step sizes
                if not progress:
                    step_sizes *= 0.5
                    if np.all(step_sizes < min_step):
                        break

        return best_x, best_f
