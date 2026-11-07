"""Rewritten optimizer for BBOB-style Weierstrass f16."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+lambda)-ES with adaptive step size and Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Multi-start: evaluate diverse points to find a good basin
        candidates = [np.zeros(dim)]
        for _ in range(9):
            candidates.append(rng.uniform(-4, 4, dim))

        best_x = None
        best_f = np.inf
        for c in candidates:
            fc = func(c)
            if fc < best_f:
                best_f = fc
                best_x = c.copy()

        calls_made = len(candidates)

        # (1+lambda)-ES parameters
        lam = 10  # population size
        sigma = 1.0
        success_threshold = 0.25  # target success rate
        window_size = 20
        successes = []

        # Phase 1: (1+lambda)-ES main search
        while calls_made < budget:
            if time.time() > deadline:
                break

            # Check if we should switch to polish phase
            remaining = budget - calls_made
            if remaining < 60:
                break

            # Generate lambda trial points
            offspring = []
            for _ in range(lam):
                z = rng.standard_normal(dim)
                trial_x = best_x + sigma * z
                trial_x = np.clip(trial_x, -5, 5)
                offspring.append(trial_x)

            # Evaluate offspring
            best_trial_f = np.inf
            best_trial_x = None
            for tx in offspring:
                if calls_made >= budget:
                    break
                if time.time() > deadline:
                    break
                tf = func(tx)
                calls_made += 1
                if tf < best_trial_f:
                    best_trial_f = tf
                    best_trial_x = tx

            # Update best if improvement found
            if best_trial_x is not None and best_trial_f < best_f:
                best_f = best_trial_f
                best_x = best_trial_x.copy()
                successes.append(1)
            else:
                successes.append(0)

            # Adapt sigma using success rate on sliding window
            if len(successes) >= window_size:
                recent = successes[-window_size:]
                success_rate = sum(recent) / window_size
                if success_rate > success_threshold + 0.1:
                    sigma *= 1.10  # increase step significantly
                elif success_rate > success_threshold:
                    sigma *= 1.03
                elif success_rate < success_threshold - 0.1:
                    sigma *= 0.90  # decrease step significantly
                elif success_rate < success_threshold:
                    sigma *= 0.97
                sigma = max(1e-8, min(sigma, 10.0))

        # Phase 2: Nelder-Mead polish
        if calls_made < budget and time.time() < deadline:
            best_x, best_f = self._nelder_mead(func, best_x, best_f, calls_made, budget, deadline, dim, rng)

        return best_x, best_f

    def _nelder_mead(self, func, x, f, calls_made, budget, deadline, dim, rng):
        """Nelder-Mead simplex method for local polishing."""
        # Initialize simplex
        simplex = [x.copy() + np.zeros(dim)]
        fvals = [f]

        # Add perturbed points
        for i in range(dim):
            perturbation = x.copy()
            if f > 0:
                perturbation[i] += 0.5
            else:
                perturbation[i] -= 0.5
            perturbation = np.clip(perturbation, -5, 5)
            simplex.append(perturbation)
            fv = func(perturbation)
            calls_made += 1
            fvals.append(fv)

        # Sort by function value
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]

        alpha = 1.0  # reflection
        gamma = 2.0  # expansion
        rho = 0.5   # contraction
        sigma_nm = 0.5  # shrink

        max_iter = min(budget - calls_made, 200)

        for _ in range(max_iter):
            if calls_made >= budget or time.time() > deadline:
                break

            # Check convergence
            if max(fvals) - min(fvals) < 1e-10:
                break

            # Centroid of all but worst point
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflect
            x_reflect = centroid + alpha * (centroid - simplex[-1])
            x_reflect = np.clip(x_reflect, -5, 5)
            f_reflect = func(x_reflect)
            calls_made += 1

            if f_reflect < fvals[0]:
                # Expand
                x_expand = centroid + gamma * (x_reflect - centroid)
                x_expand = np.clip(x_expand, -5, 5)
                f_expand = func(x_expand)
                calls_made += 1
                if f_expand < f_reflect:
                    simplex[-1] = x_expand
                    fvals[-1] = f_expand
                else:
                    simplex[-1] = x_reflect
                    fvals[-1] = f_reflect
            elif f_reflect < fvals[-2]:
                simplex[-1] = x_reflect
                fvals[-1] = f_reflect
            else:
                # Contract
                if f_reflect < fvals[-1]:
                    x_contract = centroid + rho * (simplex[-1] - centroid)
                else:
                    x_contract = centroid - rho * (centroid - simplex[-1])
                x_contract = np.clip(x_contract, -5, 5)
                f_contract = func(x_contract)
                calls_made += 1
                if f_contract < fvals[-1]:
                    simplex[-1] = x_contract
                    fvals[-1] = f_contract
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        fvals[i] = func(simplex[i])
                        calls_made += 1
                        if calls_made >= budget or time.time() > deadline:
                            break
                    if calls_made >= budget or time.time() > deadline:
                        break

            # Sort simplex
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]

        # Return best point
        best_idx = np.argmin(fvals)
        return simplex[best_idx], fvals[best_idx]
# EVOLVE-BLOCK-END
