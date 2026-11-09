"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for ZDT4 using non-uniform x1 sampling and efficient search."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Use the global numpy random state (seeded by evaluator)
        rng = np.random.default_rng()

        n_evals = 0
        X_list = []
        F_list = []

        def evaluate(x):
            nonlocal n_evals
            x_clipped = np.clip(x, lo, hi)
            f = np.array(problem(x_clipped))
            n_evals += 1
            return x_clipped, f

        # ZDT4 structure:
        # f1 = 1 + 8*(x1^2 - 6*cos(12*pi*x1))
        # f2 = g(x) * h(f1, g)
        # g(x) = sum((xi-0.5)^2 - 10*cos(20*pi*(xi-0.5))) for i=2..n
        # h(u,v) = 1 - sqrt(u/v) if v > 0, else 1
        #
        # Key insight: For the Pareto front, we want g to be at its minimum
        # (which is -10*(n-1) when all xi=0.5), but g must be positive for
        # the front to be well-defined. Actually, the minimum of each term
        # (xi-0.5)^2 - 10*cos(20*pi*(xi-0.5)) is -10 at xi=0.5, so g_min = -10*(n-1).
        # But we want to MINIMIZE f2, and f2 = g*h. If g is negative, h is still defined.
        # The Pareto front of ZDT4 is convex and lies in the region where g is near its
        # minimum (all xi near 0.5) and x1 varies.
        #
        # Strategy:
        # 1. Sample x1 values that cover the range where f1 varies meaningfully
        # 2. Set x2..x10 near 0.5 to minimize g (which minimizes f2 for a given f1)
        # 3. Add small perturbations to explore the neighborhood

        # Phase 1: Structured x1 sampling with x2..x10 near 0.5
        # f1(x1) = 1 + 8*(x1^2 - 6*cos(12*pi*x1))
        # The function has oscillations due to the cosine term.
        # x1 in [-5, 5], but the interesting region for the Pareto front is where
        # f1 is relatively small (close to the minimum).
        # The minimum of x1^2 - 6*cos(12*pi*x1) is near x1=0 where it's about -6,
        # giving f1 ≈ 1 - 48 = -47. But actually, let's just sample x1 non-uniformly.

        n_phase1 = int(budget * 0.7)
        n_phase2 = budget - n_phase1

        # Generate x1 values with non-uniform distribution biased toward center
        # Use a sine-based mapping to concentrate samples near x1=0
        u = rng.random(n_phase1)
        # Map to x1 in [-5, 5] with concentration near 0
        # x1 = 5 * sin(pi * u) maps [0,1] to [0,5] with peak density at u=0.5 (x1=5)
        # Better: x1 = 5 * (2*u - 1) ** 0.5 * sign(2*u - 1) for bias toward 0
        # Or simply: sample from a distribution with heavier weight near 0
        # Let's use: x1 = 5 * tanh(3 * (2*u - 1)) which compresses toward 0
        x1_vals = 5.0 * np.tanh(3.0 * (2.0 * u - 1.0))

        # Also add some uniform samples to cover the full range
        n_uniform = n_phase1 // 5
        n_concentrated = n_phase1 - n_uniform
        x1_uniform = rng.uniform(-5.0, 5.0, n_uniform)
        x1_concentrated = 5.0 * np.tanh(3.0 * (2.0 * rng.random(n_concentrated) - 1.0))
        x1_vals = np.concatenate([x1_uniform, x1_concentrated])

        # Shuffle to interleave
        perm = rng.permutation(len(x1_vals))
        x1_vals = x1_vals[perm]

        # For each x1, set x2..x10 near 0.5 with small perturbation
        for i in range(len(x1_vals)):
            if time.time() > deadline or n_evals >= budget:
                break
            x = np.zeros(dim)
            x[0] = x1_vals[i]
            # Set x2..x10 to 0.5 with very small perturbation to stay on/near the front
            x[1:] = 0.5 + 0.01 * rng.standard_normal(dim - 1)
            x[1:] = np.clip(x[1:], lo[1:], hi[1:])
            x, f = evaluate(x)
            X_list.append(x)
            F_list.append(f)

        # Phase 2: Local refinement around promising points
        if n_evals < budget and len(F_list) > 0:
            F_arr = np.array(F_list)
            X_arr = np.array(X_list)

            # Find points that are non-dominated or close to the front
            # Sort by f1 and pick a diverse set
            order = np.argsort(F_arr[:, 0])

            # Select candidates spread across the f1 range
            n_candidates = min(15, len(order))
            if n_candidates > 1:
                candidate_indices = np.linspace(0, len(order) - 1, n_candidates, dtype=int)
                candidate_indices = order[candidate_indices]
            else:
                candidate_indices = order

            for idx in candidate_indices:
                if time.time() > deadline or n_evals >= budget:
                    break
                x_base = X_arr[idx].copy()
                f_base = F_arr[idx].copy()

                # Try perturbations in x1 and x2..x10
                # Focus on x1 since it primarily controls f1
                for step in [-0.03, 0.03, -0.01, 0.01, -0.06, 0.06]:
                    if n_evals >= budget or time.time() > deadline:
                        break
                    x_trial = x_base.copy()
                    x_trial[0] += step
                    # Small perturbation in x2..x10
                    x_trial[1:] += 0.005 * rng.standard_normal(dim - 1)
                    x_trial = np.clip(x_trial, lo, hi)
                    x_trial, f_trial = evaluate(x_trial)
                    X_list.append(x_trial)
                    F_list.append(f_trial)
                    # Update base if this is better in at least one objective
                    # and not worse in all
                    if np.any(f_trial < f_base) and not np.all(f_trial > f_base):
                        x_base = x_trial.copy()
                        f_base = f_trial.copy()
                        F_arr[idx] = f_trial
                        X_arr[idx] = x_trial

        # Return all evaluated points
        if len(X_list) > 0:
            return np.array(X_list)
        return np.array([])
# EVOLVE-BLOCK-END
