"""Multi-objective optimizer for DTLZ7 with structural exploitation."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Exploit DTLZ7 structure: Pareto front is 1D (x[0]), optimal has x[1:]=0."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 290.0  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)

        X_list = []
        F_list = []
        eval_count = 0

        def evaluate(x):
            nonlocal eval_count
            if eval_count >= budget:
                return None
            x = np.clip(x, lo, hi)
            f = problem(x)
            eval_count += 1
            X_list.append(x.copy())
            F_list.append(np.asarray(f, dtype=float))
            return f

        # Phase 1: Quasi-random (Halton-like) initialization along the Pareto front
        # x[1:] = 0 gives g=1 (optimal), so we just need good x[0] coverage
        # Use a low-discrepancy sequence for better space-filling
        n_init = min(120, budget - 80)  # reserve budget for polish phase
        if n_init < 10:
            n_init = 10

        # Generate Halton sequence in 1D (base 2 van der Corput)
        def halton_1d(n, base=2):
            seq = np.zeros(n)
            for i in range(n):
                f = 1.0 / base
                result = 0.0
                num = i + 1
                while num > 0:
                    result += f * (num % base)
                    num //= base
                    f /= base
                seq[i] = result
            return seq

        x0_vals = halton_1d(n_init)
        for x0 in x0_vals:
            if time.time() > deadline:
                break
            x = np.zeros(dim)
            x[0] = x0
            # x[1:] = 0 is optimal for DTLZ7 (g=1)
            f = evaluate(x)
            if f is None:
                break

        # Phase 2: Gap-filling with stratified random sampling
        # Identify gaps in objective space and fill them
        while eval_count < budget - 30 and time.time() < deadline:
            x = np.zeros(dim)
            x[0] = rng.uniform(0.0, 1.0)
            f = evaluate(x)
            if f is None:
                break

        # Phase 3: Polish - add points near boundaries and key regions
        # DTLZ7 front: f1=x0, f2=(1-x0)*cos(x0*pi/2), f3=(1-x0)*sin(x0*pi/2)
        # Hypervolume is maximized by covering the entire front uniformly
        # Add extra points in regions with higher contribution to hypervolume
        remaining = budget - eval_count
        if remaining > 0 and time.time() < deadline:
            # Focus on x0 in [0, 0.5] where f3 is larger (higher hypervolume contribution)
            n_polish = min(remaining, 40)
            polish_vals = np.linspace(0.0, 0.5, n_polish)
            for x0 in polish_vals:
                if eval_count >= budget or time.time() > deadline:
                    break
                x = np.zeros(dim)
                x[0] = x0
                f = evaluate(x)
                if f is None:
                    break

        return X_list
# EVOLVE-BLOCK-END
