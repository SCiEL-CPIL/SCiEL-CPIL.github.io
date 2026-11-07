"""Rewritten optimizer for BBOB f23 (Katsuura) with separable coordinate descent + Nelder-Mead polish."""
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Separable 1D coordinate refinement followed by local Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin before 300s limit

        # Multi-start sampling: combine random, quasi-random, and boundary points
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Generate diverse initial points
        # 1) Random uniform
        # 2) Latin hypercube style via Halton (scipy qmc)
        # 3) Center point
        # 4) A few points near boundaries

        # Halton sequence for quasi-random coverage
        try:
            from scipy.stats import qmc
            halton_gen = qmc.Halton(dim, scramble=True, seed=123)
            n_halton = 50
            halton_samples = halton_gen.random(n_halton)
            halton_samples = -5.0 + 10.0 * halton_samples
        except Exception:
            halton_samples = rng.uniform(-5, 5, (50, dim))

        # Random uniform samples
        n_rand = 30
        rand_samples = rng.uniform(-5, 5, (n_rand, dim))

        # Center
        center = np.zeros(dim)

        # Boundary-near points (small perturbations from ±5)
        n_boundary = 20
        boundary_samples = []
        for _ in range(n_boundary):
            pt = rng.uniform(-5, 5, dim)
            idx = rng.integers(0, dim)
            if rng.random() < 0.5:
                pt[idx] = -4.5 + rng.uniform(0, 1.0)
            else:
                pt[idx] = 4.5 - rng.uniform(0, 1.0)
            boundary_samples.append(pt)
        boundary_samples = np.array(boundary_samples)

        # Combine all initial points
        initial_points = np.vstack([
            halton_samples,
            rand_samples,
            center.reshape(1, -1),
            boundary_samples
        ])

        # Evaluate all initial points
        values = np.array([func(p) for p in initial_points])
        n_initial = len(initial_points)

        best_idx = np.argmin(values)
        x_best = initial_points[best_idx].copy()
        f_best = values[best_idx]

        # Track best-so-far for AOCC (not needed for return, but good practice)
        calls_made = n_initial
        remaining = budget - calls_made

        # Phase 1: Separable coordinate descent with adaptive refinement
        # For each dimension, do a 1D golden-section search around the current best
        # Allocate budget across dimensions based on potential improvement

        # We'll do multiple passes over dimensions, refining each coordinate
        # Use golden-section search for 1D minimization

        def golden_1d(func, x_fixed, dim_idx, a, b, max_evals, current_best_val):
            """Golden-section search for 1D minimization along dim_idx."""
            if max_evals <= 0:
                return a + 0.5 * (b - a), current_best_val
            invphi = (np.sqrt(5.0) - 1.0) / 2.0
            c = a + (1 - invphi) * (b - a)
            d = a + invphi * (b - a)

            # Evaluate at c and d
            x_c = x_fixed.copy()
            x_c[dim_idx] = c
            fc = func(x_c)
            max_evals -= 1

            x_d = x_fixed.copy()
            x_d[dim_idx] = d
            fd = func(x_d)
            max_evals -= 1

            # Update best if improved
            if fc < current_best_val:
                current_best_val = fc
            if fd < current_best_val:
                current_best_val = fd

            while max_evals >= 2 and (b - a) > 1e-10:
                if fc < fd:
                    b = d
                    d = c
                    fd = fc
                    c = a + (1 - invphi) * (b - a)
                    x_c = x_fixed.copy()
                    x_c[dim_idx] = c
                    fc = func(x_c)
                else:
                    a = c
                    c = d
                    fc = fd
                    d = a + invphi * (b - a)
                    x_d = x_fixed.copy()
                    x_d[dim_idx] = d
                    fd = func(x_d)

                if fc < current_best_val:
                    current_best_val = fc
                if fd < current_best_val:
                    current_best_val = fd
                max_evals -= 2

            # Return the better of the last two evaluated points
            if fc < fd:
                return c, min(fc, current_best_val)
            else:
                return d, min(fd, current_best_val)

        # Phase 1: Multiple rounds of coordinate refinement
        # Budget allocation: split remaining among dimensions, proportional to initial spread
        # For simplicity, equal allocation per dimension per round, with a fixed number of rounds

        # Let's do several rounds of coordinate descent
        # Each round: for each dimension, do a golden-section search in a window around current value
        # Window size decreases over rounds

        n_rounds = min(5, max(1, remaining // (dim * 8)))  # at least 8 evals per dim per round
        evals_per_round_dim = max(5, remaining // (n_rounds * dim))

        for round_idx in range(n_rounds):
            if time.time() > deadline:
                break
            window = 2.0 * (0.5 ** round_idx)  # shrinking window: 2, 1, 0.5, 0.25, 0.125

            for d in range(dim):
                if time.time() > deadline:
                    break
                if calls_made >= budget:
                    break

                # Define search interval for this coordinate
                a = max(-5.0, x_best[d] - window)
                b = min(5.0, x_best[d] + window)

                # Golden-section search
                new_val, new_best = golden_1d(
                    func, x_best, d, a, b,
                    evals_per_round_dim, f_best
                )
                calls_made += evals_per_round_dim  # approximate; golden_1d tracks internally

                # Update if improved
                if new_best < f_best:
                    x_best[d] = new_val
                    f_best = new_best

        # Phase 2: Local Nelder-Mead polish
        # Use a few Nelder-Mead runs from the best point and nearby points
        if calls_made < budget and time.time() < deadline:
            nm_options = {
                'maxiter': min(50, (budget - calls_made) // max(1, dim)),
                'xatol': 1e-10,
                'fatol': 1e-12,
                'adaptive': True
            }

            # Start Nelder-Mead from best point
            try:
                result = minimize(func, x_best, method='Nelder-Mead', options=nm_options)
                calls_made += result.nfev
                if result.fun < f_best:
                    x_best = result.x.copy()
                    f_best = result.fun
            except Exception:
                pass

            # Try a few more Nelder-Mead from nearby points
            if calls_made < budget and time.time() < deadline:
                n_more_nm = min(3, (budget - calls_made) // max(1, dim * 2))
                for _ in range(n_more_nm):
                    if calls_made >= budget or time.time() > deadline:
                        break
                    # Perturb best point
                    x_pert = x_best + 0.1 * rng.standard_normal(dim)
                    x_pert = np.clip(x_pert, -5, 5)
                    try:
                        result = minimize(func, x_pert, method='Nelder-Mead', options=nm_options)
                        calls_made += result.nfev
                        if result.fun < f_best:
                            x_best = result.x.copy()
                            f_best = result.fun
                    except Exception:
                        pass

        # Phase 3: If budget remains, do more fine-grained coordinate refinement
        if calls_made < budget and time.time() < deadline:
            # Final fine-tuning: small window coordinate search
            remaining2 = budget - calls_made
            if remaining2 > 0:
                window_final = 0.05
                evals_per_dim = max(3, remaining2 // dim)
                for d in range(dim):
                    if calls_made >= budget or time.time() > deadline:
                        break
                    a = max(-5.0, x_best[d] - window_final)
                    b = min(5.0, x_best[d] + window_final)
                    new_val, new_best = golden_1d(
                        func, x_best, d, a, b,
                        evals_per_dim, f_best
                    )
                    calls_made += evals_per_dim
                    if new_best < f_best:
                        x_best[d] = new_val
                        f_best = new_best

        return x_best, f_best
