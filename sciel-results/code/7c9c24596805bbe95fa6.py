import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_f = np.inf
        best_x = np.zeros(dim)
        evals = 0

        def eval_func(x):
            nonlocal evals, best_f, best_x
            x = np.clip(x, -5, 5)
            f = func(x)
            evals += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Multi-start Powell's method with stratified Latin Hypercube initialization
        # Phase 1: Global exploration with stratified sampling
        # Phase 2: Local exploitation with Powell's method from best candidates

        # Determine number of initial samples for global phase
        # Reserve budget for local optimization
        n_global = 100  # Number of global samples
        n_local_starts = 5  # Number of local optimization starts

        # Stratified Latin Hypercube sampling
        def stratified_sample(n_samples, dim, rng):
            """Generate stratified Latin Hypercube samples in [-5, 5]^dim"""
            samples = np.zeros((n_samples, dim))
            for d in range(dim):
                # Create stratified intervals
                edges = np.linspace(-5, 5, n_samples + 1)
                for i in range(n_samples):
                    # Sample uniformly within each stratum
                    samples[i, d] = rng.uniform(edges[i], edges[i + 1])
            return samples

        # Global phase: evaluate stratified samples
        global_samples = stratified_sample(n_global, dim, rng)
        global_fvals = []
        for x in global_samples:
            if evals >= budget:
                break
            if time.time() > deadline:
                break
            f = eval_func(x)
            global_fvals.append(f)

        global_fvals = np.array(global_fvals)

        # Select top candidates for local optimization
        # Take the best min(n_local_starts, len(global_fvals)) points
        n_select = min(n_local_starts, len(global_fvals))
        if n_select == 0:
            # Fallback: just return best found
            return best_x, best_f

        # Sort by fitness and take top candidates
        sorted_indices = np.argsort(global_fvals)[:n_select]
        local_starts = global_samples[sorted_indices]

        # Phase 2: Local optimization with Powell's method
        # Powell's method is derivative-free and efficient for separable functions
        # Use maxiter to limit evaluations per local run

        # Estimate remaining budget
        remaining_budget = budget - evals
        # Allocate budget to local starts, with more budget for better starts
        # Use a simple allocation: equal split with some reserve
        evals_per_local = max(10, remaining_budget // (n_select + 1))

        for i, x0 in enumerate(local_starts):
            if evals >= budget:
                break
            if time.time() > deadline:
                break

            # Powell's method
            try:
                res = minimize(
                    eval_func,
                    x0,
                    method='Powell',
                    bounds=[(-5, 5)] * dim,
                    options={
                        'maxiter': 100,
                        'maxfev': evals_per_local,
                        'xatol': 1e-10,
                        'fatol': 1e-12,
                    }
                )
                # Update best if needed (eval_func already tracks best)
            except Exception:
                pass

            # Check if we should do another round of local refinement
            if evals < budget and time.time() < deadline:
                # Do a second pass with tighter tolerance from the result
                try:
                    res2 = minimize(
                        eval_func,
                        res.x,
                        method='Powell',
                        bounds=[(-5, 5)] * dim,
                        options={
                            'maxiter': 50,
                            'maxfev': min(evals_per_local, budget - evals),
                            'xatol': 1e-12,
                            'fatol': 1e-14,
                        }
                    )
                except Exception:
                    pass

        # Final check: if we still have budget, do some random restarts
        # to potentially find better basins
        while evals < budget and time.time() < deadline:
            x_new = rng.uniform(-5, 5, dim)
            f_new = eval_func(x_new)
            if f_new < best_f * 0.9:  # Significant improvement
                # Do local optimization from this point
                remaining = budget - evals
                if remaining > 20:
                    try:
                        minimize(
                            eval_func,
                            x_new,
                            method='Powell',
                            bounds=[(-5, 5)] * dim,
                            options={
                                'maxiter': 50,
                                'maxfev': min(remaining, 100),
                                'xatol': 1e-10,
                                'fatol': 1e-12,
                            }
                        )
                    except Exception:
                        pass

        return best_x, best_f
