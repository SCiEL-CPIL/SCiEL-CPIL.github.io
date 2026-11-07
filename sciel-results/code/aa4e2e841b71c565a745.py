"""Iterated Local Search with Pattern Search for BBOB f19 (Griewank-Rosenbrock)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Optimizer using Iterated Local Search (ILS) with Pattern Search for local refinement.
    Multi-start with stratified sampling to cover the search space, then exploit smooth
    structure with pattern search (a derivative-free local method well-suited to
    Griewank-Rosenbrock's landscape).
    """
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        calls_made = 0
        best_x = None
        best_f = np.inf

        def safe_call(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return None
            try:
                x = np.asarray(x, dtype=np.float64)
                if x.shape != (dim,):
                    x = x.reshape(dim)
                x = np.clip(x, -5.0, 5.0)
                f = float(func(x))
                if not np.isfinite(f):
                    f = 1e10
                calls_made += 1
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
                return f
            except Exception:
                return None

        def pattern_search(x0, max_iters=50, initial_step=0.5):
            """
            Pattern Search: a simple derivative-free local optimization method.
            Explores along coordinate axes with adaptive step size.
            """
            nonlocal calls_made, best_x, best_f
            x = np.array(x0, dtype=np.float64)
            f = safe_call(x)
            if f is None:
                return x, f if f is not None else np.inf
            step = initial_step
            n_iter = 0
            no_improve = 0
            max_no_improve = 5

            while n_iter < max_iters and calls_made < budget and time.time() < deadline:
                improved = False
                for d in range(dim):
                    if calls_made >= budget or time.time() > deadline:
                        break
                    # Try positive direction
                    x_trial = x.copy()
                    x_trial[d] += step
                    f_trial = safe_call(x_trial)
                    if f_trial is not None and f_trial < f:
                        x = x_trial
                        f = f_trial
                        improved = True
                        step *= 1.5  # Expand step in this direction
                        break
                    # Try negative direction
                    x_trial = x.copy()
                    x_trial[d] -= step
                    f_trial = safe_call(x_trial)
                    if f_trial is not None and f_trial < f:
                        x = x_trial
                        f = f_trial
                        improved = True
                        step *= 1.5
                        break

                if not improved:
                    step *= 0.5
                    no_improve += 1
                    if no_improve >= max_no_improve or step < 1e-6:
                        break
                else:
                    no_improve = 0
                n_iter += 1

            return x, f

        # --- Phase 1: Stratified multi-start sampling ---
        # Use a stratified Latin Hypercube-like grid to ensure good coverage
        n_starts = min(20, budget // 10)
        rng = np.random.RandomState(42)

        start_points = []
        # Include origin
        start_points.append(np.zeros(dim))
        # Include corners of the box (subsampled)
        for _ in range(min(8, n_starts - 1)):
            pt = rng.uniform(-4.0, 4.0, size=dim)
            start_points.append(pt)
        # Add some structured points
        for d in range(dim):
            if len(start_points) >= n_starts:
                break
            pt = np.zeros(dim)
            pt[d] = 3.0
            start_points.append(pt)
            if len(start_points) >= n_starts:
                break
            pt = np.zeros(dim)
            pt[d] = -3.0
            start_points.append(pt)

        # Evaluate initial points
        for pt in start_points:
            if calls_made >= budget or time.time() > deadline:
                break
            f = safe_call(pt)

        if best_x is None:
            best_x = np.zeros(dim)
            f = safe_call(best_x)
            if f is None:
                best_f = 1e10

        # --- Phase 2: Iterated Local Search ---
        # From the best point, run pattern search, then perturb to escape local minima
        current_x = best_x.copy()
        current_f = best_f
        perturb_scale = 1.0
        rng2 = np.random.RandomState(123)

        max_ils_iters = 30
        for it in range(max_ils_iters):
            if calls_made >= budget or time.time() > deadline - 5.0:
                break

            # Local refinement with pattern search
            x_local, f_local = pattern_search(current_x, max_iters=30, initial_step=0.3)

            if f_local < best_f:
                best_f = f_local
                best_x = x_local.copy()

            # Perturb to escape local minimum
            perturb = rng2.normal(0, perturb_scale, size=dim)
            x_perturbed = x_local + perturb
            x_perturbed = np.clip(x_perturbed, -5.0, 5.0)

            f_perturbed = safe_call(x_perturbed)
            if f_perturbed is None:
                break

            # Accept if better or with some probability (simulated annealing-like)
            if f_perturbed < current_f:
                current_x = x_perturbed.copy()
                current_f = f_perturbed
                perturb_scale *= 0.9  # Reduce perturbation on improvement
            else:
                # Accept with probability based on how much worse it is
                delta = f_perturbed - current_f
                if delta < 1.0:
                    current_x = x_perturbed.copy()
                    current_f = f_perturbed
                perturb_scale *= 1.1  # Increase perturbation on stagnation
                perturb_scale = min(perturb_scale, 3.0)

            # Also try from the global best occasionally
            if it % 5 == 4 and best_f < current_f:
                current_x = best_x.copy()
                current_f = best_f

        # --- Phase 3: Final polish on the best point ---
        if calls_made < budget and time.time() < deadline:
            x_final, f_final = pattern_search(best_x, max_iters=40, initial_step=0.1)
            if f_final < best_f:
                best_f = f_final
                best_x = x_final.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            f = safe_call(best_x)
            if f is None:
                best_f = 1e10

        return best_x, best_f
# EVOLVE-BLOCK-END
