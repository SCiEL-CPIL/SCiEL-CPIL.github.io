"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Hybrid multi-start Nelder-Mead with adaptive (1+1)-ES refinement.
    
    Strategy:
      - Phase 1: Random sampling to find diverse promising regions.
      - Phase 2: Multi-start Nelder-Mead from best samples with adaptive simplex scaling.
      - Phase 3: Adaptive (1+1)-ES with restarts for fine-tuning around best point.
      
    Combines the geometric adaptivity of Nelder-Mead for ill-conditioned funnels
    with the robustness of (1+1)-ES for final convergence.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 315.0  # safety margin from 330s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        calls_used = 0
        best_fx = float('inf')
        best_x = np.zeros(dim)

        def check_time():
            return time.time() < deadline - 3.0

        def check_budget():
            return calls_used < budget

        def safe_call(x):
            """Call function with budget and time tracking."""
            nonlocal calls_used, best_fx, best_x
            if not check_budget() or not check_time():
                return None
            fx = func(x)
            calls_used += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            return fx

        # ==================== PHASE 1: Random Sampling ====================
        # Sample diverse points to identify promising regions
        n_samples = min(30, budget // 10)
        samples = []

        for _ in range(n_samples):
            if not check_budget() or not check_time():
                break
            x = rng.uniform(-5, 5, dim)
            fx = safe_call(x)
            if fx is not None:
                samples.append((fx, x.copy()))

        if not samples:
            return best_x, best_fx

        samples.sort(key=lambda t: t[0])
        best_fx, best_x = samples[0]

        # Select top diverse seeds for Nelder-Mead
        seeds = []
        for fx, x in samples[:8]:
            # Ensure diversity: minimum distance between seeds
            if all(np.linalg.norm(x - sx) > 0.5 for _, sx in seeds):
                seeds.append((fx, x))
        if not seeds:
            seeds = [samples[0]]

        # ==================== PHASE 2: Multi-start Nelder-Mead ====================
        # Allocate budget across Nelder-Mead runs
        remaining_after_sampling = budget - calls_used
        n_nm_runs = min(len(seeds), 4)  # Limit to 4 NM runs
        per_nm_budget = max(20, remaining_after_sampling // (n_nm_runs + 1))

        def nm_step(simplex, simplex_f):
            """One Nelder-Mead iteration. Returns updated simplex and simplex_f."""
            nonlocal calls_used, best_fx, best_x

            # Sort simplex by function value
            order = np.argsort(simplex_f)
            simplex = simplex[order]
            simplex_f = simplex_f[order]

            # Update global best
            if simplex_f[0] < best_fx:
                best_fx = simplex_f[0]
                best_x = simplex[0].copy()

            # Centroid of all but worst point
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            worst = simplex[-1]
            reflected = centroid + 1.0 * (centroid - worst)
            reflected = np.clip(reflected, -5, 5)

            if not check_budget() or not check_time():
                return simplex, simplex_f

            f_reflected = func(reflected)
            calls_used += 1
            if f_reflected < best_fx:
                best_fx = f_reflected
                best_x = reflected.copy()

            if f_reflected < simplex_f[0]:
                # Expansion
                if not check_budget() or not check_time():
                    simplex[-1] = reflected
                    simplex_f[-1] = f_reflected
                    return simplex, simplex_f

                expanded = centroid + 2.0 * (reflected - centroid)
                expanded = np.clip(expanded, -5, 5)
                f_expanded = func(expanded)
                calls_used += 1
                if f_expanded < best_fx:
                    best_fx = f_expanded
                    best_x = expanded.copy()

                if f_expanded < f_reflected:
                    simplex[-1] = expanded
                    simplex_f[-1] = f_expanded
                else:
                    simplex[-1] = reflected
                    simplex_f[-1] = f_reflected

            elif f_reflected < simplex_f[-2]:
                # Simple reflection
                simplex[-1] = reflected
                simplex_f[-1] = f_reflected

            else:
                # Contraction
                if f_reflected < simplex_f[-1]:
                    # Outside contraction
                    contracted = centroid + 0.5 * (reflected - centroid)
                else:
                    # Inside contraction
                    contracted = centroid + 0.5 * (worst - centroid)

                contracted = np.clip(contracted, -5, 5)

                if not check_budget() or not check_time():
                    return simplex, simplex_f

                f_contracted = func(contracted)
                calls_used += 1
                if f_contracted < best_fx:
                    best_fx = f_contracted
                    best_x = contracted.copy()

                if f_contracted < simplex_f[-1]:
                    simplex[-1] = contracted
                    simplex_f[-1] = f_contracted
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        if not check_budget() or not check_time():
                            break
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        simplex_f[i] = func(simplex[i])
                        calls_used += 1
                        if simplex_f[i] < best_fx:
                            best_fx = simplex_f[i]
                            best_x = simplex[i].copy()

            return simplex, simplex_f

        def run_nelder_mead(x0, sigma0, max_iter):
            """Run Nelder-Mead from initial point x0 with initial step sigma0."""
            nonlocal calls_used, best_fx, best_x

            if not check_budget() or not check_time():
                return

            # Build initial simplex
            simplex = np.zeros((dim + 1, dim))
            simplex[0] = x0
            for i in range(dim):
                simplex[i + 1] = x0.copy()
                simplex[i + 1][i] += sigma0
                simplex[i + 1] = np.clip(simplex[i + 1], -5, 5)

            # Evaluate simplex
            simplex_f = np.zeros(dim + 1)
            for i in range(dim + 1):
                if not check_budget() or not check_time():
                    return
                simplex_f[i] = func(simplex[i])
                calls_used += 1
                if simplex_f[i] < best_fx:
                    best_fx = simplex_f[i]
                    best_x = simplex[i].copy()

            # Main loop
            for _ in range(max_iter):
                if not check_budget() or not check_time():
                    break

                simplex, simplex_f = nm_step(simplex, simplex_f)

                # Check convergence
                f_range = np.max(simplex_f) - np.min(simplex_f)
                if f_range < 1e-12:
                    break

        # Run Nelder-Mead from each seed
        # Allocate more iterations to better seeds
        for i, (fx0, x0) in enumerate(seeds[:n_nm_runs]):
            if not check_budget() or not check_time():
                break

            # Adaptive sigma: larger for worse seeds, smaller for better ones
            if i == 0:
                sigma0 = 1.5
                max_iter = int(per_nm_budget * 0.5)
            elif i == 1:
                sigma0 = 1.0
                max_iter = int(per_nm_budget * 0.3)
            else:
                sigma0 = 0.5
                max_iter = int(per_nm_budget * 0.15)

            max_iter = max(10, max_iter)
            run_nelder_mead(x0, sigma0, max_iter)

        # ==================== PHASE 3: Adaptive (1+1)-ES Refinement ====================
        # Fine-tune around the best point found so far
        if check_budget() and check_time():
            x = best_x.copy()
            fx = best_fx
            sigma = 0.3  # Start with small step for refinement
            stall_count = 0
            max_stall = 15

            remaining_es = budget - calls_used
            for _ in range(remaining_es):
                if not check_budget() or not check_time():
                    break

                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_used += 1

                if fy < fx:
                    x = y.copy()
                    fx = fy
                    sigma *= 1.1
                    stall_count = 0
                    if fx < best_fx:
                        best_fx = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.9
                    stall_count += 1

                # Restart if stalled
                if stall_count >= max_stall:
                    # Reset to best point with moderate step size
                    x = best_x.copy()
                    fx = best_fx
                    sigma = 0.5
                    stall_count = 0
                    max_stall = min(max_stall + 5, 50)

        return best_x, best_fx
# EVOLVE-BLOCK-END
