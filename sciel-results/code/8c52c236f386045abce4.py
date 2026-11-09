# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Adaptive (1+1)-ES with Hybrid Initialization (Halton + Lattice) and Adaptive Restarts.
    
    This optimizer combines:
    1. Structured Initialization: Halton sequence for uniform coverage + Lattice points 
       for integer minima (Rastrigin) + Strategic points.
    2. Adaptive (1+1)-ES: Uses 1/5th success rule for sigma adaptation.
    3. Adaptive Restarts: Escapes local minima via global or local restarts based on 
       stagnation and progress.
    4. Final Polish: Tightens search precision in the last phase of the budget.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        # Safety margin: 295s limit to stay well under 300s/330s kill
        deadline = start_time + 295.0 

        rng = np.random.default_rng(123)

        best_x = np.zeros(dim)
        best_f = np.inf
        calls = 0

        # Helper to update best
        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        def time_exceeded():
            return time.time() > deadline

        # ---------------------------------------------------------
        # Phase 1: Hybrid Structured Initialization
        # ---------------------------------------------------------
        
        # 1. Strategic Points (High priority for early AOCC)
        strategic_points = [
            np.zeros(dim),
            np.ones(dim) * 1.0,
            np.ones(dim) * -1.0,
            np.ones(dim) * 2.0,
            np.ones(dim) * -2.0,
            np.ones(dim) * 3.0,
            np.ones(dim) * -3.0,
            np.ones(dim) * 0.5,
            np.ones(dim) * -0.5,
        ]
        
        for pt in strategic_points:
            if calls >= budget or time_exceeded():
                break
            pt = np.clip(pt, -5, 5)
            f = func(pt)
            calls += 1
            update_best(pt, f)

        # 2. Lattice Points (Integer grid) - Crucial for Rastrigin (f3, f15)
        # Sample a subset of the integer lattice [-4, 4]^5
        # Full grid is 9^5 = 59049, too many. We sample 40 random lattice points.
        lattice_vals = [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]
        n_lattice = 40
        for _ in range(n_lattice):
            if calls >= budget or time_exceeded():
                break
            pt = np.array([rng.choice(lattice_vals) for _ in range(dim)])
            pt = np.clip(pt, -5, 5)
            f = func(pt)
            calls += 1
            update_best(pt, f)

        # 3. Halton Sequence for uniform quasi-random coverage
        def halton_sequence(index, base):
            q = 1.0
            r = 0.0
            while index > 0:
                q /= base
                r += q * (index % base)
                index //= base
            return r

        primes = [2, 3, 5, 7, 11]
        n_halton = 30
        for i in range(1, n_halton + 1):
            if calls >= budget or time_exceeded():
                break
            pt = np.zeros(dim)
            for j in range(dim):
                pt[j] = halton_sequence(i, primes[j])
            pt = pt * 10.0 - 5.0
            pt = np.clip(pt, -5, 5)
            f = func(pt)
            calls += 1
            update_best(pt, f)

        # 4. Random Uniform Points to break symmetry
        n_random = 20
        for _ in range(n_random):
            if calls >= budget or time_exceeded():
                break
            pt = rng.uniform(-5, 5, dim)
            pt = np.clip(pt, -5, 5)
            f = func(pt)
            calls += 1
            update_best(pt, f)

        # ---------------------------------------------------------
        # Phase 2: Adaptive (1+1)-ES with Restarts
        # ---------------------------------------------------------
        
        x = best_x.copy()
        fx = best_f
        sigma = 1.0  # Initial step size
        
        # Track success rate for 1/5th rule
        success_window = []
        window_size = 15
        
        # Stagnation tracking
        stagnation = 0
        no_improve_count = 0
        
        # Adaptive parameters
        sigma_min = 0.001  # Lower floor for final polish
        sigma_max = 4.0
        
        # Budget allocation: use last 10% for polish
        polish_start = int(budget * 0.90)

        while calls < budget and not time_exceeded():
            # Determine if we're in polish phase
            in_polish = calls >= polish_start
            if in_polish:
                # Tighten sigma for final precision
                sigma = min(sigma, 0.05)

            # Generate offspring
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls += 1

            # Track success
            success = fy < fx
            success_window.append(success)
            if len(success_window) > window_size:
                success_window.pop(0)

            # 1/5th success rule for sigma adaptation
            if len(success_window) >= window_size:
                success_rate = sum(success_window) / len(success_window)
                if success_rate > 0.3:
                    sigma = min(sigma * 1.2, sigma_max)
                elif success_rate < 0.1:
                    sigma = max(sigma * 0.7, sigma_min)

            # Update current solution
            if success:
                x, fx = y, fy
                stagnation = 0
            else:
                stagnation += 1

            # Update global best
            if fy < best_f:
                best_f = fy
                best_x = y.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1

            # Adaptive restart conditions
            progress_ratio = calls / budget
            # Stagnation limit: more patient early, less later
            stagnation_limit = int(30 * (1.0 - 0.5 * progress_ratio))
            no_improve_limit = int(50 * (1.0 - 0.5 * progress_ratio))

            if in_polish:
                stagnation_limit = 10
                no_improve_limit = 15

            # Restart if stagnated (local steps failed)
            if stagnation >= stagnation_limit:
                # Probability of global exploration increases if we've been stuck a long time
                explore_prob = min(0.2 + 0.02 * no_improve_count, 0.7)

                if rng.random() < explore_prob:
                    # Global exploration: Random point or Halton-inspired
                    if rng.random() < 0.5:
                        x = rng.uniform(-5, 5, dim)
                    else:
                        idx = int(rng.integers(1, n_halton + 1))
                        x = np.zeros(dim)
                        for j in range(dim):
                            x[j] = halton_sequence(idx + calls, primes[j])
                        x = x * 10.0 - 5.0
                    x = np.clip(x, -5, 5)
                    sigma = 2.0  # Reset sigma for broad exploration
                else:
                    # Local exploration around best
                    local_sigma = max(sigma * 2.0, 0.5)
                    x = np.clip(best_x + local_sigma * rng.standard_normal(dim), -5, 5)
                    sigma = local_sigma

                fx = func(x)
                calls += 1
                stagnation = 0
                no_improve_count = 0
                success_window = []

                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

                if calls >= budget:
                    break

            # Aggressive restart if no global improvement for a long time
            if no_improve_count >= no_improve_limit:
                # Try a few random restarts and pick the best one to continue from
                best_restart_x = x
                best_restart_f = fx
                n_restarts = 3
                for _ in range(n_restarts):
                    if calls >= budget or time_exceeded():
                        break
                    cand = rng.uniform(-5, 5, dim)
                    cand_f = func(cand)
                    calls += 1
                    if cand_f < best_restart_f:
                        best_restart_f = cand_f
                        best_restart_x = cand
                    if cand_f < best_f:
                        best_f = cand_f
                        best_x = cand.copy()

                x = best_restart_x
                fx = best_restart_f
                sigma = 1.5
                stagnation = 0
                no_improve_count = 0
                success_window = []

                if calls >= budget:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
