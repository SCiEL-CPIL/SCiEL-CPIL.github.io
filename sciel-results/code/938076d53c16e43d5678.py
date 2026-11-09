# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Adaptive (1+1)-ES with Halton-based initialization, CMA-like covariance adaptation, and smart restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit

        rng = np.random.default_rng(123)

        # Phase 1: Diverse initialization using Halton sequence for uniform coverage
        best_x = np.zeros(dim)
        best_f = func(best_x)
        calls = 1

        # Halton sequence for uniform coverage of [-5, 5]^dim
        def halton_sequence(index, base):
            q = 1.0
            r = 0.0
            while index > 0:
                q /= base
                r += q * (index % base)
                index //= base
            return r

        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

        # Generate Halton points for initial exploration
        n_halton = 30
        halton_points = np.zeros((n_halton, dim))
        for i in range(1, n_halton + 1):
            for j in range(dim):
                base = primes[j % len(primes)]
                halton_points[i - 1, j] = halton_sequence(i, base)
        # Scale to [-5, 5]
        halton_points = halton_points * 10.0 - 5.0

        # Strategic points for common BBOB patterns
        strategic_points = [
            np.ones(dim) * 2.0,
            np.ones(dim) * -2.0,
            np.ones(dim) * 1.0,
            np.ones(dim) * -1.0,
            np.ones(dim) * 3.0,
            np.ones(dim) * -3.0,
            np.ones(dim) * 0.5,
            np.ones(dim) * -0.5,
        ]

        # Evaluate Halton points first (uniform coverage)
        for pt in halton_points:
            if calls >= budget:
                break
            if time.time() > deadline:
                break
            pt = np.clip(pt, -5, 5)
            f = func(pt)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = pt.copy()

        # Evaluate strategic points
        for pt in strategic_points:
            if calls >= budget:
                break
            if time.time() > deadline:
                break
            pt = np.clip(pt, -5, 5)
            f = func(pt)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = pt.copy()

        # Phase 2: (1+1)-ES with covariance adaptation and smart restarts
        x = best_x.copy()
        fx = best_f
        sigma = 1.5  # Initial step size

        # Covariance matrix for directional adaptation
        C = np.eye(dim)
        C_sqrt = np.eye(dim)
        inv_C_sqrt = np.eye(dim)

        # Track success rate for 1/5th rule adaptation
        success_window = []
        window_size = 12

        # Stagnation tracking
        stagnation = 0
        no_improve_count = 0

        # Adaptive parameters
        sigma_min = 0.003
        sigma_max = 5.0

        # Budget allocation: use last 12% for polish
        polish_start = int(budget * 0.88)

        # Track direction of successful steps for covariance update
        recent_success_dirs = []
        max_dir_history = 8

        while calls < budget:
            if time.time() > deadline:
                break

            # Determine if we're in polish phase
            in_polish = calls >= polish_start
            if in_polish:
                sigma = min(sigma, 0.08)

            # Generate offspring using covariance-aware step
            z = rng.standard_normal(dim)
            step = C_sqrt @ z
            y = np.clip(x + sigma * step, -5, 5)
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
                if success_rate > 0.35:
                    sigma = min(sigma * 1.15, sigma_max)
                elif success_rate < 0.15:
                    sigma = max(sigma * 0.75, sigma_min)

            # Update current solution
            if success:
                x, fx = y, fy
                stagnation = 0
                # Track successful direction for covariance update
                direction = (y - (y - sigma * step)) / (sigma + 1e-12)
                direction = direction / (np.linalg.norm(direction) + 1e-12)
                recent_success_dirs.append(direction)
                if len(recent_success_dirs) > max_dir_history:
                    recent_success_dirs.pop(0)

                # Update covariance if we have enough successful directions
                if len(recent_success_dirs) >= 4:
                    dirs = np.array(recent_success_dirs)
                    # Rank-1 covariance update: C = (1-c)*C + c * (v v^T)
                    c = 0.3
                    v = np.mean(dirs, axis=0)
                    v_norm = np.linalg.norm(v)
                    if v_norm > 0.1:
                        v = v / v_norm
                        C = (1 - c) * C + c * np.outer(v, v)
                        # Ensure C is positive definite
                        C = 0.5 * (C + C.T)
                        # Re-orthonormalize via eigendecomposition
                        try:
                            eigvals, eigvecs = np.linalg.eigh(C)
                            eigvals = np.clip(eigvals, 1e-6, None)
                            C = (eigvecs * eigvals) @ eigvecs.T
                            C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
                            inv_C_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                        except np.linalg.LinAlgError:
                            pass
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
            stagnation_limit = int(35 * (1.0 - 0.4 * progress_ratio))
            no_improve_limit = int(55 * (1.0 - 0.4 * progress_ratio))

            if in_polish:
                stagnation_limit = 12
                no_improve_limit = 20

            # Restart if stagnated
            if stagnation >= stagnation_limit:
                explore_prob = min(0.35 + 0.012 * no_improve_count, 0.85)

                if rng.random() < explore_prob:
                    # Global exploration using Halton-inspired point
                    idx = int(rng.integers(1, n_halton + 1))
                    pt = np.zeros(dim)
                    for j in range(dim):
                        base = primes[j % len(primes)]
                        pt[j] = halton_sequence(idx + calls, base)
                    x = np.clip(pt * 10.0 - 5.0, -5, 5)
                    sigma = 2.5
                else:
                    # Local exploration around best with increased sigma
                    local_sigma = max(sigma * 2.5, 0.8)
                    x = np.clip(best_x + local_sigma * rng.standard_normal(dim), -5, 5)
                    sigma = local_sigma

                fx = func(x)
                calls += 1
                stagnation = 0
                no_improve_count = 0
                success_window = []
                recent_success_dirs = []
                # Reset covariance to identity on restart
                C = np.eye(dim)
                C_sqrt = np.eye(dim)
                inv_C_sqrt = np.eye(dim)

                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

                if calls >= budget:
                    break

            # Aggressive restart if no global improvement
            if no_improve_count >= no_improve_limit:
                # Try multiple random restarts and pick the best
                best_restart_x = x
                best_restart_f = fx
                n_restarts = 4
                for _ in range(n_restarts):
                    if calls >= budget:
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
                sigma = 2.0
                stagnation = 0
                no_improve_count = 0
                success_window = []
                recent_success_dirs = []
                C = np.eye(dim)
                C_sqrt = np.eye(dim)
                inv_C_sqrt = np.eye(dim)

                if calls >= budget:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
