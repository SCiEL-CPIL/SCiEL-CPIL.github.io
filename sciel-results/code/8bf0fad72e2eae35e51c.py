"""Combined optimizer: (1+1)-ES core with CMA-ES-inspired restarts and adaptive scaling."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule, augmented with adaptive restarts."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        n = self.dim
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s kill

        # Best solution found so far
        best_x = None
        best_f = np.inf
        calls = 0

        def make_call(x):
            nonlocal calls, best_x, best_f
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Quick initial exploration to find a good starting point
        # Use a few random points to bracket the search
        init_points = []
        # Center
        init_points.append(np.zeros(n))
        # A few random points
        for _ in range(4):
            init_points.append(rng.uniform(-5, 5, n))
        # Axis-aligned points (helpful for rotated/shifted sphere)
        for i in range(n):
            pt = np.zeros(n)
            pt[i] = 3.0
            init_points.append(pt)
            pt[i] = -3.0
            init_points.append(pt)

        init_fxs = []
        for pt in init_points:
            if calls >= self.budget:
                break
            fx = make_call(pt)
            init_fxs.append(fx)

        # Pick the best initial point
        if init_fxs:
            best_init_idx = int(np.argmin(init_fxs))
            x0 = init_points[best_init_idx].copy()
            fx0 = init_fxs[best_init_idx]
        else:
            x0 = np.zeros(n)
            fx0 = np.inf

        # Phase 2: (1+1)-ES with adaptive sigma and restart logic
        sigma = 1.5  # Initial step size
        sigma_min = 1e-10
        sigma_max = 10.0
        x = x0.copy()
        fx = fx0

        # Track success rate for adaptive sigma (1/5 rule variant)
        success_count = 0
        total_steps = 0
        # For restart detection: track if we've made no progress for a while
        no_progress_calls = 0
        restart_threshold = 50  # Restart if no improvement for this many calls
        max_restarts = 5
        restarts = 0

        # CMA-ES-inspired: maintain a covariance-like structure via
        # tracking recent successful directions
        # Simple version: keep a memory of successful step directions
        success_dirs = []
        max_dir_memory = 20

        while calls < self.budget:
            if time.time() > deadline:
                break

            # Generate candidate
            # Blend: mostly isotropic, slightly biased toward recent success directions
            z = rng.standard_normal(n)

            # If we have success directions, add a small component along them
            if success_dirs and len(success_dirs) > 3:
                # Average of recent success directions (normalized)
                avg_dir = np.mean(success_dirs, axis=0)
                avg_dir_norm = np.linalg.norm(avg_dir)
                if avg_dir_norm > 1e-8:
                    avg_dir /= avg_dir_norm
                    # Mix: 80% isotropic, 20% directional
                    z = 0.8 * z + 0.2 * avg_dir

            y = x + sigma * z
            fy = make_call(y)

            total_steps += 1

            if fy < fx:
                # Success
                # Record direction
                step_dir = (y - x) / (sigma * np.linalg.norm(z) + 1e-15)
                success_dirs.append(step_dir)
                if len(success_dirs) > max_dir_memory:
                    success_dirs.pop(0)

                x, fx = y, fy
                success_count += 1
                no_progress_calls = 0

                # 1/5 rule: adjust sigma based on success rate
                # Use a window of recent steps
                window = min(total_steps, 20)
                if total_steps >= 5:
                    # Estimate success rate over recent window
                    # Simple: if last few were successes, increase; else decrease
                    recent_successes = sum(1 for d in success_dirs[-window:] if d is not None)
                    # Actually, let's just use the classic 1/5 rule on a rolling basis
                    # For simplicity, use a fixed-rate approach
                    if success_count > 0 and total_steps > 0:
                        rate = success_count / total_steps
                        if rate > 0.2:
                            sigma *= 1.1
                        elif rate < 0.15:
                            sigma *= 0.9
                        # Clamp
                        sigma = np.clip(sigma, sigma_min, sigma_max)
            else:
                # Failure
                sigma *= 0.9
                sigma = np.clip(sigma, sigma_min, sigma_max)
                no_progress_calls += 1

            # Check for restart
            if no_progress_calls >= restart_threshold and restarts < max_restarts:
                restarts += 1
                no_progress_calls = 0
                success_dirs = []
                # Restart from current best with larger sigma
                x = best_x.copy()
                fx = best_f
                sigma = max(sigma * 3.0, 2.0)
                sigma = min(sigma, sigma_max)

        return best_x, best_f
# EVOLVE-BLOCK-END
