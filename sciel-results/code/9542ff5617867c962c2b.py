"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Rastrigin-aware coordinate local search with multi-basin exploration."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 330s kill

        call_count = 0
        max_calls = self.budget
        best_f = float('inf')
        best_x = np.zeros(self.dim)
        dim = self.dim

        def counted_func(x):
            nonlocal call_count, best_f, best_x
            if call_count >= max_calls:
                raise RuntimeError("Budget exceeded")
            if time.time() > deadline:
                raise RuntimeError("Time limit exceeded")
            call_count += 1
            x = np.clip(x, -5, 5)
            f = func(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        rng = np.random.default_rng(42)

        # Phase 1: Coarse random sampling to identify good basins
        n_random = min(40, max_calls // 10)
        for i in range(n_random):
            if call_count >= max_calls or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            f = counted_func(x)

        # Phase 2: Rastrigin-aware coordinate local search
        # The Büche-Rastrigin function has periodic ridges with period 2*pi
        # Local minima lie at integer multiples of 2*pi (shifted by the hidden offset)
        # We use coordinate-wise line search with ridge-jumping capability

        # Try multiple starting points from the random sampling
        # Sort by fitness and take top candidates
        candidates = []
        # Re-sample to get more candidates for the local search
        n_candidates = min(15, max_calls // 30)
        for i in range(n_candidates):
            if call_count >= max_calls or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            f = counted_func(x)
            candidates.append((f, x.copy()))

        candidates.sort(key=lambda t: t[0])

        # Apply local search to top candidates
        for f_init, x_init in candidates:
            if call_count >= max_calls or time.time() > deadline:
                break

            x = x_init.copy()
            f_curr = f_init
            improved = True
            max_iter_local = 50

            for iter_local in range(max_iter_local):
                if call_count >= max_calls or time.time() > deadline:
                    break

                best_dim_improvement = None
                best_dim_x = None
                best_dim_f = f_curr

                for d in range(dim):
                    # Try coordinate-wise moves with different step sizes
                    # Include ridge-jumping steps (multiples of ~2*pi)
                    step_sizes = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
                    for step in step_sizes:
                        for direction in [-1, 1]:
                            x_trial = x.copy()
                            x_trial[d] += direction * step
                            if x_trial[d] < -5 or x_trial[d] > 5:
                                continue
                            f_trial = counted_func(x_trial)
                            if f_trial < best_dim_f:
                                best_dim_f = f_trial
                                best_dim_x = x_trial.copy()
                                best_dim_improvement = d

                    if call_count >= max_calls or time.time() > deadline:
                        break

                if best_dim_improvement is None:
                    # No improvement found; try smaller steps
                    for d in range(dim):
                        for step in [0.1, 0.2, 0.05]:
                            for direction in [-1, 1]:
                                x_trial = x.copy()
                                x_trial[d] += direction * step
                                if x_trial[d] < -5 or x_trial[d] > 5:
                                    continue
                                f_trial = counted_func(x_trial)
                                if f_trial < best_dim_f:
                                    best_dim_f = f_trial
                                    best_dim_x = x_trial.copy()
                                    best_dim_improvement = d
                        if call_count >= max_calls or time.time() > deadline:
                            break
                    if best_dim_improvement is None:
                        break

                x = best_dim_x
                f_curr = best_dim_f

                if f_curr < best_f:
                    best_f = f_curr
                    best_x = x.copy()

        # Phase 3: Simulated Annealing refinement on the best point found
        if call_count < max_calls and time.time() < deadline:
            remaining = max_calls - call_count
            if remaining > 20:
                x_sa = best_x.copy()
                f_sa = best_f
                
                # Temperature schedule: start high, cool down
                T_start = 10.0
                T_end = 0.01
                n_sa_iters = min(remaining - 10, 100)
                
                for i in range(n_sa_iters):
                    if call_count >= max_calls or time.time() > deadline:
                        break
                    
                    # Linear cooling
                    T = T_start * (T_end / T_start) ** (i / max(n_sa_iters - 1, 1))
                    
                    # Generate a move: either a small random perturbation or a ridge jump
                    if rng.random() < 0.3:
                        # Ridge jump: move one coordinate by ~2*pi
                        d = rng.integers(0, dim)
                        jump = rng.choice([-2 * np.pi, 2 * np.pi])
                        x_trial = x_sa.copy()
                        x_trial[d] += jump
                        if x_trial[d] < -5 or x_trial[d] > 5:
                            x_trial[d] = np.clip(x_trial[d], -5, 5)
                    else:
                        # Small random perturbation
                        x_trial = x_sa + rng.normal(0, 0.5, dim)
                        x_trial = np.clip(x_trial, -5, 5)
                    
                    f_trial = counted_func(x_trial)
                    delta_f = f_trial - f_sa
                    
                    # Accept if better, or with probability exp(-delta_f / T)
                    if delta_f < 0 or rng.random() < np.exp(-delta_f / T):
                        x_sa = x_trial
                        f_sa = f_trial
                        if f_sa < best_f:
                            best_f = f_sa
                            best_x = x_sa.copy()

        # Phase 4: Final fine coordinate search if budget remains
        if call_count < max_calls and time.time() < deadline:
            remaining = max_calls - call_count
            if remaining > 10:
                x = best_x.copy()
                f_curr = best_f
                fine_steps = [0.01, 0.005, 0.002, 0.001, 0.0005]
                for step in fine_steps:
                    if call_count >= max_calls or time.time() > deadline:
                        break
                    for d in range(dim):
                        for direction in [-1, 1]:
                            if call_count >= max_calls or time.time() > deadline:
                                break
                            x_trial = x.copy()
                            x_trial[d] += direction * step
                            if x_trial[d] < -5 or x_trial[d] > 5:
                                continue
                            f_trial = counted_func(x_trial)
                            if f_trial < f_curr:
                                x = x_trial
                                f_curr = f_trial
                                if f_curr < best_f:
                                    best_f = f_curr
                                    best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
