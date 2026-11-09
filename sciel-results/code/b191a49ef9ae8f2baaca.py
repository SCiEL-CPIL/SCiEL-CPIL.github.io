"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """FPS-based global exploration -> coordinate sweep -> adaptive (1+1)-ES."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_used = 0

        def safe_call(x):
            nonlocal calls_used
            if calls_used >= budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            try:
                val = float(func(np.asarray(x, dtype=np.float64)))
                calls_used += 1
                return val
            except Exception:
                calls_used += 1
                return np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        def time_ok():
            return calls_used < budget and time.time() <= deadline

        def farthest_point_sampling(n_samples, dim, rng):
            """
            Generate n_samples points using Farthest Point Sampling.
            This ensures uniform coverage of the [-5, 5]^dim space.
            """
            points = []
            # Start with center or a random point
            first = rng.uniform(-5, 5, dim)
            points.append(first)
            
            for _ in range(n_samples - 1):
                # Find the point that maximizes minimum distance to all existing points
                min_dists = np.full(dim, np.inf)
                # Vectorized: for each candidate grid point, compute min dist to existing
                # Use a coarse grid for efficiency
                grid_size = 15
                candidates = []
                for i in range(grid_size):
                    for j in range(grid_size):
                        for k in range(grid_size):
                            for l in range(grid_size):
                                for m in range(grid_size):
                                    candidates.append([
                                        -5 + 10 * i / (grid_size - 1),
                                        -5 + 10 * j / (grid_size - 1),
                                        -5 + 10 * k / (grid_size - 1),
                                        -5 + 10 * l / (grid_size - 1),
                                        -5 + 10 * m / (grid_size - 1)
                                    ])
                candidates = np.array(candidates)
                
                # Compute min distance from each candidate to all existing points
                existing = np.array(points)
                # Distance matrix: (n_candidates, n_existing)
                diffs = candidates[:, np.newaxis, :] - existing[np.newaxis, :, :]
                dists = np.sqrt(np.sum(diffs ** 2, axis=2))
                min_dists = np.min(dists, axis=1)
                
                # Pick the candidate with max min-dist
                best_idx = np.argmax(min_dists)
                points.append(candidates[best_idx])
                
                # Add some jitter to avoid exact grid points
                points[-1] = np.clip(points[-1] + rng.normal(0, 0.3, dim), -5, 5)
            
            return np.array(points)

        def fast_coordinate_sweep(x, fx):
            """Fast coordinate descent using fixed step sizes."""
            nonlocal calls_used
            x = x.copy()
            fx = fx
            steps = [0.5, 1.0, 2.0, 4.0]
            
            for _ in range(2):
                if not time_ok():
                    break
                for d in range(dim):
                    if not time_ok():
                        break
                    base_val = x[d]
                    current_best_step = 0
                    current_best_f = fx
                    
                    for step in steps:
                        for sign in [1, -1]:
                            new_val = base_val + sign * step
                            if new_val < -5.0 or new_val > 5.0:
                                continue
                            y = x.copy()
                            y[d] = new_val
                            fy = safe_call(y)
                            if fy < current_best_f:
                                current_best_f = fy
                                current_best_step = sign * step
                            
                    if current_best_step != 0:
                        x[d] += current_best_step
                        fx = current_best_f
                        update_best(x, fx)
            return x, fx

        def es_refine(x, fx, max_iterations=None):
            """Adaptive (1+1)-ES refinement."""
            nonlocal calls_used
            x = x.copy()
            fx = fx
            sigma = 1.0
            
            while time_ok():
                if max_iterations is not None and max_iterations <= 0:
                    break
                if max_iterations is not None:
                    max_iterations -= 1
                
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = safe_call(y)
                
                if fy < fx:
                    x, fx = y, fy
                    update_best(x, fx)
                    sigma *= 1.1
                else:
                    sigma *= 0.85
                
                if sigma < 1e-6:
                    break

            return x, fx

        # ---- Phase 1: FPS-based global exploration ----
        # Use FPS for uniform coverage - critical for multimodal functions
        n_coarse = 30
        fps_points = farthest_point_sampling(n_coarse, dim, rng)
        
        coarse_x = []
        coarse_f = []
        for x in fps_points:
            if not time_ok():
                break
            fx = safe_call(x)
            if np.isfinite(fx):
                coarse_x.append(x)
                coarse_f.append(fx)

        if not coarse_x:
            x0 = np.zeros(dim)
            fx0 = safe_call(x0)
            update_best(x0, fx0 if np.isfinite(fx0) else 0.0)
            return best_x, best_f

        # Sort by fitness
        order = np.argsort(coarse_f)
        k = min(5, len(coarse_x))
        top_indices = order[:k]

        best_x = coarse_x[top_indices[0]].copy()
        best_f = coarse_f[top_indices[0]]

        # ---- Phase 2 & 3: Refinement with harmonic budget allocation ----
        remaining = budget - calls_used
        
        # Harmonic weights: better starts get more budget
        weights = np.array([1.0 / (i + 1) for i in range(k)])
        weights /= weights.sum()

        for i, idx in enumerate(top_indices):
            if not time_ok():
                break
            
            x = coarse_x[idx].copy()
            fx = coarse_f[idx]
            
            alloc_calls = int(remaining * weights[i])
            
            # Coordinate sweep first (cheap, deterministic)
            x, fx = fast_coordinate_sweep(x, fx)
            
            # ES refinement
            max_es_iters = max(10, alloc_calls // 2)
            x, fx = es_refine(x, fx, max_es_iters)

        # Final polish if budget remains
        if time_ok() and (budget - calls_used) > 20:
            x = best_x.copy()
            fx = best_f
            es_refine(x, fx, max_iterations=(budget - calls_used))

        return best_x, best_f
# EVOLVE-BLOCK-END
