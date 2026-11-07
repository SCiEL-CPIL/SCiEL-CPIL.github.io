"""Combined optimizer: Lattice-aware basin hopping + (1+1)-ES refinement for Rastrigin f3."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Lattice-aware basin hopping followed by (1+1)-ES fine-tuning for Rastrigin f3."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 285  # Safety margin before 330s kill
        rng = np.random.default_rng(42)

        calls = [0]

        def call_func(x):
            calls[0] += 1
            return func(x)

        def time_ok():
            return time.time() < deadline and calls[0] < budget

        # ---------- Phase 1: Lattice-Aware Initial Exploration ----------
        # Rastrigin minima are at integer points (shifted). 
        # We evaluate integer points in [-4, 4]^5 to find a good basin center.
        best_x = None
        best_f = np.inf
        
        # Generate a set of integer candidates
        # We don't need all 9^5 = 59049 points, just a good subset.
        # Use a Sobol-like or random sample of integer points.
        n_int_samples = 50
        int_candidates = []
        
        # Always include origin and its immediate neighbors
        int_candidates.append(np.zeros(dim, dtype=int))
        for i in range(dim):
            for delta in [-1, 1]:
                pt = np.zeros(dim, dtype=int)
                pt[i] += delta
                int_candidates.append(pt)
        
        # Random integer points
        while len(int_candidates) < n_int_samples:
            pt = rng.integers(-4, 5, size=dim)
            int_candidates.append(pt)
        
        for pt in int_candidates:
            if not time_ok():
                break
            x = pt.astype(float)
            fx = call_func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = call_func(best_x)

        # ---------- Phase 2: Simulated Annealing on Integer Lattice ----------
        # Move between integer points to find the global basin.
        # This exploits the periodicity: jumping by 1 unit moves to the next basin.
        
        x_current_int = np.round(best_x).astype(int)
        x_current_int = np.clip(x_current_int, -4, 4)
        f_current = call_func(x_current_int.astype(float))
        
        if f_current < best_f:
            best_f = f_current
            best_x = x_current_int.astype(float).copy()
            
        x_best_int = x_current_int.copy()

        T_start = 10.0
        T_end = 0.1
        n_sa_steps = 100
        cool_rate = (T_end / T_start) ** (1.0 / n_sa_steps)
        T = T_start

        for step in range(n_sa_steps):
            if not time_ok():
                break

            # Propose a move: change 1 coordinate by +/-1
            ax = rng.integers(0, dim)
            delta = 1 if rng.random() < 0.5 else -1
            x_prop_int = x_current_int.copy()
            x_prop_int[ax] += delta
            x_prop_int = np.clip(x_prop_int, -4, 4)
            
            # If we didn't move (hit boundary and delta was outward), try opposite or skip
            if np.array_equal(x_prop_int, x_current_int):
                continue

            x_prop = x_prop_int.astype(float)
            f_prop = call_func(x_prop)

            # Accept if better, or with probability exp(-delta_f / T)
            if f_prop < f_current or rng.random() < np.exp(-(f_prop - f_current) / max(T, 1e-8)):
                x_current_int = x_prop_int
                f_current = f_prop
                
                if f_prop < best_f:
                    best_f = f_prop
                    best_x = x_prop.copy()
                    x_best_int = x_prop_int.copy()

            T *= cool_rate

        # ---------- Phase 3: Local Refinement (1+1)-ES ----------
        # Now that we have a good basin center (x_best), refine it continuously.
        # (1+1)-ES is fast and effective for smooth local minima.
        
        if time_ok():
            sigma = 0.5  # Start with a step size larger than lattice spacing to allow slight moves, but small enough to stay in basin
            # Actually, since we are at the integer point, the minimum is very close. 
            # A smaller sigma is better for refinement.
            sigma = 0.2 
            
            # Run for a significant portion of remaining budget
            max_steps = budget - calls[0] - 50 # Keep some for final polish
            for _ in range(max_steps):
                if not time_ok():
                    break
                step = sigma * rng.standard_normal(dim)
                y = np.clip(best_x + step, -5, 5)
                fy = call_func(y)
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                    sigma *= 1.05
                else:
                    sigma *= 0.95
                if sigma < 0.001:
                    break

        # ---------- Phase 4: Final Fine-Tuning ----------
        # Use remaining budget for very fine adjustments
        if time_ok():
            sigma = 0.05
            for _ in range(budget - calls[0]):
                if not time_ok():
                    break
                step = sigma * rng.standard_normal(dim)
                y = np.clip(best_x + step, -5, 5)
                fy = call_func(y)
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                    sigma *= 1.02
                else:
                    sigma *= 0.98
                if sigma < 0.0005:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
