"""STRUCTURAL CHANGE: Mixed random/integer initialization with Nelder-Mead refinement for BBOB f4."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Mixed initialization + Nelder-Mead refinement optimizer for BBOB f4."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 330s kill limit

        call_count = 0
        max_calls = self.budget
        best_x = np.zeros(self.dim)
        best_f = float('inf')

        def counted_func(x):
            nonlocal call_count, best_x, best_f
            if time.time() > deadline:
                raise RuntimeError("Time limit exceeded")
            if call_count >= max_calls:
                raise RuntimeError("Budget exceeded")

            call_count += 1
            x = np.asarray(x, dtype=np.float64).ravel()[:self.dim]
            x = np.clip(x, -5.0, 5.0)
            f = float(func(x))

            if f < best_f:
                best_f = f
                best_x = x.copy()

            return f

        rng = np.random.RandomState(42)

        # Phase 1: Initialization
        # Generate a mix of random points and points near integer coordinates
        # Rastrigin minima are near integers in the local coordinate system.
        # Since the function is shifted and rotated, the "integer" structure is rotated.
        # However, sampling near integers in the global space still provides good coverage
        # of the potential basin locations if the shift is not too large relative to the grid.
        
        num_init = 100
        candidates = []
        
        # 50% Random uniform
        for _ in range(num_init // 2):
            if call_count >= max_calls or time.time() > deadline:
                break
            x = rng.uniform(-4.0, 4.0, size=self.dim)
            f = counted_func(x)
            candidates.append((f, x))

        # 50% Near-integer (Integer + small noise)
        for _ in range(num_init // 2):
            if call_count >= max_calls or time.time() > deadline:
                break
            # Choose random integers in [-4, 4]
            x_int = rng.randint(-4, 5, size=self.dim).astype(float)
            # Add small noise to explore the basin around the integer
            x_noise = rng.uniform(-0.3, 0.3, size=self.dim)
            x = x_int + x_noise
            x = np.clip(x, -5.0, 5.0)
            f = counted_func(x)
            candidates.append((f, x))

        # Sort candidates by function value
        candidates.sort(key=lambda t: t[0])
        
        # Phase 2: Local Refinement with Nelder-Mead
        # Take the top 10 candidates and refine them
        num_refine = min(10, len(candidates))
        
        for i in range(num_refine):
            if call_count >= max_calls or time.time() > deadline:
                break
            
            f_init, x_init = candidates[i]
            
            # Use Nelder-Mead for refinement as it is derivative-free and robust
            # to the local structure of Rastrigin
            try:
                res = minimize(
                    lambda x: counted_func(x),
                    x_init,
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
                )
            except Exception:
                pass

        # Phase 3: Final Polish
        # If we have budget left, do a final high-precision polish on the best point found
        if call_count < max_calls and time.time() < deadline and best_f < float('inf'):
            try:
                res = minimize(
                    lambda x: counted_func(x),
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
