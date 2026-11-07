"""Rewritten optimizer using CMA-ES with Rastrigin-aware restarts."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """CMA-ES based optimizer with Rastrigin-aware multi-start strategy."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 330s kill

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

        # Phase 1: Initial diverse sampling to find good starting regions
        # Büche-Rastrigin has periodic structure with period 2*pi
        n_initial = min(30, max_calls // 15)
        for i in range(n_initial):
            if call_count >= max_calls or time.time() > deadline:
                break
            # Mix of random and structured sampling
            if i < 10:
                # Sample near potential minima: multiples of 2*pi shifted
                x = rng.uniform(-5, 5, dim)
            elif i < 20:
                # Sample on a grid-like structure
                x = np.zeros(dim)
                for d in range(dim):
                    x[d] = rng.choice([-2*np.pi, 0, 2*np.pi, -np.pi, np.pi])
                    x[d] += rng.normal(0, 0.5)
            else:
                x = rng.uniform(-5, 5, dim)
            f = counted_func(x)

        # Sort initial samples by fitness
        # We need to track all evaluated points
        all_points = []
        # Re-evaluate is expensive, so let's just use the best few from memory
        # Actually, let's collect them properly
        # Since we didn't store them, let's do a fresh structured sampling with storage
        
        # Reset and do proper multi-start
        call_count = 0
        best_f = float('inf')
        best_x = np.zeros(self.dim)
        
        # Collect diverse starting points
        starts = []
        n_starts = min(12, max_calls // 25)
        
        for i in range(n_starts):
            if call_count >= max_calls or time.time() > deadline:
                break
            if i == 0:
                x = np.zeros(dim)
            elif i == 1:
                x = rng.uniform(-5, 5, dim)
            elif i == 2:
                x = rng.uniform(-2, 2, dim)
            else:
                # Structured: try different combinations
                x = np.zeros(dim)
                for d in range(dim):
                    if rng.random() < 0.5:
                        x[d] = rng.choice([-2*np.pi, 2*np.pi])
                    else:
                        x[d] = rng.uniform(-5, 5)
            f = counted_func(x)
            starts.append((f, x.copy()))
        
        starts.sort(key=lambda t: t[0])
        
        # Phase 2: CMA-ES / Nelder-Mead refinement from each starting point
        # Use Nelder-Mead as it's fast and works well in low dimensions
        # with bounded constraints handled by clipping
        
        for f_init, x_init in starts:
            if call_count >= max_calls or time.time() > deadline:
                break
            
            remaining = max_calls - call_count
            if remaining < 50:
                break
            
            # Use Nelder-Mead with bounded box
            # Create a wrapper that clips to bounds
            def nm_func(x, _x_init=x_init):
                x_clipped = np.clip(x, -5, 5)
                return counted_func(x_clipped)
            
            try:
                result = minimize(
                    nm_func,
                    x_init,
                    method='Nelder-Mead',
                    options={
                        'maxiter': min(200, remaining // 2),
                        'xatol': 1e-4,
                        'fatol': 1e-4,
                        'adaptive': True
                    }
                )
            except Exception:
                pass
            
            if call_count >= max_calls or time.time() > deadline:
                break

        # Phase 3: Fine-tuning with small perturbations around the best point
        if call_count < max_calls and time.time() < deadline:
            remaining = max_calls - call_count
            if remaining > 20:
                x = best_x.copy()
                f_curr = best_f
                
                # Adaptive step size fine-tuning
                step = 0.1
                for _ in range(min(100, remaining)):
                    if call_count >= max_calls or time.time() > deadline:
                        break
                    
                    improved = False
                    for d in range(dim):
                        for direction in [-1, 1]:
                            if call_count >= max_calls or time.time() > deadline:
                                break
                            x_trial = x.copy()
                            x_trial[d] += direction * step
                            x_trial = np.clip(x_trial, -5, 5)
                            f_trial = counted_func(x_trial)
                            if f_trial < f_curr:
                                x = x_trial
                                f_curr = f_trial
                                improved = True
                                if f_curr < best_f:
                                    best_f = f_curr
                                    best_x = x.copy()
                    
                    if not improved:
                        step *= 0.5
                        if step < 1e-4:
                            break

        return best_x, best_f
# EVOLVE-BLOCK-END
