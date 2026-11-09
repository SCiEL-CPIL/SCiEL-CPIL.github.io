# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Hybrid (1+1)-ES with CMA-ES-style covariance adaptation and deadline guard.
    Combines the simple, fast (1+1)-ES structure (inspiration 1/2) with
    covariance matrix adaptation (inspiration 3) to handle Rosenbrock's valley.
    
    Strategy:
    1. Use a small initial random search to find a good starting point.
    2. Run a covariance-adapting (1+1)-ES that learns the local geometry.
    3. If time/budget remains, continue polishing with the same mechanism.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.perf_counter()
        deadline = start_time + 280.0  # 300s limit, 20s safety margin
        rng = np.random.default_rng(0)
        n = self.dim
        b = self.budget

        # Track best solution found so far
        best_x = None
        best_f = np.inf
        call_count = 0

        def safe_call(x):
            nonlocal best_x, best_f, call_count
            if call_count >= b:
                return np.inf
            if time.perf_counter() > deadline:
                return np.inf
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            try:
                fx = float(func(x))
            except Exception:
                fx = np.inf
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # --- Phase 1: Initial random search to find a good starting point ---
        # Use ~5% of budget for initial exploration (min 5, max 25)
        n_init = max(5, min(25, b // 20))
        for _ in range(n_init):
            if call_count >= b or time.perf_counter() > deadline:
                break
            x0 = rng.uniform(-5.0, 5.0, n)
            safe_call(x0)

        if best_x is None:
            # Fallback: should not happen if b > 0
            best_x = np.zeros(n)
            best_f = safe_call(best_x)

        # --- Phase 2: Covariance-adapting (1+1)-ES ---
        # This is a (1+1)-ES that maintains a covariance matrix C to adapt
        # the search direction to the local landscape geometry.
        x = best_x.copy()
        fx = best_f
        sigma = 1.0  # Initial step size
        C = np.eye(n)  # Covariance matrix
        C_inv = np.eye(n)
        
        # Learning rate for covariance update
        # For (1+1), we use a simple update: on success, move toward the direction
        # that worked; on failure, shrink slightly.
        c_c = 0.5  # Covariance adaptation rate
        
        # Step size adaptation (1/5 rule variant)
        sigma_up = 1.1
        sigma_down = 0.9
        
        # Run until budget or deadline
        while call_count < b and time.perf_counter() < deadline:
            # Generate candidate using current covariance
            z = rng.standard_normal(n)
            # Cholesky-like transform: C @ z (approximate via eigendecomp if needed, but for n=5, C@z is fine)
            y = np.clip(x + sigma * (C @ z), -5.0, 5.0)
            fy = safe_call(y)
            
            if fy < fx:
                # Success: move to y, increase sigma, adapt C toward the direction
                direction = (y - x) / (sigma * (np.linalg.norm(z) + 1e-10))
                # Update C: emphasize the direction that worked
                # Simple rank-1 update: C = (1 - c_c) * C + c_c * outer(direction, direction)
                C = (1 - c_c) * C + c_c * np.outer(direction, direction)
                # Symmetrize
                C = (C + C.T) / 2
                # Ensure positive definite
                try:
                    C_inv = np.linalg.inv(C)
                except np.linalg.LinAlgError:
                    C_inv = np.eye(n)
                    C = np.eye(n)
                
                x = y
                fx = fy
                sigma *= sigma_up
            else:
                # Failure: shrink sigma, optionally shrink C slightly
                sigma *= sigma_down
                # Don't update C on failure to avoid overfitting to noise
                # Or slightly decay toward identity to reset if stuck
                # C = (1 - 0.1 * c_c) * C + 0.1 * c_c * np.eye(n)
                pass
            
            # Safety: if sigma becomes too small, reset C and sigma
            if sigma < 1e-6:
                sigma = 0.5
                C = np.eye(n)
                C_inv = np.eye(n)
        
        # --- Phase 3: Final polish if budget remains ---
        # If we have time and budget, do a few more steps with smaller sigma
        if call_count < b and time.perf_counter() < deadline and best_x is not None:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1  # Small step for polishing
            C = np.eye(n)
            
            # Run a simple (1+1)-ES with fixed small sigma
            for _ in range(min(50, b - call_count)):
                if call_count >= b or time.perf_counter() > deadline:
                    break
                z = rng.standard_normal(n)
                y = np.clip(x + sigma * z, -5.0, 5.0)
                fy = safe_call(y)
                if fy < fx:
                    x = y
                    fx = fy
                    sigma *= 1.1
                else:
                    sigma *= 0.9
                
                if sigma < 1e-8:
                    break

        if best_x is None:
            best_x = np.zeros(n)
            best_f = np.inf

        return best_x, best_f
# EVOLVE-BLOCK-END
