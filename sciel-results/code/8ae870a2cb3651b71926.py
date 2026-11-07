"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Adaptive ridge-following optimizer for BBOB f13 (Sharp ridge).
    
    Strategy:
    1. Initial exploration with random restarts and (1+1)-ES to find a good basin.
    2. Powell's method to align search directions with the ridge orientation.
    3. Final local refinement.
    
    Key insight: f13 has a sharp ridge, so standard ES struggles to move along it.
    Powell's method adaptively constructs directions that follow the ridge.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls_made = 0
        self.start_time = None

    def _time_ok(self, margin=10.0):
        """Check if we have enough time left before the 330s deadline."""
        if self.start_time is None:
            return True
        elapsed = time.time() - self.start_time
        return elapsed < 330.0 - margin

    def _call(self, func, x):
        """Call func with budget tracking and time guard."""
        if not self._time_ok():
            return None
        self.calls_made += 1
        if self.calls_made > self.budget:
            return None
        return func(x)

    def _powell(self, func, x0, fx0, max_calls, time_limit=10.0):
        """
        Powell's method for line search optimization.
        Returns (best_x, best_fx, calls_used).
        """
        n = self.dim
        x = x0.copy()
        fx = fx0
        calls_used = 0
        start_t = time.time()
        
        # Initialize directions to standard basis
        directions = [np.eye(n, dtype=float)[i] for i in range(n)]
        
        max_iter = max_calls // (n + 1)
        for _ in range(max_iter):
            if not self._time_ok(margin=time_limit):
                break
            if calls_used >= max_calls:
                break
            
            # Line search along each direction
            old_x = x.copy()
            old_fx = fx
            min_fx = fx
            min_dir_idx = 0
            
            for i in range(n):
                if not self._time_ok(margin=time_limit) or calls_used >= max_calls:
                    break
                dir_i = directions[i]
                
                # Find best step along this direction using golden section or simple search
                # Simple approach: try a few steps
                best_step = 0.0
                best_fx_dir = fx
                
                # Try positive and negative steps
                step_sizes = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0]
                for s in step_sizes:
                    for sign in [1, -1]:
                        if not self._time_ok(margin=time_limit) or calls_used >= max_calls:
                            break
                        y = x + sign * s * dir_i
                        y = np.clip(y, -5, 5)
                        fy = self._call(func, y)
                        if fy is None:
                            break
                        calls_used += 1  # Already counted in _call, but track locally too
                        if fy < best_fx_dir:
                            best_fx_dir = fy
                            best_step = sign * s
                    if not self._time_ok(margin=time_limit) or calls_used >= max_calls:
                        break
                
                if best_step != 0.0:
                    x = np.clip(x + best_step * dir_i, -5, 5)
                    fx = best_fx_dir
                    if fx < min_fx:
                        min_fx = fx
                        min_dir_idx = i
            
            # Replace the direction that gave the largest improvement with the new direction
            new_dir = x - old_x
            if np.linalg.norm(new_dir) > 1e-10:
                directions[min_dir_idx] = new_dir / np.linalg.norm(new_dir)
            
            # Check for convergence
            if abs(fx - old_fx) < 1e-12:
                break
        
        return x, fx, calls_used

    def _local_es(self, func, x0, fx0, sigma0, max_calls, time_limit=10.0):
        """
        (1+1)-ES with adaptive step size.
        Returns (best_x, best_fx, calls_used).
        """
        n = self.dim
        x = x0.copy()
        fx = fx0
        sigma = sigma0
        calls_used = 0
        start_t = time.time()
        rng = np.random.default_rng(42)
        
        for _ in range(max_calls):
            if not self._time_ok(margin=time_limit) or calls_used >= max_calls:
                break
            
            y = np.clip(x + sigma * rng.standard_normal(n), -5, 5)
            fy = self._call(func, y)
            if fy is None:
                break
            calls_used += 1
            
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.1
            else:
                sigma *= 0.9
        
        return x, fx, calls_used

    def __call__(self, func):
        self.start_time = time.time()
        n = self.dim
        budget = self.budget
        
        # Phase 1: Multiple random restarts with short ES to find a good starting point
        # Use ~30% of budget for exploration
        explore_budget = max(50, budget // 3)
        n_restarts = max(3, explore_budget // 15)
        calls_per_restart = explore_budget // n_restarts
        
        best_x = None
        best_fx = float('inf')
        
        rng = np.random.default_rng(123)
        
        for r in range(n_restarts):
            if not self._time_ok(margin=15.0) or self.calls_made >= budget:
                break
            
            # Random start
            x = rng.uniform(-5, 5, n)
            fx = self._call(func, x)
            if fx is None:
                break
            self.calls_made += 1
            
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            
            # Short ES from this point
            x_es, fx_es, calls_used = self._local_es(
                func, x, fx, sigma0=1.0, 
                max_calls=calls_per_restart - 1, 
                time_limit=15.0
            )
            self.calls_made += calls_used
            
            if fx_es < best_fx:
                best_fx = fx_es
                best_x = x_es.copy()
        
        if best_x is None:
            # Fallback: just return a random point
            x = np.zeros(n)
            fx = self._call(func, x)
            if fx is not None:
                self.calls_made += 1
                best_x, best_fx = x, fx
            else:
                best_x, best_fx = np.zeros(n), 0.0
        
        # Phase 2: Powell's method to align with the ridge
        # Use ~40% of remaining budget
        remaining = budget - self.calls_made
        powell_budget = max(50, int(remaining * 0.4))
        
        x_powell, fx_powell, calls_used = self._powell(
            func, best_x, best_fx, 
            max_calls=powell_budget, 
            time_limit=15.0
        )
        self.calls_made += calls_used
        
        if fx_powell < best_fx:
            best_fx = fx_powell
            best_x = x_powell.copy()
        
        # Phase 3: Final local ES refinement
        remaining = budget - self.calls_made
        es_budget = max(20, remaining)
        
        x_final, fx_final, calls_used = self._local_es(
            func, best_x, best_fx, sigma0=0.5,
            max_calls=es_budget,
            time_limit=10.0
        )
        self.calls_made += calls_used
        
        if fx_final < best_fx:
            best_fx = fx_final
            best_x = x_final.copy()
        
        return best_x, best_fx
# EVOLVE-BLOCK-END
