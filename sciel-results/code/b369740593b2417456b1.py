"""Optimizer using pattern search with adaptive restarts for BBOB f18."""
import numpy as np
import time

class Optimizer:
    """Pattern search with sparse random restarts for ill-conditioned functions."""
    
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
    
    def _clip(self, x):
        return np.clip(x, -5.0, 5.0)
    
    def _pattern_search(self, func, x0, fx0, budget_remaining, deadline, rng, global_best_f, global_best_x):
        """Pattern search from starting point x0."""
        d = self.dim
        calls_used = 0
        x = x0.copy()
        fx = fx0
        
        # Initial step size
        step = 1.0
        min_step = 1e-8
        max_step = 10.0
        
        # Build direction set: orthogonal + diagonal
        directions = []
        # Orthogonal directions
        for i in range(d):
            for sign in [1.0, -1.0]:
                dir_vec = np.zeros(d)
                dir_vec[i] = sign
                directions.append(dir_vec)
        # Diagonal directions (all coordinates same sign)
        for sign in [1.0, -1.0]:
            directions.append(sign * np.ones(d))
        # Some random directions
        for _ in range(d):
            dir_vec = rng.standard_normal(d)
            norm = np.linalg.norm(dir_vec)
            if norm > 1e-10:
                directions.append(dir_vec / norm)
        
        # Pattern search loop
        no_improve_count = 0
        max_no_improve = 30
        
        while calls_used < budget_remaining:
            if time.time() > deadline - 2.0:
                break
            
            improved = False
            
            for dir_vec in directions:
                if calls_used >= budget_remaining:
                    break
                if time.time() > deadline - 2.0:
                    break
                
                # Try expanding step
                for _ in range(5):
                    y = self._clip(x + step * dir_vec)
                    fy = func(y)
                    calls_used += 1
                    
                    if fy < fx:
                        fx = fy
                        x = y
                        improved = True
                        no_improve_count = 0
                        # Expand step
                        step = min(step * 2.0, max_step)
                        break
                    else:
                        # Try shrinking
                        step = max(step * 0.5, min_step)
                        if step <= min_step:
                            break
                
                if step <= min_step:
                    step = max_step * 0.1  # Reset to moderate step
            
            if not improved:
                no_improve_count += 1
                # Shrink all steps
                step = max(step * 0.5, min_step)
                if no_improve_count >= max_no_improve or step <= min_step * 1.1:
                    break
            
            # Occasionally re-randomize directions
            if no_improve_count % 5 == 0 and no_improve_count > 0:
                for i in range(len(directions)):
                    if i >= 2 * d + 2:  # Only re-randomize the random ones
                        dir_vec = rng.standard_normal(d)
                        norm = np.linalg.norm(dir_vec)
                        if norm > 1e-10:
                            directions[i] = dir_vec / norm
        
        return x, fx, calls_used
    
    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin
        dim = self.dim
        budget = self.budget
        
        rng = np.random.default_rng(42)
        
        best_x = np.zeros(dim)
        best_f = np.inf
        total_calls = 0
        
        # Phase 1: Sparse random sampling to find promising regions
        n_initial = min(30, budget // 10)
        
        # Evaluate origin
        x0 = np.zeros(dim)
        f0 = func(x0)
        total_calls += 1
        best_x, best_f = x0.copy(), f0
        
        # Random samples
        candidates = [(x0.copy(), f0)]
        for i in range(n_initial - 1):
            if total_calls >= budget:
                break
            if time.time() > deadline - 10.0:
                break
            xi = rng.uniform(-4.5, 4.5, size=dim)
            fi = func(xi)
            total_calls += 1
            candidates.append((xi.copy(), fi))
            if fi < best_f:
                best_f = fi
                best_x = xi.copy()
        
        # Sort by fitness
        candidates.sort(key=lambda t: t[1])
        
        # Select diverse starting points
        selected_starts = [candidates[0][0].copy()]
        for x, f in candidates[1:]:
            if len(selected_starts) >= min(8, len(candidates)):
                break
            # Check diversity
            too_close = False
            for s in selected_starts:
                if np.linalg.norm(x - s) < 0.8:
                    too_close = True
                    break
            if not too_close:
                selected_starts.append(x.copy())
        
        # Add a few more random diverse points if needed
        while len(selected_starts) < 10 and total_calls < budget - 20:
            x_new = rng.uniform(-4.0, 4.0, size=dim)
            too_close = False
            for s in selected_starts:
                if np.linalg.norm(x_new - s) < 0.5:
                    too_close = True
                    break
            if not too_close:
                selected_starts.append(x_new)
        
        # Phase 2: Pattern search from each starting point
        calls_per_run = max(20, (budget - total_calls) // len(selected_starts))
        
        for start_idx, x0 in enumerate(selected_starts):
            if total_calls >= budget:
                break
            if time.time() > deadline - 5.0:
                break
            
            remaining = budget - total_calls
            max_calls_this = min(calls_per_run, remaining)
            
            if max_calls_this < 5:
                continue
            
            # Get function value at starting point
            if start_idx == 0:
                fx0 = f0
            else:
                # Check if we already evaluated this point
                fx0 = None
                for x_cand, f_cand in candidates:
                    if np.allclose(x0, x_cand, atol=1e-10):
                        fx0 = f_cand
                        break
                if fx0 is None:
                    fx0 = func(x0)
                    total_calls += 1
                    if fx0 < best_f:
                        best_f = fx0
                        best_x = x0.copy()
            
            # Run pattern search
            x_new, fx_new, calls_used = self._pattern_search(
                func, x0, fx0, max_calls_this, deadline, rng, best_f, best_x
            )
            total_calls += calls_used
            
            if fx_new < best_f:
                best_f = fx_new
                best_x = x_new.copy()
        
        # Phase 3: Final refinement with smaller steps from best point
        if total_calls < budget and time.time() < deadline - 5.0:
            remaining = budget - total_calls
            x_final, fx_final, calls_used = self._pattern_search(
                func, best_x.copy(), best_f, remaining, deadline, rng, best_f, best_x
            )
            total_calls += calls_used
            if fx_final < best_f:
                best_f = fx_final
                best_x = x_final.copy()
        
        return best_x, best_f
