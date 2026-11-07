import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _halton_sequence(self, n, dim):
        """Generate Halton sequence points in [0, 1]^dim."""
        points = np.zeros((n, dim))
        for i in range(1, n + 1):
            for d_idx in range(dim):
                base = d_idx + 2
                f = 1.0
                r = 0.0
                i_temp = i
                while i_temp > 0:
                    f /= base
                    r += f * (i_temp % base)
                    i_temp //= base
                points[i - 1, d_idx] = r
        return points

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0
        d = self.dim
        total_budget = self.budget
        
        best_x = None
        best_f = np.inf
        calls_used = 0

        def track_call(x):
            nonlocal best_x, best_f, calls_used
            x_clipped = np.clip(x, -5.0, 5.0)
            f = func(x_clipped)
            calls_used += 1
            if f < best_f:
                best_f = f
                best_x = x_clipped.copy()
            return f

        # Phase 1: Halton Initialization
        # Use 40 points for initial sampling to cover the space well.
        n_halton = min(40, total_budget)
        halton_pts = self._halton_sequence(n_halton, d)
        for pt in halton_pts:
            if calls_used >= total_budget or time.time() > deadline:
                break
            x = -5.0 + 10.0 * pt
            track_call(x)

        if best_x is None:
            track_call(np.zeros(d))

        # Phase 2: CMA-ES Optimization
        # Implement a lightweight CMA-ES or use a robust local search like Nelder-Mead with restarts.
        # Given the budget (500) and dimension (5), CMA-ES is very effective.
        # However, implementing full CMA-ES is complex. Let's use a simplified version or Nelder-Mead with careful tuning.
        # The previous solution used Nelder-Mead from top 5 candidates.
        # Let's try a CMA-ES implementation since it's known to be superior for this type of problem.
        
        # Since we can't rely on external libraries other than numpy/scipy, and scipy's CMA-ES is not built-in (it's in cma package which might not be available), 
        # we should stick to scipy.optimize.minimize if available, or implement a simple evolution strategy.
        # The prompt says "Python standard library, NumPy and SciPy are available".
        # So scipy.optimize.minimize is safe.
        
        # Let's refine the Nelder-Mead approach with more aggressive restarts or a better initial population.
        # Actually, let's try a simple (1+1) ES or (mu+lambda) ES which is robust and easy to implement correctly.
        # But CMA-ES is better. Let's check if we can use scipy.optimize.minimize with method='Nelder-Mead' effectively.
        
        # The current score is 0.9826. The best score seen is 1.0722 (inspiration 1).
        # Inspiration 1 used numpy, time; defines Optimizer. It might have used a better strategy.
        # Let's try to implement a basic CMA-ES manually to avoid dependency issues and gain control.
        
        # CMA-ES Parameters
        pop_size = max(4, int(4 + 3 * np.log(d)))  # 4 + 3*ln(5) ~ 7
        sigma = 2.0  # Initial step size
        x = best_x.copy()
        B = np.eye(d)
        D = np.ones(d)
        ps = np.zeros(d)
        pc = np.zeros(d)
        cs = (pop_size + 2) / (d + pop_size + 5)
        cmu = min(1 - cs, 2 / ((d + 2) ** 2 + pop_size))
        ds = 1 + 2 * max(0, np.sqrt((pop_size - 1) / (d + 1)) - 1) + cs
        cc = (4 + d / pop_size) / (d + 4 + 2 * pop_size / d)
        hsig = (1 + 6 / (d + pop_size)) / np.sqrt(2 * (d + 1))
        chi_n = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d ** 2))
        
        # We need to track the calls carefully.
        # Each generation requires pop_size evaluations.
        # Budget remaining: total_budget - calls_used
        # Number of generations: (total_budget - calls_used) // pop_size
        
        max_generations = (total_budget - calls_used) // pop_size
        if max_generations < 1:
            return best_x, best_f

        for gen in range(max_generations):
            if time.time() > deadline:
                break
            
            # Generate population
            weights = np.ones(pop_size) / pop_size  # Simple uniform weights for now
            
            # Sample points
            pop = []
            for i in range(pop_size):
                z = np.random.randn(d)
                y = sigma * (B @ (D * z))
                x_new = x + y
                pop.append((x_new, y))
            
            # Evaluate population
            pop_f = []
            for x_new, y in pop:
                if calls_used >= total_budget or time.time() > deadline:
                    break
                f_val = track_call(x_new)
                pop_f.append((f_val, x_new, y))
            
            if len(pop_f) < pop_size:
                break
                
            # Sort by fitness
            pop_f.sort(key=lambda item: item[0])
            
            # Update best
            if pop_f[0][0] < best_f:
                best_f = pop_f[0][0]
                best_x = pop_f[0][1].copy()
                
            # Compute new center
            new_x = np.zeros(d)
            for i in range(pop_size):
                new_x += weights[i] * pop_f[i][1]
            
            # Update evolution path for covariance matrix
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) / sigma * (B @ (D @ ((new_x - x) / sigma)))
            # mu_eff is usually the effective population size, here approx pop_size
            mu_eff = pop_size
            
            # Update evolution path for mean
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1)))) < (1 + 6 / (d + pop_size)) * chi_n
            pc = (1 - cc) * pc + hsig * cc * (new_x - x) / sigma
            
            # Update covariance matrix
            delta_hsig = (1 - hsig) * cc * (2 - cc)
            C = (1 - cmu * ds) * C if 'C' in locals() else None # This is getting complicated without a full implementation
            
            # Let's simplify: just use Nelder-Mead from the best point found so far in the ES, 
            # or restart Nelder-Mead if stagnation occurs.
            # Actually, the previous solution with Nelder-Mead from top 5 Halton points scored 0.9826.
            # Let's try a hybrid: Halton init, then CMA-ES-like restarts or just a very robust Nelder-Mead.
            
            # Let's abandon the manual CMA-ES due to complexity and risk of bugs.
            # Instead, let's use multiple Nelder-Mead restarts from diverse points.
            # But we already used Halton.
            
            # Let's try: After Halton, run Nelder-Mead from the best point.
            # If it stagnates (no improvement for N calls), restart from the second best, etc.
            
            break # Exit the CMA-ES loop and go to Nelder-Mead phase

        # Phase 2: Nelder-Mead Local Optimization
        # Sort all points evaluated so far by function value
        # We didn't keep a full history in the ES loop above, but we did in the Halton phase.
        # Let's restart the strategy:
        
        # 1. Halton init (40 points)
        # 2. Run Nelder-Mead from the best point found in Halton.
        # 3. If Nelder-Mead finishes or stagnates, run from the next best Halton point.
        
        # We need to keep track of all evaluated points.
        # Let's rewrite the __call__ to be cleaner.
        
        pass

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0
        d = self.dim
        total_budget = self.budget
        
        best_x = None
        best_f = np.inf
        calls_used = 0
        history = []

        def track_call(x):
            nonlocal best_x, best_f, calls_used
            x_clipped = np.clip(x, -5.0, 5.0)
            f = func(x_clipped)
            calls_used += 1
            history.append((f, x_clipped.copy()))
            if f < best_f:
                best_f = f
                best_x = x_clipped.copy()
            return f

        # Phase 1: Halton Initialization
        n_halton = min(40, total_budget)
        halton_pts = self._halton_sequence(n_halton, d)
        for pt in halton_pts:
            if calls_used >= total_budget or time.time() > deadline:
                break
            x = -5.0 + 10.0 * pt
            track_call(x)

        if best_x is None:
            track_call(np.zeros(d))

        # Phase 2: Local Optimization from Top Candidates
        history.sort(key=lambda item: item[0])
        
        remaining_budget = total_budget - calls_used
        if remaining_budget <= 0:
            return best_x, best_f

        # Select top 5 unique candidates
        top_candidates = [x for f, x in history[:5]]
        
        # Run Nelder-Mead from each top candidate until budget is exhausted
        for idx, start_x in enumerate(top_candidates):
            if calls_used >= total_budget or time.time() > deadline:
                break
            
            def wrapped_func(x):
                nonlocal calls_used
                if calls_used >= total_budget or time.time() > deadline:
                    return np.inf
                calls_used += 1
                return func(np.clip(x, -5.0, 5.0))

            try:
                from scipy.optimize import minimize
                res = minimize(
                    wrapped_func, 
                    start_x, 
                    method='Nelder-Mead',
                    options={
                        'maxiter': 200, 
                        'xatol': 1e-8, 
                        'fatol': 1e-10,
                        'adaptive': True
                    }
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                pass

        return best_x, best_f
