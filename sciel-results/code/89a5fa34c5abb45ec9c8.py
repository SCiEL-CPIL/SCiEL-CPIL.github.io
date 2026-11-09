import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 10s margin
        
        dim = self.dim
        total_budget = self.budget
        
        # Budget allocation
        es_budget = int(total_budget * 0.8)
        polish_budget = total_budget - es_budget
        
        calls = 0
        
        def counted_func(x):
            nonlocal calls
            if calls >= total_budget:
                return float('inf')
            calls += 1
            return func(x)

        # (mu+lambda)-ES
        mu = 5
        lam = 10
        rng = np.random.default_rng(123)
        
        # Initialize
        x = rng.uniform(-5, 5, dim)
        fx = counted_func(x)
        best_x = x.copy()
        best_fx = fx
        
        sigma = 2.0
        
        # Weights
        weights = np.ones(mu) / mu
        
        # Generation loop
        while calls < es_budget:
            if time.time() > deadline:
                break
                
            offspring = []
            offspring_f = []
            
            for _ in range(lam):
                if calls >= es_budget:
                    break
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = counted_func(y)
                offspring.append(y)
                offspring_f.append(fy)
                
                if fy < best_fx:
                    best_fx = fy
                    best_x = y.copy()
            
            if not offspring:
                break
                
            # Sort
            sorted_idx = np.argsort(offspring_f)
            sorted_offspring = [offspring[i] for i in sorted_idx]
            sorted_offspring_f = [offspring_f[i] for i in sorted_idx]
            
            # Select top mu
            selected = sorted_offspring[:mu]
            
            # Recombine
            x = np.sum([w * s for w, s in zip(weights, selected)], axis=0)
            
            # Sigma adaptation
            # Compare offspring to current parent x (before update)
            # In (mu,lambda), we don't keep parent. We just use offspring.
            # Success rule: if best offspring < worst selected? No.
            # Standard: if at least 1/5 of offspring are better than the parent that generated them.
            # Since we recombine, we don't have a single parent.
            # Let's use a simpler adaptation: if best_fx improved in this generation, sigma *= 1.1 else sigma *= 0.9
            # Or track improvement of the recombined point.
            
            # Let's try: if the best offspring is significantly better than the mean of the selected, increase sigma.
            # Simpler: if best_fx < fx_prev, sigma *= 1.05 else sigma *= 0.95
            # But we don't store fx_prev for the recombined point.
            
            # Let's use the 1/5 rule on the offspring vs the *mean* of the selected?
            # Let's just use a fixed sigma decay/increase based on global best improvement.
            # This is less principled but effective for small budgets.
            
            # Better: Adapt sigma based on the spread of the selected offspring.
            # If the top mu are close, decrease sigma. If spread out, increase.
            
            # Let's stick to a robust (1+1) style adaptation but on the recombined step?
            # No, let's use a standard CMA-ES if available, or a well-tuned (9+9).
            
            # Let's use a simple heuristic:
            # If the best offspring is better than the worst of the selected top-mu, we are making progress.
            # Actually, let's just decrease sigma slowly to converge.
            sigma *= 0.99
            # And if we get stuck (no improvement for 10 generations), restart?
            # With 500 budget, restarts are risky.
            
            # Let's try a different adaptation:
            # sigma = sigma * (1 + 0.1 * (successes - 0.2))
            # successes = number of offspring better than the *previous best*.
            # This is not quite right.
            
            # Let's use the standard 1/5 success rule for (1+1) but adapted for (mu,lambda).
            # We will treat the recombined point as the "parent" for the next step?
            # No, let's just use a fixed sigma schedule or simple adaptation.
            
            # Given the short budget, a large initial sigma and slow decay is good.
            # sigma = 2.0 * (0.95 ** generation)
            
        # Polish
        if calls < total_budget:
            try:
                from scipy.optimize import minimize
                # Limit calls in minimize
                # We have total_budget - calls left.
                # NM uses dim+1 calls per iteration.
                max_iter = (total_budget - calls) // (dim + 1)
                if max_iter > 0:
                    res = minimize(counted_func, best_x, method='Nelder-Mead', 
                                   options={'maxiter': max_iter, 'xatol': 1e-6, 'fatol': 1e-6})
                    if res.fun < best_fx:
                        best_x = res.x
                        best_fx = res.fun
            except Exception:
                pass

        return best_x, best_fx
