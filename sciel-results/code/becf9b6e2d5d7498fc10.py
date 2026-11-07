import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0  # Safety margin
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = float('inf')
        calls = 0
        eval_history = []

        def evaluate(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return best_f
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            calls += 1
            eval_history.append((fx, x.copy()))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def run_local(x0, method='Powell', max_iter=50):
            nonlocal calls
            if calls >= budget or time.time() >= deadline:
                return
            try:
                def objective(x):
                    return evaluate(x)
                
                if method == 'Powell':
                    minimize(
                        objective,
                        x0,
                        method='Powell',
                        bounds=[(-5.0, 5.0)] * n,
                        options={'maxiter': max_iter, 'ftol': 1e-8}
                    )
                else: # Nelder-Mead
                    minimize(
                        objective,
                        x0,
                        method='Nelder-Mead',
                        options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
                    )
            except Exception:
                pass

        # Phase 1: Exploration
        # Use 100 Sobol points for good coverage
        sobol = qmc.Sobol(d=n, scramble=True, seed=42)
        num_sobol = 100
        sobol_points = sobol.random_base2(m=7) # 128 points
        sobol_points = sobol_points[:num_sobol]
        sobol_points = sobol_points * 10.0 - 5.0
        
        for x in sobol_points:
            if calls >= budget or time.time() >= deadline:
                break
            evaluate(x)
            
        # Add a few points near the center and corners just in case
        extra_points = [
            np.zeros(n),
            np.ones(n) * 2.0,
            np.ones(n) * -2.0,
            np.array([2.0, 2.0, 2.0, 2.0, 2.0]),
            np.array([-2.0, -2.0, -2.0, -2.0, -2.0]),
        ]
        for x in extra_points:
            if calls >= budget or time.time() >= deadline:
                break
            evaluate(x)

        # Phase 2: Select top candidates
        if eval_history and calls < budget and time.time() < deadline:
            sorted_hist = sorted(eval_history, key=lambda t: t[0])
            # Select top 5 diverse points
            selected = []
            for fx, x in sorted_hist:
                is_diverse = True
                for _, sx in selected:
                    if np.linalg.norm(x - sx) < 0.5:
                        is_diverse = False
                        break
                if is_diverse:
                    selected.append((fx, x))
                if len(selected) >= 5:
                    break
            
            # If not enough diverse, fill with next best
            if len(selected) < 5:
                for fx, x in sorted_hist:
                    if (fx, x) not in selected:
                        selected.append((fx, x))
                    if len(selected) >= 5:
                        break

            # Phase 3: Local Optimization
            # Allocate remaining budget to local searches
            remaining_budget = budget - calls
            if remaining_budget > 0:
                calls_per_search = max(10, remaining_budget // len(selected))
                
                for fx, x in selected:
                    if calls >= budget or time.time() >= deadline:
                        break
                    # Run Powell first, it's robust for valleys
                    run_local(x, method='Powell', max_iter=calls_per_search)
                    
                    # If we still have budget and time, try NM for fine tuning
                    if calls < budget and time.time() < deadline:
                        run_local(x, method='Nelder-Mead', max_iter=10)

        # Phase 4: Final Polish
        if best_x is not None and calls < budget and time.time() < deadline:
            run_local(best_x, method='Nelder-Mead', max_iter=20)

        return best_x, best_f
