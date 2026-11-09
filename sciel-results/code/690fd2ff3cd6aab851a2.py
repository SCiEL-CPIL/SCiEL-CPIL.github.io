import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = 290  # Safety margin from 300s limit
        calls = 0
        best_x = None
        best_f = float('inf')
        rng = np.random.default_rng(42)
        
        def safe_eval(x):
            nonlocal calls, best_x, best_f
            if calls >= self.budget or (time.time() - t0) > deadline:
                return float('inf')
            calls += 1
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Global Sampling
        n_global = 20
        # Use Sobol-like or just random for simplicity, random is fine for 5D with 20 pts
        X_global = rng.uniform(-5, 5, (n_global, self.dim))
        for x in X_global:
            safe_eval(x)
            if calls >= self.budget or (time.time() - t0) > deadline:
                break

        # Get top 3 candidates
        # We need to track which x gave which f.
        # Since safe_eval doesn't return x, we should store them.
        # Let's modify Phase 1 to store (x, f)
        
        # Re-do Phase 1 with storage
        candidates = []
        calls = 0
        best_x = None
        best_f = float('inf')
        
        def safe_eval_store(x):
            nonlocal calls, best_x, best_f
            if calls >= self.budget or (time.time() - t0) > deadline:
                return float('inf')
            calls += 1
            fx = func(x)
            candidates.append((x, fx))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Reset and run Phase 1 properly
        calls = 0
        best_x = None
        best_f = float('inf')
        candidates = []
        
        X_global = rng.uniform(-5, 5, (20, self.dim))
        for x in X_global:
            safe_eval_store(x)
            if calls >= self.budget or (time.time() - t0) > deadline:
                break

        if not candidates:
            return best_x, best_f

        # Sort candidates by f
        candidates.sort(key=lambda item: item[1])
        top_k = 3
        top_candidates = candidates[:top_k]
        
        # Phase 2: Local Refinement with Nelder-Mead
        remaining_budget = self.budget - calls
        # Allocate budget: 50%, 30%, 20%
        allocs = [0.5, 0.3, 0.2]
        
        for i, (x0, f0) in enumerate(top_candidates):
            if calls >= self.budget or (time.time() - t0) > deadline:
                break
            
            budget_for_this = int(remaining_budget * allocs[i]) if i < len(allocs) else 0
            if budget_for_this < 10:
                budget_for_this = 10 # Minimum budget for NM
                
            self._nelder_mead(x0, safe_eval, budget_for_this, self.dim, deadline, t0)
            remaining_budget -= budget_for_this
            
            if calls >= self.budget or (time.time() - t0) > deadline:
                break

        return best_x, best_f

    def _nelder_mead(self, x0, f_eval, budget, dim, deadline, t0):
        # Initialize simplex
        simplex = [x0.copy()]
        # Create dim other points
        for i in range(dim):
            xi = x0.copy()
            # Small step size, maybe 0.1 or 0.5
            xi[i] += 0.5
            simplex.append(xi)
            
        # Evaluate initial simplex
        fs = []
        for x in simplex:
            fs.append(f_eval(x))
            if calls >= budget or (time.time() - t0) > deadline:
                return

        alpha, gamma, rho, sigma_ = 1.0, 2.0, 0.5, 0.5
        
        while budget > 0 and (time.time() - t0) < deadline:
            # Sort simplex
            idx = np.argsort(fs)
            simplex = [simplex[i] for i in idx]
            fs = [fs[i] for i in idx]
            
            x_c = np.mean(simplex[:-1], axis=0)
            x_w = simplex[-1]
            f_w = fs[-1]
            
            # Reflection
            x_r = x_c + alpha * (x_c - x_w)
            x_r = np.clip(x_r, -5, 5)
            f_r = f_eval(x_r)
            budget -= 1
            if budget <= 0 or (time.time() - t0) > deadline:
                return

            if f_r < fs[0]:
                # Expansion
                x_e = x_c + gamma * (x_r - x_w)
                x_e = np.clip(x_e, -5, 5)
                f_e = f_eval(x_e)
                budget -= 1
                if budget <= 0 or (time.time() - t0) > deadline:
                    return
                if f_e < f_r:
                    simplex[-1] = x_e
                    fs[-1] = f_e
                else:
                    simplex[-1] = x_r
                    fs[-1] = f_r
            elif f_r < fs[-2]:
                # Accept reflection
                simplex[-1] = x_r
                fs[-1] = f_r
            else:
                # Contraction
                if f_r < f_w:
                    x_c2 = x_c + rho * (x_r - x_c)
                else:
                    x_c2 = x_c - rho * (x_r - x_c)
                x_c2 = np.clip(x_c2, -5, 5)
                f_c = f_eval(x_c2)
                budget -= 1
                if budget <= 0 or (time.time() - t0) > deadline:
                    return
                if f_c < f_r:
                    simplex[-1] = x_c2
                    fs[-1] = f_c
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + sigma_ * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        fs[i] = f_eval(simplex[i])
                        budget -= 1
                        if budget <= 0 or (time.time() - t0) > deadline:
                            return
