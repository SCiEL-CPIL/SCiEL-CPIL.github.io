import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        budget = self.budget
        dim = self.dim
        rng = np.random.default_rng(12345)

        best_x = None
        best_f = np.inf
        calls = 0

        def try_call(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return None
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def run_asa(x_start, t0, n_iter, rng_asa):
            """
            Adaptive Simulated Annealing:
            - Uses deterministic ASA direction: x_new = x + (t/T0) * (x_prev - x) + noise
            - Actually uses the standard ASA update: 
              x_new = x + (t_k / T0) * (x_best - x) + sigma * noise
            where t_k decays geometrically.
            """
            nonlocal best_x, best_f, calls
            
            x = np.asarray(x_start, dtype=np.float64).copy()
            fx = try_call(x)
            if fx is None:
                return x
            
            x_best_local = x.copy()
            fx_best_local = fx
            t = t0
            t_min = 1e-8
            cooling = 0.95  # geometric cooling rate
            
            # Track recent history for ASA direction
            x_prev = x.copy()
            fx_prev = fx
            
            for it in range(n_iter):
                if calls >= budget or time.time() >= deadline:
                    break
                    
                # ASA direction: pull toward best, add noise scaled by temperature
                # Direction component: (x_best - x) * (t / T0)
                pull = (x_best_local - x) * (t / t0)
                
                # Noise component
                noise_scale = t * 0.5
                noise = rng_asa.standard_normal(dim) * noise_scale
                
                x_new = x + pull + noise
                x_new = np.clip(x_new, -5.0, 5.0)
                
                fx_new = try_call(x_new)
                if fx_new is None:
                    break
                
                # Metropolis acceptance: always accept if better, else with prob exp(-delta/t)
                if fx_new < fx:
                    x = x_new
                    fx = fx_new
                    x_prev = x.copy()
                    fx_prev = fx
                else:
                    delta = fx_new - fx
                    if delta < 0:
                        delta = 0
                    accept_prob = np.exp(-delta / max(t, 1e-12))
                    if rng_asa.random() < accept_prob:
                        x = x_new
                        fx = fx_new
                        x_prev = x.copy()
                        fx_prev = fx
                    else:
                        x_prev = x.copy()
                        fx_prev = fx
                
                # Update local best
                if fx < fx_best_local:
                    fx_best_local = fx
                    x_best_local = x.copy()
                
                # Cool down
                t *= cooling
                if t < t_min:
                    t = t_min
            
            return x_best_local

        def coord_descent(x_start, budget_cd, rng_cd):
            nonlocal best_x, best_f, calls
            x = np.asarray(x_start, dtype=np.float64).copy()
            f = try_call(x)
            if f is None:
                return
            step = 1.0
            no_improve = 0
            max_no_improve = 150
            while calls < budget_cd and time.time() < deadline and no_improve < max_no_improve:
                improved = False
                for i in range(dim):
                    for s in (step, -step):
                        xn = x.copy()
                        xn[i] += s
                        fn = try_call(xn)
                        if fn is None:
                            return
                        if fn < f:
                            x = xn
                            f = fn
                            improved = True
                            break
                if improved:
                    no_improve = 0
                    step *= 0.7
                else:
                    no_improve += 1
                    step *= 0.5
                if step < 1e-9:
                    break
            return x

        # ---- Phase 1: diverse random initialization ----
        # Evaluate a set of diverse starting points
        n_init = 15
        for i in range(n_init):
            if calls >= budget or time.time() >= deadline:
                break
            if i == 0:
                x0 = np.zeros(dim)
            else:
                x0 = rng.uniform(-4, 4, dim)
            try_call(x0)

        # ---- Phase 2: Multi-start ASA ----
        # Run ASA from multiple starting points with different initial temperatures
        n_asa_starts = 6
        asa_budget_each = max(10, (budget - calls - 50) // n_asa_starts)
        
        for s in range(n_asa_starts):
            if calls >= budget or time.time() >= deadline:
                break
            
            # Choose starting point: mix of current best and random perturbations
            if s == 0:
                x_start = best_x.copy()
                t0 = 2.0
            elif s == 1:
                x_start = best_x + rng.normal(0, 1.5, dim)
                x_start = np.clip(x_start, -5.0, 5.0)
                t0 = 1.5
            elif s == 2:
                x_start = best_x + rng.normal(0, 0.5, dim)
                x_start = np.clip(x_start, -5.0, 5.0)
                t0 = 1.0
            elif s == 3:
                x_start = rng.uniform(-4, 4, dim)
                t0 = 2.5
            elif s == 4:
                x_start = rng.uniform(-3, 3, dim)
                t0 = 1.8
            else:
                x_start = best_x + rng.normal(0, 0.3, dim)
                x_start = np.clip(x_start, -5.0, 5.0)
                t0 = 0.8
            
            run_asa(x_start, t0, asa_budget_each, rng)

        # ---- Phase 3: Fine-tuned ASA from current best ----
        if calls < budget - 10 and time.time() < deadline:
            fine_budget = min(50, budget - calls - 10)
            run_asa(best_x.copy(), 0.3, fine_budget, rng)

        # ---- Phase 4: Coordinate descent polish ----
        if calls < budget and time.time() < deadline:
            coord_descent(best_x.copy(), budget - calls, rng)

        return best_x, best_f
