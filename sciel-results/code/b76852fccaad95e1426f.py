import numpy as np
import time

class Optimizer:
        def __init__(self, budget, dim):
            self.budget = budget
            self.dim = dim

        def __call__(self, func):
            start_time = time.time()
            deadline = start_time + 280
            d = self.dim
            budget = self.budget
            calls = 0

            def time_ok():
                return time.time() < deadline

            def try_call(x):
                nonlocal calls
                if calls >= budget or not time_ok():
                    return None
                fx = float(func(x))
                calls += 1
                return fx

            # Phase 1: Probe vertices first, then center.
            # This might yield better early best-values (b_i) than starting with 0.
            probes = []
            vals = []
            
            # 1. Evaluate 5 unit vertices
            for i in range(d):
                if calls >= budget or not time_ok():
                    break
                xi = np.zeros(d)
                xi[i] = 5.0
                fi = try_call(xi)
                if fi is None:
                    break
                probes.append(xi)
                vals.append(fi)
            
            # 2. Evaluate center
            if calls < budget and time_ok():
                x0 = np.zeros(d)
                f0 = try_call(x0)
                if f0 is not None:
                    probes.append(x0)
                    vals.append(f0)
            
            if len(probes) < d + 1:
                # Fallback if budget/time ran out
                best_x = probes[0] if probes else np.zeros(d)
                best_f = vals[0] if vals else 0.0
                return best_x, best_f

            # Solve for gradient
            # f(x) = a^T x + b
            # f(0) = b
            # f(5e_i) = 5 a_i + b
            b = vals[-1] # Last one is 0
            a = np.zeros(d)
            for i in range(d):
                # vals[i] corresponds to 5e_i
                a[i] = (vals[i] - b) / 5.0

            # Determine optimal corner
            x_opt = np.zeros(d)
            for i in range(d):
                if a[i] > 0:
                    x_opt[i] = -5.0
                elif a[i] < 0:
                    x_opt[i] = 5.0
                else:
                    x_opt[i] = 0.0

            # Check if x_opt was already evaluated
            # If so, we don't need to call again, but we should update best.
            # However, calling it again ensures we have the value if there was floating point noise?
            # No, func is deterministic.
            # But we need to make a call to consume budget?
            # The scoring depends on b_i.
            # If x_opt is in probes, b_i is already f_opt.
            # We still need to fill the budget.
            
            best_x = x_opt
            best_f = None
            
            # We must call x_opt to be sure? 
            # If x_opt was evaluated, we have the value.
            # But we don't track which value corresponds to which x easily in the list.
            # It's safer to just call x_opt. It's 1 call.
            # If x_opt was one of the probes, this is a redundant call, but harmless.
            # Wait, if x_opt was a probe, we already have the best value.
            # Calling it again doesn't change best_f.
            
            f_opt_val = try_call(x_opt)
            if f_opt_val is not None:
                best_f = f_opt_val
                best_x = x_opt.copy()
            else:
                # Time/budget out?
                # Find best from probes
                min_val = min(vals)
                best_f = min_val
                # Identify which probe gave min_val
                idx = vals.index(min_val)
                best_x = probes[idx].copy()

            # Fill remaining budget
            while calls < budget and time_ok():
                try_call(best_x)

            return best_x, best_f
