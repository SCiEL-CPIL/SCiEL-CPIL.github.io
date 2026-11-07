import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = float('inf')
        calls = 0
        history = []

        def try_eval(x):
            nonlocal calls, best_x, best_f
            if calls >= budget or time.time() > deadline:
                return None
            x = np.clip(x, -5, 5)
            f = func(x)
            calls += 1
            history.append((f, x.copy()))
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def golden_section_1d(x_start, direction, f_start):
            """
            Golden section search along direction.
            Returns (best_x, best_f)
            """
            if calls >= budget or time.time() > deadline - 5:
                return x_start, f_start
            
            # Initial bracket: try small steps
            step = 0.1
            x_plus = x_start + step * direction
            f_plus = try_eval(x_plus)
            if f_plus is None:
                return x_start, f_start
            
            if f_plus >= f_start:
                # Try negative direction
                x_minus = x_start - step * direction
                f_minus = try_eval(x_minus)
                if f_minus is None:
                    return x_start, f_start
                if f_minus >= f_start:
                    return x_start, f_start
                # Minimum is in negative direction
                lo = -step * 2
                hi = 0.0
                x_lo = x_start + lo * direction
                f_lo = f_minus
                x_hi = x_start
                f_hi = f_start
            else:
                # Minimum is in positive direction, expand
                lo = 0.0
                hi = step
                x_lo = x_start
                f_lo = f_start
                x_hi = x_plus
                f_hi = f_plus
                
                # Expand to find bracket
                for _ in range(3):
                    if calls >= budget or time.time() > deadline - 5:
                        break
                    new_hi = hi * 2
                    if new_hi > 10.0:
                        break
                    x_new = x_start + new_hi * direction
                    f_new = try_eval(x_new)
                    if f_new is None:
                        break
                    if f_new >= f_hi:
                        break
                    lo = hi
                    x_lo = x_hi
                    f_lo = f_hi
                    hi = new_hi
                    x_hi = x_new
                    f_hi = f_new
            
            # Golden section
            gr = (np.sqrt(5) - 1) / 2
            for _ in range(8):
                if calls >= budget or time.time() > deadline - 5:
                    break
                width = hi - lo
                if width < 1e-4:
                    break
                m1 = hi - gr * width
                m2 = lo + gr * width
                x_m1 = x_start + m1 * direction
                x_m2 = x_start + m2 * direction
                f_m1 = try_eval(x_m1)
                if f_m1 is None:
                    break
                f_m2 = try_eval(x_m2)
                if f_m2 is None:
                    break
                
                if f_m1 < f_m2:
                    hi = m2
                    x_hi = x_m2
                    f_hi = f_m2
                else:
                    lo = m1
                    x_lo = x_m1
                    f_lo = f_m1
            
            # Return best point found
            candidates = [(f_start, x_start), (f_lo, x_lo), (f_hi, x_hi)]
            best = min(candidates, key=lambda t: t[0])
            return best[1], best[0]

        def nelder_mead(x0, max_calls=100):
            """
            Nelder-Mead simplex method for local optimization.
            """
            if max_calls < 10:
                return x0, best_f
            
            # Initialize simplex
            simplex = [x0.copy()]
            f_vals = [try_eval(x0)]
            if f_vals[0] is None:
                return x0, best_f
            
            # Add vertices
            for i in range(n):
                xi = x0.copy()
                xi[i] += 0.5
                simplex.append(xi)
                f_vals.append(try_eval(xi))
                if f_vals[-1] is None:
                    break
            
            if len(simplex) < 2:
                return x0, best_f
            
            # Nelder-Mead iterations
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            
            while calls < budget and time.time() < deadline - 5 and max_calls > 0:
                # Sort by function value
                order = np.argsort(f_vals)
                simplex = [simplex[i] for i in order]
                f_vals = [f_vals[i] for i in order]
                max_calls -= 1
                
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                f_r = try_eval(x_r)
                if f_r is None:
                    break
                
                if f_r < f_vals[0]:
                    # Expansion
                    x_e = centroid + gamma * (centroid - simplex[-1])
                    f_e = try_eval(x_e)
                    if f_e is None:
                        simplex[-1] = x_r
                        f_vals[-1] = f_r
                    elif f_e < f_r:
                        simplex[-1] = x_e
                        f_vals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_vals[-1] = f_r
                elif f_r < f_vals[-2]:
                    simplex[-1] = x_r
                    f_vals[-1] = f_r
                else:
                    # Contraction
                    if f_r < f_vals[-1]:
                        # Outside contraction
                        x_c = centroid + 0.5 * (x_r - centroid)
                        f_c = try_eval(x_c)
                        if f_c is None:
                            break
                        if f_c <= f_r:
                            simplex[-1] = x_c
                            f_vals[-1] = f_c
                        else:
                            # Shrink
                            for i in range(1, len(simplex)):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                f_vals[i] = try_eval(simplex[i])
                                if f_vals[i] is None:
                                    break
                    else:
                        # Inside contraction
                        x_c = centroid - 0.5 * (centroid - simplex[-1])
                        f_c = try_eval(x_c)
                        if f_c is None:
                            break
                        if f_c < f_vals[-1]:
                            simplex[-1] = x_c
                            f_vals[-1] = f_c
                        else:
                            # Shrink
                            for i in range(1, len(simplex)):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                f_vals[i] = try_eval(simplex[i])
                                if f_vals[i] is None:
                                    break
                
                # Check convergence
                if len(f_vals) > 1:
                    f_spread = max(f_vals) - min(f_vals)
                    if f_spread < 1e-6:
                        break
            
            # Return best
            best_idx = np.argmin(f_vals)
            return simplex[best_idx], f_vals[best_idx]

        # Phase 1: Global Sampling with Sobol-like sequence
        def sobol_sample(index, dim):
            """Simple low-discrepancy sequence"""
            point = []
            for j in range(dim):
                # Use different primes for each dimension
                prime = [2, 3, 5, 7, 11][j % 5]
                val = 0.0
                frac = 1.0
                idx = index
                while idx > 0:
                    frac /= prime
                    val += frac * (idx % prime)
                    idx //= prime
                point.append(val * 10.0 - 5.0)
            return np.array(point)
        
        # Sample 50 points with low-discrepancy sequence
        for i in range(1, 51):
            if calls >= 60 or time.time() > deadline - 15:
                break
            try_eval(sobol_sample(i, n))
        
        # Add center and some random points
        try_eval(np.zeros(n))
        
        for _ in range(10):
            if calls >= 70 or time.time() > deadline - 15:
                break
            try_eval(rng.uniform(-5, 5, n))

        # Phase 2: Local Optimization with Nelder-Mead
        if history:
            # Sort by function value
            history.sort(key=lambda t: t[0])
            starts = []
            for f, x in history:
                is_dup = False
                for s in starts:
                    if np.linalg.norm(x - s) < 0.8:
                        is_dup = True
                        break
                if not is_dup:
                    starts.append(x)
                if len(starts) == 3:
                    break

            remaining_budget = budget - calls
            if remaining_budget > 0 and starts:
                # Allocate more budget to better starts
                weights = []
                for i, (f, x) in enumerate(history[:len(starts)]):
                    # Exponential decay based on rank
                    weights.append(1.0 / (i + 1))
                total_w = sum(weights)
                budgets = [int(remaining_budget * w / total_w) for w in weights]
                
                for i, x_start in enumerate(starts):
                    if calls >= budget - 5 or time.time() > deadline - 5:
                        break
                    alloc = budgets[i] if i < len(budgets) else max(20, remaining_budget // (len(starts) - i))
                    if alloc < 15:
                        break
                    nelder_mead(x_start, alloc)
                    
                    # After each local search, re-evaluate best
                    if best_x is not None:
                        f_check = try_eval(best_x)
                        if f_check is None:
                            break

        return best_x, best_f
