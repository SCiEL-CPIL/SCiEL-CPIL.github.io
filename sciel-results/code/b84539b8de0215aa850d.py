import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        dim = self.dim
        budget = self.budget
        n_calls = 0
        best_x = None
        best_fx = np.inf

        def latin_hypercube_sample(n, d, low=-5.0, high=5.0):
            """Generate Latin Hypercube samples for better initial coverage."""
            points = np.zeros((n, d))
            for i in range(n):
                for j in range(d):
                    points[i, j] = low + (high - low) * (i + np.random.uniform()) / n
            return points

        def golden_section_1d(func, x_base, axis, low, high, max_evals, budget_check):
            """
            Golden section search along a 1D line.
            x_base: base point (dim-vector)
            axis: which coordinate to vary
            low, high: range for the coordinate
            Returns (best_coord, best_f, n_used)
            """
            nonlocal n_calls, best_fx, best_x
            
            gr = (np.sqrt(5.0) - 1.0) / 2.0  # golden ratio conjugate
            a = low
            b = high
            c = b - gr * (b - a)
            d = a + gr * (b - a)
            
            # Evaluate c and d
            x_c = x_base.copy()
            x_c[axis] = c
            f_c = func(x_c)
            n_calls += 1
            if n_calls > budget:
                n_calls -= 1
                return c, f_c, 1
            
            x_d = x_base.copy()
            x_d[axis] = d
            f_d = func(x_d)
            n_calls += 1
            if n_calls > budget:
                n_calls -= 1
                return c if f_c < f_d else d, min(f_c, f_d), 1
            
            evals_used = 2
            
            while (b - a) > 1e-10 and evals_used < max_evals:
                if f_c < f_d:
                    b = d
                    d = c
                    f_d = f_c
                    c = b - gr * (b - a)
                    x_c = x_base.copy()
                    x_c[axis] = c
                    f_c = func(x_c)
                    n_calls += 1
                    evals_used += 1
                else:
                    a = c
                    c = d
                    f_c = f_d
                    d = a + gr * (b - a)
                    x_d = x_base.copy()
                    x_d[axis] = d
                    f_d = func(x_d)
                    n_calls += 1
                    evals_used += 1
                
                if n_calls >= budget:
                    break
                if time.time() > deadline - 5:
                    break
                
                if f_c < best_fx:
                    best_fx = f_c
                    best_x = x_c.copy()
                if f_d < best_fx:
                    best_fx = f_d
                    best_x = x_d.copy()
            
            # Return the better of c and d
            if f_c <= f_d:
                return c, f_c, evals_used
            else:
                return d, f_d, evals_used

        # Use Latin Hypercube for better initial coverage
        num_init = 20
        init_x = latin_hypercube_sample(num_init, dim)
        init_f = np.array([func(x) for x in init_x])
        n_calls = num_init

        idx = np.argmin(init_f)
        x = init_x[idx].copy()
        best_fx = init_f[idx]
        best_x = x.copy()

        # If time is short, just do local search
        if time.time() > deadline - 10:
            try:
                res = minimize(func, x, method='Nelder-Mead', options={'maxfev': budget - n_calls, 'maxiter': budget - n_calls})
                if res.fun < best_fx:
                    best_fx = res.fun
                    best_x = res.x
            except:
                pass
            return best_x, best_fx

        # Phase 2: Coordinate-wise 1D golden section searches
        # We'll cycle through axes multiple times, with adaptive step sizes
        # For each axis, we do a 1D search around the current best point
        
        max_iter = 50  # number of full cycles through all axes
        for iteration in range(max_iter):
            if n_calls >= budget or time.time() > deadline - 10:
                break
            
            for axis in range(dim):
                if n_calls >= budget or time.time() > deadline - 5:
                    break
                
                # Determine search range for this axis
                # Use a window around the current value, expanding if needed
                current_val = best_x[axis]
                # Start with a moderate window, shrink as we converge
                window = max(0.5, 2.0 / (1.0 + iteration * 0.1))
                low = max(-5.0, current_val - window)
                high = min(5.0, current_val + window)
                
                if high - low < 1e-8:
                    continue
                
                # Perform golden section search along this axis
                # Limit evaluations per 1D search to avoid wasting budget
                max_evals_1d = min(30, budget - n_calls)
                if max_evals_1d < 4:
                    break
                
                best_coord, best_f_1d, n_used = golden_section_1d(
                    func, best_x, axis, low, high, max_evals_1d, None
                )
                
                # Update the axis value if improved
                if best_f_1d < best_fx - 1e-12:
                    best_x[axis] = best_coord
                    best_fx = best_f_1d
                
                # Even if no improvement, update the coordinate to the best found
                # (this helps with the step structure - we want to be on the flat side)
                if best_f_1d <= best_fx + 1e-12:
                    best_x[axis] = best_coord
        
        # Phase 3: Final polish with Nelder-Mead if budget remains
        remaining = budget - n_calls
        if remaining > 15:
            try:
                res = minimize(func, best_x, method='Nelder-Mead', options={'maxfev': remaining, 'maxiter': remaining, 'xatol': 1e-10, 'fatol': 1e-12})
                if res.fun < best_fx:
                    best_fx = res.fun
                    best_x = res.x
            except Exception:
                pass

        return best_x, best_fx
