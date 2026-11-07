import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        x = rng.uniform(-5, 5, self.dim)
        fx = func(x)
        calls = 1
        
        # Phase 1: Rough exploration (Hill Climbing)
        # Use 20% of budget
        explore_budget = int(self.budget * 0.2)
        sigma = 2.0
        while calls < explore_budget:
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            calls += 1
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.1 # Slightly increase step if improving
            else:
                sigma *= 0.9 # Decrease if not

        # Phase 2: Coordinate Descent with Golden Section
        invphi = (np.sqrt(5) - 1) / 2
        invphi2 = invphi * invphi
        
        while calls < self.budget:
            for i in range(self.dim):
                if calls >= self.budget:
                    break
                
                # Determine max evals for this 1D search
                # We want to finish the cycle if possible, but not exceed budget
                # Use 10 evals per 1D search
                max_evals = 10
                if calls + max_evals > self.budget:
                    max_evals = self.budget - calls
                    if max_evals <= 0:
                        break
                
                a = -5.0
                b = 5.0
                
                c = a + invphi2 * (b - a)
                d = a + invphi * (b - a)
                
                x_c = x.copy(); x_c[i] = c
                x_d = x.copy(); x_d[i] = d
                
                fc = func(x_c); calls += 1
                fd = func(x_d); calls += 1
                
                best_x = x_c if fc < fd else x_d
                best_f = min(fc, fd)
                
                for _ in range(max_evals - 2):
                    if calls >= self.budget:
                        break
                    
                    if fc < fd:
                        b = d
                        d = c
                        fd = fc
                        c = a + invphi2 * (b - a)
                        x_c = x.copy(); x_c[i] = c
                        fc = func(x_c); calls += 1
                        if fc < best_f:
                            best_f = fc
                            best_x = x_c
                    else:
                        a = c
                        c = d
                        fc = fd
                        d = a + invphi * (b - a)
                        x_d = x.copy(); x_d[i] = d
                        fd = func(x_d); calls += 1
                        if fd < best_f:
                            best_f = fd
                            best_x = x_d
                
                if best_f < fx:
                    x, fx = best_x, best_f

        return x, fx
