import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bound_lo = -5.0
        self.bound_hi = 5.0
        self.start_time = None
        self.deadline = 325.0

    def __call__(self, func):
        self.start_time = time.time()
        d = self.dim
        budget = self.budget
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        rng = np.random.default_rng(42)

        def check_time():
            if time.time() - self.start_time > self.deadline:
                return True
            return False

        def safe_eval(x):
            nonlocal calls_made
            if calls_made >= budget or check_time():
                return None
            x = np.clip(x, self.bound_lo, self.bound_hi)
            fx = func(x)
            calls_made += 1
            return fx

        # Nelder-Mead implementation
        def nelder_mead(x0, max_iter=50):
            nonlocal calls_made
            d = len(x0)
            
            # Initial evaluation
            f0 = safe_eval(x0)
            if f0 is None:
                return x0, np.inf
            simplex = [x0.copy()]
            fvals = [f0]
            
            # Initial simplex
            step = 1.0 
            for i in range(d):
                x_new = x0.copy()
                if x0[i] + step > self.bound_hi:
                    step = x0[i] - self.bound_lo
                if x0[i] - step < self.bound_lo:
                    step = self.bound_hi - x0[i]
                if step <= 0:
                    step = 0.1
                x_new[i] += step
                fx = safe_eval(x_new)
                if fx is None:
                    break
                simplex.append(x_new)
                fvals.append(fx)
                
            if len(simplex) < d + 1:
                return simplex[0], fvals[0]

            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]
            
            for _ in range(max_iter):
                if check_time() or calls_made >= budget:
                    break
                
                x_worst = simplex[-1]
                f_worst = fvals[-1]
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_ref = centroid + 2.0 * (centroid - x_worst)
                x_ref = np.clip(x_ref, self.bound_lo, self.bound_hi)
                f_ref = safe_eval(x_ref)
                if f_ref is None: break
                
                if f_ref < fvals[-2]:
                    if f_ref < fvals[0]:
                        # Expansion
                        x_exp = centroid + 2.0 * (x_ref - centroid)
                        x_exp = np.clip(x_exp, self.bound_lo, self.bound_hi)
                        f_exp = safe_eval(x_exp)
                        if f_exp is None: break
                        if f_exp < f_ref:
                            simplex[-1] = x_exp
                            fvals[-1] = f_exp
                        else:
                            simplex[-1] = x_ref
                            fvals[-1] = f_ref
                    else:
                        simplex[-1] = x_ref
                        fvals[-1] = f_ref
                else:
                    if f_ref < f_worst:
                        # Outside Contraction
                        x_con = centroid + 0.5 * (x_ref - centroid)
                        x_con = np.clip(x_con, self.bound_lo, self.bound_hi)
                        f_con = safe_eval(x_con)
                        if f_con is None: break
                        if f_con <= f_ref:
                            simplex[-1] = x_con
                            fvals[-1] = f_con
                        else:
                            # Inside Contraction
                            x_con_in = centroid + 0.5 * (centroid - x_worst)
                            x_con_in = np.clip(x_con_in, self.bound_lo, self.bound_hi)
                            f_con_in = safe_eval(x_con_in)
                            if f_con_in is None: break
                            if f_con_in < f_worst:
                                simplex[-1] = x_con_in
                                fvals[-1] = f_con_in
                            else:
                                # Shrink
                                for i in range(1, d + 1):
                                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                                    fvals[i] = safe_eval(simplex[i])
                                    if fvals[i] is None:
                                        # Fill with inf if budget exceeded
                                        fvals[i] = np.inf
                    else:
                        # Inside Contraction
                        x_con_in = centroid + 0.5 * (centroid - x_worst)
                        x_con_in = np.clip(x_con_in, self.bound_lo, self.bound_hi)
                        f_con_in = safe_eval(x_con_in)
                        if f_con_in is None: break
                        if f_con_in < f_worst:
                            simplex[-1] = x_con_in
                            fvals[-1] = f_con_in
                        else:
                            # Shrink
                            for i in range(1, d + 1):
                                simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                                fvals[i] = safe_eval(simplex[i])
                                if fvals[i] is None:
                                    fvals[i] = np.inf
                
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]
                
                if np.max(fvals) - np.min(fvals) < 1e-8:
                    break
                    
            return simplex[0], fvals[0]

        # Main Loop
        batch_size = 10
        
        while calls_made < budget and not check_time():
            batch_x = rng.uniform(self.bound_lo, self.bound_hi, (batch_size, d))
            batch_f = []
            
            for i in range(batch_size):
                if calls_made >= budget or check_time():
                    break
                fx = safe_eval(batch_x[i])
                if fx is None:
                    break
                batch_f.append((fx, batch_x[i].copy()))
                
            if not batch_f:
                break
                
            batch_f.sort()
            
            if batch_f[0][0] < best_f:
                best_f = batch_f[0][0]
                best_x = batch_f[0][1].copy()
                
            median_f = batch_f[len(batch_f)//2][0]
            if best_f < median_f - 0.1:
                x_opt, f_opt = nelder_mead(best_x, max_iter=50)
                if f_opt < best_f:
                    best_f = f_opt
                    best_x = x_opt

        return best_x, best_f
