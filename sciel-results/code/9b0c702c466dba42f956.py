import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(123)
        
        NP = 10
        F = 0.8
        CR = 0.9
        
        if budget < NP + 10:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls = 1
            while calls < budget:
                x = rng.uniform(-5, 5, dim)
                f = func(x)
                calls += 1
                if f < best_f:
                    best_f = f
                    best_x = x
            return best_x, best_f

        pop = rng.uniform(-5, 5, (NP, dim))
        fit = np.array([func(x) for x in pop])
        calls_made = NP
        
        best_idx = np.argmin(fit)
        best_x = pop[best_idx].copy()
        best_f = fit[best_idx]
        
        remaining = budget - calls_made
        no_improve_gen = 0
        max_no_improve = 10
        spread = 1.0
        
        while remaining > 0:
            gen_improved = False
            
            for i in range(NP):
                if remaining <= 0:
                    break
                
                candidates = list(range(NP))
                candidates.remove(i)
                r1, r2, r3 = rng.choice(candidates, 3, replace=False)
                
                v = pop[r1] + F * (pop[r2] - pop[r3])
                
                j_rand = rng.integers(0, dim)
                trial = pop[i].copy()
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                
                trial = np.clip(trial, -5, 5)
                
                f_trial = func(trial)
                calls_made += 1
                remaining -= 1
                
                if f_trial <= fit[i]:
                    pop[i] = trial
                    fit[i] = f_trial
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()
                        gen_improved = True
                        
            if not gen_improved:
                no_improve_gen += 1
            else:
                no_improve_gen = 0
                
            if no_improve_gen >= max_no_improve:
                # Restart
                pop[best_idx] = best_x
                fit[best_idx] = best_f
                for i in range(NP):
                    if i == best_idx:
                        continue
                    if remaining <= 0:
                        break
                    x_new = best_x + rng.normal(0, spread, dim)
                    x_new = np.clip(x_new, -5, 5)
                    f_new = func(x_new)
                    calls_made += 1
                    remaining -= 1
                    pop[i] = x_new
                    fit[i] = f_new
                    if f_new < best_f:
                        best_f = f_new
                        best_x = x_new.copy()
                        best_idx = i
                no_improve_gen = 0
                spread *= 0.5
                
        return best_x, best_f
