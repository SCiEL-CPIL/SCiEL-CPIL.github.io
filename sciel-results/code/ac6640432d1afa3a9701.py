import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin before 330s kill
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        rng = np.random.default_rng(42)
        
        lb = np.full(self.dim, -5.0)
        ub = np.full(self.dim, 5.0)
        d = self.dim
        bounds = [(-5.0, 5.0)] * d

        def call_func(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= self.budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
                
            x = np.asarray(x, dtype=np.float64)
            x = np.clip(x, lb, ub)
            
            try:
                f = float(func(x))
                if not np.isfinite(f):
                    f = np.inf
            except Exception:
                f = np.inf
                
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Strategy: Parallel Multi-Scale Differential Evolution + Local Polish
        # Three populations:
        #   - Large (NP=12): Global exploration, high crossover
        #   - Medium (NP=8): Basin identification, medium crossover
        #   - Small (NP=4): Fine-tuning, low crossover, tight neighborhood
        
        # Phase 1: Global sampling to seed populations
        num_seed = 24
        seeds = []
        for i in range(num_seed):
            if calls_made >= self.budget or time.time() > deadline:
                break
            x = rng.uniform(-5.0, 5.0, d)
            f = call_func(x)
            seeds.append((f, x.copy()))
        
        if not seeds:
            x = np.zeros(d)
            call_func(x)
            return best_x, best_f
        
        seeds.sort(key=lambda item: item[0])
        
        # Initialize three populations from best seeds
        # Large pop: spread out, use diverse seeds
        # Medium pop: closer to best
        # Small pop: very close to best
        
        def init_pop(size, spread_scale, center):
            pop = []
            for i in range(size):
                if i < len(seeds):
                    x = seeds[i][1].copy()
                else:
                    x = center + spread_scale * rng.standard_normal(d)
                    x = np.clip(x, lb, ub)
                f = call_func(x)
                pop.append((f, x.copy()))
            pop.sort(key=lambda item: item[0])
            return pop
        
        best_seed_x = seeds[0][1]
        
        # Large population for global exploration
        pop_large = init_pop(12, 3.0, best_seed_x)
        
        # Medium population for basin identification
        pop_med = init_pop(8, 1.5, best_seed_x)
        
        # Small population for fine-tuning
        pop_small = init_pop(4, 0.5, best_seed_x)
        
        # Differential evolution parameters
        def de_step(pop, F, CR, max_generations):
            """Run differential evolution for a given population."""
            n = len(pop)
            for gen in range(max_generations):
                if calls_made >= self.budget or time.time() > deadline:
                    break
                
                new_pop = list(pop)
                for i in range(n):
                    if calls_made >= self.budget or time.time() > deadline:
                        break
                    
                    # Select three distinct random indices different from i
                    while True:
                        r1, r2, r3 = rng.choice(n, size=3, replace=False)
                        if len(set([r1, r2, r3, i])) == 4:
                            break
                    
                    # Differential variation
                    mutant = pop[r1][1] + F * (pop[r2][1] - pop[r3][1])
                    
                    # Crossover
                    trial = pop[i][1].copy()
                    j_rand = rng.integers(0, d)
                    for j in range(d):
                        if rng.random() < CR or j == j_rand:
                            trial[j] = mutant[j]
                    trial = np.clip(trial, lb, ub)
                    
                    # Evaluate trial
                    f_trial = call_func(trial)
                    
                    # Selection
                    if f_trial <= pop[i][0]:
                        new_pop[i] = (f_trial, trial.copy())
                
                pop = new_pop
                pop.sort(key=lambda item: item[0])
            
            return pop
        
        # Run DE on each population with different parameters
        # Large: more generations, higher F and CR for exploration
        # Medium: medium
        # Small: fewer generations, lower F and CR for exploitation
        
        # Budget allocation for DE
        remaining = self.budget - calls_made
        if remaining > 30:
            # Large pop: ~40% of remaining
            gen_large = max(1, min(8, (remaining // 3) // 12))
            pop_large = de_step(pop_large, F=0.8, CR=0.7, max_generations=gen_large)
            
            remaining = self.budget - calls_made
            if remaining > 20:
                # Medium pop: ~35% of remaining
                gen_med = max(1, min(6, (remaining // 3) // 8))
                pop_med = de_step(pop_med, F=0.6, CR=0.5, max_generations=gen_med)
                
                remaining = self.budget - calls_made
                if remaining > 15:
                    # Small pop: ~25% of remaining
                    gen_small = max(1, min(4, (remaining // 3) // 4))
                    pop_small = de_step(pop_small, F=0.4, CR=0.3, max_generations=gen_small)
        
        # Merge all populations and find best
        all_points = pop_large + pop_med + pop_small
        all_points.sort(key=lambda item: item[0])
        
        # Phase 2: Local refinement from top candidates
        # Take top 6 diverse points and polish with L-BFGS-B + Nelder-Mead
        
        refined = []
        for f, x in all_points[:10]:
            # Check diversity: skip if too close to already refined
            min_dist = np.inf
            for rx, rf in refined:
                dist = np.linalg.norm(x - rx)
                min_dist = min(min_dist, dist)
            
            if min_dist > 0.8 or len(refined) == 0:
                refined.append((x.copy(), f))
            
            if len(refined) >= 6:
                break
        
        # Polish each refined point
        max_local_runs = len(refined)
        for idx, (x_start, f_start) in enumerate(refined):
            if calls_made >= self.budget or time.time() > deadline:
                break
            
            remaining = self.budget - calls_made
            runs_left = max(1, max_local_runs - idx)
            
            # L-BFGS-B
            maxfev = max(5, min(30, remaining // (runs_left * 2)))
            try:
                res = minimize(
                    call_func,
                    x_start,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxfev': maxfev,
                        'ftol': 1e-12,
                        'gtol': 1e-8
                    }
                )
            except Exception:
                pass
            
            # Nelder-Mead for fine-tuning
            if calls_made < self.budget and time.time() < deadline:
                nm_maxfev = max(3, min(20, (self.budget - calls_made) // runs_left))
                try:
                    res_nm = minimize(
                        call_func,
                        best_x,
                        method='Nelder-Mead',
                        options={
                            'maxfev': nm_maxfev,
                            'xatol': 1e-8,
                            'fatol': 1e-8
                        }
                    )
                except Exception:
                    pass
        
        # Phase 3: Final intensive polish if budget remains
        if calls_made < self.budget and time.time() < deadline and best_x is not None:
            remaining = self.budget - calls_made
            if remaining > 15:
                final_maxfev = min(remaining, 25)
                try:
                    res_final = minimize(
                        call_func,
                        best_x,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={
                            'maxfev': final_maxfev,
                            'ftol': 1e-15,
                            'gtol': 1e-12
                        }
                    )
                except Exception:
                    pass
                
                if calls_made < self.budget and time.time() < deadline:
                    nm_final_maxfev = min(self.budget - calls_made, 20)
                    if nm_final_maxfev > 3:
                        try:
                            res_nm_final = minimize(
                                call_func,
                                best_x,
                                method='Nelder-Mead',
                                options={
                                    'maxfev': nm_final_maxfev,
                                    'xatol': 1e-10,
                                    'fatol': 1e-10
                                }
                            )
                        except Exception:
                            pass

        if best_x is None:
            best_x = np.zeros(d)
            call_func(best_x)

        return best_x, best_f
