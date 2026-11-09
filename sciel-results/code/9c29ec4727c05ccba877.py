import numpy as np
import time
import scipy.optimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.start_time = None
        self.best_x = None
        self.best_f = float('inf')

    def __call__(self, func):
        self.start_time = time.time()
        deadline = self.start_time + 290.0  # Safety margin
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget
        
        # DE Parameters
        N = 10  # Population size
        F = 0.8  # Differential weight
        CR = 0.9  # Crossover rate
        
        # Initialize population
        pop = rng.uniform(-5, 5, (N, n))
        
        fpop = np.empty(N)
        for i in range(N):
            if self.calls >= budget or time.time() > deadline:
                break
            fpop[i] = func(pop[i])
            self.calls += 1
            if fpop[i] < self.best_f:
                self.best_f = fpop[i]
                self.best_x = pop[i].copy()
        
        if self.best_x is None:
             # Fallback if we couldn't even init
             self.best_x = np.zeros(n)
             self.best_f = func(self.best_x)
             self.calls += 1
             return self.best_x, self.best_f

        # Local refinement function using Powell
        def local_refine(x_init, max_calls):
            """
            Perform local refinement using Powell's method.
            Returns the best x and f found, and updates self.calls.
            """
            if self.calls >= budget or time.time() > deadline:
                return x_init, self.best_f
            
            # Wrapper for scipy to track calls
            local_calls = [0]
            local_best_x = [x_init.copy()]
            local_best_f = [self.best_f] # Start with current global best to avoid re-evaluating if possible? 
            # Actually, we need to evaluate x_init to be sure, but we might have just evaluated it.
            # Let's just let the wrapper handle it.
            
            def wrapper(z):
                if self.calls >= budget or time.time() > deadline or local_calls[0] >= max_calls:
                    return local_best_f[0]
                z_clipped = np.clip(z, -5, 5)
                val = func(z_clipped)
                self.calls += 1
                local_calls[0] += 1
                if val < local_best_f[0]:
                    local_best_f[0] = val
                    local_best_x[0] = z_clipped.copy()
                return val

            try:
                # Powell is good for Rosenbrock
                result = scipy.optimize.minimize(
                    wrapper, 
                    x_init, 
                    method='Powell',
                    options={'maxiter': 50, 'ftol': 1e-12, 'xtol': 1e-10, 'catol': 1e-10}
                )
                # Update global best if local found something better
                if local_best_f[0] < self.best_f:
                    self.best_f = local_best_f[0]
                    self.best_x = local_best_x[0].copy()
            except Exception:
                pass
            
            return self.best_x, self.best_f

        # Main DE Loop
        # We will run DE generations, and after each, try to refine the best.
        # To ensure we don't get stuck, we also re-inject the best into the population.
        
        # Sort initial population
        order = np.argsort(fpop)
        pop = pop[order]
        fpop = fpop[order]
        
        # Update best from sorted
        if fpop[0] < self.best_f:
            self.best_f = fpop[0]
            self.best_x = pop[0].copy()

        # Track stagnation
        last_improvement_calls = self.calls
        stagnation_threshold = 20 # If no improvement in 20 calls, consider restarting or aggressive polishing

        while self.calls < budget and time.time() < deadline:
            # 1. Generate Offspring
            for i in range(N):
                if self.calls >= budget or time.time() > deadline:
                    break
                
                # Select distinct a, b, c
                while True:
                    a, b, c = rng.choice(N, 3, replace=False)
                    if a != i and b != i and c != i:
                        break
                
                # Mutation: DE/rand/1
                v = pop[a] + F * (pop[b] - pop[c])
                
                # Crossover
                r = rng.random(n)
                j_rand = rng.integers(n)
                u = np.where(r < CR, v, pop[i])
                u[j_rand] = v[j_rand]
                u = np.clip(u, -5, 5)
                
                fu = func(u)
                self.calls += 1
                
                # Selection
                if fu < fpop[i]:
                    pop[i] = u
                    fpop[i] = fu
                    if fu < self.best_f:
                        self.best_f = fu
                        self.best_x = u.copy()
                        last_improvement_calls = self.calls

            # 2. Local Refinement
            # Refine the best individual found so far
            # We allocate a small budget for local search, e.g., 20 calls
            # This helps AOCC by quickly reducing error once in the valley
            
            # Check for stagnation to decide on refinement strategy
            is_stagnant = (self.calls - last_improvement_calls) > stagnation_threshold
            
            # If stagnant, we might do a slightly more aggressive restart or just stick to Powell
            # For now, we always try to polish if we have budget.
            
            if self.calls < budget - 5 and time.time() < deadline - 10:
                # Use a copy of the current best
                x_refine = self.best_x.copy()
                
                # Determine local budget
                # If stagnant, give it a bit more? Or keep it constant to avoid wasting time?
                # Let's keep it constant to ensure we return to global search often enough.
                local_budget = 20
                if self.calls + local_budget <= budget:
                    self.best_x, self.best_f = local_refine(x_refine, local_budget)
                
                # Update the best individual in the population to the refined one
                # This ensures DE continues from a better point
                # Find the index of the current best in pop
                # Note: pop might be stale. Let's just replace the worst or best.
                # Better: Replace the individual that corresponds to the current best if it's there, 
                # or just insert the refined best as the new best.
                # Since we track self.best_x separately, we can just ensure the population
                # contains this good point.
                # Let's replace the worst individual in the population with the refined best
                # to maintain diversity but inject the improvement.
                worst_idx = np.argmax(fpop)
                pop[worst_idx] = self.best_x
                fpop[worst_idx] = self.best_f
                
                # Re-sort population to keep it organized for DE
                order = np.argsort(fpop)
                pop = pop[order]
                fpop = fpop[order]
                
                # If we improved during local refinement, reset stagnation counter
                if self.best_f < fpop[0]: # This check is redundant if we updated pop correctly, but safe
                     last_improvement_calls = self.calls

        return self.best_x, self.best_f
