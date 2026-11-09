import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        D = self.dim
        budget = self.budget
        
        # Parameters
        lambda_pop = 10
        mu_parents = 5
        
        def run_es(start_x, start_sigma, remaining_budget):
            # Initialize population
            rng = np.random.default_rng(123)
            pop = np.zeros((lambda_pop, D))
            for i in range(lambda_pop):
                # Start near start_x, but spread out
                pop[i] = start_x + start_sigma * rng.standard_normal(D)
                pop[i] = np.clip(pop[i], -5, 5)
                
            # Evaluate initial pop
            fits = np.array([func(x) for x in pop])
            calls = lambda_pop
            
            best_x = pop[np.argmin(fits)]
            best_f = np.min(fits)
            
            sigma = start_sigma
            no_improve = 0
            
            while calls < remaining_budget:
                # Sort by fitness
                idx = np.argsort(fits)
                pop = pop[idx]
                fits = fits[idx]
                
                # Update best
                if fits[0] < best_f:
                    best_f = fits[0]
                    best_x = pop[0].copy()
                    no_improve = 0
                else:
                    no_improve += 1
                    
                # Generate offspring
                new_pop = np.zeros((lambda_pop, D))
                # Success count: how many offspring are better than the worst parent?
                # Or simpler: 1/5 rule based on the previous generation's improvement.
                # Let's use a simple success rule:
                # Count how many of the new population (generated from parents) are better than the current best parent (fits[mu-1]).
                
                # Actually, standard 1/5 rule:
                # In the previous step, we generated offspring. We need to know how many were accepted.
                # Since we are doing (lambda, mu), all offspring are evaluated.
                # Let's track the number of offspring that improved upon the best parent of the previous generation.
                
                # For simplicity, let's use a fixed step size adaptation based on the spread of the best mu.
                # Or just use the 1/5 rule:
                # success = 0
                # for i in range(lambda_pop):
                #     if fits[i] < fits[mu_parents-1]: success += 1
                # if success > lambda_pop * 0.5: sigma *= 1.5
                # else: sigma *= 0.75
                
                # This is a bit heuristic. Let's stick to a robust adaptation.
                
                # Generate offspring from weighted average of top mu parents
                # Weights: linear decreasing
                w = np.arange(mu_parents, 0, -1)
                w = w / np.sum(w)
                
                center = np.sum(w[:, None] * pop[:mu_parents], axis=0)
                
                # Estimate step size from the spread of the parents
                # Or just keep sigma constant? No, adaptive is better.
                
                # Let's use the 1/5 rule properly.
                # We need to know how many offspring from the LAST generation were better than the best parent of the PREVIOUS generation.
                # This requires storing the previous best parent fitness.
                
                # Let's simplify: Use a fixed sigma per restart, but restart often.
                # Or use a simple adaptation:
                # If best_f improved, sigma *= 1.1
                # Else sigma *= 0.9
                
                # Let's try:
                # If best_f < prev_best_f: sigma *= 1.2
                # else: sigma *= 0.8
                # But we need prev_best_f.
                
                # Let's implement the 1/5 rule correctly.
                # We need to store the fitness of the best parent from the previous generation.
                # Let's assume `prev_best_parent_f` is available.
                
                # For this implementation, I'll use a simpler adaptive scheme:
                # sigma is adjusted based on the distance between the best and the center.
                
                for i in range(lambda_pop):
                    # Pick a random parent from top mu
                    parent_idx = rng.integers(0, mu_parents)
                    # Or weighted pick
                    # parent_idx = rng.choice(mu_parents, p=w)
                    
                    # Add noise
                    new_pop[i] = center + sigma * rng.standard_normal(D)
                    new_pop[i] = np.clip(new_pop[i], -5, 5)
                    
                # Evaluate offspring
                new_fits = np.array([func(x) for x in new_pop])
                calls += lambda_pop
                
                # Update sigma based on success
                # How many new offspring are better than the best parent (fits[0] of current pop, which is the best of the previous generation)?
                # Actually, fits[0] is the best of the current population (which was the offspring of the previous step).
                # Let's compare new_fits to fits[0] (the best of the population we just evaluated).
                # If many new ones are better, we are moving in a good direction, increase sigma.
                
                success = np.sum(new_fits < fits[0])
                if success > lambda_pop * 0.5:
                    sigma *= 1.5
                else:
                    sigma *= 0.75
                    
                # Replace population
                pop = new_pop
                fits = new_fits
                
                # Check for stagnation
                if no_improve > 5 or sigma < 1e-4:
                    break
                    
            return best_x, best_f, calls

        # Main Loop
        total_calls = 0
        best_x_global = np.zeros(D)
        best_f_global = np.inf
        
        rng_global = np.random.default_rng(0)
        
        # Initial start
        start_x = np.zeros(D)
        start_sigma = 1.0
        x, f, c = run_es(start_x, start_sigma, budget - total_calls)
        best_x_global, best_f_global = x, f
        total_calls += c
        
        # Restarts
        while total_calls < budget:
            start_x = rng_global.uniform(-4, 4, D)
            start_sigma = 1.0
            
            x, f, c = run_es(start_x, start_sigma, budget - total_calls)
            
            if f < best_f_global:
                best_x_global, best_f_global = x, f
                
            total_calls += c
            
        return best_x_global, best_f_global
