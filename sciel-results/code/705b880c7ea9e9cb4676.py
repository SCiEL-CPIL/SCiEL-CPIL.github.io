    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Multi-start parameters
        n_starts = 5
        budget_per_start = 50
        total_start_budget = n_starts * budget_per_start
        
        best_overall_x = None
        best_overall_f = np.inf
        calls_used_total = 0

        # Phase 1: Multi-start
        for _ in range(n_starts):
            if calls_used_total >= budget:
                break
            if time.time() > deadline:
                break
            
            # Random start in [-4, 4]^5
            start_mean = rng.uniform(-4, 4, size=dim)
            start_sigma = 1.0
            
            # Run CMA-ES
            b_x, b_f, c_used = self._run_cmaes(func, start_mean, start_sigma, budget_per_start, rng, deadline)
            calls_used_total += c_used
            
            if b_f < best_overall_f:
                best_overall_f = b_f
                best_overall_x = b_x

        # Phase 2: Polish the best found
        remaining_budget = budget - calls_used_total
        if remaining_budget > 0 and best_overall_x is not None:
            # Restart from best found with larger sigma to explore around it?
            # Or just continue? 
            # Since we restarted, the covariance is lost.
            # Let's start a new CMA-ES from the best point with a moderate sigma.
            start_mean = best_overall_x
            start_sigma = 1.0 # Moderate sigma for local search
            b_x, b_f, c_used = self._run_cmaes(func, start_mean, start_sigma, remaining_budget, rng, deadline)
            calls_used_total += c_used
            if b_f < best_overall_f:
                best_overall_f = b_f
                best_overall_x = b_x

        if best_overall_x is None:
            best_overall_x = np.zeros(dim)
            best_overall_f = func(best_overall_x)
            calls_used_total += 1

        return (best_overall_x, best_overall_f)
