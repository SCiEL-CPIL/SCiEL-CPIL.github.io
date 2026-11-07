    import scipy.optimize as so
    
    def __call__(self, func):
        # Wrap func to count calls and check time
        calls = 0
        start_time = time.time()
        deadline = start_time + 280
        
        def wrapper(x):
            nonlocal calls
            if time.time() > deadline:
                raise TimeoutError()
            calls += 1
            if calls > self.budget:
                raise BudgetExceeded()
            return func(x)
        
        try:
            res = so.differential_evolution(
                wrapper, 
                bounds=[(-5, 5)] * self.dim, 
                maxiter=100, # DE in scipy is slow, limit iterations
                popsize=15,
                tol=1e-8,
                seed=42
            )
            # If it finished, try a local polish with Nelder-Mead or Powell
            # But DE might have used most of the budget.
            # Check remaining budget.
            remaining = self.budget - calls
            if remaining > 10:
                # Local polish
                res2 = so.minimize(
                    wrapper, 
                    res.x, 
                    method='Nelder-Mead', 
                    options={'maxiter': remaining}
                )
                if res2.fun < res.fun:
                    res = res2
            return res.x, res.fun
        except (TimeoutError, BudgetExceeded):
            # Return best found so far? 
            # We need to track best_x in wrapper.
            pass
    