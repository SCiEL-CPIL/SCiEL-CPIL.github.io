"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import cma


class Optimizer:
    """
    Optimizer using CMA-ES with multiple restarts.
    CMA-ES is well-suited for ill-conditioned continuous landscapes like Schaffers F7.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s limit
        n = self.dim
        total_budget = self.budget

        best_x = None
        best_f = np.inf

        # Use a fixed seed for reproducibility if needed, but random is fine.
        # The task mentions hidden seeded shift, but we don't know it.
        # We'll use random restarts.
        
        # Strategy:
        # 1. Initial random sampling to find a good starting point quickly.
        # 2. Run CMA-ES from that point.
        # 3. If CMA-ES stagnates or budget allows, restart from a new random point 
        #    or a perturbed best point.
        
        # Given the 500 budget and 5D problem, CMA-ES can do quite a few generations.
        # Population size lambda ~ 4 + floor(3 * log(n)) = 4 + 3*1.6 = 8.
        # 500 / 8 = ~62 generations. This is plenty for 5D.
        
        # To maximize AOCC, we want fast early progress.
        # CMA-ES with a small initial sigma and adaptive step size is good.
        
        # Let's try a multi-restart CMA-ES.
        # Restart 1: Start from center (0,0,0,0,0) or a random point?
        # Schaffers F7 is often ill-conditioned around the global minimum.
        # The global minimum is usually at 0 in standard BBOB, but here it's shifted.
        # Random initial points might be safer to avoid local minima if the shift is large.
        # However, the problem says minimizer is in [-4, 4]^5.
        
        # Let's do a few random initial evaluations to pick a good start.
        n_init = 10
        init_xs = []
        init_fxs = []
        
        for _ in range(n_init):
            if time.time() > deadline:
                break
            x = np.random.uniform(-5, 5, n)
            fx = func(x)
            init_xs.append(x)
            init_fxs.append(fx)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        if best_x is None:
            # Fallback if time ran out during init (unlikely)
            best_x = np.zeros(n)
            best_f = func(best_x)
        
        # Now run CMA-ES from the best initial point.
        # We have budget - n_init calls left.
        remaining_budget = total_budget - len(init_fxs)
        
        # CMA-ES options
        # sigma0: initial step size. Since box is [-5,5], 1.0 is reasonable.
        # bounds: [-5, 5]
        
        # We'll do up to 3 restarts.
        # Restart 1: From best_x
        # Restart 2: From a random point
        # Restart 3: From a random point
        
        restarts = 3
        budget_per_restart = max(1, remaining_budget // restarts)
        
        for r in range(restarts):
            if time.time() > deadline:
                break
            
            if r == 0:
                x0 = best_x.copy()
            else:
                # Pick a random point, possibly biased towards best_x
                # Or just a new random point to explore other basins
                x0 = np.random.uniform(-5, 5, n)
                # If we want to exploit, we could mix:
                # x0 = 0.5 * best_x + 0.5 * np.random.uniform(-5, 5, n)
                # Let's stick to pure random for diversity in restarts 2 and 3
                # Actually, for ill-conditioned functions, local search is key.
                # Maybe restart from best_x with a larger sigma?
                # CMA-ES adapts sigma. If we restart, we reset sigma.
                # Let's restart from a point near best_x but with some noise.
                x0 = best_x + np.random.normal(0, 1.0, n)
                x0 = np.clip(x0, -5, 5)

            # Define the function to pass to CMA-ES
            # CMA-ES calls func(x) and expects a float.
            # We need to track the number of function evaluations to stay within budget.
            # cma doesn't have a direct "budget" parameter that stops it exactly at N evals
            # in the simple API, but we can check time or use maxiter.
            # maxiter = budget_per_restart // lambda
            # lambda is determined by cma.
            
            opts = {
                'bounds': [-5, 5],
                'verbose': -9, # Silence output
                'seed': np.random.randint(0, 10000), # Random seed for each restart
                'maxfevals': budget_per_restart # This is the key parameter
            }
            
            # Note: cma.fmin uses maxfevals to stop.
            # It returns (x, f, result)
            
            try:
                # sigma0=1.0 is a good default for [-5,5]
                es = cma.CMA(x0, 1.0, opts)
                
                while not es.stop():
                    if time.time() > deadline:
                        break
                    solutions = es.ask()
                    fvals = [func(s) for s in solutions]
                    es.tell(solutions, fvals)
                    
                    # Update global best
                    if es.result.fbest < best_f:
                        best_f = es.result.fbest
                        best_x = es.result.xbest.copy()
                        
            except Exception:
                # If CMA-ES fails, just continue to next restart
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
